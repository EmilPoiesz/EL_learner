"""
test_llm_oracle.py
==================
LLMOracle response handling without loading a model: the HuggingFace pipeline
is replaced by a stub that returns scripted responses.

1  a parseable MQ response is cached and replayed without a second generation
2  an unparseable MQ response raises OracleError and is not cached
3  a cached response that does not parse is regenerated and overwritten
4  EQ: judge 'yes' returns None; judge 'no' parses the counterexample line
5  EQ: counterexamples the LLM's own MQ keeps rejecting end in OracleError after 3 attempts
6  a cached response is not reused under a different max_new_tokens
7  yes/no extraction: final line first, <think> block ignored, fallback to last match
8  EQ: an unusable counterexample is asked for again, with the reason in the prompt
9  the counterexample line is found and unwrapped (fences, backticks, labels, full stop)
10 names outside the signature or the roles in scope are rejected, saying which
11 Manchester parsing: keywords in any case, prefixed and hyphenated names
12 sampling: Ollama's settings reach generate(); each seed has its own cache entries
13 combined EQ: "yes" returns None; a counterexample (even after "Yes,") is parsed
14 natural language: rendering and parsing round-trip; an EQ in natural language
15 EQ: a counterexample H already entails is asked for again, saying so
16 EQ: its own system message and a budget of at least EQ_MIN_NEW_TOKENS
"""

from __future__ import annotations

import re

import pytest

from learner.cache.cache import LLMCache
from learner.el_algorithm import ELConcept, GCI, OracleError, step
from learner import llm_prompts as prompts
from learner.el_syntax import gci_to_nl, parse_manchester_gci, parse_nl_gci
from learner.llm_oracle import EQ_MIN_NEW_TOKENS, EQSettings, LLMOracle, query_kind
from learner.llm_prompts import CHECK_SYSTEM_PROMPTS
from learner.upstream import OLLAMA_SAMPLING, PAPER_SYSTEM_PROMPTS, upstream_natural_language

_GCI = GCI(ELConcept({"A"}), ELConcept({"B"}))


class _StubPipe:
    """Stands in for transformers.pipeline: returns scripted responses in order."""

    def __init__(self, responses: list[str]):
        self.responses = list(responses)
        self.calls = 0
        self.prompts: list[str] = []
        self.systems: list[str] = []
        self.budgets: list[int] = []
        self.kwargs: list[dict] = []

    def __call__(self, messages, generation_config=None, **kwargs):
        self.calls += 1
        self.kwargs.append(kwargs)
        self.prompts.append(messages[-1]["content"])
        self.systems.append(messages[0]["content"])
        self.budgets.append(generation_config.max_new_tokens if generation_config else None)
        reply = self.responses.pop(0)
        return [{"generated_text": messages + [{"role": "assistant", "content": reply}]}]


def _oracle(tmp_path, responses: list[str]) -> LLMOracle:
    # Bypass __init__: it loads a model and needs a running H-reasoner.
    oracle = LLMOracle.__new__(LLMOracle)
    oracle.mq_count = oracle.eq_count = 0
    oracle._pipe = _StubPipe(responses)
    oracle._model_name = "stub"
    oracle._max_new_tokens = 16
    oracle._sampling = None
    oracle._pad_token_id = 0
    oracle._prompt_style = "ours"
    oracle._verbose = False
    oracle._system_msg = "system"
    oracle._sig = {"A", "B"}
    oracle._roles = set()
    oracle._h_reasoner = lambda gci: False
    oracle._cache = LLMCache(enabled=True, db_path=str(tmp_path / "cache.db"))
    oracle._seed = 0
    oracle._eq = EQSettings()
    oracle._eq_language = "manchester"
    oracle._eq_system_msg = None          # None: the MQ system message
    oracle._rejected = ()
    oracle._failed = ()
    return oracle


def test_1_parseable_response_cached(tmp_path):
    oracle = _oracle(tmp_path, ["Thinking...\nyes"])
    assert oracle.MQ(_GCI) is True
    assert oracle.MQ(_GCI) is True           # second call served from the cache
    assert oracle._pipe.calls == 1


def test_2_unparseable_response_not_cached(tmp_path):
    oracle = _oracle(tmp_path, ["Let me think about whether", "no"])
    with pytest.raises(OracleError, match="Unparseable MQ"):
        oracle.MQ(_GCI)
    assert oracle.MQ(_GCI) is False          # regenerated, not replayed
    assert oracle._pipe.calls == 2


def test_3_stale_cached_response_regenerated(tmp_path):
    oracle = _oracle(tmp_path, ["yes"])
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": prompts.mq_prompt({"A", "B"}, set(), _GCI)},
    ]
    params = {"max_new_tokens": 16, "do_sample": False}
    oracle._cache.set("stub", messages, "cut off mid-sentence", params)   # as an older version cached it
    assert oracle.MQ(_GCI) is True
    assert oracle._cache.get("stub", messages, params) == "yes"


def test_4_eq_judge_and_counterexample(tmp_path):
    oracle = _oracle(tmp_path, ["yes"])
    assert oracle.EQ(set()) is None

    (tmp_path / "b").mkdir()
    # judge, counterexample, then the MQ confirming the counterexample
    oracle = _oracle(tmp_path / "b", ["no", "Reasoning...\nA SubClassOf B", "yes"])
    assert oracle.EQ(set()) == _GCI


def test_5_inconsistent_counterexample_raises(tmp_path):
    # judge; then three counterexamples, each rejected by the model's own MQ
    # (the repeated A ⊑ B is answered from the MQ cache, so no third "no").
    oracle = _oracle(tmp_path, ["no", "A SubClassOf B", "no", "A SubClassOf B", "B SubClassOf A", "no"])
    with pytest.raises(OracleError, match="no usable counterexample in 3 attempts.*you answered no"):
        oracle.EQ(set())
    assert oracle._pipe.responses == []


def test_6_cache_separated_by_budget(tmp_path):
    oracle = _oracle(tmp_path, ["yes", "no"])
    assert oracle.MQ(_GCI) is True
    oracle._max_new_tokens = 1024             # same prompt, larger budget
    assert oracle.MQ(_GCI) is False           # asked again, not replayed
    assert oracle._pipe.calls == 2


@pytest.mark.parametrize("response, expected", [
    ("yes", "yes"),
    ("No.", "no"),
    ("Yes — there is no axiom contradicting it.", "yes"),
    ("I think the answer is no.\nAnswer: Yes", "yes"),
    ("<think>Is it yes? Maybe no.</think>\n\nNo", "no"),
    ("<think>yes</think>\nThe ontology does not say.", None),
    ("Reasoning says yes.\nFinal verdict follows.", "yes"),
    ("I cannot tell.", None),
])
def test_7_extract_yes_no(response, expected):
    assert LLMOracle._extract_yes_no(response) == expected


def test_8_unusable_counterexample_asked_again(tmp_path):
    oracle = _oracle(tmp_path, ["no", "Cat SubClassOf Animal", "A SubClassOf B", "yes"])
    assert oracle.EQ(set()) == _GCI
    first, retry = oracle._pipe.prompts[1], oracle._pipe.prompts[2]
    assert "could not be used" not in first
    assert "Animal, Cat are not concept names in scope" in retry


@pytest.mark.parametrize("response", [
    "A SubClassOf B",
    "A SubClassOf B.",
    "Answer: `A SubClassOf B`",
    "```\nA SubClassOf B\n```",
    "- A SubClassOf B",
    '"A subclassof B"',
    "<think>Maybe B SubClassOf A?</think>\nA SubClassOf B",
])
def test_9_answer_line_unwrapped(tmp_path, response):
    oracle = _oracle(tmp_path, [])
    assert oracle._parse_counterexample(response) == _GCI


def test_10_names_out_of_scope_rejected(tmp_path):
    oracle = _oracle(tmp_path, [])
    oracle._roles = {"r"}
    assert oracle._parse_counterexample("A SubClassOf r some B") == GCI(
        ELConcept({"A"}), ELConcept(existentials={("r", ELConcept({"B"}))}))
    with pytest.raises(ValueError, match="s is not an object property in scope"):
        oracle._parse_counterexample("A SubClassOf s some B")
    with pytest.raises(ValueError, match="C is not a concept name in scope"):
        oracle._parse_counterexample("A SubClassOf C")
    with pytest.raises(ValueError, match="not of the form"):
        oracle._parse_counterexample("A is a kind of B")


@pytest.mark.parametrize("text, expected", [
    ("A subClassOf B", _GCI),
    ("A SubClassOf: B", _GCI),
    ("A AND C SubClassOf B", GCI(ELConcept({"A", "C"}), ELConcept({"B"}))),
    ("A SubClassOf has-part SOME ex:Leg",
     GCI(ELConcept({"A"}), ELConcept(existentials={("has-part", ELConcept({"ex:Leg"}))}))),
    ("owl:Thing SubClassOf B", GCI(ELConcept(), ELConcept({"B"}))),
    # A bracketed conjunction (Qwen brackets the whole right-hand side, cex25-cex27).
    ("Bird SubClassOf (Animal and Vertebrate and (has_part some Wing))",
     GCI(ELConcept({"Bird"}), ELConcept({"Animal", "Vertebrate"},
                                        {("has_part", ELConcept({"Wing"}))}))),
    ("(A and C) SubClassOf ((B and (r some (D and E))))",
     GCI(ELConcept({"A", "C"}), ELConcept({"B"}, {("r", ELConcept({"D", "E"}))}))),
    ("(A and B) and (r some (C and D)) SubClassOf E",
     GCI(ELConcept({"A", "B"}, {("r", ELConcept({"C", "D"}))}), ELConcept({"E"}))),
])
def test_11_manchester_parsing(text, expected):
    assert parse_manchester_gci(text) == expected


def test_12_sampling_settings_and_cache_by_seed(tmp_path):
    configs = []
    oracle = _oracle(tmp_path, ["yes", "no"])
    pipe = oracle._pipe
    oracle._pipe = lambda messages, generation_config=None: (configs.append(generation_config),
                                                             pipe(messages))[1]
    oracle._sampling = {"do_sample": True, "temperature": 0.8, **OLLAMA_SAMPLING, "seed": 0}
    assert oracle.MQ(_GCI) is True
    assert oracle.MQ(_GCI) is True            # same seed: the first answer, from the cache
    oracle._sampling = {**oracle._sampling, "seed": 1}
    assert oracle.MQ(_GCI) is False           # another seed draws its own answer
    assert pipe.calls == 2
    config = configs[0]
    assert config.do_sample and config.temperature == 0.8
    assert (config.top_k, config.top_p, config.repetition_penalty) == (40, 0.9, 1.1)



@pytest.mark.parametrize("response, expected", [
    ("yes", None),
    ("Yes.", None),
    ("A SubClassOf B", _GCI),
    ("Yes, for example:\nA SubClassOf B", _GCI),
])
def test_13_combined_eq(tmp_path, response, expected):
    # "yes" ends learning; a counterexample is then confirmed by the model's own MQ.
    oracle = _oracle(tmp_path, [response, "yes"])
    oracle._eq = EQSettings(mode="combined")
    assert oracle.EQ(set()) == expected
    assert "If H captures everything that is true in O, reply with exactly one word: yes" \
        in oracle._pipe.prompts[0]


def test_14_natural_language(tmp_path):
    B, C = ELConcept({"B"}), ELConcept({"C"})
    for gci in [
        _GCI,
        GCI(ELConcept({"A"}), ELConcept(existentials={("r", B)})),
        GCI(ELConcept({"A"}, {("r", B)}), C),
        GCI(ELConcept(existentials={("r", ELConcept({"B", "C"}))}), ELConcept({"A"})),
        GCI(ELConcept({"A"}), ELConcept(existentials={("r", ELConcept(existentials={("s", B)}))})),
        GCI(ELConcept(existentials={("r", ELConcept())}), ELConcept({"A"})),
    ]:
        assert parse_nl_gci(gci_to_nl(gci)) == gci
    assert gci_to_nl(GCI(ELConcept(existentials={("r", ELConcept({"B", "C"}))}), ELConcept({"A"}))) \
        == "something that r some (B that is also C) is a subcategory of A"

    oracle = _oracle(tmp_path, ["no", "'A' is a subcategory of 'B'.", "yes"])
    oracle._eq_language = "nl"
    assert oracle.EQ({GCI(ELConcept({"B"}), ELConcept({"A"}))}) == _GCI
    assert "  B is a subcategory of A\n" in oracle._pipe.prompts[0]
    with pytest.raises(ValueError, match="not of the form 'C is a subcategory of D'"):
        oracle._parse_counterexample("A SubClassOf B")


def test_15_h_entailed_counterexample_asked_again(tmp_path):
    other = GCI(ELConcept({"B"}), ELConcept({"A"}))
    oracle = _oracle(tmp_path, ["no", "B SubClassOf A", "A SubClassOf B", "yes"])
    oracle._h_reasoner = lambda gci: gci == other
    assert oracle.EQ({other}) == _GCI
    assert "could not be used: 'B SubClassOf A' already follows from H" in oracle._pipe.prompts[2]


def test_16_eq_system_message_and_budget(tmp_path):
    oracle = _oracle(tmp_path, ["no", "A SubClassOf B", "yes"])
    oracle._max_new_tokens = 2               # the paper's MQ budget
    oracle._eq_system_msg = "eq system"
    assert oracle.EQ(set()) == _GCI
    assert oracle._pipe.systems == ["eq system", "eq system", "system"]   # judge, counterexample, MQ
    assert oracle._pipe.budgets == [EQ_MIN_NEW_TOKENS, EQ_MIN_NEW_TOKENS, 2]


@pytest.mark.parametrize("language, response", [
    ("manchester", "An A SubClassOf the B."),
    ("manchester", "A A SubClassOf B"),          # the article, then the concept named A
    ("nl", "An A is a subcategory of a B."),
    ("nl", "A A is a subcategory of B"),
])
def test_17_leading_article_stripped(tmp_path, language, response):
    # Mistral wrote "A Professor SubClassOf Teacher", copying the example's A.
    oracle = _oracle(tmp_path, [])
    oracle._eq_language = language
    assert oracle._parse_counterexample(response) == _GCI


def test_18_article_kept_when_stripping_does_not_help(tmp_path):
    oracle = _oracle(tmp_path, [])
    with pytest.raises(ValueError, match="'A Cat SubClassOf B' is not of the form"):
        oracle._parse_counterexample("A Cat SubClassOf B")
    prompt = prompts.eq_counterexample_prompt({"A", "B"}, set(), set())
    assert "  C SubClassOf D\n" in prompt and "A SubClassOf B" not in prompt


_A_R_B = GCI(ELConcept({"A"}), ELConcept(existentials={("r", ELConcept({"B"}))}))


@pytest.mark.parametrize("language, response, expected", [
    ("manchester", "A SubClassOf r B", _A_R_B),           # "Project SubClassOf employed_by University"
    ("manchester", "A r B", _A_R_B),                      # "Bird has_part Wing"
    ("manchester", "An A r a B.", _A_R_B),
    ("manchester", "A and r B SubClassOf B",
     GCI(ELConcept({"A"}, {("r", ELConcept({"B"}))}), ELConcept({"B"}))),
    ("manchester", "A SubClassOf r some B", _A_R_B),      # already right: unchanged
    ("nl", "A is r B", _A_R_B),                           # "Teacher is employed_by University"
    ("nl", "A is a subcategory of something that r B", _A_R_B),
])
def test_19_role_without_some(tmp_path, language, response, expected):
    oracle = _oracle(tmp_path, [])
    oracle._roles = {"r"}
    oracle._eq_language = language
    assert oracle._parse_counterexample(response) == expected


def test_20_role_repair_needs_a_role_in_scope(tmp_path):
    oracle = _oracle(tmp_path, [])
    oracle._roles = {"r"}
    with pytest.raises(ValueError, match="'A s B' is not of the form"):
        oracle._parse_counterexample("A s B")
    with pytest.raises(ValueError, match="not of the form"):  # an explanation stays unusable
        oracle._parse_counterexample("A r B but does not follow from H, as H says nothing")


@pytest.mark.parametrize("check, expected_prompt", [
    ("mq", "A SubClassOf B"),
    ("nl", upstream_natural_language("A SubClassOf B")),
])
def test_21_counterexample_check_language(tmp_path, check, expected_prompt):
    # paper-advanced: the counterexample in Manchester syntax, its check as eq_check says.
    oracle = _oracle(tmp_path, ["no", "A SubClassOf B", "True"])
    oracle._prompt_style, oracle._system_msg = "paper-advanced", PAPER_SYSTEM_PROMPTS["paper-advanced"]
    oracle._eq = EQSettings(check=check)
    assert oracle.EQ(set()) == _GCI
    assert oracle._pipe.prompts[2] == expected_prompt
    assert oracle.mq_count == 1


@pytest.mark.parametrize("response, expected", [
    # Olivia job 2703958: the statement, then an explanation.
    ("A r B\n\nThis statement is true in O (birds have wings) but does not follow from H.", _A_R_B),
    ("A r B\n(This statement is true in O, but does not follow from H.)", _A_R_B),
    ("Reasoning first.\nA SubClassOf B", _GCI),                 # the last line still comes first
    ("B SubClassOf A\nA SubClassOf B", _GCI),
])
def test_22_statement_on_an_earlier_line(tmp_path, response, expected):
    oracle = _oracle(tmp_path, [])
    oracle._roles = {"r"}
    assert oracle._parse_counterexample(response) == expected


def test_23_no_line_parses_reports_the_last(tmp_path):
    oracle = _oracle(tmp_path, [])
    with pytest.raises(ValueError, match="'Because H says nothing' is not of the form"):
        oracle._parse_counterexample("Cat has wings\nBecause H says nothing.")


@pytest.mark.parametrize("variant", ["s2", "s3", "s4"])
def test_24_check_system_prompt(tmp_path, variant):
    # The check asks under the variant; an ordinary MQ keeps the paper's system prompt.
    oracle = _oracle(tmp_path, ["no", "A SubClassOf B", "True", "True"])
    oracle._prompt_style, oracle._system_msg = "paper-advanced", PAPER_SYSTEM_PROMPTS["paper-advanced"]
    oracle._eq = EQSettings(check="nl", check_system=variant)
    assert oracle.EQ(set()) == _GCI
    assert oracle._pipe.prompts[2] == upstream_natural_language("A SubClassOf B")
    assert oracle._pipe.systems[2] == CHECK_SYSTEM_PROMPTS[variant]
    assert oracle.MQ(GCI(ELConcept({"B"}), ELConcept({"A"}))) is True
    assert oracle._pipe.systems[3] == PAPER_SYSTEM_PROMPTS["paper-advanced"]
    assert "has_part some Wheel" in CHECK_SYSTEM_PROMPTS[variant]
    assert ("How to read the statements" in CHECK_SYSTEM_PROMPTS[variant]) == (variant == "s3")


@pytest.mark.parametrize("memory", [False, True])
def test_25_rejections_remembered_across_eqs(tmp_path, memory):
    # EQ 1: A ⊑ B rejected, then B ⊑ A accepted.  Without eq_memory EQ 2 asks the
    # same prompts and replays EQ 1 from the cache, A ⊑ B first (as in cex8); with
    # it, EQ 2's first counterexample prompt lists A ⊑ B and is asked anew.
    oracle = _oracle(tmp_path, ["no", "A SubClassOf B", "no", "B SubClassOf A", "yes",
                                "B SubClassOf A"])
    oracle._eq = EQSettings(memory=memory)
    assert oracle.EQ(set()) == GCI(ELConcept({"B"}), ELConcept({"A"}))
    assert oracle.EQ(set()) == GCI(ELConcept({"B"}), ELConcept({"A"}))
    assert oracle._pipe.calls == (6 if memory else 5)
    if memory:
        assert "do not propose it again: 'A SubClassOf B'" in oracle._pipe.prompts[5]
    assert oracle._rejected == (("A SubClassOf B",) if memory else ())


def test_26_only_counterexamples_sampled(tmp_path):
    configs = []
    oracle = _oracle(tmp_path, ["no", "A SubClassOf B", "yes"])
    pipe = oracle._pipe
    oracle._pipe = lambda messages, generation_config=None: (configs.append(generation_config),
                                                             pipe(messages))[1]
    oracle._eq, oracle._seed = EQSettings(temperature=1.0), 3
    assert oracle.EQ(set()) == _GCI
    judge, counterexample, check = configs
    assert not judge.do_sample and not check.do_sample
    assert counterexample.do_sample and counterexample.temperature == 1.0


def test_27_eq_settings():
    with pytest.raises(ValueError, match="mode must be one of"):
        EQSettings(mode="guess")
    with pytest.raises(ValueError, match="check_system must be one of"):
        EQSettings(check_system="s9")
    with pytest.raises(ValueError, match="attempts must be at least 1"):
        EQSettings(attempts=0)
    with pytest.raises(ValueError, match="generation must be one of"):
        EQSettings(generation="g9")
    with pytest.raises(ValueError, match="needs mode 'judge'"):
        EQSettings(mode="combined", generation="g1")
    assert EQSettings(generation="g3").describe() == "judge, counterexample prompt g3"
    assert EQSettings().describe() == "judge"
    assert EQSettings(check="nl", check_system="s2", memory=True, temperature=1.0).describe(3) == (
        "judge, counterexamples checked by a natural-language MQ, check system prompt s2, "
        "rejections remembered across EQs, counterexamples sampled at temperature 1.0, seed 3")


def test_28_eq_attempts(tmp_path):
    # judge; one unusable answer, and with attempts=1 that is the end.
    oracle = _oracle(tmp_path, ["no", "not a statement"])
    oracle._eq = EQSettings(attempts=1)
    with pytest.raises(OracleError, match="no usable counterexample in 1 attempts"):
        oracle.EQ(set())


@pytest.mark.parametrize("generation, present, absent", [
    ("default", ["C SubClassOf r some D"], ["Rules:", "Car", "|"]),
    ("g1", ["C SubClassOf r some D", "Rules:", "never r D", "not, or, only"], ["Car"]),
    ("g2", ["Car SubClassOf has_component some Wheel", "not about O"], ["C SubClassOf", "Rules:"]),
    ("g3", ["Subject | relation | Object", "Car | is-a | Vehicle"], ["SubClassOf", "Rules:"]),
    ("g2b", ["Car SubClassOf has_component some Wheel", "powered_by some Engine SubClassOf"],
     ["(", "C SubClassOf"]),
    ("g3b", ["three parts separated by |", "Car | is-a | Vehicle"],
     ["Subject", "Object:", "(", "SubClassOf"]),
])
def test_29_generation_prompts(generation, present, absent):
    prompt = prompts.eq_counterexample_prompt({"A", "B"}, {"r"}, set(), generation=generation)
    for text in present:
        assert text in prompt
    question = prompt.split("Name one counterexample")[1]
    for text in absent:
        assert text not in question
    with pytest.raises(ValueError, match="Manchester syntax only"):
        prompts.eq_counterexample_prompt({"A"}, set(), set(), language="nl", generation="g1")


@pytest.mark.parametrize("text, expected", [
    ("Bird | is-a | Animal", "Bird SubClassOf Animal"),
    ("| Bird | is a | Animal |", "Bird SubClassOf Animal"),
    ("Bird|ISA|Animal", "Bird SubClassOf Animal"),
    ("`Bird | has_part | Wing`", "Bird SubClassOf has_part some Wing"),
])
def test_30_triple_parsing(text, expected):
    from learner.el_syntax import gci_to_triple, parse_triple
    gci = parse_triple(text.strip("`"))
    assert gci == parse_manchester_gci(expected)
    assert parse_triple(gci_to_triple(gci)) == gci
    assert gci_to_triple(parse_manchester_gci("A and B SubClassOf C")) is None
    for bad in ("Bird | is-a", "Bird | has part | Wing", "Bird SubClassOf Animal"):
        with pytest.raises(ValueError):
            parse_triple(bad)


def test_31_g3_eq(tmp_path):
    # judge no; a Manchester answer is unusable under G3, said in triple terms; the
    # triple B | is-a | A is rejected by the check, and the retry line names it as a
    # triple; then A | r | B is accepted.
    oracle = _oracle(tmp_path, ["no", "A SubClassOf B", "B | is-a | A", "no",
                                "A | r | B", "yes"])
    oracle._roles = {"r"}
    oracle._eq = EQSettings(generation="g3")
    assert oracle.EQ(set()) == parse_manchester_gci("A SubClassOf r some B")
    assert "is not of the form 'Subject | relation | Object'" in oracle._pipe.prompts[2]
    assert "asked whether 'B | is-a | A' is true in O" in oracle._pipe.prompts[4]


@pytest.mark.parametrize("response, expected", [
    ("(Bird lives_in Water)", "Bird SubClassOf lives_in some Water"),
    ("(Cat, eats, Meat)", "Cat SubClassOf eats some Meat"),
    ("(Bird, Has_Part, Wing)", "Bird SubClassOf has_part some Wing"),
    ("(Elephant SubClassOf Herbivore)", "Elephant SubClassOf Herbivore"),
    ("(Elephant, is-a, Mammal)", "Elephant SubClassOf Mammal"),
])
def test_32_brackets_and_tuples(tmp_path, response, expected):
    # cex14: every G2 answer was wrapped in brackets, often as a tuple.
    oracle = _oracle(tmp_path, [])
    oracle._sig, oracle._roles = {"Bird", "Cat", "Elephant", "Herbivore", "Mammal", "Meat",
                                  "Water", "Wing"}, {"eats", "has_part", "lives_in"}
    assert oracle._parse_counterexample(response) == parse_manchester_gci(expected)


@pytest.mark.parametrize("response, expected", [
    ("Subject | is-a | Object: Bird | is-a | Vertebrate", "Bird SubClassOf Vertebrate"),
    ("Cat | eats | Fish: While it's possible for a cat to eat fish", "Cat SubClassOf eats some Fish"),
    ("(Cat | is-a | Mammal)", "Cat SubClassOf Mammal"),
])
def test_33_triple_readings(tmp_path, response, expected):
    # cex15: the format before the answer, or an explanation after a colon.
    oracle = _oracle(tmp_path, [])
    oracle._sig, oracle._roles = {"Bird", "Cat", "Fish", "Mammal", "Vertebrate"}, {"eats"}
    oracle._eq = EQSettings(generation="g3b")
    assert oracle._parse_counterexample(response) == parse_manchester_gci(expected)
    with pytest.raises(ValueError, match=re.escape("'Subject | relation | Object'")):
        oracle._parse_counterexample("Subject | lays | Egg: Every bird lays eggs")


@pytest.mark.parametrize("text, expected", [
    ("Teacher SubClassOf Professor", "Is it true that every Teacher is a Professor?"),
    ("Elephant SubClassOf Animal", "Is it true that every Elephant is an Animal?"),
    ("Bird SubClassOf has_part some Wing",
     "Is it true that every Bird is related by has_part to some Wing?"),
    ("Animal and (eats some Meat) SubClassOf Carnivore",
     "Is it true that every Animal that is related by eats to some Meat is a Carnivore?"),
])
def test_34_every_question(text, expected):
    assert prompts.every_question(parse_manchester_gci(text)) == expected


@pytest.mark.parametrize("system, question, expected_prompt", [
    ("s1", "paper", "Can A be considered a subcategory of 'B'?"),
    ("s4", "every", "Is it true that every A is a B?"),
    ("paper", "every", "Is it true that every A is a B?"),
])
def test_35_check_question_and_system(tmp_path, system, question, expected_prompt):
    # judge, counterexample, then the check as the settings say.
    oracle = _oracle(tmp_path, ["no", "A SubClassOf B", "True"])
    oracle._prompt_style = "paper-advanced"
    oracle._system_msg = PAPER_SYSTEM_PROMPTS["paper-advanced"]
    oracle._eq = EQSettings(check="nl", check_system=system, check_question=question)
    assert oracle.EQ(set()) == _GCI
    assert oracle._pipe.prompts[2] == expected_prompt
    assert oracle._pipe.systems[2] == (CHECK_SYSTEM_PROMPTS[system] or PAPER_SYSTEM_PROMPTS["paper-advanced"])
    with pytest.raises(ValueError, match="check_question must be one of"):
        EQSettings(check_question="why")



@pytest.mark.parametrize("memory", [False, True])
def test_36_failed_eq_reasons_carried(tmp_path, memory):
    # EQ 1: an unusable answer and a rejected one (attempts=2), so it fails.  EQ 2's
    # first prompt names both (the rejection through the memory line when memory is on),
    # so it is not the cached EQ 1 prompt; its answer is accepted, which clears the carry.
    # EQ 2's judge is asked again too: after a failed EQ its prompt says so (test 40).
    oracle = _oracle(tmp_path, ["no", "not a statement", "A SubClassOf B", "no",
                                "no", "B SubClassOf A", "yes"])
    oracle._eq = EQSettings(attempts=2, memory=memory)
    with pytest.raises(OracleError):
        oracle.EQ(set())
    assert oracle.EQ(set()) == GCI(ELConcept({"B"}), ELConcept({"A"}))
    second = oracle._pipe.prompts[5]
    assert "'not a statement' is not of the form" in second
    assert ("do not propose it again: 'A SubClassOf B'" in second) == memory
    assert ("asked whether 'A SubClassOf B' is true in O" in second) == (not memory)
    assert oracle._failed == ()



def test_37_thinking_model_answer_after_think(tmp_path):
    # A template that opens <think> (Olmo 3 Think): only the text after </think> is
    # read; without </think> the budget ran out, which reads as no answer (False for
    # the paper's MQ) and is counted.
    oracle = _oracle(tmp_path, ["Birds are animals, so... </think>\n\nTrue",
                                "Let me think about whether every"])
    oracle._prompt_style = "paper-advanced"
    oracle._system_msg = PAPER_SYSTEM_PROMPTS["paper-advanced"]
    oracle._thinks = True
    assert oracle.MQ(_GCI) is True
    assert oracle.MQ(GCI(ELConcept({"B"}), ELConcept({"A"}))) is False
    assert oracle.truncated == 1


def test_38_model_generation_defaults_off(tmp_path):
    # transformers 4.50-4.57 fill settings left at the library default from the model's own
    # generation config (Olmo: sampling at T 0.6, cex20); use_model_defaults=False stops that.
    # A generate() without the parameter (transformers 5) gets nothing.
    from learner.llm_oracle import model_defaults_off

    class Old:
        def generate(self, inputs=None, generation_config=None, use_model_defaults=None, **kwargs):
            pass

    class New:
        def generate(self, inputs=None, generation_config=None, **kwargs):
            pass

    assert model_defaults_off(Old()) == {"use_model_defaults": False}
    assert model_defaults_off(New()) == {}
    assert model_defaults_off(object()) == {}

    oracle = _oracle(tmp_path, ["yes"])
    oracle._generate_kwargs = model_defaults_off(Old())
    assert oracle.MQ(_GCI) is True
    assert oracle._pipe.kwargs == [{"use_model_defaults": False}]


def test_39_cut_off_reasoning_not_cached(tmp_path):
    # Under the paper's MQ a reply cut off before </think> reads as False; that is the
    # answer to this query, but it is not cached, so the next ask goes to the model.
    oracle = _oracle(tmp_path, ["Let me think about whether every",
                                "Birds are animals. </think> True"])
    oracle._prompt_style = "paper-advanced"
    oracle._system_msg = PAPER_SYSTEM_PROMPTS["paper-advanced"]
    oracle._thinks = True
    assert oracle.MQ(_GCI) is False
    assert oracle.MQ(_GCI) is True            # asked again, not replayed as False
    assert oracle.MQ(_GCI) is True            # the full answer is cached
    assert (oracle._pipe.calls, oracle.truncated) == (2, 1)

    # A cut-off reply cached by an older version counts as a miss.
    (tmp_path / "stale").mkdir()
    stale = _oracle(tmp_path / "stale", ["Done. </think> True"])
    stale._prompt_style, stale._system_msg = "paper-advanced", PAPER_SYSTEM_PROMPTS["paper-advanced"]
    stale._thinks = True
    from learner.upstream import upstream_gci_text
    messages = [{"role": "system", "content": stale._system_msg},
                {"role": "user", "content": upstream_gci_text(_GCI)}]
    stale._cache.set("stub", messages, "cut off mid-thought", {"max_new_tokens": 16, "do_sample": False})
    assert stale.MQ(_GCI) is True and stale._pipe.calls == 1


def test_40_repeated_eq_is_a_new_prompt(tmp_path):
    # A failed EQ with the same reasons as the one before, or a judge reply that does not
    # parse, used to leave the next EQ's prompts unchanged: a greedy model (or the cache)
    # then failed the same way again.  Now the prompts say how many EQs failed in a row.
    oracle = _oracle(tmp_path, ["no", "not a statement", "not a statement", "not a statement",
                                "no", "not a statement", "not a statement", "not a statement",
                                "maybe", "no", "A SubClassOf B", "yes"])
    for _ in range(2):
        with pytest.raises(OracleError, match="no usable counterexample"):
            oracle.EQ(set())
    first, second = oracle._pipe.prompts[:4], oracle._pipe.prompts[4:8]
    assert not any("in a row" in p for p in first)
    assert all("The previous equivalence query in a row gave no usable answer" in p for p in second)
    assert set(first).isdisjoint(second)
    with pytest.raises(OracleError, match="Unparseable EQ judge"):
        oracle.EQ(set())                                 # the judge says "maybe"
    assert "The previous 2 equivalence queries in a row" in oracle._pipe.prompts[8]
    assert oracle.EQ(set()) == _GCI                      # a usable counterexample resets the count
    assert "The previous 3 equivalence queries in a row" in oracle._pipe.prompts[9]
    assert oracle._failed_eqs == 0


@pytest.mark.parametrize("truth", [None, lambda gci: True, lambda gci: False])
def test_41_labels_never_change_a_decision(tmp_path, capsys, truth):
    # The model's own check decides; the labeller (ELK over O) only annotates the log
    # and eq_outcomes, whatever it says.
    responses = ["no", "not a statement", "B SubClassOf A", "no", "A SubClassOf B", "yes"]
    oracle = _oracle(tmp_path, responses)
    oracle._truth, oracle._verbose = truth, True
    assert oracle.EQ(set()) == _GCI
    out = capsys.readouterr().out
    if truth is None:
        assert dict(oracle.eq_outcomes) == {"unusable": 1, "rejected": 1, "accepted": 1}
        assert "in O]" not in out
    else:
        word = "true" if truth(_GCI) else "false"
        assert dict(oracle.eq_outcomes) == {"unusable": 1, f"{word} rejected": 1, f"{word} accepted": 1}
        assert f"rejected by the model's own MQ [{word} in O]" in out
        assert f"counterexample A ⊑ B [{word} in O]" in out


def test_42_learners_notation_read_back():
    import random
    from learner.el_syntax import parse_dl_gci
    E = lambda atoms=(), ex=(): ELConcept(frozenset(atoms), frozenset(ex))
    rnd = random.Random(0)

    def concept(depth):
        atoms = rnd.sample(["A", "B", "Cat", "has-part", "ex:Meat"], rnd.randint(0, 2))
        ex = [(rnd.choice(["r", "has_part", "ex:eats"]), concept(depth - 1))
              for _ in range(rnd.randint(0, 2 if depth else 0))]
        return E(atoms, ex)

    for _ in range(300):
        gci = GCI(concept(2), concept(2))
        assert parse_dl_gci(str(gci)) == gci
    assert parse_dl_gci("Professor ⊑ ∃teaches.(Module)") == GCI(E({"Professor"}), E((), {("teaches", E({"Module"}))}))
    with pytest.raises(ValueError):
        parse_dl_gci("Professor SubClassOf Person")


@pytest.mark.parametrize("as_check", [False, True])
def test_43_mqs_asked_as_the_check(tmp_path, as_check):
    # mq_asked_as="check": the learner's MQs take the check's form (natural language, S4),
    # so one statement is not asked in two ways within a run (cex19).
    gci = GCI(ELConcept({"A"}), ELConcept(existentials={("r", ELConcept({"B"}))}))
    oracle = _oracle(tmp_path, ["True"])
    oracle._prompt_style = "paper-advanced"
    oracle._system_msg = PAPER_SYSTEM_PROMPTS["paper-advanced"]
    oracle._roles = {"r"}
    oracle._eq = EQSettings(check="nl", check_system="s4")
    oracle._mq_as_check = as_check
    assert oracle.MQ(gci) is True
    assert oracle.mq_count == 1
    if as_check:
        assert oracle._pipe.prompts == ["Can A be considered a subcategory of 'something that r some B'?"]
        assert oracle._pipe.systems == [CHECK_SYSTEM_PROMPTS["s4"]]
    else:
        assert oracle._pipe.prompts == ["A SubClassOf r some B"]
        assert oracle._pipe.systems == [PAPER_SYSTEM_PROMPTS["paper-advanced"]]


def test_44_counterexample_budget(tmp_path):
    # A reasoning model needs room to think and then write a statement (cex20: 29 of
    # 38 attempts cut off); the judge and the check are yes/no answers and keep theirs.
    oracle = _oracle(tmp_path, ["no", "not a statement", "A SubClassOf B", "yes"])
    oracle._eq = EQSettings(max_new_tokens=4096)
    assert oracle.EQ(set()) == _GCI
    assert oracle._pipe.budgets == [EQ_MIN_NEW_TOKENS, 4096, 4096, 16]   # judge, 2 attempts, check
    with pytest.raises(ValueError, match="max_new_tokens must be at least 1"):
        EQSettings(max_new_tokens=0)
    assert "4096 tokens per counterexample" in EQSettings(max_new_tokens=4096).describe()


def test_45_query_stats(tmp_path):
    # For estimating the cost on other models: model calls by step of the algorithm
    # and kind of query, with cache hits apart; MQs by step on the oracle.
    oracle = _oracle(tmp_path, ["no", "A SubClassOf B", "yes"])
    with step("EQ"):
        assert oracle.EQ(set()) == _GCI                 # judge, counterexample, check (an MQ)
    with step("saturation"):
        assert oracle.MQ(_GCI) is True                  # the check's answer, from the cache
    stats = oracle.query_stats
    assert set(stats) == {("EQ", "EQ (judge)"), ("EQ", "EQ (counterexample)"), ("EQ", "MQ"),
                          ("saturation", "MQ")}
    assert stats[("EQ", "MQ")]["generated"] == 1 and stats[("EQ", "MQ")]["seconds"] >= 0
    assert (stats[("saturation", "MQ")]["asked"], stats[("saturation", "MQ")]["cached"],
            stats[("saturation", "MQ")]["generated"]) == (1, 1, 0)
    assert oracle.mq_by_step == {"EQ": 1, "saturation": 1}
    assert query_kind("MQ (check, s4 system prompt, every question)") == "MQ (check)"
    assert query_kind("EQ (counterexample or yes)") == "EQ (counterexample or yes)"


def test_46_eq_language_apart_from_the_mq_style(tmp_path):
    # cex23: MQs under paper-advanced (Manchester), EQs in the paper's natural language,
    # with G2b's car examples in it; the answer is parsed in natural language.
    oracle = _oracle(tmp_path, ["no", "A is a subcategory of B", "True"])
    oracle._prompt_style = "paper-advanced"
    oracle._system_msg = PAPER_SYSTEM_PROMPTS["paper-advanced"]
    oracle._eq = EQSettings(language="nl", generation="g2b")
    oracle._eq_language = "nl"
    oracle._sig, oracle._roles = {"A", "B", "C"}, {"r"}
    H = {GCI(ELConcept({"C"}), ELConcept(existentials={("r", ELConcept({"A"}))}))}
    assert oracle.EQ(H) == _GCI
    judge, counterexample = oracle._pipe.prompts[:2]
    assert "C is a subcategory of something that r some A" in judge
    assert "Car is a subcategory of something that has_component some Wheel" in counterexample
    assert "SubClassOf" not in judge + counterexample
    assert oracle._pipe.prompts[2] == "A SubClassOf B"           # the check: the paper's MQ
    with pytest.raises(ValueError, match="language must be one of"):
        EQSettings(language="french")
    assert prompts.eq_counterexample_prompt({"A"}, set(), set(), language="nl", generation="g2b")
    with pytest.raises(ValueError, match="Manchester syntax only"):
        prompts.eq_counterexample_prompt({"A"}, set(), set(), language="nl", generation="g2")
