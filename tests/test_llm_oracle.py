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
"""

from __future__ import annotations

import pytest

from learner.cache.cache import LLMCache
from learner.el_algorithm import ELConcept, GCI, OracleError
from learner.llm_oracle import LLMOracle, parse_manchester_gci

_GCI = GCI(ELConcept({"A"}), ELConcept({"B"}))


class _StubPipe:
    """Stands in for transformers.pipeline: returns scripted responses in order."""

    def __init__(self, responses: list[str]):
        self.responses = list(responses)
        self.calls = 0
        self.prompts: list[str] = []

    def __call__(self, messages, generation_config=None):
        self.calls += 1
        self.prompts.append(messages[-1]["content"])
        reply = self.responses.pop(0)
        return [{"generated_text": messages + [{"role": "assistant", "content": reply}]}]


def _oracle(tmp_path, responses: list[str]) -> LLMOracle:
    # Bypass __init__: it loads a model and needs a running H-reasoner.
    oracle = LLMOracle.__new__(LLMOracle)
    oracle.mq_count = oracle.eq_count = 0
    oracle._pipe = _StubPipe(responses)
    oracle._model_name = "stub"
    oracle._max_new_tokens = 16
    oracle._pad_token_id = 0
    oracle._prompt_style = "ours"
    oracle._verbose = False
    oracle._system_msg = "system"
    oracle._sig = {"A", "B"}
    oracle._roles = set()
    oracle._cache = LLMCache(enabled=True, db_path=str(tmp_path / "cache.db"))
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
        {"role": "user", "content": LLMOracle._build_mq_prompt({"A", "B"}, set(), _GCI)},
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
])
def test_11_manchester_parsing(text, expected):
    assert parse_manchester_gci(text) == expected
