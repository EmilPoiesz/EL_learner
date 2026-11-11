"""
LLMOracle: an Oracle whose MQs and EQs a local language model answers.

The text it sends and reads comes from learner/el_syntax.py (EL as text),
learner/llm_prompts.py (this project's prompts) and learner/upstream.py (the
original ExactLearner-LLM's).
"""

from __future__ import annotations

import logging
import os
import re
from collections import Counter
from dataclasses import dataclass
from functools import partial
from typing import Callable, Optional, TypeVar

from transformers import pipeline, GenerationConfig, set_seed

from learner.cache.cache import LLMCache
from learner.el_algorithm import Oracle, OracleError, ELConcept, GCI
from learner.el_syntax import (IS_A_PATTERN, NAME_PATTERN, gci_names, gci_to_manchester, gci_to_nl,
                               gci_to_triple, parse_manchester_gci, parse_nl_gci, parse_triple,
                               without_brackets)
from learner.hypothesis_reasoner import HypothesisReasoner
from learner import llm_prompts as prompts
from learner.llm_prompts import (CHECK_QUESTIONS, CHECK_SYSTEM_PROMPTS, GENERATIONS,
                                 MANCHESTER_ONLY_GENERATIONS, TRIPLE_GENERATIONS)
from learner.upstream import (OLLAMA_SAMPLING, PAPER_NL_STYLES, PAPER_SYSTEM_PROMPTS, desaturate,
                              paper_is_true, rhs_conjuncts, upstream_gci_text,
                              upstream_natural_language)

logger = logging.getLogger(__name__)

T = TypeVar("T")

PROMPT_STYLES = ("ours", *PAPER_SYSTEM_PROMPTS)

# How an EQ is asked: "judge" asks yes/no whether H is complete, then for a
# counterexample; "combined" asks once for a counterexample or "yes".
EQ_MODES = ("judge", "combined")
# How the learner's own MQs are asked (Phase 1 and refinement): "paper", the paper's MQ in the
# run's prompt style; "check", exactly as the counterexample check is asked (EQSettings check,
# check_system, check_question), so that one statement gets one answer within a run.
MQ_ASKED_AS = ("paper", "check")
EQ_CHECKS = ("mq", "nl")
# Generation budget of an EQ answer, at least: a counterexample is a line of text,
# which the 2 tokens of the paper's MQs would cut off.
EQ_MIN_NEW_TOKENS = 128


def after_think(text: str) -> str:
    """The text after the last </think> (all of *text* without one): the answer of a
    model that reasons first, whether its chat template opens <think> (``_thinks``) or
    it writes <think>…</think> itself.  The one place this cut is made."""
    return text.rsplit("</think>", 1)[-1]


def model_defaults_off(model) -> dict:
    """Keyword arguments for generate() that keep the model's own generation defaults
    out of every query.  transformers 4.50-4.57 replace each setting a GenerationConfig
    leaves at the library default with the model's default (``use_model_defaults=None``):
    Olmo-3.1-32B-Think's do_sample=True and temperature 0.6 turned greedy queries into
    sampled ones, and a counterexample temperature of 1.0 into 0.6 (cex20).  With
    ``use_model_defaults=False`` only the special tokens (eos, bos, pad) come from the
    model.  Versions whose generate() has no such parameter get nothing."""
    import inspect
    try:
        params = inspect.signature(model.generate).parameters
    except (TypeError, ValueError, AttributeError):
        return {}
    return {"use_model_defaults": False} if "use_model_defaults" in params else {}


def sampling_params(temperature: float, seed: int) -> dict | None:
    """Generation settings for sampling at *temperature* with Ollama's other
    defaults, or None (greedy) at 0.  The seed only keys the cache."""
    if temperature <= 0:
        return None
    return {"do_sample": True, "temperature": temperature, **OLLAMA_SAMPLING, "seed": seed}


@dataclass(frozen=True)
class EQSettings:
    """
    How LLMOracle asks its equivalence queries.

    mode : str
        ``"judge"`` (default): an EQ asks whether H is complete (yes/no), then
        for a counterexample.  ``"combined"``: one prompt asks for a
        counterexample, or "yes" if H is complete.  Either way, EQs have their
        own system message (``llm_prompts.eq_system_message``) and a budget of
        at least ``EQ_MIN_NEW_TOKENS``.
    check : str
        How a counterexample is checked before it is used (it must be
        O-entailed): ``"mq"`` (default), by an MQ in the run's own prompt style;
        ``"nl"``, with a paper prompt style, by the paper's MQ in natural language
        (same system message), whatever the style.
    check_system : str
        With a paper prompt style, the system prompt of that check: ``"paper"``
        (default: the MQs' own), or a variant from ``CHECK_SYSTEM_PROMPTS``
        (``"s1"`` … ``"s4"``).  The MQs of learning keep the paper's.
    check_question : str
        With a paper prompt style, the question of that check (``CHECK_QUESTIONS``):
        ``"paper"`` (default), the MQ as ``check`` says; ``"every"``, "Is it
        true that every C …?" (``llm_prompts.every_question``).
    memory : bool
        False (default): an EQ's retry prompts list only why that EQ's earlier
        answers failed.  True: they also list, from the first attempt on, the
        counterexamples the check rejected at earlier EQs.
    temperature : float
        0 (default): counterexamples are generated as the other queries are.
        Above 0: only the counterexample prompts are sampled, at this
        temperature with ``OLLAMA_SAMPLING`` and the oracle's seed; the MQs, the
        check and the judge keep the oracle's temperature.
    attempts : int
        Attempts at a usable counterexample per EQ (default 3: the first ask and
        two retries).
    max_new_tokens : int or None
        The generation budget of the counterexample prompts only.  None
        (default): the oracle's, but at least ``EQ_MIN_NEW_TOKENS``.  The judge
        and the check keep that one: they are yes/no answers, while a reasoning
        model must think and then write a statement (cex20).
    generation : str
        How the counterexample is asked for (``llm_prompts.GENERATIONS``):
        ``"default"``, the placeholder examples; ``"g1"``, with the forms allowed
        listed as rules; ``"g2"``, concrete examples about cars; ``"g3"``, as
        "Subject | relation | Object" (``el_syntax.parse_triple``), and the
        feedback lines name statements that way too; ``"g2b"``, ``"g3b"``: G2
        without brackets, G3 with its format explained in prose.  Only with mode
        ``"judge"``; G1, G2 and G2b only in Manchester syntax.
    """

    mode: str = "judge"
    check: str = "mq"
    check_system: str = "paper"
    check_question: str = "paper"
    memory: bool = False
    temperature: float = 0.0
    attempts: int = 3
    generation: str = "default"
    max_new_tokens: int | None = None

    def __post_init__(self):
        if self.mode not in EQ_MODES:
            raise ValueError(f"mode must be one of {EQ_MODES}, not {self.mode!r}")
        if self.check not in EQ_CHECKS:
            raise ValueError(f"check must be one of {EQ_CHECKS}, not {self.check!r}")
        if self.check_system not in CHECK_SYSTEM_PROMPTS:
            raise ValueError(f"check_system must be one of {tuple(CHECK_SYSTEM_PROMPTS)}, "
                             f"not {self.check_system!r}")
        if self.check_question not in CHECK_QUESTIONS:
            raise ValueError(f"check_question must be one of {CHECK_QUESTIONS}, "
                             f"not {self.check_question!r}")
        if self.attempts < 1:
            raise ValueError(f"attempts must be at least 1, not {self.attempts}")
        if self.max_new_tokens is not None and self.max_new_tokens < 1:
            raise ValueError(f"max_new_tokens must be at least 1, not {self.max_new_tokens}")
        if self.generation not in GENERATIONS:
            raise ValueError(f"generation must be one of {GENERATIONS}, not {self.generation!r}")
        if self.generation != "default" and self.mode != "judge":
            raise ValueError(f"generation {self.generation} needs mode 'judge'")

    def describe(self, seed: int = 0) -> str:
        """The settings that differ from the defaults, for a run's header."""
        return (self.mode
                + (", counterexamples checked by a natural-language MQ" if self.check == "nl" else "")
                + (f", check system prompt {self.check_system}" if self.check_system != "paper" else "")
                + (f", check question {self.check_question}" if self.check_question != "paper" else "")
                + (", rejections remembered across EQs" if self.memory else "")
                + (f", counterexamples sampled at temperature {self.temperature}, seed {seed}"
                   if self.temperature > 0 else "")
                + (f", {self.attempts} attempts per EQ" if self.attempts != 3 else "")
                + (f", counterexample prompt {self.generation}" if self.generation != "default" else "")
                + (f", {self.max_new_tokens} tokens per counterexample" if self.max_new_tokens else ""))


# ---------------------------------------------------------------------------
# LLMOracle
# ---------------------------------------------------------------------------

class LLMOracle(Oracle):
    """
    An Oracle backed by a local HuggingFace language model.

    O-entailment queries (MQ and EQ) are answered by prompting the model
    using Manchester syntax.  H-entailment is delegated to a
    HypothesisReasoner (Java/ELK), exactly as in ReasonerOracle.

    Parameters
    ----------
    model_name_or_path : str
        HuggingFace model identifier or local path,
        e.g. "meta-llama/Llama-3.2-1B-Instruct".
    signature : set[str] | None
        Σ_O — the set of atomic concept names in the target ontology.
        Required when ``ontology_path`` is not given; when both are given,
        ``signature`` wins (e.g. the OWL API's, ``utils.java_utils.load_ontology``).
    h_reasoner : HypothesisReasoner
        A running HypothesisReasoner instance that will be kept in sync
        with H and used to answer H-entailment queries.
    max_new_tokens : int
        Maximum tokens the LLM may generate per query.  Reasoning models
        may produce long chain-of-thought before the final yes/no; 1024 is
        a safe default.  Set lower for non-reasoning models to save time.
    device : str | int
        Device for the HuggingFace pipeline, e.g. "cpu", "cuda", or a
        CUDA device index.
    ontology_path : str | None
        Optional path to a target ontology file (.ttl or .owl).  When
        provided the oracle operates in *grounded* mode: the full set of
        parsed GCIs is placed in the system prompt so the LLM reasons
        from the given axioms rather than from its own knowledge.
        The signature and roles are also derived from the file, unless
        ``signature`` / ``roles`` are given.
    include_axioms : bool
        With ``ontology_path``: put O's axioms in the system prompt (True,
        ``--oracle axioms``) or only take the signature and roles from the file (False,
        ``--oracle signature``: the model answers from what it knows, as in the paper).
    prompt_style : str
        ``"ours"`` (default): this project's yes/no prompts.
        ``"paper-simple"`` / ``"paper-advanced"``, and their natural-language
        forms ``"paper-nl-simple"`` / ``"paper-nl-advanced"``: the original ExactLearner-LLM's
        MQ setup — its system prompt, the axiom alone as the user message,
        True/False read with ``paper_is_true``, and its AxiomSimplifier: an MQ
        that H already entails is answered True without the model, and a
        conjunction on the right is asked one conjunct at a time, skipping the
        conjuncts H entails, after both sides are desaturated against H.  EQ
        prompts stay this project's (in natural language with ``paper-nl-*``,
        else in Manchester syntax); use PACOracle for the paper's equivalence
        queries.
    roles : set[str] | None
        Role names in scope.  ``None``: those in the file's parsed axioms, or
        none without ``ontology_path``.
    temperature : float
        0 (default): greedy decoding.  Above 0: sampling at this temperature
        with Ollama's other defaults (``OLLAMA_SAMPLING``), as upstream's
        requests were answered.  Each distinct query is still answered once
        per run, from the cache.
    seed : int
        With ``temperature`` > 0: seeds the sampling, and is part of the cache
        key, so runs with different seeds draw their own answers.
    eq_settings : EQSettings
        How the EQs are asked (see ``EQSettings``); the defaults otherwise.
    mq_asked_as : str
        ``"paper"`` (default): the learner's MQs (Phase 1, refinement) are the paper's MQ
        in the prompt style.  ``"check"``: they are asked exactly as the counterexample
        check is (``EQSettings``), so the model is not asked one statement in two forms
        within a run (cex19: half the accepted counterexamples were denied by the next
        refinement MQ).  Needs a paper prompt style.
    truth : callable, optional
        Whether O entails a GCI (e.g. ELK over the ontology file).  Only to label
        the proposed counterexamples in the log and in ``eq_outcomes``: it is
        consulted after each decision and never changes one, so the model's
        answers alone still decide what is learned.

    A reasoning model whose chat template opens <think> (Olmo 3 Think) answers
    after its reasoning: only the text after </think> is read, and a response
    without </think> (the budget ran out while reasoning) is read as no answer,
    counted in ``truncated``.
    """

    # Set in __init__ from the chat template; class defaults for oracles built without it.
    _thinks = False
    truncated = 0
    _generate_kwargs: dict = {}   # model_defaults_off(model)
    _failed_eqs = 0               # EQs in a row that gave no usable answer
    _mq_as_check = False          # mq_asked_as == "check"
    _truth: Callable[[GCI], bool] | None = None   # labels proposals for the log only
    eq_outcomes: Counter | None = None            # what each counterexample attempt came to

    def __init__(
        self,
        model_name_or_path: str,
        signature: set[str] | None,
        h_reasoner: HypothesisReasoner,
        max_new_tokens: int = 1024,
        device: str | int = "cpu",
        verbose: bool = False,
        ontology_path: str | None = None,
        include_axioms: bool = True,
        prompt_style: str = "ours",
        roles: set[str] | None = None,
        temperature: float = 0.0,
        seed: int = 0,
        eq_settings: EQSettings = EQSettings(),
        truth: Callable[[GCI], bool] | None = None,
        mq_asked_as: str = "paper",
    ):
        super().__init__()
        if prompt_style not in PROMPT_STYLES:
            raise ValueError(f"prompt_style must be one of {PROMPT_STYLES}, not {prompt_style!r}")
        self._prompt_style = prompt_style
        if mq_asked_as not in MQ_ASKED_AS:
            raise ValueError(f"mq_asked_as must be one of {MQ_ASKED_AS}, not {mq_asked_as!r}")
        if mq_asked_as == "check" and prompt_style not in PAPER_SYSTEM_PROMPTS:
            raise ValueError("mq_asked_as 'check' needs a paper prompt style")
        self._mq_as_check = mq_asked_as == "check"
        self._eq = eq_settings
        self._truth = truth
        self.eq_outcomes = Counter()
        self._eq_language = "nl" if prompt_style in PAPER_NL_STYLES else "manchester"
        if eq_settings.generation in MANCHESTER_ONLY_GENERATIONS and self._eq_language != "manchester":
            raise ValueError(f"generation {eq_settings.generation} is written in Manchester "
                             f"syntax only, not with prompt style {prompt_style}")
        self._rejected: tuple[str, ...] = ()   # with memory: the check's rejections so far
        self._failed: tuple[str, ...] = ()     # why the answers of the EQs that just failed could not be used
        self._h_reasoner = h_reasoner
        self._max_new_tokens = max_new_tokens
        self._seed = seed
        self._sampling = sampling_params(temperature, seed)
        if self._sampling or eq_settings.temperature > 0:
            set_seed(seed)
        self._verbose = verbose
        self._pipe = pipeline(
            "text-generation",
            model=model_name_or_path,
            device=device,
        )
        # Models without a pad token (Mistral, Llama, …) pad with EOS anyway; passing
        # it explicitly stops generate() warning about it on every query.
        tokenizer = self._pipe.tokenizer
        self._pad_token_id = (
            tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        )
        self._thinks = tokenizer.apply_chat_template(
            [{"role": "user", "content": "x"}], tokenize=False, add_generation_prompt=True,
        ).rstrip().endswith("<think>")
        self._generate_kwargs = model_defaults_off(self._pipe.model)
        self._model_name = model_name_or_path
        cache_enabled = os.getenv("LLM_CACHE_ENABLED", "1") != "0"
        cache_db = os.getenv("LLM_CACHE_DB", "llm_cache.db")
        store_prompts = os.getenv("LLM_CACHE_STORE_PROMPTS", "0") == "1"
        self._cache = LLMCache(
            enabled=cache_enabled,
            db_path=cache_db,
            store_prompts=store_prompts,
        )

        if ontology_path is not None:
            from utils.owl_parser import extract_ontology
            # Axioms the parsed GCIs leave out; the axioms prompt does not show them.
            self.skipped_axioms: list[str] = []
            parsed_sig, self._O = extract_ontology(ontology_path, self.skipped_axioms)
            self._sig = set(signature) if signature is not None else parsed_sig
            self._roles: set[str] = (set(roles) if roles is not None
                                     else set().union(*(gci_names(g)[1] for g in self._O)))
            if include_axioms:
                self._system_msg = prompts.grounded_system_message(self._sig, self._roles, self._O)
            else:
                self._system_msg = prompts.PARAMETRIC_SYSTEM_MSG
        else:
            if signature is None:
                raise ValueError("Either 'signature' or 'ontology_path' must be provided.")
            self._sig = signature
            self._O: set[GCI] | None = None
            self._roles = set(roles or ())
            self._system_msg = prompts.PARAMETRIC_SYSTEM_MSG

        if prompt_style in PAPER_SYSTEM_PROMPTS:
            system = PAPER_SYSTEM_PROMPTS[prompt_style]
            if ontology_path is not None and include_axioms:
                axioms = "\n".join(f"  {gci_to_manchester(g)}" for g in sorted(self._O, key=str))
                system += f"\n\nOntology axioms:\n{axioms}"
            self._system_msg = system

        with_axioms = ontology_path is not None and include_axioms
        self._eq_system_msg = prompts.eq_system_message(self._O if with_axioms else None,
                                                        self._eq_language)

    # ------------------------------------------------------------------
    # Oracle.signature / Oracle.axioms
    # ------------------------------------------------------------------

    @property
    def axioms(self) -> set[GCI]:
        """The parsed target axioms — only available in grounded mode."""
        if self._O is None:
            raise AttributeError("axioms is only available when ontology_path was provided.")
        return self._O

    @property
    def signature(self) -> set[str]:
        return self._sig

    # ------------------------------------------------------------------
    # H-entailment — delegated to the HypothesisReasoner
    # ------------------------------------------------------------------

    def make_H_MQ(self, H: set[GCI]):
        return self._h_reasoner

    def on_H_add(self, gci: GCI) -> None:
        self._h_reasoner.add(gci)

    # ------------------------------------------------------------------
    # LLM query helper
    # ------------------------------------------------------------------

    def _query(self, prompt: str, parse: Callable[[str], T], label: str = "QUERY",
               axiom: str | None = None, system: str | None = None,
               max_new_tokens: int | None = None, sampling: dict | None = None) -> T:
        """
        Ask the LLM and return ``parse(response)``.

        A response is cached only once ``parse`` accepts it, so a response that
        cannot be parsed (e.g. reasoning cut off by max_new_tokens) is never
        replayed from the cache.  Nor is a reasoning model's response cut off
        before </think>, even where ``parse`` reads "no answer" as False (the
        paper's MQ): it is that query's answer, but not one to replay.  A cached
        response that no longer parses or was cut off, left by an older version,
        is treated as a miss and regenerated.  *axiom*: the
        query's GCI, shown after the prompt in verbose output.  *system*,
        *max_new_tokens*, *sampling*: instead of the oracle's own (for EQs).
        """
        shown = prompt + (f"\n[AXIOM]\n{axiom}" if axiom else "")
        messages = [
            {"role": "system", "content": self._system_msg if system is None else system},
            {"role": "user",   "content": prompt},
        ]
        # Part of the cache key: a response generated under other settings is not reused.
        params = {"max_new_tokens": max_new_tokens or self._max_new_tokens,
                  **(sampling or self._sampling or {"do_sample": False})}

        # Cache lookup
        cached = self._cache.get(self._model_name, messages, params)
        if cached is not None and self._cut_off(cached):
            logger.warning("Cached %s response was cut off while reasoning; regenerating.", label)
            cached = None
        if cached is not None:
            if self._verbose:
                print(f"\n--- {label} (CACHED) ---")
                print("[PROMPT]\n" + shown)
                print("[RESPONSE]\n" + cached)
            try:
                return parse(self._answer(cached))
            except ValueError:
                logger.warning("Cached %s response does not parse; regenerating.", label)

        # The seed only keys the cache; set_seed in __init__ seeded the generator.
        config = {k: v for k, v in params.items() if k != "seed"}
        generation_config = GenerationConfig(**config, pad_token_id=self._pad_token_id)

        outputs = self._pipe(messages, generation_config=generation_config, **self._generate_kwargs)
        response = outputs[0]["generated_text"][-1]["content"].strip()

        if self._verbose:
            print(f"\n--- {label} ---")
            print("[PROMPT]\n" + shown)
            print("[RESPONSE]\n" + response)

        answer = self._answer(response)
        if self._thinks and not answer and self._verbose:
            print("[TRUNCATED: no </think>, read as no answer]")
        result = parse(answer)  # raises ValueError before anything is cached

        # Cache store
        if not self._cut_off(response):
            self._cache.set(self._model_name, messages, response, params)
        return result

    # ------------------------------------------------------------------
    # Reading answers
    # ------------------------------------------------------------------

    def _cut_off(self, response: str) -> bool:
        """A reasoning model's *response* that ends before </think>: no answer."""
        return self._thinks and "</think>" not in response

    def _answer(self, response: str) -> str:
        """The answer in *response*: all of it, or for a model that reasons in
        <think> first (``_thinks``), the text after </think>; "" when there is
        none (counted in ``truncated``)."""
        if not self._thinks:
            return response
        if self._cut_off(response):
            self.truncated += 1
            return ""
        return after_think(response).strip()

    @staticmethod
    def _extract_yes_no(response: str) -> str | None:
        """Return 'yes' or 'no' as the answer given in *response*, or None.

        Reasoning models think before answering, and DeepSeek-style models wrap
        that in <think>…</think>, so only the text after the last </think> is
        read.  The verdict is taken from its last non-empty line, using the
        first yes/no there ("Yes — no axiom contradicts it" is a yes).  Only
        when that line has none is the last yes/no anywhere in the text used.
        """
        text = after_think(response).lower()
        lines = [ln for ln in text.splitlines() if ln.strip()]
        if lines:
            final = re.findall(r'\b(yes|no)\b', lines[-1])
            if final:
                return final[0]
        matches = re.findall(r'\b(yes|no)\b', text)
        return matches[-1] if matches else None

    @classmethod
    def _parse_yes_no(cls, response: str, what: str) -> bool:
        """Return True for 'yes', False for 'no'; raise OracleError otherwise
        (often reasoning cut off by max_new_tokens, which a retry would repeat)."""
        answer = cls._extract_yes_no(response)
        if answer is None:
            raise OracleError(f"Unparseable {what} response: {response!r}")
        return answer == "yes"

    @staticmethod
    def _answer_lines(response: str) -> list[str]:
        """
        The lines of *response* that may hold the counterexample, in the order
        they are tried: the last non-empty line that is not a code fence, then the
        others from the top (Mistral wrote the statement first and an explanation
        under it, jobs 4380986-2703958), each without the wrapping a model may add
        (backticks, quotes, a list dash, an "Answer:" label, a full stop).
        """
        text = after_think(response)
        lines = [ln.strip() for ln in text.splitlines() if ln.strip() and ln.strip().strip("`")]
        if not lines:
            raise ValueError(f"no answer line in {response!r}")

        def unwrap(line: str) -> str:
            line = re.sub(r'^[-*]\s+', '', line)
            line = re.sub(r'^(answer|counterexample|gci)\s*:\s*', '', line, flags=re.IGNORECASE)
            line = line.strip().strip('`"\'').strip()
            return line.rstrip('.').strip()

        return [unwrap(line) for line in [lines[-1], *lines[:-1]]]

    def _parse_counterexample(self, response: str) -> GCI:
        """
        The GCI named in an EQ counterexample *response*, in the EQ language
        (Manchester syntax, or the natural language of ``gci_to_nl``): from the
        first of ``_answer_lines`` that gives one.  Raises ValueError, saying why
        the last line could not be used, when none does (not a statement, or a
        concept name outside Σ or a role outside the roles in scope).
        """
        lines = self._answer_lines(response)
        try:
            return self._parse_line(lines[0])
        except ValueError as error:
            for line in lines[1:]:
                try:
                    return self._parse_line(line)
                except ValueError:
                    pass
            raise error

    def _parse_line(self, line: str) -> GCI:
        """The GCI on one answer *line*: as written, else as ``_repaired_lines``
        reads it.  The error says why the line as written could not be used."""
        try:
            return self._parse_gci_in_scope(line)
        except ValueError as error:
            # Tried only after the line as written, so that a concept named A, or a
            # role used correctly, is never rewritten.
            for repaired in self._repaired_lines(line):
                try:
                    return self._parse_gci_in_scope(repaired)
                except ValueError:
                    pass
            raise error

    def _repaired_lines(self, line: str) -> list[str]:
        """Readings of an unparsable answer *line* that fix slips Mistral made
        (jobs 4380986-87, 4381253-54), most conservative first: an article before
        a name, and a role in scope used without "some" ("Bird has_part Wing",
        "Project SubClassOf employed_by University"); each also inside brackets
        around the whole line, and a tuple "(Cat, eats, Meat)" as "Cat eats Meat"
        (cex14).  For a triple answer, ``_triple_readings``."""
        if self._eq.generation in TRIPLE_GENERATIONS:
            return self._triple_readings(line)
        def add_some(text: str) -> str:
            for role in sorted(self._roles, key=len, reverse=True):
                text = re.sub(rf"(?<![\w:.\-]){re.escape(role)}\s+(?!some\b)(?=[\w(])",
                              f"{role} some ", text, flags=re.IGNORECASE)
            return text

        def add_subclass(text: str) -> str:
            # "Bird has_part some Wing", with no SubClassOf: Bird SubClassOf has_part some Wing.
            if re.search(r"\b(?:subclassof|subcategory)\b", text, flags=re.IGNORECASE):
                return text
            match = re.match(rf"^({NAME_PATTERN})\s+(?:is\s+)?({NAME_PATTERN}\s+some\s+.+)$", text,
                             flags=re.IGNORECASE)
            if not match or match.group(2).split()[0] not in self._roles:
                return text
            if self._eq_language == "nl":
                return f"{match.group(1)} is a subcategory of something that {match.group(2)}"
            return f"{match.group(1)} SubClassOf {match.group(2)}"

        # An English article before a concept name ("A Professor SubClassOf Teacher"),
        # not before a keyword or a role (then it is a concept named A), nor in
        # "is a subcategory of".
        keywords = "|".join(["subcategory", "subclassof", "and", "some", "is", "that",
                             *map(re.escape, self._roles)])

        def strip_articles(text: str) -> str:
            return re.sub(rf"(?<![\w:.\-])(?:a|an|the)\s+(?!(?:{keywords})(?![\w:.\-]))(?=[\w(])",
                          "", text, flags=re.IGNORECASE)

        def tuple_reading(text: str) -> str | None:
            parts = [p.strip() for p in text.split(",")]
            if len(parts) != 3:
                return None
            subject, relation, obj = parts
            if re.fullmatch(IS_A_PATTERN, relation, flags=re.IGNORECASE):
                is_a = "is a subcategory of" if self._eq_language == "nl" else "SubClassOf"
                return f"{subject} {is_a} {obj}"
            return add_subclass(add_some(f"{subject} {relation} {obj}"))

        candidates = []
        for base in dict.fromkeys([line, without_brackets(line)]):
            stripped = strip_articles(base)
            candidates += [base, stripped, add_subclass(add_some(base)),
                           add_subclass(add_some(stripped)), tuple_reading(base)]
        return [c for i, c in enumerate(candidates)
                if c is not None and c != line and c not in candidates[:i]]

    @staticmethod
    def _triple_readings(line: str) -> list[str]:
        """Readings of an unparsable triple *line*: the parts around a colon, each
        also without enclosing brackets.  The model wrote the format before its
        answer ("Subject | is-a | Object: Bird | is-a | Vertebrate") or an
        explanation after it ("Cat | eats | Fish: While …", cex15).  A colon inside
        a name (ex:hasPart) has no space after it, so it does not split."""
        parts = [p.strip() for p in re.split(r":(?:\s+|$)", line) if p.strip()]
        candidates = [r for p in parts for r in (p, without_brackets(p))]
        return [c for i, c in enumerate(candidates) if c != line and c not in candidates[:i]]

    def _parse_gci_in_scope(self, line: str) -> GCI:
        """The GCI written on *line* in the EQ language, or ValueError, saying why
        (not a statement, or names outside the scope)."""
        if self._eq.generation in TRIPLE_GENERATIONS:
            parse, form = parse_triple, "'Subject | relation | Object'"
        elif self._eq_language == "nl":
            parse, form = parse_nl_gci, "'C is a subcategory of D'"
        else:
            parse, form = parse_manchester_gci, "'C SubClassOf D' in Manchester syntax"
        try:
            gci = parse(line)
        except ValueError:
            raise ValueError(f"{line!r} is not of the form {form}")
        atoms, roles = gci_names(gci)
        unknown = sorted(atoms - self._sig)
        unknown_roles = sorted(roles - self._roles)
        problems = []
        if unknown:
            problems.append(f"{', '.join(unknown)} {'is not a concept name' if len(unknown) == 1 else 'are not concept names'} in scope")
        if unknown_roles:
            problems.append(f"{', '.join(unknown_roles)} {'is not an object property' if len(unknown_roles) == 1 else 'are not object properties'} in scope")
        if problems:
            raise ValueError(f"{line!r}: " + "; ".join(problems))
        return gci

    def _parse_combined(self, response: str) -> Optional[GCI]:
        """None when a combined EQ *response* says yes (H is complete), else its
        counterexample, as ``_parse_counterexample``."""
        text = after_think(response)
        lines = [ln.strip() for ln in text.splitlines() if ln.strip().strip("`")]
        statement = re.search(r"subclassof|subcategory", text, flags=re.IGNORECASE)
        if lines and re.match(r"^\W*yes\b", lines[0], flags=re.IGNORECASE) and not statement:
            return None
        return self._parse_counterexample(response)

    # ------------------------------------------------------------------
    # Oracle interface — O-entailment via LLM
    # ------------------------------------------------------------------

    def _MQ(self, gci: GCI) -> bool:
        """
        Membership query: does O entail lhs ⊑ rhs?

        The LLM is prompted for a yes/no answer in Manchester syntax.
        Raises ValueError if the response cannot be parsed as yes or no.
        With a paper prompt style, see ``_paper_MQ``.
        """
        if self._prompt_style in PAPER_SYSTEM_PROMPTS:
            if self._mq_as_check:
                return self._check_style_MQ(gci, "MQ (as check")
            return self._paper_MQ(gci)
        return self._query(
            prompts.mq_prompt(self._sig, self._roles, gci),
            parse=lambda r: self._parse_yes_no(r, f"MQ for '{gci_to_manchester(gci)}'"),
            label="MQ",
        )

    def _paper_MQ(self, gci: GCI, nl: bool | None = None, system: str | None = None,
                  label: str = "MQ", question: Callable[[GCI], str] | None = None) -> bool:
        """
        MQ as the original ExactLearner-LLM's LLMEngine with its AxiomSimplifier:
        True without asking if H entails the GCI; otherwise the right-hand
        conjuncts H does not entail are kept, both sides are desaturated (concept
        names that H makes redundant in a node are dropped), and each remaining
        right-hand conjunct is asked on its own (the axiom alone as the user
        message), stopping at the first that is not True.  *nl*: ask in natural
        language (default: as the prompt style does).  *system*: instead of the
        style's system prompt.  *question*: writes each conjunct's question
        instead (then *nl* does not apply).
        """
        if nl is None:
            nl = self._prompt_style in PAPER_NL_STYLES
        if self._h_reasoner(gci):
            return True
        unknown = [c for c in rhs_conjuncts(gci.rhs) if not self._h_reasoner(GCI(gci.lhs, c))]
        rhs = ELConcept(frozenset().union(*(c.atoms for c in unknown)),
                        frozenset().union(*(c.existentials for c in unknown)))
        # Dropping a name only generalises, so equivalence is "H entails new ⊑ old".
        lhs = desaturate(gci.lhs, lambda c, old=gci.lhs: self._h_reasoner(GCI(c, old)))
        rhs = desaturate(rhs, lambda c, old=rhs: self._h_reasoner(GCI(c, old)))
        for conjunct in rhs_conjuncts(rhs):
            part = GCI(lhs, conjunct)
            text = upstream_gci_text(part)
            if question is not None:
                text = question(part)
            elif nl:
                text = upstream_natural_language(text)
            # The axiom in this project's syntax, for scripts/score_mqs.py.
            if not self._query(text, parse=paper_is_true, label=label, axiom=gci_to_manchester(part),
                               system=system):
                return False
        return True

    def _EQ(self, hypothesis: set[GCI]) -> Optional[GCI]:
        """
        Equivalence query: is H equivalent to O?

        With mode ``"judge"``: ask yes/no whether H captures all of O, and if
        not, ask for one counterexample.  With ``"combined"``: ask once for a
        counterexample, or "yes" if H captures all of O.  H is shown, and the
        counterexample asked for, in natural language with a ``paper-nl-*`` prompt
        style, else in Manchester syntax.

        Returns None if the LLM confirms equivalence, otherwise the counterexample.
        A counterexample is usable when it parses, uses only names in scope, does
        not already follow from H, and the LLM's own MQ accepts it (it must be
        O-entailed).  An unusable one is asked for again, up to ``attempts``
        times in all, with the reasons the earlier answers failed added to the
        prompt (with ``memory``, also the counterexamples rejected at earlier
        EQs).  Raises OracleError when the judge's answer does not parse or no
        attempt gives a usable counterexample.  After such failed EQs, every prompt of
        the next one opens with how many failed in a row (``repeated_feedback``), so a
        repeated EQ is never the same prompt.  With ``verbose``, each step's
        outcome is printed after the exchange, as ``[EQ n] ...``.
        """
        settings, language = self._eq, self._eq_language
        ask = dict(system=self._eq_system_msg,
                   max_new_tokens=max(self._max_new_tokens, EQ_MIN_NEW_TOKENS))
        plain = gci_to_nl if language == "nl" else gci_to_manchester
        # The feedback names statements as the answer is asked for: as triples with G3.
        render = (lambda gci: gci_to_triple(gci) or plain(gci)) if settings.generation in TRIPLE_GENERATIONS else plain
        # After failed EQs every prompt of this one says so, so that it is never the same
        # prompt again (a greedy model would give the same answer, or the cache would).
        again = [prompts.repeated_feedback(self._failed_eqs)] if self._failed_eqs else []
        if settings.mode == "judge":
            try:
                equivalent = self._query(
                    prompts.eq_judge_prompt(self._sig, self._roles, hypothesis, language, again),
                    parse=lambda r: self._parse_yes_no(r, "EQ judge"),
                    label="EQ (judge)", **ask,
                )
            except OracleError:
                self._failed_eqs += 1
                raise
            self._eq_log(f"judge: {'yes, H is complete' if equivalent else 'no'}")
            if equivalent:
                self._failed_eqs = 0
                return None
        if settings.mode == "combined":
            build, parse = prompts.eq_combined_prompt, self._parse_combined
            label = "EQ (counterexample or yes)"
        else:
            build = partial(prompts.eq_counterexample_prompt, generation=settings.generation)
            parse = self._parse_counterexample
            label = "EQ (counterexample)"
        sampling = sampling_params(settings.temperature, self._seed)
        generate = {**ask, **({"max_new_tokens": settings.max_new_tokens} if settings.max_new_tokens else {})}
        reasons: list[str] = []   # why each earlier answer could not be used
        # Carried into the next EQ if this one fails (the learner may ask again with H
        # unchanged, and the same prompt would get the same cached answer); a
        # rejection only without memory, which lists it anyway.
        carry: list[str] = []
        earlier = again + [prompts.rejected_earlier_feedback(text) for text in self._rejected]
        earlier += [prompts.unusable_feedback(r) for r in self._failed]
        for attempt in range(1, settings.attempts + 1):
            feedback = earlier + [prompts.unusable_feedback(r) for r in reasons]
            prompt = build(self._sig, self._roles, hypothesis, feedback, language)
            try:
                counterexample = self._query(prompt, parse=parse, label=label,
                                             sampling=sampling, **generate)
            except ValueError as exc:
                reasons.append(str(exc))
                carry.append(reasons[-1])
                self._outcome("unusable")
                self._eq_log(f"attempt {attempt}: unusable, {exc}")
                continue
            if counterexample is None:
                self._eq_log(f"attempt {attempt}: yes, H is complete")
                self._failed, self._failed_eqs = (), 0
                return None
            if self._h_reasoner(counterexample):
                reasons.append(f"'{render(counterexample)}' already follows from H")
                carry.append(reasons[-1])
                self._outcome("follows from H")
                self._eq_log(f"attempt {attempt}: {counterexample} already follows from H")
                continue
            if self._check_counterexample(counterexample):
                label = self._label(counterexample, "accepted")
                self._eq_log(f"attempt {attempt}: counterexample {counterexample}{label}")
                self._failed, self._failed_eqs = (), 0
                return counterexample
            # The model named an axiom it then rejects: its answers are inconsistent.
            reasons.append(f"asked whether '{render(counterexample)}' is true in O, you answered no")
            if not settings.memory:
                carry.append(reasons[-1])
            if settings.memory and render(counterexample) not in self._rejected:
                self._rejected += (render(counterexample),)
            label = self._label(counterexample, "rejected")
            self._eq_log(f"attempt {attempt}: {counterexample} rejected by the model's own MQ{label}")
        self._eq_log(f"no usable counterexample in {settings.attempts} attempts")
        self._failed += tuple(r for r in dict.fromkeys(carry) if r not in self._failed)
        self._failed_eqs += 1
        raise OracleError(
            f"no usable counterexample in {settings.attempts} attempts: " + "; ".join(reasons)
        )

    def _check_counterexample(self, gci: GCI) -> bool:
        """The model's own MQ on a counterexample, as the settings' ``check`` and
        ``check_system`` say; counted as an MQ."""
        check, system, question = self._eq.check, self._eq.check_system, self._eq.check_question
        if self._prompt_style in PAPER_SYSTEM_PROMPTS and (
                check == "nl" or system != "paper" or question != "paper"):
            self.mq_count += 1
            return self._check_style_MQ(gci, "MQ (check")
        return self.MQ(gci)

    def _check_style_MQ(self, gci: GCI, label: str) -> bool:
        """The paper's MQ as the counterexample check asks it: in natural language with
        ``check == "nl"``, under the check's system prompt, with the check's question."""
        check, system, question = self._eq.check, self._eq.check_system, self._eq.check_question
        return self._paper_MQ(
            gci, nl=True if check == "nl" else None,
            system=CHECK_SYSTEM_PROMPTS[system],
            question=prompts.every_question if question == "every" else None,
            label=f"{label}, {system} system prompt" + (", every question)" if question == "every" else ")"))

    def _outcome(self, kind: str) -> None:
        if self.eq_outcomes is None:
            self.eq_outcomes = Counter()
        self.eq_outcomes[kind] += 1

    def _label(self, gci: GCI, decision: str) -> str:
        """Count a checked proposal as accepted or rejected, and, with ``truth``, as true
        or false in O; the log's " [true in O]" / " [false in O]".  Called after the
        decision, which it does not touch."""
        if self._truth is None:
            self._outcome(decision)
            return ""
        true = bool(self._truth(gci))
        self._outcome(f"{'true' if true else 'false'} {decision}")
        return " [true in O]" if true else " [false in O]"

    def _eq_log(self, outcome: str) -> None:
        if self._verbose:
            print(f"\n[EQ {self.eq_count}] {outcome}")

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def close(self) -> None:
        try:
            self._h_reasoner.close()
        except Exception:
            logger.debug("Error closing H-reasoner", exc_info=True)

        try:
            self._cache.close()
        except Exception:
            logger.debug("Error closing cache", exc_info=True)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
