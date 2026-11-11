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
from typing import Callable, Optional, TypeVar

from transformers import pipeline, GenerationConfig, set_seed

from learner.cache.cache import LLMCache
from learner.el_algorithm import Oracle, OracleError, ELConcept, GCI
from learner.el_syntax import gci_names, gci_to_manchester, parse_manchester_gci
from learner.hypothesis_reasoner import HypothesisReasoner
from learner import llm_prompts as prompts
from learner.upstream import (OLLAMA_SAMPLING, PAPER_NL_STYLES, PAPER_SYSTEM_PROMPTS, desaturate,
                              paper_is_true, rhs_conjuncts, upstream_gci_text,
                              upstream_natural_language)

logger = logging.getLogger(__name__)

T = TypeVar("T")

PROMPT_STYLES = ("ours", *PAPER_SYSTEM_PROMPTS)


def sampling_params(temperature: float, seed: int) -> dict | None:
    """Generation settings for sampling at *temperature* with Ollama's other
    defaults, or None (greedy) at 0.  The seed only keys the cache."""
    if temperature <= 0:
        return None
    return {"do_sample": True, "temperature": temperature, **OLLAMA_SAMPLING, "seed": seed}


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
        from the given axioms rather than from parametric knowledge.
        The signature and roles are also derived from the file, unless
        ``signature`` / ``roles`` are given.
    include_axioms : bool
        With ``ontology_path``: put O's axioms in the system prompt (True,
        "taught") or only take the signature and roles from the file (False,
        "parametric": the model answers from what it knows, as in the paper).
    prompt_style : str
        ``"ours"`` (default): this project's yes/no prompts.
        ``"paper-simple"`` / ``"paper-advanced"``, and their natural-language
        forms ``"paper-nl-simple"`` / ``"paper-nl-advanced"``: the original ExactLearner-LLM's
        MQ setup — its system prompt, the axiom alone as the user message,
        True/False read with ``paper_is_true``, and its AxiomSimplifier: an MQ
        that H already entails is answered True without the model, and a
        conjunction on the right is asked one conjunct at a time, skipping the
        conjuncts H entails, after both sides are desaturated against H.  EQ
        prompts stay this project's; use PACOracle for the paper's equivalence
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
    """

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
    ):
        super().__init__()
        if prompt_style not in PROMPT_STYLES:
            raise ValueError(f"prompt_style must be one of {PROMPT_STYLES}, not {prompt_style!r}")
        self._prompt_style = prompt_style
        self._h_reasoner = h_reasoner
        self._max_new_tokens = max_new_tokens
        self._sampling = sampling_params(temperature, seed)
        if self._sampling:
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
            # Axioms the parsed GCIs leave out; the taught prompt does not show them.
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
               axiom: str | None = None) -> T:
        """
        Ask the LLM and return ``parse(response)``.

        A response is cached only once ``parse`` accepts it, so a response that
        cannot be parsed (e.g. reasoning cut off by max_new_tokens) is never
        replayed from the cache.  A cached response that no longer parses, left
        by an older version, is treated as a miss and regenerated.  *axiom*: the
        query's GCI, shown after the prompt in verbose output.
        """
        shown = prompt + (f"\n[AXIOM]\n{axiom}" if axiom else "")
        messages = [
            {"role": "system", "content": self._system_msg},
            {"role": "user",   "content": prompt},
        ]
        # Part of the cache key: a response generated under other settings is not reused.
        params = {"max_new_tokens": self._max_new_tokens, **(self._sampling or {"do_sample": False})}

        # Cache lookup
        cached = self._cache.get(self._model_name, messages, params)
        if cached is not None:
            if self._verbose:
                print(f"\n--- {label} (CACHED) ---")
                print("[PROMPT]\n" + shown)
                print("[RESPONSE]\n" + cached)
            try:
                return parse(cached)
            except ValueError:
                logger.warning("Cached %s response does not parse; regenerating.", label)

        # The seed only keys the cache; set_seed in __init__ seeded the generator.
        config = {k: v for k, v in params.items() if k != "seed"}
        generation_config = GenerationConfig(**config, pad_token_id=self._pad_token_id)

        outputs = self._pipe(messages, generation_config=generation_config)
        response = outputs[0]["generated_text"][-1]["content"].strip()

        if self._verbose:
            print(f"\n--- {label} ---")
            print("[PROMPT]\n" + shown)
            print("[RESPONSE]\n" + response)

        result = parse(response)  # raises ValueError before anything is cached

        # Cache store
        self._cache.set(self._model_name, messages, response, params)
        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_yes_no(response: str) -> str | None:
        """Return 'yes' or 'no' as the answer given in *response*, or None.

        Reasoning models think before answering, and DeepSeek-style models wrap
        that in <think>…</think>, so only the text after the last </think> is
        read.  The verdict is taken from its last non-empty line, using the
        first yes/no there ("Yes — no axiom contradicts it" is a yes).  Only
        when that line has none is the last yes/no anywhere in the text used.
        """
        text = response.rsplit("</think>", 1)[-1].lower()
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
    def _answer_line(response: str) -> str:
        """
        The line of *response* that holds a counterexample: the last non-empty
        line that is not a code fence, without the wrapping a model may add
        (backticks, quotes, a list dash, an "Answer:" label, a full stop).
        """
        text = response.rsplit("</think>", 1)[-1]
        lines = [ln.strip() for ln in text.splitlines() if ln.strip() and ln.strip().strip("`")]
        if not lines:
            raise ValueError(f"no answer line in {response!r}")
        line = lines[-1]
        line = re.sub(r'^[-*]\s+', '', line)
        line = re.sub(r'^(answer|counterexample|gci)\s*:\s*', '', line, flags=re.IGNORECASE)
        line = line.strip().strip('`"\'').strip()
        return line.rstrip('.').strip()

    def _parse_counterexample(self, response: str) -> GCI:
        """
        The GCI named in an EQ counterexample *response*.  Raises ValueError,
        saying why, when the answer line does not parse or uses a concept name
        outside Σ or a role outside the roles in scope.
        """
        line = self._answer_line(response)
        try:
            gci = parse_manchester_gci(line)
        except ValueError:
            raise ValueError(f"{line!r} is not of the form 'C SubClassOf D' in Manchester syntax")
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
            return self._paper_MQ(gci)
        return self._query(
            prompts.mq_prompt(self._sig, self._roles, gci),
            parse=lambda r: self._parse_yes_no(r, f"MQ for '{gci_to_manchester(gci)}'"),
            label="MQ",
        )

    def _paper_MQ(self, gci: GCI) -> bool:
        """
        MQ as the original ExactLearner-LLM's LLMEngine with its AxiomSimplifier:
        True without asking if H entails the GCI; otherwise the right-hand
        conjuncts H does not entail are kept, both sides are desaturated (concept
        names that H makes redundant in a node are dropped), and each remaining
        right-hand conjunct is asked on its own (the axiom alone as the user
        message, as the OWL API renders it, or in upstream's natural language
        with a ``paper-nl-*`` style), stopping at the first that is not True.
        """
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
            if nl:
                text = upstream_natural_language(text)
            # The axiom in this project's syntax, for scripts/score_mqs.py.
            if not self._query(text, parse=paper_is_true, label="MQ", axiom=gci_to_manchester(part)):
                return False
        return True

    # Attempts at a usable counterexample per EQ (the first ask and two retries).
    _EQ_ATTEMPTS = 3

    def _EQ(self, hypothesis: set[GCI]) -> Optional[GCI]:
        """
        Equivalence query: is H equivalent to O?

        Step 1 — ask the LLM yes/no whether H captures all of O.
        Step 2 — if no, ask for one counterexample GCI in Manchester syntax.

        Returns None if the LLM confirms equivalence, otherwise the counterexample.
        A counterexample is usable when it parses, uses only names in scope, and
        the LLM's own MQ accepts it (it must be O-entailed).  An unusable one is
        asked for again, up to ``_EQ_ATTEMPTS`` times in all, with the reasons the
        earlier answers failed added to the prompt.  Raises OracleError when the
        judge's answer does not parse or no attempt gives a usable counterexample.
        Whether H already entails it is checked by learn_el_terminology.
        """
        equivalent = self._query(
            prompts.eq_judge_prompt(self._sig, self._roles, hypothesis),
            parse=lambda r: self._parse_yes_no(r, "EQ judge"),
            label="EQ (judge)",
        )
        if equivalent:
            return None
        reasons: list[str] = []   # why each earlier answer could not be used
        for _ in range(self._EQ_ATTEMPTS):
            feedback = [prompts.unusable_feedback(r) for r in reasons]
            try:
                counterexample = self._query(
                    prompts.eq_counterexample_prompt(self._sig, self._roles, hypothesis, feedback),
                    parse=self._parse_counterexample,
                    label="EQ (counterexample)",
                )
            except ValueError as exc:
                reasons.append(str(exc))
                continue
            if self.MQ(counterexample):
                return counterexample
            # The model named an axiom it then rejects: its answers are inconsistent.
            reasons.append(
                f"asked whether O entails '{gci_to_manchester(counterexample)}', you answered no"
            )
        raise OracleError(
            f"no usable counterexample in {self._EQ_ATTEMPTS} attempts: " + "; ".join(reasons)
        )

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
