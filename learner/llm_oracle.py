from __future__ import annotations

import logging
import re
import os
from learner.cache.cache import LLMCache
from typing import Callable, Optional, Sequence, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")

from transformers import pipeline, GenerationConfig

from learner.el_algorithm import Oracle, OracleError, ELConcept, GCI
from learner.hypothesis_reasoner import HypothesisReasoner


# ---------------------------------------------------------------------------
# Manchester syntax serialization
# ---------------------------------------------------------------------------

def concept_to_manchester(concept: ELConcept) -> str:
    """Serialize an ELConcept to Manchester syntax."""
    parts: list[str] = []
    for atom in sorted(concept.atoms):
        parts.append(atom)
    
    # x here is the existential ∃r.A, where the x[0] is r and x[1] is A. We sort
    # alphabetically by relation r, then alphabetically by concept name.
    for role, filler in sorted(concept.existentials, key=lambda x: (x[0], str(x[1]))):
        filler_str = concept_to_manchester(filler)
        if (len(filler.atoms) + len(filler.existentials)) > 1: filler_str = "(" + filler_str + ")"
        parts.append(f"({role} some {filler_str})")
    if not parts:
        return "owl:Thing"
    return " and ".join(parts)


def gci_to_manchester(gci: GCI) -> str:
    return f"{concept_to_manchester(gci.lhs)} SubClassOf {concept_to_manchester(gci.rhs)}"


def gci_names(gci: GCI) -> tuple[set[str], set[str]]:
    """The concept names and the role names that occur in *gci*."""
    atoms: set[str] = set()
    roles: set[str] = set()

    def _walk(concept: ELConcept) -> None:
        atoms.update(concept.atoms)
        for role, filler in concept.existentials:
            roles.add(role)
            _walk(filler)

    _walk(gci.lhs)
    _walk(gci.rhs)
    return atoms, roles


def hypothesis_to_manchester(H: set[GCI]) -> str:
    if not H:
        return "  (empty)"
    return "\n".join(f"  {gci_to_manchester(gci)}" for gci in sorted(H, key=str))


# ---------------------------------------------------------------------------
# Manchester syntax parsing
# ---------------------------------------------------------------------------

def parse_manchester_gci(s: str) -> GCI:
    """
    Parse a GCI in Manchester syntax: "C SubClassOf D".  The keywords
    (SubClassOf, and, some) may be in any case, and "SubClassOf:" (Manchester
    frame syntax) is accepted too.

    Raises ValueError if the format is not recognised.
    """
    match = re.match(r'^(.+?)\s+SubClassOf:?\s+(.+)$', s.strip(), re.DOTALL | re.IGNORECASE)
    if not match:
        raise ValueError(f"Cannot parse GCI (expected 'C SubClassOf D'): {s!r}")
    lhs = parse_manchester_concept(match.group(1).strip())
    rhs = parse_manchester_concept(match.group(2).strip())
    return GCI(lhs=lhs, rhs=rhs)


# A concept or role name: word characters, with ':', '.' or '-' inside but not at
# either end (prefixed names such as ex:hasPart, hyphenated ones such as has-part;
# "B." is the name B and a full stop).
_NAME = r'\w(?:[\w:.\-]*\w)?'


def parse_manchester_concept(s: str) -> ELConcept:
    """
    Parse a Manchester syntax EL concept expression into an ELConcept.

    Supports:
      owl:Thing
      A
      A and B
      r some C
      A and (r some C)
      (A and B) and (r some (C and D))
    """
    s = s.strip()
    if s.lower() == "owl:thing":
        return ELConcept()

    parts = _split_top_level_and(s)

    atoms: set[str] = set()
    existentials: set[tuple] = set()

    for part in parts:
        part = part.strip()
        # Strip matching outer parentheses
        while (
            part.startswith("(")
            and part.endswith(")")
            and _matching_paren(part, 0) == len(part) - 1
        ):
            part = part[1:-1].strip()

        some_match = re.match(rf'^({_NAME})\s+some\s+(.+)$', part, re.DOTALL | re.IGNORECASE)
        if some_match:
            role = some_match.group(1)
            filler_str = some_match.group(2).strip()
            while (
                filler_str.startswith("(")
                and filler_str.endswith(")")
                and _matching_paren(filler_str, 0) == len(filler_str) - 1
            ):
                filler_str = filler_str[1:-1].strip()
            existentials.add((role, parse_manchester_concept(filler_str)))
        elif re.match(rf'^{_NAME}$', part):
            if part.lower() != "owl:thing":
                atoms.add(part)
        else:
            raise ValueError(f"Cannot parse Manchester concept fragment: {part!r}")

    return ELConcept(atoms=frozenset(atoms), existentials=frozenset(existentials))


def _matching_paren(s: str, open_pos: int) -> int:
    """Return the index of the closing paren matching s[open_pos]=='('."""
    depth = 0
    for i in range(open_pos, len(s)):
        if s[i] == '(':
            depth += 1
        elif s[i] == ')':
            depth -= 1
            if depth == 0:
                return i
    raise ValueError(f"Unmatched parenthesis in: {s!r}")


def _split_top_level_and(s: str) -> list[str]:
    """Split s on ' and ' tokens that are not inside parentheses."""
    parts: list[str] = []
    depth = 0
    current: list[str] = []
    i = 0
    while i < len(s):
        if s[i] == '(':
            depth += 1
            current.append(s[i])
        elif s[i] == ')':
            depth -= 1
            current.append(s[i])
        elif depth == 0 and s[i:i + 5].lower() == ' and ':
            parts.append(''.join(current))
            current = []
            i += 5
            continue
        else:
            current.append(s[i])
        i += 1
    parts.append(''.join(current))
    return parts


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
        ``signature`` wins.
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
    roles : set[str] | None
        Role names in scope.  ``None``: those in the file's parsed axioms, or
        none without ``ontology_path``.
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
        roles: set[str] | None = None,
    ):
        super().__init__()
        self._h_reasoner = h_reasoner
        self._max_new_tokens = max_new_tokens
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
            parsed_sig, self._O = extract_ontology(ontology_path)
            self._sig = set(signature) if signature is not None else parsed_sig
            self._roles: set[str] = set(roles) if roles is not None else self._extract_roles(self._O)
            self._system_msg = self._build_grounded_system_message(self._sig, self._roles, self._O)
        else:
            if signature is None:
                raise ValueError("Either 'signature' or 'ontology_path' must be provided.")
            self._sig = signature
            self._O: set[GCI] | None = None
            self._roles = set(roles or ())
            self._system_msg = (
                "You are a precise ontology reasoning assistant. "
                "Follow the output format exactly as specified. "
                "Do not add any explanation, preamble, or punctuation beyond what is asked."
            )

    # ------------------------------------------------------------------
    # Grounded system message (teaching step)
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_roles(gcis: set[GCI]) -> set[str]:
        """Collect every role name that appears in any existential in the GCI set."""
        roles: set[str] = set()
        def _walk(concept: ELConcept) -> None:
            for role, filler in concept.existentials:
                roles.add(role)
                _walk(filler)
        for gci in gcis:
            _walk(gci.lhs)
            _walk(gci.rhs)
        return roles

    @staticmethod
    def _build_grounded_system_message(sig: set[str], roles: set[str], gcis: set[GCI]) -> str:
        sig_str   = ", ".join(sorted(sig))
        roles_str = ", ".join(sorted(roles)) if roles else "(none)"
        gcis_str  = "\n".join(f"  {gci_to_manchester(g)}" for g in sorted(gcis, key=str))
        return (
            "You are a precise ontology reasoning assistant.\n"
            "You have been given the following ontology O in Manchester syntax.\n\n"
            f"Concept names: {sig_str}\n"
            f"Object properties: {roles_str}\n\n"
            "Axioms:\n"
            f"{gcis_str}\n\n"
            "Answer questions about what O entails. "
            "Follow the output format exactly as specified. "
            "Do not add explanation, preamble, or punctuation beyond what is asked."
        )

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
    # Prompt construction
    # ------------------------------------------------------------------

    @staticmethod
    def _build_mq_prompt(sig: set[str], roles: set[str], gci: GCI) -> str:
        sig_str   = ", ".join(sorted(sig))
        roles_str = ", ".join(sorted(roles)) if roles else "(none)"
        return (
            f"Concept names in scope: {sig_str}.\n"
            f"Object properties in scope: {roles_str}.\n"
            "Does the target ontology O entail the following GCI?\n"
            f"  {gci_to_manchester(gci)}\n"
            "Reply with exactly one word: yes or no."
        )

    @staticmethod
    def _build_eq_judge_prompt(sig: set[str], roles: set[str], H: set[GCI]) -> str:
        sig_str   = ", ".join(sorted(sig))
        roles_str = ", ".join(sorted(roles)) if roles else "(none)"
        h_str = hypothesis_to_manchester(H)
        return (
            f"Concept names in scope: {sig_str}.\n"
            f"Object properties in scope: {roles_str}.\n\n"
            "Hypothesis H:\n"
            f"{h_str}\n\n"
            "Does H capture all subsumptions entailed by the target ontology O?\n"
            "Reply with exactly one word: yes or no."
        )

    @staticmethod
    def _build_eq_counterexample_prompt(
        sig: set[str], roles: set[str], H: set[GCI], feedback: Sequence[str] = (),
    ) -> str:
        """*feedback*: why earlier answers to this prompt could not be used, one
        line each.  Generation is deterministic, so a retry must change the prompt."""
        sig_str   = ", ".join(sorted(sig))
        roles_str = ", ".join(sorted(roles)) if roles else "(none)"
        h_str = hypothesis_to_manchester(H)
        retry = "".join(f"{line}\n" for line in feedback)
        return (
            f"Concept names in scope: {sig_str}.\n"
            f"Object properties in scope: {roles_str}.\n\n"
            "Hypothesis H:\n"
            f"{h_str}\n\n"
            + (f"{retry}\n" if retry else "")
            + "Name one GCI that is entailed by target ontology O but is NOT entailed by H.\n"
            "Output exactly one line in Manchester syntax, e.g.: Cat SubClassOf Animal\n"
            "Output nothing else."
        )

    # ------------------------------------------------------------------
    # LLM query helper
    # ------------------------------------------------------------------

    def _query(self, prompt: str, parse: Callable[[str], T], label: str = "QUERY") -> T:
        """
        Ask the LLM and return ``parse(response)``.

        A response is cached only once ``parse`` accepts it, so a response that
        cannot be parsed (e.g. reasoning cut off by max_new_tokens) is never
        replayed from the cache.  A cached response that no longer parses, left
        by an older version, is treated as a miss and regenerated.
        """
        messages = [
            {"role": "system", "content": self._system_msg},
            {"role": "user",   "content": prompt},
        ]
        # Part of the cache key: a response generated under another budget is not reused.
        params = {"max_new_tokens": self._max_new_tokens, "do_sample": False}

        # Cache lookup
        cached = self._cache.get(self._model_name, messages, params)
        if cached is not None:
            if self._verbose:
                print(f"\n--- {label} (CACHED) ---")
                print("[PROMPT]\n" + prompt)
                print("[RESPONSE]\n" + cached)
            try:
                return parse(cached)
            except ValueError:
                logger.warning("Cached %s response does not parse; regenerating.", label)

        generation_config = GenerationConfig(**params, pad_token_id=self._pad_token_id)

        outputs = self._pipe(messages, generation_config=generation_config)
        response = outputs[0]["generated_text"][-1]["content"].strip()

        if self._verbose:
            print(f"\n--- {label} ---")
            print("[PROMPT]\n" + prompt)
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
        """
        return self._query(
            self._build_mq_prompt(self._sig, self._roles, gci),
            parse=lambda r: self._parse_yes_no(r, f"MQ for '{gci_to_manchester(gci)}'"),
            label="MQ",
        )

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
            self._build_eq_judge_prompt(self._sig, self._roles, hypothesis),
            parse=lambda r: self._parse_yes_no(r, "EQ judge"),
            label="EQ (judge)",
        )
        if equivalent:
            return None
        reasons: list[str] = []   # why each earlier answer could not be used
        for _ in range(self._EQ_ATTEMPTS):
            feedback = [f"Your previous answer could not be used: {r}." for r in reasons]
            try:
                counterexample = self._query(
                    self._build_eq_counterexample_prompt(self._sig, self._roles, hypothesis, feedback),
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
