from __future__ import annotations

import logging
import re
import os
from learner.cache.cache import LLMCache
from typing import Optional

logger = logging.getLogger(__name__)

from transformers import pipeline, GenerationConfig

from learner.el_algorithm import Oracle, ELConcept, GCI
from learner.hypothesis_reasoner import HypothesisReasoner


# ---------------------------------------------------------------------------
# Manchester syntax serialization
# ---------------------------------------------------------------------------

def concept_to_manchester(concept: ELConcept) -> str:
    """Serialize an ELConcept to Manchester syntax."""
    parts: list[str] = []
    for atom in sorted(concept.atoms):
        parts.append(atom)
    for role, filler in sorted(concept.existentials, key=lambda x: x[0]):
        filler_str = concept_to_manchester(filler)
        if (len(filler.atoms) + len(filler.existentials)) > 1:
            parts.append(f"({role} some ({filler_str}))")
        else:
            parts.append(f"({role} some {filler_str})")
    if not parts:
        return "owl:Thing"
    return " and ".join(parts)


def gci_to_manchester(gci: GCI) -> str:
    return f"{concept_to_manchester(gci.lhs)} SubClassOf {concept_to_manchester(gci.rhs)}"


def hypothesis_to_manchester(H: set[GCI]) -> str:
    if not H:
        return "  (empty)"
    return "\n".join(f"  {gci_to_manchester(gci)}" for gci in sorted(H, key=str))


# ---------------------------------------------------------------------------
# Manchester syntax parsing
# ---------------------------------------------------------------------------

def parse_manchester_gci(s: str) -> GCI:
    """
    Parse a GCI in Manchester syntax: "C SubClassOf D".

    Raises ValueError if the format is not recognised.
    """
    match = re.match(r'^(.+?)\s+SubClassOf\s+(.+)$', s.strip(), re.DOTALL)
    if not match:
        raise ValueError(f"Cannot parse GCI (expected 'C SubClassOf D'): {s!r}")
    lhs = parse_manchester_concept(match.group(1).strip())
    rhs = parse_manchester_concept(match.group(2).strip())
    return GCI(lhs=lhs, rhs=rhs)


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
    if s == "owl:Thing":
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

        some_match = re.match(r'^(\w+)\s+some\s+(.+)$', part, re.DOTALL)
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
        elif re.match(r'^\w[\w:]*$', part):
            if part != "owl:Thing":
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
        elif depth == 0 and s[i:i + 5] == ' and ':
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
    signature : set[str]
        Σ_O — the set of atomic concept names in the target ontology.
    h_reasoner : HypothesisReasoner
        A running HypothesisReasoner instance that will be kept in sync
        with H and used to answer H-entailment queries.
    max_new_tokens : int
        Maximum tokens the LLM may generate per query.  64 is sufficient
        for yes/no MQ answers; EQ responses may need more.
    device : str | int
        Device for the HuggingFace pipeline, e.g. "cpu", "cuda", or a
        CUDA device index.
    """

    def __init__(
        self,
        model_name_or_path: str,
        signature: set[str],
        h_reasoner: HypothesisReasoner,
        max_new_tokens: int = 128,
        device: str | int = "cpu",
        verbose: bool = False,
    ):
        super().__init__()
        self._sig = signature
        self._h_reasoner = h_reasoner
        self._max_new_tokens = max_new_tokens
        self._verbose = verbose
        self._pipe = pipeline(
            "text-generation",
            model=model_name_or_path,
            device=device,
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

    # ------------------------------------------------------------------
    # Oracle.signature
    # ------------------------------------------------------------------

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
    def _build_mq_prompt(sig: set[str], gci: GCI) -> str:
        sig_str = ", ".join(sorted(sig))
        return (
            f"Concept names in scope: {sig_str}.\n"
            "Does the target ontology O entail the following GCI?\n"
            f"  {gci_to_manchester(gci)}\n"
            "Reply with exactly one word: yes or no."
        )

    @staticmethod
    def _build_eq_judge_prompt(sig: set[str], H: set[GCI]) -> str:
        sig_str = ", ".join(sorted(sig))
        h_str = hypothesis_to_manchester(H)
        return (
            f"Concept names in scope: {sig_str}.\n\n"
            "Hypothesis H:\n"
            f"{h_str}\n\n"
            "Does H capture all subsumptions entailed by the target ontology O?\n"
            "Reply with exactly one word: yes or no."
        )

    @staticmethod
    def _build_eq_counterexample_prompt(sig: set[str], H: set[GCI]) -> str:
        sig_str = ", ".join(sorted(sig))
        h_str = hypothesis_to_manchester(H)
        return (
            f"Concept names in scope: {sig_str}.\n\n"
            "Hypothesis H:\n"
            f"{h_str}\n\n"
            "Name one GCI that is entailed by target ontology O but is NOT entailed by H.\n"
            "Output exactly one line in Manchester syntax, e.g.: Cat SubClassOf Animal\n"
            "Output nothing else."
        )

    # ------------------------------------------------------------------
    # LLM query helper
    # ------------------------------------------------------------------

    def _query(self, prompt: str, label: str = "QUERY") -> str:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a precise ontology reasoning assistant. "
                    "Follow the output format exactly as specified. "
                    "Do not add any explanation, preamble, or punctuation beyond what is asked."
                ),
            },
            {"role": "user", "content": prompt},
        ]

        # Cache lookup
        cached = self._cache.get(self._model_name, messages)
        if cached is not None:
            if self._verbose:
                print(f"\n--- {label} (CACHED) ---")
                print("[PROMPT]\n" + prompt)
                print("[RESPONSE]\n" + cached)
            return cached

        generation_config = GenerationConfig(
            max_new_tokens=self._max_new_tokens,
            do_sample=False,
        )

        outputs = self._pipe(messages, generation_config=generation_config)
        response = outputs[0]["generated_text"][-1]["content"].strip()

        # Cache store
        self._cache.set(self._model_name, messages, response)

        if self._verbose:
            print(f"\n--- {label} ---")
            print("[PROMPT]\n" + prompt)
            print("[RESPONSE]\n" + response)

        return response

    # ------------------------------------------------------------------
    # Oracle interface — O-entailment via LLM
    # ------------------------------------------------------------------

    def _MQ(self, gci: GCI) -> bool:
        """
        Membership query: does O entail lhs ⊑ rhs?

        The LLM is prompted for a yes/no answer in Manchester syntax.
        Raises ValueError if the response cannot be parsed as yes or no.
        """
        response = self._query(self._build_mq_prompt(self._sig, gci), label="MQ")
        lower = response.lower()
        if lower.startswith("yes"):
            return True
        if lower.startswith("no"):
            return False
        raise ValueError(
            f"Unparseable MQ response for '{gci_to_manchester(gci)}': {response!r}"
        )

    def _EQ(self, hypothesis: set[GCI]) -> Optional[GCI]:
        """
        Equivalence query: is H equivalent to O?

        Step 1 — ask the LLM yes/no whether H captures all of O.
        Step 2 — if no, ask for one counterexample GCI in Manchester syntax.

        Returns None if the LLM confirms equivalence, otherwise parses and
        returns the counterexample GCI.
        Raises ValueError if either response cannot be parsed.
        """
        judge = self._query(self._build_eq_judge_prompt(self._sig, hypothesis), label="EQ (judge)").lower()
        if judge.startswith("yes"):
            return None
        if not judge.startswith("no"):
            raise ValueError(f"Unparseable EQ judge response: {judge!r}")
        ce = self._query(self._build_eq_counterexample_prompt(self._sig, hypothesis), label="EQ (counterexample)")
        return parse_manchester_gci(ce.strip())

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
