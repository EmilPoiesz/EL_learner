"""
This project's own prompts (the paper's are in learner/upstream.py): the system
messages, the yes/no MQ, and the EQ prompts and their retry lines.  Plain
functions of the scope and H.
"""

from __future__ import annotations

from typing import Sequence

from learner.el_algorithm import GCI
from learner.el_syntax import gci_to_manchester, hypothesis_to_manchester


def _scope(sig: set[str], roles: set[str]) -> tuple[str, str]:
    return ", ".join(sorted(sig)), ", ".join(sorted(roles)) if roles else "(none)"


# ---------------------------------------------------------------------------
# System messages
# ---------------------------------------------------------------------------

PARAMETRIC_SYSTEM_MSG = (
    "You are a precise ontology reasoning assistant. "
    "Follow the output format exactly as specified. "
    "Do not add any explanation, preamble, or punctuation beyond what is asked."
)


def grounded_system_message(sig: set[str], roles: set[str], gcis: set[GCI]) -> str:
    """The MQ system message that teaches the model O (``include_axioms``)."""
    sig_str, roles_str = _scope(sig, roles)
    gcis_str = "\n".join(f"  {gci_to_manchester(g)}" for g in sorted(gcis, key=str))
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


# ---------------------------------------------------------------------------
# MQ (prompt style "ours")
# ---------------------------------------------------------------------------

def mq_prompt(sig: set[str], roles: set[str], gci: GCI) -> str:
    sig_str, roles_str = _scope(sig, roles)
    return (
        f"Concept names in scope: {sig_str}.\n"
        f"Object properties in scope: {roles_str}.\n"
        "Does the target ontology O entail the following GCI?\n"
        f"  {gci_to_manchester(gci)}\n"
        "Reply with exactly one word: yes or no."
    )


# ---------------------------------------------------------------------------
# EQ
# ---------------------------------------------------------------------------

def eq_judge_prompt(sig: set[str], roles: set[str], H: set[GCI]) -> str:
    sig_str, roles_str = _scope(sig, roles)
    return (
        f"Concept names in scope: {sig_str}.\n"
        f"Object properties in scope: {roles_str}.\n\n"
        "Hypothesis H:\n"
        f"{hypothesis_to_manchester(H)}\n\n"
        "Does H capture all subsumptions entailed by the target ontology O?\n"
        "Reply with exactly one word: yes or no."
    )


def eq_counterexample_prompt(sig: set[str], roles: set[str], H: set[GCI],
                             feedback: Sequence[str] = ()) -> str:
    """*feedback*: why earlier answers to this prompt could not be used, one
    line each.  Generation is deterministic, so a retry must change the prompt."""
    sig_str, roles_str = _scope(sig, roles)
    retry = "".join(f"{line}\n" for line in feedback)
    return (
        f"Concept names in scope: {sig_str}.\n"
        f"Object properties in scope: {roles_str}.\n\n"
        "Hypothesis H:\n"
        f"{hypothesis_to_manchester(H)}\n\n"
        + (f"{retry}\n" if retry else "")
        + "Name one GCI that is entailed by target ontology O but is NOT entailed by H.\n"
        "Output exactly one line in Manchester syntax, e.g.: Cat SubClassOf Animal\n"
        "Output nothing else."
    )


def unusable_feedback(reason: str) -> str:
    """The retry line for an answer of this EQ that could not be used."""
    return f"Your previous answer could not be used: {reason}."
