"""
What the original ExactLearner-LLM does, reproduced: its system prompts, how it
reads an answer, and its AxiomSimplifier.  This project's own prompts are in
learner/llm_prompts.py.
"""

from __future__ import annotations

from learner.el_algorithm import ELConcept, existential_key


# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

# System prompts of the original ExactLearner-LLM (src/main/java/org/configurations/
# experiments/manchester-{simple,advanced}.yml), as the YAML folds and trims them.
PAPER_SYSTEM_PROMPTS: dict[str, str] = {
    "paper-simple": "Answer with only True or False.",
    "paper-advanced": (
        "You need to classify the following statements as True or False. The statement "
        "will be provided in either Manchester OWL syntax or natural language. Strictly "
        "follow these guidelines:\n"
        "  1. answer with only True or False;\n"
        "  2. entities with has part relation are not in a subclass relation;\n"
        "  3. take a deep breath before answering;\n"
        "  4. if you are unsure about the classification, answer with False."
    ),
}


# ---------------------------------------------------------------------------
# Answers and the AxiomSimplifier
# ---------------------------------------------------------------------------

def paper_is_true(response: str) -> bool:
    """
    Upstream's ``Result.isStrictlyTrue``: True only if the response, lowercased,
    without full stops and literal "\\n" and trimmed, is exactly "true".
    Anything else, including an unparseable answer, counts as False.
    """
    return response.lower().replace(".", "").replace("\\n", "").strip() == "true"


def rhs_conjuncts(concept: ELConcept) -> list[ELConcept]:
    """The conjuncts of a concept: each atom and each existential on its own."""
    return (
        [ELConcept({a}) for a in sorted(concept.atoms)]
        + [ELConcept(existentials={e}) for e in sorted(concept.existentials, key=existential_key)]
    )
