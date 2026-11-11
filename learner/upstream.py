"""
What the original ExactLearner-LLM does, reproduced: how it writes an MQ (the OWL
API's Manchester rendering, its natural-language rewrite), its system prompts, how
it reads an answer, its AxiomSimplifier, and the sampling its runs were answered
with.  This project's own prompts are in learner/llm_prompts.py.
"""

from __future__ import annotations

from learner.el_algorithm import ELConcept, GCI, existential_key


# ---------------------------------------------------------------------------
# The original ExactLearner-LLM's query text
# ---------------------------------------------------------------------------

def _upstream_conjuncts(concept: ELConcept) -> list[ELConcept]:
    return ([ELConcept({a}) for a in concept.atoms]
            + [ELConcept(set(), {e}) for e in concept.existentials])


def _upstream_order(concept: ELConcept) -> tuple:
    """The OWL API 5's order of class expressions: named classes by IRI, then
    intersections, then existentials by property and filler.  owl:Thing is a named
    class whose IRI sorts after the target's names, as it does for the paper's five
    ontologies (their namespaces sort before http://www.w3.org/2002/07/owl#)."""
    parts = _upstream_conjuncts(concept)
    if not parts:
        return (1, 1, "")
    if len(parts) > 1:
        return (2, sorted(map(_upstream_order, parts)))
    if concept.atoms:
        return (1, 0, next(iter(concept.atoms)))
    (role, filler), = concept.existentials
    return (3, role, _upstream_order(filler))


def upstream_manchester(concept: ELConcept, nested: bool = False) -> str:
    """*concept* as upstream's LLMEngine sends it: the OWL API 5's Manchester
    renderer with local names, its line breaks turned into single spaces.  Unlike
    ``concept_to_manchester``, ⊤ is "Thing" and an outermost existential has no
    parentheses."""
    parts = _upstream_conjuncts(concept)
    if not parts:
        return "Thing"
    if len(parts) > 1:
        text = " and ".join(upstream_manchester(p, nested=True) for p in sorted(parts, key=_upstream_order))
    elif concept.atoms:
        return next(iter(concept.atoms))
    else:
        (role, filler), = concept.existentials
        text = f"{role} some {upstream_manchester(filler, nested=True)}"
    return f"({text})" if nested else text


def upstream_gci_text(gci: GCI) -> str:
    return f"{upstream_manchester(gci.lhs)} SubClassOf {upstream_manchester(gci.rhs)}"


def upstream_natural_language(message: str) -> str:
    """Upstream's NLPLLMEngine.addExtraSemantic, kept as it is: it splits the
    Manchester text at the first " and " or " some " before looking at brackets,
    so a nested filler comes out half-rewritten."""
    if " SubClassOf " in message:
        lhs, rhs = message.split(" SubClassOf ", 1)
        return (f"Can {upstream_natural_language(lhs)} be considered a subcategory of "
                f"'{upstream_natural_language(rhs)}'?")
    if " and " in message:
        first, rest = message.split(" and ", 1)
        return f"{upstream_natural_language(first)} that is also {upstream_natural_language(rest)}"
    if " some " in message:
        role, filler = message.split(" some ", 1)
        return f"something that {upstream_natural_language(role)} some {upstream_natural_language(filler)}"
    if message.startswith("(") and message.endswith(")"):
        return upstream_natural_language(message[1:-1])
    return message


# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

# System prompts of the original ExactLearner-LLM (src/main/java/org/configurations/
# exp2/{manchester,nlp}-{simple,advanced}.yml), as the YAML folds and trims them.  The
# natural-language styles share them; only the query text differs (queryFormat: nlp).
_PAPER_ADVANCED_SYSTEM = (
        "You need to classify the following statements as True or False. The statement "
        "will be provided in either Manchester OWL syntax or natural language. Strictly "
        "follow these guidelines:\n"
        "  1. answer with only True or False;\n"
        "  2. entities with has part relation are not in a subclass relation;\n"
        "  3. take a deep breath before answering;\n"
        "  4. if you are unsure about the classification, answer with False."
)
PAPER_SYSTEM_PROMPTS: dict[str, str] = {
    "paper-simple": "Answer with only True or False.",
    "paper-advanced": _PAPER_ADVANCED_SYSTEM,
    "paper-nl-simple": "Answer with only True or False.",
    "paper-nl-advanced": _PAPER_ADVANCED_SYSTEM,
}
PAPER_NL_STYLES = ("paper-nl-simple", "paper-nl-advanced")


# ---------------------------------------------------------------------------
# Sampling, answers and the AxiomSimplifier
# ---------------------------------------------------------------------------

# Ollama's sampling defaults, which upstream's requests (options: num_predict only)
# ran under; the temperature is set per oracle.  Its repeat_last_n (64) window has
# no transformers counterpart: the penalty here covers the whole prompt.
OLLAMA_SAMPLING = {"top_k": 40, "top_p": 0.9, "repetition_penalty": 1.1}


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

