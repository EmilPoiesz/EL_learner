"""
This project's own prompts (the paper's are in learner/upstream.py): the system
messages, the yes/no MQ, the EQ prompts and their retry lines, and the system
prompts of the counterexample check.  Plain functions of the scope and H; the
prompt variants of the EQ experiments (cex1-cex20) are made here.
"""

from __future__ import annotations

from typing import Sequence

from learner.el_algorithm import GCI
from learner.el_syntax import gci_to_manchester, gci_to_nl, hypothesis_to_manchester, hypothesis_to_nl


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


_EQ_SYSTEM_INTRO = (
    "You are an expert in ontology engineering. You are helping to learn a target "
    "ontology O step by step. You will be shown the concept names and object "
    "properties in scope and the hypothesis H, the statements learned so far. A "
    "counterexample is a statement that is true in O but does not follow from H."
)
_EQ_SYSTEM_OUTRO = "Follow the output format exactly as specified, without explanation."
# The model's own knowledge of the domain is O (--oracle signature, the paper's setting).
EQ_SYSTEM_SIGNATURE = (
    f"{_EQ_SYSTEM_INTRO} O describes the domain that these names are about; answer "
    f"from your own knowledge of that domain. {_EQ_SYSTEM_OUTRO}"
)


def eq_system_message(axioms: set[GCI] | None, language: str = "manchester") -> str:
    """The EQ system message: ``EQ_SYSTEM_SIGNATURE``, or with *axioms* (--oracle axioms)
    O itself, in the EQ language."""
    if axioms is None:
        return EQ_SYSTEM_SIGNATURE
    render = gci_to_nl if language == "nl" else gci_to_manchester
    text = "\n".join(f"  {render(g)}" for g in sorted(axioms, key=str))
    return f"{_EQ_SYSTEM_INTRO} O is:\n{text}\n\n{_EQ_SYSTEM_OUTRO}"


# ---------------------------------------------------------------------------
# Counterexample check
# ---------------------------------------------------------------------------

# System prompts for the check of an EQ counterexample (``eq_check_system``), variants
# of the paper's advanced one. Under its rule 2 Mistral answered False to every
# has_part MQ (1143 in the animals logs, experiment cex6), so no has_part
# counterexample could pass. S2 states what the rule presumably means; S3 adds how
# EL statements read. The examples are from another domain (cars), so that they say
# nothing about a target ontology; the natural-language ones are worded as the check
# asks (upstream_natural_language).
_CHECK_RULES_HEAD = (
    "You need to classify the following statements as True or False. The statement "
    "will be provided in either Manchester OWL syntax or natural language. Strictly "
    "follow these guidelines:\n"
    "  1. answer with only True or False;\n"
    "  2. a part is not a subcategory of the whole it belongs to: \"Can Wheel be "
    "considered a subcategory of 'Car'?\" is False. But having a part can be true of "
    "every member of a category: \"Can Car be considered a subcategory of 'something "
    "that has_part some Wheel'?\" is True, since every car has a wheel;\n"
    "  3. take a deep breath before answering;\n"
    "  4. if you are unsure about the classification, answer with False."
)
_CHECK_EL_READING = (
    "\n\nHow to read the statements:\n"
    "  - \"A is a subcategory of B\" (A SubClassOf B): every A is a B.\n"
    "  - \"A is a subcategory of something that r some B\" (A SubClassOf r some B): "
    "every A is related by r to at least one B.\n"
    "  - \"A that is also B\" (A and B): something that is both an A and a B.\n"
    "A statement is True when it holds for every member of the category on the left."
)
# S4 adds what the check got wrong in cex12-cex17, where it accepted 30 false
# counterexamples, most of them again and again (the analysis "Is the check
# needed?" after cex17): a property of some members taken for all (Professor ⊑ ∃teaches.Project,
# Elephant ⊑ ∃has_part.Leg), a subcategory the wrong way round (Teacher ⊑ Professor),
# a relation the wrong way round (Egg ⊑ ∃lays.Bird).  Examples about cars, as in S2.
_CHECK_EVERY_MEMBER = (
    "\n\nA statement about a category is True only if it holds for every member of the "
    "category, by definition, not just for some or for typical members. It must also hold "
    "in the direction it is written:\n"
    "  - every Car has_part some Wheel: True;\n"
    "  - every Car has_part some Sunroof: False, only some cars have one;\n"
    "  - every Vehicle is a Car: False, it is the other way round: every Car is a Vehicle;\n"
    "  - every Wheel has_part some Car: False, the relation goes the other way: the car "
    "has the wheel."
)
CHECK_SYSTEM_PROMPTS: dict[str, str | None] = {
    "paper": None,                                   # the run's own MQ system prompt
    "s1": "Answer with only True or False.",         # the paper's simple one (upstream.py)
    "s2": _CHECK_RULES_HEAD,                         # rule 2 restated
    "s3": _CHECK_RULES_HEAD + _CHECK_EL_READING,     # S2 + how EL statements read
    "s4": _CHECK_RULES_HEAD + _CHECK_EVERY_MEMBER,   # S2 + every member, and direction
}

# The question the check asks (EQSettings.check_question): "paper", the paper's MQ
# (upstream_natural_language with --eq-check nl: "Can Teacher be considered a
# subcategory of 'Professor'?"), or "every", which says what the statement claims:
# "Is it true that every Teacher is a Professor?".  "Can … be considered" may invite
# a yes to what holds for some members only.
CHECK_QUESTIONS = ("paper", "every")


def _article(name: str) -> str:
    return "an" if name[:1].lower() in "aeiou" else "a"


def _concept_phrase(concept) -> str:
    """A concept as a noun phrase: "Animal that is also Mammal that is related by eats
    to some Meat"; "thing" for owl:Thing."""
    atoms = sorted(concept.atoms)
    head = " that is also ".join(atoms) if atoms else "thing"
    clauses = [f"is related by {role} to some {_concept_phrase(filler)}"
               for role, filler in sorted(concept.existentials, key=lambda x: (x[0], str(x[1])))]
    return " that ".join([head, *clauses]) if clauses else head


def every_question(gci: GCI) -> str:
    """The check as "Is it true that every C …?": "every Bird is a Vertebrate",
    "every Bird is related by has_part to some Wing"."""
    parts = [f"is {_article(a)} {a}" for a in sorted(gci.rhs.atoms)]
    parts += [f"is related by {role} to some {_concept_phrase(filler)}"
              for role, filler in sorted(gci.rhs.existentials, key=lambda x: (x[0], str(x[1])))]
    return f"Is it true that every {_concept_phrase(gci.lhs)} {' and '.join(parts)}?"


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

# The statement forms an EQ answer may take, with placeholder names so that the
# examples do not suggest an axiom of the ontology.
# Not A: the model read "A SubClassOf B" as the article and wrote
# "A Professor SubClassOf Teacher" (jobs 4380986-87).
EQ_EXAMPLES = {
    "manchester": ("C SubClassOf D", "C SubClassOf r some D", "C and (r some D) SubClassOf E"),
    "nl": ("C is a subcategory of D", "C is a subcategory of something that r some D",
           "C that is also something that r some D is a subcategory of E"),
}
_EQ_FORMAT = {"manchester": "in Manchester syntax", "nl": "in this form"}

# How the counterexample is asked for (EQSettings.generation; experiments cex13-cex17,
# the generation variants).  In cex12 most answers could not be used: negation, names out
# of scope, the placeholder copied, explanations.
#   default  the EQ_EXAMPLES placeholders
#   g1       the same, with the forms allowed and the slips seen listed as rules
#   g2       concrete examples about cars instead of placeholders, with role names no
#            target ontology has (S2 already uses cars for the check)
#   g3       a structured answer, "Subject | relation | Object" (el_syntax.parse_triple):
#            EL by construction, but only A ⊑ B and A ⊑ ∃r.B
#   g2b      G2 without brackets: in cex14 every answer was wrapped in them ("(Cat, eats,
#            Meat)"), perhaps copied from "(about cars, not about O)" or the third example
#   g3b      G3 with the format explained in prose: in cex15 the model copied the shape
#            "Subject | is-a | Object: every Subject is an Object", writing "Subject"
#            itself, or the format line before its answer
# G1, G2 and G2b are written in Manchester syntax; G3 and G3b ask the same in either language.
GENERATIONS = ("default", "g1", "g2", "g3", "g2b", "g3b")
MANCHESTER_ONLY_GENERATIONS = ("g1", "g2", "g2b")
TRIPLE_GENERATIONS = ("g3", "g3b")         # the answer is a triple (el_syntax.parse_triple)
EQ_EXAMPLES_G2 = ("Car SubClassOf Vehicle", "Car SubClassOf has_component some Wheel",
                  "Vehicle and (powered_by some Engine) SubClassOf MotorVehicle")
EQ_EXAMPLES_G2B = ("Car SubClassOf Vehicle", "Car SubClassOf has_component some Wheel",
                   "Vehicle and powered_by some Engine SubClassOf MotorVehicle")
_G1_RULES = (
    "Rules:\n"
    "  - C, D, E and r stand for names in scope: replace them, do not write them.\n"
    "  - Use only the concept names and object properties in scope, spelled as given.\n"
    "  - An object property is always followed by some and one concept: r some D, never r D.\n"
    "  - Combine with and only. not, or, only, commas and comments are not allowed.\n"
    "  - The statement can only add to H: it cannot say that a statement of H is false.\n"
    "  - Write the statement alone on one line, with no explanation before or after it.\n"
)
_G3_FORMAT = (
    "Output exactly one line of the form Subject | relation | Object, where Subject and "
    "Object are concept names in scope and relation is is-a or an object property in scope:\n"
    "  Subject | is-a | Object: every Subject is an Object;\n"
    "  Subject | r | Object: every Subject is related by r to at least one Object.\n"
    "For example (about cars, not about O):\n"
    "  Car | is-a | Vehicle\n"
    "  Car | has_component | Wheel\n"
)
_G3B_FORMAT = (
    "Write it as one line of three parts separated by |, in this order: a concept name in "
    "scope, a relation, and a concept name in scope. The relation is is-a when every member of the "
    "first concept is a member of the second, or an object property in scope when every "
    "member of the first concept is related by it to at least one member of the second.\n"
    "For example, about cars:\n"
    "  Car | is-a | Vehicle\n"
    "  Car | has_component | Wheel\n"
)


def _eq_context(sig: set[str], roles: set[str], H: set[GCI], language: str,
                feedback: Sequence[str] = ()) -> str:
    """The scope and H, then *feedback*: why earlier answers to the prompt
    could not be used, one line each (a deterministic model needs a changed
    prompt to answer differently)."""
    sig_str, roles_str = _scope(sig, roles)
    h_str = hypothesis_to_nl(H) if language == "nl" else hypothesis_to_manchester(H)
    retry = "".join(f"{line}\n" for line in feedback)
    return (
        f"Concept names in scope: {sig_str}.\n"
        f"Object properties in scope: {roles_str}.\n\n"
        "Hypothesis H:\n"
        f"{h_str}\n\n"
        + (f"{retry}\n" if retry else "")
    )


def eq_judge_prompt(sig: set[str], roles: set[str], H: set[GCI],
                    language: str = "manchester", feedback: Sequence[str] = ()) -> str:
    return _eq_context(sig, roles, H, language, feedback) + (
        "Does H capture everything that is true in O?\n"
        "Reply with exactly one word: yes or no."
    )


def eq_counterexample_prompt(sig: set[str], roles: set[str], H: set[GCI],
                             feedback: Sequence[str] = (), language: str = "manchester",
                             generation: str = "default") -> str:
    if generation in MANCHESTER_ONLY_GENERATIONS and language != "manchester":
        raise ValueError(f"generation {generation} is written in Manchester syntax only")
    head = _eq_context(sig, roles, H, language, feedback) + (
        "Name one counterexample: a statement that is true in O but does not follow from H.\n")
    if generation == "g3":
        return head + _G3_FORMAT + "Output nothing else."
    if generation == "g3b":
        return head + _G3B_FORMAT + "Output only that line, with no explanation."
    if generation == "g2":
        examples = "".join(f"  {e}\n" for e in EQ_EXAMPLES_G2)
        return head + (f"Output exactly one line {_EQ_FORMAT[language]}, for example "
                       f"(about cars, not about O):\n{examples}Output nothing else.")
    if generation == "g2b":
        examples = "".join(f"  {e}\n" for e in EQ_EXAMPLES_G2B)
        return head + (f"Output exactly one line {_EQ_FORMAT[language]}, for example, "
                       f"about cars:\n{examples}Output nothing else.")
    examples = "".join(f"  {e}\n" for e in EQ_EXAMPLES[language])
    return head + (
        f"Output exactly one line {_EQ_FORMAT[language]}, for example:\n"
        f"{examples}"
        + (_G1_RULES if generation == "g1" else "")
        + "Output nothing else."
    )


def eq_combined_prompt(sig: set[str], roles: set[str], H: set[GCI],
                       feedback: Sequence[str] = (), language: str = "manchester") -> str:
    examples = "".join(f"  {e}\n" for e in EQ_EXAMPLES[language])
    return _eq_context(sig, roles, H, language, feedback) + (
        "If H captures everything that is true in O, reply with exactly one word: yes.\n"
        "Otherwise name one counterexample: a statement that is true in O but does not "
        "follow from H.\n"
        f"Output exactly one line: yes, or the counterexample {_EQ_FORMAT[language]}, "
        "for example:\n"
        f"{examples}"
        "Output nothing else."
    )


def repeated_feedback(failed: int) -> str:
    """The line that opens an EQ's prompts after *failed* EQs in a row gave no usable
    answer.  The learner asks again with H unchanged; without it, a greedy model would
    get the same prompt, and the same answer, again."""
    queries = "equivalence query" if failed == 1 else f"{failed} equivalence queries"
    return f"The previous {queries} in a row gave no usable answer; this is a new one."


def unusable_feedback(reason: str) -> str:
    """The retry line for an answer of this EQ that could not be used."""
    return f"Your previous answer could not be used: {reason}."


def rejected_earlier_feedback(statement: str) -> str:
    """The line for a counterexample the check rejected at an earlier EQ (``memory``)."""
    return (f"Rejected at an earlier equivalence query, do not propose it again: "
            f"'{statement}' (asked whether it is true in O, you answered no).")
