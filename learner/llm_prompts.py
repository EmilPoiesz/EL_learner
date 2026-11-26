"""
This project's own prompts (the paper's are in learner/upstream.py): the system
messages, the yes/no MQ, the EQ prompts and their retry lines, and the system
prompts of the counterexample check.  Plain functions of the scope and H; the
prompt variants of the EQ experiments (cex1-cex20) are made here.
"""

from __future__ import annotations

import ast
import re
from typing import Sequence

from learner.el_algorithm import GCI, existential_key
from learner.el_syntax import gci_to_manchester, gci_to_nl, hypothesis_to_manchester, hypothesis_to_nl


def _scope(sig: set[str] | Sequence[str], roles: set[str]) -> tuple[str, str]:
    """The concept names and roles as listed in a prompt: sorted, except a *sig* given
    as a list or tuple, which keeps its order."""
    names = list(sig) if isinstance(sig, (list, tuple)) else sorted(sig)
    return ", ".join(names), ", ".join(sorted(roles)) if roles else "(none)"


# ---------------------------------------------------------------------------
# System messages (the MQ's are the paper's, upstream.PAPER_SYSTEM_PROMPTS)
# ---------------------------------------------------------------------------

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

# The check's system prompt (EQSettings.check_system): the paper's simple one, S1, since
# cex28 (with the domain note since cex44).  The variants tried before it were removed on
# 2026-11-26: the run's own MQ prompt (under paper-advanced its rule 2 made Mistral reject
# every has_part statement, cex6), S2-S4 (that prompt with its rules restated, examples
# about cars, cex8-cex27) and "literal" (read literally, the MQ-wording bench only).
CHECK_SYSTEM_PROMPTS: dict[str, str] = {"s1": "Answer with only True or False."}

# What the check's system prompt says about the domain (EQSettings.check_domain): "off",
# nothing; "names" (2026-11-17), a note listing the run's own concept names and roles,
# saying they come from one domain.  In the MQ benchmark Qwen3-14B read a name in its
# widest everyday sense ("not every Teacher is employed by a University: some teach at
# schools") and, under "literal", asked for facts it had not been given ("no data is
# provided, so I can't verify it").  The domain is named only through the signature, so
# the note is the same kind of text for every ontology, never written for one.
CHECK_DOMAINS = ("off", "names")


def domain_note(sig: set[str], roles: set[str]) -> str:
    """The "names" note: the signature (sorted, without owl:Thing) and the advice to read
    the names in the sense of their shared domain, from the model's own knowledge."""
    concepts, relations = _scope(sorted(n for n in sig if n not in ("owl:Thing", "Thing")), roles)
    return ("The statements use these category names: " + concepts + "; and these relations: "
            + relations + ". They all come from one domain. Read each name in the sense it has "
            "in that domain, together with the other names, and answer from your own knowledge "
            "of that domain.")


def check_system(base: str, domain: str, sig: set[str], roles: set[str]) -> str:
    """The check's system prompt: *base* (a ``CHECK_SYSTEM_PROMPTS`` text), followed by
    the domain note when *domain* asks for one."""
    if domain == "off":
        return base
    return f"{base}\n\n{domain_note(sig, roles)}"

# The question the check asks (EQSettings.check_question): the paper's MQ in natural
# language ("Can Teacher be considered a subcategory of 'Professor'?").  "every" ("Is it
# true that every Teacher is a Professor?") and "atonce" (the same with conjunctions
# spelled out) were only benchmarked (no wording won on both ontologies) and were removed
# on 2026-11-26.
CHECK_QUESTIONS = ("paper",)


# ---------------------------------------------------------------------------
# EQ
# ---------------------------------------------------------------------------

# The statement forms a combined EQ's answer may take (eq_combined_prompt), with
# placeholder names so that the examples do not suggest an axiom of the ontology.
# Not A: the model read "A SubClassOf B" as the article and wrote
# "A Professor SubClassOf Teacher" (jobs 4380986-87).
EQ_EXAMPLES = {
    "manchester": ("C SubClassOf D", "C SubClassOf r some D", "C and (r some D) SubClassOf E"),
    "nl": ("C is a subcategory of D", "C is a subcategory of something that r some D",
           "C that is also something that r some D is a subcategory of E"),
}
_EQ_FORMAT = {"manchester": "in Manchester syntax", "nl": "in this form"}

# How the counterexample is asked for (EQSettings.generation): g2b, examples about cars
# with role names no target ontology has.  The variants tried before it (cex13-cex17:
# default placeholders, g1 with rules, g2 with brackets, g3/g3b as triples
# "Subject | relation | Object") were removed on 2026-11-26; every run since cex18 used
# g2b.  In natural language the examples are gci_to_nl of the Manchester ones (cex23).
GENERATIONS = ("g2b",)
EQ_EXAMPLES_G2B = {
    "manchester": ("Car SubClassOf Vehicle", "Car SubClassOf has_component some Wheel",
                   "Vehicle and powered_by some Engine SubClassOf MotorVehicle"),
    "nl": ("Car is a subcategory of Vehicle",
           "Car is a subcategory of something that has_component some Wheel",
           "Vehicle that is also something that powered_by some Engine is a subcategory of MotorVehicle"),
}


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
                             allow_none: bool = False) -> str:
    """*allow_none* (EQSettings.none): the one-word answer "none" is allowed when there
    is no counterexample."""
    examples = "".join(f"  {e}\n" for e in EQ_EXAMPLES_G2B[language])
    return _eq_context(sig, roles, H, language, feedback) + (
        "Name one counterexample: a statement that is true in O but does not follow from H.\n"
        + (NONE_LINE if allow_none else "")
        + f"Output exactly one line {_EQ_FORMAT[language]}, for example, about cars:\n"
        f"{examples}Output nothing else.")


# EQSettings.none: the way out when the model knows nothing true that H misses.
NONE_LINE = ("If every statement you know to be true in O already follows from H, reply with "
             "exactly one word instead: none.\n")


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


def unusable_feedback(reason: str, times: int = 1) -> str:
    """The retry line for an answer of this EQ that could not be used; *times*: how
    often it failed this way in this EQ (said when more than once).  Each reason gets
    one line, and the count keeps the prompt changing when the model repeats itself:
    an identical prompt would get the identical (cached) answer for every attempt left."""
    again = f" (you gave this answer {times} times)" if times > 1 else ""
    return f"Your previous answer could not be used{again}: {reason}."


# Why an answer failed, explained (EQSettings.explain, cex28).  In cex25-cex27 Qwen
# repeated the same failures within a run: negation ("Researcher and not Professor
# SubClassOf Person", 17x on university), a concept without SubClassOf, OWL keywords EL
# lacks, roles out of scope (the inverse "employs"), and statements H already implies,
# often trivially ("Professor and Researcher SubClassOf Person", "Bird and has_part some
# Wing SubClassOf Bird").  The bare reason says what failed, not what to do instead; the
# explanations name no concept or role of the signature.
_NOT_EL = ("not", "or", "only", "all", "exactly", "min", "max", "value", "that", "intersectionOf",
           "unionOf", "complementOf", "DisjointWith", "EquivalentTo", "Type")
_FORM = {"manchester": "'C SubClassOf D'", "nl": "'C is a subcategory of D'"}


# EQSettings.explain_form (cex55): three failures Qwen3-32B repeated in cex54 that the
# explanations above missed.  "¬" got only the generic form line (the negation check
# reads letters); inverse roles ("inverse(r)", "r^", "r inv") got it 7-9 times in a row,
# each time in a new notation; "Nothing" was called an out-of-scope name, so the model
# was never told that EL cannot say two classes share no members.
_INVERSE = re.compile(r"\binverse\b|\binverseOf\b|\binv\b|\^|⁻", flags=re.I)
_NOTHING = {"Nothing", "owl:Nothing", "⊥"}
# Whole names only (code review 2026-11-26, finding 17): a substring test fired on
# names such as BottomFeeder.
_DISJOINT_WORD = re.compile(r"(?<![\w:])disjoint(?:with|union(?:of)?)?(?![\w:])", flags=re.I)


def _names(text: str) -> set[str]:
    """The names and the symbol ⊥ in *text*, each whole."""
    return set(re.findall(r"⊥|[\w:]+", text))
_DISJOINT = ("EL statements cannot say that two classes share no members: there is no "
             "'Nothing' and no disjointness. Name something that is true of every member instead.")


def _explain_form(answer: str, reason: str, language: str) -> str:
    """The cex55 explanations (EQSettings.explain_form), or "" when none applies."""
    if "not of the form" in reason:
        if "¬" in answer:
            return ("EL statements cannot say what is not the case: there is no 'not' and no '¬'. "
                    "Name something that is true of every member instead.")
        if _DISJOINT_WORD.search(answer) or _names(answer) & _NOTHING:
            return _DISJOINT
        if _INVERSE.search(answer):
            forward = ("'C SubClassOf r some D'" if language == "manchester"
                       else "'C is a subcategory of something that r some D'")
            return ("There are no inverse roles: a role can only be followed forward, from the left "
                    f"side. {forward} says that every C has an r to some D; if what you mean is true "
                    "of every member of the class on the other side, state it about that class.")
    m = re.search(r": (.*?) (?:is not a concept name|are not concept names) in scope", reason)
    if m and any(n.strip() in _NOTHING for n in m.group(1).split(",")):
        return _DISJOINT
    return ""


def explain_unusable(reason: str, language: str = "manchester", form: bool = False) -> str:
    """One sentence on what to do instead, for the *reason* an answer could not be used
    (the parser's or the scope check's ValueError).  The answer is the quoted line.
    *form* (EQSettings.explain_form): also explain '¬', inverse roles and 'Nothing'."""
    # The answer as repr() quoted it: in double quotes when it holds an apostrophe
    # (code review 2026-11-26, finding 16: such answers got no explanation).
    quoted = re.match(r"""^('(?:[^'\\]|\\.)*'|"(?:[^"\\]|\\.)*")(?: is not of the form|: )""", reason, flags=re.S)
    answer = ast.literal_eval(quoted.group(1)) if quoted else ""
    words = set(re.findall(r"[A-Za-z]+", answer))
    if form and (extra := _explain_form(answer, reason, language)):
        return extra
    if "not of the form" in reason:
        if words & {"not", "Not", "NOT", "no", "No", "without"}:
            return ("EL statements cannot say what is not the case: there is no 'not'. "
                    "Name something that is true of every member instead.")
        used = [w for w in _NOT_EL if w in words and not (w == "that" and language == "nl")]
        if used:
            return (f"'{used[0]}' cannot be used: a statement may only use "
                    + ("'SubClassOf', 'and' and 'some'." if language == "manchester"
                       else "'is a subcategory of', 'that is also' and 'something that r some'."))
        if not re.search(r"subclassof|subcategory", answer, flags=re.I):
            return (f"That names a concept, not a statement: a statement has the form "
                    f"{_FORM[language]}, saying that every C is a D.")
        return f"A statement has the form {_FORM[language]}, written as in the examples."
    # Signature-independent: no guess at which listed role was meant (a name-prefix match
    # suggested employed_by for employs, but would suggest hasSex for hasSister).
    if re.search(r"(?:is not an object property|are not object properties) in scope", reason):
        return ("Use only the object properties listed above. For a relation in the other "
                "direction than a listed one, use the listed one with the two sides swapped.")
    if "not a concept name" in reason or "not concept names" in reason:
        return "Use only the concept names listed above; a statement about other names cannot be used."
    return ""


def explain_follows(trivial: bool) -> str:
    """One sentence on a counterexample that already follows from H: *trivial* when it
    holds of anything (its right-hand side is part of its left-hand side)."""
    if trivial:
        return ("It is true of anything, since its right-hand side is already part of its "
                "left-hand side, so it says nothing new. Name a statement whose right-hand side "
                "adds something.")
    return ("H already implies it, so it is not a counterexample: name a statement that is "
            "true in O but that H does not imply.")


def explain_rejected() -> str:
    """One sentence on a counterexample the model's own check rejected."""
    return "By your own answer it is not true in O, so it is not a counterexample: name a different statement."


def rejected_earlier_feedback(statement: str) -> str:
    """The line for a counterexample the check rejected at an earlier EQ (``memory``)."""
    return (f"Rejected at an earlier equivalence query, do not propose it again: "
            f"'{statement}' (asked whether it is true in O, you answered no).")
