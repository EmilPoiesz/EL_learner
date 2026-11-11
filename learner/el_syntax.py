"""
EL concepts and GCIs as text: Manchester syntax and this project's natural
language, written and parsed.  Pure functions, no model.
"""

from __future__ import annotations

import re

from learner.el_algorithm import ELConcept, GCI, existential_key


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
# Natural language for EQs
# ---------------------------------------------------------------------------
# Upstream's phrases ("that is also", "something that r some", "subcategory"), but
# with a filler of several conjuncts in brackets, so that the text maps back to
# Manchester syntax word for word (upstream_natural_language leaves brackets
# half-rewritten, and the paper has no natural-language EQ to stay faithful to).

def concept_to_nl(concept: ELConcept, nested: bool = False) -> str:
    parts = sorted(concept.atoms) + [
        f"something that {role} some {concept_to_nl(filler, nested=True)}"
        for role, filler in sorted(concept.existentials, key=existential_key)
    ]
    if not parts:
        return "owl:Thing"
    text = " that is also ".join(parts)
    return f"({text})" if nested and len(parts) > 1 else text


def gci_to_nl(gci: GCI) -> str:
    return f"{concept_to_nl(gci.lhs)} is a subcategory of {concept_to_nl(gci.rhs)}"


def hypothesis_to_nl(H: set[GCI]) -> str:
    if not H:
        return "  (empty)"
    return "\n".join(f"  {gci_to_nl(gci)}" for gci in sorted(H, key=str))


def parse_nl_gci(s: str) -> GCI:
    """The GCI of a statement written as gci_to_nl writes it.  Raises ValueError."""
    sides = re.split(r"\s+is an? subcategory of\s+", s.strip(), maxsplit=1, flags=re.IGNORECASE)
    if len(sides) != 2:
        raise ValueError(f"no 'is a subcategory of' in {s!r}")

    def to_manchester(text: str) -> str:
        text = re.sub(r"\bthat is also\b", "and", text.strip().strip("'\""), flags=re.IGNORECASE)
        return re.sub(r"\bsomething that\s+", "", text, flags=re.IGNORECASE)

    return parse_manchester_gci(f"{to_manchester(sides[0])} SubClassOf {to_manchester(sides[1])}")


# ---------------------------------------------------------------------------
# Triples: "Subject | relation | Object" (EQ answers with generation G3)
# ---------------------------------------------------------------------------

IS_A = "is-a"
# The relation of a triple that means SubClassOf, as the model may write it.
IS_A_PATTERN = r"is[\s_\-]?an?|isa|subclassof"


def gci_to_triple(gci: GCI) -> str | None:
    """A ⊑ B as "A | is-a | B", A ⊑ ∃r.B as "A | r | B", for concept names A and B;
    None for any other GCI."""
    lhs, rhs = gci.lhs, gci.rhs
    if len(lhs.atoms) != 1 or lhs.existentials:
        return None
    subject = next(iter(lhs.atoms))
    if len(rhs.atoms) == 1 and not rhs.existentials:
        return f"{subject} | {IS_A} | {next(iter(rhs.atoms))}"
    if not rhs.atoms and len(rhs.existentials) == 1:
        role, filler = next(iter(rhs.existentials))
        if len(filler.atoms) == 1 and not filler.existentials:
            return f"{subject} | {role} | {next(iter(filler.atoms))}"
    return None


def parse_triple(s: str) -> GCI:
    """The GCI of "Subject | relation | Object": Subject ⊑ Object when the relation
    is is-a (also "is a", "isa", "is_a", "SubClassOf", in any case), else
    Subject ⊑ ∃relation.Object.  Raises ValueError."""
    parts = [p.strip().strip("`'\"").strip() for p in s.strip().strip("|").split("|")]
    if len(parts) != 3:
        raise ValueError(f"expected 'Subject | relation | Object': {s!r}")
    subject, relation, obj = parts
    is_a = re.fullmatch(IS_A_PATTERN, relation, flags=re.IGNORECASE)
    for name in (subject, obj) if is_a else (subject, relation, obj):
        if not re.fullmatch(NAME_PATTERN, name):
            raise ValueError(f"{name!r} is not a name: {s!r}")
    lhs = ELConcept(frozenset({subject}))
    if is_a:
        return GCI(lhs, ELConcept(frozenset({obj})))
    return GCI(lhs, ELConcept(frozenset(), frozenset({(relation, ELConcept(frozenset({obj})))})))


def without_brackets(text: str) -> str:
    """*text* without brackets that enclose all of it: "((A SubClassOf B))" → "A SubClassOf B"."""
    text = text.strip()
    while text.startswith("(") and _matching_paren_or_none(text, 0) == len(text) - 1:
        text = text[1:-1].strip()
    return text


def _matching_paren_or_none(s: str, open_pos: int) -> int | None:
    depth = 0
    for i in range(open_pos, len(s)):
        depth += {"(": 1, ")": -1}.get(s[i], 0)
        if depth == 0:
            return i
    return None


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
NAME_PATTERN = r'\w(?:[\w:.\-]*\w)?'


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

        some_match = re.match(rf'^({NAME_PATTERN})\s+some\s+(.+)$', part, re.DOTALL | re.IGNORECASE)
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
        elif re.match(rf'^{NAME_PATTERN}$', part):
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

