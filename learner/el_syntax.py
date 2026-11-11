"""
EL concepts and GCIs as text in Manchester syntax, written and parsed.  Pure
functions, no model.
"""

from __future__ import annotations

import re

from learner.el_algorithm import ELConcept, GCI


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

