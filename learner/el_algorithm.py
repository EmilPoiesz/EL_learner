from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Optional, Tuple
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data structures for EL concept expressions
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ELConcept:
    """
    An EL concept expression.  We represent it in a *normalised* structural form:

      concept = (atoms, existentials)

    where
      atoms         – frozenset of atomic concept names (strings); the implicit
                      conjunction of named classes.  ∅ represents ⊤.
      existentials  – frozenset of (role_name, ELConcept) pairs; the ∃r.C parts.

    The full concept is the conjunction of all atoms and all existentials.

    Examples
    --------
    ⊤                    → ELConcept(frozenset(), frozenset())
    A                    → ELConcept(frozenset({"A"}), frozenset())
    A ⊓ ∃r.B             → ELConcept(frozenset({"A"}), frozenset({("r", ELConcept({"B"}, {}))}))
    A ⊓ B ⊓ ∃r.C         → ELConcept(frozenset({"A", "B"}), frozenset({("r", ELConcept({"C"}, {}))}))
    ∃r.(A ⊓ ∃s.B)        → ELConcept(frozenset(), frozenset({("r", ELConcept({"A"}, {("s", ELConcept({"B"}, {}))}))}))

    """
    atoms: frozenset[str] = field(default_factory=frozenset)
    existentials: frozenset[Tuple[str, "ELConcept"]] = field(default_factory=frozenset)

    def __post_init__(self):
        # Enforce immutable field types even when caller passes regular sets/lists.
        object.__setattr__(self, "atoms", frozenset(self.atoms))
        object.__setattr__(self, "existentials", frozenset(self.existentials))

    # ------------------------------------------------------------------
    # Pretty-printing
    # ------------------------------------------------------------------
    def __str__(self) -> str:
        parts = sorted(self.atoms)
        for role, filler in sorted(self.existentials, key=lambda x: (x[0], str(x[1]))):
            parts.append(f"∃{role}.({filler})")
        if not parts:
            return "⊤"
        return " ⊓ ".join(parts)

    def __repr__(self) -> str:
        return f"ELConcept({str(self)})"


@dataclass(frozen=True)
class GCI:
    """
    A General Concept Inclusion  lhs ⊑ rhs.
    Both lhs and rhs are ELConcepts.
    """
    lhs: ELConcept
    rhs: ELConcept

    def __str__(self) -> str:
        return f"{self.lhs} ⊑ {self.rhs}"

    def __repr__(self) -> str:
        return f"GCI({str(self)})"


# ---------------------------------------------------------------------------
# Signature helpers
# ---------------------------------------------------------------------------

def signature(terminology: set[GCI]) -> set[str]:
    """
    Return Σ_H – the set of all atomic concept names appearing in a terminology.
    """
    names: set[str] = set()
    for gci in terminology:
        names |= _concept_atoms(gci.lhs)
        names |= _concept_atoms(gci.rhs)
    return names


def _concept_atoms(concept: ELConcept) -> set[str]:
    """Recursively collect all atomic concept names in an EL concept."""
    atoms = set(concept.atoms)
    for _, filler in concept.existentials:
        atoms |= _concept_atoms(filler)
    return atoms


# ---------------------------------------------------------------------------
# Oracle abstract base class
# ---------------------------------------------------------------------------

class Oracle(ABC):
    """
    Abstract base class defining the interface that any oracle must implement.
    In the learning framework, the oracle represents the "teacher" that knows
    the target terminology O.  The learner interacts with it through four
    abstract members:

      1. signature        – Σ_O, the set of atomic concept names in O.
      2. MQ(gci)          – membership query: does O entail this GCI?
      3. EQ(H)            – equivalence query: given the current hypothesis
                            H, return a GCI entailed by O but not by H,
                            or None if H ≡ O.
      4. make_H_MQ(H)     – return a callable that answers H-entailment queries.
      5. on_H_add(gci)    – called after each GCI is added to H; use this to
                            keep any external H-reasoner in sync with H.

    Call counts are tracked automatically.  After learning, inspect
    ``oracle.mq_count`` and ``oracle.eq_count``, or call ``reset_counts()``
    to zero them before a new run.
    """

    def __init__(self) -> None:
        self.mq_count: int = 0
        self.eq_count: int = 0

    def reset_counts(self) -> None:
        """Reset MQ and EQ call counters to zero."""
        self.mq_count = 0
        self.eq_count = 0

    @property
    @abstractmethod
    def signature(self) -> set[str]:
        """
        Σ_O – the set of atomic concept names in the target ontology O.
        The learner is given this upfront as background knowledge.
        """

    def MQ(self, gci: GCI) -> bool:
        """
        Membership query: return True iff O entails the given GCI.

        Increments ``mq_count`` and delegates to ``_MQ``.
        """
        self.mq_count += 1
        return self._MQ(gci)

    def EQ(self, hypothesis: set[GCI]) -> Optional[GCI]:
        """
        Equivalence query: given the current hypothesis H, return either:
          - A *positive* counterexample: a GCI entailed by O but not by H, or
          - None, meaning H ≡ O and learning is complete.

        Increments ``eq_count`` and delegates to ``_EQ``.
        """
        self.eq_count += 1
        return self._EQ(hypothesis)

    @abstractmethod
    def _MQ(self, gci: GCI) -> bool:
        """Oracle-specific membership query implementation."""

    @abstractmethod
    def _EQ(self, hypothesis: set[GCI]) -> Optional[GCI]:
        """Oracle-specific equivalence query implementation."""

    @abstractmethod
    def make_H_MQ(self, H: set[GCI]) -> Callable[[GCI], bool]:
        """
        Return a callable  H_MQ(gci) -> bool  that answers whether the current
        hypothesis H entails gci.

        A complete H-entailment check requires a DL reasoner (e.g. ELK via
        HypothesisReasoner); the EL closure is not computable by structural
        inspection alone.

        H is passed by reference; keep the returned callable in sync with H
        by implementing on_H_add to push each new GCI to the underlying reasoner.
        """

    @abstractmethod
    def on_H_add(self, gci: GCI) -> None:
        """
        Called by learn_el_terminology immediately after a GCI is added to H.

        Keep any external H-reasoner in sync so that make_H_MQ stays correct
        as H grows.  If no external state needs updating, implement as a no-op.
        """

# ---------------------------------------------------------------------------
# O-essential GCI computation helper functions
# ---------------------------------------------------------------------------

def _make_filler_rebuild(
    role: str,
    old_filler: ELConcept,
    current_atoms: frozenset,
    current_existentials: frozenset,
    outer_rebuild: Callable[[ELConcept], ELConcept],
) -> Callable[[ELConcept], ELConcept]:
    """
    Create a function to rebuild the full RHS concept with an updated filler for a specific existential.
    """
    def rebuild(new_filler: ELConcept) -> ELConcept:
        # Rebuild the full RHS concept with the updated filler for this existential.
        new_exs = frozenset(
            {(r, s) for r, s in current_existentials if not (r == role and s == old_filler)} | {(role, new_filler)}
        )
        return outer_rebuild(ELConcept(current_atoms, new_exs))
    return rebuild


def saturate_concept_rhs(
    concept: ELConcept,
    signature: set[str],
    MQ: Callable[[GCI], bool],
    lhs: ELConcept,
    _rebuild: Callable[[ELConcept], ELConcept] = lambda x: x,
) -> ELConcept:
    """
    Saturate the RHS concept tree, per the paper's concept saturation rule:

        For each node in the tree and each A′ ∈ signature,
        if O ⊨ A ⊑ C′  (where C′ is the concept with A′ added to that node),
        add A′ to that node.

    The check is always anchored to ``lhs`` (the atomic A of the full GCI).
    ``_rebuild`` is an internal continuation used during recursion to reconstruct
    the full RHS from a modified subtree, so the MQ is always  lhs ⊑ full_rhs.
    External callers only pass the first four arguments.
    """
    atoms = set(concept.atoms)
    existentials = set(concept.existentials)

    changed = True
    while changed:
        changed = False

        new_atoms = set(atoms)
        for atom_prime in signature:
            if atom_prime in new_atoms:
                continue
            # New RHS concept with atom_prime added at this node; the rest of the tree is unchanged.
            candidate = ELConcept(frozenset(new_atoms | {atom_prime}), frozenset(existentials))
            # Check if O entails lhs ⊑ candidate (where candidate is the full RHS with atom_prime added at this node).
            if MQ(GCI(lhs, _rebuild(candidate))):
                new_atoms.add(atom_prime)
                changed = True
        atoms = new_atoms

        new_existentials = set()
        for role, filler in existentials:
            # Create a rebuild function for this specific existential so that recursive saturation of the filler can reconstruct the full RHS correctly.
            filler_rebuild = _make_filler_rebuild(
                role, filler, frozenset(atoms), frozenset(existentials), _rebuild
            )
            # Recursively saturate the filler concept; the rebuild function ensures that the MQ check is always against the full RHS with the updated filler.
            saturated_filler = saturate_concept_rhs(filler, signature, MQ, lhs, filler_rebuild)
            new_existentials.add((role, saturated_filler))

            if saturated_filler != filler:
                changed = True
        existentials = new_existentials

    return ELConcept(frozenset(atoms), frozenset(existentials))

def _try_merge_in_concept(lhs: ELConcept, MQ: Callable, node: ELConcept, rebuild: Callable[[ELConcept], ELConcept]) -> tuple[ELConcept, bool]:
    """
    Attempt ONE valid sibling merge anywhere in the `node` subtree.

    ``rebuild`` is a continuation: given a replacement for ``node``, it
    reconstructs the *full* RHS concept so the T-entailment check is always
    performed against  lhs ⊑ full_rhs,  mirroring the original ExactLearner's
    ``merging()`` which validates every candidate with
    ``myEngineForT.entailed(cl ⊑ tmp.transformToClassExpression())``.

    Returns (new_full_rhs, True) when a valid merge is found, or
    (rebuild(node), False) when none exists.
    """
    # --- try merging at this node ---
    role_map: dict = {}
    for role, sub in node.existentials:
        role_map.setdefault(role, []).append(sub)

    for role, subs in role_map.items():
        if len(subs) >= 2:
            merged_sub = ELConcept(
                atoms=frozenset().union(*(s.atoms for s in subs)),
                existentials=frozenset().union(*(s.existentials for s in subs)),
            )
            new_node_exs = frozenset(
                {(r, s) for r, s in node.existentials if r != role}
                | {(role, merged_sub)}
            )
            candidate_node = ELConcept(node.atoms, new_node_exs)
            candidate_root = rebuild(candidate_node)
            if MQ(GCI(lhs=lhs, rhs=candidate_root)):
                return candidate_root, True

    # --- recurse into children ---
    for target_role, target_sub in node.existentials:
        def child_rebuild(new_sub, *, r=target_role, s=target_sub):
            new_exs = frozenset(
                {(r2, s2) for r2, s2 in node.existentials if not (r2 == r and s2 == s)}
                | {(r, new_sub)}
            )
            return rebuild(ELConcept(node.atoms, new_exs))

        result, found = _try_merge_in_concept(lhs, MQ, target_sub, child_rebuild)
        if found:
            return result, found

    return rebuild(node), False

def sibling_merge(concept: ELConcept, lhs: ELConcept, MQ: Callable) -> ELConcept:
    """
    Repeatedly merge same-role sibling existentials until no more merges apply.

    Each merge step is validated against O: ``MQ(GCI(lhs, full_rhs_after_merge))``
    must return True before the merge is accepted.  This mirrors the original
    ExactLearner's ``mergeRight()`` / ``merging()`` which guards every candidate
    with ``myEngineForT.entailed(cl ⊑ merged)``.
    """
    current = concept
    while True:
        new, changed = _try_merge_in_concept(lhs, MQ, current, lambda x: x)
        if not changed:
            break
        current = new
    return current

def is_subsumed_by(c1: ELConcept, c2: ELConcept, H: set[GCI]) -> bool:
    """Checks if c1 ⊑ c2 given the current H."""
    # 1. Trivial case: Top concept
    if not c2.atoms and not c2.existentials:
        return True

    # 2. Check Atoms: All atoms in c2 must be present in c1
    # (Note: This is a basic check; a full version would check H for atom-to-atom matches)
    if not c2.atoms.issubset(c1.atoms):
        # Even if not a subset, H might say Atom1 ⊑ Atom2
        # For now, let's ensure at least basic subset logic works
        return False

    # 3. Check Existentials: For every ∃r.D in c2, there must be an ∃r.E in c1 such that E ⊑ D
    for r_c2, D in c2.existentials:
        found_match = False
        for r_c1, E in c1.existentials:
            if r_c1 == r_c2 and is_subsumed_by(E, D, H):
                found_match = True
                break
        if not found_match:
            return False

    return True

def entails_in_H(gci_query: GCI, H: set[GCI]) -> bool:
    """Returns True if H |= gci_query.lhs ⊑ gci_query.rhs"""
    # Check if any GCI in our hypothesis makes this true
    for learned_gci in H:
        # If (query.lhs ⊑ learned.lhs) and (learned.rhs ⊑ query.rhs)
        # then H entails (query.lhs ⊑ query.rhs)
        if is_subsumed_by(gci_query.lhs, learned_gci.lhs, H):
            if is_subsumed_by(learned_gci.rhs, gci_query.rhs, H):
                return True
    return False

def remove_subtree(root, target_node, role, sub_to_remove):
    if root is target_node:
        new_existentials = {
            (r, sub) for (r, sub) in root.existentials
            if not (r == role and sub == sub_to_remove)
        }
        return ELConcept(root.atoms, new_existentials)

    new_existentials = set()
    for r, sub in root.existentials:
        new_sub = remove_subtree(sub, target_node, role, sub_to_remove)
        new_existentials.add((r, new_sub))

    return ELConcept(root.atoms, new_existentials)

def decompose_rhs(lhs, concept, original_concept, hypothesis, MQ, H_MQ=None):
    """
    Returns a new GCI after applying ONE decomposition step,
    or None if no decomposition applies.
    """
    if H_MQ is None:
        def H_MQ(gci):
            return entails_in_H(gci, hypothesis)

    # Traverse nodes (DFS)
    stack = [(None, None, concept)]
    # (parent, role_from_parent, current_node)

    while stack:
        parent, role, node = stack.pop()


        is_root = (parent is None)
        # Restrict atoms at root to original RHS atoms only
        if is_root:
            valid_atoms = node.atoms & original_concept.atoms
        else:
            valid_atoms = node.atoms

        for A_prime in valid_atoms:
        # Check decomposition condition at this node
        #for A_prime in node.atoms:
            for r, sub in node.existentials:

                candidate_rhs = ELConcept(set(), {(r, sub)})

                A_prime_concept = ELConcept(frozenset({A_prime}))
                if MQ(GCI(A_prime_concept, candidate_rhs)):
                    # --- CASE (a): learn new inclusion ---
                    if not H_MQ(GCI(A_prime_concept, candidate_rhs)):
                        return GCI(A_prime_concept, candidate_rhs)

                    # --- CASE (b): remove subtree (only if result is novel) ---
                    else:
                        new_concept = remove_subtree(concept, node, r, sub)
                        candidate = GCI(lhs, new_concept)
                        if not H_MQ(candidate):
                            return candidate

        # Continue traversal
        for r, sub in node.existentials:
            stack.append((node, r, sub))

    return None

def saturate_left(lhs: ELConcept, hypothesis: set[GCI], signature: set[str], H_MQ=None) -> ELConcept:
    """
    If H |= concept ⊑ A', then add A' to the atoms of the concept.
    This must be done recursively for all nested concepts in existentials.
    """
    if H_MQ is None:
        def H_MQ(gci):
            return entails_in_H(gci, hypothesis)

    # 1. Saturate the nested concepts first (bottom-up)
    new_exs = []
    for role, nested_concept in lhs.existentials:
        new_exs.append((role, saturate_left(nested_concept, hypothesis, signature, H_MQ)))

    # Create a temporary concept with saturated children to check the current node
    intermediate_C = ELConcept(lhs.atoms, frozenset(new_exs))

    # 2. Add any concept name from the vocabulary that is implied by H
    new_atoms = set(intermediate_C.atoms)
    for A_prime in signature:
        if H_MQ(GCI(intermediate_C, ELConcept(frozenset({A_prime})))):
            new_atoms.add(A_prime)

    return ELConcept(atoms=frozenset(new_atoms), existentials=frozenset(new_exs))

def _unsaturate_node(
    node: ELConcept,
    rhs: ELConcept,
    MQ: Callable[[GCI], bool],
    rebuild: Callable[[ELConcept], ELConcept],
) -> ELConcept:
    """
    Recursively try to remove atoms from `node` and all descendant nodes.

    `rebuild(new_node)` reconstructs the full LHS concept given a replacement
    for `node`, so O-entailment is always checked against the complete GCI.
    Changes are committed immediately; each child's rebuild closure captures the
    most up-to-date parent state so subsequent siblings see prior removals.
    """
    # --- Try dropping atoms at this level ---
    current_node = node
    for atom in list(node.atoms):
        candidate = ELConcept(
            atoms=frozenset(current_node.atoms - {atom}),
            existentials=current_node.existentials,
        )
        if MQ(GCI(rebuild(candidate), rhs)):
            current_node = candidate

    # --- Recurse into existential fillers ---
    new_exs: list[tuple[str, ELConcept]] = []
    for role, sub in list(current_node.existentials):
        # Capture current_node by value via default arg so each child's rebuild
        # reflects all prior sibling modifications.
        def child_rebuild(
            new_sub: ELConcept,
            *,
            r: str = role,
            s: ELConcept = sub,
            frozen_node: ELConcept = current_node,
        ) -> ELConcept:
            updated_exs = frozenset(
                {(r2, s2) for r2, s2 in frozen_node.existentials if not (r2 == r and s2 == s)}
                | {(r, new_sub)}
            )
            return rebuild(ELConcept(frozen_node.atoms, updated_exs))

        new_sub = _unsaturate_node(sub, rhs, MQ, child_rebuild)
        new_exs.append((role, new_sub))

        # Update current_node so the next sibling's rebuild closure is current.
        current_node = ELConcept(
            current_node.atoms,
            frozenset(
                {(r2, s2) for r2, s2 in current_node.existentials if not (r2 == role and s2 == sub)}
                | {(role, new_sub)}
            ),
        )

    return ELConcept(current_node.atoms, frozenset(new_exs))


def unsaturate_left(lhs: ELConcept, rhs: ELConcept, MQ: Callable[[GCI], bool]) -> ELConcept:
    """
    Concept desaturation on the left side of the inclusion lhs ⊑ rhs.

    Mirrors Java ExactLearner.unsaturateLeft(): iterates over every node in the
    LHS concept tree and, for each atomic concept name in the node's label,
    attempts to remove it.  The removal is kept when O still entails the GCI
    (checked via MQ); otherwise the atom is restored.

    This is applied after decompose_left to minimise the left-hand side.
    """
    return _unsaturate_node(lhs, rhs, MQ, rebuild=lambda x: x)


def decompose_left(lhs: ELConcept, rhs: ELConcept, hypothesis: set[GCI], MQ: Callable[[GCI], bool], H_MQ=None, signature: set[str] = None) -> GCI:
    if H_MQ is None:
        def H_MQ(gci):
            return entails_in_H(gci, hypothesis)

    def _find_undiscovered_atom(concept: ELConcept) -> Optional[ELConcept]:
        """Return A′ ∈ ΣO s.t. O ⊨ concept ⊑ A′ and H ⊭ concept ⊑ A′, or None."""
        if signature is not None:
            for A in signature:
                if A in concept.atoms:
                    continue  # trivially entailed; not informative
                A_concept = ELConcept(frozenset({A}), frozenset())
                if MQ(GCI(concept, A_concept)) and not H_MQ(GCI(concept, A_concept)):
                    return A_concept
            return None
        # Fallback: check only against rhs (original behaviour when no signature given)
        if MQ(GCI(concept, rhs)) and not H_MQ(GCI(concept, rhs)):
            return rhs
        return None

    # Case 1: Try subtrees (Cd) — non-root nodes only (existential fillers).
    # Per the paper: if O ⊨ Cd ⊑ A′ and H ⊭ Cd ⊑ A′ for some A′ ∈ ΣO,
    # replace C ⊑ A by Cd ⊑ A′.  A′ need not equal the original rhs.
    for role, sub_concept in lhs.existentials:
        a_prime = _find_undiscovered_atom(sub_concept)
        if a_prime is not None:
            return decompose_left(sub_concept, a_prime, hypothesis, MQ, H_MQ, signature)

    # Case 2: Try pruning (C \ d)
    # Try removing one existential at a time from the top level
    for ex in lhs.existentials:
        reduced_exs = frozenset([e for e in lhs.existentials if e != ex])
        lhs_minus_sub_concept = ELConcept(lhs.atoms, reduced_exs)

        if MQ(GCI(lhs_minus_sub_concept, rhs)) and not H_MQ(GCI(lhs_minus_sub_concept, rhs)):
            return decompose_left(lhs_minus_sub_concept, rhs, hypothesis, MQ, H_MQ, signature)

    # If no further decomposition is possible, C is O-essential
    return GCI(lhs, rhs)


# ---------------------------------------------------------------------------
# O-essential GCI computation  (the core of the algorithm)
# ---------------------------------------------------------------------------

def compute_right_essential(lhs: ELConcept, rhs: ELConcept, hypothesis: set[GCI], MQ: Callable[[GCI], bool], signature: set[str], H_MQ=None) -> GCI:
    """
    Compute a *right O-essential* α from the GCI  lhs ⊑ rhs
    (used in line 6 of Algorithm 1, when lhs ∈ Σ_O ∩ N_C).

    Intuition
    ---------
    We have a counterexample  C' ⊑ D'  where C' is an atomic concept already
    in the learner's signature.  The current hypothesis H already has some GCIs
    with C' on the left-hand side; their right-hand sides form a conjunction F.

    A *right O-essential* α is the "most informative" strengthening of D' that:
      - is still entailed by O, and
      - cannot be derived from H alone.

    Concretely, we compute:
      rhs_refined = D' ⊓  ⋂{F' | C' ⊑ F' ∈ H}

    and return  C' ⊑ rhs_refined.

    This corresponds to line 6:
      Compute a right O-essential α from C' ⊑ D' ⊓ ⊓_{C'⊑F'∈H} F'
    """

    # Collect all right-hand sides of H-axioms whose lhs == C'
    rhs_parts_from_H = [
        gci.rhs for gci in hypothesis
        if gci.lhs == lhs
    ]

    # Conjoin them all with D' by merging atoms and existentials
    combined_atoms = set(rhs.atoms)
    combined_existentials = set(rhs.existentials)
    for part in rhs_parts_from_H:
        combined_atoms |= part.atoms
        combined_existentials |= part.existentials

    rhs_refined = ELConcept(
        atoms=frozenset(combined_atoms),
        existentials=frozenset(combined_existentials),
    )

    # 1) Concept Saturation
    rhs_saturated = saturate_concept_rhs(rhs_refined, signature, MQ, lhs=lhs)

    # 2) Sibling Merge — each merge step is validated against T via MQ so that
    #    merging two same-role existentials (e.g. ∃r.D1 and ∃r.D2) is only
    #    accepted when O actually entails lhs ⊑ ∃r.(D1 ⊓ D2).  This mirrors
    #    the original ExactLearner's merging() guard:
    #        T.entailed(cl ⊑ merged_expression)
    rhs_merged = sibling_merge(rhs_saturated, lhs=lhs, MQ=MQ)

    logger.info(f"refined {rhs_refined}")
    logger.info(f"saturated {rhs_saturated}")
    logger.info(f"merged {rhs_merged}")

    # 3) Decomposition
    candidate = decompose_rhs(lhs, rhs_merged, rhs, hypothesis, MQ, H_MQ)

    if candidate is not None:
        # Validate the decomposed candidate is actually entailed by O.
        # decompose_rhs can produce a subtree-stripped GCI that over-extends
        # the rhs by carrying saturated atoms that O does not actually entail
        # from the given lhs.
        if MQ(candidate):
            logger.info(f"decomposed {candidate}")
            return candidate
        else:
            logger.debug(
                "Decomposed candidate %s not confirmed by O; "
                "skipping decomposition and continuing to merged form.",
                candidate,
            )

    return GCI(lhs=lhs, rhs=rhs_merged)


def compute_left_essential(lhs: ELConcept, rhs: ELConcept, hypothesis: set[GCI], MQ: Callable[[GCI], bool], signature: set[str], H_MQ=None) -> GCI:
    """
    Compute a *left O-essential* α from the GCI  lhs ⊑ rhs
    (used in line 8 of Algorithm 1, when rhs ∈ Σ_O ∩ N_C but lhs is not atomic).

    Intuition
    ---------
    We want to *weaken* the left-hand side as much as possible while the GCI
    remains entailed by O and cannot be derived from H alone.
    """

    lhs_saturated = saturate_left(lhs, hypothesis, signature, H_MQ)
    decomposed = decompose_left(lhs_saturated, rhs, hypothesis, MQ, H_MQ, signature)
    lhs_essential = unsaturate_left(decomposed.lhs, decomposed.rhs, MQ)

    return GCI(lhs=lhs_essential, rhs=decomposed.rhs)


# ---------------------------------------------------------------------------
# Normalise a counterexample into C' ⊑ D' with C' or D' atomic
# ---------------------------------------------------------------------------

def normalise_counterexample(counterexample: GCI, signature: set[str], MQ: Callable[[GCI], bool]) -> GCI:
    """
    Given a positive counterexample  C ⊑ D  (entailed by O, not by H),
    produce a *normalised* GCI  C' ⊑ D'  such that either:
      - C' ∈ Σ_O ∩ N_C  (C' is an atomic concept in the oracle's signature), or
      - D' ∈ Σ_O ∩ N_C  (D' is an atomic concept in the oracle's signature).

    This corresponds to line 4 of Algorithm 1.

    Strategy
    --------
    In a well-formed counterexample from the oracle, either the LHS or the RHS
    (or both) will already be atomic concepts in Σ_O.  We check that and return
    the GCI unchanged if the condition is already met.

    If neither side is atomic (e.g. both are conjunctions), we attempt to find
    a sub-GCI by:
      1. Trying each atomic part of rhs as a stand-alone rhs.
      2. Trying each atomic part of lhs as a stand-alone lhs.
    """
    # Already normalised?
    lhs_atomic = (
        len(counterexample.lhs.atoms) == 1
        and not counterexample.lhs.existentials
        and next(iter(counterexample.lhs.atoms)) in signature
    )
    rhs_atomic = (
        len(counterexample.rhs.atoms) == 1
        and not counterexample.rhs.existentials
        and next(iter(counterexample.rhs.atoms)) in signature
    )

    if lhs_atomic or rhs_atomic:
        return counterexample

    # Try to find an atomic rhs that is still entailed
    for atom in counterexample.rhs.atoms:
        if atom in signature:
            candidate = GCI(
                lhs=counterexample.lhs,
                rhs=ELConcept(atoms=frozenset({atom})),
            )
            if MQ(candidate):
                return candidate

    # Try to find an atomic lhs
    for atom in counterexample.lhs.atoms:
        if atom in signature:
            candidate = GCI(
                lhs=ELConcept(atoms=frozenset({atom})),
                rhs=counterexample.rhs,
            )
            if MQ(candidate):
                return candidate

    # Fall back: return original (may already be normalisable by the caller)
    logger.warning(
        "Could not normalise counterexample %s to have an atomic side; "
        "returning as-is.  The algorithm may not terminate.",
        counterexample,
    )
    return counterexample


# ---------------------------------------------------------------------------
# Algorithm 1 – main learning procedure
# ---------------------------------------------------------------------------

def learn_el_terminology(oracle: Oracle, max_iterations: int = 1000) -> set[GCI]:
    """
    Algorithm 1: Learn an EL terminology from an equivalence oracle.

    Parameters
    ----------
    oracle : Oracle
        The teacher.  Must implement:
          - oracle.signature     → set[str]              (Σ_O, the atomic concept names)
          - oracle.MQ(gci)       → bool                  (membership query against O)
          - oracle.EQ(H)         → GCI | None            (equivalence query)
          - oracle.make_H_MQ(H)  → Callable[[GCI], bool] (membership query against H)
          - oracle.on_H_add(gci) → None                  (called after each GCI is added to H)

    max_iterations : int
        Safety cap to prevent infinite loops during development/debugging.

    Returns
    -------
    H : set[GCI]
        A terminology equivalent to O (O ≡ H).

    Algorithm walkthrough
    ---------------------
    Line 1  Initialise H with all *atomic* subsumptions entailed by O.
            These are GCIs  A ⊑ B  where both A and B are in Σ_O ∩ N_C.
            They are cheap to enumerate: |Σ_O|² checks.

    Lines 2–11  Main loop:
        Line 3   Ask the oracle for a positive counterexample GCI  C ⊑ D.
        Line 4   Normalise it so one side is atomic (∈ Σ_O ∩ N_C).
        Lines 5–9  Decide whether to compute a *right* or *left* O-essential:
            - If C' is atomic (lhs is atomic): compute a RIGHT-essential.
              This means we strengthen the rhs by conjoining the rhs-parts
              already in H (they were not enough, so we must add more).
            - Else (D' is atomic, rhs is atomic): compute a LEFT-essential.
              This means we find the *weakest* lhs that still forces D'.
        Line 10  Add the O-essential α to H.

    Line 12  Return H.
    """

    # ------------------------------------------------------------------
    # Line 1: Initialise H with all atomic subsumptions A ⊑ B entailed by O
    # ------------------------------------------------------------------
    logger.info("=== Phase 1: Enumerating atomic subsumptions ===")
    H: set[GCI] = set()
    H_MQ = oracle.make_H_MQ(H)
    sigma = list(oracle.signature)

    for a_name in sigma:
        for b_name in sigma:
            if a_name == b_name:
                # A ⊑ A is trivially true; no need to add to H
                continue
            gci = GCI(
                lhs=ELConcept(atoms=frozenset({a_name})),
                rhs=ELConcept(atoms=frozenset({b_name})),
            )
            if oracle.MQ(gci):
                logger.debug("  Adding atomic subsumption: %s", gci)
                H.add(gci)

    logger.info("  |H| after atomic init = %d", len(H))

    # ------------------------------------------------------------------
    # Lines 2–11: Main refinement loop
    # ------------------------------------------------------------------
    logger.info("=== Phase 2: Iterative refinement ===")
    for iteration in range(max_iterations):

        # Line 3: Ask the oracle for a positive counterexample
        counterexample = oracle.EQ(H)

        if counterexample is None:
            # oracle says H ≡ O  →  we are done
            logger.info("Oracle confirmed H ≡ O after %d iteration(s).", iteration)
            break

        logger.info("Iteration %d | Counterexample: %s", iteration + 1, counterexample)

        # Line 4: Normalise so that C' or D' is in Σ_O ∩ N_C
        normalised = normalise_counterexample(
            counterexample=counterexample,
            signature=oracle.signature,
            MQ=oracle.MQ
        )
        logger.info("  Normalised:  %s", normalised)

        C_prime = normalised.lhs
        D_prime = normalised.rhs

        # Determine whether C' is atomic (∈ Σ_O ∩ N_C)
        C_prime_is_atomic = (
            len(C_prime.atoms) == 1
            and not C_prime.existentials
            and next(iter(C_prime.atoms)) in oracle.signature
        )

        if C_prime_is_atomic:

            # Lines 5–6: C' ∈ Σ_O ∩ N_C  →  compute a RIGHT O-essential
            #
            # We conjoin D' with all rhs-parts of current H-axioms whose lhs = C'.
            # Rationale: H already says "C' ⊑ F'" for some F's, but that is not
            # enough to derive the counterexample.  We need to add a *stronger*
            # right-hand side.

            logger.info("  C' is atomic → computing RIGHT O-essential")
            alpha = compute_right_essential(
                lhs=C_prime,
                rhs=D_prime,
                hypothesis=H,
                MQ=oracle.MQ,
                signature=oracle.signature,
                H_MQ=H_MQ,
            )
        else:
            # Lines 7–8: D' ∈ Σ_O ∩ N_C  →  compute a LEFT O-essential
            #
            # We minimise the lhs of the counterexample: find the *weakest*
            # sub-concept of C' that still forces D' under O.
            logger.info("  D' is atomic → computing LEFT O-essential")
            alpha = compute_left_essential(
                lhs=C_prime,
                rhs=D_prime,
                hypothesis=H,
                MQ=oracle.MQ,
                signature=oracle.signature,
                H_MQ=H_MQ,
            )

        # Line 10: Add the essential GCI to H
        logger.info("  Adding to H: %s", alpha)
        H.add(alpha)
        oracle.on_H_add(alpha)
        logger.info("  |H| = %d", len(H))

    else:
        logger.warning(
            "Reached max_iterations=%d without convergence. "
            "H may not yet be equivalent to O.",
            max_iterations,
        )

    # Line 12: Return H
    return H