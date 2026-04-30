"""
test_el_algorithm.py
====================
Pytest test suite for the EL learning algorithm (el_algorithm.py).

Each test targets a specific function or branch of Algorithm 1, using the
minimal ontology in ontologies/test_minimal.ttl (and hand-coded oracles where
isolation matters).

Run with:
    pytest                          # ELK reasoner, compact output
    pytest -v                       # verbose output
    pytest --reasoner=hermit        # HermiT reasoner
    pytest --reasoner=hermit -v

Tests 1, 2, 14, 17, 23, 35–37 require a live reasoner and are skipped
automatically when the chosen reasoner cannot start.  All others always run.

Test catalogue
--------------
 1  saturate_concept_rhs   — atoms propagated transitively via MQ (lhs = A)
 2  saturate_concept_rhs   — filler of ∃r.C gets D added because O ⊨ B ⊑ ∃r.(C⊓D)
 3  sibling_merge          — two same-role branches merged; one existential with C⊓D filler
 4  sibling_merge          — pairwise merge within three siblings; all-at-once rejected
 5  sibling_merge          — single ∃r.C branch left unchanged
 6  sibling_merge          — different-role siblings (∃r.C, ∃s.C) are never merged
 7  decompose_rhs case (a) — inner node fires, emits B ⊑ ∃s.F
 8  decompose_rhs          — root atom B ≢_O A included; decomposition fires
 9  decompose_rhs          — root atom B ≡_O A excluded; returns None
10  decompose_rhs case (b) — ∃s.F already in H; subtree stripped from filler
11  decompose_rhs          — no atom+existential node → returns None
12  saturate_left          — H: A⊑B adds B to the LHS of concept A
13  saturate_left          — transitivity: H has A⊑B and B⊑C; C added via H_MQ
14  decompose_left case 1  — prune ∃r.C; A⊓B still forces E → returns A⊓B ⊑ E
15  decompose_left         — no existentials to prune or zoom; returns None
16  decompose_left         — pruned=⊤ fails; C implies no new A' → returns None
17  decompose_left case 2  — subtree zoom: G ⊑ E fires
18  unsaturate_left        — both atoms essential; LHS unchanged
19  unsaturate_left        — one redundant atom removed from LHS
20  unsaturate_left        — nested atom in existential filler removed
21  unsaturate_left        — real MQ: B dropped from A⊓B (A⊑B in O collapses LHS)
22  compute_right_essential — ∃r.C: saturation adds B to root, decomposition fires → GCI 3
23  compute_right_essential — sibling merge fires; two ∃r. → ∃r.(C⊓D)
24  compute_right_essential — decompose_rhs case (a) inside right-essential
25  compute_left_essential  — A⊓B ⊑ E: B dropped because A⊑B in O; result is A ⊑ E
26  compute_left_essential  — existential-only LHS kept intact (GCI 5)
27  compute_left_essential  — H-saturation must precede decompose_left (paper page 3)
28  normalise_counterexample — already normalised (atomic LHS)
29  normalise_counterexample — already normalised (atomic RHS)
30  normalise_counterexample — compound RHS; atomic CE found and confirmed as positive CE
31  normalise_counterexample — case (b) H_MQ filter: atoms already in H are skipped
32  normalise_counterexample — case 1: recursive reduction on matching existential filler
33  normalise_counterexample — case 2: atomic LHS via Lemma 3 (no matching filler in C)
34  normalise_counterexample — case 1 depth-2 recursion: ∃r.∃r.A ⊑ ∃r.∃r.B → A ⊑ B
35  learn_el_terminology    — completeness: every GCI in O is entailed by H
36  learn_el_terminology    — soundness: every GCI in H is entailed by O
37  learn_el_terminology    — equivalence: EQ returns None after learning
38  learn_el_terminology    — Phase 1 calls on_H_add for each atomic subsumption
"""

from __future__ import annotations

import pytest

from learner.el_algorithm import (
    ELConcept,
    GCI,
    compute_left_essential,
    compute_right_essential,
    decompose_left,
    decompose_rhs,
    learn_el_terminology,
    normalise_counterexample,
    saturate_concept_rhs,
    saturate_left,
    sibling_merge,
    unsaturate_left,
)

# ---------------------------------------------------------------------------
# Shared concept constants   Σ = {A,B,C,D,E,F,G},  roles r, s
# ---------------------------------------------------------------------------

A   = ELConcept(atoms=frozenset({"A"}))
B   = ELConcept(atoms=frozenset({"B"}))
C   = ELConcept(atoms=frozenset({"C"}))
D   = ELConcept(atoms=frozenset({"D"}))
E   = ELConcept(atoms=frozenset({"E"}))
F   = ELConcept(atoms=frozenset({"F"}))
G   = ELConcept(atoms=frozenset({"G"}))
TOP = ELConcept()

rA   = ELConcept(existentials=frozenset({("r", A)}))
rC   = ELConcept(existentials=frozenset({("r", C)}))
rD   = ELConcept(existentials=frozenset({("r", D)}))
sF   = ELConcept(existentials=frozenset({("s", F)}))

AB    = ELConcept(atoms=frozenset({"A", "B"}))
CD    = ELConcept(atoms=frozenset({"C", "D"}))
BsF   = ELConcept(atoms=frozenset({"B"}), existentials=frozenset({("s", F)}))
rBsF  = ELConcept(existentials=frozenset({("r", BsF)}))
rC_rD = ELConcept(existentials=frozenset({("r", C), ("r", D)}))
rCD   = ELConcept(existentials=frozenset({("r", CD)}))

SIG = {"A", "B", "C", "D", "E", "F", "G"}

# Target ontology O — mirrors ontologies/test_minimal.ttl exactly.
TARGET_O: set[GCI] = {
    GCI(A, B),        # GCI 1
    GCI(A, rC),       # GCI 2
    GCI(B, rCD),      # GCI 3  (single ∃r.(C⊓D))
    GCI(B, rC),       # GCI 3a (entailed by GCI 3)
    GCI(B, rD),       # GCI 3b (entailed by GCI 3)
    GCI(AB, E),       # GCI 4
    GCI(rC, F),       # GCI 5
    GCI(G, E),        # GCI 6a
    GCI(G, sF),       # GCI 6b
    GCI(A, rBsF),     # GCI 7
    GCI(B, sF),       # GCI 8
}

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def mq(reasoner_oracle):
    """
    Return an MQ callable backed by the live reasoner when available, or a
    structural fallback that checks direct membership in TARGET_O otherwise.
    The fallback is sufficient for tests that don't need the full closure.
    """
    if reasoner_oracle is not None:
        return reasoner_oracle.MQ

    def _structural(gci: GCI) -> bool:
        if gci in TARGET_O:
            return True
        for mid in TARGET_O:
            if (
                GCI(gci.lhs, mid.lhs) in TARGET_O
                and GCI(mid.rhs, gci.rhs) in TARGET_O
            ):
                return True
        return False

    return _structural


@pytest.fixture
def h_mq():
    """H_MQ for tests where H = ∅ — nothing is entailed by the hypothesis."""
    return lambda gci: False


@pytest.fixture
def req(reasoner_oracle, reasoner_name):
    """Skip the calling test if no reasoner is available."""
    if reasoner_oracle is None:
        pytest.skip(f"Reasoner {reasoner_name!r} unavailable")
    return reasoner_oracle


@pytest.fixture(scope="session")
def learned_h(integration_oracle):
    """
    Run learn_el_terminology once against the integration oracle and cache the
    result.  Tests 35–37 share this fixture so the algorithm executes exactly
    once per session.
    """
    if integration_oracle is None:
        pytest.skip("Reasoner unavailable — integration test skipped")
    return learn_el_terminology(integration_oracle, max_iterations=100)


# ===========================================================================
# 1–2  saturate_concept_rhs
# ===========================================================================


def test_01_saturate_atoms(req, mq):
    result = saturate_concept_rhs(A, SIG, mq, lhs=A)
    assert result.atoms == frozenset({"A", "B", "E", "F"})


def test_02_saturate_filler(req, mq):
    # lhs = B; O ⊨ B ⊑ ∃r.(C⊓D) (GCI 3), so filler C must gain atom D.
    result = saturate_concept_rhs(rC, SIG, mq, lhs=B)
    inner = next(f for role, f in result.existentials if role == "r")
    assert inner.atoms == frozenset({"C", "D"})


# ===========================================================================
# 3–6  sibling_merge
# ===========================================================================


def test_03_sibling_merge_two_branches(mq):
    # Two same-role branches ∃r.C and ∃r.D must be merged into a single
    # ∃r.(C⊓D) because O ⊨ B ⊑ ∃r.(C⊓D) (GCI 3).
    merged = sibling_merge(rC_rD, lhs=B, MQ=mq)
    assert len(merged.existentials) == 1
    _, filler = next(iter(merged.existentials))
    assert filler.atoms == frozenset({"C", "D"})


def test_04_sibling_merge_pairwise_within_three():
    # Three same-role siblings: ∃r.A, ∃r.B, ∃r.C.
    # O entails merging A with B (leaving C separate) but NOT the 3-way merge A⊓B⊓C.
    three_siblings = ELConcept(existentials=frozenset({("r", A), ("r", B), ("r", C)}))
    after_merge    = ELConcept(existentials=frozenset({("r", AB), ("r", C)}))

    def mock_mq(gci):
        return gci.rhs == after_merge

    result = sibling_merge(three_siblings, lhs=A, MQ=mock_mq)
    assert len(result.existentials) == 2
    fillers = {f for _, f in result.existentials}
    assert AB in fillers
    assert C in fillers


def test_05_sibling_merge_single_branch(mq):
    assert sibling_merge(rC, lhs=A, MQ=mq) == rC


def test_06_sibling_merge_different_roles_not_merged():
    # ∃r.C and ∃s.C share the same filler but have different roles; they must
    # never be merged, regardless of what MQ says.
    rC_sC = ELConcept(existentials=frozenset({("r", C), ("s", C)}))
    result = sibling_merge(rC_sC, lhs=A, MQ=lambda _: True)
    assert len(result.existentials) == 2
    assert {role for role, _ in result.existentials} == {"r", "s"}


# ===========================================================================
# 7–11  decompose_rhs
# ===========================================================================


def test_07_decompose_rhs_case_a(mq):
    # Case (a): inner node has {B, ∃s.F}; O |= B ⊑ ∃s.F; H empty → emit B ⊑ ∃s.F.
    result = decompose_rhs(A, rBsF, mq, lambda _: False)
    assert result == GCI(B, sF)


def test_08_decompose_rhs_root_atom_not_o_equiv(mq):
    # Root has atom B (e.g. added by prior saturation) and existential ∃r.C.
    # B ≢_O A (O ⊭ B ⊑ A), so the paper allows A'=B at the root.
    B_rC = ELConcept(atoms=frozenset({"B"}), existentials=frozenset({("r", C)}))

    def mock_mq(gci: GCI) -> bool:
        # O |= B ⊑ ∃r.C  (triggers case (a) with A'=B at root).
        # All other queries (including B ⊑ A and A ⊑ B) return False,
        # confirming B ≢_O A.
        return gci == GCI(B, rC)

    result = decompose_rhs(A, B_rC, mock_mq, lambda _: False)
    assert result == GCI(B, rC)


def test_09_decompose_rhs_root_atom_o_equiv_excluded():
    # Root has atom B where B ≡_O A (O |= B ⊑ A and O |= A ⊑ B).
    # The paper excludes such A' at the root — decomposing with an O-equivalent
    # atom would not make progress toward a smaller counterexample.
    B_rC = ELConcept(atoms=frozenset({"B"}), existentials=frozenset({("r", C)}))

    def mock_mq(gci: GCI) -> bool:
        if gci == GCI(B, A) or gci == GCI(A, B):
            return True
        if gci == GCI(B, rC):
            return True
        return False

    result = decompose_rhs(A, B_rC, mock_mq, lambda _: False)
    assert result is None


def test_10_decompose_rhs_case_b(mq):
    # Case (b): H already knows B ⊑ ∃s.F → strip ∃s.F subtree, return A ⊑ ∃r.B.
    H: set[GCI] = {GCI(B, sF)}
    result = decompose_rhs(A, rBsF, mq, lambda gci: gci in H)
    expected = GCI(A, ELConcept(existentials=frozenset({("r", B)})))
    assert result == expected


def test_11_decompose_rhs_no_mixed_node(mq):
    # ∃r.C: root has only an existential, filler C has only an atom → no mixed node.
    result = decompose_rhs(A, rC, mq, lambda _: False)
    assert result is None


# ===========================================================================
# 12–13  saturate_left
# ===========================================================================


def test_12_saturate_left_adds_implied_atom():
    # H contains A ⊑ B; saturating {A} via H should add B.
    H = {GCI(A, B)}
    result = saturate_left(A, H, SIG, lambda gci: gci in H)
    assert result.atoms == frozenset({"A", "B"})


def test_13_saturate_left_transitivity():
    # H: A ⊑ B, B ⊑ C.  A complete H_MQ (as a real reasoner provides) exposes
    # the transitive closure, so saturate_left must add both B and C to {A}.
    H = {GCI(A, B), GCI(B, C)}
    h_mq_transitive = lambda gci: gci in {GCI(A, B), GCI(B, C), GCI(A, C)}
    result = saturate_left(A, H, SIG, h_mq_transitive)
    assert frozenset({"A", "B", "C"}) <= result.atoms


# ===========================================================================
# 14–17  decompose_left
# ===========================================================================


def test_14_decompose_left_case1_prune(req, mq):
    # LHS = A ⊓ B ⊓ ∃r.C, RHS = E.
    # Case 1 (prune): remove ∃r.C → pruned = A⊓B. O |= A⊓B ⊑ E (GCI 4) → fires.
    lhs = ELConcept(atoms=frozenset({"A", "B"}), existentials=frozenset({("r", C)}))
    result = decompose_left(lhs, E, mq, lambda _: False, SIG)
    assert result == GCI(AB, E)


def test_15_decompose_left_no_existentials(mq):
    # LHS = A⊓B has no existentials — no non-root node to prune or zoom into.
    result = decompose_left(AB, E, mq, lambda _: False, SIG)
    assert result is None


def test_16_decompose_left_prune_fails_no_zoom(mq):
    # LHS = ∃r.C. Case 1: pruned = ⊤; O ⊭ ⊤ ⊑ F. Case 2: O ⊭ C ⊑ A' for
    # any A' ∈ SIG (GCI 5 says ∃r.C ⊑ F, not C ⊑ F). Both cases fail → None.
    result = decompose_left(rC, F, mq, lambda _: False, SIG)
    assert result is None


def test_17_decompose_left_case2_subtree_zoom(req, mq):
    # LHS = ∃r.G, RHS = F. Case 1: pruned = ⊤; O ⊭ ⊤ ⊑ F.
    # Case 2: O |= G ⊑ E (GCI 6a) and H ⊭ G ⊑ E → fires, returning G ⊑ E.
    rG = ELConcept(existentials=frozenset({("r", G)}))
    result = decompose_left(rG, F, mq, lambda _: False, SIG)
    assert result == GCI(G, E)


# ===========================================================================
# 18–21  unsaturate_left
# ===========================================================================


def test_18_unsaturate_left_essential_atoms_unchanged():
    # Mock where only A⊓B ⊑ E holds; neither A alone nor B alone forces E.
    def mock_mq(gci: GCI) -> bool:
        return gci.lhs == AB and gci.rhs == E

    result = unsaturate_left(AB, E, mock_mq)
    assert result.atoms == frozenset({"A", "B"})


def test_19_unsaturate_left_redundant_atom_removed():
    # Mock where A alone suffices (A ⊑ E holds), so B is redundant in A⊓B ⊑ E.
    def mock_mq(gci: GCI) -> bool:
        if gci.rhs != E:
            return False
        return gci.lhs in (AB, A)

    result = unsaturate_left(AB, E, mock_mq)
    assert result.atoms == frozenset({"A"})
    assert not result.existentials


def test_20_unsaturate_left_nested_filler_atom_removed():
    # LHS = ∃r.(A⊓B), RHS = F.
    # Mock: ∃r.A ⊑ F holds, so B is redundant inside the filler.
    rA_only = ELConcept(existentials=frozenset({("r", A)}))
    rAB     = ELConcept(existentials=frozenset({("r", AB)}))

    def mock_mq(gci: GCI) -> bool:
        if gci.rhs != F:
            return False
        return gci.lhs in (rAB, rA_only)

    result = unsaturate_left(rAB, F, mock_mq)
    assert len(result.existentials) == 1
    _, filler = next(iter(result.existentials))
    assert filler.atoms == frozenset({"A"})


def test_21_unsaturate_left_real_mq_b_dropped(mq):
    # O contains A ⊑ B (GCI 1), which makes A alone sufficient to satisfy
    # A⊓B ⊑ E (GCI 4).  unsaturate_left should therefore drop B from the LHS.
    result = unsaturate_left(AB, E, mq)
    assert result.atoms == frozenset({"A"})
    assert not result.existentials


# ===========================================================================
# 22–24  compute_right_essential
# ===========================================================================


def test_22_right_essential_simple(mq):
    # GCI 2: A ⊑ ∃r.C as counterexample with lhs = A.
    # Saturation adds B, E, F to the root and D to the filler → A⊓B⊓E⊓F⊓∃r.(C⊓D).
    # Decomposition fires: B ≢_O A and O ⊨ B ⊑ ∃r.(C⊓D) (GCI 3) → lhs becomes B.
    # Exhaustive loop: re-saturate ∃r.(C⊓D) with lhs=B → adds F (via GCI 3+5).
    result = compute_right_essential(A, rC, set(), mq, SIG, lambda _: False)
    expected_rhs = ELConcept(
        atoms=frozenset({"F"}),
        existentials=frozenset({("r", ELConcept(atoms=frozenset({"C", "D"})))}),
    )
    assert result == GCI(B, expected_rhs)


def test_23_right_essential_sibling_merge(req, mq):
    # GCI 3: H has B ⊑ ∃r.C; new CE is B ⊑ ∃r.D → merge → ∃r.(C⊓D).
    H: set[GCI] = {GCI(B, rC)}
    result = compute_right_essential(B, rD, H, mq, SIG, lambda gci: gci in H)
    r, filler = next(iter(result.rhs.existentials))
    assert len(result.rhs.existentials) == 1
    assert r == "r"
    assert filler.atoms == frozenset({"C", "D"})


def test_24_right_essential_decompose_case_a(mq):
    # GCI 7: A ⊑ ∃r.(B⊓∃s.F).
    # After saturation, decomposition fires at the inner node: B ≢_O A and
    # O ⊨ B ⊑ ∃s.F (GCI 8) → lhs becomes B, rhs becomes ∃s.F.
    # Exhaustive loop: re-saturate ∃s.F with lhs=B → adds F (via GCI 3+5).
    result = compute_right_essential(A, rBsF, set(), mq, SIG, lambda _: False)
    expected_rhs = ELConcept(
        atoms=frozenset({"F"}),
        existentials=frozenset({("s", F)}),
    )
    assert result == GCI(B, expected_rhs)


# ===========================================================================
# 25–27  compute_left_essential
# ===========================================================================


def test_25_left_essential_conjunctive_lhs(mq):
    # GCI 4: A ⊓ B ⊑ E.  O contains A ⊑ B (GCI 1), so A alone already forces E.
    # The desaturation step removes B, leaving just A as the minimal precondition.
    result = compute_left_essential(AB, E, set(), mq, SIG, lambda _: False)
    assert result.lhs.atoms == frozenset({"A"})
    assert result.rhs == E


def test_26_left_essential_existential_lhs(mq):
    # GCI 5: ∃r.C ⊑ F; existential-only LHS returned unchanged.
    result = compute_left_essential(rC, F, set(), mq, SIG, lambda _: False)
    assert result == GCI(rC, F)


def test_27_left_essential_h_saturation_enables_decompose():
    # CE: ∃r.C ⊑ E.  H = {C ⊑ B}.  O |= C⊓B ⊑ E, but O ⊭ C ⊑ E alone.
    # Without H-saturation decompose_left sees inner node C and case 2 finds
    # nothing.  After H-saturation adds B to C, the inner node becomes C⊓B
    # and case 2 fires: O |= C⊓B ⊑ E and H ⊭ C⊓B ⊑ E.
    # (Paper page 3: "can only be decomposed … if we apply concept saturation
    # for H first".)
    CB = ELConcept(atoms=frozenset({"C", "B"}))
    rC_concept = ELConcept(existentials=frozenset({("r", C)}))
    H = {GCI(C, B)}

    def mock_mq(gci):
        return gci.lhs == CB and gci.rhs == E

    result = compute_left_essential(rC_concept, E, H, mock_mq, SIG, lambda gci: gci in H)
    # decompose_left zooms into C⊓B → GCI(C⊓B, E).
    # unsaturate_left cannot drop either atom (neither alone satisfies mock_mq).
    assert result == GCI(CB, E)


# ===========================================================================
# 28–34  normalise_counterexample
# ===========================================================================


def test_28_normalise_atomic_lhs(mq, h_mq):
    ce = GCI(A, rC)
    assert normalise_counterexample(ce, SIG, mq, h_mq) == ce


def test_29_normalise_atomic_rhs(mq, h_mq):
    ce = GCI(AB, E)
    assert normalise_counterexample(ce, SIG, mq, h_mq) == ce


def test_30_normalise_compound_rhs_finds_atomic_ce(mq, h_mq):
    # LHS = A⊓B (non-atomic), RHS = E⊓F (two atoms).
    # normalise_counterexample must find an atomic RHS that is a genuine
    # positive CE: O-entailed, H-not-entailed, and in Σ_O.
    rhs = ELConcept(atoms=frozenset({"E", "F"}))
    result = normalise_counterexample(GCI(AB, rhs), SIG, mq, h_mq)
    assert len(result.rhs.atoms) == 1
    assert not result.rhs.existentials
    atom = next(iter(result.rhs.atoms))
    assert atom in SIG
    assert mq(result)
    assert not h_mq(result)


def test_31_normalise_case_b_hmq_filter(mq):
    # H already entails C ⊑ A and C ⊑ B (trivially true conjuncts); the
    # function must skip those and find E instead (O |= A⊓B ⊑ E, GCI 4).
    blocked = {A, B}
    h_mq_partial = lambda gci: ELConcept(atoms=gci.rhs.atoms) in blocked and not gci.rhs.existentials
    result = normalise_counterexample(GCI(AB, E), SIG, mq, h_mq_partial)
    assert result.rhs not in blocked
    assert len(result.rhs.atoms) == 1
    assert not result.rhs.existentials
    assert mq(result)


def test_32_normalise_case1_recursive(h_mq):
    # Case 1: C = ∃r.A, D = ∃r.B.  Case (b) finds nothing.  C has conjunct
    # ∃r.A matching role r of D; O |= A ⊑ B (sub-CE).
    # The function recurses on A ⊑ B, which is already normalised (lhs atomic).
    rB = ELConcept(existentials=frozenset({("r", B)}))
    ce = GCI(rA, rB)

    def mock_mq(gci):
        if gci == GCI(rA, rB):
            return True
        if gci == GCI(A, B):
            return True
        return False

    result = normalise_counterexample(ce, SIG, mock_mq, h_mq)
    assert result == GCI(A, B)


def test_33_normalise_case2_atomic_lhs(h_mq):
    # Case 2: C = ∃r.D, D = ∃r.C.  Case (b) finds nothing.  C has conjunct
    # ∃r.D but O ⊭ D ⊑ C (Case 1 skipped).  O |= A ⊑ ∃r.C (GCI 2), so
    # Case 2 returns the atomic-LHS CE A ⊑ ∃r.C.
    ce = GCI(rD, rC)

    def mock_mq(gci):
        if gci == GCI(rD, rC):
            return True
        if gci == GCI(A, rC):
            return True
        return False

    result = normalise_counterexample(ce, SIG, mock_mq, h_mq)
    assert result == GCI(A, rC)
    assert len(result.lhs.atoms) == 1
    assert not result.lhs.existentials


def test_34_normalise_case1_depth2_recursion():
    # CE: ∃r.∃r.A ⊑ ∃r.∃r.B.
    # Step 1: case 1 fires (LHS has ∃r.∃r.A, role matches) → recurse on ∃r.A ⊑ ∃r.B.
    # Step 2: case 1 fires again (LHS has ∃r.A, role matches) → recurse on A ⊑ B.
    # Step 3: A is atomic → return A ⊑ B immediately.
    rrA = ELConcept(existentials=frozenset({("r", rA)}))
    rB  = ELConcept(existentials=frozenset({("r", B)}))
    rrB = ELConcept(existentials=frozenset({("r", rB)}))
    ce  = GCI(rrA, rrB)

    positive = {ce, GCI(rA, rB), GCI(A, B)}
    result = normalise_counterexample(ce, SIG, lambda gci: gci in positive, lambda _: False)
    assert result == GCI(A, B)


# ===========================================================================
# 35–38  learn_el_terminology
# ===========================================================================


def test_35_completeness(learned_h, integration_oracle):
    """Every GCI in O must be entailed by the learned H."""
    missing = [g for g in TARGET_O if not integration_oracle._h_reasoner.entails(g)]
    assert missing == [], f"GCIs in O not entailed by H: {missing}"


def test_36_soundness(learned_h, integration_oracle):
    """Every GCI in H must be semantically entailed by O."""
    spurious = [g for g in learned_h if not integration_oracle.MQ(g)]
    assert spurious == [], f"GCIs in H not entailed by O: {spurious}"


def test_37_equivalence(learned_h, integration_oracle):
    """The oracle must confirm H ≡ O (EQ returns None)."""
    remaining_ce = integration_oracle.EQ(learned_h)
    assert remaining_ce is None, f"Oracle found counterexample: {remaining_ce}"


def test_38_phase1_calls_on_h_add():
    # Regression: on_H_add was previously only called inside the main loop,
    # leaving the H-reasoner uninformed about Phase 1 atomic subsumptions.
    on_h_add_calls: list[GCI] = []

    class MockOracle:
        signature = {"A", "B"}
        mq_count = 0
        eq_count = 0

        def MQ(self, gci):
            self.mq_count += 1
            return gci == GCI(A, B)

        def EQ(self, H):
            self.eq_count += 1
            return None  # converge immediately; we only care about Phase 1 behaviour

        def make_H_MQ(self, H):
            return lambda gci: gci in H

        def on_H_add(self, gci):
            on_h_add_calls.append(gci)

    learn_el_terminology(MockOracle(), max_iterations=1)
    assert GCI(A, B) in on_h_add_calls, (
        "on_H_add was not called for the Phase 1 atomic subsumption A ⊑ B"
    )
