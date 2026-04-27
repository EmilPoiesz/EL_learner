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

Tests 3, 4, 8, 12, 13, 14 do not require a live reasoner and always run.
All others are skipped automatically when the chosen reasoner cannot start.

Test catalogue
--------------
 1  saturate_concept_rhs   — atoms propagated transitively via MQ (lhs = A)
 2  saturate_concept_rhs   — filler of ∃r.C gets D added because O ⊨ B ⊑ ∃r.(C⊓D)
 3  sibling_merge — two same-role branches merged into ∃r.(C⊓D)
 4  sibling_merge — single ∃r.C branch left unchanged
 5  decompose_rhs case (a) — inner node fires, emits B ⊑ ∃s.F
 6  decompose_rhs case (b) — ∃s.F already in H; subtree stripped from filler
 7  decompose_rhs          — no atom+existential node → returns None
 8  saturate_left          — H: A⊑B adds B to the LHS of concept A
 9  decompose_left         — redundant ∃r.C stripped; essential {A,B} kept
10  decompose_left         — both atoms essential; {A,B} ⊑ E returned intact
11  decompose_left         — existential-only LHS kept intact
12  unsaturate_left        — mock: both atoms essential; LHS unchanged
13  unsaturate_left        — mock: one atom redundant; removed from LHS
14  unsaturate_left        — mock: nested atom in existential filler removed
15  unsaturate_left        — real MQ: B dropped from A⊓B (A⊑B in O collapses LHS)
16  compute_right_essential — simple ∃r.C (GCI 2 scenario)
17  compute_right_essential — sibling merge fires; two ∃r. → ∃r.(C⊓D) (GCI 3)
18  compute_right_essential — decompose_rhs case (a) inside right-essential
19  compute_left_essential  — A⊓B ⊑ E: B dropped because A⊑B in O (GCI 4)
20  compute_left_essential  — existential-only LHS kept intact (GCI 5)
21  normalise_counterexample — already normalised (atomic LHS)
22  normalise_counterexample — already normalised (atomic RHS)
23  normalise_counterexample — compound RHS; extracts one atomic RHS entailed by O
23c normalise_counterexample — case (b) H_MQ filter: atoms already in H are skipped
23d normalise_counterexample — case 1: recursive reduction on matching existential filler
23e normalise_counterexample — case 2: atomic LHS via Lemma 3 (no matching filler in C)
24  Integration             — full learn_el_terminology on test_minimal.ttl
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
    result.  All three test_24_* sub-tests share this fixture so the algorithm
    executes exactly once per session.
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
    # lhs = B; O ⊨ B ⊑ ∃r.(C⊓D) (GCI 3), so filler C must gain atom D
    result = saturate_concept_rhs(rC, SIG, mq, lhs=B)
    inner = next(f for role, f in result.existentials if role == "r")
    assert inner.atoms == frozenset({"C", "D"})


# ===========================================================================
# 3–4  sibling_merge  (no reasoner required)
# ===========================================================================


def test_03a_sibling_merge_two_branches_one_existential(mq):
    merged = sibling_merge(rC_rD, lhs=B, MQ=mq)
    assert len(merged.existentials) == 1


def test_03b_sibling_merge_two_branches_filler_atoms(mq):
    merged = sibling_merge(rC_rD, lhs=B, MQ=mq)
    _, filler = next(iter(merged.existentials))
    assert filler.atoms == frozenset({"C", "D"})


def test_04_sibling_merge_single_branch(mq):
    assert sibling_merge(rC, lhs=A, MQ=mq) == rC


# ===========================================================================
# 5–7  decompose_rhs
# ===========================================================================


def test_05_decompose_rhs_case_a(mq):
    # Case (a): inner node has {B, ∃s.F}; O |= B ⊑ ∃s.F; H empty → emit B ⊑ ∃s.F.
    result = decompose_rhs(A, rBsF, rBsF, set(), mq, lambda _: False)
    assert result == GCI(B, sF)


def test_06_decompose_rhs_case_b(mq):
    # Case (b): H already knows B ⊑ ∃s.F → strip ∃s.F subtree, return A ⊑ ∃r.B.
    H: set[GCI] = {GCI(B, sF)}
    result = decompose_rhs(A, rBsF, rBsF, H, mq, lambda gci: gci in H)
    expected = GCI(A, ELConcept(existentials=frozenset({("r", B)})))
    assert result == expected


def test_07_decompose_rhs_no_mixed_node(mq):
    # ∃r.C: root has only existential, filler C has only atom → no mixed node.
    result = decompose_rhs(A, rC, rC, set(), mq, lambda _: False)
    assert result is None


# ===========================================================================
# 8  saturate_left  (no reasoner required)
# ===========================================================================


def test_08_saturate_left():
    # H contains A ⊑ B; saturating {A} via H should add B.
    H = {GCI(A, B)}
    result = saturate_left(A, H, SIG, lambda gci: gci in H)
    assert result.atoms == frozenset({"A", "B"})


# ===========================================================================
# 9–11  decompose_left
# ===========================================================================


def test_09_decompose_left_redundant_existential(req, mq):
    # LHS = A ⊓ B ⊓ ∃r.C, RHS = E.  O |= A ⊓ B ⊑ E, so ∃r.C is redundant.
    lhs = ELConcept(atoms=frozenset({"A", "B"}), existentials=frozenset({("r", C)}))
    result = decompose_left(lhs, E, set(), mq, lambda _: False)
    assert result == GCI(AB, E)


def test_10_decompose_left_both_atoms_essential(mq):
    # O ⊭ A ⊑ E and O ⊭ B ⊑ E alone; both atoms are essential.
    result = decompose_left(AB, E, set(), mq, lambda _: False)
    assert result.lhs.atoms == frozenset({"A", "B"})


def test_11_decompose_left_existential_only(mq):
    # ∃r.C is the only component; it is essential.
    result = decompose_left(rC, F, set(), mq, lambda _: False)
    assert result == GCI(rC, F)


def test_11b_decompose_left_case1_new_rhs(req, mq):
    # lhs = ∃r.G, rhs = F.
    # O |= G ⊑ E (GCI 6a) but O ⊭ G ⊑ F, so the old rhs-only check would not
    # fire.  With the new check against all of ΣO, Case 1 fires with A′ = E and
    # the result is GCI(G, E) — not GCI(∃r.G, F) and not GCI(G, F).
    rG = ELConcept(existentials=frozenset({("r", G)}))
    result = decompose_left(rG, F, set(), mq, lambda _: False, signature=SIG)
    assert result == GCI(G, E)


# ===========================================================================
# 12–15  unsaturate_left
# ===========================================================================


def test_12_unsaturate_left_essential_atoms_unchanged():
    # Mock where only A⊓B ⊑ E holds; neither A alone nor B alone forces E.
    # Neither atom can be dropped, so the LHS must be returned intact.
    def mock_mq(gci: GCI) -> bool:
        return gci.lhs == AB and gci.rhs == E

    result = unsaturate_left(AB, E, mock_mq)
    assert result.atoms == frozenset({"A", "B"})


def test_13_unsaturate_left_redundant_atom_removed():
    # Mock where A alone suffices (A ⊑ E holds), so B is redundant in A⊓B ⊑ E.
    def mock_mq(gci: GCI) -> bool:
        if gci.rhs != E:
            return False
        return gci.lhs in (AB, A)

    result = unsaturate_left(AB, E, mock_mq)
    assert result.atoms == frozenset({"A"})
    assert not result.existentials


def test_14_unsaturate_left_nested_filler_atom_removed():
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


def test_15_unsaturate_left_real_mq_b_dropped(mq):
    # O contains A ⊑ B (GCI 1), which makes A alone sufficient to satisfy
    # A⊓B ⊑ E (GCI 4).  unsaturate_left should therefore drop B from the LHS.
    result = unsaturate_left(AB, E, mq)
    assert result.atoms == frozenset({"A"})
    assert not result.existentials


# ===========================================================================
# 16–18  compute_right_essential
# ===========================================================================


def test_16_right_essential_simple(mq):
    # GCI 2: A ⊑ ∃r.C as counterexample with lhs = A.
    # Concept saturation (paper rule) uses A as the LHS oracle, so it adds
    # B, E, F to the root (O ⊨ A ⊑ B, E, F) and D to the filler C
    # (O ⊨ A ⊑ ∃r.(C⊓D) via A ⊑ B ⊑ ∃r.(C⊓D)).  No decomposition fires
    # because the original RHS has no root atoms to pivot on.
    result = compute_right_essential(A, rC, set(), mq, SIG, lambda _: False)
    expected_rhs = ELConcept(
        atoms=frozenset({"A", "B", "E", "F"}),
        existentials=frozenset({("r", ELConcept(atoms=frozenset({"C", "D"})))}),
    )
    assert result == GCI(A, expected_rhs)


def test_17_right_essential_sibling_merge(req, mq):
    # GCI 3: H has B ⊑ ∃r.C; new CE is B ⊑ ∃r.D → merge → ∃r.(C⊓D).
    H: set[GCI] = {GCI(B, rC)}
    result = compute_right_essential(B, rD, H, mq, SIG, lambda gci: gci in H)
    r, filler = next(iter(result.rhs.existentials))
    assert len(result.rhs.existentials) == 1
    assert r == "r"
    assert filler.atoms == frozenset({"C", "D"})


def test_18_right_essential_decompose_rhs_case_a(mq):
    # GCI 7: A ⊑ ∃r.(B⊓∃s.F); decompose_rhs case (a) fires → returns B ⊑ ∃s.F.
    result = compute_right_essential(A, rBsF, set(), mq, SIG, lambda _: False)
    assert result == GCI(B, sF)


# ===========================================================================
# 19–20  compute_left_essential
# ===========================================================================


def test_19a_left_essential_conjunctive_lhs_atoms(mq):
    # GCI 4: A ⊓ B ⊑ E.  O contains A ⊑ B (GCI 1), so A alone already forces E
    # (A ⊑ B makes the LHS A ⊓ B collapse to A under O).  The desaturation step
    # removes B, leaving just A as the minimal sufficient precondition.
    result = compute_left_essential(AB, E, set(), mq, SIG, lambda _: False)
    assert result.lhs.atoms == frozenset({"A"})


def test_19b_left_essential_conjunctive_lhs_rhs(mq):
    result = compute_left_essential(AB, E, set(), mq, SIG, lambda _: False)
    assert result.rhs == E


def test_20_left_essential_existential_lhs(mq):
    # GCI 5: ∃r.C ⊑ F; existential-only LHS returned unchanged.
    result = compute_left_essential(rC, F, set(), mq, SIG, lambda _: False)
    assert result == GCI(rC, F)


# ===========================================================================
# 21–23  normalise_counterexample
# ===========================================================================


def test_21_normalise_atomic_lhs(mq, h_mq):
    ce = GCI(A, rC)
    assert normalise_counterexample(ce, SIG, mq, h_mq) == ce


def test_22_normalise_atomic_rhs(mq, h_mq):
    ce = GCI(AB, E)
    assert normalise_counterexample(ce, SIG, mq, h_mq) == ce


def test_23a_normalise_compound_rhs_atomic(mq, h_mq):
    # LHS = A⊓B (non-atomic), RHS = E⊓F (two atoms).
    # normalise_counterexample must split the RHS to one atom entailed by O.
    rhs = ELConcept(atoms=frozenset({"E", "F"}))
    result = normalise_counterexample(GCI(AB, rhs), SIG, mq, h_mq)
    assert len(result.rhs.atoms) == 1
    assert not result.rhs.existentials


def test_23b_normalise_compound_rhs_valid_atom(mq, h_mq):
    # The function searches all of Σ_O for an atomic RHS that is a positive
    # counterexample (O-entailed, H-not-entailed).  The result atom need not
    # come from the original RHS — any A ∈ Σ_O with O |= A⊓B ⊑ A qualifies.
    rhs = ELConcept(atoms=frozenset({"E", "F"}))
    result = normalise_counterexample(GCI(AB, rhs), SIG, mq, h_mq)
    assert len(result.rhs.atoms) == 1
    assert not result.rhs.existentials
    atom = next(iter(result.rhs.atoms))
    assert atom in SIG
    assert mq(result)
    assert not h_mq(result)


def test_23c_normalise_case_b_hmq_filter(mq):
    # H already entails C ⊑ A and C ⊑ B (trivially true conjuncts); the
    # function must skip those and find E instead (O |= A⊓B ⊑ E, GCI 4).
    blocked = {A, B}
    h_mq_partial = lambda gci: ELConcept(atoms=gci.rhs.atoms) in blocked and not gci.rhs.existentials
    result = normalise_counterexample(GCI(AB, E), SIG, mq, h_mq_partial)
    assert result.rhs not in blocked
    assert len(result.rhs.atoms) == 1
    assert not result.rhs.existentials
    assert mq(result)


def test_23d_normalise_case1_recursive(h_mq):
    # Case 1: C = ∃r.A, D = ∃r.B.  Case (b) finds nothing (∃r.A ⊑ atom never
    # holds).  C has conjunct ∃r.A matching role r of D; O |= A ⊑ B (sub-CE).
    # The function recurses on A ⊑ B, which is already normalised (lhs atomic).
    rB = ELConcept(existentials=frozenset({("r", B)}))
    ce = GCI(rA, rB)

    def mock_mq(gci):
        if gci == GCI(rA, rB):   # CE itself is valid
            return True
        if gci == GCI(A, B):     # sub-CE for Case 1 recursion
            return True
        return False              # case (b) finds nothing

    result = normalise_counterexample(ce, SIG, mock_mq, h_mq)
    assert result == GCI(A, B)


def test_23e_normalise_case2_atomic_lhs(h_mq):
    # Case 2: C = ∃r.D, D = ∃r.C.  Case (b) finds nothing.  C has conjunct
    # ∃r.D but O ⊭ D ⊑ C (Case 1 skipped).  O |= A ⊑ ∃r.C (GCI 2), so
    # Case 2 returns the atomic-LHS CE A ⊑ ∃r.C.
    ce = GCI(rD, rC)

    def mock_mq(gci):
        if gci == GCI(rD, rC):   # CE itself is valid
            return True
        if gci == GCI(A, rC):    # Case 2: A ⊑ ∃r.C (GCI 2 analogue)
            return True
        return False              # case (b) and Case 1 find nothing

    result = normalise_counterexample(ce, SIG, mock_mq, h_mq)
    assert result == GCI(A, rC)
    assert len(result.lhs.atoms) == 1
    assert not result.lhs.existentials


# ===========================================================================
# 24  Integration: learn_el_terminology on test_minimal.ttl
# ===========================================================================


def test_24a_completeness(learned_h, integration_oracle):
    """Every GCI in O must be entailed by the learned H."""
    missing = [g for g in TARGET_O if not integration_oracle._h_reasoner.entails(g)]
    assert missing == [], f"GCIs in O not entailed by H: {missing}"


def test_24b_soundness(learned_h, integration_oracle):
    """Every GCI in H must be semantically entailed by O."""
    spurious = [g for g in learned_h if not integration_oracle.MQ(g)]
    assert spurious == [], f"GCIs in H not entailed by O: {spurious}"


def test_24c_equivalence(learned_h, integration_oracle):
    """The oracle must confirm H ≡ O (EQ returns None)."""
    remaining_ce = integration_oracle.EQ(learned_h)
    assert remaining_ce is None, f"Oracle found counterexample: {remaining_ce}"
