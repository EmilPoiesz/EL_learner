"""
test_el_algorithm.py
====================
Pytest test suite for the EL learning algorithm (EL_algorithm.py).

Each test targets a specific function or branch of Algorithm 1, using the
minimal ontology in ontologies/test_minimal.ttl (and hand-coded oracles where
isolation matters).

Run with:
    pytest                          # ELK reasoner, compact output
    pytest -v                       # verbose output
    pytest --reasoner=hermit        # HermiT reasoner
    pytest --reasoner=hermit -v

Tests 3, 4, 8 do not require a live reasoner and always run.
All others are skipped automatically when the chosen reasoner cannot start.

Test catalogue
--------------
 1  saturate_concept_rhs   — atoms propagated transitively via MQ
 2  saturate_concept_rhs   — filler of ∃r.A saturated recursively to ∃r.(A⊓B⊓E⊓F)
 3  sibling_merge — two same-role branches merged into ∃r.(C⊓D)
 4  sibling_merge — single ∃r.C branch left unchanged
 5  decompose_rhs case (a) — inner node fires, emits B ⊑ ∃s.F
 6  decompose_rhs case (b) — ∃s.F already in H; subtree stripped from filler
 7  decompose_rhs          — no atom+existential node → returns None
 8  saturate_left          — H: A⊑B adds B to the LHS of concept A
 9  decompose_left         — redundant ∃r.C stripped; essential {A,B} kept
10  decompose_left         — both atoms essential; {A,B} ⊑ E returned intact
11  decompose_left         — existential-only LHS kept intact
12  compute_right_essential — simple ∃r.C (GCI 2 scenario)
13  compute_right_essential — sibling merge fires; two ∃r. → ∃r.(C⊓D) (GCI 3)
14  compute_right_essential — decompose_rhs case (a) inside right-essential
15  compute_left_essential  — conjunctive LHS; both atoms essential (GCI 4)
16  compute_left_essential  — existential-only LHS kept intact (GCI 5)
17  normalise_counterexample — already normalised (atomic LHS)
18  normalise_counterexample — already normalised (atomic RHS)
19  normalise_counterexample — compound RHS; extracts one atomic RHS entailed by O
20  Integration             — full learn_el_terminology on test_minimal.ttl
"""

from __future__ import annotations

import pytest

from el_algorithm import (
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
def req(reasoner_oracle, reasoner_name):
    """Skip the calling test if no reasoner is available."""
    if reasoner_oracle is None:
        pytest.skip(f"Reasoner {reasoner_name!r} unavailable")
    return reasoner_oracle


@pytest.fixture(scope="session")
def learned_h(integration_oracle):
    """
    Run learn_el_terminology once against the integration oracle and cache the
    result.  All three test_20_* sub-tests share this fixture so the algorithm
    executes exactly once per session.
    """
    if integration_oracle is None:
        pytest.skip("Reasoner unavailable — integration test skipped")
    return learn_el_terminology(integration_oracle, max_iterations=100)


# ===========================================================================
# 1–2  saturate_concept_rhs
# ===========================================================================


def test_01_saturate_atoms(req, mq):
    result = saturate_concept_rhs(A, SIG, mq)
    assert result.atoms == frozenset({"A", "B", "E", "F"})


def test_02_saturate_filler(req, mq):
    result = saturate_concept_rhs(rA, SIG, mq)
    inner = next(f for role, f in result.existentials if role == "r")
    assert inner.atoms == frozenset({"A", "B", "E", "F"})


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
    result = decompose_rhs(A, rBsF, rBsF, set(), mq)
    assert result == GCI(B, sF)


def test_06_decompose_rhs_case_b(mq):
    # Case (b): H already knows B ⊑ ∃s.F → strip ∃s.F subtree, return A ⊑ ∃r.B.
    H: set[GCI] = {GCI(B, sF)}
    result = decompose_rhs(A, rBsF, rBsF, H, mq)
    expected = GCI(A, ELConcept(existentials=frozenset({("r", B)})))
    assert result == expected


def test_07_decompose_rhs_no_mixed_node(mq):
    # ∃r.C: root has only existential, filler C has only atom → no mixed node.
    result = decompose_rhs(A, rC, rC, set(), mq)
    assert result is None


# ===========================================================================
# 8  saturate_left  (no reasoner required)
# ===========================================================================


def test_08_saturate_left():
    # H contains A ⊑ B; saturating {A} via H should add B.
    result = saturate_left(A, {GCI(A, B)}, SIG)
    assert result.atoms == frozenset({"A", "B"})


# ===========================================================================
# 9–11  decompose_left
# ===========================================================================


def test_09_decompose_left_redundant_existential(req, mq):
    # LHS = A ⊓ B ⊓ ∃r.C, RHS = E.  O |= A ⊓ B ⊑ E, so ∃r.C is redundant.
    lhs = ELConcept(atoms=frozenset({"A", "B"}), existentials=frozenset({("r", C)}))
    result = decompose_left(lhs, E, set(), mq)
    assert result == AB


def test_10_decompose_left_both_atoms_essential(mq):
    # O ⊭ A ⊑ E and O ⊭ B ⊑ E alone; both atoms are essential.
    result = decompose_left(AB, E, set(), mq)
    assert result.atoms == frozenset({"A", "B"})


def test_11_decompose_left_existential_only(mq):
    # ∃r.C is the only component; it is essential.
    result = decompose_left(rC, F, set(), mq)
    assert result == rC


# ===========================================================================
# 12–14  compute_right_essential
# ===========================================================================


def test_12_right_essential_simple(mq):
    # GCI 2: A ⊑ ∃r.C — no merging or decomposition needed.
    result = compute_right_essential(A, rC, set(), mq, SIG)
    assert result == GCI(A, rC)


def test_13_right_essential_sibling_merge(req, mq):
    # GCI 3: H has B ⊑ ∃r.C; new CE is B ⊑ ∃r.D → merge → ∃r.(C⊓D).
    H: set[GCI] = {GCI(B, rC)}
    result = compute_right_essential(B, rD, H, mq, SIG)
    r, filler = next(iter(result.rhs.existentials))
    assert len(result.rhs.existentials) == 1
    assert r == "r"
    assert filler.atoms == frozenset({"C", "D"})


def test_14_right_essential_decompose_rhs_case_a(mq):
    # GCI 7: A ⊑ ∃r.(B⊓∃s.F); decompose_rhs case (a) fires → returns B ⊑ ∃s.F.
    result = compute_right_essential(A, rBsF, set(), mq, SIG)
    assert result == GCI(B, sF)


# ===========================================================================
# 15–16  compute_left_essential
# ===========================================================================


def test_15a_left_essential_conjunctive_lhs_atoms(mq):
    # GCI 4: A ⊓ B ⊑ E; both atoms are essential.
    result = compute_left_essential(AB, E, set(), mq, SIG)
    assert result.lhs.atoms == frozenset({"A", "B"})


def test_15b_left_essential_conjunctive_lhs_rhs(mq):
    result = compute_left_essential(AB, E, set(), mq, SIG)
    assert result.rhs == E


def test_16_left_essential_existential_lhs(mq):
    # GCI 5: ∃r.C ⊑ F; existential-only LHS returned unchanged.
    result = compute_left_essential(rC, F, set(), mq, SIG)
    assert result == GCI(rC, F)


# ===========================================================================
# 17–19  normalise_counterexample
# ===========================================================================


def test_17_normalise_atomic_lhs(mq):
    ce = GCI(A, rC)
    assert normalise_counterexample(ce, SIG, mq) == ce


def test_18_normalise_atomic_rhs(mq):
    ce = GCI(AB, E)
    assert normalise_counterexample(ce, SIG, mq) == ce


def test_19a_normalise_compound_rhs_atomic(mq):
    # LHS = A⊓B (non-atomic), RHS = E⊓F (two atoms).
    # normalise_counterexample must split the RHS to one atom entailed by O.
    rhs = ELConcept(atoms=frozenset({"E", "F"}))
    result = normalise_counterexample(GCI(AB, rhs), SIG, mq)
    assert len(result.rhs.atoms) == 1
    assert not result.rhs.existentials


def test_19b_normalise_compound_rhs_valid_atom(mq):
    # Both O |= A⊓B ⊑ E (GCI 4) and O |= A⊓B ⊑ F (via A⊑∃r.C⊑F) hold.
    rhs = ELConcept(atoms=frozenset({"E", "F"}))
    result = normalise_counterexample(GCI(AB, rhs), SIG, mq)
    assert next(iter(result.rhs.atoms)) in {"E", "F"}


# ===========================================================================
# 20  Integration: learn_el_terminology on test_minimal.ttl
# ===========================================================================


def test_20a_completeness(learned_h, integration_oracle):
    """Every GCI in O must be entailed by the learned H."""
    missing = [g for g in TARGET_O if not integration_oracle._h_reasoner.entails(g)]
    assert missing == [], f"GCIs in O not entailed by H: {missing}"


def test_20b_soundness(learned_h, integration_oracle):
    """Every GCI in H must be semantically entailed by O."""
    spurious = [g for g in learned_h if not integration_oracle.MQ(g)]
    assert spurious == [], f"GCIs in H not entailed by O: {spurious}"


def test_20c_equivalence(learned_h, integration_oracle):
    """The oracle must confirm H ≡ O (EQ returns None)."""
    remaining_ce = integration_oracle.EQ(learned_h)
    assert remaining_ce is None, f"Oracle found counterexample: {remaining_ce}"
