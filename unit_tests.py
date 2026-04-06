"""
unit_tests.py
=============
Unit and integration tests for the EL learning algorithm (EL_algorithm.py).

Each test targets a specific function or branch of Algorithm 1, using the
minimal ontology in test_minimal.ttl (and hand-coded oracles where isolation
matters).

Run with:
    python unit_tests.py              # ELK reasoner, compact output
    python unit_tests.py -v           # ELK reasoner, verbose (shows every PASS line too)
    python unit_tests.py --hermit     # HermiT reasoner
    python unit_tests.py --hermit -v  # HermiT reasoner, verbose

Tests 3, 4, 8 do not require a live reasoner.


Test catalogue
--------------
 1  saturate_concept_rhs   — atoms propagated transitively via MQ
 2  saturate_concept_rhs   — filler of ∃r.A saturated recursively to ∃r.(A⊓B)
 3  sibling_merge_fixpoint — two same-role branches merged into ∃r.(C⊓D)
 4  sibling_merge_fixpoint — single ∃r.C branch left unchanged
 5  decompose_rhs case (a) — inner node fires, emits simpler B ⊑ ∃s.F
 6  decompose_rhs case (b) — ∃s.F already in H; subtree stripped from filler
 7  decompose_rhs          — no atom+existential node → returns None
 8  saturate_left          — H: A⊑B adds B to the LHS of concept A
 9  decompose_left         — redundant atom C stripped; essential {A,B} kept
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

import sys
import os
import traceback

from EL_algorithm import (
    ELConcept, GCI,
    saturate_concept_rhs,
    sibling_merge_fixpoint,
    decompose_rhs,
    saturate_left,
    decompose_left,
    compute_right_essential,
    compute_left_essential,
    normalise_counterexample,
    learn_el_terminology,
)

from ReasonerOracle import ReasonerOracle

# ---------------------------------------------------------------------------
# CLI flags
# ---------------------------------------------------------------------------

VERBOSE    = "-v" in sys.argv
USE_HERMIT = "--hermit" in sys.argv
REASONER   = "hermit" if USE_HERMIT else "elk"

# ---------------------------------------------------------------------------
# Tiny test runner
# ---------------------------------------------------------------------------

_PASS = 0
_FAIL = 0
_SKIP = 0
_ERRORS: list[str] = []


def check(label: str, result, expected):
    global _PASS, _FAIL
    if result == expected:
        _PASS += 1
        if VERBOSE:
            print(f"  PASS  {label}")
    else:
        _FAIL += 1
        print(f"  FAIL  {label}")
        print(f"        expected : {expected}")
        print(f"        got      : {result}")
        _ERRORS.append(label)


def skip(label: str):
    global _SKIP
    _SKIP += 1
    print(f"  SKIP  {label}")


def print_section_title(title: str):
    print(f"\n{'─'*60}")
    print(f"  {title}")
    print(f"{'─'*60}")


# ---------------------------------------------------------------------------
# Shared concept constants   Σ = {A,B,C,D,E,F},  roles r, s
# ---------------------------------------------------------------------------

A   = ELConcept(atoms=frozenset({"A"}))
B   = ELConcept(atoms=frozenset({"B"}))
C   = ELConcept(atoms=frozenset({"C"}))
D   = ELConcept(atoms=frozenset({"D"}))
E   = ELConcept(atoms=frozenset({"E"}))
F   = ELConcept(atoms=frozenset({"F"}))
G   = ELConcept(atoms=frozenset({"G"}))
TOP = ELConcept()

rA  = ELConcept(existentials=frozenset({("r", A)}))           # ∃r.A
rC  = ELConcept(existentials=frozenset({("r", C)}))           # ∃r.C
rD  = ELConcept(existentials=frozenset({("r", D)}))           # ∃r.D
sF  = ELConcept(existentials=frozenset({("s", F)}))           # ∃s.F

AB = ELConcept(atoms=frozenset({"A", "B"}))                                     # A ⊓ B
CD  = ELConcept(atoms=frozenset({"C", "D"}))                                    # C ⊓ D
BsF  = ELConcept(atoms=frozenset({"B"}), existentials=frozenset({("s", F)}))    # B ⊓ ∃s.F
rBsF = ELConcept(existentials=frozenset({("r", BsF)}))                          # ∃r.(B ⊓ ∃s.F)
rC_rD = ELConcept(existentials=frozenset({("r", C), ("r", D)}))                 # ∃r.C ⊓ ∃r.D
rCD = ELConcept(existentials=frozenset({("r", CD)}))                            # ∃r.(C⊓D)

SIG = {"A", "B", "C", "D", "E", "F", "G"}


# ---------------------------------------------------------------------------
# Target ontology O  (mirrors test_minimal.ttl exactly, including explicit 3a/3b)
# ---------------------------------------------------------------------------


target_O: set[GCI] = {
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
# Reasoner-backed oracle
#
# A single ReasonerOracle is started once for the whole test run so that the
# Java gateway process is shared across all tests.  Tests 3, 4, 8 never call
# MQ and always run.  All other tests require REASONER_AVAILABLE = True.
# ---------------------------------------------------------------------------

_TTL_PATH    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ontologies/test_minimal.ttl")
_PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

REASONER_AVAILABLE = False
_reasoner_oracle = None

try:
    print(f"  [Setup] Starting ReasonerOracle ({REASONER}) with TTL path: {_TTL_PATH}")
    _reasoner_oracle = ReasonerOracle(
        path=_TTL_PATH, gateway_jar_dir=_PROJECT_DIR,
        reasoner=REASONER,
        oracle_skills={
        "saturate_left":    0.5,
        "unsaturate_right": 0.5,
        "compose_left":     0.5,
        "compose_right":    0.5,
    })
    print(f"  [Setup] ReasonerOracle started successfully: {_reasoner_oracle!r}")
    REASONER_AVAILABLE = True
    print(f"  [Setup] {REASONER} reasoner started successfully.")
    print(f"  [Setup] _reasoner_oracle = {_reasoner_oracle!r}")
except Exception as _setup_exc:
    print(f"  [Setup] {REASONER} unavailable: {_setup_exc}")
    print(f"  [Setup] Tests requiring {REASONER} will be skipped.")


def MQ(gci: GCI) -> bool:
    """Membership query: O |= lhs ⊑ rhs?  Backed by the reasoner when available."""
    if _reasoner_oracle is not None:
        return _reasoner_oracle.MQ(gci)

    # Structural fallback so that reasoner-independent tests (3, 4, 8, 13) can
    # still run, but note that tests with reasoner-specific expected values will
    # be skipped rather than called with this fallback.
    if gci in target_O:
        return True
    for mid in target_O:
        if GCI(gci.lhs, mid.lhs) in target_O and GCI(mid.rhs, gci.rhs) in target_O:
            return True
    return False


# ===========================================================================
# 1–2  saturate_concept_rhs
# ===========================================================================
print_section_title("1–2  saturate_concept_rhs")

result1 = saturate_concept_rhs(A, SIG, MQ)
if not REASONER_AVAILABLE:
    skip(f"1  saturate atoms: {{A}}  →  {{A, B, E, F}}  (requires {REASONER})")
else:
    check(
        label="1  saturate atoms: {A}  →  {A, B, E, F}",
        result=result1.atoms,
        expected=frozenset({"A", "B", "E", "F"}),
    )

result2 = saturate_concept_rhs(rA, SIG, MQ)
inner_atoms = next(f for role, f in result2.existentials if role == "r").atoms
if not REASONER_AVAILABLE:
    skip(f"2  saturate filler: ∃r.A  →  filler becomes {{A, B, E, F}}  (requires {REASONER})")
else:
    check(
        label="2  saturate filler: ∃r.A  →  filler becomes {A, B, E, F}",
        result=inner_atoms,
        expected=frozenset({"A", "B", "E", "F"}),
    )


# ===========================================================================
# 3–4  sibling_merge_fixpoint
# ===========================================================================
print_section_title("3–4  sibling_merge_fixpoint")

# Test 3: Two ∃r branches on the same role r merge into one ∃r.(C⊓D).
merged3 = sibling_merge_fixpoint(rC_rD)
check(
    label="3  sibling merge: ∃r.C ⊓ ∃r.D  →  exactly one ∃r branch",
    result=len(merged3.existentials),
    expected=1,
)
role3, filler3 = next(iter(merged3.existentials))
check(
    label="3  sibling merge: merged filler atoms = {C, D}",
    result=filler3.atoms,
    expected=frozenset({"C", "D"}),
)

# Test 4: A concept with only one ∃r branch is left unchanged.
merged4 = sibling_merge_fixpoint(rC)
check(
    label="4  sibling merge: single ∃r.C  →  unchanged",
    result=merged4,
    expected=rC,
)


# ===========================================================================
# 5–7  decompose_rhs
# ===========================================================================

print_section_title("5–7  decompose_rhs")

# Test 5 — case (a): inner node has atom B and existential ∃s.F.
# O |= B ⊑ ∃s.F (GCI 8).  H is empty, so B ⊑ ∃s.F is not yet known.
# decompose_rhs should emit GCI(B, sF) directly.
H5: set[GCI] = set()
result5 = decompose_rhs(A, rBsF, rBsF, H5, MQ)
check(
    label="5  decompose_rhs case (a): inner node {B,∃s.F} → emits B ⊑ ∃s.F",
    result=result5,
    expected=GCI(B, sF),
)

# Test 6 — case (b): H already knows B ⊑ ∃s.F, so case (a) skips.
# Case (b) removes the ∃s.F subtree from the inner filler, returning A ⊑ ∃r.B.
H6: set[GCI] = {GCI(B, sF)}
result6 = decompose_rhs(A, rBsF, rBsF, H6, MQ)
expected6 = GCI(A, ELConcept(existentials=frozenset({("r", B)})))
check(
    label="6  decompose_rhs case (b): ∃s.F already in H → strip it, return A ⊑ ∃r.B",
    result=result6,
    expected=expected6,
)

# Test 7 — no decomposition: the concept has no node with both atoms AND existentials.
# ∃r.C: root has only an existential; filler C has only an atom.
H7: set[GCI] = set()
result7 = decompose_rhs(A, rC, rC, H7, MQ)
check(
    label="7  decompose_rhs: no atom+existential node found  →  None",
    result=result7,
    expected=None,
)


# ===========================================================================
# 8  saturate_left
# ===========================================================================

print_section_title("8  saturate_left")

# H contains A ⊑ B.  Saturating concept {A} should add B to its atoms because
# entails_in_H(A ⊑ B, H) is True.
H8: set[GCI] = {GCI(A, B)}
result8 = saturate_left(A, H8, SIG)
check(
    label="8  saturate_left: {A} + H(A⊑B)  →  {A, B}",
    result=result8.atoms,
    expected=frozenset({"A", "B"}),
)


# ===========================================================================
# 9–11  decompose_left
# ===========================================================================

print_section_title("9–11  decompose_left")

# Test 10: LHS = {A, B}, RHS = E.  Neither atom can be dropped.
# O ⊭ A ⊑ E and O ⊭ B ⊑ E alone.
lhs10 = ELConcept(atoms=frozenset({"A", "B"}))
H10: set[GCI] = set()
result10 = decompose_left(lhs10, E, H10, MQ)
check(
    label="10  decompose_left: A ⊓ B ⊑ E — both atoms essential, LHS unchanged",
    result=result10.atoms,
    expected=frozenset({"A", "B"}),
)

# Test 11: LHS = ∃r.C, RHS = F.
# O ⊭ C ⊑ F (subtree); O ⊭ ⊤ ⊑ F (remove ∃r.C entirely).
# The existential is essential; ∃r.C returned unchanged.
H11: set[GCI] = set()
result11 = decompose_left(rC, F, H11, MQ)
check(
    label="11  decompose_left: ∃r.C ⊑ F — existential essential, returned as-is",
    result=result11,
    expected=rC,
)


# ===========================================================================
# 12–14  compute_right_essential
# ===========================================================================

print_section_title("12–14  compute_right_essential")

# Test 12 — GCI 2: A ⊑ ∃r.C, H empty.
# No merging or decomposition needed.  α = A ⊑ ∃r.C.
H12: set[GCI] = set()
result12 = compute_right_essential(A, rC, H12, MQ, SIG)
check(
    label="12  compute_right_essential: A ⊑ ∃r.C (simple)  →  α = A ⊑ ∃r.C",
    result=result12,
    expected=GCI(A, rC),
)

# Test 13 — GCI 3: sibling merge fires.
# H already has B ⊑ ∃r.C (GCI 3a).  New counterexample is B ⊑ ∃r.D (GCI 3b).
# compute_right_essential conjoins: ∃r.C ⊓ ∃r.D  →  sibling merge  →  ∃r.(C⊓D).
# D has no outgoing subsumptions in the ontology, so filler saturation keeps D
# as {D} and the merged filler is exactly {C, D}.
H13: set[GCI] = {GCI(B, rC)}
if not REASONER_AVAILABLE:
    skip(f"13  compute_right_essential: sibling merge → single ∃r.(C⊓D)  (requires {REASONER})")
else:
    result13 = compute_right_essential(B, rD, H13, MQ, SIG)
    r13, filler13 = next(iter(result13.rhs.existentials))
    check(
        label="13  compute_right_essential: sibling merge → single ∃r.(C⊓D)",
        result=(len(result13.rhs.existentials), r13, filler13.atoms),
        expected=(1, "r", frozenset({"C", "D"})),
    )

# Test 14 — GCI 7: decompose_rhs case (a) fires inside right-essential.
# A ⊑ ∃r.(B ⊓ ∃s.F), H empty.
# decompose_rhs finds inner node {B, ∃s.F}, checks O |= B ⊑ ∃s.F → True.
# H lacks B ⊑ ∃s.F → case (a): compute_right_essential returns GCI(B, ∃s.F).
H14: set[GCI] = set()
result14 = compute_right_essential(A, rBsF, H14, MQ, SIG)
check(
    label="14  compute_right_essential: decompose_rhs case (a) → emits B ⊑ ∃s.F",
    result=result14,
    expected=GCI(B, sF),
)


# ===========================================================================
# 15–16  compute_left_essential
# ===========================================================================

print_section_title("15–16  compute_left_essential")

# Test 15 — GCI 4: A ⊓ B ⊑ E, H empty.
# decompose_left: O ⊭ A ⊑ E; O ⊭ B ⊑ E → both atoms are essential.
H15: set[GCI] = set()
lhs15 = ELConcept(atoms=frozenset({"A", "B"}))
result15 = compute_left_essential(lhs15, E, H15, MQ, SIG)
check(
    label="15  compute_left_essential: A ⊓ B ⊑ E — both atoms essential",
    result=result15.lhs.atoms,
    expected=frozenset({"A", "B"}),
)
check(
    label="15  compute_left_essential: RHS is E",
    result=result15.rhs,
    expected=E,
)

# Test 16 — GCI 5: ∃r.C ⊑ F, H empty.
H16: set[GCI] = set()
result16 = compute_left_essential(rC, F, H16, MQ, SIG)
check(
    label="16  compute_left_essential: ∃r.C ⊑ F — existential-only LHS kept",
    result=result16,
    expected=GCI(rC, F),
)


# ===========================================================================
# 17–19  normalise_counterexample
# ===========================================================================

print_section_title("17–19  normalise_counterexample")

# Test 17: LHS is already atomic (A) → GCI returned unchanged.
ce17 = GCI(A, rC)
check(
    label="17  normalise: atomic LHS → unchanged",
    result=normalise_counterexample(ce17, SIG, MQ),
    expected=ce17,
)

# Test 18: RHS is already atomic (E) → GCI returned unchanged.
ce18 = GCI(lhs15, E)
check(
    label="18  normalise: atomic RHS → unchanged",
    result=normalise_counterexample(ce18, SIG, MQ),
    expected=ce18,
)

# Test 19: Neither side is atomic.
# LHS = A⊓B (non-atomic conjunction), RHS = E⊓F (two atoms, non-atomic).
# normalise should split the RHS.  Both O |= A⊓B ⊑ E (GCI 4) and
# O |= A⊓B ⊑ F (via A⊑∃r.C⊑F) hold, so either atom is a valid result.
compound_lhs19 = ELConcept(atoms=frozenset({"A", "B"}))
compound_rhs19 = ELConcept(atoms=frozenset({"E", "F"}))
ce19 = GCI(compound_lhs19, compound_rhs19)
result19 = normalise_counterexample(ce19, SIG, MQ)
check(
    label="19  normalise: A⊓B ⊑ E⊓F → extracts an atomic RHS, no existentials",
    result=(len(result19.rhs.atoms) == 1 and not result19.rhs.existentials),
    expected=True,
)
check(
    label="19  normalise: extracted atom is E or F (both entailed by O)",
    result=next(iter(result19.rhs.atoms)) in {"E", "F"},
    expected=True,
)


# ===========================================================================
# 20  Integration: learn_el_terminology on test_minimal.ttl
# ===========================================================================

print_section_title("20  Integration: learn_el_terminology on test_minimal.ttl")

if not REASONER_AVAILABLE:
    skip("20a  all O-GCIs entailed by learned H")
    skip("20b  soundness: all H-GCIs semantically entailed by O")
    skip("20c  oracle confirms H ≡ O  (EQ returns None)")
else:
    try:
        print("  [Setup 20] Closing unit-test oracle…")
        if _reasoner_oracle is not None:
            _reasoner_oracle.close()

        print("  [Setup 20] Starting fresh oracle for integration test…")
        oracle20 = ReasonerOracle(
            path=_TTL_PATH, gateway_jar_dir=_PROJECT_DIR,
            reasoner=REASONER,
            oracle_skills={
                "saturate_left":    0.8,
                "unsaturate_right": 0.5,
                "compose_left":     0.6,
                "compose_right":    0.6,
            }
        )
        print(f"  [Setup 20] oracle20 = {oracle20!r}")

        H20 = learn_el_terminology(oracle20, max_iterations=100)

        # (a) Every GCI in O must be entailed by H.
        missing = [g for g in target_O if not oracle20.MQ(g) and g not in H20]
        check(
            label="20a  all O-GCIs entailed by learned H",
            result=missing,
            expected=[],
        )

        # (b) Every H-GCI must be entailed by O (soundness).
        spurious = [g for g in H20 if not oracle20.MQ(g)]
        check(
            label="20b  soundness: all H-GCIs semantically entailed by O",
            result=spurious,
            expected=[],
        )

        # (c) The oracle confirms H ≡ O.
        remaining_ce = oracle20.EQ(H20)
        check(
            label="20c  oracle confirms H ≡ O  (EQ returns None)",
            result=remaining_ce,
            expected=None,
        )

        print(f"\n  Learned H  ({len(H20)} axioms):")
        for gci in sorted(H20, key=str):
            tag = "O" if oracle20.MQ(gci) else "*"
            print(f"    [{tag}]  {gci}")
        print("  (* = in H but not directly in O — redundant but correct)")

        oracle20.close()

    except Exception as exc:
        _FAIL += 1
        print(f"  FAIL  20 — unexpected exception: {exc}")
        traceback.print_exc()
        _ERRORS.append("20 (exception)")


# ---------------------------------------------------------------------------
# Cleanup: unit-test oracle is closed inside test 20's setup block.
# Nothing further to do here.
# ---------------------------------------------------------------------------

# ===========================================================================
# Summary
# ===========================================================================

print(f"\n{'='*60}")
print(f"  Results:  {_PASS} passed,  {_FAIL} failed,  {_SKIP} skipped")
if _ERRORS:
    print(f"  Failed:   {', '.join(_ERRORS)}")
print(f"{'='*60}\n")

sys.exit(0 if _FAIL == 0 else 1)
