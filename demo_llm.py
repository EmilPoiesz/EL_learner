"""
demo_llm.py — EL learner driven by a local HuggingFace LLM oracle.

Model
-----
HuggingFaceTB/SmolLM2-1.7B-Instruct  (~1.7 B parameters)
  - No authentication required
  - Instruction-tuned, follows structured yes/no and Manchester syntax prompts
  - Small enough to run on CPU (slow) or a single consumer GPU

Concept of the demo
-------------------
We want to learn what the LLM "knows" about animal taxonomy.  The LLM acts
as the implicit target ontology O: its yes/no answers to MQ calls and its
counterexample GCIs during EQ calls define O.  We use the EL learning
algorithm to reconstruct that knowledge as a set of GCIs.

The signature Σ_O is declared upfront so both the learner and the LLM know
which concept names are in scope.

Usage
-----
  python demo_llm.py              # run full learning loop
  python demo_llm.py --dry-run    # only print prompts, no model loaded
  python demo_llm.py --device cuda
"""

from __future__ import annotations

import os
import sys

from EL_algorithm import ELConcept, GCI, learn_el_terminology
from HypothesisReasoner import HypothesisReasoner, _find_free_port
from LLMOracle import (
    LLMOracle,
    gci_to_manchester,
    hypothesis_to_manchester,
)
from ReasonerOracle import ReasonerOracle  # reuse jar-finding helpers

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL = "HuggingFaceTB/SmolLM2-1.7B-Instruct"

# Σ_O — concept names the LLM will reason about.
#
# Kinship domain: atomic subsumptions (Father ⊑ Parent, etc.) are found in
# Phase 1.  Conjunctive GCIs such as  Male ⊓ Parent ⊑ Father  cannot be
# derived from atomic pairs alone, so they must come as EQ counterexamples.
SIGNATURE: set[str] = {"Person", "Male", "Female", "Parent", "Father", "Mother"}

# A few hand-picked GCIs to probe during the dry-run preview
PREVIEW_GCIS: list[GCI] = [
    GCI(ELConcept({"Father"}),             ELConcept({"Parent"})),   # True
    GCI(ELConcept({"Father"}),             ELConcept({"Female"})),   # False
    GCI(ELConcept({"Male", "Parent"}),     ELConcept({"Father"})),   # True (conjunctive LHS)
    GCI(ELConcept({"Female", "Parent"}),   ELConcept({"Mother"})),   # True (conjunctive LHS)
]

# Hypothesis that already contains the atomic subsumptions — used in the
# dry-run EQ preview to show what the counterexample prompt looks like.
PREVIEW_HYPOTHESIS: set[GCI] = {
    GCI(ELConcept({"Father"}), ELConcept({"Parent"})),
    GCI(ELConcept({"Father"}), ELConcept({"Male"})),
    GCI(ELConcept({"Mother"}), ELConcept({"Parent"})),
    GCI(ELConcept({"Mother"}), ELConcept({"Female"})),
    GCI(ELConcept({"Father"}), ELConcept({"Person"})),
    GCI(ELConcept({"Mother"}), ELConcept({"Person"})),
    GCI(ELConcept({"Parent"}), ELConcept({"Person"})),
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_classpath() -> str:
    """Assemble the Java classpath needed for ELK + OWLGateway."""
    gateway_jar_dir = os.path.dirname(os.path.abspath(__file__))
    hermit_jar = ReasonerOracle._find_hermit_jar()
    py4j_jar   = ReasonerOracle._find_py4j_jar()
    elk_jar    = ReasonerOracle._find_elk_jar(gateway_jar_dir)
    log4j_jars = ReasonerOracle._find_log4j_jars()
    return os.pathsep.join([gateway_jar_dir, hermit_jar, py4j_jar, elk_jar] + log4j_jars)


def _section(title: str) -> None:
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Dry-run: show prompts without loading a model
# ---------------------------------------------------------------------------

def dry_run() -> None:
    _section("Dry-run — prompt preview (no model loaded)")

    # Instantiate a dummy oracle just to use its prompt builders
    dummy = object.__new__(LLMOracle)
    dummy._sig = SIGNATURE
    dummy._h_reasoner = None
    dummy._max_new_tokens = 128

    print("\n--- MQ prompts ---\n")
    for gci in PREVIEW_GCIS:
        prompt = dummy._build_mq_prompt(gci)
        print(f"GCI: {gci_to_manchester(gci)}")
        print("-" * 40)
        print(prompt)
        print()

    print("\n--- EQ prompts ---\n")
    print(f"H ({len(PREVIEW_HYPOTHESIS)} GCI(s)):")
    print(hypothesis_to_manchester(PREVIEW_HYPOTHESIS))
    print("\n[Step 1 — judge equivalence]")
    print("-" * 40)
    print(dummy._build_eq_judge_prompt(PREVIEW_HYPOTHESIS))
    print("\n[Step 2 — request counterexample, sent only if Step 1 answers 'no']")
    print("-" * 40)
    print(dummy._build_eq_counterexample_prompt(PREVIEW_HYPOTHESIS))


# ---------------------------------------------------------------------------
# Full demo: load the model and run the learner
# ---------------------------------------------------------------------------

def run_demo(device: str = "cpu", verbose: bool = False) -> None:
    _section(f"LLM Oracle Demo  —  {MODEL}")
    print(f"\n  Signature Σ_O : {sorted(SIGNATURE)}")
    print(f"  Device        : {device}")

    # ------------------------------------------------------------------
    # Build the H-reasoner (ELK via Java/py4j)
    # ------------------------------------------------------------------
    print("\n  Building classpath and starting ELK H-reasoner …")
    classpath = _build_classpath()
    h_port    = _find_free_port()
    h_reasoner = HypothesisReasoner(classpath, h_port, "elk")
    print("  H-reasoner ready.")

    # ------------------------------------------------------------------
    # Load the LLM oracle
    # ------------------------------------------------------------------
    print(f"\n  Loading model '{MODEL}' …")
    oracle = LLMOracle(
        model_name_or_path=MODEL,
        signature=SIGNATURE,
        h_reasoner=h_reasoner,
        max_new_tokens=128,
        device=device,
        verbose=verbose,
    )
    print("  Model loaded.")

    # ------------------------------------------------------------------
    # Quick MQ sanity check
    # ------------------------------------------------------------------
    _section("MQ sanity check")
    for gci in PREVIEW_GCIS:
        result = oracle.MQ(gci)
        print(f"  MQ({gci_to_manchester(gci)}) = {result}")

    # ------------------------------------------------------------------
    # Run the learning algorithm
    # ------------------------------------------------------------------
    _section("Learning EL terminology from LLM oracle")
    print("  (This may take a while — each MQ/EQ call queries the LLM)\n")

    H = learn_el_terminology(oracle, max_iterations=30)

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------
    _section("Result")
    print(f"\n  |H| = {len(H)}\n")
    print("  Learned GCIs (Manchester syntax):")
    for gci in sorted(H, key=str):
        print(f"    {gci_to_manchester(gci)}")

    oracle.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    dry     = "--dry-run" in sys.argv
    verbose = "-v" in sys.argv
    device  = "cpu"
    for arg in sys.argv[1:]:
        if arg.startswith("--device="):
            device = arg.split("=", 1)[1]

    if dry:
        dry_run()
    else:
        run_demo(device=device, verbose=verbose)
