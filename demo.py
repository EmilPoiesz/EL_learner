"""
demo.py — EL learner demo, supporting both oracle backends.

Usage
-----
  # ReasonerOracle (default)
  python demo.py
  python demo.py --reasoner hermit
  python demo.py --ontology ontologies/medical.ttl
  python demo.py -v

  # LLMOracle
  python demo.py --oracle llm
  python demo.py --oracle llm --device mps
  python demo.py --oracle llm --dry-run
  python demo.py --oracle llm -v
"""

from __future__ import annotations

import argparse
import logging
import os

from el_algorithm import ELConcept, GCI, Oracle, learn_el_terminology

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _section(title: str) -> None:
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# ReasonerOracle demo
# ---------------------------------------------------------------------------

def _run_reasoner_demo(
    ontology: str,
    reasoner: str,
    verbose: bool,
) -> None:
    from reasoner_oracle import ReasonerOracle

    ttl_path    = os.path.join(os.path.dirname(os.path.abspath(__file__)), ontology)
    project_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "java")

    _section("EL Learner — ReasonerOracle")
    print(f"\n  Ontology : {ontology}")
    print(f"  Reasoner : {reasoner}")

    try:
        with ReasonerOracle(
            path=ttl_path,
            gateway_jar_dir=project_dir,
            reasoner=reasoner,
            oracle_skills={
                "saturate_left":    0.8,
                "unsaturate_right": 0.5,
                "compose_left":     0.6,
                "compose_right":    0.6,
            },
        ) as oracle:
            _run_and_report(oracle, verbose=verbose)
    except Exception as exc:
        print(f"\n  ✗  Failed: {exc}")


def _run_and_report(oracle: Oracle, verbose: bool) -> None:
    """Run the learner and print a comparison report against oracle.axioms."""
    if not verbose:
        logging.disable(logging.INFO)
    H = learn_el_terminology(oracle, max_iterations=20)
    logging.disable(logging.NOTSET)

    mq_calls = oracle.mq_count
    eq_calls = oracle.eq_count

    target_O  = oracle.axioms
    redundant = [g for g in H if g not in target_O]

    _section("Result")
    print(f"\n  |H| = {len(H)}  (target |O| = {len(target_O)})")
    print(f"  Redundant axioms (entailed but not primitive): {len(redundant)}")
    print("\n  Learned H:")
    for gci in sorted(H, key=str):
        tag = "  " if gci in target_O else "* "
        print(f"    {tag}{gci}")
    print("  (* = derivable from O but not a primitive axiom)")

    extra   = [g for g in H if not oracle.MQ(g)]
    covered = all(oracle.MQ(g) or g in H for g in target_O)
    print()
    if covered and not extra:
        print("  ✓  H ≡ O")
    else:
        print("  ✗  H ≢ O")

    print(f"\n  MQ calls: {mq_calls}  |  EQ calls: {eq_calls}")


# ---------------------------------------------------------------------------
# LLMOracle demo
# ---------------------------------------------------------------------------

_LLM_MODEL     = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
_LLM_SIGNATURE: set[str] = {"Person", "Male", "Female", "Parent", "Father", "Mother"}

# A few representative GCIs used in the dry-run preview and MQ sanity check.
_PREVIEW_GCIS: list[GCI] = [
    GCI(ELConcept({"Father"}),           ELConcept({"Parent"})),
    GCI(ELConcept({"Father"}),           ELConcept({"Female"})),
    GCI(ELConcept({"Male", "Parent"}),   ELConcept({"Father"})),
    GCI(ELConcept({"Female", "Parent"}), ELConcept({"Mother"})),
]

# Partial hypothesis used in the EQ dry-run preview.
_PREVIEW_HYPOTHESIS: set[GCI] = {
    GCI(ELConcept({"Father"}), ELConcept({"Parent"})),
    GCI(ELConcept({"Father"}), ELConcept({"Male"})),
    GCI(ELConcept({"Mother"}), ELConcept({"Parent"})),
    GCI(ELConcept({"Mother"}), ELConcept({"Female"})),
    GCI(ELConcept({"Father"}), ELConcept({"Person"})),
    GCI(ELConcept({"Mother"}), ELConcept({"Person"})),
    GCI(ELConcept({"Parent"}), ELConcept({"Person"})),
}


def _llm_dry_run() -> None:
    from llm_oracle import LLMOracle, gci_to_manchester, hypothesis_to_manchester

    _section("Dry-run — prompt preview (no model loaded)")

    print("\n--- MQ prompts ---\n")
    for gci in _PREVIEW_GCIS:
        print(f"GCI: {gci_to_manchester(gci)}")
        print("-" * 40)
        print(LLMOracle._build_mq_prompt(_LLM_SIGNATURE, gci))
        print()

    print("\n--- EQ prompts ---\n")
    print(f"H ({len(_PREVIEW_HYPOTHESIS)} GCI(s)):")
    print(hypothesis_to_manchester(_PREVIEW_HYPOTHESIS))
    print("\n[Step 1 — judge equivalence]")
    print("-" * 40)
    print(LLMOracle._build_eq_judge_prompt(_LLM_SIGNATURE, _PREVIEW_HYPOTHESIS))
    print("\n[Step 2 — request counterexample]")
    print("-" * 40)
    print(LLMOracle._build_eq_counterexample_prompt(_LLM_SIGNATURE, _PREVIEW_HYPOTHESIS))


def _run_llm_demo(model: str, device: str, verbose: bool) -> None:
    from utils.java_utils import build_classpath
    from hypothesis_reasoner import HypothesisReasoner
    from llm_oracle import LLMOracle, gci_to_manchester

    _section(f"EL Learner — LLMOracle  ({model})")
    print(f"\n  Signature Σ_O : {sorted(_LLM_SIGNATURE)}")
    print(f"  Device        : {device}")

    gateway_jar_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "java")
    print("\n  Starting ELK H-reasoner …")
    h_reasoner = HypothesisReasoner(build_classpath(gateway_jar_dir), "elk")
    print("  H-reasoner ready.")

    print(f"\n  Loading model '{model}' …")
    with LLMOracle(
        model_name_or_path=model,
        signature=_LLM_SIGNATURE,
        h_reasoner=h_reasoner,
        max_new_tokens=128,
        device=device,
        verbose=verbose,
    ) as oracle:
        print("  Model loaded.")

        _section("MQ sanity check")
        for gci in _PREVIEW_GCIS:
            print(f"  MQ({gci_to_manchester(gci)}) = {oracle.MQ(gci)}")

        _section("Learning EL terminology from LLM oracle")
        print("  (Each MQ/EQ call queries the LLM — this may take a while)\n")
        oracle.reset_counts()
        H = learn_el_terminology(oracle, max_iterations=30)
        mq_calls = oracle.mq_count
        eq_calls = oracle.eq_count

        _section("Result")
        print(f"\n  |H| = {len(H)}\n")
        print("  Learned GCIs (Manchester syntax):")
        for gci in sorted(H, key=str):
            print(f"    {gci_to_manchester(gci)}")
        print(f"\n  MQ calls: {mq_calls}  |  EQ calls: {eq_calls}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="EL learner demo.  Choose an oracle backend with --oracle.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--oracle", choices=["reasoner", "llm"], default="reasoner",
        help="Oracle backend to use.",
    )

    # --- ReasonerOracle options ---
    reasoner_grp = parser.add_argument_group("ReasonerOracle options")
    reasoner_grp.add_argument(
        "--reasoner", choices=["elk", "hermit"], default="elk",
        help="DL reasoner (only used with --oracle reasoner).",
    )
    reasoner_grp.add_argument(
        "--ontology", default="ontologies/medical.ttl",
        help="Path to the target ontology Turtle file.",
    )

    # --- LLMOracle options ---
    llm_grp = parser.add_argument_group("LLMOracle options")
    llm_grp.add_argument(
        "--model", default=_LLM_MODEL,
        help="HuggingFace model name or local path.",
    )
    llm_grp.add_argument(
        "--device", default="cpu",
        help="Inference device: cpu, cuda, or mps.",
    )
    llm_grp.add_argument(
        "--dry-run", action="store_true",
        help="Print prompts without loading the model (LLM oracle only).",
    )

    # --- Shared ---
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()

    if args.oracle == "reasoner":
        _run_reasoner_demo(
            ontology=args.ontology,
            reasoner=args.reasoner,
            verbose=args.verbose,
        )
    else:
        if args.dry_run:
            _llm_dry_run()
        else:
            _run_llm_demo(
                model=args.model,
                device=args.device,
                verbose=args.verbose,
            )
