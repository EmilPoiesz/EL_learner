import logging
import os
import sys

from EL_algorithm import Oracle, learn_el_terminology
from Reasoner import ReasonerOracle


def run_and_report(label: str, oracle: Oracle, verbose: bool = False) -> int:
    """Run the learner with *oracle*, print a report, return |H|."""
    print("\n" + "="*60)
    print(f"  {label}")
    print("="*60)

    if not verbose:
        logging.disable(logging.INFO)
    H = learn_el_terminology(oracle, max_iterations=20)
    logging.disable(logging.NOTSET)

    target_O = oracle._O

    print(f"\n  Iterations to convergence: {len(H) - len([g for g in H if g in target_O])}"
          f" extra axiom(s) accumulated")
    print(f"  |H| = {len(H)}  (target |O| = {len(target_O)})")

    print("\n  Learned H:")
    for gci in sorted(H, key=str):
        tag = "  " if gci in target_O else "* "   # * marks redundant-but-correct axioms
        print(f"    {tag}{gci}")
    print("  (* = derivable from O but not a primitive axiom)")

    extra   = [g for g in H if not oracle.MQ(g)]
    covered = all(oracle.MQ(g) or g in H for g in target_O)
    if covered and not extra:
        print("\n  ✓  H ≡ O")
    else:
        print("\n  ✗  H ≢ O")

    return len(H)


def demo(filename: str = "ontologies/medical.ttl", verbose: bool = False, reasoner: str = "elk"):
    """
    Demonstrate the algorithm on a tiny hand-coded EL terminology loaded from medical.ttl.
    """
    print("\n" + "="*60)
    print(" EL Learner – Demo")
    print("="*60 + "\n")

    print(f"\n  Loading Turtle file: {filename}")
    print(f"  Reasoner: {reasoner}")

    _TTL_PATH    = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
    _PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
    try:
        reasoning_oracle = ReasonerOracle(
            path=_TTL_PATH,
            gateway_jar_dir=_PROJECT_DIR,
            reasoner=reasoner,
            oracle_skills={
                "saturate_left":    0.8,
                "unsaturate_right": 0.5,
                "compose_left":     0.6,
                "compose_right":    0.6,
            }
        )
        result = run_and_report(f"Run: Reasoning Oracle on medical.ttl ({reasoner})", reasoning_oracle, verbose=verbose)
    except Exception as exc:
        print(f"  ✗  Turtle loader failed: {exc}")
        return

    print("\n" + "="*60)
    print("  Summary")
    print("="*60)
    print(f"  Reasoning oracle: |H| = {result}")


if __name__ == "__main__":
    verbose = "-v" in sys.argv

    reasoner = "elk"
    for arg in sys.argv[1:]:
        if arg.startswith("--reasoner="):
            reasoner = arg.split("=", 1)[1]
            break

    demo(verbose=verbose, reasoner=reasoner)