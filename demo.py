"""
demo.py — EL learner demo, supporting both oracle backends.

Usage
-----
  # ReasonerOracle (default)
  python demo.py
  python demo.py --reasoner hermit
  python demo.py --ontology ontologies/medical.ttl
  python demo.py -v

  # LLMOracle (parametric — uses the model's training knowledge)
  python demo.py --oracle llm
  python demo.py --oracle llm --device mps
  python demo.py --oracle llm --dry-run
  python demo.py --oracle llm -v

  # LLMOracle in grounded mode (taught — ontology placed in system prompt)
  python demo.py --oracle taught --ontology ontologies/medical.ttl
  python demo.py --oracle taught --ontology ontologies/medical.ttl --model <hf-model> --device mps

  # As the original ExactLearner-LLM paper: signature from the ontology, axioms
  # withheld, its prompts and True/False answers, PAC-sampling equivalence queries
  python demo.py --oracle parametric --ontology ontologies/animals.owl \\
                 --prompt paper-advanced --eq pac --model <hf-model> --device cuda
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import TYPE_CHECKING, Callable

from learner.el_algorithm import ELConcept, GCI, Oracle, learn_el_terminology

if TYPE_CHECKING:
    from utils.java_utils import OntologyInfo

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _section(title: str) -> None:
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def _with_eq(oracle: Oracle, eq: str, epsilon: float, delta: float, seed: int | None,
             info: OntologyInfo) -> Oracle:
    """Wrap *oracle* in PAC-sampling equivalence queries when ``eq == "pac"``, over
    the ontology as the OWL API reads it (*info*): its roles, owl:Thing as a concept
    name when the file mentions it, and upstream's h."""
    if eq != "pac":
        return oracle
    from learner.pac import PACOracle
    pac = PACOracle(oracle, roles=set(info.roles), epsilon=epsilon, delta=delta,
                    hypothesis_size=info.size, seed=seed or 0, top=info.top)
    s = pac.sampler
    print(f"  EQ       : PAC sampling, ε={epsilon} δ={delta} seed={s.seed}, "
          f"|X|={s.instance_space_size}, budget {s.budget} statements (h={info.size})"
          + (", owl:Thing among the concept names" if info.top else ""))
    return pac


# ---------------------------------------------------------------------------
# ReasonerOracle demo
# ---------------------------------------------------------------------------

def _run_reasoner_demo(
    ontology: str,
    reasoner: str,
    verbose: bool,
    output: str | None = None,
    seed: int | None = 0,
    eq: str = "oracle",
    epsilon: float = 0.2,
    delta: float = 0.1,
) -> bool:
    from learner.reasoner_oracle import ReasonerOracle

    ontology_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ontology)
    project_dir   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "java")

    _section("EL Learner — ReasonerOracle")
    print(f"\n  Ontology : {ontology}")
    print(f"  Reasoner : {reasoner}")

    try:
        with ReasonerOracle(
            path=ontology_path,
            gateway_jar_dir=project_dir,
            reasoner=reasoner,
            seed=seed,
            oracle_skills={
                "saturate_left":    0.8,
                "unsaturate_right": 0.5,
                "compose_left":     0.6,
                "compose_right":    0.6,
            },
        ) as oracle:
            _run_and_report(_with_eq(oracle, eq, epsilon, delta, seed, oracle.info), verbose=verbose,
                            output=output, info=oracle.info)
    except Exception as exc:
        print(f"\n  ✗  Failed: {exc}")
        return False
    return True


def _run_and_report(oracle: Oracle, verbose: bool, output: str | None = None, judge: Callable[[GCI], bool] | None = None,
                    info: OntologyInfo | None = None) -> None:
    """Run the learner and print a comparison report against oracle.axioms.
    ``info``: the ontology as the OWL API reads it, whose classes (with owl:Thing
    when the file mentions it) and roles the paper metric is computed over."""
    # With PAC sampling the sample budget bounds the run; every counterexample is
    # one iteration, so a cap of 50 would cut a yes-biased LLM's run short.
    max_iterations = 100_000 if getattr(oracle, "sampler", None) is not None else 50
    if not verbose:
        logging.disable(logging.INFO)
    try:
        H = learn_el_terminology(oracle, max_iterations=max_iterations)
    finally:
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

    H_MQ = oracle.make_H_MQ(H)
    O_MQ = judge or oracle.MQ
    extra   = [g for g in H if not O_MQ(g)]   # soundness:    O ⊨ H
    missing = [g for g in target_O if not H_MQ(g)] # completeness: H ⊨ O

    # The original ExactLearner-LLM's score: H vs O on every statement of four shapes.
    from learner.evaluation import score_ontology
    from learner.pac import role_names
    if info is not None:
        score = score_ontology(O_MQ, H_MQ, info.classes, info.roles, info.top)
    else:
        score = score_ontology(O_MQ, H_MQ, oracle.signature, role_names(target_O))

    # The parsed GCIs (target_O) are the file's EL GCIs only; O reasons over the
    # whole file.  Completeness then also needs H to entail every statement O does.
    skipped = getattr(getattr(oracle, "inner", oracle), "skipped_axioms", [])
    if skipped:
        print(f"\n  Note: {len(skipped)} axiom(s) of the file are not EL GCIs and not among the "
              f"parsed axioms (|O| = {len(target_O)}); O-reasoning uses them. Completeness is "
              "checked on the parsed axioms and on the paper's statements.")
    if not missing and not extra and not score.fn:
        print("  ✓  H ≡ O")
    else:
        print("  ✗  H ≢ O")
        for g in missing: print(f"      missing: {g}")
        if score.fn:
            print(f"      missing: {score.fn} of the paper's statements that O entails")
        for g in extra:   print(f"      extra:   {g}")

    print(f"\n  MQ calls: {mq_calls}  |  EQ calls: {eq_calls}")
    sampler = getattr(oracle, "sampler", None)
    if sampler is not None:   # PACOracle
        print(f"  PAC samples: {sampler.provided} of {sampler.budget}  |  "
              f"MQs while sampling: {oracle.sample_mq_count}")

    print(f"\n  Ontology score (paper metric, {score.total} statements, {score.tp + score.fn} entailed by O):")
    print(f"    TP {score.tp}  FN {score.fn}  FP {score.fp}  TN {score.tn}")
    print(f"    accuracy {score.accuracy:.3f}  recall {score.recall:.3f}  "
          f"precision {score.precision:.3f}  F1 {score.f1:.3f}")

    if output:
        from utils.owl_parser import save_ontology
        save_ontology(H, output)
        print(f"\n  Saved learned ontology to: {output}")


# ---------------------------------------------------------------------------
# Grounded (taught / parametric) LLMOracle demo
# ---------------------------------------------------------------------------

def _run_grounded_llm_demo(
    ontology: str,
    model: str,
    device: str,
    verbose: bool,
    output: str | None = None,
    max_new_tokens: int | None = None,
    include_axioms: bool = True,
    prompt_style: str = "ours",
    eq: str = "oracle",
    epsilon: float = 0.2,
    delta: float = 0.1,
    seed: int | None = 0,
) -> bool:
    """
    LLM oracle over an ontology file, scored against it.  ``include_axioms``:
    the axioms are in the system prompt ("taught") or withheld ("parametric",
    the paper's setting: the file only gives the signature and the reference).
    """
    from utils.java_utils import build_classpath
    from learner.hypothesis_reasoner import HypothesisReasoner
    from learner.llm_oracle import LLMOracle

    ontology_path   = os.path.join(os.path.dirname(os.path.abspath(__file__)), ontology)
    gateway_jar_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "java")

    mode = "taught" if include_axioms else "parametric"
    _section(f"EL Learner — LLMOracle (grounded / {mode})  ({model})")
    print(f"\n  Ontology : {ontology}")
    print(f"  Device   : {device}")
    print(f"  Prompt   : {prompt_style}")
    if max_new_tokens is not None:
        print(f"  Max new tokens : {max_new_tokens}")

    classpath = build_classpath(gateway_jar_dir)
    print("\n  Starting ELK H-reasoner …")
    h_reasoner = HypothesisReasoner(classpath, "elk")
    print("  H-reasoner ready.")

    print(f"\n  Loading model '{model}' …")
    try:
        # The judge reasons over the ontology file as the OWL API reads it, which is
        # also where the signature, the roles and the PAC budget's h come from.
        with HypothesisReasoner(classpath, "elk") as judge:
            info = judge.load_ontology(ontology_path)
            with LLMOracle(
                model_name_or_path=model,
                signature=set(info.classes),
                roles=set(info.roles),
                h_reasoner=h_reasoner,
                ontology_path=ontology_path,
                include_axioms=include_axioms,
                prompt_style=prompt_style,
                device=device,
                verbose=verbose,
                # None keeps LLMOracle's default budget.
                **({} if max_new_tokens is None else {"max_new_tokens": max_new_tokens}),
            ) as oracle:
                print("  Model loaded.")
                print(f"  Σ = {sorted(oracle.signature)}")
                print(f"  Roles = {sorted(info.roles)}")
                learner_oracle = _with_eq(oracle, eq, epsilon, delta, seed, info)
                print()
                _run_and_report(learner_oracle, verbose=verbose, output=output, judge=judge, info=info)
    except Exception as exc:
        print(f"\n  ✗  Failed: {exc}")
        return False
    return True


# ---------------------------------------------------------------------------
# LLMOracle demo
# ---------------------------------------------------------------------------

_LLM_MODEL     = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
_LLM_SIGNATURE: set[str] = {"Person", "Male", "Female", "Parent", "Father", "Mother"}
_LLM_ROLES: set[str] = set()

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
    from learner.llm_oracle import LLMOracle, gci_to_manchester, hypothesis_to_manchester

    _section("Dry-run — prompt preview (no model loaded)")

    print("\n--- MQ prompts ---\n")
    for gci in _PREVIEW_GCIS:
        print(f"GCI: {gci_to_manchester(gci)}")
        print("-" * 40)
        print(LLMOracle._build_mq_prompt(_LLM_SIGNATURE, _LLM_ROLES, gci))
        print()

    print("\n--- EQ prompts ---\n")
    print(f"H ({len(_PREVIEW_HYPOTHESIS)} GCI(s)):")
    print(hypothesis_to_manchester(_PREVIEW_HYPOTHESIS))
    print("\n[Step 1 — judge equivalence]")
    print("-" * 40)
    print(LLMOracle._build_eq_judge_prompt(_LLM_SIGNATURE, _LLM_ROLES, _PREVIEW_HYPOTHESIS))
    print("\n[Step 2 — request counterexample]")
    print("-" * 40)
    print(LLMOracle._build_eq_counterexample_prompt(_LLM_SIGNATURE, _LLM_ROLES, _PREVIEW_HYPOTHESIS))


def _run_llm_demo(
    model: str,
    device: str,
    verbose: bool,
    output: str | None = None,
    max_new_tokens: int | None = None,
) -> bool:
    from utils.java_utils import build_classpath
    from learner.hypothesis_reasoner import HypothesisReasoner
    from learner.llm_oracle import LLMOracle, gci_to_manchester

    _section(f"EL Learner — LLMOracle  ({model})")
    print(f"\n  Signature Σ_O : {sorted(_LLM_SIGNATURE)}")
    print(f"  Device        : {device}")

    gateway_jar_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "java")
    print("\n  Starting ELK H-reasoner …")
    h_reasoner = HypothesisReasoner(build_classpath(gateway_jar_dir), "elk")
    print("  H-reasoner ready.")

    print(f"\n  Loading model '{model}' …")
    try:
        with LLMOracle(
            model_name_or_path=model,
            signature=_LLM_SIGNATURE,
            h_reasoner=h_reasoner,
            max_new_tokens=128 if max_new_tokens is None else max_new_tokens,
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
            H = learn_el_terminology(oracle, max_iterations=50)
            mq_calls = oracle.mq_count
            eq_calls = oracle.eq_count

            _section("Result")
            print(f"\n  |H| = {len(H)}\n")
            print("  Learned GCIs (Manchester syntax):")
            for gci in sorted(H, key=str):
                print(f"    {gci_to_manchester(gci)}")
            print(f"\n  MQ calls: {mq_calls}  |  EQ calls: {eq_calls}")

            if output:
                from utils.owl_parser import save_ontology
                save_ontology(H, output)
                print(f"\n  Saved learned ontology to: {output}")
    except Exception as exc:
        print(f"\n  ✗  Failed: {exc}")
        return False
    return True


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="EL learner demo.  Choose an oracle backend with --oracle.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--oracle", choices=["reasoner", "llm", "taught", "parametric"], default="reasoner",
        help="Oracle backend: 'reasoner' (Java/ELK); 'llm' (LLM on a fixed kinship "
             "signature); 'taught' (LLM with the ontology's axioms in its prompt); "
             "'parametric' (LLM given only the ontology's signature, as in the paper). "
             "'taught' and 'parametric' are scored against --ontology.",
    )
    parser.add_argument(
        "--eq", choices=["oracle", "pac"], default="oracle",
        help="Equivalence queries: the oracle's own ('oracle': ELK, or the LLM judge), "
             "or PAC sampling as in the paper ('pac'; reasoner, taught, parametric).",
    )
    parser.add_argument("--epsilon", type=float, default=0.2, help="PAC accuracy ε (with --eq pac).")
    parser.add_argument("--delta", type=float, default=0.1, help="PAC confidence δ (with --eq pac).")

    # --- ReasonerOracle options ---
    reasoner_grp = parser.add_argument_group("ReasonerOracle options")
    reasoner_grp.add_argument(
        "--reasoner", choices=["elk", "hermit"], default="elk",
        help="DL reasoner (only used with --oracle reasoner).",
    )
    reasoner_grp.add_argument(
        "--seed", type=int, default=0,
        help="Seed for the oracle skills' random choices (--oracle reasoner) "
             "and for the PAC statement sequence (--eq pac).",
    )
    reasoner_grp.add_argument(
        "--ontology", default="ontologies/animals.owl",
        help="Path to the target ontology file (.ttl Turtle or .owl RDF/XML).",
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
        "--max-new-tokens", type=int, default=None, metavar="N",
        help="Generation budget per LLM query.  Unset: 2 with a paper prompt "
             "(as upstream), else 1024 for taught/parametric and 128 for --oracle llm.  "
             "Reasoning models need room to finish thinking before their answer.",
    )
    llm_grp.add_argument(
        "--prompt", choices=["ours", "paper-simple", "paper-advanced"], default="ours",
        help="MQ prompts: this project's yes/no prompts, or the original "
             "ExactLearner-LLM's True/False prompts (taught/parametric only).",
    )
    llm_grp.add_argument(
        "--dry-run", action="store_true",
        help="Print prompts without loading the model (LLM oracle only).",
    )

    # --- Shared ---
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument(
        "--output", default=None, metavar="FILE",
        help="Save the learned ontology to FILE (.ttl for Turtle, .owl/.xml for RDF/XML).",
    )

    args = parser.parse_args()

    if args.output and os.sep not in args.output and "/" not in args.output:
        args.output = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "ontologies", "learned", args.output,
        )

    ok = True
    if args.oracle == "reasoner":
        ok = _run_reasoner_demo(
            ontology=args.ontology,
            reasoner=args.reasoner,
            verbose=args.verbose,
            output=args.output,
            seed=args.seed,
            eq=args.eq,
            epsilon=args.epsilon,
            delta=args.delta,
        )
    elif args.oracle in ("taught", "parametric"):
        max_new_tokens = args.max_new_tokens
        if max_new_tokens is None and args.prompt != "ours":
            max_new_tokens = 2   # upstream's maxTokens for its True/False prompts
        ok = _run_grounded_llm_demo(
            ontology=args.ontology,
            model=args.model,
            device=args.device,
            verbose=args.verbose,
            output=args.output,
            max_new_tokens=max_new_tokens,
            include_axioms=args.oracle == "taught",
            prompt_style=args.prompt,
            eq=args.eq,
            epsilon=args.epsilon,
            delta=args.delta,
            seed=args.seed,
        )
    else:
        if args.dry_run:
            _llm_dry_run()
        else:
            ok = _run_llm_demo(
                model=args.model,
                device=args.device,
                verbose=args.verbose,
                output=args.output,
                max_new_tokens=args.max_new_tokens,
            )

    # Non-zero on failure, so scripts and batch jobs see it.
    sys.exit(0 if ok else 1)
