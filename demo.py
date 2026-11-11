"""
demo.py — EL learner demo, supporting both oracle backends.

Usage
-----
  # ReasonerOracle (default)
  python demo.py
  python demo.py --reasoner hermit
  python demo.py --ontology ontologies/medical.ttl
  python demo.py -v

  # LLMOracle on a fixed kinship signature (the model's own knowledge)
  python demo.py --oracle llm
  python demo.py --oracle llm --device mps
  python demo.py --oracle llm --dry-run
  python demo.py --oracle llm -v

  # LLMOracle with the ontology's axioms in the system prompt (--oracle axioms)
  python demo.py --oracle axioms --ontology ontologies/medical.ttl
  python demo.py --oracle axioms --ontology ontologies/medical.ttl --model <hf-model> --device mps

  # As the original ExactLearner-LLM paper: signature from the ontology, axioms
  # withheld, its prompts and True/False answers, PAC-sampling equivalence queries
  python demo.py --oracle signature --ontology ontologies/animals.owl \\
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
    from learner.llm_oracle import EQSettings
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


# The order counterexample attempts are reported in (LLMOracle.eq_outcomes).
OUTCOMES = ("unusable", "follows from H", "true accepted", "true rejected",
            "false rejected", "false accepted", "accepted", "rejected")


def format_outcomes(outcomes) -> str:
    """One line: what the counterexample attempts came to.  true / false: whether O
    entails the proposal (ELK); the learner never sees these labels."""
    parts = [f"{kind} {outcomes[kind]}" for kind in OUTCOMES if outcomes.get(kind)]
    return "Counterexample attempts: " + " · ".join(parts) + "  (true/false: in O, by ELK; not seen by the learner)"


def _run_and_report(oracle: Oracle, verbose: bool, output: str | None = None, judge: Callable[[GCI], bool] | None = None,
                    info: OntologyInfo | None = None, max_eqs: int | None = None,
                    precomputation: bool = True, max_failed_eqs: int = 0) -> None:
    """Run the learner and print a comparison report against oracle.axioms.
    ``info``: the ontology as the OWL API reads it, whose classes (with owl:Thing
    when the file mentions it) and roles the paper metric is computed over.
    ``max_eqs``: stop after this many equivalence queries (one per iteration).
    ``precomputation``: start H with an MQ for every ordered pair of concept names
    (Phase 1); without it H starts empty and grows only from counterexamples.
    ``max_failed_eqs``: EQs in a row without a usable answer before learning stops."""
    # With PAC sampling the sample budget bounds the run; every counterexample is
    # one iteration, so a cap of 50 would cut a yes-biased LLM's run short.
    max_iterations = 100_000 if getattr(oracle, "sampler", None) is not None else 50
    if max_eqs is not None:
        max_iterations = max_eqs
    if not verbose:
        logging.disable(logging.INFO)
    try:
        H = learn_el_terminology(oracle, max_iterations=max_iterations, precomputation=precomputation,
                                 max_failed_eqs=max_failed_eqs)
    finally:
        logging.disable(logging.NOTSET)

    mq_calls = oracle.mq_count
    eq_calls = oracle.eq_count
    # A reasoning model cut off before </think> (LLMOracle.truncated): read as no answer.
    truncated = getattr(oracle, "truncated", 0)

    target_O  = oracle.axioms

    H_MQ = oracle.make_H_MQ(H)
    O_MQ = judge or oracle.MQ
    extra   = [g for g in H if not O_MQ(g)]   # soundness:    O ⊨ H
    missing = [g for g in target_O if not H_MQ(g)] # completeness: H ⊨ O
    # Not an axiom of O, but O entails it.  Only these are derivable: an LLM oracle
    # can also put GCIs in H that O does not entail (the extra ones).
    redundant = [g for g in H if g not in target_O and g not in extra]

    _section("Result")
    print(f"\n  |H| = {len(H)}  (target |O| = {len(target_O)})")
    print(f"  Redundant axioms (entailed but not primitive): {len(redundant)}")
    if extra:
        print(f"  Unsound axioms (not entailed by O): {len(extra)}")
    print("\n  Learned H:")
    for gci in sorted(H, key=str):
        tag = "✗ " if gci in extra else "  " if gci in target_O else "* "
        print(f"    {tag}{gci}")
    print("  (* = derivable from O but not a primitive axiom"
          + ("; ✗ = not entailed by O)" if extra else ")"))

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

    print(f"\n  MQ calls: {mq_calls}  |  EQ calls: {eq_calls}"
          + (f"  |  cut off while reasoning: {truncated}" if truncated else ""))
    outcomes = getattr(oracle, "eq_outcomes", None)
    if outcomes:
        print("  " + format_outcomes(outcomes))
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
# LLMOracle over an ontology file (--oracle axioms / signature)
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
    temperature: float = 0.0,
    eq_settings: EQSettings | None = None,
    mq_asked_as: str = "paper",
    max_eqs: int | None = None,
    precomputation: bool = True,
    max_failed_eqs: int = 0,
) -> bool:
    """
    LLM oracle over an ontology file, scored against it.  ``include_axioms``:
    the axioms are in the system prompt ("axioms") or withheld ("signature",
    the paper's setting: the file only gives the signature and the reference).
    """
    from utils.java_utils import build_classpath
    from learner.hypothesis_reasoner import HypothesisReasoner
    from learner.llm_oracle import EQSettings, LLMOracle
    eq_settings = eq_settings or EQSettings()

    ontology_path   = os.path.join(os.path.dirname(os.path.abspath(__file__)), ontology)
    gateway_jar_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "java")

    mode = "axioms" if include_axioms else "signature"
    _section(f"EL Learner — LLMOracle (grounded / {mode})  ({model})")
    print(f"\n  Ontology : {ontology}")
    print(f"  Device   : {device}")
    print(f"  Prompt   : {prompt_style}")
    if max_new_tokens is not None:
        print(f"  Max new tokens : {max_new_tokens}")
    if temperature > 0:
        print(f"  Sampling : temperature {temperature}, Ollama's defaults, seed {seed or 0}")
    if eq == "oracle":
        language = "natural language" if prompt_style.startswith("paper-nl") else "Manchester syntax"
        print(f"  EQ       : the LLM, {eq_settings.describe(seed or 0)} ({language})"
              + (f", at most {max_eqs}" if max_eqs is not None else ""))
    if not precomputation:
        print("  Phase 1  : skipped, H starts empty (no MQ per pair of concept names)")
    if max_failed_eqs:
        print(f"  Stopping : after {max_failed_eqs + 1} EQs in a row without a usable answer")
    if mq_asked_as != "paper":
        print("  MQs      : asked as the counterexample check is (Phase 1 and refinement)")

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
                temperature=temperature,
                seed=seed or 0,
                eq_settings=eq_settings,
                truth=judge,   # labels proposals in the log; never used for learning
                mq_asked_as=mq_asked_as,
                # None keeps LLMOracle's default budget.
                **({} if max_new_tokens is None else {"max_new_tokens": max_new_tokens}),
            ) as oracle:
                print("  Model loaded.")
                print(f"  Σ = {sorted(oracle.signature)}")
                print(f"  Roles = {sorted(info.roles)}")
                learner_oracle = _with_eq(oracle, eq, epsilon, delta, seed, info)
                print()
                _run_and_report(learner_oracle, verbose=verbose, output=output, judge=judge, info=info,
                                max_eqs=max_eqs, precomputation=precomputation,
                                max_failed_eqs=max_failed_eqs)
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
    from learner import llm_prompts as prompts
    from learner.el_syntax import gci_to_manchester, hypothesis_to_manchester

    _section("Dry-run — prompt preview (no model loaded)")

    print("\n--- MQ prompts ---\n")
    for gci in _PREVIEW_GCIS:
        print(f"GCI: {gci_to_manchester(gci)}")
        print("-" * 40)
        print(prompts.mq_prompt(_LLM_SIGNATURE, _LLM_ROLES, gci))
        print()

    print("\n--- EQ prompts ---\n")
    print(f"H ({len(_PREVIEW_HYPOTHESIS)} GCI(s)):")
    print(hypothesis_to_manchester(_PREVIEW_HYPOTHESIS))
    print("\n[Step 1 — judge equivalence]")
    print("-" * 40)
    print(prompts.eq_judge_prompt(_LLM_SIGNATURE, _LLM_ROLES, _PREVIEW_HYPOTHESIS))
    print("\n[Step 2 — request counterexample]")
    print("-" * 40)
    print(prompts.eq_counterexample_prompt(_LLM_SIGNATURE, _LLM_ROLES, _PREVIEW_HYPOTHESIS))

    # --oracle axioms/signature: EQs have their own system message, and can be
    # asked in one prompt (--eq-mode combined) or in natural language (paper-nl-*).
    print("\n[EQ system message, --oracle signature]")
    print("-" * 40)
    print(prompts.EQ_SYSTEM_SIGNATURE)
    print("\n[--eq-mode combined — counterexample or yes]")
    print("-" * 40)
    print(prompts.eq_combined_prompt(_LLM_SIGNATURE, _LLM_ROLES, _PREVIEW_HYPOTHESIS))
    print("\n[--eq-mode combined, natural language (--prompt paper-nl-*)]")
    print("-" * 40)
    print(prompts.eq_combined_prompt(_LLM_SIGNATURE, _LLM_ROLES, _PREVIEW_HYPOTHESIS,
                                              language="nl"))


def _run_llm_demo(
    model: str,
    device: str,
    verbose: bool,
    output: str | None = None,
    max_new_tokens: int | None = None,
) -> bool:
    from utils.java_utils import build_classpath
    from learner.hypothesis_reasoner import HypothesisReasoner
    from learner.el_syntax import gci_to_manchester
    from learner.llm_oracle import LLMOracle

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

# The oracle kinds' names before 2026-11-11, still accepted.
ORACLE_ALIASES = {"taught": "axioms", "parametric": "signature"}


def _oracle_kind(name: str) -> str:
    return ORACLE_ALIASES.get(name, name)


def build_parser() -> argparse.ArgumentParser:
    """demo.py's command line (learner/run_config.py's catalogue mirrors its LLM options)."""
    parser = argparse.ArgumentParser(
        description="EL learner demo.  Choose an oracle backend with --oracle.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--oracle", type=_oracle_kind, choices=["reasoner", "llm", "axioms", "signature"], default="reasoner",
        help="Oracle backend: 'reasoner' (Java/ELK); 'llm' (LLM on a fixed kinship "
             "signature); 'axioms' (LLM with the ontology's axioms in its prompt); "
             "'signature' (LLM given only the ontology's concept and role names, answering "
             "from its own knowledge, as in the paper).  'axioms' and 'signature' are scored "
             "against --ontology.  The old names 'taught' and 'parametric' still work.",
    )
    parser.add_argument(
        "--eq", choices=["oracle", "pac"], default="oracle",
        help="Equivalence queries: the oracle's own ('oracle': ELK, or the LLM judge), "
             "or PAC sampling as in the paper ('pac'; reasoner, axioms, signature).",
    )
    parser.add_argument(
        "--eq-mode", choices=["judge", "combined"], default="judge",
        help="With an LLM's own equivalence queries (--eq oracle; axioms, signature): "
             "'judge' asks yes/no whether H is complete, then for a counterexample; "
             "'combined' asks once for a counterexample, or yes if H is complete.  "
             "In natural language with --prompt paper-nl-*, else in Manchester syntax.",
    )
    parser.add_argument(
        "--eq-check", choices=["mq", "nl"], default="mq",
        help="With an LLM's own equivalence queries and a paper prompt: check each "
             "counterexample with an MQ in the run's own prompt style ('mq'), or in "
             "the paper's natural-language form ('nl'), whatever the style.",
    )
    parser.add_argument(
        "--eq-check-system", choices=["paper", "s1", "s2", "s3", "s4"], default="paper",
        help="With an LLM's own equivalence queries and a paper prompt: the system prompt "
             "of the counterexample check.  'paper': the MQs' own; 's1': the paper's simple "
             "one; 's2': the advanced one with its has-part rule restated; 's3': s2 and how "
             "EL statements read; 's4': s2 and 'every member, in the direction written' "
             "(learner/llm_prompts.py CHECK_SYSTEM_PROMPTS).",
    )
    parser.add_argument(
        "--eq-check-question", choices=["paper", "every"], default="paper",
        help="With an LLM's own equivalence queries and a paper prompt: the question of the "
             "counterexample check.  'paper': the MQ as --eq-check says; 'every': \"Is it "
             "true that every C ...?\".",
    )
    parser.add_argument(
        "--eq-memory", action=argparse.BooleanOptionalAction, default=False,
        help="With an LLM's own equivalence queries: list the counterexamples the check "
             "rejected at earlier EQs in every later counterexample prompt "
             "(--no-eq-memory turns a config's memory off).",
    )
    parser.add_argument(
        "--eq-temperature", type=float, default=0.0, metavar="T",
        help="With an LLM's own equivalence queries: sample only the counterexample "
             "prompts at T (Ollama's other defaults, seeded by --seed); the MQs, the "
             "check and the judge keep --temperature.  0 (default): as the other queries.",
    )
    parser.add_argument(
        "--eq-generation", choices=["default", "g1", "g2", "g3", "g2b", "g3b"], default="default",
        help="With an LLM's own equivalence queries (judge mode): how the counterexample "
             "is asked for.  'default': placeholder examples; 'g1': with the EL forms "
             "allowed listed as rules; 'g2': concrete examples about cars; 'g3': as "
             "'Subject | relation | Object'; 'g2b', 'g3b': g2 without brackets, g3 with the "
             "format in prose (learner/llm_prompts.py GENERATIONS).",
    )
    parser.add_argument(
        "--max-eqs", type=int, default=None, metavar="N",
        help="Stop after N equivalence queries (default: 50, or the PAC budget).",
    )
    parser.add_argument(
        "--eq-attempts", type=int, default=3, metavar="N",
        help="With an LLM's own equivalence queries: attempts at a usable counterexample per EQ.",
    )
    parser.add_argument(
        "--max-failed-eqs", type=int, default=0, metavar="N",
        help="With an LLM over an ontology (axioms, signature): tolerate N EQs in a row "
             "without a usable answer before learning stops (default 0: stop at the first); "
             "the next EQ asks again, with the earlier failures in its prompt.  Failed EQs "
             "count towards --max-eqs.",
    )
    parser.add_argument(
        "--no-precompute", action="store_true",
        help="With an LLM over an ontology (axioms, signature): skip Phase 1, the MQ for "
             "every ordered pair of concept names, so H starts empty and grows only from "
             "counterexamples.",
    )
    parser.add_argument(
        "--precompute", dest="no_precompute", action="store_false",
        help="Run Phase 1 after all, when a config (or an earlier flag) skips it.",
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
             "(as upstream), else 1024 for axioms/signature and 128 for --oracle llm.  "
             "Reasoning models need room to finish thinking before their answer.",
    )
    llm_grp.add_argument(
        "--prompt", choices=["ours", "paper-simple", "paper-advanced", "paper-nl-simple", "paper-nl-advanced"],
        default="ours",
        help="MQ prompts: this project's yes/no prompts, or the original "
             "ExactLearner-LLM's True/False prompts, the axiom in Manchester syntax or, "
             "with paper-nl-*, in its natural-language form (axioms/signature only).",
    )
    llm_grp.add_argument(
        "--mq-asked-as", choices=["paper", "check"], default="paper",
        help="With a paper prompt: ask the learner's own MQs (Phase 1, refinement) as the "
             "paper does ('paper'), or exactly as the counterexample check is asked "
             "('check': --eq-check, --eq-check-system, --eq-check-question), so that one "
             "statement gets one answer within a run.",
    )
    llm_grp.add_argument(
        "--temperature", type=float, default=0.0, metavar="T",
        help="0 (default): greedy decoding.  Above 0: sample at T with Ollama's other "
             "defaults, as the paper's runs were answered (axioms/signature only); "
             "--seed then also seeds the sampling.",
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
    parser.add_argument(
        "--config", default=None, metavar="FILE",
        help="A run configuration (JSON, learner/run_config.py): its settings become the "
             "flags above, and flags given here override them.  With several ontologies or "
             "seeds in the file, pick one with --ontology / --seed.",
    )
    return parser


def parse_args(parser: argparse.ArgumentParser, argv: list[str] | None = None) -> argparse.Namespace:
    """The command line, with --config's settings placed before the flags given."""
    argv = sys.argv[1:] if argv is None else list(argv)
    known, _ = parser.parse_known_args(argv)
    if not known.config:
        return parser.parse_args(argv)
    from learner import run_config
    try:
        config = run_config.load(known.config)
        args = parser.parse_args(run_config.to_argv(config) + argv)
        run_config.check_run(config, args.ontology, args.seed, argv)
        prompts = run_config.value(config, "oracle.mq.prompt")
        if isinstance(prompts, list) and args.prompt not in prompts:
            raise run_config.ConfigError(f"oracle.mq.prompt: {args.prompt} is not in the config ({', '.join(prompts)})")
    except run_config.ConfigError as exc:
        parser.error(f"--config {known.config}: {exc}")
    return args


if __name__ == "__main__":
    parser = build_parser()
    args = parse_args(parser)

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
    elif args.oracle in ("axioms", "signature"):
        max_new_tokens = args.max_new_tokens
        if max_new_tokens is None and args.prompt != "ours":
            max_new_tokens = 2   # upstream's maxTokens for its True/False prompts
        from learner.llm_oracle import EQSettings   # loads transformers
        try:
            eq_settings = EQSettings(
                mode=args.eq_mode,
                check=args.eq_check,
                check_system=args.eq_check_system,
                check_question=args.eq_check_question,
                memory=args.eq_memory,
                temperature=args.eq_temperature,
                generation=args.eq_generation,
                attempts=args.eq_attempts,
            )
        except ValueError as exc:   # a combination the settings refuse, e.g. combined + g2
            parser.error(str(exc))
        if args.mq_asked_as == "check" and args.prompt == "ours":
            parser.error("--mq-asked-as check needs a paper prompt (--prompt paper-*)")
        ok = _run_grounded_llm_demo(
            ontology=args.ontology,
            model=args.model,
            device=args.device,
            verbose=args.verbose,
            output=args.output,
            max_new_tokens=max_new_tokens,
            include_axioms=args.oracle == "axioms",
            prompt_style=args.prompt,
            eq=args.eq,
            epsilon=args.epsilon,
            delta=args.delta,
            seed=args.seed,
            temperature=args.temperature,
            eq_settings=eq_settings,
            max_eqs=args.max_eqs,
            precomputation=not args.no_precompute,
            max_failed_eqs=args.max_failed_eqs,
            mq_asked_as=args.mq_asked_as,
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
