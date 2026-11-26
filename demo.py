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
import json
import logging
import os
import sys
import time
from typing import TYPE_CHECKING, Callable

from learner.el_algorithm import STEPS, GCI, Oracle, learn_el_terminology

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
    reuse_counterexamples: bool = False,
    saturation: str = "paper",
    max_eqs: int | None = None,
    precomputation: bool = True,
    max_failed_eqs: int = 0,
) -> bool:
    from learner.reasoner_oracle import ReasonerOracle

    ontology_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ontology)
    project_dir   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "java")

    _section("EL Learner — ReasonerOracle")
    print(f"\n  Ontology : {ontology}")
    print(f"  Reasoner : {reasoner}")
    if reuse_counterexamples:
        print("  Reuse    : a counterexample H still misses is processed again before the next EQ")
    if saturation != "paper":
        print("  Saturate : names tried general first by H's hierarchy; candidates H rules out are not asked")

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
                            output=output, info=oracle.info, max_eqs=max_eqs,
                            precomputation=precomputation, max_failed_eqs=max_failed_eqs,
                            reuse_counterexamples=reuse_counterexamples, saturation=saturation)
    except Exception as exc:
        print(f"\n  ✗  Failed: {exc}")
        return False
    return True


# The order counterexample attempts are reported in (LLMOracle.eq_outcomes).
OUTCOMES = ("none", "unusable", "follows from H", "true accepted", "true rejected",
            "false rejected", "false accepted", "accepted", "rejected")


def format_outcomes(outcomes) -> str:
    """One line: what the counterexample attempts came to.  true / false: whether O
    entails the proposal (ELK); the learner never sees these labels."""
    parts = [f"{kind} {outcomes[kind]}" for kind in OUTCOMES if outcomes.get(kind)]
    return "Counterexample attempts: " + " · ".join(parts) + "  (true/false: in O, by ELK; not seen by the learner)"


def query_report(oracle: Oracle, seconds: float) -> list[str]:
    """The report's lines on what learning asked: the MQs by step of the algorithm,
    and for an LLM oracle its model calls by step and kind of query, with tokens and
    time; then one line "QUERIES {json}" with all of it, for scripts."""
    base = getattr(oracle, "inner", oracle)        # PAC: the inner oracle counts sampling too
    by_step = dict(getattr(base, "mq_by_step", {}))
    order = [s for s in (*STEPS, "other") if by_step.get(s)]
    lines = [f"Learning took {seconds:.0f} s.  MQs by step: "
             + (" · ".join(f"{s} {by_step[s]}" for s in order) or "none")]
    stats = getattr(base, "query_stats", None) or {}
    rank = {s: i for i, s in enumerate((*STEPS, "other"))}
    rows = sorted(stats.items(), key=lambda kv: (rank.get(kv[0][0], len(rank)), kv[0][1]))
    if rows:
        total = {k: sum(v[k] for _, v in rows) for k in rows[0][1]}
        head = f"    {'step / query':34} {'asked':>6} {'cached':>6} {'model':>6} {'prompt tok':>10} {'new tok':>8} {'seconds':>8} {'s/call':>6}"
        lines += ["Model calls (model = generated; tokens and seconds of the model calls):", head]
        for name, v in [(f"{s} / {k}", v) for (s, k), v in rows] + [("total", total)]:
            per = v["seconds"] / v["generated"] if v["generated"] else 0
            lines.append(f"    {name:34} {v['asked']:6} {v['cached']:6} {v['generated']:6} "
                         f"{v['prompt_tokens']:10} {v['new_tokens']:8} {v['seconds']:8.1f} {per:6.2f}")
    record = {"seconds": round(seconds, 1), "mq_by_step": by_step,
              "model_calls": [{"step": s, "query": k, **{f: round(x, 2) for f, x in v.items()}}
                              for (s, k), v in rows]}
    return lines + ["QUERIES " + json.dumps(record, ensure_ascii=False)]


def _run_and_report(oracle: Oracle, verbose: bool, output: str | None = None, judge: Callable[[GCI], bool] | None = None,
                    info: OntologyInfo | None = None, max_eqs: int | None = None,
                    precomputation: bool = True, max_failed_eqs: int = 0,
                    reuse_counterexamples: bool = False, saturation: str = "paper",
                    repair: bool = False, repair_majority: bool = False) -> None:
    """Run the learner and print a comparison report against oracle.axioms.
    ``info``: the ontology as the OWL API reads it, whose classes (with owl:Thing
    when the file mentions it) and roles the paper metric is computed over.
    ``max_eqs``: stop after this many equivalence queries (one per iteration).
    ``precomputation``: start H with an MQ for every ordered pair of concept names
    (Phase 1); without it H starts empty and grows only from counterexamples.
    ``max_failed_eqs``: EQs in a row without a usable answer before learning stops.
    ``reuse_counterexamples``: process a counterexample again, without an EQ, while H
    still misses it after a round that extended H.  ``saturation``: ``"paper"`` or
    ``"hierarchy"`` (saturation also uses H's subclass hierarchy; learn_el_terminology)."""
    # With PAC sampling the sample budget bounds the run; every counterexample is
    # one iteration, so a cap of 50 would cut a yes-biased LLM's run short.
    max_iterations = 100_000 if getattr(oracle, "sampler", None) is not None else 50
    if max_eqs is not None:
        max_iterations = max_eqs
    if not verbose:
        logging.disable(logging.INFO)
    start = time.perf_counter()
    try:
        H = learn_el_terminology(oracle, max_iterations=max_iterations, precomputation=precomputation,
                                 max_failed_eqs=max_failed_eqs,
                                 reuse_counterexamples=reuse_counterexamples, saturation=saturation,
                                 repair=repair, repair_majority=repair_majority)
    finally:
        logging.disable(logging.NOTSET)
    # Before scoring, which may ask the oracle again.
    queries = query_report(oracle, time.perf_counter() - start)

    mq_calls = oracle.mq_count
    eq_calls = oracle.eq_count
    # A reasoning model cut off before </think> (LLMOracle.truncated): read as no answer.
    # PAC: the inner LLM oracle's count (the wrapper has none).
    truncated = getattr(getattr(oracle, "inner", oracle), "truncated", 0)

    target_O  = oracle.axioms

    H_MQ = oracle.make_H_MQ(H)
    O_MQ = judge or oracle.MQ
    extra   = [g for g in H if not O_MQ(g)]   # soundness:    O ⊨ H
    missing = [g for g in target_O if not H_MQ(g)] # completeness: H ⊨ O
    # Not an axiom of O, but O entails it.  Only these are derivable: an LLM oracle
    # can also put GCIs in H that O does not entail (the extra ones).
    redundant = [g for g in H if g not in target_O and g not in extra]

    repairs = getattr(oracle, "repair_log", None)
    if repairs:
        _section("Repairs")
        for ev in repairs:
            print(f"\n  EQ {ev['eq']}: {ev['proposal']}  ({ev['outcome']})")
            for c in ev["culprits"]:
                print(f"      follows from {c}")
            for r in ev["refuted"]:
                print(f"      refuted      {r}")
            for w in ev["weakened"]:
                print(f"      weakened     {w}")
            for n in ev.get("still_entailed", []):
                print(f"      still entailed {n}")
            for o in ev.get("second_opinion", []):
                print(f"      asked again  {o}")
        refuted = [str(n) for n in getattr(oracle, "refuted", [])]
        print(f"\n  REPAIRS {json.dumps({'repairs': len(repairs), 'refuted': refuted}, ensure_ascii=False)}")

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
    for line in queries:
        print("  " + line)

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
    prompt_style: str = "paper-advanced",
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
    reuse_counterexamples: bool = False,
    saturation: str = "paper",
    repair: bool = False,
    thinking: bool = True,
    repair_temperature: float = 0.0,
    repair_majority: bool = False,
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
    if reuse_counterexamples:
        print("  Reuse    : a counterexample H still misses is processed again before the next EQ")
    if saturation != "paper":
        print("  Saturate : names tried general first by H's hierarchy; candidates H rules out are not asked")
    if repair:
        print("  Repair   : proposals that follow from H are used to find and weaken wrong GCIs"
              + (f"; its questions sampled afresh at temperature {repair_temperature} (never from the cache)"
                 if repair_temperature > 0 else "")
              + ("; a denied part is asked again, refuted by a majority of 3" if repair_majority else ""))
    if not thinking:
        print("  Thinking : off (enable_thinking=False in the chat template)")
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
                thinking=thinking,
                repair_temperature=repair_temperature,
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
                                max_failed_eqs=max_failed_eqs,
                                reuse_counterexamples=reuse_counterexamples,
                                saturation=saturation, repair=repair, repair_majority=repair_majority)
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


def _count(text: str) -> int:
    """A whole number ≥ 1 (a token budget): 0 would only fail at the first query, after the model loaded."""
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError(f"expected a whole number ≥ 1, not {text}")
    return value


def _fraction(text: str, below: float | None = None) -> float:
    """A number > 0 (and < *below*): PAC's ε and δ (0 divides by zero, δ ≥ 1 has no log)."""
    value = float(text)
    if not value > 0 or (below is not None and value >= below):
        raise argparse.ArgumentTypeError(
            f"expected a number > 0{f' and < {below:g}' if below is not None else ''}, not {text}")
    return value


def build_parser() -> argparse.ArgumentParser:
    """demo.py's command line (learner/run_config.py's catalogue mirrors its LLM options)."""
    parser = argparse.ArgumentParser(
        description="EL learner demo.  Choose an oracle backend with --oracle.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--oracle", type=_oracle_kind, choices=["reasoner", "axioms", "signature"], default="reasoner",
        help="Oracle backend: 'reasoner' (Java/ELK); 'axioms' (LLM with the ontology's axioms in its prompt); "
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
             "In the language --eq-language says.",
    )
    parser.add_argument(
        "--eq-language", choices=["style", "manchester", "nl"], default="style",
        help="With an LLM's own equivalence queries: the language of the EQ prompts and "
             "answers.  'style' (default): natural language with --prompt paper-nl-*, else "
             "Manchester syntax; 'nl': the paper's 'C is a subcategory of D' whatever the "
             "MQ prompt.",
    )
    parser.add_argument(
        "--eq-check", choices=["nl"], default="nl",
        help="With an LLM's own equivalence queries: each counterexample is checked with the "
             "paper's MQ in natural language ('nl', the only form since 2026-11-26).",
    )
    parser.add_argument(
        "--eq-check-system", choices=["s1"], default="s1",
        help="With an LLM's own equivalence queries: the system prompt of the counterexample "
             "check, 's1', the paper's simple one (the variants before it were removed on "
             "2026-11-26; learner/llm_prompts.py CHECK_SYSTEM_PROMPTS).",
    )
    parser.add_argument(
        "--eq-check-question", choices=["paper"], default="paper",
        help="With an LLM's own equivalence queries: the question of the counterexample "
             "check, the paper's MQ ('every' and 'atonce' were removed on 2026-11-26).",
    )
    parser.add_argument(
        "--eq-check-domain", choices=["off", "names"], default="off",
        help="With an LLM's own equivalence queries and a paper prompt: what the check's "
             "system prompt says about the domain.  'off': nothing; 'names': a note listing "
             "the run's concept names and roles, which come from one domain, to be read in "
             "that domain's sense (learner/llm_prompts.py domain_note).",
    )
    parser.add_argument(
        "--eq-memory", action=argparse.BooleanOptionalAction, default=False,
        help="With an LLM's own equivalence queries: list the counterexamples the check "
             "rejected at earlier EQs in every later counterexample prompt "
             "(--no-eq-memory turns a config's memory off).",
    )
    parser.add_argument(
        "--eq-explain", action=argparse.BooleanOptionalAction, default=False,
        help="With an LLM's own equivalence queries: each retry line also says what to do "
             "instead of the failed answer (no negation, only the listed names, a statement "
             "not a concept, nothing H already implies).",
    )
    parser.add_argument(
        "--eq-explain-form", action=argparse.BooleanOptionalAction, default=False,
        help="With --eq-explain: also explain '¬', inverse roles and 'Nothing' (EL has no "
             "negation, no inverse roles, no disjointness).",
    )
    parser.add_argument(
        "--eq-mask", type=int, default=0, metavar="N",
        help="With an LLM's own equivalence queries: after the first accepted "
             "counterexample, a masked phase of N accepted counterexamples, whose prompts "
             "show an empty H and leave the concepts H is about out of scope (default 0: off).",
    )
    parser.add_argument(
        "--eq-none", action=argparse.BooleanOptionalAction, default=False,
        help="With an LLM's own equivalence queries (mode judge): the counterexample prompt "
             "allows the one-word answer 'none' when the model knows nothing true that H "
             "misses; it ends learning like the judge's yes. Not offered in the masked phase.",
    )
    parser.add_argument(
        "--eq-temperature", type=float, default=0.0, metavar="T",
        help="With an LLM's own equivalence queries: sample only the counterexample "
             "prompts at T (Ollama's other defaults, seeded by --seed); the MQs, the "
             "check and the judge keep --temperature.  0 (default): as the other queries.",
    )
    parser.add_argument(
        "--eq-max-new-tokens", type=_count, default=None, metavar="N",
        help="With an LLM's own equivalence queries: the generation budget of the "
             "counterexample prompts only; the MQs, the judge and the check keep "
             "--max-new-tokens.  Unset: --max-new-tokens, but at least 128.",
    )
    parser.add_argument(
        "--eq-generation", choices=["g2b"], default="g2b",
        help="With an LLM's own equivalence queries (judge mode): how the counterexample "
             "is asked for.  Only 'g2b': examples about cars (the variants tried in "
             "cex13-cex17 were removed; learner/llm_prompts.py GENERATIONS).",
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
        help="Tolerate N EQs in a row "
             "without a usable answer before learning stops (default 0: stop at the first); "
             "the next EQ asks again, with the earlier failures in its prompt.  Failed EQs "
             "count towards --max-eqs.",
    )
    parser.add_argument(
        "--repair", action=argparse.BooleanOptionalAction, default=False,
        help="Not in the paper: when a counterexample proposal already follows from H, ask about "
             "the parts of the H axioms it follows from, refute the parts the teacher denies, and "
             "weaken H; refuted statements are answered no from then on.  Needs an LLM's own "
             "EQs (--oracle axioms/signature, --eq oracle).",
    )
    parser.add_argument(
        "--saturation", choices=["paper", "hierarchy"], default="paper",
        help="How the learner saturates a right side.  'paper': every name at every node, in "
             "alphabetical order.  'hierarchy' (not in the paper): names general first by H's "
             "subclass hierarchy, and a candidate H shows to be at least as specific as a rejected "
             "one is not asked.  Same result with a sound H, fewer MQs (13%% in the simulation).",
    )
    parser.add_argument(
        "--reuse-counterexamples", action=argparse.BooleanOptionalAction, default=False,
        help="Not in the paper: keep what a counterexample taught.  A ⊑ A′ is added to H "
             "when decomposition on the right moves from A to a name A′ of the root (already "
             "confirmed by O), and after a round that extended H the same counterexample is "
             "processed again, without an EQ, as long as H does not entail it.  Reused rounds "
             "don't count towards --max-eqs.",
    )
    parser.add_argument(
        "--repair-majority", action=argparse.BooleanOptionalAction, default=False,
        help="With --repair: a part the model denies is asked again before it is refuted; a "
             "second no refutes it, after a yes a third answer decides. Needs --repair-temperature "
             "above 0 (fresh samples).",
    )
    parser.add_argument(
        "--repair-temperature", type=float, default=0.0,
        help="With --repair: 0 (default) asks the repair step's questions like every other MQ "
             "(a question asked before is answered from the cache). Above 0: they are sampled "
             "at this temperature, from a random stream of their own, and never read from the cache.",
    )
    parser.add_argument(
        "--no-thinking", action="store_true",
        help="With an LLM over an ontology (axioms, signature): pass enable_thinking=False "
             "to the chat template, which switches a hybrid model's reasoning off (Qwen3-8B, "
             "14B, 32B); other templates ignore it.",
    )
    parser.add_argument(
        "--thinking", dest="no_thinking", action="store_false",
        help="Thinking on after all, when a config (or an earlier flag) turns it off.",
    )
    parser.add_argument(
        "--no-precompute", action="store_true",
        help="Skip Phase 1, the MQ for "
             "every ordered pair of concept names, so H starts empty and grows only from "
             "counterexamples.",
    )
    parser.add_argument(
        "--precompute", dest="no_precompute", action="store_false",
        help="Run Phase 1 after all, when a config (or an earlier flag) skips it.",
    )
    parser.add_argument("--epsilon", type=_fraction, default=0.2, help="PAC accuracy ε (with --eq pac), > 0.")
    parser.add_argument("--delta", type=lambda t: _fraction(t, below=1.0), default=0.1,
                        help="PAC confidence δ (with --eq pac), > 0 and < 1.")

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
        "--model", default=None,
        help="HuggingFace model name or local path (needed with an LLM oracle).",
    )
    llm_grp.add_argument(
        "--device", default="cpu",
        help="Inference device: cpu, cuda, or mps.",
    )
    llm_grp.add_argument(
        "--max-new-tokens", type=_count, default=None, metavar="N",
        help="Generation budget per LLM query.  Unset: 2, upstream's maxTokens for its "
             "True/False prompts.  Reasoning models need room to finish thinking before their answer.",
    )
    llm_grp.add_argument(
        "--prompt", choices=["paper-simple", "paper-advanced", "paper-nl-simple", "paper-nl-advanced"],
        default="paper-advanced",
        help="MQ prompts: the original ExactLearner-LLM's True/False prompts (default "
             "paper-advanced, the standard), the axiom in Manchester syntax or, with "
             "paper-nl-*, in its natural-language form (axioms/signature only).",
    )
    llm_grp.add_argument(
        "--mq-asked-as", choices=["paper", "check"], default="paper",
        help="With a paper prompt: ask the learner's own MQs (Phase 1, refinement) as the "
             "paper does ('paper'), or exactly as the counterexample check is asked "
             "('check': --eq-check, --eq-check-system, --eq-check-question, "
             "--eq-check-domain), so that one "
             "statement gets one answer within a run.",
    )
    llm_grp.add_argument(
        "--temperature", type=float, default=0.0, metavar="T",
        help="0 (default): greedy decoding.  Above 0: sample at T with Ollama's other "
             "defaults, as the paper's runs were answered (axioms/signature only); "
             "--seed then also seeds the sampling.",
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
    args = parser.parse_args(argv) if not known.config else _parse_with_config(parser, known.config, argv)
    # The repair acts on counterexample proposals that already follow from H.  Only an
    # LLM's own EQs make them: ELK is never wrong and PAC sampling returns statements H
    # misses, so with either it would run and never do anything (code review 2026-11-26).
    if args.repair and (args.oracle == "reasoner" or args.eq == "pac"):
        parser.error("--repair needs an LLM's own equivalence queries (--oracle axioms or signature "
                     "with --eq oracle): ELK and PAC sampling never propose a statement H already implies")
    if args.repair_majority and not args.repair:
        parser.error("--repair-majority needs --repair")
    if args.repair_majority and args.oracle in ("axioms", "signature") and not args.repair_temperature > 0:
        parser.error("--repair-majority needs --repair-temperature above 0: at 0 the second question "
                     "replays the first answer from the cache")
    return args


def _parse_with_config(parser: argparse.ArgumentParser, path: str, argv: list[str]) -> argparse.Namespace:
    """The command line after --config's settings (the flags given win)."""
    from learner import run_config
    try:
        config = run_config.load(path)
        args = parser.parse_args(run_config.to_argv(config) + argv)
        run_config.check_run(config, args.ontology, args.seed, argv)
        prompts = run_config.value(config, "oracle.mq.prompt")
        if isinstance(prompts, list) and args.prompt not in prompts:
            raise run_config.ConfigError(f"oracle.mq.prompt: {args.prompt} is not in the config ({', '.join(prompts)})")
    except run_config.ConfigError as exc:
        parser.error(f"--config {path}: {exc}")
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
            reuse_counterexamples=args.reuse_counterexamples,
            saturation=args.saturation,
            max_eqs=args.max_eqs,
            precomputation=not args.no_precompute,
            max_failed_eqs=args.max_failed_eqs,
        )
    elif args.oracle in ("axioms", "signature"):
        if not args.model:
            parser.error("--model is needed with an LLM oracle (axioms, signature)")
        max_new_tokens = args.max_new_tokens
        if max_new_tokens is None:
            max_new_tokens = 2   # upstream's maxTokens for its True/False prompts
        from learner.llm_oracle import EQSettings   # loads transformers
        try:
            eq_settings = EQSettings(
                mode=args.eq_mode,
                language=args.eq_language,
                check=args.eq_check,
                check_system=args.eq_check_system,
                check_question=args.eq_check_question,
                check_domain=args.eq_check_domain,
                memory=args.eq_memory,
                explain=args.eq_explain,
                explain_form=args.eq_explain_form,
                mask=args.eq_mask,
                none=args.eq_none,
                temperature=args.eq_temperature,
                generation=args.eq_generation,
                attempts=args.eq_attempts,
                max_new_tokens=args.eq_max_new_tokens,
            )
        except ValueError as exc:   # a combination the settings refuse, e.g. combined + g2
            parser.error(str(exc))
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
            reuse_counterexamples=args.reuse_counterexamples,
            saturation=args.saturation,
            repair=args.repair,
            thinking=not args.no_thinking,
            repair_temperature=args.repair_temperature,
            repair_majority=args.repair_majority,
            mq_asked_as=args.mq_asked_as,
        )

    # Non-zero on failure, so scripts and batch jobs see it.
    sys.exit(0 if ok else 1)
