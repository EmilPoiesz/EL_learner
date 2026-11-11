"""
Run configurations: the settings of an LLM learning run as one JSON file instead
of a row of flags, and the catalogue of every option such a file can hold.

A configuration nests like the setup it describes:

    {
      "cluster": "olivia", "model": "Mistral-7B-Instruct-v0.3",
      "ontologies": ["animals", "university"], "seeds": [0, 1, 2, 3, 4],
      "learner": {"phase1": false, "max_eqs": 50, "max_failed_eqs": 5},
      "oracle": {
        "kind": "signature",
        "mq": {"prompt": "paper-advanced"},
        "eq": {"kind": "oracle", "mode": "judge",
               "counterexample": {"prompt": "g2b", "temperature": 1.0, "memory": true},
               "check": {"asked_as": "nl", "system": "s4"}}
      }
    }

Every key is optional except ``oracle.kind`` and ``ontologies``; a missing key
takes demo.py's default.  An unknown key, a value of the wrong type or a choice
that does not exist is an error, naming the key.  ``experiment``, ``description``
and ``based_on`` are free text for the record.

``demo.py --config FILE`` turns the run options into its own flags (``to_argv``),
before the flags given on the command line, which therefore win.  The job options
(cluster, model, ontologies, seeds) are read by the batch script, which starts one
demo.py run per ontology and seed: ``python -m learner.run_config FILE get seeds``.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from typing import Any

META_KEYS = ("experiment", "description", "based_on")


@dataclass(frozen=True)
class Option:
    """One setting.  ``flag``: the demo.py flag it becomes (None: read by the batch
    script).  ``choices``: (value, what it means) pairs; for a bool, the meaning of
    true and of false.  ``default``: demo.py's default (None: unset)."""

    key: str
    title: str
    kind: str                       # choice | bool | int | float | str | names | seeds
    help: str
    flag: str | None = None
    default: Any = None
    choices: tuple[tuple[Any, str], ...] = ()
    env: str | None = None          # the batch script's variable before configs
    nullable: bool = False
    required: bool = False
    invert: bool = False            # bool whose *false* sets the flag (phase1 -> --no-precompute)
    aliases: tuple[tuple[str, str], ...] = ()   # (old name, choice it now is)
    minimum: int = 0                # for an int
    multi: bool = False             # a choice that may also be a list: one run per value
    see: tuple[str, ...] = field(default=())   # experiments that tell its story


@dataclass(frozen=True)
class Group:
    key: str
    title: str
    help: str


GROUPS: tuple[Group, ...] = (
    Group("", "Run", "One job per ontology; in each, one learning run per seed."),
    Group("learner", "Learner", "Algorithm 1 (learn_el_terminology): where H starts and when learning stops."),
    Group("oracle", "Oracle", "The model that answers the queries, and how it is asked."),
    Group("oracle.mq", "Membership queries", "Does O entail this GCI? Asked during refinement, and in Phase 1."),
    Group("oracle.eq", "Equivalence queries", "Is H complete, and if not, which statement does it miss?"),
    Group("oracle.eq.counterexample", "Counterexample",
          "How the model is asked to name a statement that is true in O but does not follow from H."),
    Group("oracle.eq.check", "Check",
          "A counterexample is used only if it parses, uses names in scope, does not follow from H, "
          "and the model accepts it when asked again as a membership query."),
)

_PAPER_MQ = ("The paper's MQ: H-entailed questions are answered without the model, a conjunction "
             "on the right is asked one conjunct at a time, and both sides are desaturated against H.")

OPTIONS: tuple[Option, ...] = (
    # --- Job -------------------------------------------------------------------
    Option("cluster", "Cluster", "choice",
           "Where the jobs run. bf16 answers differ slightly between GPU types, so runs on the two "
           "clusters are not comparable answer for answer. The batch script refuses a config for the "
           "other cluster.",
           choices=(("fox", "Fox, one A100 80GB (account ec30)."),
                    ("olivia", "Olivia, one GH200 (account nn9851k), transformers pinned to Fox's 4.57.1.")),
           nullable=True, see=("cex4-olivia", "cex6")),
    Option("model", "Model", "str",
           "The model's folder under the cluster's model root (MODEL_ROOT in scripts/hpc.env), e.g. "
           "Mistral-7B-Instruct-v0.3 or Olmo-3.1-32B-Think. Unset: hpc.env's MODEL.",
           env="MODEL", nullable=True, see=("cex20",)),
    Option("ontologies", "Ontologies", "names",
           "Target ontologies by name (ontologies/<name>.owl): one job each. The result is scored "
           "against each one.", env="ONTOLOGY", required=True),
    Option("seeds", "Seeds", "seeds",
           "One learning run per seed. A seed changes a run only where something is sampled.",
           flag="--seed", default=0, env="SEEDS", see=("cex11",)),
    # --- Learner ---------------------------------------------------------------
    Option("learner.phase1", "Phase 1", "bool",
           "Whether learning starts with an MQ for every ordered pair of concept names.",
           flag="--no-precompute", default=True, invert=True, env="PRECOMPUTE",
           choices=((True, "First an MQ for every ordered pair of concept names (272 on animals, 42 on "
                           "university); the first EQ comes after. While no counterexample is accepted, the "
                           "score measures these MQs."),
                    (False, "H starts empty: every GCI comes from a counterexample and the refinement MQs "
                            "after it, so the score measures the equivalence oracle.")),
           see=("cex12",)),
    Option("learner.max_eqs", "EQs per run", "int",
           "Stop after this many equivalence queries, failed ones included. Unset: 50, or the PAC budget.",
           flag="--max-eqs", nullable=True, env="MAX_EQS", see=("cex12", "cex19")),
    Option("learner.max_failed_eqs", "Failed EQs tolerated", "int",
           "EQs in a row without a usable counterexample before learning stops. The next EQ asks again, "
           "with the reasons the earlier answers failed in its prompt. 0: the first failed EQ ends learning.",
           flag="--max-failed-eqs", default=0, env="MAX_FAILED_EQS", see=("cex19",)),
    # --- Oracle ----------------------------------------------------------------
    Option("oracle.kind", "Oracle", "choice",
           "What the model is given. Before 2026-11-11 these were called taught (axioms) and "
           "parametric (signature); the old names are still accepted.",
           flag="--oracle", default="reasoner", env="ORACLE", required=True,
           choices=(("axioms", "The ontology's axioms are in the system prompt; the model answers from them."),
                    ("signature", "Only the concept and role names; the model answers from its own knowledge, "
                                  "as in the paper. The ontology is only what the result is scored against.")),
           aliases=(("taught", "axioms"), ("parametric", "signature"))),
    Option("oracle.temperature", "Decoding", "float",
           "0: greedy. Above 0: sampled at this temperature with Ollama's other defaults (top-k 40, top-p "
           "0.9, repeat penalty 1.1), seeded by the seed. Applies to the MQs, the judge and the check; the "
           "counterexample prompts have their own.", flag="--temperature", default=0.0, env="TEMPERATURE"),
    Option("oracle.max_new_tokens", "Tokens per answer", "int",
           "The generation budget of every answer. Unset: 2 with a paper prompt, as the paper; EQ answers "
           "always get at least 128, and the counterexample prompts can have their own. A reasoning model needs room to finish thinking: an answer cut off "
           "before </think> counts as no answer.",
           flag="--max-new-tokens", nullable=True, env="MAX_NEW_TOKENS", see=("cex20",)),
    Option("oracle.mq.prompt", "Prompt", "choice",
           "How a membership query is asked. With a natural-language style the EQs are asked in natural "
           "language too. A list runs each prompt in turn (one run per prompt and seed).",
           flag="--prompt", default="ours", env="PROMPTS", multi=True,
           choices=(("ours", "This project's yes/no question naming the scope; the whole GCI at once."),
                    ("paper-simple", "The paper's simple system prompt, 'Answer with only True or False.', and "
                                     "the axiom as the OWL API writes it. " + _PAPER_MQ),
                    ("paper-advanced", "The paper's advanced system prompt, four rules including 'entities with "
                                       "has part relation are not in a subclass relation'. " + _PAPER_MQ),
                    ("paper-nl-simple", "paper-simple with the axiom in the paper's natural language: \"Can Cat "
                                        "be considered a subcategory of 'Animal'?\""),
                    ("paper-nl-advanced", "paper-advanced in the paper's natural language.")),
           see=("cex4",)),
    Option("oracle.mq.asked_as", "Asked as", "choice",
           "How the learner's own MQs (Phase 1 and refinement) are asked. With a paper prompt only.",
           flag="--mq-asked-as", default="paper",
           choices=(("paper", "The paper's MQ in the prompt style, as the reproduction asks it."),
                    ("check", "Exactly as the counterexample check is asked (its form, system prompt and "
                              "question), so the model is not asked one statement in two ways within a run. "
                              "In cex19, half the accepted counterexamples were denied by the next "
                              "refinement MQ under the paper's prompt.")),
           see=("cex19",)),
    Option("oracle.eq.kind", "Equivalence oracle", "choice",
           "Who answers the equivalence queries.", flag="--eq", default="oracle", env="EQ",
           choices=(("oracle", "The model: it judges whether H is complete and names a counterexample."),
                    ("pac", "The paper's PAC sampling: random statements, and the first one H misses and the "
                            "model's MQ accepts is the counterexample."))),
    Option("oracle.eq.epsilon", "PAC accuracy ε", "float",
           "With pac: the sample budget is round((h·ln|X| − ln δ)/ε).", flag="--epsilon", default=0.2),
    Option("oracle.eq.delta", "PAC confidence δ", "float",
           "With pac: see ε.", flag="--delta", default=0.1),
    Option("oracle.eq.mode", "Mode", "choice",
           "How an EQ is asked.", flag="--eq-mode", default="judge", env="EQ_MODES",
           choices=(("judge", "A yes/no question whether H is complete; if not, a second prompt for a "
                              "counterexample."),
                    ("combined", "One prompt: a counterexample, or 'yes' if H is complete. Mistral said 'yes' "
                                 "at once, even to an empty H.")),
           see=("cex1", "cex4")),
    Option("oracle.eq.language", "Language", "choice",
           "The language of both EQ prompts (H, the counterexample asked for, the feedback) and of the "
           "answer, which is parsed in it.", flag="--eq-language", default="style", env="EQ_LANGUAGE",
           choices=(("style", "As the MQ prompt style: natural language with paper-nl-*, else Manchester "
                              "syntax."),
                    ("manchester", "Manchester syntax: \"Bird SubClassOf has_part some Wing\"."),
                    ("nl", "The paper's natural language: \"Bird is a subcategory of something that "
                           "has_part some Wing\", whatever the MQ prompt style.")),
           see=("cex1-cex11", "cex23")),
    Option("oracle.eq.counterexample.prompt", "Prompt", "choice",
           "How the counterexample is asked for. Only with mode judge; g1 and g2 only in Manchester "
           "syntax.", flag="--eq-generation", default="default", env="EQ_GENERATION",
           choices=(("default", "Placeholder examples: C SubClassOf D, C SubClassOf r some D, "
                                "C and (r some D) SubClassOf E."),
                    ("g1", "The placeholders, then rules: names in scope only, a property always as "
                           "'r some D', 'and' only, no negation, the statement alone on its line."),
                    ("g2", "Examples about cars, '(about cars, not about O)'. Mistral wrapped every answer "
                           "in brackets."),
                    ("g3", "A triple, 'Subject | relation | Object', with is-a or a property: EL by "
                           "construction, but only A ⊑ B and A ⊑ ∃r.B. The model copied the format line."),
                    ("g2b", "g2 without any brackets in the prompt."),
                    ("g3b", "g3 with the format explained in prose, without 'Subject' and 'Object'.")),
           see=("cex13-cex15", "cex16-cex17", "cex18")),
    Option("oracle.eq.counterexample.temperature", "Sampling", "float",
           "0: as the other queries. Above 0: only the counterexample prompts are sampled at this "
           "temperature (Ollama's other defaults), seeded by the seed, so the seeds propose different "
           "counterexamples while the MQs, the judge and the check stay as they are.",
           flag="--eq-temperature", default=0.0, env="EQ_TEMPERATURE", see=("cex11",)),
    Option("oracle.eq.counterexample.memory", "Memory", "bool",
           "Whether rejected counterexamples are remembered across EQs.",
           flag="--eq-memory", default=False, env="EQ_MEMORY",
           choices=((True, "Counterexamples the check rejected are listed in every later EQ's prompt, so "
                           "they are not proposed again."),
                    (False, "Only the current EQ's failed answers are listed.")),
           see=("cex10",)),
    Option("oracle.eq.counterexample.attempts", "Attempts per EQ", "int",
           "Attempts at a usable counterexample per EQ; each retry lists why the earlier answers "
           "could not be used.", flag="--eq-attempts", default=3, env="EQ_ATTEMPTS", minimum=1),
    Option("oracle.eq.counterexample.max_new_tokens", "Tokens per counterexample", "int",
           "The generation budget of the counterexample prompts only; the MQs, the judge and the check "
           "keep oracle.max_new_tokens. Unset: that budget, but at least 128. A reasoning model has to "
           "think and then write a statement: in cex20, 29 of 38 attempts were cut off at 2048 tokens. "
           "A budget is a ceiling: an answer that ends sooner costs no more.",
           flag="--eq-max-new-tokens", nullable=True, env="EQ_MAX_NEW_TOKENS", minimum=1, see=("cex20",)),
    Option("oracle.eq.check.asked_as", "Asked as", "choice",
           "The form of the check's membership query. With a paper prompt only.",
           flag="--eq-check", default="mq", env="EQ_CHECK",
           choices=(("mq", "An MQ in the run's own prompt style."),
                    ("nl", "The paper's natural-language MQ, \"Can Bird be considered a subcategory of "
                           "'something that has_part some Wing'?\", whatever the style.")),
           see=("cex4",)),
    Option("oracle.eq.check.system", "System prompt", "choice",
           "The check's system prompt. With a paper prompt only; the MQs of learning keep the paper's.",
           flag="--eq-check-system", default="paper", env="EQ_CHECK_SYSTEM",
           choices=(("paper", "The MQs' own. Under paper-advanced its rule 2 made Mistral reject every "
                              "has_part statement."),
                    ("s1", "The paper's simple one: 'Answer with only True or False.'"),
                    ("s2", "The advanced one with rule 2 restated: a part is not a subcategory of its whole, "
                           "but having a part can hold for every member (examples about cars)."),
                    ("s3", "s2, and how EL statements read."),
                    ("s4", "s2, and: True only if it holds for every member, by definition, in the direction "
                           "written. Fewest false accepts in the check benchmark; Olmo took it literally and "
                           "rejected true statements.")),
           see=("cex6", "cex8", "check-benchmark-the-check-on-its-own", "cex18", "cex20")),
    Option("oracle.eq.check.question", "Question", "choice",
           "The check's question. With a paper prompt only.",
           flag="--eq-check-question", default="paper", env="EQ_CHECK_QUESTION",
           choices=(("paper", "The MQ as 'asked as' says."),
                    ("every", "\"Is it true that every C …?\" It rejects more of everything, true "
                              "existentials included.")),
           see=("check-benchmark-the-check-on-its-own",)),
)

BY_KEY: dict[str, Option] = {o.key: o for o in OPTIONS}


class ConfigError(ValueError):
    pass


def _check_value(opt: Option, value: Any) -> None:
    where = f"{opt.key}: "
    if value is None:
        if opt.nullable:
            return
        raise ConfigError(where + "may not be null")
    if opt.kind == "choice":
        allowed = [c for c, _ in opt.choices]
        values = value if opt.multi and isinstance(value, list) else [value]
        if not values:
            raise ConfigError(where + "expected at least one choice")
        for v in values:
            if v not in allowed:
                raise ConfigError(where + f"{v!r} is not one of {', '.join(map(str, allowed))}")
    elif opt.kind == "bool":
        if not isinstance(value, bool):
            raise ConfigError(where + f"expected true or false, not {value!r}")
    elif opt.kind == "int":
        if isinstance(value, bool) or not isinstance(value, int) or value < opt.minimum:
            raise ConfigError(where + f"expected a whole number ≥ {opt.minimum}, not {value!r}")
    elif opt.kind == "float":
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
            raise ConfigError(where + f"expected a number ≥ 0, not {value!r}")
    elif opt.kind == "str":
        if not isinstance(value, str) or not value:
            raise ConfigError(where + f"expected a name, not {value!r}")
    elif opt.kind == "names":
        if not isinstance(value, list) or not value or not all(isinstance(v, str) and v for v in value):
            raise ConfigError(where + f"expected a non-empty list of names, not {value!r}")
    elif opt.kind == "seeds":
        if (not isinstance(value, list) or not value
                or not all(isinstance(v, int) and not isinstance(v, bool) and v >= 0 for v in value)):
            raise ConfigError(where + f"expected a non-empty list of whole numbers, not {value!r}")


def flatten(config: dict, prefix: str = "") -> dict[str, Any]:
    """{dotted key: value} for every option set in *config*; ConfigError on an unknown key."""
    out: dict[str, Any] = {}
    groups = {g.key for g in GROUPS}
    for name, value in config.items():
        key = f"{prefix}{name}"
        if not prefix and name in META_KEYS:
            continue
        if key in BY_KEY:
            value = dict(BY_KEY[key].aliases).get(value, value) if isinstance(value, str) else value
            _check_value(BY_KEY[key], value)
            out[key] = value
        elif key in groups:
            if not isinstance(value, dict):
                raise ConfigError(f"{key}: expected an object of settings")
            out.update(flatten(value, key + "."))
        else:
            here = sorted({k[len(prefix):].split(".")[0] for k in [*BY_KEY, *groups]
                           if k.startswith(prefix) and k != prefix.rstrip(".")})
            raise ConfigError(f"{key}: unknown key; expected one of {', '.join(here)}")
    return out


def load(path: str) -> dict:
    """The configuration in *path*, checked; ConfigError says what is wrong and where."""
    try:
        with open(path, encoding="utf-8") as f:
            config = json.load(f)
    except json.JSONDecodeError as exc:
        raise ConfigError(f"{path}: not valid JSON ({exc})") from None
    except OSError as exc:
        raise ConfigError(f"{path}: cannot be read ({exc.strerror or exc})") from None
    if not isinstance(config, dict):
        raise ConfigError(f"{path}: expected a JSON object")
    flat = flatten(config)
    for opt in OPTIONS:
        if opt.required and opt.key not in flat:
            raise ConfigError(f"{opt.key}: required")
    return config


def value(config: dict, key: str) -> Any:
    """The setting of *key*: the config's, else the default."""
    return flatten(config).get(key, BY_KEY[key].default)


def ontology_path(name: str) -> str:
    return name if name.endswith((".owl", ".ttl")) else f"ontologies/{name}.owl"


def to_argv(config: dict) -> list[str]:
    """demo.py flags for the run options set in *config*.  A list of one ontology or one
    seed becomes --ontology / --seed; with several, the caller picks one on the command line."""
    flat = flatten(config)
    argv: list[str] = []
    for opt in OPTIONS:
        if opt.key not in flat or flat[opt.key] is None:
            continue
        v = flat[opt.key]
        if opt.multi and isinstance(v, list):
            if len(v) != 1:
                continue                 # several: the caller picks one (check_run)
            v = v[0]
        if opt.key == "ontologies":
            if len(v) == 1:
                argv += ["--ontology", ontology_path(v[0])]
        elif opt.key == "seeds":
            if len(v) == 1:
                argv += ["--seed", str(v[0])]
        elif opt.flag is None:
            continue                     # job options: cluster, model
        elif opt.kind == "bool":
            if v != opt.default and (v is False) == opt.invert:
                argv.append(opt.flag)
            elif v != opt.default:
                raise ConfigError(f"{opt.key}: {v!r} cannot be passed as {opt.flag}")
        else:
            argv += [opt.flag, str(v)]
    return argv


def check_run(config: dict, ontology: str, seed: int, argv: list[str]) -> None:
    """Refuse a run outside the config's ontologies or seeds, or one that leaves the choice
    among several of them to demo.py's default."""
    flat = flatten(config)
    given = lambda flag: any(a == flag or a.startswith(flag + "=") for a in argv)
    names = flat.get("ontologies", [])
    if len(names) > 1 and not given("--ontology"):
        raise ConfigError(f"ontologies: the config names {', '.join(names)}; pick one with --ontology")
    if names and ontology not in {ontology_path(n) for n in names} | set(names):
        raise ConfigError(f"ontologies: {ontology} is not in the config ({', '.join(names)})")
    for opt in OPTIONS:
        v = flat.get(opt.key)
        if opt.multi and isinstance(v, list) and len(v) > 1 and not given(opt.flag):
            raise ConfigError(f"{opt.key}: the config names {', '.join(map(str, v))}; pick one with {opt.flag}")
    seeds = flat.get("seeds")
    if seeds and len(seeds) > 1 and not given("--seed"):
        raise ConfigError(f"seeds: the config names {seeds}; pick one with --seed")
    if seeds and seed not in seeds:
        raise ConfigError(f"seeds: {seed} is not in the config ({seeds})")


def main(argv: list[str]) -> int:
    """``python -m learner.run_config FILE [get KEY | check]``: for the batch script."""
    if not argv:
        print(__doc__)
        return 2
    try:
        config = load(argv[0])
        if argv[1:2] == ["get"]:
            v = value(config, argv[2])
            print(" ".join(map(str, v)) if isinstance(v, list) else ("" if v is None else v))
        elif argv[1:] in ([], ["check"]):
            for key, v in flatten(config).items():
                print(f"{key} = {json.dumps(v)}")
        else:
            print(f"unknown command {' '.join(argv[1:])}", file=sys.stderr)
            return 2
    except (ConfigError, OSError, KeyError) as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
