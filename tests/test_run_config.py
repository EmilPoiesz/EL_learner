"""
Tests for learner/run_config.py and demo.py --config.

1  the catalogue matches demo.py's parser: every flag exists, with the same default and choices
2  the catalogue's choices are the code's own lists (prompt styles, EQ settings)
3  load: unknown keys, wrong types, unknown choices and missing required keys fail, naming the key
4  to_argv: set options become flags; bools by their sense; job options and defaults are left out
5  demo.py --config: the config's flags, then the command line's, which win
6  a run must be one of the config's ontologies and seeds, and must pick when there are several
7  the command line for the batch script: get KEY, and errors as exit status 1
8  the oracle kinds' old names (taught, parametric) still work, as the new ones
9  review fixes: a missing file and attempts 0 are config errors; --seed=N counts as picking
   a seed; --no-eq-memory and --precompute undo the config's memory and skipped Phase 1
10 a list of MQ prompts: one run per prompt, so demo.py must be given one of them
"""

from __future__ import annotations

import json

import pytest

import demo
from learner import run_config
from learner.run_config import BY_KEY, OPTIONS, ConfigError

STANDARD = {
    "experiment": "standard", "description": "the cex19 setup",
    "cluster": "olivia", "model": "Mistral-7B-Instruct-v0.3",
    "ontologies": ["animals", "university"], "seeds": [0, 1, 2, 3, 4],
    "learner": {"phase1": False, "max_eqs": 50, "max_failed_eqs": 5},
    "oracle": {
        "kind": "signature",
        "mq": {"prompt": "paper-advanced"},
        "eq": {"kind": "oracle", "mode": "judge",
               "counterexample": {"prompt": "g2b", "temperature": 1.0, "memory": True, "attempts": 3},
               "check": {"asked_as": "nl", "system": "s4", "question": "paper"}},
    },
}


def _write(tmp_path, config, name="run.json"):
    path = tmp_path / name
    path.write_text(json.dumps(config))
    return str(path)


def _actions():
    return {a.option_strings[0]: a for a in demo.build_parser()._actions if a.option_strings}


def test_1_catalogue_matches_parser():
    actions = _actions()
    for opt in OPTIONS:
        if opt.flag is None:
            continue
        action = actions.get(opt.flag)
        assert action is not None, f"{opt.key}: demo.py has no {opt.flag}"
        default = (not action.default) if opt.invert else action.default
        assert default == opt.default, f"{opt.key}: demo.py's default is {action.default!r}"
        if opt.kind == "choice":
            assert {c for c, _ in opt.choices} <= set(action.choices), opt.key


def test_2_choices_are_the_codes():
    from learner.llm_oracle import EQ_CHECKS, EQ_LANGUAGES, EQ_MODES, PROMPT_STYLES
    from learner.llm_prompts import CHECK_QUESTIONS, CHECK_SYSTEM_PROMPTS, GENERATIONS
    choices = lambda key: [c for c, _ in BY_KEY[key].choices]
    assert choices("oracle.mq.prompt") == list(PROMPT_STYLES)
    assert choices("oracle.eq.mode") == list(EQ_MODES)
    assert choices("oracle.eq.language") == list(EQ_LANGUAGES)
    assert choices("oracle.eq.check.asked_as") == list(EQ_CHECKS)
    assert choices("oracle.eq.check.system") == list(CHECK_SYSTEM_PROMPTS)
    assert choices("oracle.eq.check.question") == list(CHECK_QUESTIONS)
    assert choices("oracle.eq.counterexample.prompt") == list(GENERATIONS)
    for opt in OPTIONS:
        assert opt.title and opt.help, opt.key
        assert all(text for _, text in opt.choices), opt.key


@pytest.mark.parametrize("change, message", [
    ({"oracle": {"kind": "signature", "eq": {"chek": {}}}}, "oracle.eq.chek: unknown key; expected one of"),
    ({"oracle": {"kind": "signature", "eq": {"check": {"system": "s5"}}}}, "oracle.eq.check.system: 's5' is not one of"),
    ({"learner": {"phase1": "no"}}, "learner.phase1: expected true or false"),
    ({"learner": {"max_eqs": -1}}, "learner.max_eqs: expected a whole number"),
    ({"seeds": []}, "seeds: expected a non-empty list"),
    ({"oracle": "signature"}, "oracle: expected an object"),
    ({"ontologies": None}, "ontologies: may not be null"),
])
def test_3_load_errors_name_the_key(tmp_path, change, message):
    config = {**STANDARD, **change}
    with pytest.raises(ConfigError, match=message.replace("(", r"\(")):
        run_config.load(_write(tmp_path, config))


def test_3_required_and_valid(tmp_path):
    assert run_config.load(_write(tmp_path, STANDARD)) == STANDARD
    config = {k: v for k, v in STANDARD.items() if k != "ontologies"}
    with pytest.raises(ConfigError, match="ontologies: required"):
        run_config.load(_write(tmp_path, config))
    with pytest.raises(ConfigError, match="not valid JSON"):
        (tmp_path / "bad.json").write_text("{")
        run_config.load(str(tmp_path / "bad.json"))


def test_4_to_argv():
    argv = run_config.to_argv(STANDARD)
    pairs = dict(zip(argv[::2], argv[1::2]))
    assert "--no-precompute" in argv and "--eq-memory" in argv
    for flag in ("--model", "--ontology", "--seed", "--cluster"):
        assert flag not in argv                     # job options, or several to pick from
    expected = {"--max-eqs": "50", "--max-failed-eqs": "5", "--oracle": "signature",
                "--prompt": "paper-advanced", "--eq": "oracle", "--eq-mode": "judge",
                "--eq-generation": "g2b", "--eq-temperature": "1.0", "--eq-attempts": "3",
                "--eq-check": "nl", "--eq-check-system": "s4", "--eq-check-question": "paper"}
    flags = [a for a in argv if a.startswith("--")]
    assert set(flags) == set(expected) | {"--no-precompute", "--eq-memory"}
    rest = [a for a in argv if a not in ("--no-precompute", "--eq-memory")]
    assert dict(zip(rest[::2], rest[1::2])) == expected
    one = {**STANDARD, "ontologies": ["animals"], "seeds": [3],
           "learner": {"phase1": True}, "oracle": {"kind": "axioms", "max_new_tokens": None}}
    assert run_config.to_argv(one) == ["--ontology", "ontologies/animals.owl", "--seed", "3",
                                       "--oracle", "axioms"]


def test_5_demo_config_then_command_line(tmp_path):
    path = _write(tmp_path, STANDARD)
    parser = demo.build_parser()
    args = demo.parse_args(parser, ["--config", path, "--ontology", "ontologies/university.owl",
                                    "--seed", "2", "--eq-check-system", "s2"])
    assert (args.oracle, args.prompt, args.eq_generation, args.eq_temperature) == \
        ("signature", "paper-advanced", "g2b", 1.0)
    assert args.no_precompute and args.eq_memory and args.max_failed_eqs == 5
    assert args.eq_check_system == "s2"              # the command line wins
    assert (args.ontology, args.seed) == ("ontologies/university.owl", 2)
    plain = demo.parse_args(parser, ["--oracle", "axioms"])
    assert plain.prompt == "ours" and not plain.no_precompute


@pytest.mark.parametrize("argv, message", [
    (["--seed", "0"], "pick one with --ontology"),
    (["--ontology", "ontologies/animals.owl"], "pick one with --seed"),
    (["--ontology", "ontologies/cell.owl", "--seed", "0"], "not in the config"),
    (["--ontology", "ontologies/animals.owl", "--seed", "9"], "9 is not in the config"),
])
def test_6_run_within_the_config(tmp_path, capsys, argv, message):
    path = _write(tmp_path, STANDARD)
    with pytest.raises(SystemExit):
        demo.parse_args(demo.build_parser(), ["--config", path, *argv])
    assert message in capsys.readouterr().err


def test_7_command_line(tmp_path, capsys):
    path = _write(tmp_path, STANDARD)
    assert run_config.main([path, "get", "seeds"]) == 0
    assert run_config.main([path, "get", "model"]) == 0
    assert run_config.main([path, "get", "oracle.eq.mode"]) == 0
    assert run_config.main([path, "get", "oracle.temperature"]) == 0
    assert capsys.readouterr().out.split("\n")[:4] == ["0 1 2 3 4", "Mistral-7B-Instruct-v0.3", "judge", "0.0"]
    bad = _write(tmp_path, {**STANDARD, "seeds": "all"}, "bad.json")
    assert run_config.main([bad, "get", "seeds"]) == 1
    assert "seeds: expected" in capsys.readouterr().err


def test_8_old_oracle_names(tmp_path):
    old = {**STANDARD, "oracle": {**STANDARD["oracle"], "kind": "parametric"}}
    path = _write(tmp_path, old)
    assert run_config.value(run_config.load(path), "oracle.kind") == "signature"
    assert run_config.to_argv(old)[run_config.to_argv(old).index("--oracle") + 1] == "signature"
    parser = demo.build_parser()
    assert demo.parse_args(parser, ["--oracle", "taught"]).oracle == "axioms"
    assert demo.parse_args(parser, ["--oracle", "parametric"]).oracle == "signature"
    with pytest.raises(SystemExit):
        demo.parse_args(parser, ["--oracle", "grounded"])


def test_9_review_fixes(tmp_path, capsys):
    parser = demo.build_parser()
    with pytest.raises(SystemExit):
        demo.parse_args(parser, ["--config", str(tmp_path / "typo.json")])
    assert "typo.json: cannot be read" in capsys.readouterr().err

    zero = {**STANDARD, "oracle": {**STANDARD["oracle"], "eq": {"counterexample": {"attempts": 0}}}}
    with pytest.raises(ConfigError, match="attempts: expected a whole number ≥ 1"):
        run_config.load(_write(tmp_path, zero))

    path = _write(tmp_path, STANDARD)
    args = demo.parse_args(parser, ["--config", path, "--ontology=ontologies/animals.owl", "--seed=1"])
    assert (args.ontology, args.seed) == ("ontologies/animals.owl", 1)

    args = demo.parse_args(parser, ["--config", path, "--ontology", "ontologies/animals.owl", "--seed", "0",
                                    "--no-eq-memory", "--precompute"])
    assert not args.eq_memory and not args.no_precompute
    assert demo.parse_args(parser, ["--eq-memory", "--no-precompute"]).no_precompute


def test_10_prompt_list(tmp_path, capsys):
    many = {**STANDARD, "oracle": {**STANDARD["oracle"], "mq": {"prompt": ["paper-simple", "paper-nl-advanced"],
                                                                  "asked_as": "paper"}}}
    path = _write(tmp_path, many)
    assert "--prompt" not in run_config.to_argv(many)
    pick = ["--config", path, "--ontology", "ontologies/animals.owl", "--seed", "0"]
    parser = demo.build_parser()
    assert demo.parse_args(parser, pick + ["--prompt", "paper-nl-advanced"]).prompt == "paper-nl-advanced"
    for extra, message in (([], "pick one with --prompt"), (["--prompt", "paper-advanced"], "not in the config")):
        with pytest.raises(SystemExit):
            demo.parse_args(parser, pick + extra)
        assert message in capsys.readouterr().err
    one = {**many, "oracle": {**many["oracle"], "mq": {"prompt": ["paper-simple"], "asked_as": "check"}}}
    argv = run_config.to_argv(one)
    assert argv[argv.index("--prompt") + 1] == "paper-simple"
    assert argv[argv.index("--mq-asked-as") + 1] == "check"
    with pytest.raises(ConfigError, match="'paper-s' is not one of"):
        run_config.load(_write(tmp_path, {**many, "oracle": {**many["oracle"], "mq": {"prompt": ["paper-s"]}}}, "bad.json"))

