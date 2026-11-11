"""
Tests for demo.py's run report.

1  learned H — an axiom of O is unmarked, a GCI O entails is marked *, one O does not entail ✗
2  the run report's line of counterexample outcomes, in a fixed order, only the kinds that occur
3  the query report: MQs by step in the algorithm's order, model calls by step, a QUERIES json line
"""

from __future__ import annotations

from types import SimpleNamespace

import demo
from learner.el_algorithm import ELConcept, GCI

A, B, C = (ELConcept(frozenset({n})) for n in "ABC")


def test_1_learned_h_tags(monkeypatch, capsys):
    axiom, derivable, unsound = GCI(A, B), GCI(A, ELConcept(frozenset({"A", "B"}))), GCI(C, A)
    H = {axiom, derivable, unsound}
    monkeypatch.setattr(demo, "learn_el_terminology", lambda oracle, max_iterations, precomputation, max_failed_eqs: H)
    oracle = SimpleNamespace(
        mq_count=0, eq_count=0, axioms={axiom}, signature={"A", "B", "C"},
        make_H_MQ=lambda H: lambda gci: gci in H,
    )
    entailed = {axiom, derivable}
    demo._run_and_report(oracle, verbose=False, judge=lambda gci: gci in entailed)
    out = capsys.readouterr().out
    assert f"      {axiom}\n" in out
    assert f"    * {derivable}\n" in out
    assert f"    ✗ {unsound}\n" in out
    assert "Redundant axioms (entailed but not primitive): 1" in out
    assert "Unsound axioms (not entailed by O): 1" in out
    assert f"extra:   {unsound}" in out


def test_2_outcomes_line():
    from collections import Counter
    line = demo.format_outcomes(Counter({"false accepted": 1, "unusable": 30, "true accepted": 8}))
    assert line.startswith("Counterexample attempts: unusable 30 · true accepted 8 · false accepted 1")
    assert "not seen by the learner" in line


def test_3_query_report():
    import json
    stats = {("saturation", "MQ"): {"asked": 4, "cached": 1, "generated": 3, "prompt_tokens": 300,
                                    "new_tokens": 6, "seconds": 1.5},
             ("EQ", "EQ (judge)"): {"asked": 1, "cached": 0, "generated": 1, "prompt_tokens": 900,
                                    "new_tokens": 2, "seconds": 0.5}}
    inner = SimpleNamespace(mq_by_step={"saturation": 4, "Phase 1": 2}, query_stats=stats)
    lines = demo.query_report(SimpleNamespace(inner=inner), 12.3)    # PAC: the inner oracle's
    assert lines[0] == "Learning took 12 s.  MQs by step: Phase 1 2 · saturation 4"
    rows = [line.split()[0] for line in lines[3:-1]]
    assert rows == ["EQ", "saturation", "total"]                       # in the algorithm's order
    assert lines[-2].split()[1:] == ["5", "1", "4", "1200", "8", "2.0", "0.50"]
    record = json.loads(lines[-1][len("QUERIES "):])
    assert record["mq_by_step"] == {"saturation": 4, "Phase 1": 2}
    assert record["model_calls"][0] == {"step": "EQ", "query": "EQ (judge)", "asked": 1, "cached": 0,
                                        "generated": 1, "prompt_tokens": 900, "new_tokens": 2, "seconds": 0.5}
    assert demo.query_report(SimpleNamespace(), 0)[0] == "Learning took 0 s.  MQs by step: none"
