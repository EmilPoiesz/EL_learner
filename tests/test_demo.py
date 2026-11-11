"""
Tests for demo.py's run report.

1  learned H — an axiom of O is unmarked, a GCI O entails is marked *, one O does not entail ✗
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
