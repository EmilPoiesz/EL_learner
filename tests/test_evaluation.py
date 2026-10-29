"""
test_evaluation.py
==================
The paper's ontology score (learner/evaluation.py), owl:Thing as a concept name,
and ⊤-LHS entailment in the gateway.

1  all_statements: cn(cn−1) + C(cn,2)(cn−2) + 2·cn²·rn statements; top adds ⊤ as one more name
2  the OWL API's view of each file (OntologyInfo): owl:Thing, class and role counts, and
   upstream's ontology size h (an equivalence counts once)
3  statement and entailed counts as upstream's ResultAnalyzer computes them (OWL API + ELK
   on the raw file), generations' nominals included
4  ⊤ ⊑ C: entailed when H holds it (the gateway looks for the proxy in the top node)
5  generations with ELK's own EQ: learning goes on past the parsed GCIs, which lack the
   nominal definitions, until H entails every statement O does
"""

from __future__ import annotations

import os
from math import comb

import pytest

from learner.el_algorithm import ELConcept, GCI
from learner.evaluation import all_statements, score_ontology

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ONTOLOGIES = os.path.join(_ROOT, "ontologies")


def _count(cn: int, rn: int) -> int:
    return cn * (cn - 1) + comb(cn, 2) * (cn - 2) + 2 * cn * cn * rn


@pytest.mark.parametrize("top", [False, True])
def test_1_statement_count(top):
    statements = list(all_statements({"A", "B", "C", "D"}, {"r", "s"}, top))
    assert len(statements) == _count(4 + top, 2)
    assert len(set(statements)) == len(statements) - (comb(4, 1) * 3 if top else 0)  # ⊤ ⊓ A ⊑ C is A ⊑ C
    assert (GCI(ELConcept({"A"}), ELConcept()) in statements) == top


def _oracle(name, reasoner_name):
    from learner.reasoner_oracle import ReasonerOracle
    return ReasonerOracle(os.path.join(_ONTOLOGIES, f"{name}.owl"),
                          gateway_jar_dir=os.path.join(_ROOT, "java"), reasoner=reasoner_name)


# Upstream's figures, from the OWL API (UpstreamCount/UpstreamH against the raw files).
@pytest.mark.parametrize("name, top, classes, roles, size", [
    ("animals", False, 17, 4, 12), ("cell", False, 22, 0, 24), ("football", True, 9, 3, 9),
    ("generations", True, 19, 4, 18), ("university", False, 7, 3, 4),
])
def test_2_ontology_info(reasoner_name, name, top, classes, roles, size):
    with _oracle(name, reasoner_name) as O:
        info = O.info
    assert (info.top, len(info.classes), len(info.roles), info.size) == (top, classes, roles, size)


@pytest.mark.parametrize("name, statements, entailed", [
    ("animals", 4624, 272), ("cell", 5082, 828), ("football", 1050, 163),
    ("generations", 7000, 1159), ("university", 441, 34),
])
def test_3_upstream_counts(reasoner_name, name, statements, entailed):
    if reasoner_name == "hermit" and name == "generations":
        pytest.skip("HermiT also uses generations' inverse, symmetric and functional roles, "
                    "which ELK (and so upstream) ignores: 1171 entailed")
    with _oracle(name, reasoner_name) as O:
        info = O.info
        score = score_ontology(O._MQ, O._MQ, info.classes, info.roles, info.top)
    assert (score.total, score.tp) == (statements, entailed)


def test_4_top_lhs_entailment(reasoner_name):
    from learner.hypothesis_reasoner import HypothesisReasoner
    from utils.java_utils import build_classpath
    top, person = ELConcept(), ELConcept({"Person"})
    has_head = ELConcept(existentials={("has", ELConcept({"Head"}))})
    h = HypothesisReasoner(build_classpath(os.path.join(_ROOT, "java")), reasoner=reasoner_name)
    try:
        h.add(GCI(top, person))
        h.add(GCI(person, has_head))
        assert h.entails(GCI(top, person))
        assert h.entails(GCI(top, has_head))
        assert not h.entails(GCI(top, ELConcept({"Head"})))
    finally:
        h.close()


def test_5_eq_covers_axioms_outside_el(reasoner_name):
    from learner.el_algorithm import learn_el_terminology
    with _oracle("generations", reasoner_name) as O:
        assert O.skipped_axioms
        H = learn_el_terminology(O, max_iterations=200)
        info = O.info
        h = O.make_H_MQ(H)
        score = score_ontology(O._MQ, h, info.classes, info.roles, info.top)
    assert (score.fn, score.fp) == (0, 0)
