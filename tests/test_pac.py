"""
test_pac.py
===========
PAC-sampling equivalence queries (learner/pac.py) and the paper prompt style
of LLMOracle, without a reasoner or a model.

1  budget follows round((h·ln|X| − ln δ)/ε) with |X| = cn²(cn−1) + 2·cn²·rn
2  statements have one of the three shapes and use only the signature
3  the statement sequence depends only on the seed, not on how it is drawn
4  PACOracle.EQ returns the first sample with O-yes and H-no; None once the budget is spent
5  PACOracle counts learner MQs and sampling MQs separately; H-entailed samples are not asked
6  paper_is_true is upstream's isStrictlyTrue
7  paper MQ: H-entailed GCI answered without the model
8  paper MQ: right-hand conjunction asked per conjunct, skipping those H entails
9  top: ⊤ (owl:Thing) is one more concept name, in |X| and in the statements
"""

from __future__ import annotations

import math

import pytest

from learner.cache.cache import LLMCache
from learner.el_algorithm import ELConcept, GCI
from learner.llm_oracle import LLMOracle, PAPER_SYSTEM_PROMPTS, paper_is_true
from learner.pac import PACOracle, PACSampler, role_names

CLASSES = {"A", "B", "C", "D"}
ROLES = {"r", "s"}


def _shape(gci: GCI) -> int:
    lhs, rhs = gci.lhs, gci.rhs
    if not lhs.existentials and not rhs.existentials and len(rhs.atoms) == 1:
        return 1                                    # A ⊓ B ⊑ C
    if len(lhs.atoms) == 1 and not lhs.existentials and not rhs.atoms and len(rhs.existentials) == 1:
        return 2                                    # B ⊑ ∃r.A
    if not lhs.atoms and len(lhs.existentials) == 1 and len(rhs.atoms) == 1 and not rhs.existentials:
        return 3                                    # ∃r.A ⊑ B
    return 0


def test_1_budget():
    s = PACSampler(CLASSES, ROLES, epsilon=0.2, delta=0.1, hypothesis_size=5)
    assert s.instance_space_size == 4 * 4 * 3 + 2 * 4 * 4 * 2
    assert s.budget == round((5 * math.log(s.instance_space_size) - math.log(0.1)) / 0.2)


def test_2_shapes():
    s = PACSampler(CLASSES, ROLES, hypothesis_size=20)
    shapes = set()
    while (g := s.next_statement()) is not None:
        shape = _shape(g)
        assert shape, g
        shapes.add(shape)
        if shape == 1:
            assert not (g.rhs.atoms & g.lhs.atoms)  # C differs from A and B
        names = set(g.lhs.atoms) | set(g.rhs.atoms)
        for _, filler in g.lhs.existentials | g.rhs.existentials:
            names |= filler.atoms
        assert names <= CLASSES
        assert role_names([g]) <= ROLES
    assert shapes == {1, 2, 3}


def test_3_sequence_depends_only_on_seed():
    a = PACSampler(CLASSES, ROLES, hypothesis_size=3, seed=7)
    b = PACSampler(set(sorted(CLASSES, reverse=True)), ROLES, hypothesis_size=3, seed=7)
    assert [a.next_statement() for _ in range(a.budget)] == [b.next_statement() for _ in range(b.budget)]
    c = PACSampler(CLASSES, ROLES, hypothesis_size=3, seed=8)
    d = PACSampler(CLASSES, ROLES, hypothesis_size=3, seed=7)
    assert [c.next_statement() for _ in range(5)] != [d.next_statement() for _ in range(5)]


class _Inner:
    """A minimal oracle: O is a set of GCIs, entailment is membership."""

    def __init__(self, O: set[GCI]):
        self.signature = set(CLASSES)
        self.axioms = O
        self.mq_count = 0
        self.asked: list[GCI] = []

    def MQ(self, gci):
        self.mq_count += 1
        self.asked.append(gci)
        return gci in self.axioms

    def make_H_MQ(self, H):
        return lambda gci: gci in H

    def on_H_add(self, gci):
        pass


def _first_statements(n: int, seed: int = 0) -> list[GCI]:
    s = PACSampler(CLASSES, ROLES, hypothesis_size=10, seed=seed)
    return [s.next_statement() for _ in range(n)]


def test_4_eq_finds_counterexample_then_exhausts():
    first = _first_statements(5)
    inner = _Inner({first[3]})
    pac = PACOracle(inner, roles=ROLES, hypothesis_size=10)
    H: set[GCI] = set()
    pac.make_H_MQ(H)
    assert pac.EQ(H) == first[3]
    assert pac.sampler.provided == 4
    H.add(first[3])
    assert pac.EQ(H) is None                       # budget spent, nothing else in O
    assert pac.sampler.provided == pac.sampler.budget


def test_5_counts_and_h_skip():
    first = _first_statements(3)
    inner = _Inner(set())
    pac = PACOracle(inner, roles=ROLES, hypothesis_size=10)
    H = {first[0]}                                  # H entails the first sample
    pac.make_H_MQ(H)
    pac.MQ(first[1])                                # a learner MQ
    assert pac.EQ(H) is None
    assert first[0] not in inner.asked             # H-entailed sample never asked
    assert pac.mq_count == 1
    # Sampling is with replacement: every draw of first[0] is skipped, not just the first.
    repeats = _first_statements(pac.sampler.budget).count(first[0])
    assert pac.sample_mq_count == pac.sampler.budget - repeats
    assert inner.mq_count == pac.mq_count + pac.sample_mq_count


@pytest.mark.parametrize("response, expected", [
    ("True", True), ("true.", True), (" TRUE \\n", True),
    ("False", False), ("Yes", False), ("True, because", False), ("", False),
])
def test_6_paper_is_true(response, expected):
    assert paper_is_true(response) is expected


class _StubPipe:
    def __init__(self, responses):
        self.responses = list(responses)
        self.prompts: list[str] = []

    def __call__(self, messages, generation_config=None):
        self.prompts.append(messages[-1]["content"])
        return [{"generated_text": messages + [{"role": "assistant", "content": self.responses.pop(0)}]}]


def _paper_oracle(tmp_path, responses, h_entailed: set[GCI]) -> LLMOracle:
    oracle = LLMOracle.__new__(LLMOracle)
    oracle.mq_count = oracle.eq_count = 0
    oracle._pipe = _StubPipe(responses)
    oracle._model_name = "stub"
    oracle._max_new_tokens = 2
    oracle._pad_token_id = 0
    oracle._verbose = False
    oracle._prompt_style = "paper-advanced"
    oracle._system_msg = PAPER_SYSTEM_PROMPTS["paper-advanced"]
    oracle._sig = set(CLASSES)
    oracle._roles = set(ROLES)
    oracle._h_reasoner = lambda gci: gci in h_entailed
    oracle._cache = LLMCache(enabled=True, db_path=str(tmp_path / "cache.db"))
    return oracle


def test_7_paper_mq_h_shortcut(tmp_path):
    gci = GCI(ELConcept({"A"}), ELConcept({"B"}))
    oracle = _paper_oracle(tmp_path, [], h_entailed={gci})
    assert oracle.MQ(gci) is True
    assert oracle._pipe.prompts == []


def test_8_paper_mq_splits_conjunction(tmp_path):
    lhs = ELConcept({"A"})
    rhs = ELConcept({"B", "C"}, {("r", ELConcept({"D"}))})
    known = GCI(lhs, ELConcept({"B"}))              # H entails A ⊑ B: not asked
    oracle = _paper_oracle(tmp_path, ["True", "False"], h_entailed={known})
    assert oracle.MQ(GCI(lhs, rhs)) is False
    assert oracle._pipe.prompts == ["A SubClassOf C", "A SubClassOf (r some D)"]


def test_9_top_is_one_more_concept_name():
    s = PACSampler(CLASSES, ROLES, hypothesis_size=40, top=True)
    cn, rn = len(CLASSES) + 1, len(ROLES)
    assert s.instance_space_size == cn * cn * (cn - 1) + 2 * cn * cn * rn
    top = ELConcept()
    statements = [g for g in iter(s.next_statement, None)]
    concepts = {g.lhs for g in statements} | {g.rhs for g in statements}
    concepts |= {f for g in statements for _, f in g.lhs.existentials | g.rhs.existentials}
    assert top in concepts                                    # ⊤ ⊑ C, A ⊑ ⊤, ∃r.⊤, ...
    for g in statements:                                      # A ⊓ ⊤ is just A
        assert g.lhs.atoms <= CLASSES and g.rhs.atoms <= CLASSES
    # Without top the same seed never draws ⊤.
    plain = PACSampler(CLASSES, ROLES, hypothesis_size=40)
    assert all(top not in (g.lhs, g.rhs) for g in iter(plain.next_statement, None))
