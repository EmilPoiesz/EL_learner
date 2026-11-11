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
10 paper MQ text is the OWL API 5's Manchester rendering, as upstream sends it
11 natural-language styles: upstream's addExtraSemantic rewrite, quirks included
12 the five ontologies' class namespaces sort before owl:Thing's, as upstream_manchester assumes
13 desaturation: names H makes redundant in a node are dropped from both sides before asking
"""

from __future__ import annotations

import math

import pytest

from learner.cache.cache import LLMCache
from learner.el_algorithm import ELConcept, GCI
from learner.llm_oracle import LLMOracle
from learner.upstream import (PAPER_SYSTEM_PROMPTS, desaturate, paper_is_true, upstream_gci_text,
                              upstream_natural_language)
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


def _paper_oracle(tmp_path, responses, h_entailed: set[GCI] = frozenset(), style="paper-advanced",
                  h_reasoner=None) -> LLMOracle:
    oracle = LLMOracle.__new__(LLMOracle)
    oracle.mq_count = oracle.eq_count = 0
    oracle._pipe = _StubPipe(responses)
    oracle._model_name = "stub"
    oracle._max_new_tokens = 2
    oracle._sampling = None
    oracle._pad_token_id = 0
    oracle._verbose = False
    oracle._prompt_style = style
    oracle._system_msg = PAPER_SYSTEM_PROMPTS[style]
    oracle._sig = set(CLASSES)
    oracle._roles = set(ROLES)
    oracle._h_reasoner = h_reasoner or (lambda gci: gci in h_entailed)
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
    assert oracle._pipe.prompts == ["A SubClassOf C", "A SubClassOf r some D"]


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


_E = lambda *atoms, **ex: ELConcept(set(atoms), {(r, f) for r, f in ex.items()})


@pytest.mark.parametrize("gci, text", [
    # The OWL API 5.1.20 renderer's output (render, then \n -> " ", then "  " -> " ").
    (GCI(_E("Module"), _E(employed_by=_E("Person"))), "Module SubClassOf employed_by some Person"),
    (GCI(_E("Module"), _E(employed_by=_E("Person", "Professor"))),
     "Module SubClassOf employed_by some (Person and Professor)"),
    (GCI(_E("Module"), _E(employed_by=_E(teaches=_E("Module", "Professor")))),
     "Module SubClassOf employed_by some (teaches some (Module and Professor))"),
    (GCI(_E("Module", employed_by=_E("Person")), _E("Professor")),
     "Module and (employed_by some Person) SubClassOf Professor"),
    (GCI(_E(), _E("Module")), "Thing SubClassOf Module"),
    (GCI(_E("B", "a", r=_E(), s=_E("A")), _E(r=_E("Z"))),
     "B and a and (r some Thing) and (s some A) SubClassOf r some Z"),
    # Thing is a named class whose IRI sorts after the target's: after Z here.
    (GCI(ELConcept({"A"}, {("r", _E("Z")), ("r", _E())}), _E(r=_E())),
     "A and (r some Z) and (r some Thing) SubClassOf r some Thing"),
])
def test_10_upstream_manchester(gci, text):
    assert upstream_gci_text(gci) == text


def test_11_natural_language(tmp_path):
    nl = upstream_natural_language
    assert nl("Cat SubClassOf Animal") == "Can Cat be considered a subcategory of 'Animal'?"
    assert nl("A and (r some B) SubClassOf C") == (
        "Can A that is also something that (r some B) be considered a subcategory of 'C'?")
    assert nl("A SubClassOf r some B") == "Can A be considered a subcategory of 'something that r some B'?"
    # Split at the first " and " before the brackets are looked at, as upstream does.
    assert nl("A SubClassOf r some (B and C)") == (
        "Can A be considered a subcategory of 'something that r some (B that is also C)'?")

    oracle = _paper_oracle(tmp_path, ["True"], h_entailed=set(), style="paper-nl-advanced")
    assert oracle.MQ(GCI(_E("A"), _E(r=_E("B")))) is True
    assert oracle._pipe.prompts == ["Can A be considered a subcategory of 'something that r some B'?"]


@pytest.mark.parametrize("name", ["animals", "cell", "football", "generations", "university"])
def test_12_namespaces_sort_before_owl(name):
    import pathlib
    import rdflib
    from rdflib.namespace import OWL, RDF
    g = rdflib.Graph().parse(pathlib.Path(__file__).parent.parent / "ontologies" / f"{name}.owl")
    iris = {str(c) for c in g.subjects(RDF.type, OWL.Class) if isinstance(c, rdflib.URIRef)} - {str(OWL.Thing)}
    assert iris and all(iri < str(OWL.Thing) for iri in iris)



def _told(inclusions: set[tuple[str, str]]):
    """H of atomic inclusions A ⊑ B as an H-reasoner: C ⊑ D when every name of D follows
    from a name of C and every ∃r.F of D has an ∃r.E in C with E ⊑ F (complete for such H)."""
    def up(a):
        seen, todo = {a}, [a]
        while todo:
            x = todo.pop()
            for s, b in inclusions:
                if s == x and b not in seen:
                    seen.add(b)
                    todo.append(b)
        return seen

    def sub(c: ELConcept, d: ELConcept) -> bool:
        implied = set().union(*(up(a) for a in c.atoms)) if c.atoms else set()
        return (d.atoms <= implied and
                all(any(r == s and sub(e, f) for s, e in c.existentials) for r, f in d.existentials))
    return lambda gci: sub(gci.lhs, gci.rhs)


@pytest.mark.parametrize("gci, asked", [
    # Dog ⊑ Mammal in H: Mammal is redundant beside Dog, in any node of either side.
    (GCI(_E("Dog", "Mammal"), _E("Cat")), ["Dog SubClassOf Cat"]),
    (GCI(_E("A"), _E(r=_E("Dog", "Mammal"))), ["A SubClassOf r some Dog"]),
    (GCI(_E(r=_E("Dog", "Mammal")), _E("A")), ["r some Dog SubClassOf A"]),
    (GCI(_E("A", s=_E(r=_E("Dog", "Mammal"))), _E("Cat")), ["A and (s some (r some Dog)) SubClassOf Cat"]),
    # A right-hand conjunct implied by another is dropped, so it is not asked.
    (GCI(_E("A"), _E("Dog", "Mammal")), ["A SubClassOf Dog"]),
    # Equivalent names: the first in sorted order goes, the second stays.
    (GCI(_E("B", "C"), _E("Cat")), ["C SubClassOf Cat"]),
    # Names H does not relate stay.
    (GCI(_E("A", "Cat"), _E("Dog")), ["A and Cat SubClassOf Dog"]),
])
def test_13_desaturate(tmp_path, gci, asked):
    h = _told({("Dog", "Mammal"), ("B", "C"), ("C", "B")})
    oracle = _paper_oracle(tmp_path, ["True"] * len(asked), h_reasoner=h)
    assert oracle.MQ(gci) is True
    assert oracle._pipe.prompts == asked


def test_13_desaturate_keeps_equivalence():
    h = _told({("Dog", "Mammal"), ("Mammal", "Animal")})
    concept = _E("Animal", "Dog", "Mammal", r=_E("Mammal", "Dog"))
    simplified = desaturate(concept, lambda c: h(GCI(c, concept)))
    assert simplified == _E("Dog", r=_E("Dog"))
    assert h(GCI(simplified, concept)) and h(GCI(concept, simplified))
