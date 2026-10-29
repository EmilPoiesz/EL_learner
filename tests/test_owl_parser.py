"""
test_owl_parser.py
==================
Saving a learned ontology (utils/owl_parser.save_ontology) in both formats.

1  round trip: every GCI shape, ⊤ on either side and nested existentials, read back unchanged
2  two saves in one process: the second file holds only the second H
3  saving the same H twice gives the same file
4  the file declares the ontology, every class and every object property
5  the OWL API loads the saved file and entails every saved GCI
6  extract_ontology reports the axioms it leaves out (generations: 13 definitions, 4 role axioms)
7  two IRIs with one local name: the parser and the gateway refuse the file
"""

from __future__ import annotations

import os

import pytest
from rdflib import OWL, RDF, Graph, Namespace, URIRef

from learner.el_algorithm import ELConcept, GCI
from utils.owl_parser import extract_ontology, save_ontology

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_NS = "http://example.org/learned#"
FORMATS = ["owl", "ttl"]


def C(*atoms, **existentials) -> ELConcept:
    return ELConcept(frozenset(atoms), frozenset(existentials.items()))


H = {
    GCI(C("Cat"), C("Animal")),                                   # A ⊑ B
    GCI(C("A", "B"), C("Animal")),                                # A ⊓ B ⊑ C
    GCI(C("Dog"), C(eats=C("Meat"))),                             # B ⊑ ∃r.A
    GCI(C(eats=C("Meat")), C("Carnivore")),                       # ∃r.A ⊑ B
    GCI(C("Cat"), C("Pet", has=C("Fur", on=C("Skin")))),          # nested existentials
    GCI(C(), C("Animal")),                                        # ⊤ ⊑ A
    GCI(C(), C(has=C("Part"))),                                   # ⊤ ⊑ ∃r.A
    GCI(C("Rock"), C()),                                          # A ⊑ ⊤
    GCI(C("Leg"), C(partOf=C())),                                 # A ⊑ ∃r.⊤
}


@pytest.mark.parametrize("ext", FORMATS)
def test_1_round_trip(tmp_path, ext):
    path = str(tmp_path / f"h.{ext}")
    save_ontology(H, path)
    _, back = extract_ontology(path)
    assert back == H


@pytest.mark.parametrize("ext", FORMATS)
def test_2_saves_do_not_accumulate(tmp_path, ext):
    first, second = {GCI(C("Cat"), C("Animal"))}, {GCI(C("Fish"), C("Animal"))}
    save_ontology(first, str(tmp_path / f"1.{ext}"))
    save_ontology(second, str(tmp_path / f"2.{ext}"))
    assert extract_ontology(str(tmp_path / f"2.{ext}"))[1] == second


@pytest.mark.parametrize("ext", FORMATS)
def test_3_same_h_same_file(tmp_path, ext):
    save_ontology(H, str(tmp_path / f"1.{ext}"))
    save_ontology(set(H), str(tmp_path / f"2.{ext}"))
    assert (tmp_path / f"1.{ext}").read_bytes() == (tmp_path / f"2.{ext}").read_bytes()


@pytest.mark.parametrize("ext", FORMATS)
def test_4_declarations(tmp_path, ext):
    path = str(tmp_path / f"h.{ext}")
    save_ontology(H, path)
    g = Graph()
    g.parse(path)
    ns = Namespace(_NS)
    assert (URIRef(_NS.rstrip("#")), RDF.type, OWL.Ontology) in g
    classes = {"Cat", "Animal", "A", "B", "Dog", "Meat", "Carnivore", "Pet", "Fur", "Skin", "Part", "Rock", "Leg"}
    assert {c for c in classes if (ns[c], RDF.type, OWL.Class) in g} == classes
    assert {r for r in ("eats", "has", "on", "partOf") if (ns[r], RDF.type, OWL.ObjectProperty) in g} \
        == {"eats", "has", "on", "partOf"}


@pytest.mark.parametrize("ext", FORMATS)
def test_5_owl_api_entails_saved_gcis(tmp_path, reasoner_name, ext):
    from learner.hypothesis_reasoner import HypothesisReasoner
    from utils.java_utils import build_classpath
    path = str(tmp_path / f"h.{ext}")
    save_ontology(H, path)
    h = HypothesisReasoner(build_classpath(os.path.join(_ROOT, "java")), reasoner=reasoner_name)
    try:
        info = h.load_ontology(path)
        assert info.roles == {"eats", "has", "on", "partOf"}
        assert [str(g) for g in H if not h.entails(g)] == []
    finally:
        h.close()


def test_6_skipped_axioms():
    skipped: list[str] = []
    _, gcis = extract_ontology(os.path.join(_ROOT, "ontologies", "generations.owl"), skipped)
    assert len(gcis) == 9
    assert sum(x.startswith("EquivalentClasses") for x in skipped) == 13
    assert sorted(x for x in skipped if not x.startswith("EquivalentClasses")) == [
        "FunctionalProperty hasSex", "SymmetricProperty hasSibling",
        "inverseOf of hasParent", "range of hasSex",
    ]
    for name in ("animals", "cell", "football", "university"):
        skipped = []
        extract_ontology(os.path.join(_ROOT, "ontologies", f"{name}.owl"), skipped)
        assert skipped == [], name


_CLASH = """\
@prefix a: <http://a.example.org/onto#> .
@prefix b: <http://b.example.org/onto#> .
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
a:Cat a owl:Class ; rdfs:subClassOf a:Animal .
b:Cat a owl:Class ; rdfs:subClassOf a:Animal .
a:Animal a owl:Class .
"""


def test_7_local_name_clash(tmp_path, reasoner_name):
    from py4j.protocol import Py4JJavaError
    from learner.hypothesis_reasoner import HypothesisReasoner
    from utils.java_utils import build_classpath
    path = tmp_path / "clash.ttl"
    path.write_text(_CLASH)
    with pytest.raises(ValueError, match="share a local name.*Cat"):
        extract_ontology(str(path))
    h = HypothesisReasoner(build_classpath(os.path.join(_ROOT, "java")), reasoner=reasoner_name)
    try:
        with pytest.raises(Py4JJavaError, match="share a local name"):
            h.load_ontology(str(path))
    finally:
        h.close()
