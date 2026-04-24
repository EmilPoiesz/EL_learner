import logging

from rdflib import OWL, RDF, RDFS, Graph, URIRef, BNode
from rdflib.collection import Collection
from learner.el_algorithm import ELConcept, GCI

logger = logging.getLogger(__name__)


def extract_ontology(path: str) -> tuple[set[str], set[GCI]]:
    g = Graph()
    g.parse(path)

    sig: set[str] = set()
    gcis: set[GCI] = set()
    for subj, _, obj in g.triples((None, RDFS.subClassOf, None)):
        try:
            lhs = parse_concept(g, subj, sig)
            rhs = parse_concept(g, obj, sig)
            gcis.add(GCI(lhs=lhs, rhs=rhs))
        except ValueError as exc:
            logger.warning("Skipping unrecognised SubClassOf triple: %s", exc)

    return sig, gcis


def local_name(uri: URIRef) -> str:
    uri_str = str(uri)
    if "#" in uri_str:
        return uri_str.split("#")[-1]
    return uri_str.split("/")[-1]


def parse_concept(g: Graph, node, sig: set[str] | None = None) -> ELConcept:
    if isinstance(node, URIRef):
        if node == OWL.Thing:
            return ELConcept()
        name = local_name(node)
        if sig is not None:
            sig.add(name)
        return ELConcept(atoms=frozenset({name}))

    if not isinstance(node, BNode):
        raise ValueError(f"Unexpected node type: {type(node)} for {node}")

    if (node, RDF.type, OWL.Restriction) in g:
        role_node   = g.value(node, OWL.onProperty)
        filler_node = g.value(node, OWL.someValuesFrom)
        if role_node is None or filler_node is None:
            raise ValueError(f"Malformed owl:Restriction at {node}")
        role   = local_name(role_node)
        filler = parse_concept(g, filler_node, sig)
        return ELConcept(existentials=frozenset({(role, filler)}))

    list_head = g.value(node, OWL.intersectionOf)
    if list_head is not None:
        members = list(Collection(g, list_head))
        combined_atoms: set[str] = set()
        combined_existentials: set[tuple] = set()
        for member in members:
            part = parse_concept(g, member, sig)
            combined_atoms        |= part.atoms
            combined_existentials |= part.existentials
        return ELConcept(
            atoms=frozenset(combined_atoms),
            existentials=frozenset(combined_existentials),
        )

    raise ValueError(f"Unrecognised BNode structure at {node}")
