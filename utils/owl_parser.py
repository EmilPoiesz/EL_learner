import logging

from rdflib import OWL, RDF, RDFS, Graph, Namespace, URIRef, BNode
from rdflib.collection import Collection
from learner.el_algorithm import ELConcept, GCI

logger = logging.getLogger(__name__)


def extract_ontology(path: str) -> tuple[set[str], set[GCI]]:
    g = Graph()
    g.parse(path)

    sig: set[str] = set()
    gcis: set[GCI] = set()

    # owl:subClassOf translates directly to a GCI
    for subj, _, obj in g.triples((None, RDFS.subClassOf, None)):
        try:
            lhs = parse_concept(g, subj, sig)
            rhs = parse_concept(g, obj, sig)
            gcis.add(GCI(lhs=lhs, rhs=rhs))
        except ValueError as exc:
            logger.warning("Skipping unrecognised SubClassOf triple: %s", exc)

    # owl:equivalentClass expands to two GCIs in both directions
    for subj, _, obj in g.triples((None, OWL.equivalentClass, None)):
        try:
            a = parse_concept(g, subj, sig)
            b = parse_concept(g, obj, sig)
            gcis.add(GCI(lhs=a, rhs=b))
            gcis.add(GCI(lhs=b, rhs=a))
        except ValueError as exc:
            logger.warning("Skipping unrecognised EquivalentClass triple: %s", exc)

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


def _concept_to_rdf(g: Graph, concept: ELConcept, ns: Namespace):
    if not concept.atoms and not concept.existentials:
        return OWL.Thing

    parts = []
    for atom in sorted(concept.atoms):
        parts.append(ns[atom])
    for role, filler in sorted(concept.existentials, key=lambda x: (x[0], str(x[1]))):
        restriction = BNode()
        g.add((restriction, RDF.type, OWL.Restriction))
        g.add((restriction, OWL.onProperty, ns[role]))
        filler_node = _concept_to_rdf(g, filler, ns)
        g.add((restriction, OWL.someValuesFrom, filler_node))
        parts.append(restriction)

    if len(parts) == 1:
        return parts[0]

    intersection = BNode()
    g.add((intersection, RDF.type, OWL.Class))
    list_head = BNode()
    Collection(g, list_head, parts)
    g.add((intersection, OWL.intersectionOf, list_head))
    return intersection


def save_ontology(
    gcis: set[GCI],
    path: str,
    namespace: str = "http://example.org/learned#",
) -> None:
    """Serialize a set of GCIs to an OWL file (.ttl or .owl/.xml)."""
    g = Graph()
    ns = Namespace(namespace)
    g.bind("ont", ns)

    for gci in gcis:
        lhs_node = _concept_to_rdf(g, gci.lhs, ns)
        rhs_node = _concept_to_rdf(g, gci.rhs, ns)
        g.add((lhs_node, RDFS.subClassOf, rhs_node))

    fmt = "xml" if path.endswith((".owl", ".xml", ".rdf")) else "turtle"
    g.serialize(destination=path, format=fmt)
    logger.info("Saved %d GCI(s) to %s", len(gcis), path)
