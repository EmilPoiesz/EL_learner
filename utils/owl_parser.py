import logging

from rdflib import OWL, RDF, RDFS, Graph, Namespace, URIRef, BNode
from rdflib.collection import Collection
from learner.el_algorithm import ELConcept, GCI


def _rdf_list(g: Graph, items: list) -> URIRef | BNode:
    """Build an RDF list using only rdf:first/rdf:rest, without rdf:type rdf:List.

    Collection() adds rdf:type rdf:List to each list-head BNode, which triggers
    a UserWarning in rdflib's pretty-xml serializer.  Building the list manually
    avoids that while still producing rdf:parseType="Collection" in the output.
    """
    if not items:
        return RDF.nil
    head = BNode()
    g.add((head, RDF.first, items[0]))
    g.add((head, RDF.rest, _rdf_list(g, items[1:])))
    return head

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
    g.add((intersection, OWL.intersectionOf, _rdf_list(g, parts)))
    return intersection


def save_ontology(
    gcis: set[GCI],
    path: str,
    namespace: str = "http://example.org/learned#",
) -> None:
    """Serialize a set of GCIs to an OWL file (.ttl for Turtle, .owl/.xml for OWL/XML)."""
    if path.endswith((".owl", ".xml", ".rdf")):
        _save_owl_xml(gcis, path, namespace)
    else:
        _save_turtle(gcis, path, namespace)
    logger.info("Saved %d GCI(s) to %s", len(gcis), path)


def _save_turtle(gcis: set[GCI], path: str, namespace: str) -> None:
    g = Graph()
    ns = Namespace(namespace)
    g.bind("ont", ns)
    for gci in gcis:
        lhs_node = _concept_to_rdf(g, gci.lhs, ns)
        rhs_node = _concept_to_rdf(g, gci.rhs, ns)
        g.add((lhs_node, RDFS.subClassOf, rhs_node))
    g.serialize(destination=path, format="turtle")


def _save_owl_xml(gcis: set[GCI], path: str, namespace: str) -> None:
    """
    Serialize GCIs to OWL/XML via owlready2, which produces clean output:
    short #Name URIs, rdf:parseType="Collection", no random BNode IDs.
    """
    import owlready2 as owl2

    base = namespace.rstrip("#")
    onto = owl2.get_ontology(base)

    concept_names: set[str] = set()
    role_names: set[str] = set()

    def _collect(c: ELConcept) -> None:
        concept_names.update(c.atoms)
        for r, f in c.existentials:
            role_names.add(r)
            _collect(f)

    for gci in gcis:
        _collect(gci.lhs)
        _collect(gci.rhs)

    with onto:
        classes = {n: owl2.types.new_class(n, (owl2.Thing,)) for n in sorted(concept_names)}
        for cls in classes.values():
            cls.is_a.remove(owl2.Thing)
        roles = {r: owl2.types.new_class(r, (owl2.ObjectProperty,)) for r in sorted(role_names)}

        def _expr(c: ELConcept):
            parts = [classes[a] for a in sorted(c.atoms)]
            for role, filler in sorted(c.existentials, key=lambda x: (x[0], str(x[1]))):
                parts.append(roles[role].some(_expr(filler)))
            if not parts:
                return owl2.Thing
            return parts[0] if len(parts) == 1 else owl2.And(parts)

        for gci in sorted(gcis, key=str):
            lhs_e = _expr(gci.lhs)
            rhs_e = _expr(gci.rhs)
            if isinstance(lhs_e, owl2.ThingClass):
                lhs_e.is_a.append(rhs_e)
            else:
                gca = owl2.GeneralClassAxiom(lhs_e)
                gca.is_a.append(rhs_e)

    onto.save(file=path, format="rdfxml")
