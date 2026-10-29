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


# Role axioms the parsed GCIs cannot hold.  The reasoners read them from the file
# (ELK uses ranges and role inclusions, for instance), so O may entail more than
# the parsed GCIs do.
_ROLE_AXIOM_PREDICATES = (RDFS.domain, RDFS.range, RDFS.subPropertyOf, OWL.inverseOf,
                          OWL.equivalentProperty, OWL.propertyChainAxiom)
_ROLE_CHARACTERISTICS = (OWL.FunctionalProperty, OWL.InverseFunctionalProperty, OWL.TransitiveProperty,
                         OWL.SymmetricProperty, OWL.AsymmetricProperty, OWL.ReflexiveProperty,
                         OWL.IrreflexiveProperty)


def extract_ontology(path: str, skipped: list[str] | None = None) -> tuple[set[str], set[GCI]]:
    """
    The signature and the EL GCIs of the ontology file at *path*.

    Axioms that are not EL GCIs (e.g. definitions with owl:hasValue nominals,
    role domains and ranges) are left out, with a warning; when *skipped* is a
    list, a short description of each is appended to it.  Raises ValueError when
    two IRIs share a local name, since names are identified by local name.
    """
    g = Graph()
    g.parse(path)
    _check_local_names(g, path)
    if skipped is None:
        skipped = []

    sig: set[str] = set()
    gcis: set[GCI] = set()

    # owl:subClassOf translates directly to a GCI
    for subj, _, obj in g.triples((None, RDFS.subClassOf, None)):
        try:
            lhs = parse_concept(g, subj, sig)
            rhs = parse_concept(g, obj, sig)
            gcis.add(GCI(lhs=lhs, rhs=rhs))
        except ValueError as exc:
            skipped.append(f"SubClassOf of {_describe(subj)}: {exc}")
            logger.warning("SubClassOf axiom outside EL, left out of the parsed GCIs "
                           "(the reasoners still read it from the file): %s", exc)

    # owl:equivalentClass expands to two GCIs in both directions
    for subj, _, obj in g.triples((None, OWL.equivalentClass, None)):
        try:
            a = parse_concept(g, subj, sig)
            b = parse_concept(g, obj, sig)
            gcis.add(GCI(lhs=a, rhs=b))
            gcis.add(GCI(lhs=b, rhs=a))
        except ValueError as exc:
            skipped.append(f"EquivalentClasses of {_describe(subj)}: {exc}")
            logger.warning("EquivalentClass axiom outside EL, left out of the parsed GCIs "
                           "(the reasoners still read it from the file): %s", exc)

    role_axioms = [f"{local_name(p)} of {_describe(s)}"
                   for p in _ROLE_AXIOM_PREDICATES for s in g.subjects(p, None)]
    role_axioms += [f"{local_name(t)} {_describe(s)}"
                    for t in _ROLE_CHARACTERISTICS for s in g.subjects(RDF.type, t)]
    if role_axioms:
        skipped.extend(sorted(role_axioms))
        logger.warning("%d role axiom(s) left out of the parsed GCIs (the reasoners still "
                       "read them from the file): %s", len(role_axioms), ", ".join(sorted(role_axioms)))

    return sig, gcis


def _describe(node) -> str:
    return local_name(node) if isinstance(node, URIRef) else "an anonymous class"


def _check_local_names(g: Graph, path: str) -> None:
    """
    Raise ValueError when two IRIs of *g* share a local name.  Concept and role
    names are identified by local name (here and in the Java gateway), so two
    such IRIs would silently become one name.  The W3C vocabularies are not names.
    """
    iris: dict[str, set[str]] = {}
    for triple in g:
        for node in triple:
            if isinstance(node, URIRef) and not str(node).startswith("http://www.w3.org/"):
                iris.setdefault(local_name(node), set()).add(str(node))
    clashes = {name: sorted(found) for name, found in iris.items() if len(found) > 1 and name}
    if clashes:
        raise ValueError(
            f"{path}: IRIs share a local name, which would merge them into one name: "
            + "; ".join(f"{name}: {', '.join(found)}" for name, found in sorted(clashes.items()))
        )


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
