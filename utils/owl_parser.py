import itertools
import logging
import warnings
from typing import Callable

from rdflib import OWL, RDF, RDFS, Graph, Namespace, URIRef, BNode
from rdflib.collection import Collection
from learner.el_algorithm import ELConcept, GCI


def _rdf_list(g: Graph, items: list, new_bnode: Callable[[], BNode] = BNode) -> URIRef | BNode:
    """Build an RDF list using only rdf:first/rdf:rest, without rdf:type rdf:List.

    Collection() adds rdf:type rdf:List to each list node, an assertion rdflib's
    pretty-xml serializer drops when it writes the list as
    rdf:parseType="Collection".  With only rdf:first/rdf:rest nothing is dropped.
    """
    if not items:
        return RDF.nil
    head = new_bnode()
    g.add((head, RDF.first, items[0]))
    g.add((head, RDF.rest, _rdf_list(g, items[1:], new_bnode)))
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


def _concept_to_rdf(g: Graph, concept: ELConcept, ns: Namespace,
                    new_bnode: Callable[[], BNode] = BNode):
    if not concept.atoms and not concept.existentials:
        return OWL.Thing

    parts = []
    for atom in sorted(concept.atoms):
        parts.append(ns[atom])
    for role, filler in sorted(concept.existentials, key=lambda x: (x[0], str(x[1]))):
        restriction = new_bnode()
        g.add((restriction, RDF.type, OWL.Restriction))
        g.add((restriction, OWL.onProperty, ns[role]))
        filler_node = _concept_to_rdf(g, filler, ns, new_bnode)
        g.add((restriction, OWL.someValuesFrom, filler_node))
        parts.append(restriction)

    if len(parts) == 1:
        return parts[0]

    intersection = new_bnode()
    g.add((intersection, RDF.type, OWL.Class))
    g.add((intersection, OWL.intersectionOf, _rdf_list(g, parts, new_bnode)))
    return intersection


def save_ontology(
    gcis: set[GCI],
    path: str,
    namespace: str = "http://example.org/learned#",
) -> None:
    """
    Serialize a set of GCIs to an OWL file: RDF/XML for .owl/.xml/.rdf, else Turtle.

    Both formats hold the same graph (``_ontology_graph``): an ontology header,
    a declaration of every class and object property, and the GCIs, ⊤ on either
    side included.  Saving the same GCIs again gives the same file.
    """
    g = _ontology_graph(gcis, namespace)
    if path.endswith((".owl", ".xml", ".rdf")):
        with warnings.catch_warnings():
            # rdflib warns on every list it writes as a Collection that assertions
            # other than rdf:first/rdf:rest are dropped; _rdf_list makes none.
            warnings.filterwarnings("ignore", message="Assertions on .* other than RDF.first")
            # base: names are written as #Name relative to the ontology IRI.
            g.serialize(destination=path, format="pretty-xml", base=namespace.rstrip("#"))
    else:
        g.serialize(destination=path, format="turtle")
    logger.info("Saved %d GCI(s) to %s", len(gcis), path)


def _ontology_graph(gcis: set[GCI], namespace: str) -> Graph:
    """
    The RDF graph of an OWL ontology holding *gcis*: an owl:Ontology header, an
    owl:Class for every concept name and an owl:ObjectProperty for every role,
    then one rdfs:subClassOf per GCI.  Blank nodes are numbered b1, b2, … in
    the order of the sorted GCIs, so the output does not change between saves.
    """
    ns = Namespace(namespace)
    g = Graph()
    g.bind("ont", ns)
    g.add((URIRef(namespace.rstrip("#")), RDF.type, OWL.Ontology))

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
    for name in sorted(concept_names):
        g.add((ns[name], RDF.type, OWL.Class))
    for role in sorted(role_names):
        g.add((ns[role], RDF.type, OWL.ObjectProperty))

    counter = itertools.count(1)
    new_bnode = lambda: BNode(f"b{next(counter)}")
    for gci in sorted(gcis, key=str):
        lhs = _concept_to_rdf(g, gci.lhs, ns, new_bnode)
        rhs = _concept_to_rdf(g, gci.rhs, ns, new_bnode)
        g.add((lhs, RDFS.subClassOf, rhs))
    return g
