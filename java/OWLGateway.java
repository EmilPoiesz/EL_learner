import org.semanticweb.owlapi.apibinding.OWLManager;
import org.semanticweb.owlapi.model.*;
import org.semanticweb.owlapi.reasoner.*;
import org.semanticweb.HermiT.Reasoner;
import py4j.GatewayServer;

import java.io.File;
import java.util.*;

public class OWLGateway {

    private final OWLOntologyManager manager;
    private final OWLDataFactory factory;
    private OWLOntology ontology;

    private OWLReasoner reasoner;
    private final OWLReasonerFactory reasonerFactory;

    private final boolean useElk;
    private boolean dirty = true;

    private final String baseIRI = "http://el-learner.local#";
    private long proxyCounter = 0;

    // Local name -> IRI of the classes and roles of a file loaded with load_file,
    // so that the wire format's bare names resolve to the file's own IRIs.
    private final Map<String, IRI> classIRIs = new HashMap<>();
    private final Map<String, IRI> roleIRIs  = new HashMap<>();

    // ---------------------------------------------------------------
    // Constructor
    // ---------------------------------------------------------------
    public OWLGateway(String reasonerType) throws OWLOntologyCreationException {
        manager  = OWLManager.createOWLOntologyManager();
        factory  = manager.getOWLDataFactory();
        ontology = manager.createOntology(IRI.create("http://el-learner.local/"));

        if ("elk".equalsIgnoreCase(reasonerType)) {
            reasonerFactory = loadElkFactory();
            useElk = true;
        } else {
            reasonerFactory = new Reasoner.ReasonerFactory();
            useElk = false;
        }
    }

    private OWLReasonerFactory loadElkFactory() {
        try {
            Class<?> cls = Class.forName("org.semanticweb.elk.owlapi.ElkReasonerFactory");
            return (OWLReasonerFactory) cls.getDeclaredConstructor().newInstance();
        } catch (ClassNotFoundException e) {
            throw new RuntimeException("ELK reasoner not found on classpath.", e);
        } catch (ReflectiveOperationException e) {
            throw new RuntimeException("Failed to instantiate ElkReasonerFactory.", e);
        }
    }

    // ---------------------------------------------------------------
    // Parsing
    // ---------------------------------------------------------------
    public OWLClassExpression parse(String expr) {
        expr = stripOuterParens(expr.trim());

        if (expr.equals("TOP")) return factory.getOWLThing();

        if (expr.startsWith("AND:")) {
            return parseIntersection(expr.substring(4));
        }

        if (expr.startsWith("SOME:")) {
            return parseExistential(expr);
        }

        return factory.getOWLClass(classIRIs.getOrDefault(expr, IRI.create(baseIRI + expr)));
    }

    private String stripOuterParens(String expr) {
        while (expr.startsWith("(") && expr.endsWith(")")
                && matchingClose(expr, 0) == expr.length() - 1) {
            expr = expr.substring(1, expr.length() - 1).trim();
        }
        return expr;
    }

    private OWLClassExpression parseIntersection(String content) {
        List<String> parts = splitTopLevel(content);
        Set<OWLClassExpression> conjuncts = new HashSet<>();
        for (String p : parts) {
            conjuncts.add(parse(p));
        }
        return factory.getOWLObjectIntersectionOf(conjuncts);
    }

    private OWLClassExpression parseExistential(String expr) {
        int colon = expr.indexOf(':', 5);
        String role   = expr.substring(5, colon);
        String filler = expr.substring(colon + 1);

        OWLObjectProperty prop = factory.getOWLObjectProperty(roleIRIs.getOrDefault(role, IRI.create(baseIRI + role)));
        return factory.getOWLObjectSomeValuesFrom(prop, parse(filler));
    }

    private int matchingClose(String s, int pos) {
        int depth = 0;
        for (int i = pos; i < s.length(); i++) {
            if (s.charAt(i) == '(') depth++;
            else if (s.charAt(i) == ')') {
                depth--;
                if (depth == 0) return i;
            }
        }
        return -1;
    }

    private List<String> splitTopLevel(String s) {
        List<String> parts = new ArrayList<>();
        int depth = 0, start = 0;

        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);

            if (c == '(') depth++;
            else if (c == ')') depth--;
            else if (c == ',' && depth == 0) {
                parts.add(s.substring(start, i));
                start = i + 1;
            }
        }

        parts.add(s.substring(start));
        return parts;
    }

    // ---------------------------------------------------------------
    // Ontology management
    // ---------------------------------------------------------------
    public void add_gci(String lhsExpr, String rhsExpr) throws OWLOntologyChangeException {
        OWLAxiom ax = factory.getOWLSubClassOfAxiom(parse(lhsExpr), parse(rhsExpr));
        manager.addAxiom(ontology, ax);
        dirty = true;
    }

    /**
     * Replace the ontology with the one in the file at path, read by the OWL API as
     * the original ExactLearner-LLM reads it: every axiom is kept, and the reasoner
     * uses what it supports (ELK: nominals with one individual, ranges, ...).
     */
    public void load_file(String path) throws OWLOntologyCreationException {
        if (reasoner != null) {
            reasoner.dispose();
            reasoner = null;
        }
        manager.removeOntology(ontology);
        ontology = manager.loadOntologyFromOntologyDocument(new File(path));
        classIRIs.clear();
        roleIRIs.clear();
        for (OWLClass c : ontology.getClassesInSignature()) {
            if (!c.isOWLThing() && !c.isOWLNothing()) putName(classIRIs, c.getIRI(), path);
        }
        for (OWLObjectProperty r : ontology.getObjectPropertiesInSignature()) {
            putName(roleIRIs, r.getIRI(), path);
        }
        dirty = true;
    }

    /** Map iri's local name to it; two IRIs with one local name would become one name. */
    private static void putName(Map<String, IRI> names, IRI iri, String path) {
        IRI earlier = names.put(localName(iri), iri);
        if (earlier != null && !earlier.equals(iri)) {
            throw new IllegalArgumentException(path + ": IRIs share a local name, which would merge them into one name: "
                    + localName(iri) + ": " + earlier + ", " + iri);
        }
    }

    /** As utils/owl_parser.local_name: the part after '#', else after the last '/'. */
    private static String localName(IRI iri) {
        String s = iri.toString();
        int hash = s.lastIndexOf('#');
        return hash >= 0 ? s.substring(hash + 1) : s.substring(s.lastIndexOf('/') + 1);
    }

    /** The loaded file's class names (without owl:Thing and owl:Nothing), sorted. */
    public List<String> class_names() {
        return new ArrayList<>(new TreeSet<>(classIRIs.keySet()));
    }

    /** The loaded file's object property names, sorted. */
    public List<String> role_names() {
        return new ArrayList<>(new TreeSet<>(roleIRIs.keySet()));
    }

    /** Whether owl:Thing is in the loaded file's class signature (the file mentions it). */
    public boolean mentions_thing() {
        return ontology.getClassesInSignature().stream().anyMatch(OWLClass::isOWLThing);
    }

    /**
     * The original ExactLearner-LLM's ontology size, the h of its PAC budget
     * (OntologyManipulator.computeOntologySize): the SubClassOf, EquivalentClasses,
     * SubObjectPropertyOf, EquivalentObjectProperties, ObjectPropertyDomain and
     * DisjointClasses axioms, each counted once.
     */
    public int ontology_size() {
        Set<AxiomType<?>> counted = new HashSet<>(Arrays.asList(
                AxiomType.SUBCLASS_OF, AxiomType.EQUIVALENT_CLASSES, AxiomType.SUB_OBJECT_PROPERTY,
                AxiomType.EQUIVALENT_OBJECT_PROPERTIES, AxiomType.OBJECT_PROPERTY_DOMAIN,
                AxiomType.DISJOINT_CLASSES));
        int size = 0;
        for (OWLAxiom ax : ontology.getAxioms()) {
            if (counted.contains(ax.getAxiomType())) size++;
        }
        return size;
    }

    public void clear() throws OWLOntologyChangeException {
        manager.removeAxioms(ontology, ontology.getAxioms());

        if (reasoner != null) {
            reasoner.dispose();
            reasoner = null;
        }

        dirty = true;
    }

    // ---------------------------------------------------------------
    // Reasoning
    // ---------------------------------------------------------------
    public boolean entails(String lhsExpr, String rhsExpr) throws OWLOntologyChangeException {
        OWLClassExpression lhs = parse(lhsExpr);
        OWLClassExpression rhs = parse(rhsExpr);

        ensureReasoner();

        if (useElk) {
            return elkEntails(lhs, rhs);
        } else {
            return hermitEntails(lhs, rhs);
        }
    }

    private void ensureReasoner() {
        if (!dirty) return;

        if (reasoner != null) {
            reasoner.dispose();
        }

        reasoner = reasonerFactory.createReasoner(ontology);
        dirty = false;
    }

    private boolean hermitEntails(OWLClassExpression lhs, OWLClassExpression rhs) {
        OWLAxiom ax = factory.getOWLSubClassOfAxiom(lhs, rhs);
        return reasoner.isEntailed(ax);
    }

    private boolean elkEntails(OWLClassExpression lhs, OWLClassExpression rhs)
            throws OWLOntologyChangeException {

        OWLClass proxy = factory.getOWLClass(
                IRI.create(baseIRI + "proxy" + (proxyCounter++))
        );

        OWLAxiom proxyAxiom = factory.getOWLSubClassOfAxiom(rhs, proxy);

        manager.addAxiom(ontology, proxyAxiom);
        reasoner.flush();

        // ⊤ ⊑ rhs makes the proxy equivalent to ⊤, and getSuperClasses lists strict
        // superclasses only, so for a ⊤ LHS the proxy is looked for in the top node.
        boolean result = lhs.isOWLThing()
                ? reasoner.getTopClassNode().contains(proxy)
                : reasoner.getSuperClasses(lhs, false).containsEntity(proxy);

        manager.removeAxiom(ontology, proxyAxiom);
        reasoner.flush();

        return result;
    }

    // ---------------------------------------------------------------
    // Entry point
    // ---------------------------------------------------------------
    public static void main(String[] args) throws Exception {
        int port = args.length > 0 ? Integer.parseInt(args[0]) : 25333;
        String reasonerType = args.length > 1 ? args[1] : "elk";

        OWLGateway gateway = new OWLGateway(reasonerType);
        GatewayServer server = new GatewayServer(gateway, port);

        server.start();
        int listeningPort = server.getListeningPort();

        System.out.println("READY:" + listeningPort);
        System.out.flush();
    }
}