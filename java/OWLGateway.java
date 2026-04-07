import org.semanticweb.owlapi.apibinding.OWLManager;
import org.semanticweb.owlapi.model.*;
import org.semanticweb.owlapi.reasoner.*;
import org.semanticweb.HermiT.Reasoner;
import py4j.GatewayServer;

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

        return factory.getOWLClass(IRI.create(baseIRI + expr));
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

        OWLObjectProperty prop = factory.getOWLObjectProperty(IRI.create(baseIRI + role));
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

        boolean result = reasoner.getSuperClasses(lhs, false).containsEntity(proxy);

        manager.removeAxiom(ontology, proxyAxiom);
        reasoner.flush();

        return result;
    }

    // ---------------------------------------------------------------
    // Entry point
    // ---------------------------------------------------------------
    public static void main(String[] args) throws Exception {
        int port = args.length > 0 ? Integer.parseInt(args[0]) : 25333;
        String reasonerType = args.length > 1 ? args[1] : "hermit";

        OWLGateway gateway = new OWLGateway(reasonerType);
        GatewayServer server = new GatewayServer(gateway, port);

        server.start();
        int listeningPort = server.getListeningPort();

        System.out.println("READY:" + listeningPort);
        System.out.flush();
    }
}