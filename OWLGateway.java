import org.semanticweb.owlapi.apibinding.OWLManager;
import org.semanticweb.owlapi.model.*;
import org.semanticweb.owlapi.reasoner.OWLReasoner;
import org.semanticweb.HermiT.Reasoner;
import py4j.GatewayServer;
import java.util.*;

public class OWLGateway {

    private final OWLOntologyManager manager;
    private OWLOntology ontology;
    private final OWLDataFactory factory;
    private OWLReasoner reasoner;
    private boolean dirty = true;
    private final String baseIRI = "http://el-learner.local#";

    public OWLGateway() throws OWLOntologyCreationException {
        manager  = OWLManager.createOWLOntologyManager();
        factory  = manager.getOWLDataFactory();
        ontology = manager.createOntology(IRI.create("http://el-learner.local/"));
    }

    // ---------------------------------------------------------------
    // Parse a concept expression string into an OWLClassExpression.
    //
    // Grammar (after stripping outer parens):
    //   TOP            → owl:Thing
    //   AND:(e),(e),...→ intersection
    //   SOME:r:(e)     → existential restriction
    //   <name>         → named class
    // ---------------------------------------------------------------
    public OWLClassExpression parse(String expr) {
        expr = expr.trim();
        // Strip outer parens
        while (expr.startsWith("(") && expr.endsWith(")") && matchingClose(expr, 0) == expr.length()-1) {
            expr = expr.substring(1, expr.length()-1).trim();
        }
        if (expr.equals("TOP")) return factory.getOWLThing();
        if (expr.startsWith("AND:")) {
            List<String> parts = splitTopLevel(expr.substring(4));
            Set<OWLClassExpression> conjuncts = new HashSet<>();
            for (String p : parts) conjuncts.add(parse(p));
            return factory.getOWLObjectIntersectionOf(conjuncts);
        }
        if (expr.startsWith("SOME:")) {
            // SOME:roleName:(filler)
            int colonAfterRole = expr.indexOf(':', 5);
            String role   = expr.substring(5, colonAfterRole);
            String filler = expr.substring(colonAfterRole + 1);
            OWLObjectProperty prop = factory.getOWLObjectProperty(IRI.create(baseIRI + role));
            return factory.getOWLObjectSomeValuesFrom(prop, parse(filler));
        }
        return factory.getOWLClass(IRI.create(baseIRI + expr));
    }

    /** Find the index of the closing paren matching the open paren at pos. */
    private int matchingClose(String s, int pos) {
        int depth = 0;
        for (int i = pos; i < s.length(); i++) {
            if (s.charAt(i) == '(') depth++;
            else if (s.charAt(i) == ')') { depth--; if (depth == 0) return i; }
        }
        return -1;
    }

    /** Split comma-separated list at top level (not inside parens). */
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
    // Add GCI and query entailment
    // ---------------------------------------------------------------

    public void add_gci(String lhsExpr, String rhsExpr) throws OWLOntologyChangeException {
        OWLSubClassOfAxiom axiom = factory.getOWLSubClassOfAxiom(parse(lhsExpr), parse(rhsExpr));
        manager.addAxiom(ontology, axiom);
        dirty = true;
    }

    public void clear() throws OWLOntologyChangeException {
        manager.removeAxioms(ontology, ontology.getAxioms());
        if (reasoner != null) { reasoner.dispose(); reasoner = null; }
        dirty = true;
    }

    public boolean entails(String lhsExpr, String rhsExpr) {
        if (dirty) {
            if (reasoner != null) reasoner.dispose();
            // Reasoner is HermiT
            reasoner = new Reasoner.ReasonerFactory().createReasoner(ontology);
            dirty = false;
        }
        return reasoner.isEntailed(
            factory.getOWLSubClassOfAxiom(parse(lhsExpr), parse(rhsExpr))
        );
    }

    public static void main(String[] args) throws Exception {
        OWLGateway gateway = new OWLGateway();
        int port = args.length > 0 ? Integer.parseInt(args[0]) : 25333;
        GatewayServer server = new GatewayServer(gateway, port);
        server.start();
        System.out.println("OWLGateway started on port " + port);
        System.out.flush();
    }
}
