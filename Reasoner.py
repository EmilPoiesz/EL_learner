import logging
import random
import subprocess
import time
import owlready2
import os
import glob
import socket

from py4j.java_gateway import JavaGateway, GatewayParameters

from EL_algorithm import Oracle, ELConcept, GCI
from typing import Optional

from rdflib import OWL, RDF, RDFS, Graph, URIRef, BNode
from rdflib.collection import Collection

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# OWL API + HermiT/ELK reasoner bridge via py4j
# ---------------------------------------------------------------------------

def _encode(concept: ELConcept) -> str:
    """
    Encode an ELConcept as a string for the OWLGateway Java process.

    Grammar:
      TOP                  → owl:Thing
      A                    → named class A
      AND:(e1),(e2),...    → intersection
      SOME:r:(filler)      → existential restriction  ∃r.filler
    """
    if not concept.atoms and not concept.existentials:
        return "TOP"
    parts = list(concept.atoms)
    for role, filler in concept.existentials:
        parts.append(f"SOME:{role}:({_encode(filler)})")
    if len(parts) == 1:
        return parts[0]
    return "AND:" + ",".join(f"({p})" for p in parts)


def _find_free_port() -> int:
    """Bind to port 0 to let the OS pick an available port, then return it."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _wait_for_port(port: int, timeout: float = 10.0) -> None:
    """Block until *port* on localhost is accepting connections, or raise."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                return
        except OSError:
            time.sleep(0.1)
    raise RuntimeError(f"Port {port} did not open within {timeout:.0f} s.")


def _wait_for_gateway(proc: subprocess.Popen, port: int, timeout: float = 10.0) -> None:
    """Wait for the gateway port, and on failure raise with the Java stderr output."""
    try:
        _wait_for_port(port, timeout)
    except RuntimeError:
        proc.terminate()
        try:
            _, stderr_bytes = proc.communicate(timeout=3)
        except Exception:
            stderr_bytes = b""
        stderr_str = stderr_bytes.decode(errors="replace").strip()
        msg = f"OWLGateway Java process failed to start on port {port}."
        if stderr_str:
            msg += f"\nJava stderr:\n{stderr_str}"
        raise RuntimeError(msg) from None


class HypothesisReasoner:
    """
    A reasoner dedicated to the current hypothesis H.

    Maintains a running Java OWLGateway process whose ontology is kept in
    sync with H via add() calls.  Used by ReasonerOracle to provide a
    complete H-entailment check (make_H_MQ / on_H_add).
    """

    def __init__(self, classpath: str, port: int, reasoner: str = "elk"):
        self._port = port
        self._java_proc = subprocess.Popen(
            ["java", "-cp", classpath, "OWLGateway", str(port), reasoner],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        _wait_for_gateway(self._java_proc, port)
        
        self._gw  = JavaGateway(
            gateway_parameters=GatewayParameters(port=port, eager_load=True),
            callback_server_parameters=None,
        )
        self._owl = self._gw.entry_point

    def add(self, gci: "GCI") -> None:
        self._owl.add_gci(_encode(gci.lhs), _encode(gci.rhs))

    def entails(self, gci: "GCI") -> bool:
        return self._owl.entails(_encode(gci.lhs), _encode(gci.rhs))

    def __call__(self, gci: "GCI") -> bool:
        return self.entails(gci)

    def close(self) -> None:
        try:
            self._gw.close()
        except Exception:
            pass
        try:
            self._java_proc.terminate()
            self._java_proc.wait(timeout=5)
        except Exception:
            pass

    def __del__(self):
        self.close()


class ReasonerOracle(Oracle):
    """
    An Oracle backed by the OWL API and a configurable DL reasoner via a py4j gateway.

    The gateway is a small Java process (OWLGateway.java) that wraps
    OWL API + the chosen reasoner and exposes three methods over py4j:
      - add_gci(lhs_str, rhs_str)  — load an axiom
      - entails(lhs_str, rhs_str)  — subsumption query
      - clear()                    — reset the ontology

    Parameters
    ----------
    path : str
        Path to the OWL/Turtle ontology file.
    gateway_jar_dir : str, optional
        Directory containing OWLGateway.class.  Defaults to the directory
        of this source file, so compiling OWLGateway.java in the project
        directory is all that is needed.
    reasoner : str, optional
        Which DL reasoner to use: ``"hermit"`` (default) or ``"elk"``.
        ELK is an OWL 2 EL reasoner; it is faster but only supports the EL
        profile.  HermiT is a full OWL 2 DL reasoner.
        When using ELK the ``elk-owlapi-standalone-*.jar`` must be on the
        classpath (place it in the project directory or set ELK_JAR).
    """

    @staticmethod
    def _search_for_jar(pattern: str, roots: list[str]) -> str | None:
        """Return the first jar matching *pattern* found under any of *roots*, or None."""
        for root in roots:
            matches = glob.glob(os.path.join(root, "**", pattern), recursive=True)
            if matches:
                return matches[0]
        return None

    @staticmethod
    def _find_hermit_jar() -> str:
        jar = os.path.join(os.path.dirname(owlready2.__file__), "hermit", "HermiT.jar")
        if not os.path.exists(jar):
            raise FileNotFoundError(f"HermiT.jar not found at {jar}")
        return jar

    @staticmethod
    def _find_py4j_jar() -> str:
        roots = [
            "/usr/local/share/py4j",
            "/usr/share/py4j",
            "/opt/homebrew",
            os.path.expanduser("~/.local/share/py4j"),
        ]
        venv = os.environ.get("VIRTUAL_ENV")
        if venv:
            roots.append(venv)
        jar = ReasonerOracle._search_for_jar("py4j*.jar", roots)
        if jar is None:
            raise FileNotFoundError("py4j jar not found. Check your venv or set VIRTUAL_ENV.")
        return jar

    @staticmethod
    def _find_elk_jar(gateway_jar_dir: str) -> str:
        """
        Locate the ELK standalone jar.
        """
        env_jar = os.environ.get("ELK_JAR")
        if env_jar:
            if not os.path.exists(env_jar):
                raise FileNotFoundError(f"ELK_JAR={env_jar} does not exist.")
            return env_jar

        roots = [gateway_jar_dir]
        venv = os.environ.get("VIRTUAL_ENV")
        if venv:
            roots.append(venv)
        roots += ["/usr/local/share", "/usr/share", "/opt/homebrew", os.path.expanduser("~/.local/share")]

        jar = ReasonerOracle._search_for_jar("elk-owlapi-standalone*.jar", roots)
        if jar is None:
            raise FileNotFoundError(
                "ELK jar not found. Download elk-owlapi-standalone-0.4.2-bin.jar from "
                "https://repo1.maven.org/maven2/org/semanticweb/elk/elk-owlapi-standalone/0.4.2/ "
                "and place it in the project directory (rename to elk-owlapi-standalone-0.4.2.jar), "
                "or set the ELK_JAR environment variable."
            )
        return jar

    @staticmethod
    def _find_log4j_jars() -> list[str]:
        """
        Find log4j jars required by ELK 0.4.2.  The owlready2 pellet directory
        ships log4j-1.2-api (legacy API bridge), log4j-api, and log4j-core,
        which together satisfy org.apache.log4j.Logger at runtime.
        """
        pellet_dir = os.path.join(os.path.dirname(owlready2.__file__), "pellet")
        return glob.glob(os.path.join(pellet_dir, "log4j*.jar"))


    def __init__(
        self,
        path: str,
        gateway_jar_dir: str | None = None,
        port: int = 25333,
        oracle_skills: dict[str, float] | None = None,
        reasoner: str = "elk",
    ):
        self._sig, self._O = extract_ontology(path)
        self._reasoner_type = reasoner.lower()

        # Default gateway_jar_dir: same directory as this source file,
        # which is where OWLGateway.class should be compiled.
        if gateway_jar_dir is None:
            gateway_jar_dir = os.path.dirname(os.path.abspath(__file__))
        logger.info("Using gateway_jar_dir: %s", gateway_jar_dir)

        hermit_jar = self._find_hermit_jar()
        py4j_jar   = self._find_py4j_jar()

        jars = [gateway_jar_dir, hermit_jar, py4j_jar]
        if self._reasoner_type == "elk":
            jars.append(self._find_elk_jar(gateway_jar_dir))
            jars.extend(self._find_log4j_jars())

        classpath = os.pathsep.join(jars)
        logger.debug("Java classpath: %s", classpath)
        logger.info("Starting OWLGateway Java process (reasoner=%s)...", self._reasoner_type)
        self._java_proc = subprocess.Popen(
            ["java", "-cp", classpath, "OWLGateway", str(port), self._reasoner_type],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )

        # GatewayServer.start() launches a daemon thread and returns before
        # the socket is bound, so the ready message can arrive slightly before
        # the port is actually accepting connections.  Poll until it is.
        _wait_for_gateway(self._java_proc, port)

        self._gw  = JavaGateway(
            gateway_parameters=GatewayParameters(port=port, eager_load=True),
            callback_server_parameters=None,
        )
        self._owl = self._gw.entry_point

        # Load all O-axioms into the O-reasoner
        for gci in self._O:
            self._owl.add_gci(_encode(gci.lhs), _encode(gci.rhs))

        # Start a second gateway for H-entailment on a free port
        h_port = _find_free_port()
        self._h_reasoner = HypothesisReasoner(classpath, h_port, self._reasoner_type)

        # oracle_skills: map skill name → probability in [0, 1].
        # Supported skills: "saturate_left", "unsaturate_right",
        #                   "compose_left", "compose_right".
        self._oracle_skills: dict[str, float] = oracle_skills or {}

        logger.info("ReasonerOracle ready. Σ = %s", self._sig)

    def close(self) -> None:
        """Shut down both the O-reasoner and H-reasoner Java processes."""
        try:
            self._gw.close()
            self._java_proc.terminate()
            self._java_proc.wait(timeout=5)
        except Exception:
            pass
        try:
            self._h_reasoner.close()
        except Exception:
            pass

    def __del__(self):
        self.close()

    @property
    def signature(self) -> set[str]:
        return self._sig

    def make_H_MQ(self, H):
        """Return the H-reasoner as the H-entailment callable."""
        return self._h_reasoner

    def on_H_add(self, gci: GCI) -> None:
        """Keep the H-reasoner in sync when a GCI is added to H."""
        self._h_reasoner.add(gci)

    def MQ(self, gci: GCI) -> bool:
        """Membership query via HermiT: O |= lhs ⊑ rhs?"""
        return self._owl.entails(_encode(gci.lhs), _encode(gci.rhs))

    # ------------------------------------------------------------------
    # Oracle skills — structural transformations of counterexamples
    # ------------------------------------------------------------------

    def _is_counterexample(self, gci: GCI) -> bool:
        """True iff O |= gci and H ⊭ gci."""
        return (self._owl.entails(_encode(gci.lhs), _encode(gci.rhs))
                and not self._h_reasoner.entails(gci))

    def _saturate_left(self, gci: GCI) -> GCI:
        """
        Add concept names from the signature to the LHS.

        In EL, lhs ⊑ rhs implies (lhs ⊓ X) ⊑ rhs by monotonicity, so every
        candidate is O-valid automatically.  We only need to check H ⊭ candidate.
        """
        current = gci
        for atom in sorted(self._sig):
            if atom in current.lhs.atoms:
                continue
            candidate = GCI(
                lhs=ELConcept(atoms=current.lhs.atoms | frozenset({atom}),
                              existentials=current.lhs.existentials),
                rhs=current.rhs,
            )
            if not self._h_reasoner.entails(candidate):
                current = candidate
        return current

    def _unsaturate_right(self, gci: GCI) -> GCI:
        """
        Remove atoms from the RHS.

        Produces a weaker consequent; both O- and H-entailment must be rechecked.
        Does not reduce the RHS to ⊤ (empty concept).
        """
        current = gci
        for atom in sorted(current.rhs.atoms):
            new_rhs = ELConcept(
                atoms=current.rhs.atoms - frozenset({atom}),
                existentials=current.rhs.existentials,
            )
            if not new_rhs.atoms and not new_rhs.existentials:
                continue  # refuse to reduce to ⊤
            candidate = GCI(lhs=current.lhs, rhs=new_rhs)
            if self._is_counterexample(candidate):
                current = candidate
        return current

    def _compose_left(self, gci: GCI) -> GCI:
        """
        Replace an LHS atom A with a subclass B using a T-axiom B ⊑ A.

        If T |= B ⊑ A and the current LHS contains A, swapping A → B gives a
        more specific LHS that is still O-entailed (by transitivity).
        """
        current = gci
        for o_ax in self._O:
            # Pattern: single named class  B ⊑ A
            if (len(o_ax.lhs.atoms) == 1 and not o_ax.lhs.existentials
                    and len(o_ax.rhs.atoms) == 1 and not o_ax.rhs.existentials):
                a = next(iter(o_ax.rhs.atoms))
                b = next(iter(o_ax.lhs.atoms))
                if a in current.lhs.atoms and b not in current.lhs.atoms:
                    candidate = GCI(
                        lhs=ELConcept(
                            atoms=(current.lhs.atoms - frozenset({a})) | frozenset({b}),
                            existentials=current.lhs.existentials,
                        ),
                        rhs=current.rhs,
                    )
                    if self._is_counterexample(candidate):
                        current = candidate
                        break
        return current

    def _compose_right(self, gci: GCI) -> GCI:
        """
        Replace an RHS atom B with a superclass D using a T-axiom B ⊑ D.

        Produces a weaker consequent that is still O-entailed by transitivity.
        """
        current = gci
        for o_ax in self._O:
            if (len(o_ax.lhs.atoms) == 1 and not o_ax.lhs.existentials
                    and len(o_ax.rhs.atoms) == 1 and not o_ax.rhs.existentials):
                b = next(iter(o_ax.lhs.atoms))
                d = next(iter(o_ax.rhs.atoms))
                if b in current.rhs.atoms and d not in current.rhs.atoms:
                    candidate = GCI(
                        lhs=current.lhs,
                        rhs=ELConcept(
                            atoms=(current.rhs.atoms - frozenset({b})) | frozenset({d}),
                            existentials=current.rhs.existentials,
                        ),
                    )
                    if self._is_counterexample(candidate):
                        current = candidate
                        break
        return current

    def EQ(self, hypothesis: set[GCI]) -> Optional[GCI]:
        """
        Equivalence query: return a (possibly transformed) counterexample GCI,
        or None if H ≡ O.

        A base counterexample is first found by scanning O for a GCI not yet
        entailed by H.  Then each enabled oracle skill is applied with its
        configured probability, potentially producing a structurally different
        GCI that is still valid (O-entailed, H-not-entailed).

        Oracle skills are configured via the ``oracle_skills`` constructor
        parameter, e.g.::

            oracle_skills={
                "saturate_left":    0.8,
                "unsaturate_right": 0.5,
                "compose_left":     0.6,
                "compose_right":    0.6,
            }
        """
        base = None
        for o_gci in self._O:
            if not self._h_reasoner.entails(o_gci):
                base = o_gci
                break
        if base is None:
            return None

        ce = base
        skill_map = [
            ("saturate_left",    self._saturate_left),
            ("unsaturate_right", self._unsaturate_right),
            ("compose_left",     self._compose_left),
            ("compose_right",    self._compose_right),
        ]
        for name, fn in skill_map:
            prob = self._oracle_skills.get(name, 0.0)
            if prob > 0.0 and random.random() < prob:
                ce = fn(ce)

        return ce


# Parsing ontologies
def extract_ontology(path: str) -> tuple[set[str], set[GCI]]:
    g = Graph()
    g.parse(path)

    # --- Signature: all named OWL classes ----------------------------
    sig: set[str] = set()
    for clause in g.subjects(RDF.type, OWL.Class):
        if isinstance(clause, URIRef):
            sig.add(local_name(clause))

    # --- GCIs: all rdfs:subClassOf triples ---------------------------
    # Also pick up classes used only as subjects/objects of subClassOf
    gcis: set[GCI] = set()
    for subj, _, obj in g.triples((None, RDFS.subClassOf, None)):
        for node in (subj, obj):
            if isinstance(node, URIRef):
                sig.add(local_name(node))
        try:
            lhs = parse_concept(g, subj)
            rhs = parse_concept(g, obj)
            gcis.add(GCI(lhs=lhs, rhs=rhs))
        except ValueError as exc:
            logger.warning("Skipping unrecognised SubClassOf triple: %s", exc)

    return sig, gcis
 
def local_name(uri: URIRef) -> str:
    """
    Extract the local name from a URI.

    For  http://example.org/onto#Doctor  →  "Doctor"
    For  http://example.org/onto/Doctor  →  "Doctor"
    """
    uri_str = str(uri)
    if "#" in uri_str:
        return uri_str.split("#")[-1]
    return uri_str.split("/")[-1]

def parse_concept(g:Graph, node) -> ELConcept:
    """
    Recursively parse an RDF node into an ELConcept.

    Named class (URIRef)
        → ELConcept(atoms={local_name})

    Anonymous intersection (BNode with owl:intersectionOf)
        → ELConcept(atoms={all named members}, existentials={...})

    Anonymous restriction (BNode with owl:onProperty + owl:someValuesFrom)
        → ELConcept(existentials={(role, filler_concept)})

    owl:Thing
        → ELConcept()  (⊤)
    """

    # Named class
    if isinstance(node, URIRef):
        if node == OWL.Thing:
            return ELConcept()   # ⊤
        return ELConcept(atoms=frozenset({local_name(node)}))

    if not isinstance(node, BNode):
        raise ValueError(f"Unexpected node type: {type(node)} for {node}")

    # owl:Restriction  →  ∃role.filler
    if (node, RDF.type, OWL.Restriction) in g:
        role_node   = g.value(node, OWL.onProperty)
        filler_node = g.value(node, OWL.someValuesFrom)
        if role_node is None or filler_node is None:
            raise ValueError(f"Malformed owl:Restriction at {node}")
        role   = local_name(role_node)
        filler = parse_concept(g, filler_node)
        return ELConcept(existentials=frozenset({(role, filler)}))

    # owl:intersectionOf  →  C1 ⊓ C2 ⊓ ...
    list_head = g.value(node, OWL.intersectionOf)
    if list_head is not None:
        members = list(Collection(g, list_head))
        combined_atoms: set[str] = set()
        combined_existentials: set[tuple] = set()
        for member in members:
            part = parse_concept(g, member)
            combined_atoms        |= part.atoms
            combined_existentials |= part.existentials
        return ELConcept(
            atoms=frozenset(combined_atoms),
            existentials=frozenset(combined_existentials),
        )

    raise ValueError(f"Unrecognised BNode structure at {node}") 
 