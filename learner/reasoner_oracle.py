import logging
import random
import subprocess
import os

from py4j.java_gateway import JavaGateway, GatewayParameters

from learner.el_algorithm import Oracle, ELConcept, GCI
from typing import Optional

from utils.java_utils import (OntologyInfo, encode, find_elk_jar, find_java, find_log4j_jars,
                              find_owlapi_jar, find_py4j_jar, load_ontology, start_gateway)
from learner.evaluation import all_statements
from learner.hypothesis_reasoner import HypothesisReasoner
from utils.owl_parser import extract_ontology

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# OWL API + ELK/HermiT reasoner bridge via py4j
# ---------------------------------------------------------------------------

class ReasonerOracle(Oracle):
    """
    An Oracle backed by the OWL API and a configurable DL reasoner via a py4j gateway.

    The gateway is a small Java process (OWLGateway.java) that wraps
    OWL API + the chosen reasoner and exposes these methods over py4j:
      - load_file(path)            — load the target ontology file
      - add_gci(lhs_str, rhs_str)  — load an axiom
      - entails(lhs_str, rhs_str)  — subsumption query
      - clear()                    — reset the ontology

    MQs are answered on the file as the OWL API reads it, so they also use the
    axioms our EL parser skips (e.g. the nominals of generations.owl); the
    signature, ``roles``, ``top`` and ``ontology_size`` come from the same read
    (``info``).  ``axioms`` stays the parsed EL GCIs, which the EQ and the oracle
    skills need as GCIs.

    Parameters
    ----------
    path : str
        Path to the OWL/Turtle ontology file.
    gateway_jar_dir : str, optional
        Directory containing OWLGateway.class.  Defaults to the project's
        ``java/`` directory, where OWLGateway.java is compiled.
    reasoner : str, optional
        Which DL reasoner to use: ``"elk"`` (default) or ``"hermit"``.
        ELK is an OWL 2 EL reasoner; it is fast and complete for the EL
        profile the learner targets.  HermiT is a full OWL 2 DL reasoner.
        When using ELK the ``elk-owlapi-standalone-*.jar`` must be on the
        classpath (place it in the java/ directory or set ELK_JAR).
    seed : int | None, optional
        Seed for the random choices of the oracle skills (default 0), so a run
        is reproducible.  ``None`` draws a fresh seed each time.
    """

    def __init__(
        self,
        path: str,
        gateway_jar_dir: str | None = None,
        oracle_skills: dict[str, float] | None = None,
        reasoner: str = "elk",
        seed: int | None = 0,
    ):
        super().__init__()
        # Axioms of the file the parsed GCIs leave out (not EL GCIs); O-reasoning
        # still uses them, since the O-reasoner reads the file itself.
        self.skipped_axioms: list[str] = []
        self._sig, self._O = extract_ontology(path, self.skipped_axioms)
        # O in a fixed order: set iteration order changes per process, and the
        # first unentailed axiom of O is the base counterexample.
        self._O_sorted: list[GCI] = sorted(self._O, key=str)
        self._rng = random.Random(seed)
        self._reasoner_type = reasoner.lower()

        # Default gateway_jar_dir: same directory as this source file,
        # which is where OWLGateway.class should be compiled.
        if gateway_jar_dir is None:
            gateway_jar_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../java")
        logger.info("Using gateway_jar_dir: %s", gateway_jar_dir)

        owlapi_jar = find_owlapi_jar()
        py4j_jar   = find_py4j_jar()

        jars = [gateway_jar_dir, owlapi_jar, py4j_jar]
        if self._reasoner_type == "elk":
            jars.append(find_elk_jar(gateway_jar_dir))
            jars.extend(find_log4j_jars())

        classpath = os.pathsep.join(jars)
        logger.debug("Java classpath: %s", classpath)
        logger.info("Starting OWLGateway Java process (reasoner=%s)...", self._reasoner_type)
        self._java_proc = subprocess.Popen(
            [find_java(), "-cp", classpath, "OWLGateway", "0", self._reasoner_type],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        # From here on a Java process runs: if the rest fails, stop what was started.
        try:
            port = start_gateway(self._java_proc)
            logger.info("OWLGateway listening on port %d", port)

            self._gw  = JavaGateway(
                gateway_parameters=GatewayParameters(port=port, eager_load=True),
                callback_server_parameters=None,
            )
            self._owl = self._gw.entry_point

            # The O-reasoner reasons over the file itself; Σ is the OWL API's class signature.
            self.info: OntologyInfo = load_ontology(self._owl, path)
            self._sig = set(self.info.classes)

            # Start a second gateway for H-entailment
            self._h_reasoner = HypothesisReasoner(classpath, self._reasoner_type)
        except BaseException:
            self.close()
            raise
        self._mq_cache: dict[GCI, bool] = {}

        # oracle_skills: map skill name → probability in [0, 1].
        # Supported skills: "saturate_left", "unsaturate_right",
        #                   "compose_left", "compose_right".
        self._oracle_skills: dict[str, float] = oracle_skills or {}

        logger.info("ReasonerOracle ready. Σ = %s", self._sig)

    def close(self) -> None:
        """Shut down both the O-reasoner and H-reasoner Java processes (those that
        were started: the constructor also calls this when it fails part-way)."""
        if hasattr(self, "_gw"):
            try:
                self._gw.close()
            except Exception:
                logger.debug("Error closing O-reasoner gateway", exc_info=True)
        if hasattr(self, "_java_proc"):
            try:
                self._java_proc.terminate()
                self._java_proc.wait(timeout=5)
            except Exception:
                logger.debug("Error shutting down O-reasoner", exc_info=True)
        if hasattr(self, "_h_reasoner"):
            try:
                self._h_reasoner.close()
            except Exception:
                logger.debug("Error shutting down H-reasoner", exc_info=True)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    @property
    def signature(self) -> set[str]:
        return self._sig

    @property
    def axioms(self) -> set[GCI]:
        """The set of GCIs parsed from the target ontology file."""
        return self._O

    @property
    def roles(self) -> set[str]:
        """The file's object property names (OWL API)."""
        return set(self.info.roles)

    def make_H_MQ(self, H):
        """Return the H-reasoner as the H-entailment callable."""
        return self._h_reasoner

    def on_H_add(self, gci: GCI) -> None:
        """Keep the H-reasoner in sync when a GCI is added to H."""
        self._h_reasoner.add(gci)

    def _MQ(self, gci: GCI) -> bool:
        """Membership query via the configured reasoner: O |= lhs ⊑ rhs?"""
        if gci not in self._mq_cache:
            self._mq_cache[gci] = self._owl.entails(encode(gci.lhs), encode(gci.rhs))
        return self._mq_cache[gci]

    # ------------------------------------------------------------------
    # Oracle skills — structural transformations of counterexamples
    # ------------------------------------------------------------------

    def _is_counterexample(self, gci: GCI) -> bool:
        """True iff O |= gci and H ⊭ gci."""
        return (self._owl.entails(encode(gci.lhs), encode(gci.rhs))
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
        for o_ax in self._O_sorted:
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
        for o_ax in self._O_sorted:
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

    def _EQ(self, hypothesis: set[GCI]) -> Optional[GCI]:
        """
        Equivalence query: return a (possibly transformed) counterexample GCI,
        or None if H ≡ O.

        A base counterexample is first found by scanning O for a GCI not yet
        entailed by H.  When the file has axioms the parsed GCIs leave out
        (``skipped_axioms``, e.g. generations.owl's nominal definitions), O
        entails more than those GCIs, so once H entails all of them the scan
        continues over the paper's statements (``learner.evaluation.all_statements``)
        for one that O entails and H does not.  Then each enabled oracle skill is
        applied with its configured probability, potentially producing a
        structurally different GCI that is still valid (O-entailed, H-not-entailed).

        Oracle skills are configured via the ``oracle_skills`` constructor
        parameter, e.g.::

            oracle_skills={
                "saturate_left":    0.8,
                "unsaturate_right": 0.5,
                "compose_left":     0.6,
                "compose_right":    0.6,
            }
        """
        # Direction 1: O ⊄ H — find a GCI in O not entailed by H
        base = None
        for o_gci in self._O_sorted:
            if not self._h_reasoner.entails(o_gci):
                base = o_gci
                break
        if base is None and self.skipped_axioms:
            # The parsed GCIs are not all of O: look for what else O entails.
            for statement in all_statements(self.info.classes, self.info.roles, self.info.top):
                if self._MQ(statement) and not self._h_reasoner.entails(statement):
                    base = statement
                    break

        # Direction 2: H ⊄ O — find a GCI in H not entailed by O
        if base is None:
            for h_gci in sorted(hypothesis, key=str):
                if not self._owl.entails(encode(h_gci.lhs), encode(h_gci.rhs)):
                    return h_gci
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
            if prob > 0.0 and self._rng.random() < prob:
                ce = fn(ce)

        return ce

