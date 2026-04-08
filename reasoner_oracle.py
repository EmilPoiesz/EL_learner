import logging
import random
import subprocess
import os

from py4j.java_gateway import JavaGateway, GatewayParameters

from el_algorithm import Oracle, ELConcept, GCI
from typing import Optional

from utils.java_utils import find_hermit_jar, find_py4j_jar, find_elk_jar, find_log4j_jars, start_gateway, encode
from hypothesis_reasoner import HypothesisReasoner
from utils.owl_parser import extract_ontology

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# OWL API + HermiT/ELK reasoner bridge via py4j
# ---------------------------------------------------------------------------

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

    def __init__(
        self,
        path: str,
        gateway_jar_dir: str | None = None,
        oracle_skills: dict[str, float] | None = None,
        reasoner: str = "elk",
    ):
        super().__init__()
        self._sig, self._O = extract_ontology(path)
        self._reasoner_type = reasoner.lower()

        # Default gateway_jar_dir: same directory as this source file,
        # which is where OWLGateway.class should be compiled.
        if gateway_jar_dir is None:
            gateway_jar_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "java")
        logger.info("Using gateway_jar_dir: %s", gateway_jar_dir)

        hermit_jar = find_hermit_jar()
        py4j_jar   = find_py4j_jar()

        jars = [gateway_jar_dir, hermit_jar, py4j_jar]
        if self._reasoner_type == "elk":
            jars.append(find_elk_jar(gateway_jar_dir))
            jars.extend(find_log4j_jars())

        classpath = os.pathsep.join(jars)
        logger.debug("Java classpath: %s", classpath)
        logger.info("Starting OWLGateway Java process (reasoner=%s)...", self._reasoner_type)
        self._java_proc = subprocess.Popen(
            ["java", "-cp", classpath, "OWLGateway", "0", self._reasoner_type],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        port = start_gateway(self._java_proc)
        logger.info("OWLGateway listening on port %d", port)

        self._gw  = JavaGateway(
            gateway_parameters=GatewayParameters(port=port, eager_load=True),
            callback_server_parameters=None,
        )
        self._owl = self._gw.entry_point

        # Load all O-axioms into the O-reasoner
        for gci in self._O:
            self._owl.add_gci(encode(gci.lhs), encode(gci.rhs))

        # Start a second gateway for H-entailment
        self._h_reasoner = HypothesisReasoner(classpath, self._reasoner_type)

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
            logger.debug("Error shutting down O-reasoner", exc_info=True)
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

    def make_H_MQ(self, H):
        """Return the H-reasoner as the H-entailment callable."""
        return self._h_reasoner

    def on_H_add(self, gci: GCI) -> None:
        """Keep the H-reasoner in sync when a GCI is added to H."""
        self._h_reasoner.add(gci)

    def _MQ(self, gci: GCI) -> bool:
        """Membership query via HermiT: O |= lhs ⊑ rhs?"""
        return self._owl.entails(encode(gci.lhs), encode(gci.rhs))

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

    def _EQ(self, hypothesis: set[GCI]) -> Optional[GCI]:
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

