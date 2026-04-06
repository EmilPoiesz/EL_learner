from typing import Optional

from EL_algorithm import Oracle, ELConcept, GCI
from HypothesisReasoner import HypothesisReasoner


class LLMOracle(Oracle):
    """
    An Oracle backed by a language model for O-entailment queries.

    The H-entailment side is delegated to a HypothesisReasoner (Java/ELK)
    passed in at construction time — the same component used by ReasonerOracle.

    Parameters
    ----------
    signature : set[str]
        Σ_O — the set of atomic concept names in the target ontology.
    h_reasoner : HypothesisReasoner
        A running HypothesisReasoner instance that will be kept in sync
        with H and used to answer H-entailment queries.
    """

    def __init__(self, signature: set[str], h_reasoner: HypothesisReasoner):
        self._sig = signature
        self._h_reasoner = h_reasoner

    @property
    def signature(self) -> set[str]:
        return self._sig

    # ------------------------------------------------------------------
    # H-entailment — delegated to the HypothesisReasoner
    # ------------------------------------------------------------------

    def make_H_MQ(self, H: set[GCI]):
        """Return the H-reasoner as the H-entailment callable."""
        return self._h_reasoner

    def on_H_add(self, gci: GCI) -> None:
        """Keep the H-reasoner in sync when a GCI is added to H."""
        self._h_reasoner.add(gci)

    # ------------------------------------------------------------------
    # O-entailment — to be implemented via LLM
    # ------------------------------------------------------------------

    def MQ(self, gci: GCI) -> bool:
        """
        Membership query: does O entail lhs ⊑ rhs?

        TODO: implement by prompting the LLM with a representation of the GCI
        and returning its True/False judgment.
        """
        raise NotImplementedError

    def EQ(self, hypothesis: set[GCI]) -> Optional[GCI]:
        """
        Equivalence query: return a GCI entailed by O but not by H, or None.

        TODO: implement by prompting the LLM to identify a counterexample GCI
        given the current hypothesis H, then verifying it against the
        H-reasoner (self._h_reasoner.entails).
        """
        raise NotImplementedError

    def close(self) -> None:
        """Shut down the H-reasoner Java process."""
        try:
            self._h_reasoner.close()
        except Exception:
            pass

    def __del__(self):
        self.close()
