import logging
import subprocess

from py4j.java_gateway import JavaGateway, GatewayParameters

from learner.el_algorithm import GCI
from utils.java_utils import encode, start_gateway

logger = logging.getLogger(__name__)


class HypothesisReasoner:
    """
    A reasoner dedicated to the current hypothesis H.

    Maintains a running Java OWLGateway process whose ontology is kept in
    sync with H via add() calls.  Used by oracles to provide a complete
    H-entailment check (make_H_MQ / on_H_add).
    """

    def __init__(self, classpath: str, reasoner: str = "elk"):
        self._java_proc = subprocess.Popen(
            ["java", "-cp", classpath, "OWLGateway", "0", reasoner],  # "0" → OS assigns a free port
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        port = start_gateway(self._java_proc)

        self._gw = JavaGateway(
            gateway_parameters=GatewayParameters(port=port, eager_load=True),
            callback_server_parameters=None,
        )
        self._owl = self._gw.entry_point
        # Positive-only cache: EL entailment is monotone, so a GCI entailed by H
        # remains entailed after any further axiom is added — cached True values
        # never go stale.
        self._entailed_cache: set[GCI] = set()

    def add(self, gci: GCI) -> None:
        self._owl.add_gci(encode(gci.lhs), encode(gci.rhs))

    def entails(self, gci: GCI) -> bool:
        if gci in self._entailed_cache:
            return True
        result = self._owl.entails(encode(gci.lhs), encode(gci.rhs))
        if result:
            self._entailed_cache.add(gci)
        return result

    def __call__(self, gci: GCI) -> bool:
        return self.entails(gci)

    def close(self) -> None:
        try:
            self._gw.close()
        except Exception:
            logger.debug("Error closing py4j gateway", exc_info=True)
        try:
            self._java_proc.terminate()
            self._java_proc.wait(timeout=5)
        except Exception:
            logger.debug("Error terminating Java process", exc_info=True)

    def __enter__(self) -> "HypothesisReasoner":
        return self

    def __exit__(self, *args) -> None:
        self.close()