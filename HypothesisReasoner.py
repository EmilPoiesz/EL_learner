import socket
import subprocess
import time

from py4j.java_gateway import JavaGateway, GatewayParameters

from EL_algorithm import ELConcept, GCI


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
    sync with H via add() calls.  Used by oracles to provide a complete
    H-entailment check (make_H_MQ / on_H_add).
    """

    def __init__(self, classpath: str, port: int, reasoner: str = "elk"):
        self._port = port
        self._java_proc = subprocess.Popen(
            ["java", "-cp", classpath, "OWLGateway", str(port), reasoner],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        _wait_for_gateway(self._java_proc, port)

        self._gw = JavaGateway(
            gateway_parameters=GatewayParameters(port=port, eager_load=True),
            callback_server_parameters=None,
        )
        self._owl = self._gw.entry_point

    def add(self, gci: GCI) -> None:
        self._owl.add_gci(_encode(gci.lhs), _encode(gci.rhs))

    def entails(self, gci: GCI) -> bool:
        return self._owl.entails(_encode(gci.lhs), _encode(gci.rhs))

    def __call__(self, gci: GCI) -> bool:
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