"""
java_utils.py — utilities for launching and connecting to OWLGateway Java processes.

Concept encoding
----------------
encode(concept)               — serialise an ELConcept to the OWLGateway wire format

Jar discovery
-------------
find_hermit_jar()             — HermiT.jar bundled with owlready2
find_py4j_jar()               — py4j runtime jar
find_elk_jar(gateway_jar_dir) — ELK OWL-API standalone jar
find_log4j_jars()             — log4j jars required by ELK 0.4.2
build_classpath(gateway_jar_dir) — assemble a full classpath string

Gateway lifecycle
-----------------
start_gateway(proc)           — block until the gateway prints READY:<port>, return the port
"""

from __future__ import annotations

import glob
import os
import queue
import subprocess
import threading

import owlready2

from learner.el_algorithm import ELConcept


# ---------------------------------------------------------------------------
# Concept encoding
# ---------------------------------------------------------------------------

def encode(concept: ELConcept) -> str:
    """
    Serialise an ELConcept to the OWLGateway wire format.

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
        parts.append(f"SOME:{role}:({encode(filler)})")
    if len(parts) == 1:
        return parts[0]
    return "AND:" + ",".join(f"({p})" for p in parts)


# ---------------------------------------------------------------------------
# Jar discovery
# ---------------------------------------------------------------------------

def _search_for_jar(pattern: str, roots: list[str]) -> str | None:
    """Return the first jar matching *pattern* found under any of *roots*, or None."""
    for root in roots:
        matches = glob.glob(os.path.join(root, "**", pattern), recursive=True)
        if matches:
            return matches[0]
    return None


def find_hermit_jar() -> str:
    jar = os.path.join(os.path.dirname(owlready2.__file__), "hermit", "HermiT.jar")
    if not os.path.exists(jar):
        raise FileNotFoundError(f"HermiT.jar not found at {jar}")
    return jar


def find_py4j_jar() -> str:
    roots = [
        "/usr/local/share/py4j",
        "/usr/share/py4j",
        "/opt/homebrew",
        os.path.expanduser("~/.local/share/py4j"),
    ]
    venv = os.environ.get("VIRTUAL_ENV")
    if venv:
        roots.append(venv)
    jar = _search_for_jar("py4j*.jar", roots)
    if jar is None:
        raise FileNotFoundError("py4j jar not found. Check your venv or set VIRTUAL_ENV.")
    return jar


def find_elk_jar(gateway_jar_dir: str) -> str:
    """Locate the ELK standalone jar."""
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

    jar = _search_for_jar("elk-owlapi-standalone*.jar", roots)
    if jar is None:
        raise FileNotFoundError(
            "ELK jar not found. Download elk-owlapi-standalone-0.4.2-bin.jar from "
            "https://repo1.maven.org/maven2/org/semanticweb/elk/elk-owlapi-standalone/0.4.2/ "
            "and place it in the project directory (rename to elk-owlapi-standalone-0.4.2.jar), "
            "or set the ELK_JAR environment variable."
        )
    return jar


def find_log4j_jars() -> list[str]:
    """
    Find log4j jars required by ELK 0.4.2.  The owlready2 pellet directory
    ships log4j-1.2-api (legacy API bridge), log4j-api, and log4j-core,
    which together satisfy org.apache.log4j.Logger at runtime.
    """
    pellet_dir = os.path.join(os.path.dirname(owlready2.__file__), "pellet")
    return glob.glob(os.path.join(pellet_dir, "log4j*.jar"))


def build_classpath(gateway_jar_dir: str) -> str:
    """Assemble the full Java classpath for OWLGateway + ELK."""
    return os.pathsep.join(
        [gateway_jar_dir, find_hermit_jar(), find_py4j_jar(), find_elk_jar(gateway_jar_dir)]
        + find_log4j_jars()
    )


# ---------------------------------------------------------------------------
# Gateway lifecycle
# ---------------------------------------------------------------------------

def start_gateway(proc: subprocess.Popen, timeout: float = 10.0) -> int:
    """
    Block until OWLGateway prints its ``READY:<port>`` line, then return the port.

    OWLGateway is launched with port 0 so the OS picks a free port; the actual
    bound port is reported on stdout as ``READY:<port>``.  Reading it here
    eliminates the TOCTOU race that existed when Python selected the port itself.

    Raises RuntimeError on timeout or if the process exits before printing READY.
    """
    result: queue.Queue[int | None] = queue.Queue()

    def _read_stdout() -> None:
        try:
            for raw in proc.stdout:
                line = raw.decode(errors="replace").strip()
                if line.startswith("READY:"):
                    result.put(int(line.split(":")[1]))
                    return
        except Exception:
            pass
        result.put(None)  # stdout closed without a READY line

    threading.Thread(target=_read_stdout, daemon=True).start()

    try:
        port = result.get(timeout=timeout)
    except queue.Empty:
        port = None

    if port is not None:
        return port

    proc.terminate()
    try:
        _, stderr_bytes = proc.communicate(timeout=3)
    except Exception:
        stderr_bytes = b""
    stderr_str = stderr_bytes.decode(errors="replace").strip()
    raise RuntimeError(
        "OWLGateway failed to start."
        + (f"\nJava stderr:\n{stderr_str}" if stderr_str else "")
    )
