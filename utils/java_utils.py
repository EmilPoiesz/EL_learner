"""
java_utils.py — utilities for launching and connecting to OWLGateway Java processes.

Concept encoding
----------------
encode(concept)               — serialise an ELConcept to the OWLGateway wire format

Jar discovery
-------------
find_owlapi_jar()             — OWL API 3 classes (shipped inside owlready2's HermiT.jar)
find_py4j_jar()               — py4j runtime jar
find_elk_jar(gateway_jar_dir) — ELK OWL-API standalone jar
find_log4j_jars()             — log4j jars required by ELK 0.4.2
build_classpath(gateway_jar_dir) — assemble a full classpath string
find_java()                   — a java executable new enough for OWLGateway (Java 11+)

Gateway lifecycle
-----------------
start_gateway(proc)           — block until the gateway prints READY:<port>, return the port
load_ontology(owl, path)      — load an ontology file into a gateway, return its OntologyInfo
"""

from __future__ import annotations

import glob
import logging
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
from dataclasses import dataclass
from functools import cache

import owlready2

from learner.el_algorithm import ELConcept

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Concept encoding
# ---------------------------------------------------------------------------

@cache
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


def find_owlapi_jar() -> str:
    """
    Locate the jar that provides the OWL API 3 classes OWLGateway compiles against.

    owlready2 bundles OWL API 3 inside its HermiT.jar, so this jar is required
    for every reasoner (ELK 0.4.2 ships without its own OWL API).  It also
    provides the HermiT reasoner, used when ``reasoner="hermit"``.
    """
    jar = os.path.join(os.path.dirname(owlready2.__file__), "hermit", "HermiT.jar")
    if not os.path.exists(jar):
        raise FileNotFoundError(f"HermiT.jar not found at {jar}")
    return jar


def find_py4j_jar() -> str:
    roots = [
        os.path.join(sys.prefix, "share", "py4j"),
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
            "and place it in the java/ directory (rename to elk-owlapi-standalone-0.4.2.jar), "
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
        [gateway_jar_dir, find_owlapi_jar(), find_py4j_jar(), find_elk_jar(gateway_jar_dir)]
        + find_log4j_jars()
    )


MIN_JAVA = 11   # OWLGateway.class is Java 11 bytecode (java_env.bash: javac --release 11)


def _java_major(java: str) -> int | None:
    """The major version of *java* (8 for "1.8.0_472", 17 for "17.0.18"), or None."""
    try:
        out = subprocess.run([java, "-version"], capture_output=True, text=True, timeout=10).stderr
    except (OSError, subprocess.TimeoutExpired):
        return None
    m = re.search(r'version "(?:1\.)?(\d+)', out)
    return int(m[1]) if m else None


@cache
def find_java() -> str:
    """
    A java executable that can run OWLGateway: the first of ``java`` on PATH,
    ``$JAVA_HOME/bin/java`` and (macOS) ``/usr/libexec/java_home -v 11+`` that is
    Java 11 or newer.  An IDE or notebook kernel often starts with a different
    PATH and JAVA_HOME than the terminal, where an older default Java comes first.
    """
    candidates = [shutil.which("java")]
    if os.environ.get("JAVA_HOME"):
        candidates.append(os.path.join(os.environ["JAVA_HOME"], "bin", "java"))
    if os.path.exists("/usr/libexec/java_home"):
        home = subprocess.run(["/usr/libexec/java_home", "-v", f"{MIN_JAVA}+"], capture_output=True, text=True)
        if home.returncode == 0:
            candidates.append(os.path.join(home.stdout.strip(), "bin", "java"))
    seen = []
    for java in dict.fromkeys(filter(None, candidates)):   # PATH and JAVA_HOME often agree
        major = _java_major(java)
        if major is not None and major >= MIN_JAVA:
            return java
        seen.append(f"{java} (Java {major or '?'})")
    raise RuntimeError(
        f"OWLGateway needs Java {MIN_JAVA} or newer; found only: {', '.join(seen) or 'no java'}. "
        "Put a newer JDK first on PATH or point JAVA_HOME at it."
    )


# ---------------------------------------------------------------------------
# Gateway lifecycle
# ---------------------------------------------------------------------------

def _drain(stream, name: str) -> None:
    """Read *stream* (a gateway's pipe) to its end, logging each line at debug level."""
    try:
        for raw in stream:
            logger.debug("OWLGateway %s: %s", name, raw.decode(errors="replace").rstrip())
    except Exception:
        pass


def start_gateway(proc: subprocess.Popen, timeout: float = 10.0) -> int:
    """
    Block until OWLGateway prints its ``READY:<port>`` line, then return the port.

    OWLGateway is launched with port 0 so the OS picks a free port; the actual
    bound port is reported on stdout as ``READY:<port>``.  Reading it here
    eliminates the TOCTOU race that existed when Python selected the port itself.

    Once READY, both pipes are read until the process ends and each line goes to
    the debug log: a pipe nobody reads fills up (about 64 KB), and the JVM then
    blocks on its next log line.  Before READY, stderr is left for the error message.

    Raises RuntimeError on timeout or if the process exits before printing READY.
    """
    result: queue.Queue[int | None] = queue.Queue()

    def _read_stdout() -> None:
        ready = False
        try:
            for raw in proc.stdout:
                line = raw.decode(errors="replace").rstrip()
                if not ready and line.startswith("READY:"):
                    ready = True
                    result.put(int(line.split(":")[1]))
                elif ready:
                    logger.debug("OWLGateway stdout: %s", line)
        except Exception:
            pass
        if not ready:
            result.put(None)  # stdout closed without a READY line

    threading.Thread(target=_read_stdout, daemon=True).start()

    try:
        port = result.get(timeout=timeout)
    except queue.Empty:
        port = None

    if port is not None:
        threading.Thread(target=_drain, args=(proc.stderr, "stderr"), daemon=True).start()
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


@dataclass(frozen=True)
class OntologyInfo:
    """
    An ontology file as the OWL API reads it (``load_ontology``), which is how the
    original ExactLearner-LLM reads it.

    classes : concept names (local names), without owl:Thing
    roles   : object property names
    top     : the file mentions owl:Thing, so upstream counts it among the classes
    size    : upstream's ontology size, the h of its PAC budget (an equivalence counts once)
    """
    classes: frozenset[str]
    roles: frozenset[str]
    top: bool
    size: int


def load_ontology(owl, path: str) -> OntologyInfo:
    """
    Load the ontology file at *path* into the gateway *owl* (a py4j entry point)
    as its whole ontology, and return what the OWL API sees in it.  Afterwards the
    gateway's ``entails`` reasons over every axiom of the file that its reasoner
    supports, including those our EL parser skips (e.g. nominals in generations.owl).
    """
    owl.load_file(os.path.abspath(path))
    return OntologyInfo(
        classes=frozenset(owl.class_names()),
        roles=frozenset(owl.role_names()),
        top=bool(owl.mentions_thing()),
        size=int(owl.ontology_size()),
    )
