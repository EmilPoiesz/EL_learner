"""
test_java_utils.py
==================
Gateway start-up and Java discovery (utils/java_utils.py), with a stand-in
process instead of the Java gateway where the gateway is not the point.

1  start_gateway returns the READY port, then keeps both pipes drained: a process
   that writes far more than a pipe buffer after READY runs to its end
2  start_gateway reports the process's stderr when it exits without READY
3  find_java finds a Java 11+
"""

from __future__ import annotations

import subprocess
import sys

import pytest

from utils.java_utils import MIN_JAVA, _java_major, find_java, start_gateway


def _process(code: str) -> subprocess.Popen:
    return subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def test_1_pipes_drained_after_ready():
    # 2 MB on each pipe, far beyond the ~64 KB a pipe holds: without draining the
    # process blocks on its first write that does not fit, and wait() times out.
    proc = _process(
        "import sys\n"
        "print('READY:4321', flush=True)\n"
        "sys.stderr.write('e' * 2_000_000); sys.stderr.flush()\n"
        "sys.stdout.write('o' * 2_000_000); sys.stdout.flush()\n"
    )
    try:
        assert start_gateway(proc) == 4321
        assert proc.wait(timeout=20) == 0
    finally:
        proc.kill()


def test_2_error_reports_stderr():
    proc = _process("import sys; sys.stderr.write('no gateway here'); sys.exit(1)")
    with pytest.raises(RuntimeError, match="no gateway here"):
        start_gateway(proc, timeout=5)


def test_3_find_java():
    try:
        java = find_java()
    except RuntimeError as exc:
        pytest.skip(str(exc))
    assert _java_major(java) >= MIN_JAVA
