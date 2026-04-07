"""
conftest.py — shared pytest fixtures for the EL learning algorithm test suite.

Fixtures
--------
reasoner_name      : str           — "elk" or "hermit" (from --reasoner CLI option)
reasoner_oracle    : ReasonerOracle | None — session-scoped oracle for unit tests
integration_oracle : ReasonerOracle | None — separate session-scoped oracle for test 20
"""

import os
import pytest

from reasoner_oracle import ReasonerOracle

_ROOT        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # project root
_TTL_PATH    = os.path.join(_ROOT, "ontologies", "test_minimal.ttl")
_PROJECT_DIR = os.path.join(_ROOT, "java")


def pytest_addoption(parser):
    parser.addoption(
        "--reasoner", choices=["elk", "hermit"], default="elk",
        help="DL reasoner backend (elk or hermit). Default: elk.",
    )


@pytest.fixture(scope="session")
def reasoner_name(request):
    return request.config.getoption("--reasoner")


def _start_oracle(reasoner_name: str, skills: dict) -> ReasonerOracle:
    return ReasonerOracle(
        path=_TTL_PATH,
        gateway_jar_dir=_PROJECT_DIR,
        reasoner=reasoner_name,
        oracle_skills=skills,
    )


@pytest.fixture(scope="session")
def reasoner_oracle(reasoner_name):
    """
    Session-scoped oracle for unit tests 1–19.
    Yields None if the reasoner process cannot start (tests that require it will skip).
    """
    try:
        oracle = _start_oracle(reasoner_name, {
            "saturate_left": 0.5, "unsaturate_right": 0.5,
            "compose_left":  0.5, "compose_right":    0.5,
        })
    except Exception:
        yield None
        return
    try:
        yield oracle
    finally:
        oracle.close()


@pytest.fixture(scope="session")
def integration_oracle(reasoner_name):
    """
    Fresh session-scoped oracle for test_20_integration.
    A separate process is used so the H-reasoner starts from an empty state.
    Yields None if the reasoner process cannot start.
    """
    try:
        oracle = _start_oracle(reasoner_name, {
            "saturate_left": 0.8, "unsaturate_right": 0.5,
            "compose_left":  0.6, "compose_right":    0.6,
        })
    except Exception:
        yield None
        return
    try:
        yield oracle
    finally:
        oracle.close()
