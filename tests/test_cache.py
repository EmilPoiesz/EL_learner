"""
test_cache.py
==============
Pytest suite for the LLM cache module.

Tests:
 1  cache empty at start
 2  first query populates cache
 3  repeated query hits cache (no new model call)
 4  different queries cached independently
 5  persistence across instances (SQLite)
 6  cache disabled via flag
 7  memory cache layer works (no disk hit)
 8  cleanup (DB removed)

Run with:
    pytest test_cache.py -v
"""

from __future__ import annotations

import os
import tempfile
import shutil

import pytest

from learner.cache.cache import LLMCache


# ---------------------------------------------------------------------------
# Mock LLM
# ---------------------------------------------------------------------------

class MockLLM:
    def __init__(self):
        self.calls = 0

    def query(self, prompt: str) -> str:
        self.calls += 1
        return f"response::{prompt}"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def temp_cache_dir():
    path = tempfile.mkdtemp()
    yield path
    shutil.rmtree(path)


@pytest.fixture
def cache(temp_cache_dir):
    db_path = os.path.join(temp_cache_dir, "test_cache.db")
    cache = LLMCache(enabled=True, db_path=db_path, store_prompts=True)
    yield cache
    cache.close()


@pytest.fixture
def mock_llm():
    return MockLLM()


def make_messages(prompt: str):
    return [
        {"role": "system", "content": "system"},
        {"role": "user", "content": prompt},
    ]


MODEL = "test-model"


# ===========================================================================
# 1. Cache starts empty
# ===========================================================================

def test_01_cache_empty(cache):
    messages = make_messages("hello")
    assert cache.get(MODEL, messages) is None


# ===========================================================================
# 2. First query populates cache
# ===========================================================================

def test_02_first_query_populates(cache, mock_llm):
    prompt = "hello"
    messages = make_messages(prompt)

    result = cache.get(MODEL, messages)
    assert result is None

    response = mock_llm.query(prompt)
    cache.set(MODEL, messages, response)

    cached = cache.get(MODEL, messages)
    assert cached == response
    assert mock_llm.calls == 1


# ===========================================================================
# 3. Repeated query hits cache
# ===========================================================================

def test_03_cache_hit(cache, mock_llm):
    prompt = "hello"
    messages = make_messages(prompt)

    # First call
    response = mock_llm.query(prompt)
    cache.set(MODEL, messages, response)

    # Second call (should NOT hit model)
    cached = cache.get(MODEL, messages)

    assert cached == response
    assert mock_llm.calls == 1  # critical check


# ===========================================================================
# 4. Different queries cached independently
# ===========================================================================

def test_04_multiple_queries(cache, mock_llm):
    p1 = "hello"
    p2 = "world"

    m1 = make_messages(p1)
    m2 = make_messages(p2)

    r1 = mock_llm.query(p1)
    cache.set(MODEL, m1, r1)

    r2 = mock_llm.query(p2)
    cache.set(MODEL, m2, r2)

    assert cache.get(MODEL, m1) == r1
    assert cache.get(MODEL, m2) == r2

    assert mock_llm.calls == 2


# ===========================================================================
# 5. Persistence across instances
# ===========================================================================

def test_05_persistence(temp_cache_dir):
    db_path = os.path.join(temp_cache_dir, "persist.db")

    cache1 = LLMCache(enabled=True, db_path=db_path)
    messages = make_messages("persist-test")

    cache1.set(MODEL, messages, "value123")
    cache1.close()

    # New instance
    cache2 = LLMCache(enabled=True, db_path=db_path)

    assert cache2.get(MODEL, messages) == "value123"
    cache2.close()


# ===========================================================================
# 6. Cache disabled
# ===========================================================================

def test_06_cache_disabled(temp_cache_dir, mock_llm):
    db_path = os.path.join(temp_cache_dir, "disabled.db")

    cache = LLMCache(enabled=False, db_path=db_path)

    messages = make_messages("no-cache")

    assert cache.get(MODEL, messages) is None

    cache.set(MODEL, messages, "value")

    # Still None because disabled
    assert cache.get(MODEL, messages) is None

    cache.close()


# ===========================================================================
# 7. Memory layer works (no duplicate disk hit)
# ===========================================================================

def test_07_memory_layer(cache, mock_llm):
    prompt = "memory-test"
    messages = make_messages(prompt)

    response = mock_llm.query(prompt)
    cache.set(MODEL, messages, response)

    # First get → loads from disk into memory
    r1 = cache.get(MODEL, messages)

    # Second get → should hit memory
    r2 = cache.get(MODEL, messages)

    assert r1 == r2 == response
    assert mock_llm.calls == 1


# ===========================================================================
# 8. Hash stability (same input → same key)
# ===========================================================================

def test_08_hash_stability(cache):
    m1 = make_messages("same")
    m2 = make_messages("same")

    key1 = cache._make_key(MODEL, m1)
    key2 = cache._make_key(MODEL, m2)

    assert key1 == key2


# ===========================================================================
# 9. Hash difference (different input → different key)
# ===========================================================================

def test_09_hash_difference(cache):
    m1 = make_messages("a")
    m2 = make_messages("b")

    key1 = cache._make_key(MODEL, m1)
    key2 = cache._make_key(MODEL, m2)

    assert key1 != key2