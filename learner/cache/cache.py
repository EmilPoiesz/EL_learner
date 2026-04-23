from learner.cache import PATH as CACHE_PATH
from typing import Optional

from .hashing import stable_hash
from .backend import SQLiteCacheBackend


class LLMCache:
    def __init__(
        self,
        enabled: bool = True,
        db_path: str = CACHE_PATH / "llm_cache.db",
        store_prompts: bool = False,
    ):
        self.enabled = enabled
        self.store_prompts = store_prompts
        self.backend = SQLiteCacheBackend(db_path) if enabled else None
        self.memory_cache: dict[str, str] = {}

    def _make_key(self, model: str, messages: list[dict]) -> str:
        return stable_hash({
            "model": model,
            "messages": messages,
        })

    def get(self, model: str, messages: list[dict]) -> Optional[str]:
        if not self.enabled:
            return None

        key = self._make_key(model, messages)

        # 1. memory
        if key in self.memory_cache:
            return self.memory_cache[key]

        # 2. disk
        value = self.backend.get(key)
        if value is not None:
            self.memory_cache[key] = value
        return value

    def set(self, model: str, messages: list[dict], value: str):
        if not self.enabled:
            return

        key = self._make_key(model, messages)

        self.memory_cache[key] = value

        prompt_str = None
        if self.store_prompts:
            prompt_str = str(messages)

        self.backend.set(key, value, prompt=prompt_str)

    def close(self):
        if self.backend:
            self.backend.close()