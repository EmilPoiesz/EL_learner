from typing import Optional

from .hashing import stable_hash
from .backend import SQLiteCacheBackend


class LLMCache:
    def __init__(
        self,
        enabled: bool = True,
        db_path: str = "llm_cache.db",
        store_prompts: bool = False,
    ):
        self.enabled = enabled
        self.store_prompts = store_prompts
        self.backend = SQLiteCacheBackend(db_path) if enabled else None
        self.memory_cache: dict[str, str] = {}

    def _make_key(self, model: str, messages: list[dict], params: Optional[dict] = None) -> str:
        # ``params`` holds the generation settings (e.g. max_new_tokens): the same
        # prompt under a different budget can give a different response, so it
        # must not share a cache entry.  Without params the key is as before.
        payload = {"model": model, "messages": messages}
        if params is not None:
            payload["params"] = params
        return stable_hash(payload)

    def get(self, model: str, messages: list[dict], params: Optional[dict] = None) -> Optional[str]:
        if not self.enabled:
            return None

        key = self._make_key(model, messages, params)

        # 1. memory
        if key in self.memory_cache:
            return self.memory_cache[key]

        # 2. disk
        value = self.backend.get(key)
        if value is not None:
            self.memory_cache[key] = value
        return value

    def set(self, model: str, messages: list[dict], value: str, params: Optional[dict] = None):
        if not self.enabled:
            return

        key = self._make_key(model, messages, params)

        self.memory_cache[key] = value

        prompt_str = None
        if self.store_prompts:
            prompt_str = str(messages)

        self.backend.set(key, value, prompt=prompt_str)

    def close(self):
        if self.backend:
            self.backend.close()