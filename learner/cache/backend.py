import sqlite3
from typing import Optional
from learner.cache import PATH as CACHE_PATH


class SQLiteCacheBackend:
    def __init__(self, path: str = CACHE_PATH / "llm_cache.db"):
        self.conn = sqlite3.connect(path)
        self._init_db()

    def _init_db(self):
        cur = self.conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS cache (
                key TEXT PRIMARY KEY,
                value TEXT
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS prompts (
                key TEXT PRIMARY KEY,
                prompt TEXT
            )
        """)
        self.conn.commit()

    def get(self, key: str) -> Optional[str]:
        cur = self.conn.cursor()
        cur.execute("SELECT value FROM cache WHERE key = ?", (key,))
        row = cur.fetchone()
        return row[0] if row else None

    def set(self, key: str, value: str, prompt: Optional[str] = None):
        cur = self.conn.cursor()
        cur.execute(
            "INSERT OR REPLACE INTO cache (key, value) VALUES (?, ?)",
            (key, value),
        )
        if prompt is not None:
            cur.execute(
                "INSERT OR REPLACE INTO prompts (key, prompt) VALUES (?, ?)",
                (key, prompt),
            )
        self.conn.commit()

    def close(self):
        self.conn.close()