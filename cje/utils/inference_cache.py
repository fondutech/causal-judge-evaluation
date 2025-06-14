"""Lightweight SQLite-backed cache for LLM generations & log-probs.

This is **not** a full-blown vector database – it's a simple key-value store
that maps `(model, prompt, response)` → value where *value* is either:

• The generation dict from `generate_with_logprobs`
• A float representing `sequence_logp`

The cache lives in ``.cache/llm_cache.sqlite`` by default (override with the
``CJE_CACHE_DIR`` environment variable or pass an explicit path).
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Optional, Tuple, Union
import threading
import os

_DEFAULT_CACHE = Path(os.getenv("CJE_CACHE_DIR", ".cache/llm_cache.sqlite"))
_LOCK = threading.Lock()

_SCHEMA = """
CREATE TABLE IF NOT EXISTS kv (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


def _hash_key(parts: Tuple[Any, ...]) -> str:
    """Create a deterministic string key from tuple parts."""
    import hashlib, json as _json

    flat = _json.dumps(parts, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(flat.encode()).hexdigest()


class LLMCache:
    """Very small wrapper around SQLite for JSON-serialisable blobs."""

    def __init__(self, path: Optional[Union[Path, str]] = None):
        self.path = Path(path or _DEFAULT_CACHE)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path)
        self._conn.execute(_SCHEMA)
        self._conn.commit()

    def get(self, *parts: Any) -> Optional[Any]:
        key = _hash_key(tuple(parts))
        with _LOCK:
            cursor = self._conn.execute("SELECT value FROM kv WHERE key = ?", (key,))
            row = cursor.fetchone()
            if row:
                return json.loads(row[0])
        return None

    def set(self, value: Any, *parts: Any) -> None:
        key = _hash_key(tuple(parts))
        blob = json.dumps(value, ensure_ascii=False)
        with _LOCK:
            self._conn.execute(
                "INSERT OR REPLACE INTO kv (key, value) VALUES (?, ?)", (key, blob)
            )
            self._conn.commit()

    def close(self) -> None:
        with _LOCK:
            self._conn.close()
