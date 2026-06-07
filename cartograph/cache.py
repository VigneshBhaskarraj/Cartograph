"""Content-hash embedding cache for incremental re-indexing.

Embedding (especially via Ollama) is the slow part of indexing. We key cached
vectors by a SHA256 of the exact `embed_text`, so re-indexing a repo only embeds
symbols whose embedded text actually changed — unchanged ones are reused. Stored as
JSON under the (gitignored) cache dir, namespaced per embedder so hash/Ollama models
never collide. Correctness of edge deletion is handled by the full graph rebuild;
this cache only removes the recompute cost.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path


def embed_key(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()


def _safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", name)


class EmbeddingCache:
    """A `{sha256(embed_text): vector}` store, persisted as JSON."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.data: dict[str, list[float]] = {}

    @classmethod
    def load(cls, path: Path) -> "EmbeddingCache":
        c = cls(path)
        if c.path.exists():
            try:
                c.data = json.loads(c.path.read_text())
            except (json.JSONDecodeError, OSError):
                c.data = {}
        return c

    @classmethod
    def for_embedder(cls, cache_dir: Path, embedder_name: str) -> "EmbeddingCache":
        return cls.load(Path(cache_dir) / f"emb.{_safe(embedder_name)}.json")

    def get(self, text: str, dim: int | None = None) -> list[float] | None:
        v = self.data.get(embed_key(text))
        if v is None:
            return None
        if dim is not None and len(v) != dim:
            return None  # model/dim changed under the same name; recompute
        return v

    def put(self, text: str, vector: list[float]) -> None:
        self.data[embed_key(text)] = vector

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data))
