"""Read-only query service over an indexed graph.

This is the logic the MCP server (and CLI) expose. It has no MCP dependency so it
stays unit-testable offline. It auto-selects the query-time embedder from the
metadata recorded at index time, so it can't silently mismatch (hash query over
Ollama vectors). Retrieval reads only the graph — never source files.
"""

from __future__ import annotations

from pathlib import Path

from .embed import get_embedder
from .retrieve import Retriever
from .store import DEFAULT_DIM, Store


def embedder_from_store(store: Store):
    """Rebuild the embedder that was used to index this store (falls back to env/hash)."""
    backend = store.get_meta("embedder_backend")
    model = store.get_meta("embedder_model") or None
    dim = store.get_meta("embedding_dim")
    dim = int(dim) if dim else DEFAULT_DIM
    if backend:
        return get_embedder(backend, dim=dim, model=model)
    return None  # let Retriever infer (env or hash)


class CartographService:
    """Opens a graph once and answers structural/semantic queries."""

    def __init__(self, db_path: str | Path, embedder=None):
        if not Path(db_path).exists():
            raise FileNotFoundError(f"no graph at {db_path}; run `cartograph index` first")
        self.store = Store(db_path)
        if embedder is None:
            embedder = embedder_from_store(self.store)
        self.retriever = Retriever(self.store, embedder=embedder)
        # `rerank` is only offered if the retriever supports it (M2 reranker, PR #6).
        self.modes = {"vector", "graph", "lexical", "hybrid"}
        if hasattr(self.retriever, "reranked"):
            self.modes.add("rerank")

    def close(self) -> None:
        self.store.close()

    def _node(self, node_id: str, score: float | None = None) -> dict | None:
        n = self.store.get_node(node_id)
        if n is None:
            return None
        doc = (n.get("docstring") or "")
        out = {
            "id": n["id"],
            "kind": n["kind"],
            "name": n["name"],
            "qualified_name": n["qualified_name"],
            "file_path": n["file_path"],
            "start_line": n["start_line"],
            "signature": n["signature"],
            "docstring": doc[:500],
        }
        if score is not None:
            out["score"] = round(float(score), 4)
        return out

    # -- tools ----------------------------------------------------------------
    def query(self, text: str, mode: str = "hybrid", k: int = 10) -> list[dict]:
        """Hybrid (default) retrieval; returns ranked nodes with scores."""
        if mode not in self.modes:
            raise ValueError(f"mode must be one of {sorted(self.modes)}")
        hits = self.retriever.retrieve(text, mode=mode, k=k)
        return [n for n in (self._node(i, s) for i, s in hits) if n]

    def semantic_search(self, text: str, k: int = 10) -> list[dict]:
        """Pure vector-ANN search over node embeddings."""
        return self.query(text, mode="vector", k=k)

    def get_node(self, node_id: str) -> dict | None:
        """Full detail for one node by id."""
        return self._node(node_id)

    def neighbors(self, node_id: str, hops: int = 1) -> list[dict]:
        """Nodes within `hops` edges (calls/inheritance/imports/containment)."""
        return [n for n in (self._node(i) for i in self.store.neighbors(node_id, hops=hops)) if n]

    def shortest_path(self, src: str, dst: str) -> list[dict]:
        """Ordered nodes on a shortest path between two node ids ([] if none)."""
        return [n for n in (self._node(i) for i in self.store.shortest_path(src, dst)) if n]

    def stats(self) -> dict[str, int]:
        return self.store.counts()
