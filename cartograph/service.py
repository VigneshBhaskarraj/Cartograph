"""Read-only query service over an indexed graph.

This is the logic the MCP server (and CLI) expose. It has no MCP dependency so it
stays unit-testable offline. It auto-selects the query-time embedder from the
metadata recorded at index time, so it can't silently mismatch (hash query over
Ollama vectors). Retrieval reads only the graph — never source files.
"""

from __future__ import annotations

import os
from pathlib import Path

from .embed import get_embedder
from .model import EDGE_TYPES
from .retrieve import Retriever
from .store import DEFAULT_DIM, SCHEMA_VERSION, Store

REQUIRED_TABLES = {"CodeNode", *EDGE_TYPES}


def open_graph(db_path: str | Path, read_only: bool = True) -> Store:
    """Open an existing graph with friendly failures: a missing path must not
    silently create an empty DB, and a graph built by an older/newer Cartograph
    (missing rel tables, other schema_version) must say so, not crash mid-query."""
    p = Path(db_path)
    if not p.exists():
        raise FileNotFoundError(f"no graph at {p}; run `cartograph index <path> --db {p}` first")
    store = Store(p, read_only=read_only)
    missing = REQUIRED_TABLES - store.table_names()
    version = store.get_meta("schema_version") if not missing else None
    if missing or (version is not None and version != SCHEMA_VERSION):
        store.close()
        why = (f"missing tables: {', '.join(sorted(missing))}" if missing
               else f"schema_version {version} != {SCHEMA_VERSION}")
        raise RuntimeError(
            f"graph at {p} was built by an incompatible Cartograph ({why}); "
            "re-run `cartograph index`")
    return store


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

    def __init__(self, db_path: str | Path, embedder=None, reranker=None):
        # Read-only: queries never write, accidental writes become errors, and
        # several read-only servers can share one graph. (It does NOT allow a
        # concurrent writer — Kuzu is multi-reader OR single-writer; see docs/mcp.md.)
        self.store = open_graph(db_path, read_only=True)
        if embedder is None:
            embedder = embedder_from_store(self.store)
        # Build a reranker from env (CARTOGRAPH_RERANKER=ollama, CARTOGRAPH_RERANK_MODEL=...)
        # so `rerank` mode actually reranks instead of silently degrading to hybrid.
        if reranker is None and os.environ.get("CARTOGRAPH_RERANKER"):
            from .rerank import get_reranker
            reranker = get_reranker()
        self.retriever = Retriever(self.store, embedder=embedder, reranker=reranker)
        self.modes = {"vector", "graph", "lexical", "hybrid"}
        # Only advertise `rerank` when a reranker is actually wired in.
        if reranker is not None and hasattr(self.retriever, "reranked"):
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
        k = max(1, min(int(k), 100))  # k<=0 would slice wrong; huge k is an agent typo
        hits = self.retriever.retrieve(text, mode=mode, k=k)
        return [n for n in (self._node(i, s) for i, s in hits) if n]

    def semantic_search(self, text: str, k: int = 10) -> list[dict]:
        """Pure vector-ANN search over node embeddings."""
        return self.query(text, mode="vector", k=k)

    def _rid(self, ref: str) -> str | None:
        """Resolve a user-supplied reference (id, qualified name, or bare name) to the
        most-direct node id."""
        ids = self.store.resolve_ids(ref)
        return ids[0] if ids else None

    def get_node(self, ref: str) -> dict | None:
        """Full detail for a node by id or qualified name (e.g. httpx._client.Client.send)."""
        rid = self._rid(ref)
        return self._node(rid) if rid else None

    def resolve(self, ref: str) -> list[dict]:
        """All nodes a reference matches (use to disambiguate a bare name)."""
        return [n for n in (self._node(i) for i in self.store.resolve_ids(ref)) if n]

    def neighbors(self, node_id: str, direction: str = "both", relation: str | None = None,
                  hops: int = 1) -> list[dict]:
        """Adjacent nodes. At hops=1 each result is labeled with `relation`
        (CALLS/INHERITS/IMPORTS/CONTAINS/DOCUMENTS) and `direction` (out=this node is
        the source, in=this node is the target). `direction` and `relation` filter the
        edges. hops>1 expands undirected/unlabeled for broad context. Accepts an id or
        qualified name."""
        rid = self._rid(node_id)
        if rid is None:
            return []
        if hops <= 1:
            types = [relation.upper()] if relation else None
            out = []
            for rel in self.store.relations(rid, direction=direction, types=types):
                node = self._node(rel["id"])
                if node:
                    out.append({**node, "relation": rel["relation"], "direction": rel["direction"]})
            return out
        return [n for n in (self._node(i) for i in self.store.neighbors(rid, hops=hops)) if n]

    def calls(self, node_id: str) -> list[dict]:
        """What this node calls (outgoing CALLS edges). Accepts an id or qualified name."""
        return self.neighbors(node_id, direction="out", relation="CALLS")

    def callers(self, node_id: str) -> list[dict]:
        """What calls this node (incoming CALLS edges). Accepts an id or qualified name."""
        return self.neighbors(node_id, direction="in", relation="CALLS")

    def shortest_path(self, src: str, dst: str) -> list[dict]:
        """Ordered nodes on a shortest path between two nodes (ids or qualified names)."""
        s, d = self._rid(src), self._rid(dst)
        if s is None or d is None:
            return []
        return [n for n in (self._node(i) for i in self.store.shortest_path(s, d)) if n]

    def stats(self) -> dict[str, int]:
        return self.store.counts()
