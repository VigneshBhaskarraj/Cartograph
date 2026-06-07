"""End-to-end indexing: source path -> extract -> embed -> Kuzu store."""

from __future__ import annotations

from pathlib import Path

from .cache import EmbeddingCache
from .embed import get_embedder
from .extract import extract_paths
from .model import Graph
from .store import DEFAULT_DIM, Store


def _python_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(p for p in path.rglob("*.py") if "__pycache__" not in p.parts)


def build_graph(path: Path) -> Graph:
    files = _python_files(path)
    if not files:
        raise FileNotFoundError(f"no Python files under {path}")
    return extract_paths(files, root=path)


def embed_graph(graph: Graph, embedder=None, cache: EmbeddingCache | None = None) -> tuple[int, int]:
    """Embed every node, reusing `cache` for unchanged embed_text. Returns (reused, embedded)."""
    embedder = embedder or get_embedder()
    dim = getattr(embedder, "dim", None)
    pending_idx, pending_text = [], []
    reused = 0
    for i, n in enumerate(graph.nodes):
        cached = cache.get(n.embed_text, dim) if cache is not None else None
        if cached is not None:
            n.embedding = cached
            reused += 1
        else:
            pending_idx.append(i)
            pending_text.append(n.embed_text)
    if pending_text:
        vectors = embedder.embed_batch(pending_text)
        for i, v in zip(pending_idx, vectors):
            graph.nodes[i].embedding = v
            if cache is not None:
                cache.put(graph.nodes[i].embed_text, v)
    return reused, len(pending_text)


def index_path(path: Path, db_path: Path, dim: int = DEFAULT_DIM, embedder=None, overwrite: bool = True,
               use_cache: bool = True) -> Store:
    """Parse → embed → store. Returns the open Store.

    The Kuzu vector column is sized to the embedder's *actual* output dimension, so
    swapping models (hash 768, nomic 768, mxbai 1024, …) just works without touching
    the schema. A content-hash embedding cache (under <db_dir>/cache) makes re-indexing
    a changed repo fast: only changed symbols are re-embedded.
    """
    embedder = embedder or get_embedder(dim=dim)
    graph = build_graph(path)
    cache = None
    if use_cache:
        cache = EmbeddingCache.for_embedder(Path(db_path).parent / "cache", getattr(embedder, "name", "hash"))
    reused, embedded = embed_graph(graph, embedder=embedder, cache=cache)
    if cache is not None:
        cache.save()
    actual_dim = len(graph.nodes[0].embedding) if graph.nodes and graph.nodes[0].embedding else dim
    store = Store.create(db_path, dim=actual_dim, overwrite=overwrite)
    store.cache_stats = (reused, embedded)  # surfaced to the CLI for reporting
    store.load(graph, dim=actual_dim)
    # Record the embedder so a reader (CLI query, MCP server) reconstructs a matching one.
    name = getattr(embedder, "name", "hash")
    backend, _, model = name.partition(":")
    store.set_meta("embedder_backend", backend)
    store.set_meta("embedder_model", model)
    store.set_meta("embedding_dim", str(actual_dim))
    return store
