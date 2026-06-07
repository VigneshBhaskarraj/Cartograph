"""End-to-end indexing: source path -> extract -> embed -> Kuzu store."""

from __future__ import annotations

from pathlib import Path

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


def embed_graph(graph: Graph, embedder=None) -> None:
    embedder = embedder or get_embedder()
    texts = [n.embed_text for n in graph.nodes]
    vectors = embedder.embed_batch(texts)
    for n, v in zip(graph.nodes, vectors):
        n.embedding = v


def index_path(path: Path, db_path: Path, dim: int = DEFAULT_DIM, embedder=None, overwrite: bool = True) -> Store:
    """Parse → embed → store. Returns the open Store.

    The Kuzu vector column is sized to the embedder's *actual* output dimension, so
    swapping models (hash 768, nomic 768, mxbai 1024, …) just works without touching
    the schema.
    """
    embedder = embedder or get_embedder(dim=dim)
    graph = build_graph(path)
    embed_graph(graph, embedder=embedder)
    actual_dim = len(graph.nodes[0].embedding) if graph.nodes and graph.nodes[0].embedding else dim
    store = Store.create(db_path, dim=actual_dim, overwrite=overwrite)
    store.load(graph, dim=actual_dim)
    # Record the embedder so a reader (CLI query, MCP server) reconstructs a matching one.
    name = getattr(embedder, "name", "hash")
    backend, _, model = name.partition(":")
    store.set_meta("embedder_backend", backend)
    store.set_meta("embedder_model", model)
    store.set_meta("embedding_dim", str(actual_dim))
    return store
