"""Thin CLI over the pipeline and retriever. No HTML viz (MVP non-goal)."""

from __future__ import annotations

import os
from pathlib import Path

import typer

from .embed import get_embedder
from .pipeline import index_path, update_index
from .retrieve import Retriever
from .store import DEFAULT_DIM, Store

app = typer.Typer(add_completion=False, help="Cartograph — local-first hybrid code retrieval.")

DEFAULT_DB = os.environ.get("CARTOGRAPH_DB", "cartograph-out/graph.kuzu")
QUERY_MODES = ("vector", "graph", "lexical", "hybrid")


def _open_graph_or_exit(db: str) -> Store:
    """Open for reading with the service's friendly failures instead of tracebacks
    (a bare Store(db) would silently create an empty DB at a mistyped path)."""
    from .service import open_graph

    try:
        return open_graph(db)
    except (FileNotFoundError, RuntimeError) as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(1)


@app.command()
def index(
    path: Path = typer.Argument(..., help="Python file or directory to index."),
    db: str = typer.Option(DEFAULT_DB, help="Kuzu DB path."),
    embedder: str = typer.Option(None, help="Embedder backend: hash | ollama."),
    no_cache: bool = typer.Option(False, "--no-cache", help="Ignore the embedding cache; re-embed all."),
    resolver: str = typer.Option("heuristic", help="Call resolver: heuristic | jedi (needs --extra resolve)."),
) -> None:
    """Parse → embed → store a codebase into the graph (re-embeds only changed symbols)."""
    try:
        store = index_path(path, Path(db), dim=DEFAULT_DIM, embedder=get_embedder(embedder),
                           overwrite=True, use_cache=not no_cache, resolver=resolver)
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(1)
    counts = store.counts()
    reused, embedded = getattr(store, "cache_stats", (0, 0))
    store.close()
    typer.echo(f"Indexed {path} -> {db}")
    typer.echo(f"  embeddings: {embedded} computed, {reused} reused from cache")
    for kkey, v in sorted(counts.items()):
        typer.echo(f"  {kkey}: {v}")


@app.command()
def update(
    path: Path = typer.Argument(..., help="Python/SQL file or directory to re-index."),
    db: str = typer.Option(DEFAULT_DB, help="Kuzu DB path."),
    embedder: str = typer.Option(None, help="Embedder backend: hash | ollama."),
    resolver: str = typer.Option("heuristic", help="Call resolver: heuristic | jedi."),
) -> None:
    """Incremental re-index: instant no-op when nothing changed; otherwise a
    cache-accelerated rebuild (re-embeds only changed symbols, no stale edges)."""
    summary = update_index(path, Path(db), dim=DEFAULT_DIM,
                           embedder=get_embedder(embedder) if embedder else None, resolver=resolver)
    typer.echo(f"{summary['status']}: {path} -> {db}")
    if summary["status"] in ("updated", "indexed", "rebuilt"):
        typer.echo(f"  changed={len(summary['changed'])} deleted={len(summary['deleted'])} "
                   f"| nodes {summary.get('created', 0)} recreated, {summary.get('removed', 0)} removed "
                   f"| embeddings {summary.get('embedded', 0)} computed, {summary.get('reused', 0)} reused")
    elif summary["status"] == "up-to-date":
        typer.echo("  nothing changed — no re-embed, no rebuild")


@app.command()
def query(
    text: str = typer.Argument(..., help="Natural-language or symbol query."),
    db: str = typer.Option(DEFAULT_DB, help="Kuzu DB path."),
    mode: str = typer.Option("hybrid", help="vector | graph | lexical | hybrid"),
    k: int = typer.Option(10, help="Top-k results."),
    embedder: str = typer.Option(None, help="Embedder backend: hash | ollama."),
) -> None:
    """Query the graph. Default mode runs the hybrid (RRF) fusion."""
    if mode not in QUERY_MODES:
        typer.echo(f"unknown mode {mode!r}; choose from: {', '.join(QUERY_MODES)}", err=True)
        raise typer.Exit(2)
    store = _open_graph_or_exit(db)
    # Auto-detect the embedder recorded at index time unless overridden, so the query
    # embeds with the same model the graph was built with (no hash-vs-ollama mismatch).
    from .service import embedder_from_store

    emb = get_embedder(embedder) if embedder else embedder_from_store(store)
    retriever = Retriever(store, embedder=emb)
    try:
        hits = retriever.retrieve(text, mode=mode, k=k)
    except RuntimeError as e:  # e.g. Ollama unreachable — message is already actionable
        typer.echo(str(e), err=True)
        raise typer.Exit(1)
    if not hits:
        typer.echo("(no results)")
    for rank, (node_id, score) in enumerate(hits, 1):
        node = store.get_node(node_id)
        label = node["qualified_name"] if node else node_id
        kind = node["kind"] if node else "?"
        typer.echo(f"{rank:2}. [{kind}] {label}  ({score:.4f})")
    store.close()


@app.command()
def demo(
    path: Path = typer.Argument(..., help="Python file to run the M0 vertical slice on."),
) -> None:
    """M0 slice: index one file, then answer one query via BOTH vector and graph."""
    store = index_path(path, Path("cartograph-out/demo.kuzu"), overwrite=True)
    retriever = Retriever(store)
    typer.echo("== vector ==")
    for nid, s in retriever.vector("function", k=3):
        typer.echo(f"  {nid}  ({s:.3f})")
    typer.echo("== graph (2-hop) ==")
    for nid, s in retriever.graph("function", k=3):
        typer.echo(f"  {nid}  ({s:.3f})")
    store.close()


@app.command()
def serve(db: str = typer.Option(DEFAULT_DB, help="Kuzu DB path to serve.")) -> None:
    """Run the MCP server (stdio) exposing the graph to coding agents.

    Requires the optional `mcp` extra: `uv sync --extra mcp`.
    """
    import os

    os.environ.setdefault("CARTOGRAPH_DB", db)
    from .mcp_server import main as serve_main

    try:
        serve_main()
    except ModuleNotFoundError:  # pragma: no cover - mcp extra absent
        typer.echo("MCP SDK not installed. Run: uv sync --extra mcp", err=True)
        raise typer.Exit(1)


@app.command()
def stats(db: str = typer.Option(DEFAULT_DB, help="Kuzu DB path.")) -> None:
    """Print node/edge counts for an indexed graph."""
    store = _open_graph_or_exit(db)
    for kkey, v in sorted(store.counts().items()):
        typer.echo(f"{kkey}: {v}")
    store.close()


if __name__ == "__main__":
    app()
