"""Thin CLI over the pipeline and retriever. No HTML viz (MVP non-goal)."""

from __future__ import annotations

from pathlib import Path

import typer

from .embed import get_embedder
from .pipeline import index_path
from .retrieve import Retriever
from .store import DEFAULT_DIM, Store

app = typer.Typer(add_completion=False, help="Cartograph — local-first hybrid code retrieval.")

DEFAULT_DB = "cartograph-out/graph.kuzu"


@app.command()
def index(
    path: Path = typer.Argument(..., help="Python file or directory to index."),
    db: str = typer.Option(DEFAULT_DB, help="Kuzu DB path."),
    embedder: str = typer.Option(None, help="Embedder backend: hash | ollama."),
) -> None:
    """Parse → embed → store a codebase into the graph."""
    store = index_path(path, Path(db), dim=DEFAULT_DIM, embedder=get_embedder(embedder), overwrite=True)
    counts = store.counts()
    store.close()
    typer.echo(f"Indexed {path} -> {db}")
    for kkey, v in sorted(counts.items()):
        typer.echo(f"  {kkey}: {v}")


@app.command()
def query(
    text: str = typer.Argument(..., help="Natural-language or symbol query."),
    db: str = typer.Option(DEFAULT_DB, help="Kuzu DB path."),
    mode: str = typer.Option("hybrid", help="vector | graph | lexical | hybrid"),
    k: int = typer.Option(10, help="Top-k results."),
    embedder: str = typer.Option(None, help="Embedder backend: hash | ollama."),
) -> None:
    """Query the graph. Default mode runs the hybrid (RRF) fusion."""
    store = Store(db)
    retriever = Retriever(store, embedder=get_embedder(embedder))
    hits = retriever.retrieve(text, mode=mode, k=k)
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
    store = Store(db)
    for kkey, v in sorted(store.counts().items()):
        typer.echo(f"{kkey}: {v}")
    store.close()


if __name__ == "__main__":
    app()
