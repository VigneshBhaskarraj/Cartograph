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
    try:
        summary = update_index(path, Path(db), dim=DEFAULT_DIM,
                               embedder=get_embedder(embedder) if embedder else None, resolver=resolver)
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(1)
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

    try:
        emb = get_embedder(embedder) if embedder else embedder_from_store(store)
        retriever = Retriever(store, embedder=emb)
        hits = retriever.retrieve(text, mode=mode, k=k)
    except (ValueError, RuntimeError) as e:
        # bad --embedder name, dim mismatch, Ollama unreachable, non-loopback host —
        # every message is already actionable; don't bury it in a traceback.
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


# -- structural commands ------------------------------------------------------
# The same graph surface the MCP server exposes, but over the shell: any agent
# that can run a command can use the graph — no MCP wiring required.

def _service_or_exit(db: str):
    from .service import CartographService

    try:
        return CartographService(db)
    except (FileNotFoundError, RuntimeError) as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(1)


def _echo_nodes(nodes: list[dict], empty_msg: str) -> None:
    if not nodes:
        typer.echo(empty_msg)
        return
    for n in nodes:
        rel = f" <{n['relation']}:{n['direction']}>" if "relation" in n else ""
        typer.echo(f"[{n['kind']}] {n['qualified_name']}{rel}  ({n['file_path']}:{n['start_line']})")
        if n.get("signature"):
            typer.echo(f"    {n['signature']}")


@app.command()
def node(
    ref: str = typer.Argument(..., help="Node id, qualified name (a.b.C.m), or bare name."),
    db: str = typer.Option(DEFAULT_DB, help="Kuzu DB path."),
) -> None:
    """Full detail for one symbol (docstring included). Use `resolve` for ambiguity."""
    svc = _service_or_exit(db)
    n = svc.get_node(ref)
    if n is None:
        typer.echo(f"no node matches {ref!r}", err=True)
        svc.close()
        raise typer.Exit(1)
    for key in ("kind", "qualified_name", "file_path", "start_line", "signature"):
        typer.echo(f"{key}: {n[key]}")
    if n.get("docstring"):
        typer.echo(f"docstring: {n['docstring']}")
    svc.close()


@app.command()
def resolve(
    ref: str = typer.Argument(..., help="Qualified name or bare name to disambiguate."),
    db: str = typer.Option(DEFAULT_DB, help="Kuzu DB path."),
) -> None:
    """All symbols a reference could mean (then use the qualified name you want)."""
    svc = _service_or_exit(db)
    _echo_nodes(svc.resolve(ref), f"no node matches {ref!r}")
    svc.close()


def _known_ref_or_exit(svc, ref: str) -> None:
    """An unknown symbol must error, not read as 'known symbol, empty result' —
    agents act on that difference."""
    if svc.get_node(ref) is None:
        typer.echo(f"no node matches {ref!r} (try `cartograph resolve`)", err=True)
        svc.close()
        raise typer.Exit(1)


@app.command()
def calls(
    ref: str = typer.Argument(..., help="The caller: node id, qualified name, or bare name."),
    db: str = typer.Option(DEFAULT_DB, help="Kuzu DB path."),
) -> None:
    """What this symbol calls (outgoing CALLS edges)."""
    svc = _service_or_exit(db)
    _known_ref_or_exit(svc, ref)
    _echo_nodes(svc.calls(ref), f"{ref!r} calls nothing the graph knows about")
    svc.close()


@app.command()
def callers(
    ref: str = typer.Argument(..., help="The callee: node id, qualified name, or bare name."),
    db: str = typer.Option(DEFAULT_DB, help="Kuzu DB path."),
) -> None:
    """What calls this symbol (incoming CALLS edges)."""
    svc = _service_or_exit(db)
    _known_ref_or_exit(svc, ref)
    _echo_nodes(svc.callers(ref), f"nothing in the graph calls {ref!r}")
    svc.close()


@app.command()
def impact(
    ref: str = typer.Argument(..., help="A table, table.column, or code symbol."),
    db: str = typer.Option(DEFAULT_DB, help="Kuzu DB path."),
) -> None:
    """Blast radius across the code<->data bridge: for a column/table, every
    function that can reach it ("what breaks if I drop users.email"); for code,
    every table/column it can touch. Offline, from the graph alone."""
    svc = _service_or_exit(db)
    result = svc.impact(ref)
    if result is None:
        typer.echo(f"no node matches {ref!r} (try `cartograph resolve`)", err=True)
        svc.close()
        raise typer.Exit(1)
    t = result["target"]
    typer.echo(f"[{t['kind']}] {t['qualified_name']}  ({result['direction']})")
    if result["direction"] == "data->code":
        shown_d, shown_t = len(result["direct_code"]), len(result["transitive_callers"])
        typer.echo(f"\ndirectly touched by ({shown_d}):")
        _echo_nodes(result["direct_code"], "  (nothing touches it directly)")
        typer.echo(f"\nreached through scope/callers ({shown_t}):")
        _echo_nodes(result["transitive_callers"], "  (no transitive callers)")
        total = result["total_code_paths"]
        suffix = f" (showing {shown_d + shown_t} of {total})" if result["truncated"] else ""
        typer.echo(f"\ntotal code paths affected: {total}{suffix}")
    else:
        shown_d, shown_t = len(result["direct_data"]), len(result["transitive_data"])
        typer.echo(f"\ntouches directly ({shown_d}):")
        _echo_nodes(result["direct_data"], "  (no direct data access)")
        typer.echo(f"\ntouches through callees ({shown_t}):")
        _echo_nodes(result["transitive_data"], "  (none)")
        total = result["total_data_touched"]
        suffix = f" (showing {shown_d + shown_t} of {total})" if result["truncated"] else ""
        typer.echo(f"\ntotal tables/columns touched: {total}{suffix}")
    typer.echo("\nnote: the CALLS expansion over-approximates (INFERRED edges); the bridge can miss"
               "\nORM attribute access, and FK/JOIN ripple between tables is not followed.")
    svc.close()


@app.command()
def path(
    src: str = typer.Argument(..., help="Start symbol (id or qualified name)."),
    dst: str = typer.Argument(..., help="End symbol (id or qualified name)."),
    db: str = typer.Option(DEFAULT_DB, help="Kuzu DB path."),
) -> None:
    """Shortest connection between two symbols — how does A reach B?"""
    svc = _service_or_exit(db)
    nodes = svc.shortest_path(src, dst)
    if not nodes:
        typer.echo(f"no path between {src!r} and {dst!r}")
    for i, n in enumerate(nodes):
        typer.echo(f"{i + 1}. [{n['kind']}] {n['qualified_name']}")
    svc.close()


@app.command()
def viz(
    db: str = typer.Option(DEFAULT_DB, help="Kuzu DB path."),
    out: str = typer.Option("cartograph-out/graph.html", help="Output HTML file."),
    title: str = typer.Option(None, help="Page title (default: DB name)."),
    iterations: int = typer.Option(200, help="Force-layout iterations (more = nicer, slower)."),
) -> None:
    """Export an interactive 3D map of the graph to a single offline HTML file.

    A viewer, never a retrieval path: rotate/zoom/search, click a symbol for its
    neighborhood, trace shortest paths, filter edge types and EXTRACTED/INFERRED.
    The file is fully self-contained — no CDN, no network calls, shareable as-is.
    """
    from .viz import write_viz

    try:
        summary = write_viz(db, out, title=title, iterations=iterations)
    except (FileNotFoundError, RuntimeError, ValueError) as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(1)
    typer.echo(f"Wrote {summary['out']} ({summary['nodes']} nodes, "
               f"{summary['links']} links, {summary['bytes'] // 1024} KB)")
    typer.echo("Open it in a browser — everything runs locally.")


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
    # Plain assignment: an inherited CARTOGRAPH_DB must not silently override an
    # explicit --db (the flag's default already comes from the env at startup).
    os.environ["CARTOGRAPH_DB"] = db
    from .mcp_server import main as serve_main

    try:
        serve_main()
    except ModuleNotFoundError:  # pragma: no cover - mcp extra absent
        typer.echo("MCP SDK not installed. Run: uv sync --extra mcp", err=True)
        raise typer.Exit(1)
    except (FileNotFoundError, RuntimeError) as e:  # missing/old graph — say so cleanly
        typer.echo(str(e), err=True)
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
