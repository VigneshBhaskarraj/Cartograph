"""End-to-end indexing: source path -> extract -> embed -> Kuzu store."""

from __future__ import annotations

import hashlib
from pathlib import Path

from .cache import EmbeddingCache
from .embed import get_embedder
from .extract import extract_paths
from .model import Graph
from .store import DEFAULT_DIM, SCHEMA_VERSION, Store


# Indexing a project root must not sweep up its virtualenv, vendored deps, or VCS
# internals — embedding a .venv costs hours of Ollama time and poisons retrieval.
SKIP_DIRS = {"__pycache__", ".git", ".hg", ".svn", ".venv", "venv", ".env", "env",
             "node_modules", ".tox", ".nox", ".mypy_cache", ".pytest_cache", ".ruff_cache",
             "dist", "build", ".eggs", "site-packages"}


def _files(path: Path, suffix: str) -> list[Path]:
    if path.is_file():
        return [path] if path.suffix == suffix else []
    out = []
    for p in path.rglob(f"*{suffix}"):
        rel_parts = p.relative_to(path).parts[:-1]  # dirs below the root only
        if any(part in SKIP_DIRS or part.startswith(".") for part in rel_parts):
            continue
        out.append(p)
    return sorted(out)


def _file_digests(path: Path) -> dict[str, str]:
    """{repo-relative path: sha256(content)} for every indexed source file."""
    pkg_parent = path.parent
    out: dict[str, str] = {}
    for f in _files(path, ".py") + _files(path, ".sql") + _files(path, ".ts") + _files(path, ".tsx"):
        rel = f.relative_to(pkg_parent).as_posix() if f.is_relative_to(pkg_parent) else f.name
        out[rel] = hashlib.sha256(f.read_bytes()).hexdigest()
    return out


def build_graph(path: Path, resolver: str = "heuristic") -> Graph:
    py_files = _files(path, ".py")
    graph = extract_paths(py_files, root=path, resolver=resolver) if py_files else Graph()
    sql_files = _files(path, ".sql")
    if sql_files:
        try:
            from .sql_extract import extract_sql_paths
            schema = extract_sql_paths(sql_files, root=path)
            graph.nodes.extend(schema.nodes)
            graph.edges.extend(schema.edges)
        except ModuleNotFoundError:
            pass  # sqlglot not installed; skip SQL (install with `--extra sql`)
    ts_files = _files(path, ".ts") + _files(path, ".tsx")
    if ts_files:
        try:
            from .ts_extract import extract_ts_paths
            tsg = extract_ts_paths(ts_files, root=path)
            graph.nodes.extend(tsg.nodes)
            graph.edges.extend(tsg.edges)
        except ModuleNotFoundError:
            pass  # tree-sitter-typescript not installed; skip (install with `--extra ts`)
    _extract_embedded_sql(graph)
    _bridge_models_to_tables(graph)
    if not graph.nodes:
        raise FileNotFoundError(f"no Python or SQL files under {path}")
    return graph


def _extract_embedded_sql(graph: Graph) -> None:
    """Pull SQL out of Python string literals: CREATE TABLE -> table/column nodes,
    DML -> QUERIES edges (function -> table). The bridge for raw-SQL (non-ORM) apps."""
    from .model import EXTRACTED, Edge

    units = [u for n in graph.nodes for u in n.extra.get("sql", [])]
    if not units:
        return
    try:
        from .sql_extract import extract_embedded_sql
    except ModuleNotFoundError:
        return  # sqlglot not installed
    new_nodes, contains, pending_fks, pending_queries, pending_joins, pending_cols = extract_embedded_sql(units)

    by_qual = {n.qualified_name: n for n in graph.nodes if n.kind == "table"}
    col_quals = {n.qualified_name for n in graph.nodes if n.kind == "column"}
    by_id = {n.id for n in graph.nodes}
    # Dedup tables AND columns by qualified name: the same CREATE TABLE appearing in
    # a .sql file and embedded in Python must not mint orphan duplicate column nodes
    # (they pollute candidates and double-count in eval gold sets).
    for n in new_nodes:
        if n.id in by_id:
            continue
        if n.kind == "table" and n.qualified_name in by_qual:
            continue
        if n.kind == "column" and n.qualified_name in col_quals:
            continue
        graph.nodes.append(n)
        by_id.add(n.id)
        if n.kind == "table":
            by_qual[n.qualified_name] = n
        elif n.kind == "column":
            col_quals.add(n.qualified_name)
    valid = by_id
    seen = {(e.type, e.src, e.dst) for e in graph.edges}
    for e in contains:
        if e.src in valid and e.dst in valid and (e.type, e.src, e.dst) not in seen:
            seen.add((e.type, e.src, e.dst))
            graph.edges.append(e)
    by_name: dict[str, Node] = {}
    col_by_qual: dict[str, Node] = {}
    for n in graph.nodes:
        if n.kind == "table":
            by_name.setdefault(n.name, n)
        elif n.kind == "column":
            col_by_qual[n.qualified_name] = n

    def _table(ref):
        return by_qual.get(ref) or by_name.get(ref.rsplit(".", 1)[-1])

    def _edge(etype, src, dst):
        if src in valid and dst in valid and (etype, src, dst) not in seen:
            seen.add((etype, src, dst))
            graph.edges.append(Edge(etype, src, dst, EXTRACTED))

    for src_id, ref in pending_fks:
        tgt = _table(ref)
        if tgt:
            _edge("REFERENCES", src_id, tgt.id)
    for owner_id, ref in pending_queries:
        tgt = _table(ref)
        if tgt:
            _edge("QUERIES", owner_id, tgt.id)
    for a, b in pending_joins:  # table <-> table relationship from a query JOIN
        ta, tb = _table(a), _table(b)
        if ta and tb and ta.id != tb.id:
            _edge("JOINS", ta.id, tb.id)
    for owner_id, tbl, col in pending_cols:  # function -> specific column
        cn = col_by_qual.get(f"{tbl}.{col}")
        if cn:
            _edge("QUERIES", owner_id, cn.id)


def _bridge_models_to_tables(graph: Graph) -> None:
    """Link ORM model classes to their SQL tables (MAPS_TO) — the code<->schema bridge.
    Mapping comes from an explicit `__tablename__`, so it's EXTRACTED."""
    from .model import EXTRACTED, Edge

    by_qual = {n.qualified_name: n for n in graph.nodes if n.kind == "table"}
    by_name: dict[str, Node] = {}
    for n in graph.nodes:
        if n.kind == "table":
            by_name.setdefault(n.name, n)
    seen = {(e.type, e.src, e.dst) for e in graph.edges}
    for n in graph.nodes:
        tn = n.extra.get("tablename") if n.kind == "class" else None
        if not tn:
            continue
        tgt = by_qual.get(tn) or by_name.get(tn.rsplit(".", 1)[-1])
        if tgt is not None and ("MAPS_TO", n.id, tgt.id) not in seen:
            seen.add(("MAPS_TO", n.id, tgt.id))
            graph.edges.append(Edge("MAPS_TO", n.id, tgt.id, EXTRACTED))


def embed_graph(graph: Graph, embedder=None, cache: EmbeddingCache | None = None) -> tuple[int, int]:
    """Embed every node, reusing `cache` for unchanged embed_text. Returns (reused, embedded)."""
    embedder = embedder or get_embedder()
    dim = getattr(embedder, "dim", None)
    if not getattr(embedder, "dim_is_exact", True):
        dim = None  # unconfirmed width (Ollama before its first call) must not void cache hits
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
               use_cache: bool = True, resolver: str = "heuristic") -> Store:
    """Parse → embed → store. Returns the open Store.

    The Kuzu vector column is sized to the embedder's *actual* output dimension, so
    swapping models (hash 768, nomic 768, mxbai 1024, …) just works without touching
    the schema. A content-hash embedding cache (under <db_dir>/cache) makes re-indexing
    a changed repo fast: only changed symbols are re-embedded. `resolver`: 'heuristic'
    or 'jedi' (receiver-type call resolution).
    """
    embedder = embedder or get_embedder(dim=dim)
    graph = build_graph(path, resolver=resolver)
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
    store.set_meta("schema_version", SCHEMA_VERSION)
    # Per-file content hashes, so `update` can detect what changed without re-embedding.
    for rel, sha in _file_digests(path).items():
        store.set_meta(f"file:{rel}", sha)
    return store


def diff_files(path: Path, db_path: Path) -> dict[str, list[str]]:
    """Compare on-disk files to the hashes recorded in the graph. Returns
    {changed, added, deleted} (paths). 'changed' includes 'added' for convenience."""
    cur = _file_digests(path)
    store = Store(db_path)
    prev = {k[len("file:"):]: v for k, v in store.all_meta().items() if k.startswith("file:")}
    store.close()
    changed = sorted(r for r, s in cur.items() if prev.get(r) != s)
    added = sorted(set(cur) - set(prev))
    deleted = sorted(set(prev) - set(cur))
    return {"changed": changed, "added": added, "deleted": deleted}


def update_index(path: Path, db_path: Path, dim: int = DEFAULT_DIM, embedder=None,
                 resolver: str = "heuristic", use_cache: bool = True) -> dict:
    """Incremental re-index. No-op when nothing changed (instant). Otherwise a
    **row-level delta**: rebuild the graph in memory (re-parse is cheap; embeddings
    come from the content-hash cache), then keep unchanged node rows (matched by
    content_sha) and only delete/recreate changed ones, refreshing edges. Falls back
    to a full rebuild if the embedder (or its dimension) changed; with no explicit
    embedder it adopts the one recorded in the graph at index time."""
    path, db_path = Path(path), Path(db_path)
    if not db_path.exists():
        index_path(path, db_path, dim=dim, embedder=embedder, resolver=resolver).close()
        return {"status": "indexed", "changed": sorted(_file_digests(path)), "added": [],
                "deleted": [], "created": 0, "removed": 0}
    delta = diff_files(path, db_path)
    if not (delta["changed"] or delta["deleted"]):
        return {"status": "up-to-date", **delta, "embedded": 0, "reused": 0, "created": 0, "removed": 0}

    meta_store = Store(db_path)
    prev_dim = meta_store.get_meta("embedding_dim")
    prev_backend = meta_store.get_meta("embedder_backend")
    prev_model = meta_store.get_meta("embedder_model") or ""
    meta_store.close()
    if embedder is None and prev_backend:
        # A plain `update` must keep the index's embedder: defaulting to hash on an
        # ollama-built graph would mix vector spaces (both 768-dim — nothing errors).
        try:
            embedder = get_embedder(prev_backend, dim=int(prev_dim) if prev_dim else dim,
                                    model=prev_model or None)
        except ValueError:
            pass  # unknown recorded backend (hand-built store) — use the default
    embedder = embedder or get_embedder(dim=dim)
    try:
        graph = build_graph(path, resolver=resolver)
    except FileNotFoundError:
        # Every source file is gone: the correct delta is "delete everything",
        # not a crash that leaves the stale graph serving answers.
        store = Store(db_path)
        removed = len(store.node_shas())
        store.delete_all_edges()
        store.conn.execute("MATCH (c:CodeNode) DETACH DELETE c")
        for rel in delta["deleted"]:
            store.delete_meta(f"file:{rel}")
        store.close()
        return {"status": "updated", **delta, "embedded": 0, "reused": 0,
                "created": 0, "removed": removed}
    cache = EmbeddingCache.for_embedder(db_path.parent / "cache", getattr(embedder, "name", "hash")) if use_cache else None
    reused, embedded = embed_graph(graph, embedder=embedder, cache=cache)
    if cache is not None:
        cache.save()
    actual_dim = len(graph.nodes[0].embedding) if graph.nodes and graph.nodes[0].embedding else dim

    name = getattr(embedder, "name", "hash")
    backend, _, model = name.partition(":")
    # A dim change can't share the fixed-size vector column; an *explicit* embedder
    # change must not keep old vectors on unchanged rows — either way, full rebuild.
    if (prev_dim is not None and int(prev_dim) != actual_dim) or (
            prev_backend is not None and (prev_backend, prev_model) != (backend, model)):
        index_path(path, db_path, dim=actual_dim, embedder=embedder, overwrite=True, resolver=resolver).close()
        return {"status": "rebuilt", **delta, "embedded": embedded, "reused": reused, "created": 0, "removed": 0}

    store = Store(db_path)

    db_sha = store.node_shas()
    keep = {n.id for n in graph.nodes if db_sha.get(n.id) == n.content_sha}
    delete_ids = [i for i in db_sha if i not in keep]
    create_nodes = [n for n in graph.nodes if n.id not in keep]
    store.delete_all_edges()
    store.delete_nodes(delete_ids)
    store.load_nodes(create_nodes, actual_dim)
    store.load_edges(graph.edges)
    name = getattr(embedder, "name", "hash")
    backend, _, model = name.partition(":")
    store.set_meta("embedder_backend", backend)
    store.set_meta("embedder_model", model)
    store.set_meta("embedding_dim", str(actual_dim))
    store.set_meta("schema_version", SCHEMA_VERSION)
    for rel, sha in _file_digests(path).items():
        store.set_meta(f"file:{rel}", sha)
    for rel in delta["deleted"]:
        store.delete_meta(f"file:{rel}")
    store.close()
    return {"status": "updated", **delta, "embedded": embedded, "reused": reused,
            "created": len(create_nodes), "removed": len(delete_ids)}
