"""End-to-end indexing: source path -> extract -> embed -> Kuzu store."""

from __future__ import annotations

import hashlib
from pathlib import Path

from .cache import EmbeddingCache
from .embed import get_embedder
from .extract import extract_paths
from .model import Graph
from .store import DEFAULT_DIM, SCHEMA_VERSION, Store

TS_JS_SUFFIXES = (".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs")


# Indexing a project root must not sweep up its virtualenv, vendored deps, or VCS
# internals — embedding a .venv costs hours of Ollama time and poisons retrieval.
# (Hidden dirs are skipped wholesale; `venv`/`env` only when they really are a
# virtualenv, so a legitimate source package named `env` still gets indexed.)
SKIP_DIRS = {"__pycache__", "node_modules", "site-packages", ".eggs",
             "dist", "build", "vendor", "target"}
VENV_NAMES = {"venv", "env"}


def _files(path: Path, suffix: str) -> list[Path]:
    if path.is_file():
        return [path] if path.suffix == suffix else []
    out = []
    for p in path.rglob(f"*{suffix}"):
        rel_parts = p.relative_to(path).parts[:-1]  # dirs below the root only
        sub, skip = path, False
        for part in rel_parts:
            sub = sub / part
            if (part in SKIP_DIRS or part.startswith(".")
                    or (part in VENV_NAMES and (sub / "pyvenv.cfg").exists())):
                skip = True
                break
        if not skip:
            out.append(p)
    return sorted(out)


def _file_digests(path: Path) -> dict[str, str]:
    """{repo-relative path: sha256(content)} for every indexed source file."""
    pkg_parent = path.parent
    out: dict[str, str] = {}
    for f in (f for suf in (".py", ".sql", ".java", ".go", *TS_JS_SUFFIXES) for f in _files(path, suf)):
        rel = f.relative_to(pkg_parent).as_posix() if f.is_relative_to(pkg_parent) else f.name
        out[rel] = hashlib.sha256(f.read_bytes()).hexdigest()
    return out


def _warn_skipped(files: list, lang: str, extra: str) -> None:
    """Found source files but their extractor isn't installed: silence here means a
    stranger indexes a Java repo and gets an almost-empty graph with no clue why."""
    import warnings

    warnings.warn(
        f"found {len(files)} {lang} files but their extractor is not installed — "
        f"they were NOT indexed. Install with: uv sync --extra {extra}",
        stacklevel=2)


def build_graph(path: Path, resolver: str = "heuristic") -> Graph:
    py_files = _files(path, ".py")
    graph = extract_paths(py_files, root=path, resolver=resolver) if py_files else Graph()
    sql_files = _files(path, ".sql")
    if sql_files:
        try:
            from .sql_extract import extract_sql_paths
            schema = _dedup_schema(extract_sql_paths(sql_files, root=path))
            graph.nodes.extend(schema.nodes)
            graph.edges.extend(schema.edges)
        except ImportError:
            _warn_skipped(sql_files, "SQL", "sql")
    java_files = _files(path, ".java")
    if java_files:
        try:
            from .java_extract import extract_java_paths
            jg = extract_java_paths(java_files, root=path)
            graph.nodes.extend(jg.nodes)
            graph.edges.extend(jg.edges)
        except ImportError:
            _warn_skipped(java_files, "Java", "java")
    go_files = _files(path, ".go")
    if go_files:
        try:
            from .go_extract import extract_go_paths
            gg = extract_go_paths(go_files, root=path)
            graph.nodes.extend(gg.nodes)
            graph.edges.extend(gg.edges)
        except ImportError:
            _warn_skipped(go_files, "Go", "go")
    ts_files = [f for suf in TS_JS_SUFFIXES for f in _files(path, suf)]
    if ts_files:
        try:
            from .ts_extract import extract_ts_paths
            tsg = extract_ts_paths(ts_files, root=path)
            graph.nodes.extend(tsg.nodes)
            graph.edges.extend(tsg.edges)
        except ImportError:
            _warn_skipped(ts_files, "TypeScript/JavaScript", "ts")
    # Every language extractor mints external stubs as ext::<target>; a polyglot
    # repo (import redis + require('redis')) would hit a duplicate-primary-key
    # crash at load. Same id = same external package — keep the first.
    seen_ids: set[str] = set()
    deduped = []
    for n in graph.nodes:
        if n.id in seen_ids:
            continue
        seen_ids.add(n.id)
        deduped.append(n)
    graph.nodes = deduped
    _extract_embedded_sql(graph)
    _bridge_models_to_tables(graph)
    if not graph.nodes:
        raise FileNotFoundError(
            f"no supported source files (.py/.js/.ts/.java/.go/.sql) under {path} — "
            "language extras may be missing (see README)")
    return graph


def _dedup_schema(schema: Graph) -> Graph:
    """The same logical table often appears in several .sql files (one schema per
    DB dialect — spring-petclinic ships h2+mysql+postgres). Duplicate table/column
    nodes split the code<->data bridge: MAPS_TO attaches to one copy while a query
    resolves the other. Keep the first node per (kind, qualified_name) and remap
    edges onto the keepers. Known limit: same-named tables in genuinely
    DIFFERENT schemas (multi-service monorepos) merge too — scope the index
    path per service if that matters."""
    keep: dict[tuple[str, str], str] = {}
    remap: dict[str, str] = {}
    nodes = []
    for n in schema.nodes:
        key = (n.kind, n.qualified_name)
        if key in keep:
            remap[n.id] = keep[key]
        else:
            keep[key] = n.id
            nodes.append(n)
    edges, seen = [], set()
    for e in schema.edges:
        src, dst = remap.get(e.src, e.src), remap.get(e.dst, e.dst)
        if src == dst or (e.type, src, dst) in seen:
            continue
        seen.add((e.type, src, dst))
        e.src, e.dst = src, dst
        edges.append(e)
    schema.nodes, schema.edges = nodes, edges
    return schema


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
    cols = {c.qualified_name: c for c in graph.nodes if c.kind == "column"}
    for n in graph.nodes:
        tn = n.extra.get("tablename") if n.kind == "class" else None
        if not tn:
            continue
        tgt = by_qual.get(tn) or by_name.get(tn.rsplit(".", 1)[-1])
        if tgt is None:
            continue
        conf = n.extra.get("tablename_confidence", EXTRACTED)
        if ("MAPS_TO", n.id, tgt.id) not in seen:
            seen.add(("MAPS_TO", n.id, tgt.id))
            graph.edges.append(Edge("MAPS_TO", n.id, tgt.id, conf))
        # Column-level mapping (JPA @Column): entity class -> table.column nodes.
        if n.extra.get("columns"):
            for col in n.extra["columns"]:
                cn = cols.get(f"{tgt.name}.{col}") or cols.get(f"{tgt.qualified_name}.{col}")
                if cn is not None and ("MAPS_TO", n.id, cn.id) not in seen:
                    seen.add(("MAPS_TO", n.id, cn.id))
                    graph.edges.append(Edge("MAPS_TO", n.id, cn.id, conf))


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
    # Digests are captured BEFORE parsing (G5-B3): hashed after, a file edited
    # mid-run would record the new sha against the old parse and read as
    # "up-to-date" forever. Captured first, the worst case is one redundant
    # re-index on the next update.
    digests = _file_digests(path)
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
    # Kuzu auto-commits each statement, so a crash mid-load would leave a
    # partial graph that passes every table check. The dirty flag is durably
    # set before the first row lands and cleared only after the last meta —
    # open_graph refuses dirty graphs (G5-B1).
    store.set_meta("dirty", "1")
    # Version goes in before any rows: a crash mid-load then reads as "interrupted"
    # (the dirty gate's message) rather than "incompatible Cartograph" (the
    # missing-version gate's), which would send the user to the wrong fix.
    store.set_meta("schema_version", SCHEMA_VERSION)
    store.load(graph, dim=actual_dim)
    # Record the embedder so a reader (CLI query, MCP server) reconstructs a matching one.
    name = getattr(embedder, "name", "hash")
    backend, _, model = name.partition(":")
    store.set_meta("embedder_backend", backend)
    store.set_meta("embedder_model", model)
    store.set_meta("embedding_dim", str(actual_dim))
    # Per-file content hashes, so `update` can detect what changed without re-embedding.
    for rel, sha in digests.items():
        store.set_meta(f"file:{rel}", sha)
    store.delete_meta("dirty")
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
    if not path.exists():
        # A typo'd source path must not look like "every file was deleted" — the
        # all-deleted handler below would silently wipe the whole graph.
        raise FileNotFoundError(f"source path {path} does not exist")
    if not db_path.exists():
        index_path(path, db_path, dim=dim, embedder=embedder, resolver=resolver).close()
        return {"status": "indexed", "changed": sorted(_file_digests(path)), "added": [],
                "deleted": [], "created": 0, "removed": 0}
    delta = diff_files(path, db_path)
    meta_store = Store(db_path)
    dirty = meta_store.get_meta("dirty") is not None
    prev_dim = meta_store.get_meta("embedding_dim")
    prev_backend = meta_store.get_meta("embedder_backend")
    prev_model = meta_store.get_meta("embedder_model") or ""
    meta_store.close()
    if not dirty and not (delta["changed"] or delta["deleted"]):
        return {"status": "up-to-date", **delta, "embedded": 0, "reused": 0, "created": 0, "removed": 0}
    if embedder is None and prev_backend:
        # A plain `update` must keep the index's embedder: defaulting to hash on an
        # ollama-built graph would mix vector spaces (both 768-dim — nothing errors).
        try:
            embedder = get_embedder(prev_backend, dim=int(prev_dim) if prev_dim else dim,
                                    model=prev_model or None)
        except ValueError:
            pass  # unknown recorded backend (hand-built store) — use the default
    embedder = embedder or get_embedder(dim=dim)
    digests = _file_digests(path)  # before parsing — see index_path (G5-B3)
    if dirty:
        # A previous index/update died mid-write: the row-level delta below would
        # diff against a half-written graph. A full rebuild is the deterministic
        # repair (embeddings come from the cache, so it's cheap).
        st = index_path(path, db_path, dim=dim, embedder=embedder, overwrite=True, resolver=resolver)
        reused, embedded = getattr(st, "cache_stats", (0, 0))
        st.close()
        return {"status": "rebuilt", **delta, "embedded": embedded, "reused": reused,
                "created": 0, "removed": 0}
    try:
        graph = build_graph(path, resolver=resolver)
    except FileNotFoundError:
        # Every source file is gone: the correct delta is "delete everything",
        # not a crash that leaves the stale graph serving answers.
        store = Store(db_path)
        removed = len(store.node_shas())
        store.set_meta("dirty", "1")  # crash guard — see index_path
        store.delete_all_edges()
        store.conn.execute("MATCH (c:CodeNode) DETACH DELETE c")
        for rel in delta["deleted"]:
            store.delete_meta(f"file:{rel}")
        store.delete_meta("dirty")
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
    store.set_meta("dirty", "1")  # crash guard — see index_path
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
    for rel, sha in digests.items():
        store.set_meta(f"file:{rel}", sha)
    for rel in delta["deleted"]:
        store.delete_meta(f"file:{rel}")
    store.delete_meta("dirty")
    store.close()
    return {"status": "updated", **delta, "embedded": embedded, "reused": reused,
            "created": len(create_nodes), "removed": len(delete_ids)}
