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
from .store import DEFAULT_DIM, SCHEMA_VERSION, Store, expected_codenode_columns, schema_ddl

REQUIRED_TABLES = {"CodeNode", "Meta", *EDGE_TYPES}

# Result/traversal bounds. Tool consumers are agents: an unbounded neighborhood or
# resolve list lands in a model's context window, so every list is capped and says
# so via a trailing {"truncated": True, "total": N, "note": ...} sentinel rather
# than silently dropping rows. MAX_HOPS=8 covers the eval set's deepest MULTIHOP
# question with headroom (Kuzu's hard ceiling is 30 — exceeding it is a raw crash).
MAX_HOPS = 8
NEIGHBOR_CAP = 200
RESOLVE_CAP = 25

# Always-available retrieval modes; `rerank` joins at runtime when a reranker is
# configured. The CLI imports this for its pre-open check — one source of truth.
QUERY_MODES = ("hybrid", "vector", "graph", "lexical")


def _heal_missing_rel_tables(p: Path, missing: set[str]) -> None:
    """Create rel tables a newer Cartograph added to the schema. They are created
    empty — there is nothing to migrate — so an old graph keeps working instead of
    forcing a full re-index just to pass the table-set gate. Needs brief write
    access; raises if another process holds the DB."""
    import warnings

    rel_ddl = {s.split()[3]: s for s in schema_ddl() if s.strip().startswith("CREATE REL TABLE")}
    store = Store(p, read_only=False)
    try:
        for name in sorted(missing):
            store.conn.execute(rel_ddl[name])
        store.set_meta("schema_version", SCHEMA_VERSION)
    finally:
        store.close()
    warnings.warn(
        f"graph at {p} predates the {', '.join(sorted(missing))} table(s); created them "
        "empty. If this corpus contains SQL, re-run `cartograph index` to populate them.",
        stacklevel=3)


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
    if missing and missing <= set(EDGE_TYPES) and store.get_meta("schema_version") in (None, SCHEMA_VERSION):
        # Only rel tables are absent and the recorded version (if any) matches:
        # self-heal in place rather than demanding a re-index. A version of None
        # is deliberately healable here even though the gate below treats
        # missing-version-with-all-tables as incompatible: graphs missing these
        # rel tables PREDATE the version meta entirely (the tables and the meta
        # landed in that order), so None is their expected fingerprint — and the
        # column gate below still validates the healed graph's structure before
        # admitting it. A versionless graph with all tables has no such
        # provenance story, so it stays rejected.
        store.close()
        try:
            _heal_missing_rel_tables(p, missing)
        except Exception:  # e.g. another process holds the writer lock —
            pass  # fall through to the incompatible-graph error below
        store = Store(p, read_only=read_only)
        missing = REQUIRED_TABLES - store.table_names()
        version = store.get_meta("schema_version") if not missing else None
    # The gate checks three layers, most-specific message first: table set,
    # recorded version (absence IS incompatibility — a versionless graph predates
    # the contract or never finished indexing), then CodeNode columns (a column
    # added without a SCHEMA_VERSION bump used to slip through and crash mid-query).
    why = None
    if missing:
        why = f"missing tables: {', '.join(sorted(missing))}"
    elif version is None:
        why = "no recorded schema_version"
    elif version != SCHEMA_VERSION:
        why = f"schema_version {version} != {SCHEMA_VERSION}"
    else:
        gap = expected_codenode_columns() - store.codenode_columns()
        if gap:
            why = f"CodeNode is missing columns: {', '.join(sorted(gap))}"
    if why:
        store.close()
        raise RuntimeError(
            f"graph at {p} was built by an incompatible Cartograph ({why}); "
            "re-run `cartograph index`")
    if store.get_meta("dirty") is not None:
        # An index/update died mid-write. The table set looks fine but the rows
        # are partial (e.g. nodes without edges) — refusing here beats silently
        # serving garbage. `cartograph update` detects the flag and rebuilds.
        store.close()
        raise RuntimeError(
            f"graph at {p} was left mid-write by an interrupted index run; "
            "re-run `cartograph update` (or `cartograph index`) to repair it")
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
        self.modes = set(QUERY_MODES)
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

    def suggest(self, ref: str, limit: int = 5) -> list[str]:
        """Closest qualified names for a reference that failed to resolve — the
        'did you mean' candidates attached to unknown-ref errors. Tiered policy:
        fuzzy match on the last path segment first (catches typos like
        `Dog.speek`), then case-insensitive substring over qualified names
        (catches partial recall). Pool is every non-external symbol."""
        import difflib

        pool = self.store.symbol_names()
        last = ref.rsplit(".", 1)[-1].lower()
        names = {n.lower(): qn for n, qn in pool}
        close = difflib.get_close_matches(last, names, n=limit, cutoff=0.6)
        if close:
            return [names[c] for c in close]
        return [qn for _, qn in pool if last in qn.lower()][:limit]

    def _rid_or_raise(self, ref: str) -> str:
        """Resolve or raise an actionable error. An unknown symbol must error, not
        read as 'known symbol, empty result' — agents act on that difference (the
        CLI already guards this; this gives the MCP surface parity)."""
        rid = self._rid(ref)
        if rid is None:
            candidates = self.suggest(ref)
            hint = f"; did you mean: {', '.join(candidates)}" if candidates else ""
            raise ValueError(f"no node matches {ref!r}{hint}")
        return rid

    def get_node(self, ref: str, strict: bool = False) -> dict | None:
        """Full detail for a node by id or qualified name (e.g. httpx._client.Client.send).
        `strict=True` raises (with 'did you mean' candidates) instead of returning
        None for unknown refs — what the MCP surface wants; the CLI's None checks
        keep the lenient default."""
        rid = self._rid_or_raise(ref) if strict else self._rid(ref)
        return self._node(rid) if rid else None

    @staticmethod
    def _cap(out: list[dict], cap: int, hint: str) -> list[dict]:
        """Truncate a result list to `cap`, appending a sentinel that says so."""
        if len(out) > cap:
            total = len(out)
            out = out[:cap]
            out.append({"truncated": True, "total": total,
                        "note": f"showing {cap} of {total}; {hint}"})
        return out

    def resolve(self, ref: str) -> list[dict]:
        """All nodes a reference matches, most-direct first (use to disambiguate a
        bare name). Capped at RESOLVE_CAP with a trailing truncation sentinel."""
        out = [n for n in (self._node(i) for i in self.store.resolve_ids(ref)) if n]
        return self._cap(out, RESOLVE_CAP, f"qualify the name (e.g. Class.{ref})")

    def neighbors(self, node_id: str, direction: str = "both", relation: str | None = None,
                  hops: int = 1) -> list[dict]:
        """Adjacent nodes. At hops=1 each result is labeled with `relation`
        (CALLS/INHERITS/IMPORTS/CONTAINS/DOCUMENTS) and `direction` (out=this node is
        the source, in=this node is the target). `direction` and `relation` filter the
        edges. hops>1 expands undirected/unlabeled for broad context. Accepts an id or
        qualified name. hops is clamped to 1..MAX_HOPS and results to NEIGHBOR_CAP —
        out-of-range values note the clamp in a trailing sentinel instead of erroring
        (Kuzu hard-crashes above 30, and a hub node's full neighborhood would flood
        an agent's context). Unknown refs raise with 'did you mean' candidates."""
        # Validate filters for EVERY hops value: the multi-hop expansion ignores
        # them by design (unlabeled), but a typo'd value must still be an error,
        # not silently dropped (review follow-up on G5-A2).
        if direction not in ("out", "in", "both"):
            raise ValueError(f"direction must be one of ['both', 'in', 'out'], got {direction!r}")
        if relation is not None and relation.upper() not in EDGE_TYPES:
            raise ValueError(f"unknown relation {relation!r}; valid: {sorted(EDGE_TYPES)}")
        rid = self._rid_or_raise(node_id)
        clamped = max(1, min(int(hops), MAX_HOPS))
        if clamped <= 1:
            types = [relation.upper()] if relation else None
            out = []
            for rel in self.store.relations(rid, direction=direction, types=types):
                node = self._node(rel["id"])
                if node:
                    out.append({**node, "relation": rel["relation"], "direction": rel["direction"]})
        else:
            out = [n for n in (self._node(i) for i in self.store.neighbors(rid, hops=clamped)) if n]
        out = self._cap(out, NEIGHBOR_CAP, "filter by relation/direction or lower hops")
        if clamped != hops:
            out.append({"note": f"hops={hops} clamped to {clamped} (valid range 1..{MAX_HOPS})"})
        return out

    def calls(self, node_id: str) -> list[dict]:
        """What this node calls (outgoing CALLS edges). Accepts an id or qualified name."""
        return self.neighbors(node_id, direction="out", relation="CALLS")

    def callers(self, node_id: str) -> list[dict]:
        """What calls this node (incoming CALLS edges). Accepts an id or qualified name."""
        return self.neighbors(node_id, direction="in", relation="CALLS")

    # -- impact (code <-> data blast radius) -----------------------------------
    _DATA_KINDS = ("table", "column")
    _BRIDGE = ("QUERIES", "MAPS_TO")  # code -> data edges

    def _typed_adj(self) -> dict:
        """Lazy one-time adjacency over typed edges: forward/reverse CALLS and the
        code->data bridge, all in RAM (built from the graph, never source files)."""
        if not hasattr(self, "_adj"):
            calls_fwd: dict[str, list[str]] = {}
            calls_rev: dict[str, list[str]] = {}
            to_data: dict[str, list[str]] = {}
            from_data: dict[str, list[str]] = {}
            contains: dict[str, list[str]] = {}
            for src, dst, etype, _conf in self.store.all_edges_typed():
                if etype == "CALLS":
                    calls_fwd.setdefault(src, []).append(dst)
                    calls_rev.setdefault(dst, []).append(src)
                elif etype in self._BRIDGE:
                    to_data.setdefault(src, []).append(dst)
                    from_data.setdefault(dst, []).append(src)
                elif etype == "CONTAINS":
                    contains.setdefault(src, []).append(dst)
            self._adj = {"fwd": calls_fwd, "rev": calls_rev,
                         "to_data": to_data, "from_data": from_data, "contains": contains}
        return self._adj

    @staticmethod
    def _closure(starts: list[str], step: dict[str, list[str]]) -> list[str]:
        seen, queue = set(starts), list(starts)
        while queue:
            for nxt in step.get(queue.pop(), ()):  # BFS order irrelevant for a set
                if nxt not in seen:
                    seen.add(nxt)
                    queue.append(nxt)
        return [s for s in seen if s not in starts]

    def impact(self, ref: str, max_results: int = 50) -> dict | None:
        """Blast radius across the code<->data bridge.

        For a table/column: which code touches it directly (QUERIES/MAPS_TO —
        a mapped class implicates its methods too), and every function that can
        reach that code through the call graph — i.e. "what application code
        breaks if this column changes". For code (function/class/module): every
        table/column reachable through its scope and transitive callees.

        Honesty: the CALLS expansion over-approximates (those edges are
        INFERRED). The bridge itself can MISS ORM attribute access that has no
        QUERIES edge, so results are NOT guaranteed supersets; FK/JOIN ripple
        between tables is not followed. Reads only the graph.
        """
        rid = self._rid(ref)
        if rid is None:
            return None
        node = self._node(rid)
        adj = self._typed_adj()

        def _nodes(ids: list[str]) -> list[dict]:
            return [n for n in (self._node(i) for i in ids[:max_results]) if n]

        if node["kind"] in self._DATA_KINDS:
            targets = [rid]
            if node["kind"] == "table":  # a table change implicates its columns too
                targets += adj["contains"].get(rid, [])
            else:  # a column change implicates code mapped to its parent table (ORM)
                targets += [t for t, cols in adj["contains"].items() if rid in cols
                            and (p := self.store.get_node(t)) and p["kind"] == "table"]
            direct = sorted({c for t in targets for c in adj["from_data"].get(t, [])})
            # The ORM bridge is class-level (MAPS_TO): the mapped class's methods
            # are implicated too, so they seed the caller closure alongside it.
            seeds = sorted(set(direct) | {m for c in direct for m in adj["contains"].get(c, [])})
            transitive = sorted((set(seeds) - set(direct))
                                | set(self._closure(seeds, adj["rev"]))) if direct else []
            total = len(direct) + len(transitive)
            return {"target": node, "direction": "data->code",
                    "direct_code": _nodes(direct),
                    "transitive_callers": _nodes(transitive),
                    "total_code_paths": total,
                    "truncated": len(direct) > max_results or len(transitive) > max_results}

        # Module/class scope descends CONTAINS first (a module "touches" what its
        # functions touch), then the call graph expands it.
        scope = sorted({rid, *self._closure([rid], adj["contains"])})
        reachable = sorted(set(scope) | set(self._closure(scope, adj["fwd"])))
        data = sorted({d for c in reachable for d in adj["to_data"].get(c, [])})
        direct = sorted({d for c in scope for d in adj["to_data"].get(c, [])})
        rest = [d for d in data if d not in set(direct)]
        return {"target": node, "direction": "code->data",
                "direct_data": _nodes(direct),
                "transitive_data": _nodes(rest),
                "total_data_touched": len(data),
                "truncated": len(direct) > max_results or len(rest) > max_results}

    def shortest_path(self, src: str, dst: str) -> list[dict]:
        """Ordered nodes on a shortest path between two nodes (ids or qualified names)."""
        s, d = self._rid(src), self._rid(dst)
        if s is None or d is None:
            return []
        return [n for n in (self._node(i) for i in self.store.shortest_path(s, d)) if n]

    def stats(self) -> dict[str, int]:
        return self.store.counts()
