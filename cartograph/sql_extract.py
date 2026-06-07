"""Deterministic SQL-schema extraction: DDL -> graph nodes + edges.

Parses `CREATE TABLE` statements with sqlglot (optional `sql` extra) into `table` and
`column` nodes, `CONTAINS` (table -> column) and `REFERENCES` (FK column -> referenced
table) edges. This lands app code and DB schema in *one* graph — the SPEC differentiator.
FK edges are deterministic, so they're tagged EXTRACTED. No network.

Table identity is schema-qualified (`schema.table`) so same-named tables in different
schemas don't collide; nodes are deduped so repeated / `IF NOT EXISTS` DDL (migrations)
doesn't produce duplicate primary keys.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from .model import EXTRACTED, Edge, Graph, Node


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()


def _qual(tbl) -> str:
    """Schema/catalog-qualified table name, e.g. 'analytics.users' (or 'users')."""
    parts = [getattr(tbl, "catalog", "") or "", getattr(tbl, "db", "") or "", tbl.name]
    return ".".join(p for p in parts if p)


def _table_node(qual: str, bare: str, cols: list[tuple[str, str]], rel_path: str, line: int) -> Node:
    col_summary = ", ".join(f"{c} {t}".strip() for c, t in cols)
    return Node(
        id=f"{rel_path}::table.{qual}",
        kind="table",
        name=bare,
        qualified_name=qual,
        module=rel_path,
        file_path=rel_path,
        start_line=line,
        end_line=line,
        signature=f"TABLE {qual} ({col_summary})",
        embed_text=f"table {qual}\ncolumns: {col_summary}",
        content_sha=_sha(f"{qual}|{col_summary}"),
    )


def _column_node(table_qual: str, col: str, ctype: str, rel_path: str, line: int) -> Node:
    qn = f"{table_qual}.{col}"
    return Node(
        id=f"{rel_path}::table.{qn}",
        kind="column",
        name=col,
        qualified_name=qn,
        module=rel_path,
        file_path=rel_path,
        start_line=line,
        end_line=line,
        signature=f"{col} {ctype}".strip(),
        embed_text=f"column {qn} {ctype}".strip(),
        content_sha=_sha(f"{qn}|{ctype}"),
    )


def extract_sql_source(source: str, rel_path: str, dialect: str | None = None):
    """Returns (nodes, contains_edges, pending_fks) for one SQL file.
    pending_fks: list of (fk_column_node_id, referenced_table_qualified_name)."""
    import sqlglot
    from sqlglot import exp

    nodes: list[Node] = []
    edges: list[Edge] = []
    pending_fks: list[tuple[str, str]] = []
    try:
        statements = sqlglot.parse(source, read=dialect)
    except Exception as exc:
        import warnings
        warnings.warn(f"SQL parse failed for {rel_path} ({exc}); skipping. Try a --dialect.", stacklevel=2)
        return nodes, edges, pending_fks

    for stmt in statements:
        if isinstance(stmt, exp.Create) and (stmt.kind or "").upper() == "TABLE":
            _create_to_nodes(stmt, rel_path, nodes, edges, pending_fks)

    return nodes, edges, pending_fks


def _create_to_nodes(stmt, rel_path: str, nodes: list, edges: list, pending_fks: list) -> None:
    """Append table + column nodes, CONTAINS edges, and pending FKs for one CREATE TABLE."""
    from sqlglot import exp

    tbl = stmt.find(exp.Table)
    if tbl is None:
        return
    qual = _qual(tbl)
    line = (stmt.meta or {}).get("line", 0) or 0
    col_defs = list(stmt.find_all(exp.ColumnDef))
    cols = [(c.name, (c.args.get("kind").sql() if c.args.get("kind") else "")) for c in col_defs]
    table = _table_node(qual, tbl.name, cols, rel_path, line)
    nodes.append(table)
    col_ids: dict[str, str] = {}
    for c, t in cols:
        cn = _column_node(qual, c, t, rel_path, line)
        nodes.append(cn)
        col_ids[c] = cn.id
        edges.append(Edge("CONTAINS", table.id, cn.id, EXTRACTED))
    for fk in stmt.find_all(exp.ForeignKey):
        ref = fk.args.get("reference")
        target = ref.find(exp.Table) if ref is not None else None
        if target is None:
            continue
        for lc in (i.name for i in fk.expressions):
            pending_fks.append((col_ids.get(lc, table.id), _qual(target)))
    for cdef in col_defs:
        for r in cdef.find_all(exp.Reference):
            target = r.find(exp.Table)
            if target is not None:
                pending_fks.append((col_ids.get(cdef.name, table.id), _qual(target)))


def extract_embedded_sql(units: list[tuple[str, str, str]], dialect: str | None = None):
    """SQL embedded in Python strings. `units`: (owner_id, rel_path, sql_text).
    Returns (table/col nodes, CONTAINS edges, pending_fks, pending_queries) where
    pending_queries = (owner_function_id, referenced_table_qualified_name)."""
    import sqlglot
    from sqlglot import exp

    nodes: list[Node] = []
    edges: list[Edge] = []
    pending_fks: list[tuple[str, str]] = []
    pending_queries: list[tuple[str, str]] = []
    for owner_id, rel_path, sql in units:
        try:
            statements = sqlglot.parse(sql, read=dialect)
        except Exception:
            continue
        for stmt in statements:
            if stmt is None:
                continue
            if isinstance(stmt, exp.Create) and (stmt.kind or "").upper() == "TABLE":
                _create_to_nodes(stmt, rel_path, nodes, edges, pending_fks)
            else:
                # DML / queries: the enclosing function touches these tables.
                for t in stmt.find_all(exp.Table):
                    pending_queries.append((owner_id, _qual(t)))
    return nodes, edges, pending_fks, pending_queries


def extract_sql_paths(paths: list[Path], root: Path, dialect: str | None = None) -> Graph:
    pkg_parent = root.parent if (root.is_dir() or root.is_file()) else root
    raw_nodes: list[Node] = []
    raw_edges: list[Edge] = []
    pending: list[tuple[str, str]] = []
    for path in paths:
        rel = path.relative_to(pkg_parent).as_posix() if path.is_relative_to(pkg_parent) else path.name
        nodes, edges, fks = extract_sql_source(path.read_text(encoding="utf-8", errors="replace"), rel, dialect)
        raw_nodes.extend(nodes)
        raw_edges.extend(edges)
        pending.extend(fks)

    # Dedup nodes (repeated / IF NOT EXISTS DDL) — first definition wins.
    by_id: dict[str, Node] = {}
    for n in raw_nodes:
        by_id.setdefault(n.id, n)
    nodes = list(by_id.values())

    # Resolve FKs by qualified name, falling back to bare table name.
    by_qual = {n.qualified_name: n for n in nodes if n.kind == "table"}
    by_bare: dict[str, Node] = {}
    for n in nodes:
        if n.kind == "table":
            by_bare.setdefault(n.name, n)

    seen: set[tuple[str, str, str]] = set()
    edges: list[Edge] = []
    for e in raw_edges:
        key = (e.type, e.src, e.dst)
        if e.src in by_id and e.dst in by_id and key not in seen:
            seen.add(key)
            edges.append(e)
    for src_id, ref in pending:
        tgt = by_qual.get(ref) or by_bare.get(ref.rsplit(".", 1)[-1])
        if tgt is None or src_id not in by_id:
            continue
        key = ("REFERENCES", src_id, tgt.id)
        if key not in seen:
            seen.add(key)
            edges.append(Edge("REFERENCES", src_id, tgt.id, EXTRACTED))
    return Graph(nodes=nodes, edges=edges)
