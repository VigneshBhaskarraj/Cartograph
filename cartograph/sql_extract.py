"""Deterministic SQL-schema extraction: DDL -> graph nodes + edges.

Parses `CREATE TABLE` statements with sqlglot (optional `sql` extra) into `table` and
`column` nodes, `CONTAINS` (table -> column) and `REFERENCES` (FK column -> referenced
table) edges. This lands app code and DB schema in *one* graph — the SPEC differentiator.
FK edges are deterministic, so they're tagged EXTRACTED. No network.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from .model import EXTRACTED, Edge, Graph, Node


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()


def _table_node(name: str, cols: list[tuple[str, str]], rel_path: str, line: int) -> Node:
    col_summary = ", ".join(f"{c} {t}".strip() for c, t in cols)
    return Node(
        id=f"{rel_path}::table.{name}",
        kind="table",
        name=name,
        qualified_name=name,
        module=rel_path,
        file_path=rel_path,
        start_line=line,
        end_line=line,
        signature=f"TABLE {name} ({col_summary})",
        embed_text=f"table {name}\ncolumns: {col_summary}",
        content_sha=_sha(f"{name}|{col_summary}"),
    )


def _column_node(table: str, col: str, ctype: str, rel_path: str, line: int) -> Node:
    qn = f"{table}.{col}"
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
    pending_fks: list of (fk_column_node_id, referenced_table_name)."""
    import sqlglot
    from sqlglot import exp

    nodes: list[Node] = []
    edges: list[Edge] = []
    pending_fks: list[tuple[str, str]] = []
    try:
        statements = sqlglot.parse(source, read=dialect)
    except Exception:
        return nodes, edges, pending_fks

    for stmt in statements:
        if not isinstance(stmt, exp.Create) or (stmt.kind or "").upper() != "TABLE":
            continue
        tbl = stmt.find(exp.Table)
        if tbl is None:
            continue
        tname = tbl.name
        line = (stmt.meta or {}).get("line", 0) or 0
        col_defs = list(stmt.find_all(exp.ColumnDef))
        cols = [(c.name, (c.args.get("kind").sql() if c.args.get("kind") else "")) for c in col_defs]
        table = _table_node(tname, cols, rel_path, line)
        nodes.append(table)
        col_ids: dict[str, str] = {}
        for c, t in cols:
            cn = _column_node(tname, c, t, rel_path, line)
            nodes.append(cn)
            col_ids[c] = cn.id
            edges.append(Edge("CONTAINS", table.id, cn.id, EXTRACTED))

        # FKs: table-level FOREIGN KEY (...) REFERENCES t, and inline col REFERENCES t.
        def _ref_table(node) -> str | None:
            ref = node.args.get("reference") if hasattr(node, "args") else None
            target = (ref or node).find(exp.Table)
            return target.name if target is not None else None

        for fk in stmt.find_all(exp.ForeignKey):
            rt = _ref_table(fk)
            local = [i.name for i in fk.expressions]
            for lc in local:
                src = col_ids.get(lc, table.id)
                if rt:
                    pending_fks.append((src, rt))
        for cdef in col_defs:
            for r in cdef.find_all(exp.Reference):
                target = r.find(exp.Table)
                if target is not None:
                    pending_fks.append((col_ids.get(cdef.name, table.id), target.name))

    return nodes, edges, pending_fks


def extract_sql_paths(paths: list[Path], root: Path, dialect: str | None = None) -> Graph:
    pkg_parent = root.parent if (root.is_dir() or root.is_file()) else root
    all_nodes: list[Node] = []
    all_edges: list[Edge] = []
    pending: list[tuple[str, str]] = []
    for path in paths:
        rel = path.relative_to(pkg_parent).as_posix() if path.is_relative_to(pkg_parent) else path.name
        nodes, edges, fks = extract_sql_source(path.read_text(encoding="utf-8", errors="replace"), rel, dialect)
        all_nodes.extend(nodes)
        all_edges.extend(edges)
        pending.extend(fks)

    table_index = {n.qualified_name: n for n in all_nodes if n.kind == "table"}
    seen = {(e.type, e.src, e.dst) for e in all_edges}
    for src_id, ref_table in pending:
        tgt = table_index.get(ref_table)
        if tgt is None:
            continue
        key = ("REFERENCES", src_id, tgt.id)
        if key not in seen:
            seen.add(key)
            all_edges.append(Edge("REFERENCES", src_id, tgt.id, EXTRACTED))
    return Graph(nodes=all_nodes, edges=all_edges)
