from pathlib import Path

import pytest

pytest.importorskip("sqlglot")

from cartograph.pipeline import build_graph  # noqa: E402

FIX = Path(__file__).parent / "fixtures" / "embedded_sql.py"


def test_embedded_sql_tables_fks_and_queries():
    """SQL in Python strings -> tables/columns, FK REFERENCES, and QUERIES bridge edges."""
    g = build_graph(FIX)
    by_id = {n.id: n for n in g.nodes}
    tables = {n.name for n in g.nodes if n.kind == "table"}
    assert {"users", "orders"} <= tables

    # QUERIES: the function that runs the SQL -> the table it touches.
    queries = {(by_id[e.src].name, by_id[e.dst].name) for e in g.edges if e.type == "QUERIES"}
    assert ("save_order", "orders") in queries
    assert ("recent_users", "users") in queries

    # FK parsed out of the embedded DDL.
    refs = {(by_id[e.src].qualified_name, by_id[e.dst].name)
            for e in g.edges if e.type == "REFERENCES"}
    assert ("orders.user_id", "users") in refs
    assert all(e.confidence == "EXTRACTED" for e in g.edges if e.type in ("QUERIES", "REFERENCES"))
