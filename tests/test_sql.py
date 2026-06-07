from pathlib import Path

import pytest

pytest.importorskip("sqlglot")

from cartograph.service import CartographService  # noqa: E402
from cartograph.sql_extract import extract_sql_paths  # noqa: E402

SQL = Path(__file__).parent / "fixtures" / "schema.sql"


def test_tables_columns_and_fk():
    g = extract_sql_paths([SQL], root=SQL)
    tables = {n.name for n in g.nodes if n.kind == "table"}
    assert tables == {"users", "orders"}
    cols = {n.qualified_name for n in g.nodes if n.kind == "column"}
    assert {"users.id", "users.email", "orders.user_id", "orders.total"} <= cols

    contains = [(e.src, e.dst) for e in g.edges if e.type == "CONTAINS"]
    assert len(contains) == 5  # users(id,email) + orders(id,user_id,total)

    refs = [e for e in g.edges if e.type == "REFERENCES"]
    by_id = {n.id: n for n in g.nodes}
    # orders.user_id REFERENCES users (FK), tagged EXTRACTED (deterministic)
    assert any(by_id[e.src].qualified_name == "orders.user_id"
               and by_id[e.dst].qualified_name == "users"
               and e.confidence == "EXTRACTED" for e in refs)


def test_sql_indexed_into_same_graph(tmp_path):
    """SQL schema lands in the same Kuzu graph and is queryable via the service."""
    from cartograph.pipeline import index_path

    store = index_path(SQL, tmp_path / "g.kuzu", dim=64, overwrite=True)
    counts = store.counts()
    store.close()
    assert counts.get("node:table") == 2
    assert counts.get("node:column") == 5
    assert counts.get("edge:REFERENCES", 0) >= 1

    svc = CartographService(tmp_path / "g.kuzu")
    # lexical/semantic retrieval finds a table by name
    hits = svc.query("orders", mode="lexical", k=5)
    assert any(h["kind"] == "table" and h["name"] == "orders" for h in hits)
    # the FK is traversable: orders.user_id --REFERENCES--> users
    fk = svc.neighbors("orders.user_id", direction="out", relation="REFERENCES")
    assert any(n["name"] == "users" for n in fk)
    svc.close()
