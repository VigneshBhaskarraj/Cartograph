from pathlib import Path

import pytest

pytest.importorskip("sqlglot")

from cartograph.pipeline import build_graph  # noqa: E402

FIX = Path(__file__).parent / "fixtures" / "embedded_join.py"


def test_join_and_column_level_edges():
    g = build_graph(FIX)
    by_id = {n.id: n for n in g.nodes}

    # JOIN relationship mined from the query: users <-> orders
    joins = {frozenset((by_id[e.src].name, by_id[e.dst].name)) for e in g.edges if e.type == "JOINS"}
    assert frozenset(("users", "orders")) in joins

    # Column-level QUERIES: the function -> the specific columns it reads
    col_q = {(by_id[e.src].name, by_id[e.dst].qualified_name)
             for e in g.edges if e.type == "QUERIES" and by_id[e.dst].kind == "column"}
    assert ("user_order_totals", "users.email") in col_q
    assert ("user_order_totals", "orders.total") in col_q

    # Table-level QUERIES still present
    tbl_q = {(by_id[e.src].name, by_id[e.dst].name)
             for e in g.edges if e.type == "QUERIES" and by_id[e.dst].kind == "table"}
    assert ("user_order_totals", "users") in tbl_q
    assert all(e.confidence == "EXTRACTED" for e in g.edges if e.type in ("JOINS", "QUERIES"))
