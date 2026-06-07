from pathlib import Path

import pytest

pytest.importorskip("sqlglot")  # bridging needs SQL tables to map onto

from cartograph.pipeline import build_graph, index_path  # noqa: E402
from cartograph.service import CartographService  # noqa: E402

CORPUS = Path(__file__).resolve().parents[1] / "eval" / "bridge_corpus"


def test_maps_to_bridges_model_to_table():
    """ORM model classes link to their SQL tables via MAPS_TO (EXTRACTED)."""
    g = build_graph(CORPUS)
    by_id = {n.id: n for n in g.nodes}
    pairs = {(by_id[e.src].name, by_id[e.dst].qualified_name)
             for e in g.edges if e.type == "MAPS_TO"}
    assert ("User", "users") in pairs
    assert ("Order", "orders") in pairs
    assert all(e.confidence == "EXTRACTED" for e in g.edges if e.type == "MAPS_TO")


def test_bridge_traversal_via_service(tmp_path):
    """An agent can hop model -> table (MAPS_TO) and table -> FK (REFERENCES) in one graph."""
    index_path(CORPUS, tmp_path / "g.kuzu", dim=64, overwrite=True).close()
    svc = CartographService(tmp_path / "g.kuzu")
    to_table = svc.neighbors("models.User", direction="out", relation="MAPS_TO")
    assert any(n["name"] == "users" and n["kind"] == "table" for n in to_table)
    # orders.user_id --REFERENCES--> users (FK), reachable in the same graph
    fk = svc.neighbors("orders.user_id", direction="out", relation="REFERENCES")
    assert any(n["name"] == "users" for n in fk)
    svc.close()
