"""Self-dogfood: index Cartograph's own package and assert real structure.

A regression guard on real code (not a toy fixture) — exercises extraction,
resolution, and the query service end-to-end the way an agent would.
"""

from pathlib import Path

from cartograph.pipeline import index_path
from cartograph.service import CartographService

PKG = Path(__file__).resolve().parents[1] / "cartograph"


def test_dogfood_self_index(tmp_path):
    db = tmp_path / "self.kuzu"
    store = index_path(PKG, db, dim=128, overwrite=True)
    counts = store.counts()
    store.close()
    assert counts.get("node:function", 0) + counts.get("node:method", 0) > 50

    svc = CartographService(db)
    # Real call edges resolve correctly on our own code (accepts qualified names).
    callees = {c["qualified_name"] for c in svc.calls("cartograph.pipeline.index_path")}
    assert "cartograph.pipeline.build_graph" in callees
    assert "cartograph.pipeline.embed_graph" in callees
    # Reverse direction.
    callers = {c["qualified_name"] for c in svc.callers("cartograph.pipeline.build_graph")}
    assert "cartograph.pipeline.index_path" in callers
    # shortest_path traverses a real call: index_path -> Store.load.
    path = svc.shortest_path("cartograph.pipeline.index_path", "cartograph.store.Store.load")
    assert len(path) >= 2 and path[-1]["name"] == "load"
    svc.close()
