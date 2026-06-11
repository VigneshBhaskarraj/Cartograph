from pathlib import Path

from cartograph.extract import extract_source
from cartograph.store import Store

FIX = Path(__file__).parent / "fixtures" / "sample.py"


def test_schema_tables(tmp_path):
    """M0-2: schema creates the node table and all five rel tables."""
    store = Store.create(tmp_path / "g.kuzu", dim=16)
    names = store.table_names()
    assert {"CodeNode", "CALLS", "INHERITS", "IMPORTS", "CONTAINS", "DOCUMENTS"} <= names
    store.close()


def test_load_roundtrip(tmp_path):
    """M0-4: extracted nodes + a CONTAINS edge survive a load into Kuzu."""
    fx = extract_source("def f():\n    return 1\n", "f.py", "f")
    from cartograph.model import Graph

    graph = Graph(nodes=fx.nodes, edges=fx.edges)
    store = Store.create(tmp_path / "g.kuzu", dim=16)
    store.load(graph, dim=16)
    counts = store.counts()
    assert counts.get("node:function", 0) == 1
    assert counts.get("edge:CONTAINS", 0) >= 1
    store.close()


def test_create_overwrite_clears_stale_wal_without_main_file(tmp_path):
    """Audit L1: a WAL left by a killed index run (main file gone) must not be
    replayed into the fresh DB — that bricked every subsequent index."""
    db = tmp_path / "g.kuzu"
    wal = tmp_path / "g.kuzu.wal"
    wal.write_bytes(b"stale write-ahead log garbage")
    store = Store.create(db, dim=16, overwrite=True)  # must not replay the stale WAL
    assert "CodeNode" in store.table_names()
    store.close()
    if wal.exists():  # kuzu may write its own fresh WAL; the garbage must be gone
        assert b"stale write-ahead log garbage" not in wal.read_bytes()
