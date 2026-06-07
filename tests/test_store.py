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
