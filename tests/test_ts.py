from pathlib import Path

import pytest

pytest.importorskip("tree_sitter_typescript")

from cartograph.pipeline import build_graph, index_path  # noqa: E402
from cartograph.service import CartographService  # noqa: E402

FIX = Path(__file__).parent / "fixtures" / "sample.ts"


def test_typescript_nodes_and_edges():
    g = build_graph(FIX)
    by_id = {n.id: n for n in g.nodes}
    kinds = {(n.name, n.kind) for n in g.nodes}
    assert ("Animal", "class") in kinds and ("Dog", "class") in kinds
    assert ("bark", "function") in kinds and ("greet", "function") in kinds  # arrow const
    assert ("Named", "interface") in kinds
    assert sum(1 for n in g.nodes if n.name == "speak" and n.kind == "method") == 2

    inh = {(by_id[e.src].name, by_id[e.dst].name) for e in g.edges if e.type == "INHERITS"}
    assert ("Dog", "Animal") in inh
    calls = {(by_id[e.src].name, by_id[e.dst].name) for e in g.edges if e.type == "CALLS"}
    assert ("speak", "bark") in calls  # Dog.speak -> bark
    assert ("greet", "bark") in calls
    assert any(e.type == "IMPORTS" for e in g.edges)


def test_typescript_queryable_via_service(tmp_path):
    store = index_path(FIX, tmp_path / "g.kuzu", dim=64, overwrite=True)
    store.close()
    svc = CartographService(tmp_path / "g.kuzu")
    hits = svc.calls("Dog.speak")
    assert any(h["name"] == "bark" for h in hits)
    svc.close()
