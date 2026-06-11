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


def test_import_binds_to_exact_module_not_suffix(tmp_path):
    """Audit H3: './utils' must resolve to `utils`, never to a module that merely
    ends with the same substring (`statsutils`)."""
    (tmp_path / "utils.ts").write_text("export function helper(): number { return 1; }\n")
    (tmp_path / "statsutils.ts").write_text("export function other(): number { return 2; }\n")
    (tmp_path / "main.ts").write_text("import { helper } from './utils';\nhelper();\n")
    g = build_graph(tmp_path)
    by_id = {n.id: n for n in g.nodes}
    targets = {by_id[e.dst].qualified_name for e in g.edges
               if e.type == "IMPORTS" and by_id[e.src].name == "main"}
    assert any(t.endswith(".utils") or t == "utils" for t in targets)
    assert not any("statsutils" in t for t in targets)
