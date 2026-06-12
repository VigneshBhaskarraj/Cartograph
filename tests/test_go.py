"""Go extraction (M7): structs, receiver methods, embedding, imports, calls."""

from pathlib import Path

import pytest

pytest.importorskip("tree_sitter_go")

from cartograph.pipeline import build_graph  # noqa: E402


def _graph(tmp_path):
    (tmp_path / "shop.go").write_text(
        'package shop\n\n'
        'import (\n    "fmt"\n    "acme/db"\n)\n\n'
        'type Base struct{}\n\n'
        'type Order struct {\n'
        '    Base\n'              # embedded -> INHERITS
        '    Total float64\n'
        '}\n\n'
        'type Auditable interface {\n    Audit()\n}\n\n'
        'func (o *Order) IsPaid() bool {\n'
        '    return o.check()\n'
        '}\n\n'
        'func (o *Order) check() bool { return o.Total > 0 }\n\n'
        'func NewOrder() *Order {\n'
        '    fmt.Println("x")\n'
        '    return &Order{}\n'
        '}\n')
    return build_graph(tmp_path)


def test_go_nodes_and_edges(tmp_path):
    g = _graph(tmp_path)
    by_qual = {n.qualified_name: n for n in g.nodes}
    assert by_qual["shop.Order"].kind == "class"
    assert by_qual["shop.Auditable"].kind == "interface"
    assert by_qual["shop.Order.IsPaid"].kind == "method"     # receiver method
    assert by_qual["shop.NewOrder"].kind == "function"
    by_id = {n.id: n for n in g.nodes}
    inherits = {(by_id[e.src].name, by_id[e.dst].name) for e in g.edges if e.type == "INHERITS"}
    assert ("Order", "Base") in inherits                      # struct embedding
    imports = {by_id[e.dst].qualified_name for e in g.edges if e.type == "IMPORTS"}
    assert {"fmt", "acme/db"} <= imports
    # the method CONTAINS-attaches to its receiver type, not the file module
    contains = {(by_id[e.src].name, by_id[e.dst].name) for e in g.edges if e.type == "CONTAINS"}
    assert ("Order", "IsPaid") in contains


def test_go_receiver_calls_disambiguate(tmp_path):
    g = _graph(tmp_path)
    by_id = {n.id: n for n in g.nodes}
    calls = {(by_id[e.src].qualified_name, by_id[e.dst].qualified_name)
             for e in g.edges if e.type == "CALLS"}
    assert ("shop.Order.IsPaid", "shop.Order.check") in calls  # o.check() -> own type


def test_receiver_method_attaches_across_files(tmp_path):
    """Review finding: idiomatic Go declares a type in one file and its methods in
    another — ownership must resolve package-wide, not per-file."""
    (tmp_path / "a.go").write_text("package shop\n\ntype Invoice struct{}\n")
    (tmp_path / "b.go").write_text(
        "package shop\n\nfunc (i *Invoice) Pay() bool { return true }\n")
    g = build_graph(tmp_path)
    by_id = {n.id: n for n in g.nodes}
    contains = {(by_id[e.src].name, by_id[e.dst].name) for e in g.edges if e.type == "CONTAINS"}
    assert ("Invoice", "Pay") in contains
