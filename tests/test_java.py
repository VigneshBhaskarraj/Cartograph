"""Java extraction (M7): structure, calls, nesting, and the JPA bridge."""

from pathlib import Path

import pytest

pytest.importorskip("tree_sitter_java")

from cartograph.pipeline import build_graph  # noqa: E402

FIX = Path(__file__).parent / "fixtures" / "Sample.java"


def _graph(tmp_path, extra_files=()):
    import shutil

    shutil.copy(FIX, tmp_path / "Sample.java")
    for name, text in extra_files:
        (tmp_path / name).write_text(text)
    return build_graph(tmp_path)


def test_java_nodes_and_edges(tmp_path):
    g = _graph(tmp_path)
    by_qual = {n.qualified_name: n for n in g.nodes}
    assert by_qual["com.acme.shop.Order"].kind == "class"
    assert by_qual["com.acme.shop.Order.isPaid"].kind == "method"
    assert by_qual["com.acme.shop.Auditable"].kind == "interface"
    # nested class carries true lexical scope (the audit-H2 lesson, applied up front)
    assert by_qual["com.acme.shop.Order.Builder"].kind == "class"
    assert by_qual["com.acme.shop.Order.Builder.build"].kind == "method"
    by_id = {n.id: n for n in g.nodes}
    inherits = {(by_id[e.src].name, by_id[e.dst].name) for e in g.edges if e.type == "INHERITS"}
    assert {("Order", "BaseEntity"), ("Order", "Auditable")} <= inherits
    imports = {(by_id[e.src].name, by_id[e.dst].qualified_name) for e in g.edges if e.type == "IMPORTS"}
    assert ("Sample", "javax.persistence.Entity") in imports  # external stub
    assert ("Sample", "com.acme.shop.Auditable") in imports   # resolved internal


def test_java_calls_with_this_receiver(tmp_path):
    g = _graph(tmp_path)
    by_id = {n.id: n for n in g.nodes}
    calls = {(by_id[e.src].qualified_name, by_id[e.dst].qualified_name)
             for e in g.edges if e.type == "CALLS"}
    assert ("com.acme.shop.Order.isPaid", "com.acme.shop.Order.checkTotal") in calls
    # `new Order()` inside Builder.build is a constructor call to the class
    assert ("com.acme.shop.Order.Builder.build", "com.acme.shop.Order") in calls


def test_jpa_entity_maps_to_table_and_column(tmp_path):
    pytest.importorskip("sqlglot")
    g = _graph(tmp_path, extra_files=[(
        "schema.sql",
        "CREATE TABLE orders (id INTEGER PRIMARY KEY, total_amount NUMERIC);\n")])
    by_id = {n.id: n for n in g.nodes}
    maps = {(by_id[e.src].qualified_name, by_id[e.dst].qualified_name)
            for e in g.edges if e.type == "MAPS_TO"}
    assert ("com.acme.shop.Order", "orders") in maps                 # @Table(name=...)
    assert ("com.acme.shop.Order", "orders.total_amount") in maps    # @Column(name=...)


def test_jpa_impact_blast_radius(tmp_path):
    """The moat query works on Java: dropping the column implicates the entity,
    its methods, and their callers."""
    pytest.importorskip("sqlglot")
    from cartograph.pipeline import index_path
    from cartograph.service import CartographService

    import shutil

    repo = tmp_path / "repo"
    repo.mkdir()
    shutil.copy(FIX, repo / "Sample.java")
    (repo / "schema.sql").write_text(
        "CREATE TABLE orders (id INTEGER PRIMARY KEY, total_amount NUMERIC);\n")
    db = tmp_path / "g.kuzu"
    index_path(repo, db, dim=32, overwrite=True).close()
    svc = CartographService(db)
    r = svc.impact("orders.total_amount")
    svc.close()
    quals = {n["qualified_name"] for n in r["direct_code"] + r["transitive_callers"]}
    assert any(q.endswith(".Order") for q in quals)
    assert any(q.endswith("Order.isPaid") for q in quals)  # entity method implicated


def test_enum_methods_extracted(tmp_path):
    """Review finding: enum methods live under enum_body_declarations."""
    (tmp_path / "Color.java").write_text(
        "package x;\n"
        "enum Color {\n"
        "    RED, GREEN;\n"
        "    public String label() { return name().toLowerCase(); }\n"
        "}\n")
    g = build_graph(tmp_path)
    quals = {n.qualified_name for n in g.nodes}
    assert "x.Color.label" in quals


def test_bare_entity_default_mapping_is_inferred(tmp_path):
    """Review finding: without @Table the name is a naming-strategy guess —
    the MAPS_TO must say INFERRED, and multi-word entities snake_case."""
    pytest.importorskip("sqlglot")
    (tmp_path / "OwnerPet.java").write_text(
        "package x;\nimport javax.persistence.Entity;\n"
        "@Entity\npublic class OwnerPet {}\n")
    (tmp_path / "schema.sql").write_text(
        "CREATE TABLE owner_pet (id INTEGER PRIMARY KEY);\n")
    g = build_graph(tmp_path)
    by_id = {n.id: n for n in g.nodes}
    maps = [(by_id[e.src].name, by_id[e.dst].name, e.confidence)
            for e in g.edges if e.type == "MAPS_TO"]
    assert ("OwnerPet", "owner_pet", "INFERRED") in maps
