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


def test_plain_javascript_extraction(tmp_path):
    """The JS grammar shares this walker's vocabulary — .js repos must index."""
    pytest.importorskip("tree_sitter_javascript")
    import shutil

    fix = Path(__file__).parent / "fixtures" / "sample.js"
    shutil.copy(fix, tmp_path / "sample.js")
    g = build_graph(tmp_path)
    kinds = {(n.name, n.kind) for n in g.nodes}
    assert ("Animal", "class") in kinds and ("Dog", "class") in kinds
    assert ("bark", "function") in kinds and ("greet", "function") in kinds
    assert sum(1 for n in g.nodes if n.name == "speak" and n.kind == "method") == 2
    by_id = {n.id: n for n in g.nodes}
    etypes = {e.type for e in g.edges}
    assert {"CONTAINS", "INHERITS", "IMPORTS", "CALLS"} <= etypes
    inherits = [(by_id[e.src].name, by_id[e.dst].name) for e in g.edges if e.type == "INHERITS"]
    assert ("Dog", "Animal") in inherits
    calls = {(by_id[e.src].name, by_id[e.dst].name) for e in g.edges if e.type == "CALLS"}
    assert ("speak", "bark") in calls


def test_mixed_ts_and_js_in_one_graph(tmp_path):
    pytest.importorskip("tree_sitter_javascript")
    (tmp_path / "a.ts").write_text("export function tsf(): number { return 1; }\n")
    (tmp_path / "b.js").write_text("function jsf() { return tsf(); }\n")
    g = build_graph(tmp_path)
    names = {n.name for n in g.nodes if n.kind == "function"}
    assert {"tsf", "jsf"} <= names


def test_commonjs_patterns(tmp_path):
    """Pre-ES6 Node style: exports.f, Foo.prototype.m, require() — most of the
    older JS ecosystem defines code this way (Express indexes 3x richer with it)."""
    pytest.importorskip("tree_sitter_javascript")
    (tmp_path / "app.js").write_text(
        "var util = require('./util');\n"
        "exports.init = function () {\n"
        "  return util.go();\n"
        "};\n"
        "function Router() {}\n"
        "Router.prototype.dispatch = function dispatch(req) {\n"
        "  return this.stack(req);\n"
        "};\n"
        "module.exports = function createApp() {\n"
        "  return Router();\n"
        "};\n")
    (tmp_path / "util.js").write_text("exports.go = function () { return 1; };\n")
    g = build_graph(tmp_path)
    kinds = {}
    for n in g.nodes:  # quals carry the tmpdir-derived package prefix — match by suffix
        for want in ("app.init", "app.Router.dispatch", "app.createApp", "util.go"):
            if n.qualified_name.endswith(want):
                kinds[want] = n.kind
    assert kinds["app.init"] == "function"            # exports.init
    assert kinds["app.Router.dispatch"] == "method"   # prototype method
    assert kinds["app.createApp"] == "function"       # named module.exports fn
    assert kinds["util.go"] == "function"
    by_id = {n.id: n for n in g.nodes}
    imports = {(by_id[e.src].name, by_id[e.dst].name) for e in g.edges if e.type == "IMPORTS"}
    assert ("app", "util") in imports                         # require('./util')
    calls = {(by_id[e.src].name, by_id[e.dst].name) for e in g.edges if e.type == "CALLS"}
    assert ("createApp", "Router") in calls


def test_polyglot_external_stub_dedup(tmp_path):
    """Review HIGH: import redis (py) + require('redis') (js) minted the same
    ext::redis primary key and crashed indexing on polyglot repos."""
    pytest.importorskip("tree_sitter_javascript")
    from cartograph.pipeline import index_path

    (tmp_path / "a.py").write_text("import redis\n\ndef f():\n    return 1\n")
    (tmp_path / "b.js").write_text("var redis = require('redis');\nfunction g() { return 1; }\n")
    store = index_path(tmp_path, tmp_path / "g.kuzu", dim=16, overwrite=True)
    counts = store.counts()
    store.close()
    assert counts.get("node:external", 0) == 1  # one stub, indexed without crashing


def test_js_this_and_computed_assignments_skipped(tmp_path):
    pytest.importorskip("tree_sitter_javascript")
    (tmp_path / "a.js").write_text(
        "function Ctor() {\n  this.x = function () { return 1; };\n}\n"
        "var obj = {};\nvar key = 'k';\nobj[key] = function dyn() { return 2; };\n")
    g = build_graph(tmp_path)
    names = {n.name for n in g.nodes if n.kind in ("function", "method")}
    assert "Ctor" in names
    assert "x" not in names and "dyn" not in names  # junk paths skipped


def test_generic_type_args_are_not_inheritance(tmp_path):
    """G5-C1: `extends Component<Props, State>` names ONE base. The old walk
    harvested Props and State from type_arguments too — and because each name
    was unique, every bogus edge got EXTRACTED confidence."""
    src = tmp_path / "view.ts"
    src.write_text(
        "interface Props { x: number }\n"
        "interface State { y: number }\n"
        "class Component<P, S> { }\n"
        "class MyView extends Component<Props, State> { }\n"
        "class Impl implements Props { }\n"
    )
    g = build_graph(src)
    by_id = {n.id: n for n in g.nodes}
    inh = {(by_id[e.src].name, by_id[e.dst].name) for e in g.edges if e.type == "INHERITS"}
    assert ("MyView", "Component") in inh
    assert ("MyView", "Props") not in inh and ("MyView", "State") not in inh
    assert ("Impl", "Props") in inh  # implements still works


def test_arrow_callback_calls_are_captured(tmp_path):
    """G5-C4: `items.forEach(i => doWork())` produced NO CALLS edge while the
    equivalent function-expression callback did — arrows are the dominant
    modern style, so this was a major recall hole."""
    src = tmp_path / "m.ts"
    src.write_text(
        "function doWork(i: number) { return i; }\n"
        "function run(items: number[]) {\n"
        "    items.forEach(i => doWork(i));\n"
        "}\n"
    )
    g = build_graph(src)
    by_id = {n.id: n for n in g.nodes}
    calls = {(by_id[e.src].name, by_id[e.dst].name) for e in g.edges if e.type == "CALLS"}
    assert ("run", "doWork") in calls


def test_class_field_arrow_methods_extracted(tmp_path):
    """G5-C4: `handleClick = () => {…}` (the React handler idiom) produced no
    node and lost its calls."""
    src = tmp_path / "view.ts"
    src.write_text(
        "class View {\n"
        "    render() { return 1; }\n"
        "    handleClick = () => { this.render(); };\n"
        "}\n"
    )
    g = build_graph(src)
    by_id = {n.id: n for n in g.nodes}
    kinds = {(n.name, n.kind) for n in g.nodes}
    assert ("handleClick", "method") in kinds
    calls = {(by_id[e.src].name, by_id[e.dst].name) for e in g.edges if e.type == "CALLS"}
    assert ("handleClick", "render") in calls


def test_nested_decls_in_assigned_fn_do_not_leak_to_module_scope(tmp_path):
    """G5-C4: `exports.mw = function () { function inner() {} }` minted a
    module-level `module.inner` function — wrong scope, phantom qual."""
    src = tmp_path / "mw.js"
    src.write_text(
        "exports.mw = function () {\n"
        "    function inner() { return 1; }\n"
        "    return inner;\n"
        "};\n"
    )
    g = build_graph(src)
    quals = {n.qualified_name for n in g.nodes if n.kind == "function"}
    assert "mw.mw" in quals
    assert "mw.inner" not in quals  # the scope leak
