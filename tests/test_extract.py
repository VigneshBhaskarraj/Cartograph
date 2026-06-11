from pathlib import Path

from cartograph.extract import extract_paths, extract_source

FIX = Path(__file__).parent / "fixtures" / "sample.py"


def _by_kind(graph, kind):
    return [n for n in graph.nodes if n.kind == kind]


def test_single_function():
    """M0-3: a one-function module yields a module node, a function node, and CONTAINS."""
    src = "def f():\n    return 1\n"
    fx = extract_source(src, "f.py", "f")
    kinds = sorted(n.kind for n in fx.nodes)
    assert "module" in kinds and "function" in kinds
    fn = next(n for n in fx.nodes if n.kind == "function")
    assert fn.name == "f"
    contains = [e for e in fx.edges if e.type == "CONTAINS"]
    assert any(e.dst == fn.id for e in contains)


def test_classes_methods_inheritance():
    """M1-2: classes, methods, inheritance, and call edges are all captured."""
    graph = extract_paths([FIX], root=FIX)
    classes = {n.name for n in _by_kind(graph, "class")}
    assert {"Animal", "Dog"} <= classes
    methods = {n.name for n in _by_kind(graph, "method")}
    assert "speak" in methods
    inherits = [e for e in graph.edges if e.type == "INHERITS"]
    dog = next(n for n in graph.nodes if n.name == "Dog" and n.kind == "class")
    animal = next(n for n in graph.nodes if n.name == "Animal" and n.kind == "class")
    assert any(e.src == dog.id and e.dst == animal.id for e in inherits)


def test_call_edge_inferred():
    """Dog.speak calls bark(); the call edge exists and is tagged INFERRED."""
    graph = extract_paths([FIX], root=FIX)
    bark = next(n for n in graph.nodes if n.name == "bark")
    calls = [e for e in graph.edges if e.type == "CALLS" and e.dst == bark.id]
    assert calls and all(e.confidence == "INFERRED" for e in calls)


def test_self_call_resolves_to_own_class(tmp_path):
    """`self._run()` in Foo binds to Foo._run, not Bar._run (sync/async disambiguation)."""
    src = (
        "class Foo:\n"
        "    def go(self):\n"
        "        return self._run()\n"
        "    def _run(self):\n"
        "        return 1\n\n"
        "class Bar:\n"
        "    def go(self):\n"
        "        return self._run()\n"
        "    def _run(self):\n"
        "        return 2\n"
    )
    p = tmp_path / "m.py"
    p.write_text(src)
    g = extract_paths([p], root=p)

    def node(suffix):
        return next(n for n in g.nodes if n.qualified_name.endswith(suffix))

    foo_go, foo_run, bar_run = node("Foo.go"), node("Foo._run"), node("Bar._run")
    calls = {(e.src, e.dst) for e in g.edges if e.type == "CALLS"}
    assert (foo_go.id, foo_run.id) in calls
    assert (foo_go.id, bar_run.id) not in calls  # no phantom cross-class edge


def test_jedi_resolves_receiver_type(tmp_path):
    """With resolver=jedi, `a.read()` (a: A) binds to A.read — not B.read — using type
    inference the name heuristic can't do."""
    import pytest

    pytest.importorskip("jedi")
    src = (
        "class A:\n    def read(self):\n        return 1\n\n"
        "class B:\n    def read(self):\n        return 2\n\n"
        "def use():\n    a = A()\n    return a.read()\n"
    )
    p = tmp_path / "m.py"
    p.write_text(src)
    g = extract_paths([p], root=p, resolver="jedi")

    def node(suffix):
        return next(n for n in g.nodes if n.qualified_name.endswith(suffix))

    use, a_read, b_read = node("m.use"), node("A.read"), node("B.read")
    calls = {(e.src, e.dst) for e in g.edges if e.type == "CALLS"}
    assert (use.id, a_read.id) in calls
    assert (use.id, b_read.id) not in calls


def test_rationale_node():
    """The `# WHY:` comment becomes a rationale node with a DOCUMENTS edge."""
    graph = extract_paths([FIX], root=FIX)
    rationale = _by_kind(graph, "rationale")
    assert any("bark" in n.docstring for n in rationale)
    assert any(e.type == "DOCUMENTS" for e in graph.edges)


def test_nested_scopes_get_true_qualified_names():
    """Audit H2: nested defs/classes must carry their full lexical scope — a local
    helper inside a method is NOT a method of the class (phantom-edge source)."""
    src = (
        "class Outer:\n"
        "    class Inner:\n"
        "        def m(self):\n"
        "            pass\n"
        "    def method(self):\n"
        "        def helper():\n"
        "            pass\n"
        "        return helper\n"
        "\n"
        "def top():\n"
        "    def inner():\n"
        "        pass\n"
    )
    fx = extract_source(src, "m.py", "m")
    by_qual = {n.qualified_name: n for n in fx.nodes}
    assert "m.Outer.Inner" in by_qual and by_qual["m.Outer.Inner"].kind == "class"
    assert by_qual["m.Outer.Inner.m"].kind == "method"
    helper = by_qual["m.Outer.method.helper"]
    assert helper.kind == "function"  # local def in a method is not a method
    assert "m.helper" not in by_qual and "m.Outer.helper" not in by_qual
    assert by_qual["m.top.inner"].kind == "function"
    assert "m.inner" not in by_qual


def test_local_helper_does_not_steal_self_calls():
    """Audit H2 downstream: `self.close()` in a class with no `close` method must not
    bind to a local helper that previously masqueraded as `A.close`."""
    src = (
        "class A:\n"
        "    def run(self):\n"
        "        def close():\n"
        "            pass\n"
        "        self.close()\n"
    )
    graph = extract_paths_from_source(src)
    by_id = {n.id: n for n in graph.nodes}
    calls = [(by_id[e.src].qualified_name, by_id[e.dst].qualified_name)
             for e in graph.edges if e.type == "CALLS"]
    # The same-class heuristic must not match m.A.run.close as a method of A.
    assert ("m.A.run", "m.A.close") not in calls


def extract_paths_from_source(src, tmp_name="m.py"):
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / tmp_name
        p.write_text(src)
        return extract_paths([p], root=p)


def test_ambiguous_inheritance_is_inferred(tmp_path):
    """Audit L2: with two same-named base classes, every INHERITS edge is a guess."""
    (tmp_path / "a.py").write_text("class Base:\n    pass\n")
    (tmp_path / "b.py").write_text("class Base:\n    pass\n")
    (tmp_path / "c.py").write_text("class Child(Base):\n    pass\n")
    graph = extract_paths(sorted(tmp_path.glob("*.py")), root=tmp_path)
    inherits = [e for e in graph.edges if e.type == "INHERITS"]
    assert len(inherits) == 2
    assert all(e.confidence == "INFERRED" for e in inherits)
