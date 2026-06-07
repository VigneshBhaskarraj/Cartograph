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


def test_rationale_node():
    """The `# WHY:` comment becomes a rationale node with a DOCUMENTS edge."""
    graph = extract_paths([FIX], root=FIX)
    rationale = _by_kind(graph, "rationale")
    assert any("bark" in n.docstring for n in rationale)
    assert any(e.type == "DOCUMENTS" for e in graph.edges)
