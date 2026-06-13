from pathlib import Path

import pytest

from cartograph.pipeline import index_path
from cartograph.service import CartographService, embedder_from_store
from cartograph.store import Store

FIX = Path(__file__).parent / "fixtures" / "sample.py"


@pytest.fixture
def db(tmp_path):
    store = index_path(FIX, tmp_path / "g.kuzu", dim=128, overwrite=True)
    store.close()
    return tmp_path / "g.kuzu"


def test_index_records_embedder_meta(db):
    store = Store(db)
    assert store.get_meta("embedder_backend") == "hash"
    assert store.get_meta("embedding_dim") == "128"
    store.close()


def test_embedder_from_store_matches_dim(db):
    store = Store(db)
    emb = embedder_from_store(store)
    assert emb is not None and len(emb.embed("hello")) == 128
    store.close()


def test_query_returns_nodes(db):
    svc = CartographService(db)
    hits = svc.query("dog bark sound", mode="hybrid", k=5)
    assert hits and all({"id", "qualified_name", "kind", "score"} <= set(h) for h in hits)
    assert any("bark" in h["id"] for h in hits)
    svc.close()


def test_semantic_search_and_get_node(db):
    svc = CartographService(db)
    hits = svc.semantic_search("greet someone by name", k=5)
    assert any("greet" in h["id"] for h in hits)
    node = svc.get_node(hits[0]["id"])
    assert node and node["id"] == hits[0]["id"]
    svc.close()


def test_neighbors_and_modes(db):
    svc = CartographService(db)
    dog = next(h["id"] for h in svc.query("Dog", mode="lexical", k=5) if h["name"] == "Dog")
    nbrs = svc.neighbors(dog, hops=2)
    assert any("bark" in n["id"] for n in nbrs)
    assert {"vector", "graph", "lexical", "hybrid"} <= svc.modes
    with pytest.raises(ValueError):
        svc.query("x", mode="bogus")
    svc.close()


def test_calls_and_callers_directed(db):
    """calls/callers expose direction so the agent never guesses (Dog.speak -> bark)."""
    svc = CartographService(db)
    speak = next(h["id"] for h in svc.query("speak", mode="lexical", k=10)
                 if h["qualified_name"].endswith("Dog.speak"))
    bark = next(h["id"] for h in svc.query("bark", mode="lexical", k=10)
                if h["name"] == "bark")
    callees = svc.calls(speak)
    assert any(c["id"] == bark for c in callees)
    assert all(c["relation"] == "CALLS" and c["direction"] == "out" for c in callees)
    callers = svc.callers(bark)
    assert any(c["id"] == speak for c in callers)
    assert all(c["direction"] == "in" for c in callers)
    svc.close()


def test_neighbors_labeled_and_filtered(db):
    svc = CartographService(db)
    dog = next(h["id"] for h in svc.query("Dog", mode="lexical", k=10)
               if h["name"] == "Dog" and h["kind"] == "class")
    inh = svc.neighbors(dog, direction="out", relation="INHERITS")
    assert inh and all(n["relation"] == "INHERITS" and n["direction"] == "out" for n in inh)
    assert any(n["name"] == "Animal" for n in inh)
    svc.close()


def test_tools_accept_qualified_name(db):
    """The friction from dogfooding: tools resolve a qualified name, not just the id."""
    svc = CartographService(db)
    # get_node by qualified name (no internal #line id needed)
    node = svc.get_node("m.Dog.speak") or svc.get_node("Dog.speak")
    assert node and node["qualified_name"].endswith("Dog.speak")
    # calls by qualified name resolves and returns the callee
    callees = svc.calls("Dog.speak")
    assert any(c["name"] == "bark" for c in callees)
    # resolve surfaces candidates for a bare name
    assert any(r["name"] == "bark" for r in svc.resolve("bark"))
    svc.close()


def test_unknown_ref_contract(db):
    """get_node stays lenient (None) for the CLI's checks; traversal tools and
    strict mode raise — 'unknown symbol' must not read as 'known, no edges'."""
    svc = CartographService(db)
    assert svc.get_node("does.not.exist") is None
    with pytest.raises(ValueError, match="no node matches"):
        svc.calls("does.not.exist")
    with pytest.raises(ValueError, match="no node matches"):
        svc.get_node("does.not.exist", strict=True)
    svc.close()


def test_shortest_path(db):
    """Regression: shortest_path must return a real path (Kuzu list-comp bug)."""
    svc = CartographService(db)
    path = svc.shortest_path("Dog.speak", "bark")  # Dog.speak -CALLS-> bark
    assert len(path) >= 2
    assert path[0]["qualified_name"].endswith("Dog.speak")
    assert path[-1]["name"] == "bark"
    svc.close()


def test_rerank_mode_reachable_via_env(db, monkeypatch):
    """Regression: `rerank` mode must actually rerank when configured, not silently hybrid."""
    monkeypatch.setenv("CARTOGRAPH_RERANKER", "lexical")
    svc = CartographService(db)
    assert "rerank" in svc.modes
    assert svc.query("dog bark sound", mode="rerank", k=5)
    svc.close()


def test_rerank_not_advertised_without_config(db, monkeypatch):
    monkeypatch.delenv("CARTOGRAPH_RERANKER", raising=False)
    svc = CartographService(db)
    assert "rerank" not in svc.modes
    svc.close()


def test_missing_db_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        CartographService(tmp_path / "nope.kuzu")


def test_neighbors_hops_clamped_not_crashing(db):
    """hops=50 used to leak a raw Kuzu binder error (ceiling 30); now it clamps
    to MAX_HOPS and says so in a trailing note instead of erroring."""
    svc = CartographService(db)
    dog = next(h["id"] for h in svc.query("Dog", mode="lexical", k=5) if h["name"] == "Dog")
    nbrs = svc.neighbors(dog, hops=50)
    assert "clamped" in nbrs[-1]["note"]
    clamped_ids = {n["id"] for n in nbrs if "id" in n}
    max_ids = {n["id"] for n in svc.neighbors(dog, hops=8) if "id" in n}
    assert clamped_ids == max_ids
    svc.close()


def test_invalid_direction_and_relation_raise(db):
    """A typo'd filter must be an error, not an empty result — agents act on
    the difference between 'no callers' and 'I asked the question wrong'."""
    svc = CartographService(db)
    dog = next(h["id"] for h in svc.query("Dog", mode="lexical", k=5) if h["name"] == "Dog")
    with pytest.raises(ValueError, match="direction"):
        svc.neighbors(dog, direction="sideways")
    with pytest.raises(ValueError, match="CALL"):
        svc.neighbors(dog, relation="CALL")  # typo for CALLS — list valid values
    svc.close()


def test_resolve_caps_with_truncation_sentinel(tmp_path):
    """A bare name matching half the codebase must not flood the agent's
    context — capped at 25 with an explicit {truncated, total} sentinel."""
    src = tmp_path / "many.py"
    src.write_text("\n".join(
        f"class C{i}:\n    def run(self):\n        pass" for i in range(30)))
    index_path(src, tmp_path / "g.kuzu", dim=64, overwrite=True).close()
    svc = CartographService(tmp_path / "g.kuzu")
    out = svc.resolve("run")
    assert len(out) == 26  # 25 nodes + sentinel
    sentinel = out[-1]
    assert sentinel["truncated"] is True and sentinel["total"] == 30
    assert all("id" in n for n in out[:-1])
    svc.close()


def test_unknown_ref_suggests_candidates(db):
    """Typo'd refs come back with 'did you mean' — the agent's recovery path.
    Policy is tiered: fuzzy on the last segment, then substring (G5-A3)."""
    svc = CartographService(db)
    with pytest.raises(ValueError, match="did you mean.*speak"):
        svc.calls("Dog.speek")  # typo -> fuzzy tier
    assert any(q.endswith("speak") for q in svc.suggest("speek"))
    svc.close()


def test_schema_gate_rejects_each_layer(db):
    """G5-B2: the gate must catch missing tables, missing/mismatched version,
    AND missing CodeNode columns — each with a message naming the problem.
    (Previously a versionless or column-shy graph passed and crashed mid-query.)"""
    import kuzu

    from cartograph.service import open_graph

    # version mismatch
    s = Store(db)
    s.set_meta("schema_version", "0")
    s.close()
    with pytest.raises(RuntimeError, match="schema_version 0"):
        open_graph(db)
    # missing version (treated as incompatible, not as a free pass)
    s = Store(db)
    s.delete_meta("schema_version")
    s.close()
    with pytest.raises(RuntimeError, match="no recorded schema_version"):
        open_graph(db)
    s = Store(db)
    s.set_meta("schema_version", "1")
    s.close()
    # missing CodeNode column (schema drift without a version bump)
    conn = kuzu.Connection(kuzu.Database(str(db)))
    conn.execute("ALTER TABLE CodeNode DROP module")
    conn.close()
    with pytest.raises(RuntimeError, match="missing columns: module"):
        open_graph(db)


def test_schema_gate_requires_meta_table(tmp_path):
    """A Meta-less graph can't prove its version — incompatible, not 'fine'."""
    import kuzu

    from cartograph.service import open_graph

    index_path(FIX, tmp_path / "g.kuzu", dim=64, overwrite=True).close()
    conn = kuzu.Connection(kuzu.Database(str(tmp_path / "g.kuzu")))
    conn.execute("DROP TABLE Meta")
    conn.close()
    with pytest.raises(RuntimeError, match="missing tables: Meta"):
        open_graph(tmp_path / "g.kuzu")


def test_invalid_filters_raise_even_multihop(db):
    """Review follow-up: hops>1 ignores filters by design (unlabeled expansion),
    but a typo'd filter must still raise, not silently drop."""
    svc = CartographService(db)
    with pytest.raises(ValueError, match="direction"):
        svc.neighbors("Dog", direction="sideways", hops=3)
    with pytest.raises(ValueError, match="relation"):
        svc.neighbors("Dog", relation="CALL", hops=3)
    svc.close()


def test_neighbors_surface_edge_confidence(tmp_path):
    """G6-4: the EXTRACTED/INFERRED honesty lives in the graph — agents must see it
    on every labeled edge (calls/callers/neighbors)."""
    from cartograph.pipeline import index_path
    from cartograph.service import CartographService

    fix = Path(__file__).parent / "fixtures" / "sample.py"
    db = tmp_path / "g.kuzu"
    index_path(fix, db, dim=32, overwrite=True).close()
    svc = CartographService(db)
    # CALLS edges are heuristic → INFERRED
    callers = svc.callers("bark")
    assert callers and all(c["confidence"] in ("EXTRACTED", "INFERRED") for c in callers)
    assert any(c["confidence"] == "INFERRED" for c in callers)  # speak->bark is a heuristic call
    # CONTAINS is deterministic structure → EXTRACTED
    contained = svc.neighbors("Dog", direction="out", relation="CONTAINS")
    assert contained and all(c["confidence"] == "EXTRACTED" for c in contained)
    svc.close()
