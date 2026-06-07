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


def test_missing_db_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        CartographService(tmp_path / "nope.kuzu")
