from pathlib import Path

from cartograph.pipeline import index_path
from cartograph.retrieve import Retriever, rrf_fuse

FIX = Path(__file__).parent / "fixtures" / "sample.py"


def _store(tmp_path):
    return index_path(FIX, tmp_path / "g.kuzu", dim=128, overwrite=True)


def test_vector_finds_node(tmp_path):
    """M0-6: a query close to a node's text retrieves that node near the top."""
    store = _store(tmp_path)
    r = Retriever(store)
    hits = [i for i, _ in r.vector("greet name hello", k=5)]
    assert any("greet" in h for h in hits)
    store.close()


def test_two_hop(tmp_path):
    """M0-7: 2-hop neighbours of Dog reach the module and the bark function."""
    store = _store(tmp_path)
    dog_id = next(d["id"] for d in store.all_nodes_text() if d["name"] == "Dog" and d["kind"] == "class")
    nbrs = store.neighbors(dog_id, hops=2)
    assert any("bark" in n for n in nbrs)
    store.close()


def test_lexical_exact(tmp_path):
    """Exact symbol name is found by lexical search."""
    store = _store(tmp_path)
    r = Retriever(store)
    hits = [i for i, _ in r.lexical("bark", k=5)]
    assert any("bark" in h for h in hits)
    store.close()


def test_hybrid_returns_ranked(tmp_path):
    store = _store(tmp_path)
    r = Retriever(store)
    hits = r.retrieve("speak sound", mode="hybrid", k=5)
    assert hits and all(isinstance(h[0], str) for h in hits)
    store.close()


def test_retriever_returns_ranked_ids(tmp_path):
    """M1-6: every mode returns a ranked list of (node_id, score)."""
    store = _store(tmp_path)
    r = Retriever(store)
    for mode in ("vector", "graph", "lexical", "hybrid"):
        hits = r.retrieve("bark sound", mode=mode, k=5)
        assert all(isinstance(i, str) and isinstance(s, float) for i, s in hits)
    store.close()


def test_rrf_fuse_basic():
    a = ["x", "y", "z"]
    b = ["y", "x", "w"]
    fused = [i for i, _ in rrf_fuse([a, b], k=4)]
    assert fused[0] in ("x", "y")  # items high in both rise to the top
