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


def test_graph_ppr_reaches_multihop(tmp_path):
    """PPR seeded on Dog reaches bark (Dog -CONTAINS-> speak -CALLS-> bark)."""
    store = _store(tmp_path)
    r = Retriever(store)
    hits = [i for i, _ in r.graph("Dog", k=6)]
    assert any("bark" in h for h in hits)
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


def test_rrf_fuse_weighted_signal_wins():
    """A heavily-weighted signal's top hit survives two signals that agree against it."""
    strong = ["a", "b", "c"]
    weak1 = ["z", "y", "x"]
    weak2 = ["z", "y", "x"]
    rankings = [strong, weak1, weak2]
    equal = [i for i, _ in rrf_fuse(rankings, k=3, rrf_k=60)]
    assert equal[0] == "z"  # equal weights: the two weak signals outvote the strong one
    weighted = [i for i, _ in rrf_fuse(rankings, k=3, rrf_k=60, weights=[3.0, 1.0, 1.0])]
    assert weighted[0] == "a"


def test_rrf_fuse_default_weights_match_unweighted():
    a, b = ["x", "y", "z"], ["y", "x", "w"]
    assert rrf_fuse([a, b], k=4) == rrf_fuse([a, b], k=4, weights=[1.0, 1.0])


def test_rrf_fuse_zero_weight_drops_signal():
    a, b = ["x", "y"], ["z", "w"]
    fused = [i for i, _ in rrf_fuse([a, b], k=4, weights=[1.0, 0.0])]
    assert fused == ["x", "y"]


def test_rrf_fuse_weight_count_mismatch():
    import pytest

    with pytest.raises(ValueError):
        rrf_fuse([["x"], ["y"]], weights=[1.0])


def test_hybrid_accepts_weights(tmp_path):
    store = _store(tmp_path)
    r = Retriever(store)
    hits = r.hybrid("bark sound", k=5, rrf_k=20, weights=(2.0, 0.5, 1.0), depth=50)
    assert hits and all(isinstance(i, str) and isinstance(s, float) for i, s in hits)
    store.close()


def test_retriever_rejects_mismatched_embedder_dim(tmp_path):
    """Audit M1: an explicit query embedder narrower/wider than the index must fail
    with a clear error at construction, not a numpy crash mid-query."""
    import pytest

    from cartograph.embed import HashEmbedder

    store = _store(tmp_path)  # indexed at dim=128
    with pytest.raises(ValueError, match="dim"):
        Retriever(store, embedder=HashEmbedder(dim=64))
    store.close()
