from pathlib import Path

from cartograph.pipeline import index_path
from cartograph.rerank import IdentityReranker, LexicalReranker, _parse_order, get_reranker
from cartograph.retrieve import Retriever

FIX = Path(__file__).parent / "fixtures" / "sample.py"


def test_identity_preserves_order():
    cands = [("a", "x"), ("b", "y"), ("c", "z")]
    assert IdentityReranker().rerank("q", cands) == ["a", "b", "c"]


def test_lexical_reranker_orders_by_overlap():
    cands = [
        ("a", "completely unrelated tokens"),
        ("b", "the quick brown fox jumps"),
    ]
    # Query overlaps b strongly -> b first.
    assert LexicalReranker().rerank("quick brown fox", cands)[0] == "b"


def test_parse_order_extracts_indices():
    assert _parse_order("3, 0, 5, 1", 6) == [3, 0, 5, 1]
    assert _parse_order("nonsense", 6) == []
    assert _parse_order("9,1,1,2", 3) == [1, 2]  # drops out-of-range and dupes


def test_get_reranker_default_offline():
    assert get_reranker().name == "identity"


def test_retriever_rerank_mode(tmp_path):
    """rerank mode returns a valid, reranked top-k using an offline reranker."""
    store = index_path(FIX, tmp_path / "g.kuzu", dim=128, overwrite=True)
    r = Retriever(store, reranker=LexicalReranker())
    hits = r.retrieve("dog bark sound", mode="rerank", k=5)
    assert hits and all(isinstance(i, str) and isinstance(s, float) for i, s in hits)
    # Scores are descending.
    assert [s for _, s in hits] == sorted((s for _, s in hits), reverse=True)
    store.close()
