from pathlib import Path

from cartograph.cache import EmbeddingCache, embed_key
from cartograph.embed import HashEmbedder
from cartograph.pipeline import build_graph, embed_graph, index_path

FIX = Path(__file__).parent / "fixtures" / "sample.py"


class CountingEmbedder(HashEmbedder):
    """HashEmbedder that records how many texts it actually embeds."""

    name = "counting"

    def __init__(self, dim=64):
        super().__init__(dim=dim)
        self.embedded = 0

    def embed_batch(self, texts):
        self.embedded += len(texts)
        return super().embed_batch(texts)


def test_cache_roundtrip(tmp_path):
    c = EmbeddingCache(tmp_path / "e.json")
    c.put("hello", [0.1, 0.2])
    c.save()
    again = EmbeddingCache.load(tmp_path / "e.json")
    assert again.get("hello") == [0.1, 0.2]
    assert again.get("missing") is None
    assert embed_key("hello") == embed_key("hello")


def test_cache_rejects_wrong_dim(tmp_path):
    c = EmbeddingCache(tmp_path / "e.json")
    c.put("x", [0.0, 0.0])
    assert c.get("x", dim=3) is None  # dim mismatch -> recompute
    assert c.get("x", dim=2) == [0.0, 0.0]


def test_embed_graph_reuses_cache(tmp_path):
    """Second pass over the same graph embeds nothing new."""
    emb = CountingEmbedder(dim=64)
    cache = EmbeddingCache(tmp_path / "e.json")
    g1 = build_graph(FIX)
    reused, embedded = embed_graph(g1, embedder=emb, cache=cache)
    assert reused == 0 and embedded == len(g1.nodes) and emb.embedded == len(g1.nodes)

    before = emb.embedded
    g2 = build_graph(FIX)
    reused2, embedded2 = embed_graph(g2, embedder=emb, cache=cache)
    assert embedded2 == 0 and reused2 == len(g2.nodes)
    assert emb.embedded == before  # no new embedding calls


def test_index_path_persists_and_reuses_cache(tmp_path):
    """Re-indexing the same source reuses the on-disk cache (0 recomputed)."""
    db = tmp_path / "g.kuzu"
    s1 = index_path(FIX, db, dim=128, overwrite=True)
    n = sum(1 for _ in s1.all_nodes_text())
    s1.close()
    assert s1.cache_stats == (0, n)  # first run: all computed

    s2 = index_path(FIX, db, dim=128, overwrite=True)
    reused, embedded = s2.cache_stats
    s2.close()
    assert embedded == 0 and reused == n  # second run: all reused


def test_unverified_dim_does_not_void_cache_hits(tmp_path):
    """Audit M2: an embedder whose declared dim is a guess (Ollama before its first
    call) must still reuse cached vectors of the model's true width."""
    from cartograph.model import Graph, Node
    from cartograph.pipeline import embed_graph

    class _LazyDim:
        name = "ollama:fake-1024"
        dim = 768            # the default guess
        dim_is_exact = False  # …but unconfirmed until a real call

        def embed_batch(self, texts):
            raise AssertionError("cache should have served every vector")

    cache = EmbeddingCache(tmp_path / "c.json")
    n = Node(id="x", kind="function", name="f", qualified_name="m.f", module="m",
             file_path="m.py", start_line=1, end_line=2, embed_text="def f()")
    cache.put(n.embed_text, [0.5] * 1024)  # true model width != declared dim
    reused, embedded = embed_graph(Graph(nodes=[n]), embedder=_LazyDim(), cache=cache)
    assert (reused, embedded) == (1, 0)
    assert len(n.embedding) == 1024


def test_corrupt_cache_resets_loudly(tmp_path):
    """G5-B4: a corrupted cache used to be dropped silently — for an
    Ollama-indexed monorepo that's hours of unexplained re-embedding."""
    import pytest

    p = tmp_path / "emb.hash.json"
    p.write_text("{not json")
    with pytest.warns(UserWarning, match="unreadable"):
        c = EmbeddingCache.load(p)
    assert c.data == {}


def test_save_is_atomic_no_tmp_left_behind(tmp_path):
    p = tmp_path / "emb.hash.json"
    c = EmbeddingCache(p)
    c.put("hello", [1.0, 2.0])
    c.save()
    assert EmbeddingCache.load(p).get("hello") == [1.0, 2.0]
    assert list(tmp_path.glob("*.tmp")) == []
