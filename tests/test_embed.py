from cartograph.embed import HashEmbedder, get_embedder, tokenize


def test_tokenize_splits_cases():
    """snake_case and CamelCase split into lowercase tokens (EXACT-mode recall)."""
    toks = tokenize("GZipDecoder send_single_request")
    assert {"g", "zip", "decoder", "send", "single", "request"} <= set(toks)


def test_fake_embedding_dim():
    """M0-5: the default (offline) embedder is deterministic and right-sized."""
    e = HashEmbedder(dim=64)
    a = e.embed("def speak(self) -> str")
    b = e.embed("def speak(self) -> str")
    assert len(a) == 64
    assert a == b  # deterministic
    assert e.embed("totally different text") != a


def test_get_embedder_default_is_offline():
    assert get_embedder().name == "hash"
