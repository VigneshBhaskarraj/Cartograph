import pytest

from cartograph.embed import HashEmbedder, OllamaEmbedder, get_embedder, tokenize


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


def test_remote_ollama_host_is_refused(monkeypatch):
    """Zero-egress is enforced, not just warned about (no network call is made:
    the check runs in the constructor, before any request)."""
    monkeypatch.setenv("OLLAMA_HOST", "http://ollama.example.com:11434")
    monkeypatch.delenv("CARTOGRAPH_ALLOW_REMOTE_OLLAMA", raising=False)
    with pytest.raises(RuntimeError, match="zero-egress"):
        OllamaEmbedder()


def test_remote_ollama_lookalike_host_is_refused(monkeypatch):
    monkeypatch.setenv("OLLAMA_HOST", "http://localhost.evil.com:11434")
    monkeypatch.delenv("CARTOGRAPH_ALLOW_REMOTE_OLLAMA", raising=False)
    with pytest.raises(RuntimeError, match="zero-egress"):
        OllamaEmbedder()


def test_remote_ollama_explicit_optout_warns(monkeypatch):
    monkeypatch.setenv("OLLAMA_HOST", "http://ollama.example.com:11434")
    monkeypatch.setenv("CARTOGRAPH_ALLOW_REMOTE_OLLAMA", "1")
    with pytest.warns(UserWarning, match="not loopback"):
        OllamaEmbedder()


def test_loopback_ollama_host_is_accepted(monkeypatch):
    monkeypatch.delenv("CARTOGRAPH_ALLOW_REMOTE_OLLAMA", raising=False)
    for host in ("http://127.0.0.1:11434", "http://localhost:11434", "localhost:11434"):
        monkeypatch.setenv("OLLAMA_HOST", host)
        OllamaEmbedder()  # constructor must not raise
