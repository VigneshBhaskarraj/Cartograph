"""Local embeddings. Zero egress by default.

Two backends:
- `hash` (default): deterministic feature-hashed bag-of-words over code tokens.
  Fully offline, no model download, reproducible in CI. Captures lexical overlap,
  not deep synonymy — honest about being a fallback, not a semantic model.
- `ollama`: the product default for real semantic recall — a local code-aware model
  served by Ollama on 127.0.0.1. Opt-in via CARTOGRAPH_EMBEDDER=ollama. Still local;
  the only network call is to localhost, behind an explicit flag (SPEC directive 5).

Selection: CARTOGRAPH_EMBEDDER env var, or pass a backend explicitly.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import urllib.request

from .store import DEFAULT_DIM

def _warn_if_remote(host: str) -> None:
    """Guard the zero-egress promise: warn if Ollama isn't on loopback."""
    if not any(h in host for h in ("127.0.0.1", "localhost", "0.0.0.0", "::1")):
        import warnings
        warnings.warn(f"OLLAMA_HOST={host} is not loopback — code/queries leave this machine.",
                      stacklevel=3)


_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9]*")
_CAMEL_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")


def tokenize(text: str) -> list[str]:
    """snake_case and CamelCase aware tokenizer (matters for EXACT-mode recall)."""
    out: list[str] = []
    for raw in _TOKEN_RE.findall(text):
        for part in _CAMEL_RE.split(raw):
            for piece in part.split("_"):
                if piece:
                    out.append(piece.lower())
    return out


class HashEmbedder:
    """Feature-hashing embedder: deterministic, offline, dependency-free."""

    name = "hash"

    def __init__(self, dim: int = DEFAULT_DIM):
        self.dim = dim

    def embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for tok in tokenize(text):
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
            idx = h % self.dim
            sign = 1.0 if (h >> 8) & 1 else -1.0
            vec[idx] += sign
        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0:
            vec = [v / norm for v in vec]
        return vec

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]


class OllamaEmbedder:
    """Local Ollama embeddings. Network calls go only to 127.0.0.1."""

    def __init__(self, model: str | None = None, host: str | None = None, dim: int = DEFAULT_DIM):
        model = model or os.environ.get("CARTOGRAPH_OLLAMA_MODEL", "nomic-embed-text")
        self.name = f"ollama:{model}"
        self.model = model
        self.host = (host or os.environ.get("OLLAMA_HOST") or "http://127.0.0.1:11434").rstrip("/")
        _warn_if_remote(self.host)
        self.dim = dim

    def embed(self, text: str) -> list[float]:
        payload = json.dumps({"model": self.model, "prompt": text}).encode()
        req = urllib.request.Request(
            f"{self.host}/api/embeddings", data=payload, headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read())
        return data["embedding"]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]


def get_embedder(backend: str | None = None, dim: int = DEFAULT_DIM, model: str | None = None):
    backend = backend or os.environ.get("CARTOGRAPH_EMBEDDER", "hash")
    if backend == "hash":
        return HashEmbedder(dim=dim)
    if backend == "ollama":
        return OllamaEmbedder(model=model, dim=dim)
    raise ValueError(f"unknown embedder backend: {backend!r}")

