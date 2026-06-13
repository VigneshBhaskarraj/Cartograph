"""Second-stage reranking over the fused candidate pool. Zero egress by default.

The eval shows that once embeddings are strong, equal-weight RRF dilutes the best
leg's ordering. A reranker re-scores the fused top-K by (query, node-context)
relevance to restore MRR without losing fusion's recall/CROSS coverage.

Backends (same opt-in pattern as embeddings):
- `identity` (default): returns the fused order unchanged — offline, deterministic.
- `lexical`: deterministic token-overlap reorder — offline, used as a CI-testable
  stand-in and as the fallback when the LLM output can't be parsed.
- `ollama`: LLM-as-reranker. One *listwise* prompt per query asks a local model to
  order the candidates. Local only; the single network call goes to 127.0.0.1.

Selection: CARTOGRAPH_RERANKER env var, or pass a backend explicitly.
"""

from __future__ import annotations

import json
import os
import re
import urllib.request

from .embed import tokenize

# A candidate is (node_id, context_text).
Candidate = tuple[str, str]


class IdentityReranker:
    name = "identity"

    def rerank(self, query: str, candidates: list[Candidate]) -> list[str]:
        return [c[0] for c in candidates]


class LexicalReranker:
    """Deterministic token-overlap reorder. Offline; also the parse-failure fallback."""

    name = "lexical"

    def rerank(self, query: str, candidates: list[Candidate]) -> list[str]:
        q = set(tokenize(query))
        scored = []
        for i, (cid, text) in enumerate(candidates):
            toks = set(tokenize(text))
            overlap = len(q & toks)
            scored.append((-overlap, i, cid))  # i keeps the fused order stable on ties
        scored.sort()
        return [cid for _, _, cid in scored]


def _parse_order(text: str, n: int) -> list[int]:
    """Pull candidate indices (0..n-1) out of the model's reply, in order, deduped."""
    seen: set[int] = set()
    order: list[int] = []
    for tok in re.findall(r"\d+", text):
        i = int(tok)
        if 0 <= i < n and i not in seen:
            seen.add(i)
            order.append(i)
    return order


class OllamaReranker:
    """Listwise LLM reranker via Ollama generation. One call per query."""

    def __init__(self, model: str | None = None, host: str | None = None, max_chars: int = 320):
        model = model or os.environ.get("CARTOGRAPH_RERANK_MODEL", "llama3.2")
        self.name = f"ollama:{model}"
        self.model = model
        self.host = (host or os.environ.get("OLLAMA_HOST") or "http://127.0.0.1:11434").rstrip("/")
        from .embed import _check_loopback
        _check_loopback(self.host)
        self.max_chars = max_chars
        self._fallback = LexicalReranker()
        self._warned_fallback = False

    def _warn_fallback(self, exc: Exception) -> None:
        """A configured LLM reranker silently degrading to lexical on every query
        would invalidate anything measured 'with reranking on' — say so, once."""
        if not self._warned_fallback:
            import warnings
            warnings.warn(
                f"ollama reranker unavailable ({exc}); falling back to lexical "
                "reordering for this session", stacklevel=3)
            self._warned_fallback = True

    def _prompt(self, query: str, candidates: list[Candidate]) -> str:
        lines = [
            "You rank code-search results by how well they answer the query.",
            f"Query: {query}",
            "",
            "Candidates:",
        ]
        for i, (_, text) in enumerate(candidates):
            snippet = " ".join(text.split())[: self.max_chars]
            lines.append(f"[{i}] {snippet}")
        lines += [
            "",
            "Return ONLY the candidate numbers from most to least relevant, "
            "comma-separated (most relevant first). Example: 3,0,5,1",
        ]
        return "\n".join(lines)

    def rerank(self, query: str, candidates: list[Candidate]) -> list[str]:
        if not candidates:
            return []
        payload = json.dumps({
            "model": self.model,
            "prompt": self._prompt(query, candidates),
            "stream": False,
            "options": {"temperature": 0},
        }).encode()
        req = urllib.request.Request(
            f"{self.host}/api/generate", data=payload, headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                reply = json.loads(resp.read()).get("response", "")
        except (OSError, ValueError) as exc:  # unreachable/refused host, bad model, junk JSON
            self._warn_fallback(exc)
            return self._fallback.rerank(query, candidates)
        order = _parse_order(reply, len(candidates))
        if not order:
            return self._fallback.rerank(query, candidates)
        ranked = [candidates[i][0] for i in order]
        # Append any candidates the model dropped, preserving the fused order.
        ranked_set = set(ranked)
        ranked += [cid for cid, _ in candidates if cid not in ranked_set]
        return ranked


def get_reranker(backend: str | None = None, model: str | None = None):
    backend = backend or os.environ.get("CARTOGRAPH_RERANKER", "identity")
    if backend == "identity":
        return IdentityReranker()
    if backend == "lexical":
        return LexicalReranker()
    if backend == "ollama":
        return OllamaReranker(model=model)
    raise ValueError(f"unknown reranker backend: {backend!r}")
