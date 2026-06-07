"""Hybrid retrieval: vector + graph + lexical, fused with RRF.

The core differentiator. Each signal runs independently and returns a ranked list
of node ids; RRF fuses them parameter-free. No source files are read here — only
the graph (SPEC invariant). All offline.
"""

from __future__ import annotations

import math
from collections import defaultdict

import numpy as np

from .embed import get_embedder, tokenize
from .store import Store


def _cosine_ranking(query_vec: list[float], ids: list[str], vecs: list[list[float]]) -> list[tuple[str, float]]:
    if not ids:
        return []
    mat = np.array([v if v else [0.0] * len(query_vec) for v in vecs], dtype=np.float32)
    q = np.array(query_vec, dtype=np.float32)
    qn = np.linalg.norm(q)
    mn = np.linalg.norm(mat, axis=1)
    denom = (mn * qn)
    denom[denom == 0] = 1e-9
    scores = (mat @ q) / denom
    order = np.argsort(-scores)
    return [(ids[i], float(scores[i])) for i in order]


class Retriever:
    """Loads the queryable indexes from the store once, answers many queries."""

    def __init__(self, store: Store, embedder=None):
        self.store = store
        self.ids, self.vecs = store.all_embeddings()
        if embedder is None:
            dim = next((len(v) for v in self.vecs if v), None)
            embedder = get_embedder(dim=dim) if dim else get_embedder()
        self.embedder = embedder
        self.docs = store.all_nodes_text()  # id, kind, name, qualified_name, embed_text, docstring
        self._build_lexical()

    # -- lexical (BM25) -------------------------------------------------------
    def _build_lexical(self) -> None:
        self.doc_tokens: dict[str, list[str]] = {}
        df: dict[str, int] = defaultdict(int)
        total_len = 0
        for d in self.docs:
            toks = tokenize(f"{d['name']} {d['qualified_name']} {d['embed_text']} {d['docstring']}")
            self.doc_tokens[d["id"]] = toks
            total_len += len(toks)
            for t in set(toks):
                df[t] += 1
        self.N = max(1, len(self.docs))
        self.avgdl = (total_len / self.N) if self.N else 1.0
        self.idf = {t: math.log(1 + (self.N - n + 0.5) / (n + 0.5)) for t, n in df.items()}
        self.tf: dict[str, dict[str, int]] = {}
        for doc_id, toks in self.doc_tokens.items():
            counts: dict[str, int] = defaultdict(int)
            for t in toks:
                counts[t] += 1
            self.tf[doc_id] = counts

    def lexical(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        q_toks = set(tokenize(query))
        k1, b = 1.5, 0.75
        scored: list[tuple[str, float]] = []
        for doc_id, counts in self.tf.items():
            dl = len(self.doc_tokens[doc_id]) or 1
            s = 0.0
            for t in q_toks:
                if t in counts:
                    idf = self.idf.get(t, 0.0)
                    f = counts[t]
                    s += idf * (f * (k1 + 1)) / (f + k1 * (1 - b + b * dl / self.avgdl))
            if s > 0:
                scored.append((doc_id, s))
        scored.sort(key=lambda x: -x[1])
        return scored[:k]

    # -- vector ---------------------------------------------------------------
    def vector(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        qv = self.embedder.embed(query)
        return _cosine_ranking(qv, self.ids, self.vecs)[:k]

    # -- graph ----------------------------------------------------------------
    def graph(self, query: str, k: int = 10, hops: int = 2, seed_k: int = 5, decay: float = 0.5) -> list[tuple[str, float]]:
        seeds = self.lexical(query, k=seed_k)
        if not seeds:
            return []
        scores: dict[str, float] = defaultdict(float)
        for seed_id, seed_score in seeds:
            scores[seed_id] += seed_score
            frontier = {seed_id}
            visited = {seed_id}
            for h in range(1, hops + 1):
                nxt: set[str] = set()
                for nid in frontier:
                    for nb in self.store.neighbors(nid, hops=1):
                        if nb not in visited:
                            scores[nb] += seed_score * (decay ** h)
                            nxt.add(nb)
                            visited.add(nb)
                frontier = nxt
        ranked = sorted(scores.items(), key=lambda x: -x[1])
        return ranked[:k]

    # -- fusion ---------------------------------------------------------------
    def hybrid(self, query: str, k: int = 10, rrf_k: int = 60) -> list[tuple[str, float]]:
        rankings = [
            [i for i, _ in self.vector(query, k=max(k, 20))],
            [i for i, _ in self.graph(query, k=max(k, 20))],
            [i for i, _ in self.lexical(query, k=max(k, 20))],
        ]
        return rrf_fuse(rankings, k=k, rrf_k=rrf_k)

    def retrieve(self, query: str, mode: str = "hybrid", k: int = 10) -> list[tuple[str, float]]:
        return {
            "vector": self.vector, "graph": self.graph,
            "lexical": self.lexical, "hybrid": self.hybrid,
        }[mode](query, k=k)


def rrf_fuse(rankings: list[list[str]], k: int = 10, rrf_k: int = 60) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion: score = Σ 1/(rrf_k + rank)."""
    scores: dict[str, float] = defaultdict(float)
    for ranking in rankings:
        for rank, node_id in enumerate(ranking):
            scores[node_id] += 1.0 / (rrf_k + rank + 1)
    fused = sorted(scores.items(), key=lambda x: -x[1])
    return fused[:k]
