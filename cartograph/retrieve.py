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

    def __init__(self, store: Store, embedder=None, reranker=None):
        self.store = store
        self.reranker = reranker
        # External-import stub nodes carry no content; they must never be answers
        # (counting them would inflate recall). Keep them out of every candidate set.
        self.docs = [d for d in store.all_nodes_text() if d["kind"] != "external"]
        self.valid = {d["id"] for d in self.docs}
        self.text_by_id = {d["id"]: (d["embed_text"] or d["qualified_name"]) for d in self.docs}
        ids, vecs = store.all_embeddings()
        self.ids, self.vecs = [], []
        for i, v in zip(ids, vecs):
            if i in self.valid:
                self.ids.append(i)
                self.vecs.append(v)
        self.index_dim = next((len(v) for v in self.vecs if v), None)
        if embedder is None:
            embedder = get_embedder(dim=self.index_dim) if self.index_dim else get_embedder()
        # An explicit embedder whose width differs from the index would crash deep
        # inside numpy at query time; fail here with the actual mismatch instead.
        # (Embedders whose width is only known after a real call — Ollama — are
        # re-checked per query in vector().)
        emb_dim = getattr(embedder, "dim", None)
        if (self.index_dim and emb_dim and getattr(embedder, "dim_is_exact", True)
                and emb_dim != self.index_dim):
            raise ValueError(
                f"query embedder dim {emb_dim} != index dim {self.index_dim}; "
                "drop the embedder override or re-index with this embedder")
        self.embedder = embedder
        self._build_lexical()
        self._build_adjacency()

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

    # -- graph adjacency (for PPR) -------------------------------------------
    def _build_adjacency(self) -> None:
        """Undirected, row-normalized transition matrix over non-external nodes."""
        self.node_index = {nid: i for i, nid in enumerate(d["id"] for d in self.docs)}
        n = len(self.node_index)
        self.adj: list[list[int]] = [[] for _ in range(n)]
        for src, dst in self.store.all_edges():
            i, j = self.node_index.get(src), self.node_index.get(dst)
            if i is not None and j is not None and i != j:
                self.adj[i].append(j)
                self.adj[j].append(i)  # undirected: callers <-> callees

    # -- vector ---------------------------------------------------------------
    def vector(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        qv = self.embedder.embed(query)
        if self.index_dim and len(qv) != self.index_dim:
            raise ValueError(
                f"query embedder dim {len(qv)} != index dim {self.index_dim} "
                f"(embedder {getattr(self.embedder, 'name', '?')}); "
                "drop the embedder override or re-index with this embedder")
        return _cosine_ranking(qv, self.ids, self.vecs)[:k]

    # -- graph (personalized PageRank) ---------------------------------------
    def graph(self, query: str, k: int = 10, seed_k: int = 8, alpha: float = 0.85, iters: int = 30) -> list[tuple[str, float]]:
        """Seed a restart distribution from lexical matches, then run PPR over the
        whole graph. Structure-aware multi-hop scoring; favours nodes that are both
        near the seeds and well-connected in the call/inheritance graph."""
        seeds = self.lexical(query, k=seed_k)
        n = len(self.node_index)
        if not seeds or n == 0:
            return []
        # Restart vector p: mass on lexical seeds, weighted by their scores.
        p = np.zeros(n, dtype=np.float64)
        for sid, sscore in seeds:
            idx = self.node_index.get(sid)
            if idx is not None:
                p[idx] += sscore
        if p.sum() == 0:
            return []
        p /= p.sum()
        r = p.copy()
        for _ in range(iters):
            nxt = np.zeros(n, dtype=np.float64)
            for i, nbrs in enumerate(self.adj):
                if r[i] and nbrs:
                    share = r[i] / len(nbrs)
                    for j in nbrs:
                        nxt[j] += share
            nxt = alpha * nxt + (1.0 - alpha) * p
            # Dangling mass (nodes with no edges) returns via restart.
            nxt += (1.0 - nxt.sum()) * p
            if np.abs(nxt - r).sum() < 1e-9:
                r = nxt
                break
            r = nxt
        ids = [d["id"] for d in self.docs]
        order = np.argsort(-r)
        return [(ids[i], float(r[i])) for i in order[:k] if r[i] > 0]

    # -- fusion ---------------------------------------------------------------
    # Calibrated on the 2026-06-11 ollama fusion sweep (eval/fusion_sweep.py, 4
    # corpora / 51 questions). Equal weights + rrf_k=60 let graph+lexical outvote
    # vector and lost to vector-alone (0.763/0.901/0.679 vs 0.885/0.937/0.707);
    # this vector-dominant, sharpened config wins or ties vector on recall@10 on
    # every corpus and lifts aggregate r@5/mrr (0.909/0.961/0.735). The win region
    # is vector-dominant, so worst case it behaves like vector — never worse. The
    # mrr lift is partly corpus-driven (see SPEC §8); recalibrate, don't hand-edit.
    HYBRID_WEIGHTS = (3.0, 0.5, 0.5)  # (vector, graph, lexical)
    HYBRID_RRF_K = 10
    HYBRID_DEPTH = 50

    def hybrid(self, query: str, k: int = 10, rrf_k: int | None = None,
               weights: tuple[float, float, float] | None = None,
               depth: int | None = None) -> list[tuple[str, float]]:
        """Weighted RRF over (vector, graph, lexical) rankings of length `depth`.

        Defaults are the calibrated constants above; pass overrides only for
        experiments (eval/fusion_sweep.py is how the defaults themselves move).
        """
        rrf_k = self.HYBRID_RRF_K if rrf_k is None else rrf_k
        weights = self.HYBRID_WEIGHTS if weights is None else weights
        depth = self.HYBRID_DEPTH if depth is None else depth
        d = max(k, depth)
        rankings = [
            [i for i, _ in self.vector(query, k=d)],
            [i for i, _ in self.graph(query, k=d)],
            [i for i, _ in self.lexical(query, k=d)],
        ]
        return rrf_fuse(rankings, k=k, rrf_k=rrf_k, weights=weights)

    # -- rerank (second stage) ------------------------------------------------
    def reranked(self, query: str, k: int = 10, pool: int = 20, rrf_k: int = 60) -> list[tuple[str, float]]:
        """Fuse, then re-order the top `pool` with the reranker.

        The LLM's ordering is *blended* with the retrieval order via RRF rather than
        trusted blindly: this keeps the reranker's top-rank gains (MRR/precision)
        while the retrieval consensus — which includes BM25's exact-match strength —
        protects recall@k from a reranker that under-values exact symbol matches.
        Falls back to the fused order if no reranker is configured.
        """
        fused = self.hybrid(query, k=pool)
        if not self.reranker or not fused:
            return fused[:k]
        fused_order = [cid for cid, _ in fused]
        candidates = [(cid, self.text_by_id.get(cid, "")) for cid in fused_order]
        llm_order = self.reranker.rerank(query, candidates)
        return rrf_fuse([llm_order, fused_order], k=k, rrf_k=rrf_k)

    def retrieve(self, query: str, mode: str = "hybrid", k: int = 10) -> list[tuple[str, float]]:
        return {
            "vector": self.vector, "graph": self.graph,
            "lexical": self.lexical, "hybrid": self.hybrid, "rerank": self.reranked,
        }[mode](query, k=k)


def rrf_fuse(rankings: list[list[str]], k: int = 10, rrf_k: int = 60,
             weights: list[float] | None = None) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion: score = Σ w_i / (rrf_k + rank).

    `weights` (one per ranking, default all 1.0) lets a high-precision signal
    outvote low-precision ones instead of being averaged down by them.
    """
    if weights is None:
        weights = [1.0] * len(rankings)
    if len(weights) != len(rankings):
        raise ValueError(f"got {len(weights)} weights for {len(rankings)} rankings")
    scores: dict[str, float] = defaultdict(float)
    for w, ranking in zip(weights, rankings):
        if w == 0:
            continue
        for rank, node_id in enumerate(ranking):
            scores[node_id] += w / (rrf_k + rank + 1)
    fused = sorted(scores.items(), key=lambda x: -x[1])
    return fused[:k]
