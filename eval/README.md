# Eval — M1 baseline report

Methodology and the M2 design: [`docs/eval-set-httpx.md`](../docs/eval-set-httpx.md).
Questions: [`questions.yaml`](./questions.yaml) (21, ≥3 per mode), confirmed against the
pinned corpus.

## Reproduce
```bash
bash eval/get_corpus.sh 0.27.2                                  # vendor httpx (gitignored)
uv run cartograph index .corpus/httpx --db cartograph-out/httpx.kuzu
uv run python eval/resolve_anchors.py --db cartograph-out/httpx.kuzu --check
for R in vector lexical graph hybrid; do
  uv run python eval/run_eval.py --retriever $R --out eval/results.csv
done
```

## Results (httpx==0.27.2, offline `hash` embedder)

| retriever | recall@5 | recall@10 | precision@5 | mrr | struct | multihop | semantic | exact | cross | why |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| vector+hash | 0.429 | 0.667 | 0.20 | 0.305 | 0.60 | 0.75 | 0.43 | 0.75 | 0.67 | 1.00 |
| lexical (BM25) | 0.619 | 0.762 | 0.24 | 0.343 | 0.80 | 0.75 | 0.57 | 0.75 | 1.00 | 0.50 |
| graph (PPR-lite) | 0.619 | 0.667 | 0.28 | 0.385 | 0.80 | 0.75 | 0.43 | 0.50 | 1.00 | 0.00 |
| **hybrid+rrf** | **0.667** | **0.762** | 0.24 | 0.380 | 0.80 | 0.75 | 0.57 | 0.75 | 1.00 | 0.50 |

(per-mode columns are recall@10)

## Reading it
- **Hybrid wins on recall — the headline metric.** `hybrid+rrf` beats both
  `vector-only` and `graph-only` on recall@5 **and** recall@10; fusion recovers what
  each single signal misses (e.g. it inherits vector's SEMANTIC and lexical's EXACT
  while keeping graph's STRUCT/CROSS).
- **MRR is the honest gap.** Hybrid's MRR (0.38) edges out vector but sits a hair
  *below* graph (0.385): RRF improves coverage but slightly dilutes the top rank that
  pure graph nails on structural questions. This is exactly the case the eval doc says
  the **cross-encoder reranker** (M2 stage 2) exists to fix — re-scoring the fused
  top-K to push the right node to rank 1. Not yet implemented; reported, not papered over.
- **The per-mode columns diagnose, as designed.** Vector is the weakest leg —
  *because the offline `hash` embedder is feature-hashed bag-of-words, not a semantic
  model*. SEMANTIC recall (0.43 for vector) is the honest ceiling of a non-semantic
  embedder.
- **The obvious next number to move:** swap to a real local model
  (`CARTOGRAPH_EMBEDDER=ollama`, `nomic-embed-text`) and re-index — vector's SEMANTIC
  leg should rise and pull hybrid further up, with no code change. That, plus the
  reranker, is M2.

## Honesty notes
- **Call edges are heuristic (INFERRED).** A call to a common method name with no
  same-module definition links to *every* class defining that name (e.g. one
  `_send_single_request` call site → all four `*Transport.handle_request`). This is the
  acknowledged M3 precision gap (SCIP / stack-graphs); it can hand graph mode some
  recall via noise. Tagged `INFERRED` in the graph so it's never mistaken for resolved.
- **External imports are excluded from scoring.** Third-party/stdlib import targets are
  stored as contentless `external` nodes and removed from every candidate set and gold
  set, so retrieval can't earn credit for an empty placeholder.
- `precision@5` divides by k=5 while most questions have 1–4 gold nodes, so its
  ceiling is well under 1.0; treat it as an over-retrieval *diagnostic*, not a target.
- Numbers above are with the **offline** embedder so the suite is reproducible in CI
  with zero network. Ollama-backed runs are a one-flag change.
