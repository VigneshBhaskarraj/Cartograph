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
| vector+hash | 0.429 | 0.619 | 0.16 | 0.292 | 0.60 | 0.75 | 0.43 | 0.75 | 0.67 | 0.50 |
| lexical (BM25) | 0.571 | 0.762 | 0.20 | 0.318 | 0.80 | 0.75 | 0.57 | 0.75 | 1.00 | 0.50 |
| graph (PPR-lite) | 0.619 | 0.714 | 0.28 | 0.379 | 0.80 | 0.75 | 0.43 | 0.50 | 1.00 | 0.50 |
| **hybrid+rrf** | **0.714** | **0.762** | **0.28** | **0.400** | 0.80 | 0.75 | 0.57 | 0.75 | 1.00 | 0.50 |

(per-mode columns are recall@10)

## Reading it
- **Hybrid wins overall.** `hybrid+rrf` beats both `vector-only` and `graph-only` on
  recall@5, recall@10, and MRR — fusion recovers what each single signal misses.
- **The per-mode columns diagnose, as designed.** Graph carries STRUCT/CROSS; lexical
  carries EXACT; vector is the weakest leg here — *because the offline `hash` embedder
  is feature-hashed bag-of-words, not a semantic model*. SEMANTIC recall is the
  honest ceiling of a non-semantic embedder.
- **The obvious next number to move:** swap the embedder to a real local model
  (`CARTOGRAPH_EMBEDDER=ollama`, `nomic-embed-text`) and re-run — vector's SEMANTIC
  leg should rise and pull hybrid further up, with no code change. That, plus the
  cross-encoder reranker, is M2.

## Honesty notes
- `precision@5` divides by k=5 while most questions have 1–4 gold nodes, so its
  ceiling is well under 1.0; treat it as an over-retrieval *diagnostic*, not a target.
- Numbers above are with the **offline** embedder so the suite is reproducible in CI
  with zero network. Ollama-backed runs are a one-flag change.
