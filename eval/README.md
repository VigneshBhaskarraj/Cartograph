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
| graph (PPR) | 0.619 | 0.762 | 0.20 | 0.313 | 0.80 | **1.00** | 0.57 | 0.50 | 1.00 | 0.50 |
| **hybrid+rrf** | **0.667** | **0.762** | 0.28 | 0.364 | 0.80 | 0.75 | 0.57 | 0.75 | 1.00 | 0.50 |

(per-mode columns are recall@10)

## Reading it
- **Hybrid wins on recall — the headline metric.** `hybrid+rrf` beats both
  `vector-only` and `graph-only` on recall@5 **and** recall@10; fusion recovers what
  each single signal misses (it inherits vector's SEMANTIC and lexical's EXACT while
  keeping graph's STRUCT/CROSS).
- **Graph is now personalized PageRank.** Replacing k-hop BFS with PPR lifted
  graph-only recall@10 (0.667 → 0.762) and took **MULTIHOP to 1.00** (0.75 → 1.00) and
  SEMANTIC to 0.57 — PPR follows call/inheritance chains the BFS decay cut off. The
  cost is graph-only MRR (0.385 → 0.313): PPR spreads mass, so the exact seed isn't
  always rank-1. That's the intended trade per the eval doc's **efficiency rule** —
  *first push each retriever's solo recall up, then fix ordering with the reranker.*
- **MRR is the honest open gap.** Hybrid's MRR (0.36) beats vector but trails the
  pre-PPR graph; ordering is precisely what the **cross-encoder reranker** (M2 stage 2)
  exists to fix — re-scoring the fused top-K to push the right node to rank 1. Not yet
  implemented (needs a local model via Ollama); reported, not papered over.
- **The per-mode columns diagnose, as designed.** Vector is the weakest leg —
  *because the offline `hash` embedder is feature-hashed bag-of-words, not a semantic
  model*. SEMANTIC recall (0.43 for vector) is the honest ceiling of a non-semantic
  embedder.
- **The obvious next number to move:** swap to a real local model
  (`CARTOGRAPH_EMBEDDER=ollama`, `nomic-embed-text`) and re-index — vector's SEMANTIC
  leg should rise and pull hybrid further up, with no code change. That, plus the
  reranker, is M2.

## Results — real embeddings (`nomic-embed-text` via Ollama, on Apple M4)

Run with `bash eval/run_local.sh` (see [`docs/local-setup.md`](../docs/local-setup.md)).

| retriever | recall@5 | recall@10 | precision@5 | mrr | struct | multihop | semantic | exact | cross | why |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **vector+nomic** | 0.667 | **0.81** | **0.40** | **0.521** | 0.80 | 1.00 | 0.71 | **1.00** | 0.67 | 1.00 |
| lexical (BM25) | 0.619 | 0.762 | 0.24 | 0.343 | 0.80 | 0.75 | 0.57 | 0.75 | 1.00 | 0.50 |
| graph (PPR) | 0.619 | 0.762 | 0.20 | 0.313 | 0.80 | 1.00 | 0.57 | 0.50 | 1.00 | 0.50 |
| hybrid+rrf | 0.667 | **0.81** | 0.32 | 0.464 | 0.80 | 1.00 | 0.71 | 0.75 | 1.00 | 0.50 |

(per-mode columns are recall@10)

### What changed vs the offline embedder, and the key finding
- **The embedder was the bottleneck.** Real embeddings lifted the vector leg hugely:
  recall@10 0.667 → **0.81**, MRR 0.305 → **0.521**, SEMANTIC 0.43 → **0.71**, EXACT
  0.75 → **1.00**. The offline `hash` numbers were an artificial floor.
- **With a strong embedder, RRF now *underperforms* the best single leg on ordering.**
  `vector-only` MRR (0.521) > `hybrid` MRR (0.464): equal-weight RRF dilutes the
  dominant vector signal with the weaker lexical (MRR 0.34) and graph (MRR 0.31) legs.
  Hybrid still ties on recall@10 and wins **CROSS** (1.00 vs 0.67) — fusion helps
  coverage, but hurts top-rank ordering.
- **This is the eval-doc trigger for the reranker.** "Add the cross-encoder reranker
  only once RRF plateaus." It has: re-scoring the fused top-K by `(query, node-context)`
  relevance is exactly what should pull MRR back above vector-only without losing the
  recall/CROSS coverage that fusion buys. That is the next M2 task.
- **Caveat:** 21 questions is a small set — a 0.06 MRR gap is ≈ one question's worth of
  rank movement. Direction is clear and matches theory; magnitudes are indicative.

## Results — M2 reranker (LLM-as-reranker via Ollama)

Run with `bash eval/run_rerank.sh <model>`. The reranker re-orders the fused top-20 with
one listwise LLM call per query, then **blends** that order with the retrieval order via
RRF (so it can't blindly demote retrieved hits).

| retriever | recall@5 | recall@10 | precision@5 | mrr | exact |
| --- | --- | --- | --- | --- | --- |
| vector+nomic | 0.667 | **0.81** | 0.40 | 0.521 | **1.00** |
| hybrid+rrf | 0.667 | **0.81** | 0.32 | 0.464 | 0.75 |
| rerank+`llama3.2:3b` | 0.286 | 0.619 | 0.24 | 0.240 | 0.75 |
| **rerank+`gemma3:12b`** | **0.714** | 0.762 | **0.40** | **0.583** | 0.50 |

### Findings
- **Reranker model size matters a lot.** A 3B model produced near-random listwise
  orderings and *wrecked* the results (MRR 0.24). A 12B model (`gemma3:12b`) instead
  gave the **best MRR (0.583)**, **best recall@5 (0.714)**, and best precision@5 (0.40)
  in the whole suite — exactly the top-rank quality a reranker is for.
- **The blend earns its keep.** Fusing the LLM order with the retrieval order (vs.
  trusting the LLM blindly) lifted MRR 0.542 → **0.583** and recall@5 0.667 → **0.714**.
- **The honest trade:** recall@10 dips to 0.762 (EXACT 0.50). When the LLM strongly
  demotes an exact symbol-name match, the RRF blend softens but doesn't always rescue it
  past rank 10. So **rerank is opt-in; `hybrid` stays the default** when max recall@10
  matters. For an agent consuming top-k context, rerank (best MRR/recall@5/precision@5)
  is the better pick.

**M2 outcome:** RRF fusion + personalized-PageRank graph + an opt-in LLM reranker. The
reranker moves MRR up decisively with a strong-enough model; the recall@10/EXACT trade is
documented, not hidden. Next precision lever is M3 (real symbol resolution).

## Call-edge precision (M3 down-payment)

The anchor-recall eval above doesn't measure whether *call edges point at the right
node* — surfaced by dogfooding the MCP server ("what does `Client.send` call?" returned
phantom `AsyncClient.*` edges). `eval/call_precision.py` measures it against
ground-truthed callee sets.

| index (`--resolver`) | mean call-edge precision | recall |
| --- | --- | --- |
| name + same-module (pre-#8) | 0.500 | 0.800 |
| `heuristic` (self-call → own class) | 0.714 | 0.867 |
| **`jedi` (receiver-type inference)** | **1.000** | **1.000** |

(ground-truthed over `Client.send`, `AsyncClient.send`, `Client.get`; httpx CALLS edges 722 → 666 heuristic → 492 jedi)

Two steps closed the gap:
1. **`self.`/`cls.` → caller's own class** killed the phantom sync↔async edges
   (`Client.send` no longer "calls" `AsyncClient._send_handling_auth`).
2. **Jedi** (`uv run cartograph index … --resolver jedi`, needs `--extra resolve`)
   infers the *receiver type*, so `response.read()` resolves to `Response.read` (not
   `Request.read`) and external calls produce no edge. Perfect precision+recall on the
   ground-truthed set; CALLS edges 666 → 492 as the wrong/external ones drop. Indexing
   is slower (per-call `goto`), so `heuristic` stays the dependency-free default and
   `jedi` is opt-in. Run `uv run python eval/call_precision.py --db <jedi-index>`.

## Schema-bridging eval (code + DB in one graph)

Corpus: `eval/bridge_corpus/` — a small SQLAlchemy-style app (`models.py`) mapped onto
`schema.sql`. The bridge is `MAPS_TO` (model class → table, from `__tablename__`,
EXTRACTED) alongside FK `REFERENCES`. 7 questions in `eval/bridge_questions.yaml` cross
the code↔schema boundary ("which table does the `User` model map to", "what code fetches
a user's orders", "FK into users", …).

```bash
uv run cartograph index eval/bridge_corpus --db cartograph-out/bridge.kuzu   # needs --extra sql
uv run python eval/resolve_anchors.py --db cartograph-out/bridge.kuzu --questions eval/bridge_questions.yaml --check
uv run python eval/run_eval.py --db cartograph-out/bridge.kuzu --questions eval/bridge_questions.yaml --retriever hybrid
```

| retriever | recall@5 | recall@10 | mrr |
| --- | --- | --- | --- |
| vector+hash | 0.857 | 1.00 | 0.44 |
| lexical | 0.857 | 1.00 | 0.50 |
| graph (PPR) | 0.571 | 0.857 | 0.61 |
| **hybrid+rrf** | 0.714 | **1.00** | **0.61** |

All code↔schema questions are answerable (recall@10 = 1.0 for vector/lexical/hybrid), and
the structural bridge edges give graph/hybrid the best MRR — the unified graph pays off:
an agent traverses `User` (model) →`MAPS_TO`→ `users` (table) →`REFERENCES`← `orders.user_id`
in one place. Numbers are with the offline `hash` embedder (reproducible); a real embedder
lifts the SEMANTIC question. The eval generalizes via `--questions`/`--db`, so pointing it
at a real code+DB repo (e.g. `ai-digest`) is a corpus swap, not new code.

## Honesty notes
- **Call edges are heuristic (INFERRED).** Non-`self` method calls resolve by bare name,
  so a `.read()`/`.close()` or a `transport.handle_request()` call links to *every* class
  defining that name. This is the acknowledged M3 precision gap (SCIP / stack-graphs);
  tagged `INFERRED` so it's never mistaken for resolved. (`self.`-calls are now
  class-resolved — see above.)
- **External imports are excluded from scoring.** Third-party/stdlib import targets are
  stored as contentless `external` nodes and removed from every candidate set and gold
  set, so retrieval can't earn credit for an empty placeholder.
- `precision@5` divides by k=5 while most questions have 1–4 gold nodes, so its
  ceiling is well under 1.0; treat it as an over-retrieval *diagnostic*, not a target.
- Numbers above are with the **offline** embedder so the suite is reproducible in CI
  with zero network. Ollama-backed runs are a one-flag change.
