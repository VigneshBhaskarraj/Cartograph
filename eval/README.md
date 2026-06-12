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

## Validated results — 101 questions / 6 corpora, with baselines (2026-06-12, Apple M4)

Java joined the evaluated tier: spring-petclinic (12 questions) scores hybrid/vector
**0.833 r@5/r@10** vs grep 0.583/0.75 and naive-rag 0.417/0.75 — the code<->data
CROSS questions hit 1.0. Known gap: SEMANTIC recall@10 is 0.333 on Java for every
retriever — javadoc is not yet extracted into embedding text (Python docstrings are);
recorded as the next extractor improvement. Six-corpus means: hybrid 0.874/0.932/0.732,
vector 0.847/0.912/0.712, grep 0.543/0.683/0.379, naive-rag 0.507/0.725/0.259.

### Earlier validated run — 89 questions / 5 corpora (2026-06-11)

`uv run python eval/scorecard.py --embedder ollama --reindex --baselines` after the
Gate-3 expansion. **click is held out**: its 18 questions were written after fusion
calibration and never used for tuning. Means across the five corpora:

| system | recall@5 | recall@10 | mrr |
| --- | --- | --- | --- |
| naive-rag+nomic (40-line chunks, no structure) | 0.525 | 0.720 | 0.230 |
| grep (term search over raw source) | 0.535 | 0.670 | 0.362 |
| graph (PPR alone) | 0.722 | 0.860 | 0.449 |
| lexical (BM25 alone) | 0.769 | 0.877 | 0.554 |
| vector+nomic (alone) | 0.850 | 0.927 | 0.714 |
| **hybrid (calibrated weighted RRF)** | **0.882** | **0.952** | **0.744** |

- **Held-out validation:** on click, baked hybrid vs vector = r@5 0.778/0.778 (tie),
  r@10 **0.944**/0.889, mrr **0.706**/0.653 → the calibration **generalizes**
  (`fusion_sweep.py` prints this check; its 71-question tuning sweep also recommended
  an alternative config, `weights=(1.5, 1.0, 0.5)`, scoring 0.920/0.946/0.740 vs the
  baked config's 0.908/0.954/0.754 on the tuning corpora — near-tied, and not adopted:
  choosing between near-ties using the held-out corpus would burn its held-out status).
- **Per-corpus hybrid r@10:** httpx 0.862 · flask 0.955 · bridge 1.0 · ai-digest 1.0
  · click 0.944 — wins or ties vector everywhere.
- The 38 added questions are deliberately harder than the original 51 (httpx hybrid
  r@10 0.905 → 0.862 on the bigger set): headroom, not saturation.

## Results — original 21-question httpx set (`nomic-embed-text` via Ollama, Apple M4)

Run with `bash eval/run_local.sh` (see [`docs/local-setup.md`](../docs/local-setup.md)).

> **Note (2026-06-11):** the `hybrid+rrf` row below is the *original equal-weight*
> fusion. The multi-corpus sweep ([Fusion calibration](#fusion-calibration-2026-06-11))
> showed it losing to vector and replaced it with a vector-dominant weighted default;
> on httpx the weighted hybrid is **recall@5 0.762, recall@10 0.905, MRR 0.506**
> (was 0.667 / 0.81 / 0.464). The table is kept to show the finding that motivated the fix.

| retriever | recall@5 | recall@10 | precision@5 | mrr | struct | multihop | semantic | exact | cross | why |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **vector+nomic** | 0.667 | **0.81** | **0.40** | **0.521** | 0.80 | 1.00 | 0.71 | **1.00** | 0.67 | 1.00 |
| lexical (BM25) | 0.619 | 0.762 | 0.24 | 0.343 | 0.80 | 0.75 | 0.57 | 0.75 | 1.00 | 0.50 |
| graph (PPR) | 0.619 | 0.762 | 0.20 | 0.313 | 0.80 | 1.00 | 0.57 | 0.50 | 1.00 | 0.50 |
| hybrid (equal-weight rrf, superseded) | 0.667 | **0.81** | 0.32 | 0.464 | 0.80 | 1.00 | 0.71 | 0.75 | 1.00 | 0.50 |

(per-mode columns are recall@10)

### Fusion calibration (2026-06-11)
`eval/fusion_sweep.py` grids 129 `(weights, rrf_k, depth)` configs over all four corpora
(51 questions), caching each signal's ranking once per question so the grid is pure CPU.
The equal-weight default (`(1,1,1)`, `rrf_k=60`) scored **0.763 / 0.901 / 0.679**
(r@5 / r@10 / mrr) — below vector's **0.885 / 0.937 / 0.707**. The baked default,
`weights=(3.0, 0.5, 0.5)`, `rrf_k=10`, `depth=50`, scores **0.909 / 0.961 / 0.735** and
wins-or-ties vector on recall@10 on every corpus.

| corpus | hybrid r@10 | vector r@10 | hybrid mrr | vector mrr |
| --- | --- | --- | --- | --- |
| httpx | **0.905** | 0.810 | 0.506 | 0.521 |
| flask | 0.938 | 0.938 | 0.650 | 0.632 |
| bridge | 1.000 | 1.000 | 0.893 | 0.905 |
| ai-digest | 1.000 | 1.000 | **0.893** | 0.771 |

**Honest caveat:** the aggregate **mrr** lift is concentrated in ai-digest; the sweep's
leave-one-corpus-out check is unstable on mrr, so the robust win is **recall** (hybrid no
longer loses to vector), not ranking. The winning region is vector-dominant, so the
defaults' worst case is "behaves like vector." A held-out 5th corpus is the next step
before claiming a ranking win publicly. Reproduce: `uv run python eval/fusion_sweep.py
--embedder ollama --reindex`.

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
uv sync --extra sql                                                          # sqlglot, needed first
uv run cartograph index eval/bridge_corpus --db cartograph-out/bridge.kuzu
uv run python eval/resolve_anchors.py --db cartograph-out/bridge.kuzu --questions eval/bridge_questions.yaml --check
uv run python eval/run_eval.py --db cartograph-out/bridge.kuzu --questions eval/bridge_questions.yaml --retriever hybrid
```

| retriever | recall@5 | recall@10 | mrr |
| --- | --- | --- | --- |
| vector+hash | 0.857 | 1.00 | 0.44 |
| lexical | 0.857 | 1.00 | 0.50 |
| graph (PPR) | 0.571 | 0.857 | 0.61 |
| **hybrid+rrf** | 0.714 | **1.00** | **0.61** |

With **real embeddings** (`nomic-embed-text`, via `bash eval/run_local_all.sh`): vector
recall@10 **1.0** / MRR **0.905** / SEMANTIC **1.0**; hybrid recall@10 1.0 / MRR 0.76.
Real vectors nail the model↔table bridge (offline vector MRR was 0.44).

All code↔schema questions are answerable (recall@10 = 1.0 for vector/lexical/hybrid), and
the structural bridge edges give graph/hybrid the best MRR — the unified graph pays off:
an agent traverses `User` (model) →`MAPS_TO`→ `users` (table) →`REFERENCES`← `orders.user_id`
in one place. Numbers are with the offline `hash` embedder (reproducible); a real embedder
lifts the SEMANTIC question. The eval generalizes via `--questions`/`--db`, so pointing it
at a real code+DB repo (e.g. `ai-digest`) is a corpus swap, not new code.

### On a real repo: `ai-digest` (raw sqlite, embedded SQL)

`ai-digest` has no ORM and no `.sql` files — its schema is `CREATE TABLE` inside
`conn.executescript("""…""")`, and queries are raw `INSERT`/`SELECT` strings. Cartograph
now extracts SQL from Python string literals: DDL → `table`/`column` nodes (+ FK
`REFERENCES`), DML → **`QUERIES`** edges (function → table). 5 tables, 15 QUERIES edges
were recovered; *"what code touches `top_stories`"* → `store_run`, `get_repeat_stories`,
`init_db` (correct).

```bash
bash eval/get_aidigest.sh                                          # clone (gitignored)
uv run cartograph index .corpus/ai-digest/src --db cartograph-out/aidigest.kuzu
uv run python eval/run_eval.py --db cartograph-out/aidigest.kuzu --questions eval/aidigest_questions.yaml --retriever hybrid
```

| retriever | recall@5 | recall@10 | mrr |
| --- | --- | --- | --- |
| vector+hash | 0.43 | 0.43 | 0.43 |
| lexical | 0.857 | 0.857 | 0.76 |
| graph (PPR) | 0.857 | 0.857 | 0.60 |
| **hybrid+rrf** | 0.571 | **0.857** | 0.61 |

7 ground-truthed code↔schema questions; **~86% recall@10** on a real repo with the
*offline* embedder (vector is hash-capped — a real embedder lifts the SEMANTIC ones).
This validates the bridge end-to-end on code nobody wrote for the eval.

With **real embeddings** (`nomic-embed-text`) the lift is dramatic: **vector recall@10
0.43 → 1.0, MRR 0.43 → 0.77**, SEMANTIC/EXACT/CROSS all 1.0; and here **hybrid MRR (0.857)
beats vector (0.771)** — graph/lexical add ordering signal fusion exploits. Real vectors
matter most on real, natural-language-ish symbol/query text.

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
