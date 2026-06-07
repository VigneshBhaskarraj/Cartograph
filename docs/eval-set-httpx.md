# M1 Eval Set — `httpx` corpus
### + how the results drive M2 (hybrid retrieval + reranker)

**Corpus:** `encode/httpx` (the `httpx/` package only). Clean, layered, well-known Python — and the same library family graphify benchmarked, so you get an apples-to-apples comparison. No SQL here; schema-bridging questions get added at M4 against a repo with a DB layer (or your `ai-digest`).

**How to use it:** index httpx, then run each question through your retriever. For every question, record whether the **expected anchor node(s)** appear in the top-k results, and at what rank. The `Signal that should win` column is your diagnostic — if a question fails, it tells you *which* retriever is weak.

> Expected nodes are written from httpx's real architecture but **confirm them against the exact version you index** — private helpers (`_send_*`) shift across releases.

---

## Mode legend
| Mode | What it stresses | Retriever that should carry it |
| --- | --- | --- |
| **STRUCT** | direct edges (calls, imports, inheritance) | graph (1-hop) |
| **MULTIHOP** | 2–3 hop paths | graph traversal / PPR / shortest_path |
| **SEMANTIC** | a concept whose word isn't in the node name | vector ANN |
| **EXACT** | a precise symbol name | BM25 / full-text |
| **CROSS** | a link spanning different modules ("surprising connection") | fusion (graph + vector) |
| **WHY** | design intent from comments/docstrings | rationale nodes + vector |

---

## The questions

| # | Question | Stresses | Expected anchor node(s) — *confirm on index* | Signal that should win |
| --- | --- | --- | --- | --- |
| 1 | What does `Client.send` call to actually dispatch a request? | STRUCT | `_send_handling_auth`, `_send_handling_redirects`, `_send_single_request`, `BaseTransport.handle_request` | graph (call edges) |
| 2 | Which classes implement the `BaseTransport` interface? | STRUCT | `HTTPTransport`, `MockTransport`, `WSGITransport` (+ async: `AsyncHTTPTransport`, `ASGITransport`) | graph (inheritance) |
| 3 | What are the subclasses of the base `Auth` class? | STRUCT | `BasicAuth`, `DigestAuth`, `NetRCAuth`, `FunctionAuth` | graph (inheritance) |
| 4 | Which module imports `httpcore`? | STRUCT | `_transports/default.py` | graph (import edge) |
| 5 | What calls `encode_request`? *(precision check — should NOT return unrelated hubs)* | STRUCT | request-building path in `_client.py` / `_content.py` | graph (call edges) |
| 6 | Trace the path from `Client.get` down to the transport's `handle_request`. | MULTIHOP | `get` → `request` → `send` → `_send_handling_auth` → `_send_handling_redirects` → `_send_single_request` → `handle_request` | shortest_path / graph |
| 7 | What's the async equivalent path of `Client.send`? *(must not conflate sync/async)* | MULTIHOP | `AsyncClient.send` → async `_send_*` → `AsyncBaseTransport.handle_async_request` | graph (parallel subgraph) |
| 8 | How does a `DigestAuth` challenge get from a 401 response back into a retried request? | MULTIHOP+SEMANTIC | `DigestAuth.sync_auth_flow` reads `Response` (401 / WWW-Authenticate) → builds digest header → yields modified `Request` | graph + vector |
| 9 | How is the `Timeout` config connected to the transport layer? | MULTIHOP+SEMANTIC | `Timeout` (`_config.py`) → passed via `extensions` in `_send_single_request` → `handle_request` | graph + vector |
| 10 | Where is automatic retry / connection-failure handling? *(honest answer: mostly delegated to httpcore; httpx exposes `retries` on the transport)* | SEMANTIC | `HTTPTransport(retries=...)` in `_transports/default.py` | vector |
| 11 | How are gzip and brotli compressed responses decompressed? | SEMANTIC | `_decoders.py` (`GZipDecoder`, `BrotliDecoder`, `ContentDecoder`), `Response.iter_bytes`/`iter_raw` | vector |
| 12 | Where does cookie persistence across requests happen? | SEMANTIC | `Cookies` (`_models.py`), `Client.cookies` | vector |
| 13 | How is the request body serialized from JSON, form data, or files? | SEMANTIC | `encode_request` (`_content.py`), `_multipart.py` | vector |
| 14 | Where is SSL/TLS verification configured? | SEMANTIC+EXACT | `create_ssl_context` (`_config.py`), `verify` param | vector + BM25 |
| 15 | What connects authentication to redirect handling? | CROSS | `_send_handling_auth` wraps `_send_handling_redirects`; auth re-applied per redirected request | fusion (graph + vector) |
| 16 | What links `Response` to the `Auth` classes? | CROSS | `Auth.sync_auth_flow` consumes a `Response`; `DigestAuth` reads the 401 challenge | fusion |
| 17 | How do `event_hooks` touch both requests and responses? | CROSS | `Client.event_hooks` (`{"request": [...], "response": [...]}`) invoked around `_send_single_request` | fusion |
| 18 | Show me `ClientState`. | EXACT | `ClientState` enum (`_client.py`) | BM25 |
| 19 | Where are `URL` and `QueryParams` defined? | EXACT | `_urls.py` | BM25 |
| 20 | Why must the response stream be read or explicitly closed? | WHY | rationale node from the docstring/comment in `_transports/base.py` (network-resource release) | rationale + vector |
| 21 | Why does the client refuse to reopen a closed instance? | WHY+EXACT | the `ClientState.CLOSED` guard/message in `Client.__enter__` | rationale + BM25 |

(21 questions — drop one or add your own; ~20 is the sweet spot. Aim to keep at least 3 per mode so per-mode recall is meaningful.)

---

## Scoring

Run the set and compute, **overall and per-mode**:

- **Recall@k** — fraction of questions where an expected anchor node is in the top-k retrieved (use k = 5 and k = 10). This is the headline number.
- **Precision@k** — for the questions with a small, well-defined answer set (e.g. #1–#5), how much of the top-k is actually relevant. Catches over-retrieval of god-nodes.
- **MRR** — mean reciprocal rank of the first correct node. Captures *ordering* quality, which is exactly what the reranker improves.

Record results as a small table per run so you can see movement:

```
run, retriever, recall@5, recall@10, precision@5, mrr, semantic_recall, multihop_recall, exact_recall, cross_recall
2026-06-07, vector-only,   0.55, 0.70, 0.40, 0.48, 0.83, 0.20, 0.50, 0.33
2026-06-07, graph-only,    0.50, 0.60, 0.65, 0.52, 0.17, 0.90, 0.50, 0.50
2026-06-07, hybrid+rrf,    ...
```

The per-mode columns are the whole point: they tell you *what to fix*, not just that something's off.

---

## From eval results → M2 (hybrid retrieval + reranker)

**Architecture.** Run all three retrievers independently, fuse, then optionally rerank:

1. **Candidate generation (run in parallel):**
   - *Vector* — cosine over node embeddings via Kuzu's HNSW index. Embed each node as `label + signature + docstring/comments` joined, not just the name.
   - *Graph* — seed from nodes whose label/text matches query terms, then **personalized PageRank** (or k-hop BFS expansion) over the graph; score by PPR weight / proximity to seeds.
   - *Lexical* — BM25 / full-text over node labels + qualified names + docstrings (Kuzu FTS).
2. **Fusion — start with Reciprocal Rank Fusion (RRF).** `score(node) = Σ_i 1 / (k + rank_i(node))`, k≈60. It's parameter-light, needs no per-retriever weight tuning, and is a famously strong baseline. Do **not** hand-tune weights first — RRF gets you 80% of the way.
3. **Rerank (optional second stage).** Run a small local cross-encoder (e.g. a `bge-reranker`-class model via Ollama/ONNX) over `(query, node-context)` pairs for the fused top-K → re-score. Local, no egress. Only add this once RRF plateaus.

**Tuning order — the efficiency rule:** you can't rerank a node that was never retrieved. So **first push each retriever's solo recall up** (right node lands *somewhere* in candidates), **then** tune fusion/rerank to pull it into top-k. Diagnose with the per-mode columns:

| If this mode is failing… | Fix here |
| --- | --- |
| **SEMANTIC** low | vector recall — better/code-aware embedding model, embed docstring+code together, revisit chunk granularity |
| **MULTIHOP / STRUCT** low | graph — edge coverage (are call/inheritance edges actually extracted?), PPR seeding, hop depth |
| **EXACT** low | lexical — index qualified names, fix tokenization (snake_case / CamelCase splitting) |
| **CROSS** low | fusion — RRF `k`, or this is where the reranker earns its place |

**Definition of done for M2:** `hybrid+rrf` beats both `vector-only` and `graph-only` on overall recall@10 **and** on at least three of the four per-mode columns, with MRR up. Adding the reranker should move MRR again without hurting recall.
