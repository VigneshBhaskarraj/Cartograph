# PLAN.md — Cartograph M0 + M1

Scope of this file: **Milestone 0 (vertical slice)** and **Milestone 1 (eval harness)**
only, broken into tiny, individually-verifiable tasks. Nothing here is built yet —
this is for review per `CLAUDE.md` directive 1 (plan before code). M2+ are out of
scope for this document.

## How to read a task
Each task lists **Files**, **Does** (one line), and **Verify** (the single command
that proves it works). A task is done only when its Verify command passes and the
diff has been shown. One task per change (`CLAUDE.md` directive 2).

## Working conventions
- Package source in `cartograph/`, tests in `tests/`, eval assets in `eval/`.
- Tests must run **fully offline and deterministically**. Anything touching Ollama
  goes behind an embedder interface with a deterministic **fake** backend used by
  default in tests; the real Ollama backend is opt-in (`CARTOGRAPH_EMBEDDER=ollama`).
  This honors "zero data egress by default" — even the local Ollama call is opt-in
  for tests.
- Commit style: `feat:` / `fix:` / `docs:` / `test:` / `chore:`.

## Proposed dependencies (need approval — `CLAUDE.md` directive 4)
Runtime: `kuzu`, `tree-sitter`, `tree-sitter-python`, `typer` (CLI), `numpy`
(vector math + RRF later), `pyyaml` (eval question set).
Dev: `pytest`.
Embeddings: the local **Ollama** HTTP endpoint is called via the Python **stdlib**
(`urllib.request` to `127.0.0.1:11434`) — no extra dependency, and it keeps the
egress surface to localhost only. All MIT/BSD/Apache-compatible.
No dependency is added until you approve this list.

---

## Proposed Kuzu schema (for review before implementation)

Design choice taken as the **working default** (and flagged as an open question
below): **one node table with a `kind` property**, plus **separate REL tables per
edge type**. Rationale: a single node table means one HNSW vector index and one FTS
index cover the whole corpus (simpler M0/M2), while typed edge tables keep
call/inherit/import traversals explicit for the STRUCT/MULTIHOP eval modes. Every
edge carries a `confidence` tag (`EXTRACTED` | `INFERRED`) per the invariants.

```cypher
-- Node: every code object and every rationale comment is one row, distinguished by `kind`.
CREATE NODE TABLE CodeNode (
    id              STRING,        -- stable id, e.g. "httpx/_client.py::httpx._client.Client.send"
    kind            STRING,        -- 'module' | 'class' | 'function' | 'method' | 'rationale'
    name            STRING,        -- simple name, e.g. 'send'
    qualified_name  STRING,        -- 'httpx._client.Client.send'
    file_path       STRING,        -- repo-relative path
    start_line      INT64,
    end_line        INT64,
    signature       STRING,        -- 'def send(self, request, *, stream=False) -> Response'
    docstring       STRING,        -- docstring / rationale text ('' if none)
    code            STRING,        -- raw source slice (context for embedding)
    embed_text      STRING,        -- exact text fed to the embedder: label+signature+docstring
    embedding       FLOAT[768],    -- DIMENSION DEPENDS ON CHOSEN MODEL — open question
    content_sha     STRING,        -- SHA256 of source slice; for M4 incremental cache
    PRIMARY KEY (id)
);

-- Structural edges. `confidence`: 'EXTRACTED' (deterministic) | 'INFERRED'.
-- `resolver`: 'tree-sitter' for M0/M1; 'scip'/'stack-graphs' later (M3).
CREATE REL TABLE CALLS     (FROM CodeNode TO CodeNode, confidence STRING, resolver STRING);  -- caller -> callee
CREATE REL TABLE INHERITS  (FROM CodeNode TO CodeNode, confidence STRING);                   -- subclass -> base
CREATE REL TABLE IMPORTS   (FROM CodeNode TO CodeNode, confidence STRING);                   -- module -> imported symbol/module
CREATE REL TABLE CONTAINS  (FROM CodeNode TO CodeNode);                                       -- module->class/func, class->method
CREATE REL TABLE DOCUMENTS (FROM CodeNode TO CodeNode);                                       -- rationale node -> the code it annotates
```

Indexes (created after load):
```cypher
-- M0 needs the vector index; FTS is M2 but the schema is ready for it.
CALL CREATE_VECTOR_INDEX('CodeNode', 'codenode_vec_idx', 'embedding');
-- (M2) CALL CREATE_FTS_INDEX('CodeNode', 'codenode_fts_idx', ['name','qualified_name','docstring']);
```

---

## Milestone 0 — Vertical slice
Goal: prove the **whole pipe on one path** — parse one file → one node + one edge in
Kuzu → embed it → retrieve it both ways — before any layer is broadened.

### M0-1 — Project scaffold
- **Files:** `pyproject.toml`, `cartograph/__init__.py`, `tests/__init__.py`
- **Does:** `uv init` the project and declare the approved dependencies.
- **Verify:** `uv run python -c "import kuzu, tree_sitter, tree_sitter_python; print('ok')"`

### M0-2 — Kuzu schema bootstrap
- **Files:** `cartograph/store.py`, `tests/test_store.py`, `cartograph/schema.cypher`
- **Does:** `create_schema(db_path)` opens a Kuzu DB and runs the DDL above.
- **Verify:** `uv run pytest tests/test_store.py::test_schema_tables -q`
  (asserts `CALL show_tables()` returns CodeNode, CALLS, INHERITS, IMPORTS, CONTAINS, DOCUMENTS)

### M0-3 — Extract one node + one edge from a single file
- **Files:** `cartograph/extract.py`, `tests/fixtures/sample.py`, `tests/test_extract.py`
- **Does:** tree-sitter-parse one file into in-memory `CodeNode`s + edges; minimal case
  = a `module` node + one `function` node + one `CONTAINS` edge.
- **Verify:** `uv run pytest tests/test_extract.py::test_single_function -q`

### M0-4 — Load extracted graph into Kuzu
- **Files:** `cartograph/store.py`, `tests/test_store.py`
- **Does:** `load(db, nodes, edges)` writes the extraction into the tables.
- **Verify:** `uv run pytest tests/test_store.py::test_load_roundtrip -q`
  (loads the M0-3 fixture, asserts node count and one CONTAINS edge via Cypher)

### M0-5 — Embed a node (fake by default, Ollama opt-in)
- **Files:** `cartograph/embed.py`, `tests/test_embed.py`
- **Does:** `Embedder.embed(text) -> list[float]`; deterministic **fake** backend
  (hash→vector) for tests, real Ollama backend behind `CARTOGRAPH_EMBEDDER=ollama`.
  Writes the vector into `CodeNode.embedding`.
- **Verify:** `uv run pytest tests/test_embed.py::test_fake_embedding_dim -q`
  (asserts vector length == schema dimension and is deterministic for the same text)

### M0-6 — Vector retrieval finds the node
- **Files:** `cartograph/retrieve.py`, `tests/test_retrieve.py`
- **Does:** build the HNSW index and run a top-k vector query over `CodeNode.embedding`.
- **Verify:** `uv run pytest tests/test_retrieve.py::test_vector_finds_node -q`
  (a query close to the seed node's `embed_text` returns it at rank 1)

### M0-7 — 2-hop graph traversal
- **Files:** `cartograph/retrieve.py`, `tests/test_retrieve.py`
- **Does:** `neighbors(node_id, hops=2)` returns nodes reachable within 2 hops.
- **Verify:** `uv run pytest tests/test_retrieve.py::test_two_hop -q`

### M0-8 — CLI: index one file + run a hardcoded query both ways
- **Files:** `cartograph/cli.py`, `pyproject.toml` (console-script entry)
- **Does:** `cartograph index <file>` and `cartograph query <text>` that runs **both**
  vector search and a 2-hop traversal and prints the hits.
- **Verify:** `uv run cartograph index tests/fixtures/sample.py && uv run cartograph query "the function" ` (exit 0, prints the fixture function via both paths)

### M0-9 — M0 acceptance on one real file + independent review
- **Files:** `tests/test_m0_acceptance.py`
- **Does:** run the full pipe on a **single real `httpx` source file** and answer one
  hardcoded query through both vector search and a 2-hop traversal.
- **Verify:** `uv run pytest tests/test_m0_acceptance.py -q`; then a **subagent in a
  fresh context** reviews the M0 diff against this plan and reports gaps
  (`CLAUDE.md` directive 6).

---

## Milestone 1 — Eval harness
Goal: ~20 fixed `httpx` questions with expected answer nodes; report recall@k,
precision@k, MRR, broken down per retrieval mode. Everything after M1 is measured
against this.

### M1-1 — Pin the httpx corpus
- **Files:** `eval/corpus.md` (how/what is pinned), `eval/get_corpus.sh`
- **Does:** fetch the `httpx/` package at a **pinned version/tag** into a gitignored
  dir (reproducibility — the eval doc warns private `_send_*` helpers shift).
- **Verify:** `bash eval/get_corpus.sh && test -f .corpus/httpx/_client.py`

### M1-2 — Broaden the extractor
- **Files:** `cartograph/extract.py`, `tests/test_extract.py`
- **Does:** walk a directory; emit classes, functions, methods, imports, call sites
  (tree-sitter heuristic edges, tagged `INFERRED`), inheritance, and rationale nodes
  (`# NOTE:` / `# WHY:` / docstrings) with CONTAINS/DOCUMENTS edges.
- **Verify:** `uv run pytest tests/test_extract.py -q`
  (fixtures assert one of each: CALLS, INHERITS, IMPORTS, DOCUMENTS)

### M1-3 — Index the full httpx package
- **Files:** `cartograph/cli.py` (`index <dir>`)
- **Does:** index the pinned corpus end-to-end into a Kuzu DB.
- **Verify:** `uv run cartograph index .corpus/httpx && uv run cartograph stats`
  (prints non-zero counts for nodes and each edge type)

### M1-4 — Eval question set
- **Files:** `eval/questions.yaml`
- **Does:** encode the 21 questions from `docs/eval-set-httpx.md` — each with `id`,
  `question`, `mode`, and `expected_anchors` (symbol strings).
- **Verify:** `uv run python -c "import yaml,sys; d=yaml.safe_load(open('eval/questions.yaml')); assert len(d)>=20; print(len(d))"`

### M1-5 — Resolve anchors against the indexed graph (confirm-on-index)
- **Files:** `eval/resolve_anchors.py`, `eval/anchors.resolved.json`
- **Does:** map each expected-anchor string to a real `CodeNode.id` in the indexed
  graph; **fail loudly** on any anchor that doesn't resolve (the eval doc's
  "confirm on index" step).
- **Verify:** `uv run python eval/resolve_anchors.py --check` (exit 0 only if every anchor resolves)

### M1-6 — Retriever API: vector-only and graph-only
- **Files:** `cartograph/retrieve.py`
- **Does:** `retrieve(query, mode, k) -> ranked list[node_id]` for `vector` and
  `graph` (graph = seed by lexical term match, then k-hop expansion).
- **Verify:** `uv run pytest tests/test_retrieve.py::test_retriever_returns_ranked_ids -q`

### M1-7 — Scoring
- **Files:** `eval/score.py`, `tests/test_score.py`
- **Does:** compute recall@5, recall@10, precision@5, MRR — overall and per-mode —
  from ranked results + resolved anchors.
- **Verify:** `uv run pytest tests/test_score.py -q` (hand-built ranking → known metric values)

### M1-8 — Eval runner
- **Files:** `eval/run_eval.py`
- **Does:** run a chosen retriever over all questions and emit one CSV row in the
  `docs/eval-set-httpx.md` format (run, retriever, recall@5/10, precision@5, mrr, per-mode).
- **Verify:** `uv run python eval/run_eval.py --retriever vector --out eval/results.csv && test -s eval/results.csv`

### M1-9 — Baseline report + independent review
- **Files:** `eval/results.csv`, `eval/README.md`
- **Does:** record `vector-only` and `graph-only` baseline rows (the numbers M2 must
  beat).
- **Verify:** `uv run python eval/run_eval.py --retriever vector && uv run python eval/run_eval.py --retriever graph` produce two rows; then a **subagent fresh-context review** of the M1 diff vs this plan.

---

## Open questions (please confirm before I start M0)
1. **Embedding model + dimension.** This blocks the schema's `FLOAT[N]` and M0-5.
   Candidates: `nomic-embed-text` (768, general, fast), `mxbai-embed-large` (1024),
   a code-specialized model if you prefer code-awareness over speed. Which model /
   dimension?
2. **Schema shape.** OK to proceed with **one node table + `kind`** and **per-type
   rel tables** as written above, or do you want per-kind node tables?
3. **Kuzu DB: regenerate vs commit.** I propose **regenerate per checkout** (already
   reflected in `.gitignore`), revisiting a committed "team map" later. Agree?
4. **httpx pin.** Which exact httpx version/tag should the eval index, so anchors stay
   stable? (I'll pin whatever you name in M1-1.)
5. **Dependency list.** Approve the proposed deps above (`kuzu`, `tree-sitter`,
   `tree-sitter-python`, `typer`, `numpy`, `pyyaml`, `pytest`; Ollama via stdlib)?

**Assumptions I'm making unless you object:** SCIP / stack-graphs is **deferred to
M3** (M0/M1 ship tree-sitter heuristic `INFERRED` call edges, recorded as the
precision gap); the **reranker is deferred to M2** and not part of this plan.

---

## Gate-2.5 — Fusion fix (added 2026-06-11)
The 2026-06-07 ollama scorecard showed vector-alone beating equal-weight hybrid on
every aggregate metric — fusion was diluting the strongest signal. Plan approved in
session; decision rule: hybrid must match/beat vector on mean recall@5 AND mrr, or
be demoted honestly.

### G2.5-1 — Weighted RRF
- **Files:** `cartograph/retrieve.py`, `tests/test_retrieve.py`
- **Does:** `rrf_fuse(..., weights=)` + `hybrid(..., weights=, rrf_k=, depth=)`;
  defaults unchanged (equal weights) until sweep evidence picks new ones.
- **Verify:** `uv run pytest tests/test_retrieve.py -q`

### G2.5-2 — Fusion sweep harness
- **Files:** `eval/fusion_sweep.py`
- **Does:** caches each signal's depth-50 ranking once per question, re-fuses across
  a 128-config grid (weights × rrf_k × depth), prints leaderboard + verdict vs the
  vector baseline, per-corpus consistency, optional CSV.
- **Verify:** `uv run python eval/fusion_sweep.py --embedder hash` (runs all present
  corpora end-to-end; mechanism check only — real numbers need `--embedder ollama`)

### G2.5-3 — Bake the winner ✅ (2026-06-11 ollama sweep)
- **Files:** `cartograph/retrieve.py` (hybrid defaults), `SPEC.md` §8, `eval/README.md`
- **Done:** baked `weights=(3.0, 0.5, 0.5)`, `rrf_k=10`, `depth=50`. Aggregate
  hybrid 0.909/0.961/0.735 vs vector 0.885/0.937/0.707 (r@5/r@10/mrr); wins-or-ties
  recall@10 on all 4 corpora. The sweep's leave-one-corpus-out check flagged the
  **mrr** edge as ai-digest-dependent, so the durable win is recall, not ranking —
  recorded honestly in SPEC §8. Pinned by `test_hybrid_defaults_are_calibrated_constants`.
- **Follow-up (open):** add a held-out 5th corpus and re-sweep to confirm the mrr lift
  generalizes before claiming a ranking win publicly.
- **Verify:** `uv run python eval/scorecard.py --embedder ollama --reindex` (hybrid ≥
  vector on mean recall@5 and mrr; mrr margin is corpus-sensitive — see above).

## Gate-3 — Comparative evidence (added 2026-06-11)
Closing the three publish-blockers: no external baseline, no agent benchmark,
51-question eval.

### G3-1 — External baselines ✅
- **Files:** `eval/baselines.py`, `eval/scorecard.py` (`--baselines`)
- **Done:** `grep` (term search over raw source) and `naive-rag` (40-line chunk
  embeddings, structure-blind) scored on the identical questions/gold/metrics.
  Graph node spans are used only to map text hits to gold ids — never as signal.
- **Verify:** `uv run python eval/scorecard.py --baselines` prints baseline rows.

### G3-2 — Eval scale: 89 questions / 5 corpora ✅
- **Files:** `eval/click_questions.yaml` (18, NEW held-out corpus, get_click.sh),
  `eval/questions.yaml` (+8), `eval/flask_questions.yaml` (+6),
  `eval/bridge_questions.yaml` (+3), `eval/aidigest_questions.yaml` (+3)
- **Done:** 51 → 89 questions. All authored by reading source (never by testing
  retrieval); all anchors resolve. **click is held out**: written after fusion
  calibration, excluded from any tuning — it is the generalization check.
- **Verify:** `resolve_anchors.py --check` exits 0 on all five DBs.

### G3-3 — Agent benchmark (pilot ✅, offline harness shipped)
- **Files:** `eval/agent_bench/{tasks.yaml,run_bench.py,RESULTS.md}`,
  `cartograph/cli.py` (new `node/resolve/calls/callers/path` commands — the MCP
  surface over the shell, also what the benchmark's cartograph condition uses)
- **Done:** 12 source-verified navigation tasks; pilot (Claude subagents, matched
  tool surfaces): **equal success (12/12), 42% fewer tool calls (3.0 vs 5.2)** for
  the Cartograph condition. Caveats recorded in RESULTS.md (n=12, strong driver
  model, hash-indexed graphs = conservative). `run_bench.py` reproduces offline
  with any local Ollama chat model.
- **Verify:** `uv run python eval/agent_bench/run_bench.py --tools grep|cartograph`
  (needs local Ollama; pilot method + numbers in RESULTS.md).

### G3 open follow-ups
- Ollama scorecard + sweep-verdict on the expanded 89-question set, esp. the
  held-out click corpus (runs on a machine with Ollama).
- Scale agent bench beyond n=12 and run with a weaker local model, where accuracy
  (not just efficiency) gaps are expected to appear.

---

## Implementation notes — deviations from this plan (M0/M1 as shipped)
Recorded so the drift from the approved plan is explicit (not silent):
1. **Node `id` format** is `<file>::<qualified_name>#<line>`, not `<file>::<qualified_name>`.
   The `#<line>` suffix keeps property getter/setter pairs (same qualified name) unique.
2. **Extra `module STRING` column** on `CodeNode` (used for same-module call-edge
   resolution). Not in the original Cypher; otherwise the schema matches.
3. **Vector search is brute-force NumPy cosine, not Kuzu HNSW.** Deliberate: HNSW needs
   a downloadable Kuzu extension (a network dependency that breaks "offline by default").
   Brute force is exact, offline, and identical in *recall*; HNSW is a later speed-only
   optimization. SPEC still names HNSW as the target architecture.
4. **`external` node kind** for third-party/stdlib import targets — excluded from all
   retrieval candidate sets and eval gold sets so contentless stubs can't earn recall.
5. **Metrics live in `eval/evallib.py`** (shared by runner and tests) rather than a
   separate `eval/score.py`; functionally equivalent.
6. **WHY mode** is carried by docstrings embedded on their owning code node plus
   marker-comment (`# WHY:`/`# NOTE:`) rationale nodes; standalone docstring→rationale
   nodes are a possible later refinement.
