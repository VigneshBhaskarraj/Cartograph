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

### G4-1 — `impact`: code↔data blast radius (the moat query) ✅ 2026-06-12
- **Files:** `cartograph/service.py` (typed adjacency + closures + `impact`),
  `cartograph/cli.py` (`impact`), `cartograph/mcp_server.py` (tool), docs, README
- **Does:** for a table/column — direct touchers (QUERIES/MAPS_TO; a mapped class
  implicates its methods) plus all transitive callers; for code — every
  table/column reachable through scope + callees. Deterministic ordering,
  truncation flagged. Honesty caveats recorded: CALLS expansion over-approximates,
  but the bridge can miss ORM attribute access; FK/JOIN ripple not followed.
- **Verify:** `uv run pytest tests/test_impact.py -q`

### G3 open follow-ups
- ~~Ollama scorecard + sweep-verdict on the expanded 89-question set~~ ✅ 2026-06-11:
  hybrid 0.882/0.952/0.744 vs vector 0.850/0.927/0.714 vs grep 0.535/0.670/0.362
  (means, 5 corpora); **held-out click: generalizes** (hybrid ≥ vector on all three).
  Sweep's near-tied alternative config NOT adopted — selecting on the held-out
  corpus would burn its held-out status; revisit only with a fresh held-out corpus.
- Scale agent bench beyond n=12 and run with a weaker local model, where accuracy
  (not just efficiency) gaps are expected to appear.

## Gate-5 — Hardening (added 2026-06-12) ✅ DONE 2026-06-12
Trigger: the local MCP server died with "Connection closed" — root cause was a
pre-G4-1 graph missing the empty JOINS/QUERIES rel tables (fixed operationally by
running the two `schema_ddl()` statements against `httpx-ollama.kuzu`). A three-agent
review of the full codebase followed; this gate is the prioritized result. Three
tracks: **A** interface robustness, **B** index/store integrity, **C** extractor
correctness. Track C tasks marked **[eval-first]** change graph edges and must ship
with before/after scorecard numbers (CLAUDE.md directive 3), not just passing tests.

> **Closed 2026-06-12.** All 14 tasks landed (A1-A4, B1-B4, C1-C6) across five
> commits; 155 tests green (was 127). Independent fresh-context review (directive
> 6): **GO** — all tasks verified implemented; its three follow-up findings
> (heal-policy documentation + versionless-heal test, dirty-rebuild `use_cache`
> plumbing, multi-hop filter validation) fixed and tested in the closing commit.
> C5 verified byte-identical: node/edge dumps over all 6 corpora, zero diff.
> Eval: zero regression anywhere; petclinic graph mrr 0.386 → 0.400 (C3).

### Track A — MCP/CLI robustness

### G5-A1 — Startup failures must reach the MCP client
- **Files:** `cartograph/mcp_server.py`, `cartograph/service.py`, `tests/test_mcp.py`
- **Does:** service constructed before the handshake (`mcp_server.py:93-97`) means any
  startup error = "Connection closed". Lazy-init the service (or degraded server whose
  tools return the error text). When the only incompatibility is missing **empty** rel
  tables and `schema_version` matches, auto-create them from `schema_ddl()` and proceed.
- **Verify:** `uv run pytest tests/test_mcp.py -q` (new: serve against a JOINS/QUERIES-less
  DB → tools respond, tables created)

### G5-A2 — Bound and validate every tool input/output
- **Files:** `cartograph/service.py`, `cartograph/store.py`, `tests/test_service.py`
- **Does:** clamp `hops` (raw Kuzu binder error leaks at >30; whole-graph payloads at
  2-30 — `service.py:142`); cap `resolve()` results (currently unbounded —
  `store.py:283`) with a `truncated` flag; reject invalid `direction`/`relation`
  with the actionable-ValueError style `query` already uses for `mode` (silent `[]`
  today — `store.py:313-333`).
- **Verify:** `uv run pytest tests/test_service.py -q`

### G5-A3 — Unknown-ref structured errors + protocol round-trip tests
- **Files:** `cartograph/mcp_server.py`, `cartograph/service.py`, `tests/test_mcp.py`
- **Does:** MCP `get_node`/`neighbors`/`calls` return `null`/`[]` for unknown refs —
  indistinguishable from genuinely-empty (the CLI guards this exact case at
  `cli.py:172-178`; the MCP surface doesn't). Return a structured error with top-5
  `resolve` candidates. Add `call_tool` round-trip + error-path tests (test_mcp.py
  currently asserts tool *registration* only).
- **Verify:** `uv run pytest tests/test_mcp.py -q`

### G5-A4 — Route `cartograph query` through the service
- **Files:** `cartograph/cli.py`, `tests/test_cli.py`
- **Does:** `cli.query` bypasses `CartographService` — duplicate mode validation, no
  `k` clamp, `rerank` unreachable from the CLI (`cli.py:82-114`). Route through the
  service; delete `QUERY_MODES`. Also: friendly error for `demo` (`cli.py:285`),
  `ModuleNotFoundError` check `e.name == "mcp"` (`cli.py:314`), fix stale help text
  ("Python file or directory"; "No HTML viz").
- **Verify:** `uv run pytest tests/test_cli.py -q`

### Track B — Index/store integrity

### G5-B1 — Atomic incremental updates
- **Files:** `cartograph/pipeline.py`, `cartograph/store.py`, `tests/test_incremental.py`
- **Does:** `update_index` mutates via auto-committing statements
  (`delete_all_edges` → `delete_nodes` → `load_nodes` → `load_edges`, `pipeline.py:396-416`);
  a crash mid-sequence leaves a nodes-but-no-edges graph that passes every check.
  Wrap in a Kuzu transaction, or write a `dirty=1` Meta flag before mutating and clear
  after, with `open_graph` rejecting dirty graphs. (Choice = open question Q1.)
- **Verify:** `uv run pytest tests/test_incremental.py -q` (new: simulated crash
  mid-update → next `open_graph` refuses or graph is intact)

### G5-B2 — Harden the schema gate
- **Files:** `cartograph/service.py`, `cartograph/store.py`, `tests/test_service.py`
- **Does:** gate checks table names only (`service.py:30-38`): missing
  `schema_version` is treated as *compatible*, `Meta` isn't in `REQUIRED_TABLES`,
  columns never validated. Add `Meta` to required tables; missing version =
  incompatible; stamp version *first* during indexing; validate `CodeNode` columns
  via `CALL table_info`. Add the (currently untested) rejection-path test.
- **Verify:** `uv run pytest tests/test_service.py -q`

### G5-B3 — Close the parse/digest TOCTOU + stale SQL positions
- **Files:** `cartograph/pipeline.py`, `cartograph/sql_extract.py`, `tests/test_incremental.py`
- **Does:** (a) `_file_digests` re-reads files *after* `build_graph`
  (`pipeline.py:314,412`) — a mid-index edit records the new sha against the old
  parse, permanently serving stale content; hash bytes once, feed both. (b) SQL node
  ids/shas omit position (`sql_extract.py:31-62`), so a moved `CREATE TABLE` keeps
  its old `start_line` on delta update; include position in the sha or `SET` it on
  kept rows.
- **Verify:** `uv run pytest tests/test_incremental.py -q`

### G5-B4 — Reranker honesty + small store fixes
- **Files:** `cartograph/rerank.py`, `cartograph/retrieve.py`, `cartograph/cache.py`
- **Does:** `except Exception` silently degrades a configured LLM reranker to lexical
  on every query (`rerank.py:107-111`) — narrow + `warnings.warn` once;
  `mode=rerank` silently caps at `pool=20` regardless of `k` (`retrieve.py:196`) —
  `pool = max(pool, k)`; cache save non-atomic (`cache.py:59`) — temp file +
  `os.replace`, warn on corruption reset.
- **Verify:** `uv run pytest tests/test_retrieve.py tests/test_cache.py -q`

### Track C — Extractor correctness

> **C1–C4 eval verdict (2026-06-12, hash, 6 corpora):** zero quality regression on
> every corpus/retriever; petclinic graph mrr 0.386 → 0.400 (C3's same-package
> tier removed cross-package false CALLS). C1/C4 deltas are zero by construction —
> **no TS corpus exists in the eval set** (gap recorded below); covered by unit
> tests. C2 moves tags, not edges-used-in-scoring, so retrieval is unchanged.

### G5-C1 — TS heritage: stop harvesting generic type args **[eval-first]**
- **Files:** `cartograph/ts_extract.py`, `tests/test_ts.py`
- **Does:** `class A extends Component<Props, State>` emits INHERITS to `Component`,
  `Props`, *and* `State` (`ts_extract.py:203-207`) — walk only
  extends/implements-clause expressions, skip `type_arguments` subtrees.
- **Verify:** `uv run pytest tests/test_ts.py -q` + scorecard delta on TS-bearing corpora

### G5-C2 — INHERITS confidence: unique-name matches are not EXTRACTED **[eval-first]**
- **Files:** `cartograph/extract.py`, `cartograph/ts_extract.py`,
  `cartograph/go_extract.py`, `cartograph/java_extract.py`, tests
- **Does:** all four extractors tag unique-name base-class matches `EXTRACTED`
  (`extract.py:449-461` et al.) — heuristic, violates the edge-confidence invariant
  (`class User(models.Model)` + any local `Model` → wrong edge, top confidence).
  Tag `EXTRACTED` only when the base resolves through the module/import graph;
  otherwise `INFERRED`. (Policy detail = open question Q2.)
- **Verify:** language test suites + `eval/scorecard.py` before/after (hash for
  mechanism, ollama for the recorded number)

### G5-C3 — Go/Java: `module` means package, not file **[eval-first]**
- **Files:** `cartograph/go_extract.py`, `cartograph/java_extract.py`, tests
- **Does:** `module = "<package>.<file-stem>"` makes the same-module resolution tier
  mean "same file" (`go_extract.py:219`, `java_extract.py:238`) — cross-file
  same-package calls fall through to corpus-wide name matching (verified false
  positives). Add a same-package tier before the all-candidates fallback.
- **Verify:** new cross-package collision tests + scorecard delta (petclinic, bridge)

### G5-C4 — TS recall holes: arrows everywhere **[eval-first]**
- **Files:** `cartograph/ts_extract.py`, `tests/test_ts.py`
- **Does:** calls inside arrow callbacks dropped (`ts_extract.py:237` skips
  `arrow_function` — `items.forEach(i => doWork())` → no CALLS edge); class-field
  arrow methods (`handleClick = () => {}`) not extracted (`ts_extract.py:209-219`);
  nested decls in CommonJS-assigned functions leak to module scope
  (`ts_extract.py:111-113`). Fix all three with the lexical-scoping approach
  `extract.py:_walk` already uses.
- **Verify:** `uv run pytest tests/test_ts.py -q` + scorecard delta

### G5-C5 — Unify the 4× duplicated resolution pass
- **Files:** new `cartograph/resolve.py` (name TBD), all four extractors, tests
- **Does:** ~65 near-identical lines (name index, receiver→module→all-candidates
  tiers, fan-out cap, INHERITS rule, ext-stub minting) duplicated across
  `extract.py:383-490`, `ts_extract.py:260-328`, `go_extract.py:187-251`,
  `java_extract.py:213-275` — the breeding ground for G5-C2/C3-class drift. Extract
  one shared pass. **Deliberately last in track C**: the C1-C4 tests become the
  refactor's safety net.
- **Verify:** full language suites green, graph output byte-identical on the eval
  corpora (diff node/edge dumps before/after)

### G5-C6 — Warn on parse errors; small extractor fixes
- **Files:** all extractors, `cartograph/pipeline.py`, tests
- **Does:** no extractor checks `tree.root_node.has_error` — broken files silently
  yield partial graphs; warn per-file (matches the 0f46fe6 "warn loudly" direction).
  Also: Python module docstring lost behind shebang/license comments
  (`extract.py:50-59`); embedded-SQL QUERIES/JOINS edges from bare-name fallback
  tagged EXTRACTED (`pipeline.py:204-224`) → INFERRED; SQL nodes all get
  `start_line=0` (`sql_extract.py:96`).
- **Verify:** language suites + `tests/test_sql.py -q`

### Deferred (recorded, not scheduled)
Scale work (sparse-matrix PPR, inverted-index BM25, `UNWIND` deletes, batched/deduped
Ollama embedding — fine at current corpus sizes); TS namespaces/re-exports/getters;
Go var-func-literals/type-aliases/interface-embedding; JSDoc/godoc/javadoc capture
(systematic embed-text bias against non-Python corpora — schedule when a non-Python
corpus joins the eval set); viz `layout_3d` O(n²) memory guard; viz title escaping.

### Open questions — resolved 2026-06-12
- **Q1 (G5-B1):** ✅ dirty-flag Meta. Portable, catches crashes outside any
  transaction scope, reuses existing Meta machinery.
- **Q2 (G5-C2):** ✅ same-module only. EXTRACTED when the base resolves within the
  same module (deterministic lexical resolution); cross-module unique-name matches
  demote to INFERRED.
- **Q3 (G5-A2):** ✅ clamp hops to 8 (eval's deepest MULTIHOP + headroom; Kuzu hard
  ceiling is 30). Out-of-range clamps with a note rather than erroring.

---

## Gate-6 — "impact you can trust" (added 2026-06-13)
The review panel's unanimous #1 and the bank-pilot's #1 condition: `impact` is the
moat feature, and its honesty/completeness is what a high-stakes (schema-migration)
workflow needs. The keep of the moat, hardened.

### G6-1 — Machine-readable completeness ✅ DONE 2026-06-13
- **Files:** `cartograph/service.py`, `cartograph/cli.py`, `cartograph/mcp_server.py`,
  `tests/test_impact.py`
- **Does:** every `impact` result carries a `completeness` block —
  `{exhaustive: false, advisory_only: true, limitations: [{code, detail}]}` with
  structured codes (`inferred_calls`, `orm_attribute_access`, `fk_join_ripple`) an
  agent can branch on instead of parsing prose. Purely additive (no change to which
  nodes are returned), so **eval-neutral**. CLI prints it; MCP docstring documents it.
- **Verify:** `uv run pytest tests/test_impact.py -q` (158 tests total green).

### G6-2 — FK/JOIN ripple ✅ DONE 2026-06-13
- **Does:** `impact` on a table/column follows the transitive closure of incoming
  `REFERENCES` (+ `JOINS`) so dropping `users` surfaces code touching `audit`/`orders`
  that reference it. `fk_join_ripple` retired; residual is `undeclared_schema_links`.
- **Verify:** `tests/test_impact.py` (FK-ripple tests on a referencing-table corpus).
  Pure `impact` logic — no graph change, retrieval scorecard untouched.

### G6-3 — ORM attribute capture ✅ DONE 2026-06-13
- **Does:** `self.<column>` reads in a mapped class's methods emit an INFERRED
  column-level `QUERIES` edge. Fixes the code→data false-negative (a method reading
  `self.email` now touches `users.email`) and makes data→code attribution precise.
- **Verify:** impact tests + scorecard gate (graph change): bridge/ai-digest reindexed,
  recall identical, bridge graph mrr 0.488→0.516, no regression.

### G6-4 — Per-edge confidence in responses ✅ DONE 2026-06-13
- **Does:** every `neighbors`/`calls`/`callers` result carries the edge's confidence
  (EXTRACTED vs INFERRED). Additive; surfaced in the CLI and MCP. The remaining
  `orm_attribute_access` limitation is narrowed to cross-instance access (type
  inference) — self-access is now captured.

**Status:** Gate-6 complete. Open follow-up (out of scope): cross-instance ORM
attribute access needs receiver-type inference (Jedi extension) — honestly reported as
a limitation rather than guessed.

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
