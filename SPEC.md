# Cartograph — Design Spec

**Status:** Draft v0.1 · **Owner:** Vignesh Bhaskarraj
**One line:** Local-first hybrid retrieval that turns code + SQL schemas + infra + docs into a knowledge graph an AI coding agent queries instead of grepping.

---

## 1. Problem & overview
AI coding agents waste context and tokens re-reading and grepping source files to answer structural questions ("what calls this?", "what connects auth to the database?", "where is retry logic?"). Existing tools that build a code knowledge graph either (a) rely on name-matching for relationships, (b) freeze semantic similarity at extraction time, or (c) ship file content to a cloud LLM — which is a non-starter for code that legally cannot leave the building.

Cartograph builds a queryable knowledge graph of a codebase and serves it to agents over MCP, doing **hybrid retrieval over a real store with real symbol resolution, fully offline.**

## 2. The strategic bet
**Local-first hybrid retrieval for regulated codebases (BFSI / fintech / healthcare).** Our single differentiator: correct hybrid (vector + graph + keyword) retrieval that **never makes a network call by default.** That is the niche frontier labs won't serve and enterprises can't buy off the shelf. We do not try to out-distribute incumbents — we win on correctness and privacy in a niche.

This is deliberately positioned as a fix for the structural weaknesses of cloud GraphRAG code tools (e.g. graphify): batch-frozen semantic links, name-matched call edges, an in-memory graph that doesn't scale, and a default path that leaks document content.

## 3. Target architecture (compose these — don't reinvent)
- **Extraction (free, local):** tree-sitter AST — classes, functions, imports, call sites, and inline rationale comments (`# NOTE:`, `# WHY:`, docstrings) as their own nodes. **Python only** to start. Deterministic **SQL schema** parsing (tables, foreign keys, JOIN relationships) is in MVP scope because "app code + DB schema in one graph" is the point.
- **Symbol resolution (the precision upgrade):** resolve *real* call edges, not name guesses. Evaluate **SCIP** (Sourcegraph) and **stack-graphs** (GitHub) for Python. If integration is heavy, MVP ships with tree-sitter heuristic edges, recorded as the known precision gap to close in M3.
- **Store:** **Kuzu** — one embedded engine: property graph + Cypher + native disk-based HNSW vector index + full-text search. Single file, serverless, git-friendly, MIT. Replaces in-memory NetworkX + JSON and removes any node-count visualization ceiling.
- **Embeddings (local):** a local code-aware embedding model via Ollama. Each node is embedded as `label + signature + docstring/comments` (not just the name) at ingest and stored in Kuzu's vector index.
- **Retrieval (hybrid — the core fix):** fuse three signals, then optionally rerank —
  1. **Vector ANN** (Kuzu HNSW) for semantic recall across the whole corpus.
  2. **Graph traversal / personalized PageRank** for structural multi-hop context.
  3. **BM25 / full-text** for exact symbol names.
  Fusion starts with **Reciprocal Rank Fusion (RRF)**; a small local cross-encoder reranker is a second stage added only when RRF plateaus. (Full design: `docs/eval-set-httpx.md`.)
- **Confidence, done honestly:** keep `EXTRACTED` vs `INFERRED` tags, but **calibrate** INFERRED scores against the eval set instead of hard-coding a rubric.
- **Interface:** an **MCP server** exposing `query`, `get_node`, `neighbors`, `shortest_path`, `semantic_search`. A thin CLI is fine. No HTML viz in the MVP.
- **Incremental correctness:** SHA256 content-hash cache for changed-file re-extraction, **plus correct edge cleanup when a symbol is deleted.**

## 4. Milestones
- **M0 — Vertical slice.** Index one real Python repo end-to-end: tree-sitter → Kuzu (nodes + edges) → embed nodes → answer one hardcoded query through *both* a vector search and a 2-hop graph traversal. Prove the whole pipe on one path before broadening.
- **M1 — Eval harness.** ~20 fixed questions over a known repo (`httpx`) with expected answer nodes; report recall@k, precision@k, MRR, broken down per retrieval mode. Everything after this is measured against it. See `docs/eval-set-httpx.md`.
- **M2 — Hybrid retrieval + reranker.** Fuse vector + graph + BM25 (RRF), add the reranker; beat vector-only and graph-only baselines on the eval.
- **M3 — Symbol resolution.** Replace heuristic call edges with SCIP / stack-graphs; show the precision gain on the eval set.
- **M4 — MCP server + incremental update + SQL-schema-in-graph.** Wire into Claude Code and dogfood on this repo (and/or `ai-digest`).

## 5. Stack & constraints
- Python 3.12, `uv`, `pytest`.
- Runs on Apple Silicon (M4), fully offline by default.
- MIT-compatible dependencies only.

## 6. Non-goals
- No 28-language breadth. ~~Python (+ SQL schema) first.~~ *Amended for release
  (2026-06-12):* the launch set is **Python, JavaScript/TypeScript, Java, Go (+ SQL)**
  — enough coverage that strangers can test on their own repos. Each extractor is a
  self-contained module feeding the same graph model. Further languages are
  demand-driven only; breadth without users is still a non-goal.
- ~~No browser visualization for the MVP.~~ *Amended post-MVP (2026-06-12):*
  `cartograph viz` ships as a **viewer/demo artifact only** — a static, offline,
  self-contained HTML export. The retrieval engine never depends on it, and it makes
  no network calls (the zero-egress promise extends to the page itself).
- No cloud LLM in the default path.
- Not competing on distribution or star count.

## 7. Open questions (resolve during planning, don't silently decide)
- Which local embedding model (general vs code-specialized; dimension vs speed on M4)?
- SCIP vs stack-graphs for Python, and whether to attempt it in MVP or defer to M3.
- Is the Kuzu DB committed to git (team-map convenience) or regenerated per checkout?
- Node/edge schema: one node table with a `kind` property, or separate tables per kind?
- Reranker model choice and whether it's worth the latency for the MVP.

## 8. As-built deviations from §3 (kept honest, per CLAUDE.md)
- **Vector search** is exact brute-force NumPy cosine, not Kuzu HNSW: the HNSW
  extension is a network download, which breaks offline-by-default. Identical recall;
  HNSW remains the speed-only upgrade path. **Keyword search** is a hand-rolled BM25
  over a code-aware tokenizer, not Kuzu FTS, for the same reason.
- **Fusion** is *weighted* RRF (`cartograph/retrieve.py`), calibrated on the
  2026-06-11 ollama sweep (`eval/fusion_sweep.py`, 4 corpora / 51 questions).
  Equal-weight RRF with `rrf_k=60` let two low-precision signals (graph, lexical)
  outvote the high-precision vector signal — hybrid lost to vector-alone on every
  aggregate metric (0.763/0.901/0.679 vs 0.885/0.937/0.707 for r@5/r@10/mrr). The
  shipped default — `weights=(3.0, 0.5, 0.5)`, `rrf_k=10`, `depth=50` — is
  vector-dominant: it wins or ties vector on recall@10 on **all four** corpora and
  lifts aggregate r@5/mrr to 0.909/0.961/0.735. Honest caveat: the aggregate **mrr**
  edge is partly carried by one corpus (ai-digest); leave-one-corpus-out is unstable
  on mrr, so the robust, generalizing win here is *recall* (hybrid no longer loses to
  vector), not a dramatic ranking gain. Because the winning region is vector-dominant,
  the worst case of these defaults is "behaves like vector." Defaults move only via a
  fresh sweep; a held-out 5th corpus would tighten the calibration.
- **Symbol resolution** ships tree-sitter heuristics + opt-in Jedi receiver-type
  inference (`--resolver jedi`) instead of SCIP/stack-graphs (integration weight).
  INHERITS edges resolve by base-class name: a *unique* corpus-wide match is tagged
  EXTRACTED (could in principle be misled by an external class shadowed by one
  same-named internal class); ambiguous multi-matches are honestly INFERRED.
- **Zero egress is enforced**, not promised: a non-loopback `OLLAMA_HOST` raises
  unless `CARTOGRAPH_ALLOW_REMOTE_OLLAMA=1` is set explicitly.
- The project license is **Apache-2.0** (patent grant for enterprise adoption);
  dependencies remain MIT-compatible per §5 (Kuzu itself is MIT).
- **Visualization is dependency-free by design**: the force layout runs in numpy at
  export time (vectorized 3D Fruchterman–Reingold); the browser side is hand-rolled
  vanilla JS (projection, picking, BFS focus/paths) — no vendored libraries, no CDN.
