# CLAUDE.md — Cartograph

Cartograph turns a codebase — plus its SQL schemas, infra, and docs — into a queryable knowledge graph that an AI coding agent consults instead of grepping. It is knowledge infrastructure *for agents*, exposed over MCP, and it runs fully offline. Full rationale and architecture: **SPEC.md**.

## Prime directives (the leash)
1. **Plan before code.** For any non-trivial task, work in plan mode first; show the plan and wait for my approval.
2. **Small, verifiable diffs.** One `PLAN.md` task per change. After each, run its verification command and show me the result before moving on.
3. **Eval-first.** No retrieval-quality change lands without a number that moved on the M1 eval (`docs/eval-set-httpx.md`). Quality is measured, never asserted.
4. **Ask before adding.** Never add a dependency, a new language extractor, or a network call without flagging it to me first.
5. **Zero data egress by default.** Embeddings and any LLM calls run locally (Ollama). A cloud backend is opt-in only, behind an explicit flag.
6. **Independent review.** Before calling a milestone done, use a subagent in a fresh context to review the diff against `PLAN.md` and report gaps — don't just declare success.

## Stack
- Python 3.12, managed with **`uv`**. Tests with **`pytest`**.
- **Store:** Kuzu — embedded; property graph + native HNSW vector index + full-text search, in one file.
- **Extraction:** tree-sitter (**Python first**). Deterministic **SQL schema** parsing is in scope for the MVP.
- **Embeddings:** a local code-aware model via Ollama.
- **Interface:** an MCP server exposing `query`, `get_node`, `neighbors`, `shortest_path`, `semantic_search`.
- **MIT-compatible dependencies only.**

## Architecture invariants (don't violate without updating SPEC.md)
- The graph is the single source of truth. Retrieval reads the graph; it never re-reads source files at query time.
- Retrieval is **hybrid**: vector ANN + graph traversal/PPR + BM25, fused (RRF first), reranked optionally. No single-signal retrieval in the final path.
- Every edge carries a confidence tag: `EXTRACTED` (deterministic) or `INFERRED` (score **calibrated against the eval set**, not a hard-coded rubric).
- Incremental updates use a SHA256 content cache **and** correctly delete stale edges when a symbol is removed.

## Scope discipline (non-goals for now)
- Python (+ SQL schema) only. No multi-language breadth yet.
- No browser visualization in the MVP.
- No cloud LLM in the default path.
- Not competing on distribution. We win on **correctness + privacy** in a regulated-codebase niche.

## Repo conventions
- Source in `cartograph/`. Tests in `tests/`. Specs and docs in `docs/`.
- Commit style: `feat:` / `fix:` / `docs:` / `test:` / `chore:`.
- Keep this file lean. Durable decisions and reasoning belong in `SPEC.md` or `docs/`, not here.
