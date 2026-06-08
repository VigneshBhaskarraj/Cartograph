# Cartograph

**Local-first hybrid retrieval over your codebase.** Cartograph turns code, SQL schemas, infrastructure, and docs into a queryable knowledge graph that an AI coding agent consults instead of grepping — fully offline, with no data leaving your machine.

Built as a privacy-first, correctness-first alternative to cloud GraphRAG code tools, for environments where source code can't leave the building.

> **Status:** early development. See [`SPEC.md`](./SPEC.md) for the design and [`docs/eval-set-httpx.md`](./docs/eval-set-httpx.md) for the evaluation methodology.

## How it works (planned)
- **tree-sitter** extracts code structure locally (free); SQL schemas are parsed deterministically — app code and DB schema land in one graph.
- **Kuzu** stores it all: a property graph + HNSW vector index + full-text search in a single embedded file.
- **Hybrid retrieval** fuses vector similarity, graph traversal, and keyword search, then reranks.
- A **local embedding model** and an optional **local LLM** mean zero network calls by default.
- The graph is exposed over **MCP**, so any coding agent can query structure-aware context on demand.

## Quickstart
```bash
uv sync --extra dev                         # install (Python 3.12)
uv run pytest                               # 17 tests, fully offline

# Index any Python file or package, then query the graph
uv run cartograph index path/to/pkg --db cartograph-out/graph.kuzu
uv run cartograph query "what calls encode_request" --mode hybrid --k 8
```
Modes: `vector`, `graph`, `lexical`, `hybrid` (RRF fusion). The default embedder is an
offline feature-hash model — set `CARTOGRAPH_EMBEDDER=ollama` for real local semantic
embeddings (still zero egress; only talks to `127.0.0.1`).

### Use it from Claude Code (MCP)
```bash
uv sync --extra mcp
uv run cartograph serve --db cartograph-out/graph.kuzu     # stdio MCP server
```
Exposes `query` / `semantic_search` / `get_node` / `neighbors` / `shortest_path` so an
agent queries structure instead of grepping. Wiring + tool reference: [`docs/mcp.md`](./docs/mcp.md).

## Status / roadmap
- [x] M0 — vertical slice (extract → store → embed → query, one Python file → httpx)
- [x] M1 — evaluation harness (21 questions over `httpx`, recall@k / precision@k / MRR, per-mode)
- [x] M2 — hybrid retrieval + reranker (RRF fusion + personalized-PageRank graph + opt-in LLM reranker)
- [x] M3 — real symbol resolution: `self.`-call class resolution + opt-in **Jedi** receiver-type inference (`--resolver jedi`); call-edge precision 0.50 → **1.0** on the ground-truthed set
- [x] M4 — MCP server ✅ + incremental indexing ✅ (`cartograph update`: per-file SHA change detection, instant no-op; **row-level delta** — unchanged node rows kept, only changed ones recreated; re-embeds only changed symbols) + SQL-schema-in-graph ✅ (`CREATE TABLE` → `table`/`column` nodes + FK `REFERENCES` edges via `sqlglot`, `--extra sql`) — app code + DB schema in one graph
- [x] M5 — code↔schema bridge: ORM `__tablename__` → table (`MAPS_TO`) **and** raw-SQL embedded in Python → tables + `QUERIES` edges (function → table); schema-bridging eval on a synthetic corpus (recall@10 **1.0**) **and the real `ai-digest` repo** (~0.86); generalized eval runner (`--questions`/`--db`)

**Latest eval** (httpx==0.27.2, real `nomic-embed-text` embeddings). `vector`/`hybrid`
reach recall@10 **0.81**; the opt-in **LLM reranker** (`gemma3:12b`, blended with the
fused order) leads top-rank quality — **MRR 0.583, recall@5 0.714, precision@5 0.40** —
at a small recall@10/EXACT trade, so `hybrid` stays the default. Full tables, the
offline baseline, and the reranker trade-off: [`eval/README.md`](./eval/README.md).

## License
[Apache License 2.0](./LICENSE) — permissive, with an explicit patent grant suited to
regulated-enterprise adoption. See also [`NOTICE`](./NOTICE).
