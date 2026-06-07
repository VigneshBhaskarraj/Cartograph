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

## Status / roadmap
- [ ] M0 — vertical slice (extract → store → embed → query, one Python repo)
- [ ] M1 — evaluation harness
- [ ] M2 — hybrid retrieval + reranker
- [ ] M3 — real symbol resolution (SCIP / stack-graphs)
- [ ] M4 — MCP server, incremental updates, SQL-schema-in-graph

## License
MIT
