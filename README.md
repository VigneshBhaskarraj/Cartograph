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

## Status / roadmap
- [x] M0 — vertical slice (extract → store → embed → query, one Python file → httpx)
- [x] M1 — evaluation harness (21 questions over `httpx`, recall@k / precision@k / MRR, per-mode)
- [~] M2 — hybrid retrieval + reranker (RRF fusion + personalized-PageRank graph landed; reranker pending)
- [ ] M3 — real symbol resolution (SCIP / stack-graphs)
- [ ] M4 — MCP server, incremental updates, SQL-schema-in-graph

**Latest eval** (httpx==0.27.2, offline embedder): `hybrid+rrf` leads on recall@5 (0.67)
and MRR (0.36) over both single-signal baselines and ties them on recall@10 (0.76). The
graph leg now uses personalized PageRank — graph-only MULTIHOP recall hit 1.0. The
remaining ordering gap is what the M2 reranker targets. Honest table and interpretation:
[`eval/README.md`](./eval/README.md).

## License
[Apache License 2.0](./LICENSE) — permissive, with an explicit patent grant suited to
regulated-enterprise adoption. See also [`NOTICE`](./NOTICE).
