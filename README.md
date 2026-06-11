# Cartograph

**Local-first hybrid retrieval over your codebase.** Cartograph turns code, SQL schemas, infrastructure, and docs into a queryable knowledge graph that an AI coding agent consults instead of grepping — fully offline, with no data leaving your machine.

Built as a privacy-first, correctness-first alternative to cloud GraphRAG code tools, for environments where source code can't leave the building.

> **Status:** early development. See [`SPEC.md`](./SPEC.md) for the design and [`docs/eval-set-httpx.md`](./docs/eval-set-httpx.md) for the evaluation methodology.

## How it works
- **tree-sitter** extracts code structure locally (Python + TypeScript); SQL schemas are parsed deterministically — app code and DB schema land in one graph.
- **Kuzu** stores it all: a property graph in a single embedded file. (Vector search is currently exact brute-force cosine — deliberate, fully offline; Kuzu HNSW is the planned speed-only upgrade.)
- **Hybrid retrieval** fuses vector similarity, graph traversal (personalized PageRank), and BM25 keyword search with weighted RRF, plus an opt-in local-LLM reranker.
- A **local embedding model** (Ollama) and an optional **local LLM** mean zero network calls by default — enforced, not promised: a non-loopback `OLLAMA_HOST` is refused unless explicitly allowed.
- The graph is exposed over **MCP**, so any coding agent can query structure-aware context on demand.

## Install
Runs on macOS and Linux, Python 3.12+ — everything works offline.
```bash
# as a tool (recommended for using it on your repos; sql/ts extras enable those extractors):
uv tool install "cartograph[mcp,sql,ts] @ git+https://github.com/VigneshBhaskarraj/Cartograph"
cartograph --help    # tool installs use plain `cartograph`, no `uv run` prefix

# or for development:
git clone https://github.com/VigneshBhaskarraj/Cartograph && cd Cartograph
uv sync --all-extras
uv run pytest        # the whole suite runs offline and deterministically
```

## Quickstart
```bash
# Index any Python/TypeScript/SQL file or package, then query the graph
uv run cartograph index path/to/pkg --db cartograph-out/graph.kuzu
uv run cartograph query "what calls encode_request" --mode hybrid --k 8
```
Modes: `vector`, `graph`, `lexical`, `hybrid` (weighted RRF fusion). The default embedder
is an offline feature-hash model — set `CARTOGRAPH_EMBEDDER=ollama` for real local
semantic embeddings (still zero egress; only talks to `127.0.0.1`).

### Use it from Claude Code (MCP)
```bash
uv sync --extra mcp
uv run cartograph serve --db cartograph-out/graph.kuzu     # stdio MCP server
```
Or in your project's `.mcp.json` (use **absolute** paths — the server resolves `--db`
relative to its working directory):
```json
{
  "mcpServers": {
    "cartograph": {
      "command": "uv",
      "args": ["run", "--directory", "/abs/path/to/Cartograph",
               "cartograph", "serve", "--db", "/abs/path/to/Cartograph/cartograph-out/graph.kuzu"]
    }
  }
}
```
Exposes `query` / `semantic_search` / `get_node` / `neighbors` / `calls` / `callers` /
`shortest_path` so an agent queries structure instead of grepping. Wiring + tool
reference: [`docs/mcp.md`](./docs/mcp.md).

### Measure it — including against the alternatives
Retrieval quality is measured, never asserted (see `CLAUDE.md`). The one-command
dashboard indexes five corpora (89 questions — one corpus held out from all tuning)
and scores every retriever **plus two external baselines**: `grep` over raw source,
and `naive-rag` (structure-blind chunk embeddings):
```bash
bash eval/get_corpus.sh 0.27.2 && bash eval/get_flask.sh && bash eval/get_aidigest.sh && bash eval/get_click.sh
uv run python eval/scorecard.py --baselines     # offline; --embedder ollama for real numbers
```
**Does it help an agent, though?** `eval/agent_bench/` measures exactly that: an
agent answers navigation tasks with grep-only vs cartograph-only tools. Pilot
(matched surfaces, 12 source-verified tasks): **equal success, 42% fewer tool
calls** for the Cartograph condition — full method, numbers, and caveats in
[`eval/agent_bench/RESULTS.md`](./eval/agent_bench/RESULTS.md); reproduce offline
with any local Ollama chat model via `eval/agent_bench/run_bench.py`.

## Status / roadmap
- [x] M0 — vertical slice (extract → store → embed → query, one Python file → httpx)
- [x] M1 — evaluation harness (21 questions over `httpx`, recall@k / precision@k / MRR, per-mode)
- [x] M2 — hybrid retrieval + reranker (RRF fusion + personalized-PageRank graph + opt-in LLM reranker)
- [x] M3 — real symbol resolution: `self.`-call class resolution + opt-in **Jedi** receiver-type inference (`--resolver jedi`); call-edge precision 0.50 → **1.0** on the ground-truthed set
- [x] M4 — MCP server ✅ + incremental indexing ✅ (`cartograph update`: per-file SHA change detection, instant no-op; **row-level delta** — unchanged node rows kept, only changed ones recreated; re-embeds only changed symbols) + SQL-schema-in-graph ✅ (`CREATE TABLE` → `table`/`column` nodes + FK `REFERENCES` edges via `sqlglot`, `--extra sql`) — app code + DB schema in one graph
- [x] M5 — code↔schema bridge: ORM `__tablename__` → table (`MAPS_TO`) **and** raw-SQL embedded in Python → tables + `QUERIES` edges (function → table/**column**) + `JOINS` (table↔table from query JOINs); schema-bridging eval on a synthetic corpus (recall@10 **1.0**) **and the real `ai-digest` repo** (~0.86); generalized eval runner (`--questions`/`--db`)
- [x] M6 — second language: **TypeScript/TSX** extractor via tree-sitter (`--extra ts`) — classes, interfaces, functions, arrow-const functions, methods, `extends` (INHERITS), imports, heuristic calls — into the same graph (polyglot: Python + TS in one store)

**Latest eval** (real `nomic-embed-text` embeddings, 4 corpora / 51 questions). After
calibrating fusion on the sweep (`eval/fusion_sweep.py`), **weighted `hybrid` now wins
or ties vector on recall@10 across every corpus** and lifts the aggregate to
**recall@5 0.909 / recall@10 0.961 / MRR 0.735** (vs vector 0.885 / 0.937 / 0.707).
Honest caveat: the MRR edge is partly carried by one corpus, so the durable,
generalizing win is *recall* — see [`SPEC.md`](./SPEC.md) §8. The opt-in **LLM reranker**
(`gemma3:12b`) further leads top-rank quality. Full tables, the offline baseline, and
the reranker trade-off: [`eval/README.md`](./eval/README.md).

## License
[Apache License 2.0](./LICENSE) — permissive, with an explicit patent grant suited to
regulated-enterprise adoption. See also [`NOTICE`](./NOTICE).
