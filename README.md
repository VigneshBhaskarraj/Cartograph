# Cartograph

**A queryable knowledge graph of your codebase — for AI agents, fully offline.**
Cartograph turns code **and its SQL schema** into one graph that a coding agent
consults instead of grepping. No code ever leaves your machine.

Built for environments where source can't go to the cloud (regulated / on-prem), and
for the one question neither code tools nor data-lineage tools answer alone:

> **"What application code breaks if I drop `users.email`?"**

```bash
uv run cartograph impact users.email     # → the columns, the code, and its transitive callers
```

> **Status:** alpha. The retrieval quality and the code↔data bridge are measured
> (numbers below); the API and CLI may still change. Design rationale: [`SPEC.md`](./SPEC.md).

## How it works
- **tree-sitter** extracts structure locally — **Python, JavaScript/TypeScript (incl. CommonJS), Java (incl. JPA annotations), Go** — and **SQL** schemas are parsed deterministically, so app code and database schema land in the *same* graph.
- **Kuzu** stores it as a property graph in a single embedded file (no server, no DBA).
- **Hybrid retrieval** fuses vector similarity, graph traversal (personalized PageRank), and BM25 keyword search, with an optional local-LLM reranker.
- **Zero network calls by default — enforced, not promised:** embeddings run on a local Ollama model, and a non-loopback `OLLAMA_HOST` is *refused* unless you explicitly allow it.
- The graph is served over **MCP**, so any coding agent (Claude Code, etc.) can query structure-aware context on demand.

## Install
macOS / Linux, Python 3.12+. Everything runs offline.
```bash
uv tool install "cartograph[mcp,sql,ts,java,go] @ git+https://github.com/VigneshBhaskarraj/Cartograph"
cartograph --help
```
Or for development:
```bash
git clone https://github.com/VigneshBhaskarraj/Cartograph && cd Cartograph
uv sync --all-extras && uv run pytest    # the whole suite runs offline and deterministically
```

## Quickstart
```bash
cartograph index path/to/repo --db graph.kuzu
cartograph query "where is retry logic" --mode hybrid
```
The default embedder is an offline feature-hash model; set `CARTOGRAPH_EMBEDDER=ollama`
for real local semantic embeddings (still zero egress — only `127.0.0.1`).

## Use it from a coding agent (MCP)
```bash
cartograph serve --db /abs/path/to/graph.kuzu     # stdio MCP server
```
Or add to your project's `.mcp.json` (use **absolute** paths):
```json
{
  "mcpServers": {
    "cartograph": {
      "command": "cartograph",
      "args": ["serve", "--db", "/abs/path/to/graph.kuzu"]
    }
  }
}
```
Tools: `query` · `semantic_search` · `get_node` · `neighbors` · `calls` · `callers` ·
`shortest_path` · `impact`. Full reference: [`docs/mcp.md`](./docs/mcp.md).

## The differentiator: code↔data blast radius
Code tools stop at code; data-lineage tools stop at SQL. Cartograph holds both, so
`impact` traces across the boundary — function-level, offline:
```bash
cartograph impact users.email    # a column/table → every code path that can reach it
cartograph impact store_run      # a function → every table/column it can touch
```
The radius follows the call graph, the ORM/SQL bridge (including `self.<column>`
reads), and **FK/JOIN ripple** — dropping a table also surfaces code touching tables
that reference it. Every result carries a machine-readable `completeness` block (it's
advisory, never a proof): each remaining gap — heuristic calls, cross-instance
attribute access, undeclared foreign keys — is reported as a structured code an agent
can branch on. Honest by construction.

## See it
Export an interactive **3D map** of any graph to one self-contained, offline HTML file —
rotate, search, click a symbol for its neighborhood, trace shortest paths, filter by
edge type and EXTRACTED/INFERRED confidence:
```bash
cartograph viz --db graph.kuzu --out graph.html
```

## Does it actually work? (measured, not asserted)
Real `nomic-embed-text` embeddings, **101 questions / 6 corpora** spanning Python,
Java+JPA, and raw-SQL apps; two corpora are **held out** from all tuning. Mean across
corpora, against two external baselines:

| system | recall@5 | recall@10 | mrr |
|---|---|---|---|
| `grep` over raw source | 0.58 | 0.72 | 0.40 |
| `naive-rag` (structure-blind chunks) | 0.54 | 0.76 | 0.28 |
| vector (single signal) | 0.85 | 0.91 | 0.71 |
| **Cartograph hybrid** | **0.87** | **0.93** | **0.73** |

The fusion **generalizes** — hybrid wins or ties single-signal vector on recall@10 on
all six corpora, including both held-out ones (`click` 0.944; `spring-petclinic`
0.833). Agent-task evidence: with grep-only vs Cartograph-only tools, an agent reaches
the same answers with **~42% fewer tool calls**.

Reproduce (offline):
```bash
bash eval/get_corpus.sh 0.27.2 && bash eval/get_flask.sh
uv run python eval/scorecard.py --baselines      # add --embedder ollama for the real numbers
```
Full tables, methodology, and the agent benchmark: [`eval/README.md`](./eval/README.md)
and [`eval/agent_bench/RESULTS.md`](./eval/agent_bench/RESULTS.md).

## Language support
| tier | languages | what it means |
|---|---|---|
| **Evaluated** | Python (+SQL bridge), Java (+JPA bridge) | retrieval quality measured on real corpora |
| **Structural** | TypeScript, JavaScript (ES + CommonJS), Go | extraction verified by tests + real-repo smoke (Express, go-chi); no retrieval numbers yet |

Further languages are demand-driven — open an issue.

## License
[Apache License 2.0](./LICENSE) — permissive, with an explicit patent grant. See [`NOTICE`](./NOTICE).
