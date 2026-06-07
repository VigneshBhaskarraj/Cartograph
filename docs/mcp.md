# Cartograph MCP server

Exposes the code graph to coding agents (Claude Code, etc.) over MCP, so the agent
queries structure-aware context instead of grepping. Stdio transport, fully local —
no data egress.

## Tools
| Tool | What it answers |
| --- | --- |
| `query(text, mode="hybrid", k=10)` | Ranked nodes for a question. `mode`: `hybrid` (default) \| `vector` \| `graph` \| `lexical` \| `rerank`. |
| `semantic_search(text, k=10)` | Pure vector search — concepts whose wording isn't in the code. |
| `get_node(id)` | Full detail for one node — **id or qualified name**. |
| `resolve(ref)` | Node(s) matching a reference — disambiguate a bare name. |
| `calls(id)` | **What this node calls** (outgoing CALLS edges only). |
| `callers(id)` | **What calls this node** (incoming CALLS edges only). |
| `neighbors(id, direction="both", relation="", hops=1)` | Adjacent nodes, each labeled with `relation` and `direction`. Filter by `direction` (out\|in\|both) and `relation` (e.g. CALLS, INHERITS). |
| `shortest_path(src, dst)` | Ordered nodes on a shortest path between two node ids. |

Each node result carries `id`, `kind`, `name`, `qualified_name`, `file_path`,
`start_line`, `signature`, `docstring`, and — where applicable — `score` (ranked
results) or `relation`/`direction` (neighbor results). **Direction and edge type come
from the graph**, so an agent answers "what does X call" with `calls(id)` instead of
guessing from an undirected blob.

**Node references:** every id-taking tool accepts either the internal node id
(`httpx/_client.py::httpx._client.Client.send#891`) **or** a qualified name
(`httpx._client.Client.send`) / bare name (`send`). Ambiguous names resolve to the most
direct match; use `resolve(ref)` to list all candidates and pick a precise id.

## Setup
```bash
uv sync --extra mcp                                   # install the MCP SDK
# index whatever repo you want to serve (real embeddings recommended):
CARTOGRAPH_EMBEDDER=ollama uv run cartograph index /path/to/repo --db cartograph-out/graph.kuzu --embedder ollama
```
The query-time embedder is **auto-detected** from metadata recorded at index time, so
the server matches whatever you indexed with. (If you indexed with Ollama, keep Ollama
running while the server is up.)

## Add to Claude Code
CLI:
```bash
claude mcp add cartograph -- \
  uv run --directory /abs/path/to/Cartograph cartograph serve --db cartograph-out/graph.kuzu
```
Or `.mcp.json` in your project:
```json
{
  "mcpServers": {
    "cartograph": {
      "command": "uv",
      "args": ["run", "--directory", "/abs/path/to/Cartograph",
               "cartograph", "serve", "--db", "cartograph-out/graph.kuzu"]
    }
  }
}
```
Then in Claude Code, ask things like *"use cartograph to find what Client.send calls"* —
it'll call `query`/`neighbors`/`shortest_path` against the graph.

## Notes
- The server only ever reads the graph; it never re-reads source files at query time.
- `serve` writes nothing to stdout except the MCP protocol (stdio-safe).
- `rerank` mode becomes available automatically if the M2 reranker is present.
