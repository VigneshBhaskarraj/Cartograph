# Cartograph MCP server

Exposes the code graph to coding agents (Claude Code, etc.) over MCP, so the agent
queries structure-aware context instead of grepping. Stdio transport, fully local —
no data egress.

## Tools
| Tool | What it answers |
| --- | --- |
| `query(text, mode="hybrid", k=10)` | Ranked nodes for a question. `mode`: `hybrid` (default) \| `vector` \| `graph` \| `lexical`. |
| `semantic_search(text, k=10)` | Pure vector search — concepts whose wording isn't in the code. |
| `get_node(id)` | Full detail for one node id. |
| `neighbors(id, hops=1)` | Callers/callees, base/subclasses, imports, containment around a node. |
| `shortest_path(src, dst)` | Ordered nodes on a shortest path between two node ids. |

Each node result carries `id`, `kind`, `name`, `qualified_name`, `file_path`,
`start_line`, `signature`, `docstring`, and (for ranked results) `score`.

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
