"""MCP server exposing the Cartograph graph to coding agents (e.g. Claude Code).

Tools (per SPEC): query, semantic_search, get_node, neighbors, shortest_path.
Transport is stdio. Requires the optional `mcp` extra:  uv sync --extra mcp

Config via env:
  CARTOGRAPH_DB        path to the indexed Kuzu graph (default: cartograph-out/graph.kuzu)
  CARTOGRAPH_EMBEDDER  override query embedder (normally auto-detected from the graph)

Everything runs locally; no data egress.
"""

from __future__ import annotations

import os
import sys

from .service import CartographService

DEFAULT_DB = "cartograph-out/graph.kuzu"


def build_server(service: CartographService | None = None, db_path: str | None = None):
    """Wrap a CartographService in a FastMCP server (imported lazily).

    When `service` is None it is constructed from `db_path` on the first tool
    call. Constructed eagerly in main() instead, a bad DB (missing path, old
    schema) would kill the process before the MCP handshake and the client
    would report only "Connection closed" — lazily, the same friendly error
    arrives as a tool result the agent can read and act on. A side benefit:
    fixing the DB doesn't require a server restart; the next call retries.
    """
    from mcp.server.fastmcp import FastMCP

    if service is None and db_path is None:
        raise ValueError("build_server needs a service or a db_path")

    mcp = FastMCP("cartograph")
    state: dict = {"service": service}

    def svc() -> CartographService:
        if state["service"] is None:
            state["service"] = CartographService(db_path)
        return state["service"]

    @mcp.tool()
    def query(text: str, mode: str = "hybrid", k: int = 10) -> list[dict]:
        """Search the code graph. `mode`: hybrid (default) | vector | graph | lexical
        (| rerank when the server is configured with CARTOGRAPH_RERANKER). Returns
        ranked nodes (functions/classes/methods/modules) with file, line, signature,
        docstring, and score; `k` is clamped to 1..100. Use this instead of grepping
        for structure."""
        return svc().query(text, mode=mode, k=k)

    @mcp.tool()
    def semantic_search(text: str, k: int = 10) -> list[dict]:
        """Pure semantic (vector) search over node embeddings — for concepts whose
        wording may not appear in the code (e.g. 'where is retry logic')."""
        return svc().semantic_search(text, k=k)

    @mcp.tool()
    def get_node(id: str) -> dict:
        """Full detail for one node. Accepts a node id OR a qualified name
        (e.g. `httpx._client.Client.send`) — you don't need the internal id format.
        Unknown refs are an error carrying 'did you mean' candidates — never an
        empty success."""
        return svc().get_node(id, strict=True)

    @mcp.tool()
    def resolve(ref: str) -> list[dict]:
        """Find node(s) matching a reference (qualified name or bare name). Use to get a
        precise id when a name is ambiguous (e.g. `send` -> Client.send, AsyncClient.send).
        Capped at 25 matches; a trailing {truncated, total, note} item flags the cut."""
        return svc().resolve(ref)

    @mcp.tool()
    def neighbors(id: str, direction: str = "both", relation: str = "", hops: int = 1) -> list[dict]:
        """Adjacent nodes, each labeled with `relation` (CALLS/INHERITS/IMPORTS/
        CONTAINS/DOCUMENTS) and `direction` (out=this node is the source, in=target).
        Filter with `direction` (out|in|both) and `relation` (e.g. CALLS). Don't guess
        direction — it's in the result. hops>1 expands unlabeled for broad context;
        hops is clamped to 1..8 and results to 200 — a trailing {truncated/note} item
        flags any cut. Invalid direction/relation values raise (they are not empty
        results)."""
        return svc().neighbors(id, direction=direction, relation=(relation or None), hops=hops)

    @mcp.tool()
    def calls(id: str) -> list[dict]:
        """What this node calls — outgoing CALLS edges only. Use for 'what does X call'.
        Accepts a node id or qualified name (e.g. `httpx._client.Client.send`)."""
        return svc().calls(id)

    @mcp.tool()
    def callers(id: str) -> list[dict]:
        """What calls this node — incoming CALLS edges only. Use for 'what calls X'.
        Accepts a node id or qualified name."""
        return svc().callers(id)

    @mcp.tool()
    def impact(ref: str) -> dict | None:
        """Blast radius across the code<->data bridge. For a table or table.column:
        the code touching it (incl. `self.<column>` reads on mapped classes), the
        code touching tables that FK/JOIN-reference it, and every caller that can
        reach that code ("what breaks if I drop users.email"). For a code symbol:
        every table/column reachable through its scope and transitive callees. The
        result carries a machine-readable `completeness` block (`exhaustive: false`,
        `advisory_only: true`, and a `limitations` list of codes like inferred_calls /
        orm_attribute_access / undeclared_schema_links) — treat a populated radius as
        advisory, never as a complete proof of what a schema change will break."""
        return svc().impact(ref)

    @mcp.tool()
    def shortest_path(src: str, dst: str) -> list[dict]:
        """Ordered nodes on a shortest path between two node ids (e.g. trace how one
        function reaches another). Empty if there is no path."""
        return svc().shortest_path(src, dst)

    return mcp


def main() -> None:
    db = os.environ.get("CARTOGRAPH_DB", DEFAULT_DB)
    # Preflight eagerly so a human at the terminal sees the problem immediately —
    # but stay up either way: dying here is what turns a one-line fix into an
    # opaque "Connection closed" in MCP clients. On failure the service is left
    # unset and each tool call retries construction, returning the same message.
    service = None
    try:
        service = CartographService(db)
    except (FileNotFoundError, RuntimeError) as e:
        print(f"cartograph serve: {e}", file=sys.stderr)
    server = build_server(service, db_path=db)
    server.run()  # stdio transport


if __name__ == "__main__":
    main()
