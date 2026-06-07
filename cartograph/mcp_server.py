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

from .service import CartographService

DEFAULT_DB = "cartograph-out/graph.kuzu"


def build_server(service: CartographService):
    """Wrap a CartographService in a FastMCP server (imported lazily)."""
    from mcp.server.fastmcp import FastMCP

    mcp = FastMCP("cartograph")

    @mcp.tool()
    def query(text: str, mode: str = "hybrid", k: int = 10) -> list[dict]:
        """Search the code graph. `mode`: hybrid (default) | vector | graph | lexical.
        Returns ranked nodes (functions/classes/methods/modules) with file, line,
        signature, docstring, and score. Use this instead of grepping for structure."""
        return service.query(text, mode=mode, k=k)

    @mcp.tool()
    def semantic_search(text: str, k: int = 10) -> list[dict]:
        """Pure semantic (vector) search over node embeddings — for concepts whose
        wording may not appear in the code (e.g. 'where is retry logic')."""
        return service.semantic_search(text, k=k)

    @mcp.tool()
    def get_node(id: str) -> dict | None:
        """Full detail for one node by its id (as returned by other tools)."""
        return service.get_node(id)

    @mcp.tool()
    def neighbors(id: str, hops: int = 1) -> list[dict]:
        """Nodes within `hops` edges of a node — callers/callees, base/subclasses,
        imports, and containment. Use to expand context around a symbol."""
        return service.neighbors(id, hops=hops)

    @mcp.tool()
    def shortest_path(src: str, dst: str) -> list[dict]:
        """Ordered nodes on a shortest path between two node ids (e.g. trace how one
        function reaches another). Empty if there is no path."""
        return service.shortest_path(src, dst)

    return mcp


def main() -> None:
    db = os.environ.get("CARTOGRAPH_DB", DEFAULT_DB)
    service = CartographService(db)
    server = build_server(service)
    server.run()  # stdio transport


if __name__ == "__main__":
    main()
