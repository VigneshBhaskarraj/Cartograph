import asyncio
from pathlib import Path

import pytest

pytest.importorskip("mcp")

from cartograph.mcp_server import build_server  # noqa: E402
from cartograph.pipeline import index_path  # noqa: E402
from cartograph.service import CartographService  # noqa: E402

FIX = Path(__file__).parent / "fixtures" / "sample.py"


def test_mcp_server_registers_all_tools(tmp_path):
    db = tmp_path / "g.kuzu"
    index_path(FIX, db, dim=64, overwrite=True).close()
    server = build_server(CartographService(db))
    tools = {t.name for t in asyncio.run(server.list_tools())}
    assert {
        "query", "semantic_search", "get_node", "resolve",
        "calls", "callers", "neighbors", "shortest_path",
    } <= tools
