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
        "calls", "callers", "neighbors", "shortest_path", "impact",
    } <= tools


def test_startup_error_surfaces_as_tool_error(tmp_path):
    """A bad DB must NOT kill the server pre-handshake ("Connection closed");
    the friendly message must come back through the tool call instead."""
    server = build_server(db_path=str(tmp_path / "missing.kuzu"))
    tools = {t.name for t in asyncio.run(server.list_tools())}  # handshake-equivalent works
    assert "query" in tools
    with pytest.raises(Exception, match="no graph at"):
        asyncio.run(server.call_tool("query", {"text": "anything"}))


def test_lazy_service_recovers_once_db_appears(tmp_path):
    """Fixing the DB while the server is up must not require a restart."""
    db = tmp_path / "g.kuzu"
    server = build_server(db_path=str(db))
    with pytest.raises(Exception, match="no graph at"):
        asyncio.run(server.call_tool("query", {"text": "x"}))
    index_path(FIX, db, dim=64, overwrite=True).close()
    content, result = asyncio.run(server.call_tool("query", {"text": "send"}))
    assert result  # ranked nodes came back after the retry


def test_old_schema_graph_heals_on_open(tmp_path):
    """A graph missing rel tables a newer Cartograph added (the JOINS/QUERIES
    incident) is healed in place — empty tables created, version stamped —
    instead of demanding a full re-index."""
    import warnings

    import kuzu

    db = tmp_path / "g.kuzu"
    index_path(FIX, db, dim=64, overwrite=True).close()
    conn = kuzu.Connection(kuzu.Database(str(db)))
    conn.execute("DROP TABLE QUERIES")
    conn.execute("DROP TABLE JOINS")
    conn.close()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        service = CartographService(db)
    assert any("JOINS, QUERIES" in str(w.message) for w in caught)
    assert {"QUERIES", "JOINS"} <= service.store.table_names()
    assert service.store.get_meta("schema_version") is not None
    service.store.close()


def test_call_tool_round_trips_and_error_paths(tmp_path):
    """Protocol-level coverage between service and wire — where the hops crash
    and silent-empty behaviors used to live."""
    db = tmp_path / "g.kuzu"
    index_path(FIX, db, dim=64, overwrite=True).close()
    server = build_server(CartographService(db))

    content, result = asyncio.run(server.call_tool("query", {"text": "dog bark"}))
    assert result  # ranked nodes through the wire

    with pytest.raises(Exception, match="no node matches"):
        asyncio.run(server.call_tool("get_node", {"id": "does.not.exist"}))
    with pytest.raises(Exception, match="mode must be one of"):
        asyncio.run(server.call_tool("query", {"text": "x", "mode": "bogus"}))
    with pytest.raises(Exception, match="direction"):
        asyncio.run(server.call_tool("neighbors", {"id": "Dog", "direction": "sideways"}))
    with pytest.raises(Exception, match="no node matches"):
        asyncio.run(server.call_tool("calls", {"id": "does.not.exist"}))


def test_versionless_old_graph_heals_too(tmp_path):
    """Review follow-up: the REAL incident graph predates the schema_version
    meta entirely (rel tables and version meta landed in that order), so the
    heal path must accept version=None — while the gate still rejects a
    versionless graph that has all its tables (no provenance story)."""
    import kuzu

    db = tmp_path / "g.kuzu"
    index_path(FIX, db, dim=64, overwrite=True).close()
    conn = kuzu.Connection(kuzu.Database(str(db)))
    conn.execute("DROP TABLE QUERIES")
    conn.execute("DROP TABLE JOINS")
    conn.execute("MATCH (m:Meta) WHERE m.key = 'schema_version' DELETE m")
    conn.close()

    service = CartographService(db)  # heals, stamps, admits
    assert service.store.get_meta("schema_version") is not None
    service.store.close()
