"""CLI failure modes: a stranger's first five minutes must not end in a traceback."""

from pathlib import Path

from typer.testing import CliRunner

from cartograph.cli import app

FIX = Path(__file__).parent / "fixtures" / "sample.py"
runner = CliRunner()


def test_index_query_stats_happy_path(tmp_path):
    db = str(tmp_path / "g.kuzu")
    r = runner.invoke(app, ["index", str(FIX), "--db", db])
    assert r.exit_code == 0 and "Indexed" in r.output
    r = runner.invoke(app, ["query", "bark", "--db", db, "--mode", "lexical"])
    assert r.exit_code == 0 and "bark" in r.output
    r = runner.invoke(app, ["stats", "--db", db])
    assert r.exit_code == 0 and "node:" in r.output


def test_query_missing_db_errors_without_creating_it(tmp_path):
    db = tmp_path / "missing.kuzu"
    r = runner.invoke(app, ["query", "anything", "--db", str(db)])
    assert r.exit_code == 1
    assert "no graph at" in r.output
    assert not db.exists()  # must not leave an empty DB at the mistyped path


def test_stats_missing_db_errors(tmp_path):
    r = runner.invoke(app, ["stats", "--db", str(tmp_path / "nope.kuzu")])
    assert r.exit_code == 1
    assert "no graph at" in r.output


def test_query_unknown_mode_is_rejected(tmp_path):
    r = runner.invoke(app, ["query", "x", "--db", str(tmp_path / "g.kuzu"), "--mode", "bogus"])
    assert r.exit_code == 2
    assert "unknown mode" in r.output


def test_index_refuses_to_overwrite_non_kuzu_path(tmp_path):
    target = tmp_path / "documents"
    target.mkdir()
    keep = target / "precious.txt"
    keep.write_text("do not delete")
    r = runner.invoke(app, ["index", str(FIX), "--db", str(target)])
    assert r.exit_code == 1
    assert "refusing to overwrite" in r.output
    assert keep.read_text() == "do not delete"


def test_index_empty_dir_is_friendly(tmp_path):
    src = tmp_path / "empty"
    src.mkdir()
    r = runner.invoke(app, ["index", str(src), "--db", str(tmp_path / "g.kuzu")])
    assert r.exit_code == 1
    assert "no supported source files" in r.output


def test_index_skips_venv_and_vendored_dirs(tmp_path):
    """Pointing index at a project root must not embed the virtualenv."""
    from cartograph.pipeline import _files

    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "real.py").write_text("def f():\n    pass\n")
    for junk in (".venv/lib", "node_modules/m", ".git/hooks"):
        d = tmp_path / junk
        d.mkdir(parents=True)
        (d / "junk.py").write_text("x = 1\n")
    found = _files(tmp_path, ".py")
    assert [f.name for f in found] == ["real.py"]


def test_update_missing_source_path_is_friendly(tmp_path):
    db = str(tmp_path / "g.kuzu")
    r = runner.invoke(app, ["index", str(FIX), "--db", db])
    assert r.exit_code == 0
    r = runner.invoke(app, ["update", str(tmp_path / "typo"), "--db", db])
    assert r.exit_code == 1
    assert "does not exist" in r.output


def test_query_bad_embedder_is_friendly(tmp_path):
    db = str(tmp_path / "g.kuzu")
    runner.invoke(app, ["index", str(FIX), "--db", db])
    r = runner.invoke(app, ["query", "bark", "--db", db, "--embedder", "bogus"])
    assert r.exit_code == 1
    assert "unknown embedder" in r.output


def test_serve_missing_db_stays_up_and_reports_through_mcp(tmp_path):
    """serve must NOT die on a bad DB (the client would see only "Connection
    closed") — it warns on stderr, completes the handshake, and returns the
    friendly error from tool calls. Real subprocess: a stdio server can't be
    exercised honestly under CliRunner."""
    import json
    import subprocess
    import sys

    import pytest

    pytest.importorskip("mcp")
    db = str(tmp_path / "missing.kuzu")
    frames = "\n".join(json.dumps(m) for m in [
        {"jsonrpc": "2.0", "id": 1, "method": "initialize",
         "params": {"protocolVersion": "2024-11-05", "capabilities": {},
                    "clientInfo": {"name": "test", "version": "0"}}},
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/call",
         "params": {"name": "query", "arguments": {"text": "x"}}},
    ]) + "\n"
    proc = subprocess.run(
        [sys.executable, "-c",
         f"import sys; from cartograph.cli import app; sys.argv = ['cartograph', 'serve', '--db', {db!r}]; app()"],
        # 60s was too tight under full-suite load (cold interpreter + mcp import +
        # stdio handshake on a busy CI box); generous so the test is deterministic.
        input=frames, capture_output=True, text=True, timeout=180)
    assert "no graph at" in proc.stderr  # preflight warning for the human
    replies = {m.get("id"): m for m in map(json.loads, proc.stdout.splitlines())}
    assert "serverInfo" in replies[1]["result"]  # handshake survived the bad DB
    tool_result = replies[2]["result"]
    assert tool_result["isError"] is True
    assert "no graph at" in tool_result["content"][0]["text"]  # actionable error reached the agent


def test_index_venv_skipped_only_when_real_venv(tmp_path):
    """A source package named `env` is indexed; an actual virtualenv named `env`
    (has pyvenv.cfg) is skipped."""
    from cartograph.pipeline import _files

    pkg = tmp_path / "env"
    pkg.mkdir()
    (pkg / "real.py").write_text("def f():\n    pass\n")
    venv = tmp_path / "sub" / "env"
    venv.mkdir(parents=True)
    (venv / "pyvenv.cfg").write_text("home = /usr\n")
    (venv / "junk.py").write_text("x = 1\n")
    found = [f.name for f in _files(tmp_path, ".py")]
    assert found == ["real.py"]


def test_structural_cli_commands(tmp_path):
    """node/resolve/calls/callers/path expose the MCP surface over the shell."""
    db = str(tmp_path / "g.kuzu")
    r = runner.invoke(app, ["index", str(FIX), "--db", db])
    assert r.exit_code == 0
    r = runner.invoke(app, ["node", "Dog", "--db", db])
    assert r.exit_code == 0 and "qualified_name" in r.output
    r = runner.invoke(app, ["resolve", "speak", "--db", db])
    assert r.exit_code == 0 and "speak" in r.output
    r = runner.invoke(app, ["calls", "Dog.speak", "--db", db])
    assert r.exit_code == 0 and "bark" in r.output
    r = runner.invoke(app, ["callers", "bark", "--db", db])
    assert r.exit_code == 0 and "speak" in r.output
    r = runner.invoke(app, ["path", "Dog", "bark", "--db", db])
    assert r.exit_code == 0 and "bark" in r.output
    r = runner.invoke(app, ["node", "no_such_symbol_xyz", "--db", db])
    assert r.exit_code == 1
    r = runner.invoke(app, ["calls", "x", "--db", str(tmp_path / "missing.kuzu")])
    assert r.exit_code == 1 and "no graph at" in r.output


def test_missing_language_extra_warns_loudly(tmp_path, monkeypatch):
    """A Java repo indexed without tree-sitter-java must WARN, not silently produce
    a half-empty graph (a real user hit this: petclinic scored 0 on code questions)."""
    import sys

    import pytest as _pytest

    from cartograph.pipeline import build_graph

    (tmp_path / "App.java").write_text("package x;\npublic class App {}\n")
    (tmp_path / "ok.py").write_text("def f():\n    return 1\n")
    monkeypatch.setitem(sys.modules, "cartograph.java_extract", None)  # import -> ImportError
    with _pytest.warns(UserWarning, match="Java files.*NOT indexed.*--extra java"):
        g = build_graph(tmp_path)
    assert any(n.name == "f" for n in g.nodes)  # the python half still indexed


def test_demo_missing_file_is_friendly(tmp_path):
    r = runner.invoke(app, ["demo", str(tmp_path / "nope.py")])
    assert r.exit_code == 1
    assert "Traceback" not in r.output


def test_query_rerank_reachable_from_cli(tmp_path, monkeypatch):
    """Regression (G5-A4): the CLI used to bypass the service, making rerank
    mode unreachable even with CARTOGRAPH_RERANKER set."""
    monkeypatch.setenv("CARTOGRAPH_RERANKER", "lexical")
    db = str(tmp_path / "g.kuzu")
    runner.invoke(app, ["index", str(FIX), "--db", db])
    r = runner.invoke(app, ["query", "bark", "--db", db, "--mode", "rerank"])
    assert r.exit_code == 0
    assert "[" in r.output  # ranked rows printed
