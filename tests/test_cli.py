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
    assert "no Python or SQL files" in r.output


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
