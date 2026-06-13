from pathlib import Path

from cartograph.pipeline import diff_files, index_path, update_index
from cartograph.store import Store


def _repo(tmp_path, files: dict[str, str]) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir(exist_ok=True)
    for name, txt in files.items():
        (repo / name).write_text(txt)
    return repo


def test_update_noop_when_unchanged(tmp_path):
    repo = _repo(tmp_path, {"a.py": "def f():\n    return 1\n"})
    db = tmp_path / "g.kuzu"
    index_path(repo, db, dim=32, overwrite=True).close()
    summary = update_index(repo, db, dim=32)
    assert summary["status"] == "up-to-date"
    assert summary["embedded"] == 0  # nothing re-embedded


def test_update_reflects_change_and_removes_stale(tmp_path):
    repo = _repo(tmp_path, {"a.py": "def f():\n    return 1\n\ndef g():\n    return 2\n"})
    db = tmp_path / "g.kuzu"
    index_path(repo, db, dim=32, overwrite=True).close()
    # g removed, h added
    (repo / "a.py").write_text("def f():\n    return 1\n\ndef h():\n    return 3\n")
    summary = update_index(repo, db, dim=32)
    assert summary["status"] == "updated" and "repo/a.py" in summary["changed"]
    store = Store(db)
    names = {d["name"] for d in store.all_nodes_text()}
    store.close()
    assert "h" in names and "g" not in names  # stale symbol (and its edges) gone


def test_update_detects_deleted_file(tmp_path):
    repo = _repo(tmp_path, {"a.py": "def f():\n    return 1\n", "b.py": "def g():\n    return 2\n"})
    db = tmp_path / "g.kuzu"
    index_path(repo, db, dim=32, overwrite=True).close()
    (repo / "b.py").unlink()
    delta = diff_files(repo, db)
    assert "repo/b.py" in delta["deleted"]
    update_index(repo, db, dim=32)
    store = Store(db)
    names = {d["name"] for d in store.all_nodes_text()}
    store.close()
    assert "g" not in names


def test_delta_recreates_only_changed_nodes(tmp_path):
    """Row-level delta: changing one file recreates only that file's nodes; the
    other file's nodes are preserved (not rewritten)."""
    repo = _repo(tmp_path, {"a.py": "def af():\n    return 1\n", "b.py": "def bf():\n    return 2\n"})
    db = tmp_path / "g.kuzu"
    index_path(repo, db, dim=32, overwrite=True).close()
    (repo / "b.py").write_text("def bf():\n    return 99\n")  # change only b.py
    summary = update_index(repo, db, dim=32)
    assert summary["status"] == "updated"
    # only b.py (module + bf) churns — not the whole graph
    assert summary["created"] <= 3 and summary["removed"] <= 3
    store = Store(db)
    names = {d["name"] for d in store.all_nodes_text()}
    store.close()
    assert {"af", "bf"} <= names  # a.py preserved, b.py updated


def test_update_indexes_when_db_absent(tmp_path):
    repo = _repo(tmp_path, {"a.py": "def f():\n    return 1\n"})
    db = tmp_path / "g.kuzu"
    summary = update_index(repo, db, dim=32)
    assert summary["status"] == "indexed"
    assert db.exists()


class _StubEmbedder:
    """Same dim as the index but a different identity — must force a rebuild."""

    name = "stub"
    dim = 32
    dim_is_exact = True

    def embed(self, text):
        return [1.0] + [0.0] * 31

    def embed_batch(self, texts):
        return [self.embed(t) for t in texts]


def test_update_with_different_embedder_rebuilds(tmp_path):
    """Audit H1: a same-dim embedder switch must not keep old vectors on unchanged
    rows (silent vector-space mixing); it rebuilds and records the new identity."""
    repo = _repo(tmp_path, {"a.py": "def f():\n    return 1\n"})
    db = tmp_path / "g.kuzu"
    index_path(repo, db, dim=32, overwrite=True).close()
    (repo / "b.py").write_text("def g():\n    return 2\n")
    summary = update_index(repo, db, dim=32, embedder=_StubEmbedder())
    assert summary["status"] == "rebuilt"
    store = Store(db)
    assert store.get_meta("embedder_backend") == "stub"
    store.close()


def test_plain_update_keeps_index_embedder(tmp_path, monkeypatch):
    """A plain `update` (no --embedder) adopts the embedder recorded at index time —
    even when the environment default says otherwise — so it never rebuilds or
    mixes vector spaces by accident."""
    repo = _repo(tmp_path, {"a.py": "def f():\n    return 1\n"})
    db = tmp_path / "g.kuzu"
    index_path(repo, db, dim=32, overwrite=True).close()  # records backend=hash
    (repo / "b.py").write_text("def g():\n    return 2\n")
    monkeypatch.setenv("CARTOGRAPH_EMBEDDER", "ollama")  # env must NOT win over meta
    summary = update_index(repo, db, dim=32)  # no embedder given
    assert summary["status"] == "updated"  # delta path, no rebuild, no ollama call
    store = Store(db)
    assert store.get_meta("embedder_backend") == "hash"
    store.close()


def test_update_after_all_files_deleted_empties_graph(tmp_path):
    """Audit M4: deleting every source file is a valid delta — empty the graph,
    don't crash and keep serving stale answers."""
    repo = _repo(tmp_path, {"a.py": "def f():\n    return 1\n"})
    db = tmp_path / "g.kuzu"
    index_path(repo, db, dim=32, overwrite=True).close()
    (repo / "a.py").unlink()
    summary = update_index(repo, db, dim=32)
    assert summary["status"] == "updated"
    assert summary["removed"] > 0
    store = Store(db)
    assert store.node_shas() == {}
    assert not any(k.startswith("file:") for k in store.all_meta())
    store.close()


def test_update_with_nonexistent_source_path_does_not_wipe(tmp_path):
    """Review BLOCKER: a typo'd source path must error, not masquerade as
    'all files deleted' and silently empty the graph."""
    import pytest

    repo = _repo(tmp_path, {"a.py": "def f():\n    return 1\n"})
    db = tmp_path / "g.kuzu"
    index_path(repo, db, dim=32, overwrite=True).close()
    with pytest.raises(FileNotFoundError, match="does not exist"):
        update_index(tmp_path / "typo", db, dim=32)
    store = Store(db)
    assert len(store.node_shas()) > 0  # graph untouched
    store.close()


def test_interrupted_update_is_refused_then_repaired(tmp_path, monkeypatch):
    """G5-B1: Kuzu auto-commits each statement, so a crash mid-delta used to
    leave a nodes-but-no-edges graph that passed every check. The dirty flag
    makes readers refuse it and `update` repair it with a full rebuild."""
    import pytest

    from cartograph.service import open_graph

    repo = _repo(tmp_path, {"a.py": "def f():\n    return 1\n\ndef g():\n    return f()\n"})
    db = tmp_path / "g.kuzu"
    index_path(repo, db, dim=32, overwrite=True).close()
    (repo / "a.py").write_text("def f():\n    return 2\n\ndef g():\n    return f()\n")

    # Crash after edges were wiped, before nodes were reloaded.
    def boom(self, ids):
        raise RuntimeError("simulated crash")

    with monkeypatch.context() as m:
        m.setattr(Store, "delete_nodes", boom)
        with pytest.raises(RuntimeError, match="simulated crash"):
            update_index(repo, db, dim=32)

    # Readers refuse the half-written graph instead of serving garbage...
    with pytest.raises(RuntimeError, match="interrupted"):
        open_graph(db)
    # ...even though nothing changed on disk since the crash (up-to-date must not mask dirty).
    summary = update_index(repo, db, dim=32)
    assert summary["status"] == "rebuilt"
    store = open_graph(db)  # opens cleanly again
    counts = store.counts()
    assert counts["edge:CONTAINS"] >= 1 and counts["node:function"] == 2
    store.close()


def test_mid_run_edit_is_not_permanently_stale(tmp_path, monkeypatch):
    """G5-B3 TOCTOU: digests used to be hashed AFTER parsing, so a file edited
    mid-run recorded the new sha against the old parse — 'up-to-date' forever.
    Hashed before, the next update detects the edit and re-indexes."""
    import cartograph.pipeline as pl

    repo = _repo(tmp_path, {"a.py": "def f():\n    return 1\n"})
    db = tmp_path / "g.kuzu"
    index_path(repo, db, dim=32, overwrite=True).close()
    (repo / "a.py").write_text("def f():\n    return 2\n")

    real_build = pl.build_graph

    def edits_after_parse(path, resolver="heuristic"):
        g = real_build(path, resolver=resolver)
        (repo / "a.py").write_text("def f():\n    return 3\n")  # the mid-run edit
        return g

    with monkeypatch.context() as m:
        m.setattr(pl, "build_graph", edits_after_parse)
        update_index(repo, db, dim=32)

    summary = update_index(repo, db, dim=32)
    assert summary["status"] != "up-to-date"  # the edit was detected, not masked


def test_moved_sql_table_keeps_correct_position(tmp_path):
    """G5-B3: SQL node ids carry no line, so a moved CREATE TABLE was 'kept'
    with its stale start_line on delta update. Position is in the sha now."""
    repo = _repo(tmp_path, {"schema.sql": "CREATE TABLE users (id INT);\n"})
    db = tmp_path / "g.kuzu"
    index_path(repo, db, dim=32, overwrite=True).close()
    (repo / "schema.sql").write_text("-- a comment banner\n-- pushing things down\n\nCREATE TABLE users (id INT);\n")
    update_index(repo, db, dim=32)
    store = Store(db)
    res = store.conn.execute(
        "MATCH (c:CodeNode) WHERE c.kind = 'table' AND c.name = 'users' RETURN c.start_line")
    assert res.get_next()[0] == 4  # the new position, not the stale line 1
    store.close()


def test_dirty_rebuild_honors_no_cache(tmp_path):
    """Review follow-up: the dirty-graph rebuild used to drop use_cache=False —
    `cartograph update --no-cache` on a dirty graph silently reused the cache."""
    repo = _repo(tmp_path, {"a.py": "def f():\n    return 1\n"})
    db = tmp_path / "g.kuzu"
    index_path(repo, db, dim=32, overwrite=True).close()  # populates the cache
    s = Store(db)
    s.set_meta("dirty", "1")
    s.close()
    summary = update_index(repo, db, dim=32, use_cache=False)
    assert summary["status"] == "rebuilt"
    assert summary["reused"] == 0  # nothing came from the cache
