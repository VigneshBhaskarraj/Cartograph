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
