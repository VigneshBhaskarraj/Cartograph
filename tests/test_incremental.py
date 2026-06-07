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


def test_update_indexes_when_db_absent(tmp_path):
    repo = _repo(tmp_path, {"a.py": "def f():\n    return 1\n"})
    db = tmp_path / "g.kuzu"
    summary = update_index(repo, db, dim=32)
    assert summary["status"] == "indexed"
    assert db.exists()
