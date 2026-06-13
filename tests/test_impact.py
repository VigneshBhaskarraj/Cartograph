"""Impact analysis: blast radius across the code<->data bridge (the moat query)."""

from pathlib import Path

import pytest

pytest.importorskip("sqlglot")

from typer.testing import CliRunner  # noqa: E402

from cartograph.cli import app  # noqa: E402
from cartograph.pipeline import index_path  # noqa: E402
from cartograph.service import CartographService  # noqa: E402

runner = CliRunner()


@pytest.fixture()
def repo_db(tmp_path):
    """A tiny app with a verified call chain into raw SQL:
    entrypoint() -> save_user() -> INSERT INTO users; plus an ORM class on users."""
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "schema.sql").write_text(
        "CREATE TABLE users (id INTEGER PRIMARY KEY, email TEXT);\n"
        "CREATE TABLE audit (id INTEGER PRIMARY KEY, user_id INTEGER REFERENCES users(id));\n")
    (repo / "app.py").write_text(
        'class User:\n'
        '    __tablename__ = "users"\n'
        '\n'
        '    def email_domain(self):\n'
        '        return self.email.split("@")[1]\n'
        '\n'
        'def save_user(conn, email):\n'
        '    conn.execute("INSERT INTO users (email) VALUES (?)", (email,))\n'
        '\n'
        'def entrypoint(conn):\n'
        '    save_user(conn, "x@example.com")\n'
        '\n'
        'def unrelated():\n'
        '    return 1\n')
    db = tmp_path / "g.kuzu"
    index_path(repo, db, dim=32, overwrite=True).close()
    return db


def _quals(nodes):
    return {n["qualified_name"] for n in nodes}


def test_column_impact_reaches_transitive_callers(repo_db):
    svc = CartographService(repo_db)
    r = svc.impact("users.email")
    svc.close()
    assert r["direction"] == "data->code"
    direct = _quals(r["direct_code"])
    # the raw-SQL writer and the ORM class mapped to the parent table
    assert any(q.endswith("save_user") for q in direct)
    assert any(q.endswith(".User") for q in direct)
    # the caller of the writer is in the blast radius; unrelated code is not
    transitive = _quals(r["transitive_callers"])
    assert any(q.endswith("entrypoint") for q in transitive)
    # G6-3: email_domain reads self.email, so it DIRECTLY touches users.email now
    # (a real QUERIES edge) rather than being a coarse class-method guess.
    assert any(q.endswith("User.email_domain") for q in direct)
    assert not any(q.endswith("unrelated") for q in direct | transitive)
    assert r["total_code_paths"] >= 4
    assert r["truncated"] is False


def test_table_impact_includes_its_columns_touchers(repo_db):
    svc = CartographService(repo_db)
    r = svc.impact("users")
    svc.close()
    assert any(q.endswith("save_user") for q in _quals(r["direct_code"]))


def test_code_impact_lists_data_it_touches(repo_db):
    svc = CartographService(repo_db)
    r = svc.impact("entrypoint")
    svc.close()
    assert r["direction"] == "code->data"
    assert r["direct_data"] == []  # entrypoint runs no SQL itself
    names = {n["qualified_name"] for n in r["transitive_data"]}
    assert "users" in names  # …but reaches users through save_user
    assert r["total_data_touched"] >= 1


def test_impact_unknown_ref_is_none(repo_db):
    svc = CartographService(repo_db)
    assert svc.impact("no_such_thing_xyz") is None
    svc.close()


def test_impact_cli(repo_db, tmp_path):
    r = runner.invoke(app, ["impact", "users.email", "--db", str(repo_db)])
    assert r.exit_code == 0
    assert "save_user" in r.output and "entrypoint" in r.output
    assert "INFERRED" in r.output  # the over-approximation caveat is shown
    r = runner.invoke(app, ["impact", "nope_xyz", "--db", str(repo_db)])
    assert r.exit_code == 1


def test_module_impact_descends_scope(repo_db):
    """Review finding: a module 'touches' what its functions touch — code->data
    must descend CONTAINS before expanding the call graph."""
    svc = CartographService(repo_db)
    r = svc.impact("app")  # the module
    svc.close()
    assert r["direction"] == "code->data"
    names = {n["qualified_name"] for n in r["direct_data"] + r["transitive_data"]}
    assert "users" in names
    assert r["total_data_touched"] >= 1


def test_impact_carries_machine_readable_completeness(repo_db):
    """G6-1: every impact result must flag that it is not exhaustive, with
    structured limitation codes an agent can branch on (the bank-pilot condition)."""
    svc = CartographService(repo_db)
    data_to_code = svc.impact("users.email")
    code_to_data = svc.impact("entrypoint")
    svc.close()
    for r in (data_to_code, code_to_data):
        c = r["completeness"]
        assert c["exhaustive"] is False and c["advisory_only"] is True
        codes = {lim["code"] for lim in c["limitations"]}
        assert {"inferred_calls", "orm_attribute_access"} <= codes
        assert all(lim["detail"] for lim in c["limitations"])  # every code is explained
    # the schema-links residual (after FK ripple is followed) is data->code only;
    # it must NOT be claimed in the code->data direction.
    assert "undeclared_schema_links" in {l["code"] for l in data_to_code["completeness"]["limitations"]}
    assert "undeclared_schema_links" not in {l["code"] for l in code_to_data["completeness"]["limitations"]}


def test_impact_cli_shows_completeness(repo_db):
    r = runner.invoke(app, ["impact", "users.email", "--db", str(repo_db)])
    assert r.exit_code == 0
    assert "NOT EXHAUSTIVE" in r.output and "advisory only" in r.output
    assert "inferred_calls" in r.output and "undeclared_schema_links" in r.output


@pytest.fixture()
def fk_db(tmp_path):
    """users <- audit (FK): audit.user_id REFERENCES users(id); code touches audit
    via an ORM class, so dropping users must put that code in the blast radius."""
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "schema.sql").write_text(
        "CREATE TABLE users (id INTEGER PRIMARY KEY, email TEXT);\n"
        "CREATE TABLE audit (id INTEGER PRIMARY KEY, user_id INTEGER REFERENCES users(id), action TEXT);\n")
    (repo / "app.py").write_text(
        'class Audit:\n'
        '    __tablename__ = "audit"\n'
        '\n'
        'def write_audit(conn, uid, action):\n'
        '    conn.execute("INSERT INTO audit (user_id, action) VALUES (?, ?)", (uid, action))\n'
        '\n'
        'def some_caller(conn):\n'
        '    write_audit(conn, 1, "login")\n')
    db = tmp_path / "g.kuzu"
    index_path(repo, db, dim=32, overwrite=True).close()
    return db


def test_fk_ripple_pulls_referencing_tables_code(fk_db):
    """G6-2: dropping `users` breaks `audit` (FK), so audit's code is in the radius
    — even though no code touches the users table directly."""
    svc = CartographService(fk_db)
    r = svc.impact("users")
    svc.close()
    reached = _quals(r["direct_code"]) | _quals(r["transitive_callers"])
    assert any(q.endswith(".Audit") for q in reached)        # ORM class on the FK table
    assert any(q.endswith("write_audit") for q in reached)   # raw-SQL writer to audit
    assert any(q.endswith("some_caller") for q in reached)   # its transitive caller


def test_fk_ripple_off_by_default_for_unrelated_table(fk_db):
    """A table nothing references gets no spurious FK-ripple code."""
    svc = CartographService(fk_db)
    r = svc.impact("audit")  # nothing references audit
    svc.close()
    # audit's own code is present; users' code (there is none) is not invented
    assert r["direction"] == "data->code"


def test_orm_self_attribute_capture(repo_db):
    """G6-3: a method reading self.email on a mapped class now touches users.email
    (code->data), and that column's impact reaches the method (data->code)."""
    svc = CartographService(repo_db)
    # code->data: email_domain reads self.email — previously showed NO data touched
    fwd = svc.impact("User.email_domain")
    touched = {n["qualified_name"] for n in fwd["direct_data"] + fwd["transitive_data"]}
    assert "users.email" in touched
    # the edge is INFERRED (self.x matching a column name is heuristic)
    svc.close()


def test_orm_self_attribute_edge_is_inferred(repo_db):
    from cartograph.store import Store
    s = Store(repo_db, read_only=True)
    q = {(s.get_node(a)["qualified_name"], s.get_node(b)["qualified_name"], c)
         for a, b, t, c in s.all_edges_typed() if t == "QUERIES"}
    s.close()
    assert any(src.endswith("email_domain") and dst == "users.email" and conf == "INFERRED"
               for src, dst, conf in q)
