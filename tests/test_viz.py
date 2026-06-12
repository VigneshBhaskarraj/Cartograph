"""The visualizer is a viewer over the graph: export + layout + one offline file."""

import json
from pathlib import Path

from typer.testing import CliRunner

from cartograph.cli import app
from cartograph.pipeline import index_path
from cartograph.viz import build_html, export_graph_data, layout_3d

FIX = Path(__file__).parent / "fixtures" / "sample.py"
runner = CliRunner()


def _data(tmp_path):
    store = index_path(FIX, tmp_path / "g.kuzu", dim=32, overwrite=True)
    data = export_graph_data(store)
    store.close()
    return data


def test_export_has_typed_links_and_confidence(tmp_path):
    data = _data(tmp_path)
    kinds = {n["kind"] for n in data["nodes"]}
    assert {"module", "class", "function"} <= kinds
    types = {li["y"] for li in data["links"]}
    assert "CONTAINS" in types and "CALLS" in types
    confs = {li["c"] for li in data["links"]}
    assert confs <= {"EXTRACTED", "INFERRED"}
    # links are index-based and in range
    n = len(data["nodes"])
    assert all(0 <= li["s"] < n and 0 <= li["t"] < n for li in data["links"])


def test_layout_writes_finite_coordinates(tmp_path):
    data = _data(tmp_path)
    layout_3d(data, iterations=30)
    for node in data["nodes"]:
        for axis in ("x", "y", "z"):
            assert axis in node and abs(node[axis]) <= 101


def test_html_is_self_contained(tmp_path):
    data = _data(tmp_path)
    layout_3d(data, iterations=10)
    html = build_html(data, title="t")
    assert "__CARTOGRAPH_DATA__" not in html and "__CARTOGRAPH_TITLE__" not in html
    # zero egress: a CSP forbids ALL loads/connects; no meta-refresh; no external
    # resource references (the GitHub href is navigation-only)
    assert "Content-Security-Policy" in html and "default-src 'none'" in html
    assert 'http-equiv="refresh"' not in html
    assert 'src="http' not in html and 'href="http' not in html.replace(
        'href="https://github.com/VigneshBhaskarraj/Cartograph"', "")
    raw = html.split('<script id="data" type="application/json">')[1].split("</script>")[0]
    parsed = json.loads(raw)  # < escaping is valid JSON — no un-escaping needed
    assert len(parsed["nodes"]) == len(data["nodes"])


def test_hostile_docstrings_cannot_break_the_script_block(tmp_path):
    """Review finding: `</script>` ends the data block and `<!--` + `<script` (even
    in different docstrings) swallows it. Every `<` must be \\u003c-escaped, and the
    payload must round-trip as plain JSON."""
    src = tmp_path / "hostile.py"
    src.write_text(
        '"""Module doc with </script><script>alert(1)</script>."""\n\n'
        'def f():\n'
        '    """Has <!--<script in it, and a lone </ too."""\n'
        '    return 1\n'
    )
    store = index_path(src, tmp_path / "g.kuzu", dim=16, overwrite=True)
    data = export_graph_data(store)
    store.close()
    layout_3d(data, iterations=5)
    html = build_html(data, title="hostile")
    raw = html.split('<script id="data" type="application/json">')[1].split("</script>")[0]
    assert "<" not in raw  # every angle bracket in the data is escaped
    parsed = json.loads(raw)
    doc = next(n["doc"] for n in parsed["nodes"] if n["name"] == "f")
    assert "<!--<script" in doc  # content survives intact after JSON.parse


def test_wheel_packages_the_template():
    from importlib import resources

    t = resources.files("cartograph").joinpath("viz_assets/template.html").read_text(encoding="utf-8")
    assert "__CARTOGRAPH_DATA__" in t


def test_viz_cli_writes_file(tmp_path):
    db = str(tmp_path / "g.kuzu")
    out = str(tmp_path / "graph.html")
    assert runner.invoke(app, ["index", str(FIX), "--db", db]).exit_code == 0
    r = runner.invoke(app, ["viz", "--db", db, "--out", out, "--iterations", "10"])
    assert r.exit_code == 0 and "Wrote" in r.output
    assert Path(out).stat().st_size > 10_000


def test_viz_missing_db_is_friendly(tmp_path):
    r = runner.invoke(app, ["viz", "--db", str(tmp_path / "missing.kuzu"),
                            "--out", str(tmp_path / "x.html")])
    assert r.exit_code == 1 and "no graph at" in r.output
