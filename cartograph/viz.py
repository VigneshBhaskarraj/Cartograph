"""Interactive 3D graph visualization — a viewer over the graph, never a retrieval
path (SPEC §6 amendment: visualization is sanctioned post-MVP as a *demo/trust
artifact*; the engine never depends on it).

Design: zero new dependencies. The expensive force-directed layout runs HERE, in
numpy, at export time (Fruchterman–Reingold in 3D, vectorized); the browser gets a
single self-contained HTML file with the precomputed coordinates plus a small
vanilla-JS projection/interaction engine (rotate/zoom/pick, neighborhood focus,
shortest paths, filters, 2D/3D toggle). No CDN, no vendored libraries, no network
calls from the page — the zero-egress promise extends to the visualization.

Usage:  cartograph viz --db cartograph-out/graph.kuzu --out cartograph-out/graph.html
"""

from __future__ import annotations

import json
from importlib import resources
from pathlib import Path

import numpy as np

from .store import Store


def export_graph_data(store: Store) -> dict:
    """Nodes + typed edges in the compact form the HTML app consumes.
    Links reference node indices (not ids) to keep the embedded JSON small."""
    nodes = store.all_nodes_full()
    index = {n["id"]: i for i, n in enumerate(nodes)}
    links = []
    for src, dst, etype, confidence in store.all_edges_typed():
        si, di = index.get(src), index.get(dst)
        if si is None or di is None or si == di:
            continue
        links.append({"s": si, "t": di, "y": etype, "c": confidence})
    out_nodes = []
    degree = [0] * len(nodes)
    for link in links:
        degree[link["s"]] += 1
        degree[link["t"]] += 1
    for i, n in enumerate(nodes):
        out_nodes.append({
            "id": n["id"],
            "kind": n["kind"],
            "name": n["name"],
            "qn": n["qualified_name"],
            "file": n["file_path"],
            "line": n["start_line"],
            "sig": (n["signature"] or "")[:160],
            "doc": (n["docstring"] or "")[:280],
            "deg": degree[i],
        })
    return {"nodes": out_nodes, "links": links}


def layout_3d(data: dict, iterations: int = 200, seed: int = 42) -> None:
    """Vectorized 3D Fruchterman–Reingold; writes x/y/z onto each node (the 2D
    view simply projects x/y). O(N²) per iteration in numpy — ~2k nodes lays
    out in a few seconds, which is the intended scale."""
    n = len(data["nodes"])
    if n == 0:
        return
    rng = np.random.default_rng(seed)
    pos = rng.normal(size=(n, 3)).astype(np.float64)
    pos *= 10.0 / max(1.0, np.linalg.norm(pos, axis=1).max())
    if data["links"]:
        src = np.array([li["s"] for li in data["links"]])
        dst = np.array([li["t"] for li in data["links"]])
    else:
        src = dst = np.zeros(0, dtype=int)
    volume = max(n, 2) ** (1 / 3) * 10.0
    k = 10.0  # ideal pairwise distance (constant by construction; volume drives t)
    t = volume * 0.1  # temperature, cooled linearly
    for it in range(iterations):
        diff = pos[:, None, :] - pos[None, :, :]            # (n, n, 3)
        dist = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dist, np.inf)
        dist = np.maximum(dist, 1e-9)  # coincident nodes must not NaN the layout
        # Repulsion k²/d between every pair.
        rep = (k * k) / dist
        disp = (diff / dist[..., None] * rep[..., None]).sum(axis=1)
        # Attraction d²/k along edges.
        if len(src):
            evec = pos[src] - pos[dst]
            edist = np.linalg.norm(evec, axis=1, keepdims=True)
            edist[edist == 0] = 1e-9
            force = evec / edist * (edist ** 2 / k)
            np.add.at(disp, src, -force)
            np.add.at(disp, dst, force)
        # Mild gravity keeps disconnected components from drifting away.
        disp -= pos * 0.02
        length = np.linalg.norm(disp, axis=1, keepdims=True)
        length[length == 0] = 1e-9
        step = t * (1.0 - it / iterations) + 0.01
        pos += disp / length * np.minimum(length, step)
    pos -= pos.mean(axis=0)
    scale = 100.0 / max(1.0, np.abs(pos).max())
    pos *= scale
    for i, node in enumerate(data["nodes"]):
        node["x"], node["y"], node["z"] = (round(float(v), 2) for v in pos[i])


def build_html(data: dict, title: str = "Cartograph") -> str:
    template = resources.files("cartograph").joinpath(
        "viz_assets/template.html").read_text(encoding="utf-8")
    payload = json.dumps(data, separators=(",", ":"), allow_nan=False)
    # Embedding inside <script>: per the HTML tokenizer, a literal `</script>` ends
    # the block, and `<!--` + `<script` (even split across different docstrings)
    # enters double-escaped state and swallows the real close tag. Escaping every
    # `<` as < (valid JSON) neutralizes all of it — the json_script approach.
    payload = payload.replace("<", "\\u003c")
    return (template
            .replace("__CARTOGRAPH_TITLE__", title)
            .replace("__CARTOGRAPH_DATA__", payload))


def write_viz(db_path: str | Path, out_path: str | Path, title: str | None = None,
              iterations: int = 200) -> dict:
    """Export -> layout -> single self-contained HTML. Returns summary counts."""
    from .service import open_graph

    store = open_graph(db_path)
    try:
        data = export_graph_data(store)
    finally:
        store.close()
    layout_3d(data, iterations=iterations)
    html = build_html(data, title=title or Path(db_path).stem)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    return {"nodes": len(data["nodes"]), "links": len(data["links"]),
            "bytes": len(html), "out": str(out)}
