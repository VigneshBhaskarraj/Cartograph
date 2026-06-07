"""Call-edge precision check — measures CALLS-edge *correctness*, which the
anchor-recall eval doesn't capture.

Gold callee sets are ground-truthed against the pinned httpx source (read the method
body, list what it actually calls). This is what catches name/dispatch imprecision
(phantom sync/async edges, wrong-owner `.read`/`.close`) and what M3 (real symbol
resolution) must drive up. Seed set — grow it as needed.

Usage: uv run python eval/call_precision.py --db cartograph-out/httpx.kuzu
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cartograph.store import Store  # noqa: E402

# caller qualified_name -> set of correct callee qualified_names (confirmed in source).
GOLD: dict[str, set[str]] = {
    "httpx._client.Client.send": {
        "httpx._client.BaseClient._set_timeout",
        "httpx._client.BaseClient._build_request_auth",
        "httpx._client.Client._send_handling_auth",
        "httpx._models.Response.read",
        "httpx._models.Response.close",
    },
    "httpx._client.AsyncClient.send": {
        "httpx._client.BaseClient._set_timeout",
        "httpx._client.BaseClient._build_request_auth",
        "httpx._client.AsyncClient._send_handling_auth",
        "httpx._models.Response.aread",
        "httpx._models.Response.aclose",
    },
}


def callees(store: Store, qn: str) -> set[str]:
    res = store.conn.execute(
        "MATCH (a:CodeNode {qualified_name:$q})-[:CALLS]->(b:CodeNode) RETURN DISTINCT b.qualified_name",
        {"q": qn},
    )
    out: set[str] = set()
    while res.has_next():
        out.add(res.get_next()[0])
    return out


def main() -> int:
    db = sys.argv[sys.argv.index("--db") + 1] if "--db" in sys.argv else "cartograph-out/httpx.kuzu"
    store = Store(db)
    precs, recs = [], []
    for qn, gold in GOLD.items():
        got = callees(store, qn)
        tp = len(got & gold)
        prec = tp / len(got) if got else 0.0
        rec = tp / len(gold) if gold else 0.0
        precs.append(prec)
        recs.append(rec)
        wrong = sorted(got - gold)
        print(f"{qn}\n  precision={prec:.2f} recall={rec:.2f}  (got {len(got)}, correct {tp}, gold {len(gold)})")
        if wrong:
            print(f"  false edges: {', '.join(w.split('.')[-2] + '.' + w.split('.')[-1] for w in wrong)}")
    store.close()
    mp = sum(precs) / len(precs)
    mr = sum(recs) / len(recs)
    print(f"\nMEAN call-edge precision={mp:.3f}  recall={mr:.3f}  (n={len(GOLD)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
