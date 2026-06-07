"""M1-5: confirm every expected anchor resolves to >=1 node in the indexed graph.

Fails loudly (exit 1) on any unresolved anchor — the eval doc's "confirm on index"
guard against silently scoring against symbols that shifted between releases.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from evallib import gold_ids, load_questions, open_store

DEFAULT_DB = "cartograph-out/httpx.kuzu"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=DEFAULT_DB)
    ap.add_argument("--check", action="store_true", help="exit 1 if any anchor is unresolved")
    ap.add_argument("--out", default="eval/anchors.resolved.json")
    ap.add_argument("--questions", default=None, help="path to a questions.yaml (default: the httpx set)")
    args = ap.parse_args()

    store, nodes = open_store(args.db)
    resolved: dict[str, dict] = {}
    unresolved: list[tuple[int, str]] = []
    questions = load_questions(Path(args.questions)) if args.questions else load_questions()
    for q in questions:
        per_anchor = {}
        for a in q["anchors"]:
            ids = sorted(gold_ids([a], nodes))
            per_anchor[a] = ids
            if not ids:
                unresolved.append((q["id"], a))
        resolved[str(q["id"])] = {"mode": q["mode"], "anchors": per_anchor}
    store.close()

    Path(args.out).write_text(json.dumps(resolved, indent=2))
    print(f"Resolved anchors for {len(resolved)} questions -> {args.out}")
    if unresolved:
        print(f"UNRESOLVED ({len(unresolved)}):")
        for qid, a in unresolved:
            print(f"  Q{qid}: {a}")
        if args.check:
            return 1
    else:
        print("All anchors resolved.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
