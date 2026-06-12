"""Multi-corpus scorecard: index + eval every corpus, emit one combined table.

The Gate-1 quality dashboard — run all eval corpora through all retrievers and print
a single matrix so movement (and regressions) are visible at a glance. Corpora whose
source isn't present are skipped (fetch with eval/get_corpus.sh / get_aidigest.sh or
clone). Offline by default (`hash`); pass --embedder ollama for real numbers.

Usage: uv run python eval/scorecard.py [--embedder hash|ollama] [--reindex]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))

from cartograph.embed import get_embedder  # noqa: E402
from cartograph.pipeline import index_path  # noqa: E402
from run_eval import run as run_eval  # noqa: E402

# name, source dir, db path, questions file (None = default httpx set)
CORPORA = [
    ("httpx", ".corpus/httpx", "cartograph-out/httpx.kuzu", None),
    ("flask", ".corpus/flask/src/flask", "cartograph-out/flask.kuzu", "eval/flask_questions.yaml"),
    ("bridge", "eval/bridge_corpus", "cartograph-out/bridge.kuzu", "eval/bridge_questions.yaml"),
    ("ai-digest", ".corpus/ai-digest/src", "cartograph-out/aidigest.kuzu", "eval/aidigest_questions.yaml"),
    # Held out from fusion calibration (the 2026-06-11 sweep used only the four
    # above) — click is the generalization check. Fetch: bash eval/get_click.sh
    ("click", ".corpus/click/src/click", "cartograph-out/click.kuzu", "eval/click_questions.yaml"),
    # Java corpus: canonical JPA entities + multi-dialect schema.sql (code<->data
    # bridge under eval). Fetch: bash eval/get_petclinic.sh
    ("petclinic", ".corpus/petclinic/src/main", "cartograph-out/petclinic.kuzu", "eval/petclinic_questions.yaml"),
]
RETRIEVERS = ["vector", "lexical", "graph", "hybrid"]
BASELINES = ["grep", "naive-rag"]  # external competitors (eval/baselines.py)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--embedder", default="hash")
    ap.add_argument("--reindex", action="store_true")
    ap.add_argument("--baselines", action="store_true",
                    help="also score the external baselines (grep, naive-rag)")
    args = ap.parse_args()

    rows = []
    for name, src, db, questions in CORPORA:
        if not (ROOT / src).exists():
            print(f"skip {name}: source {src} not present")
            continue
        if questions and not (ROOT / questions).exists():
            print(f"skip {name}: questions {questions} not present")
            continue
        if args.reindex or not (ROOT / db).exists():
            t0 = time.time()
            index_path(ROOT / src, ROOT / db, embedder=get_embedder(args.embedder), overwrite=True).close()
            idx_s = time.time() - t0
        else:
            idx_s = None
        for r in RETRIEVERS:
            row = run_eval(str(ROOT / db), r, args.embedder, None,
                           questions_path=Path(ROOT / questions) if questions else None)
            rows.append((name, r, row, idx_s if r == "vector" else None))
        if args.baselines:
            from baselines import run_baseline
            for b in BASELINES:
                row = run_baseline(b, ROOT / src, str(ROOT / db),
                                   Path(ROOT / questions) if questions else None, args.embedder)
                rows.append((name, row["retriever"], row, None))

    print("\n=== SCORECARD ===")
    hdr = f"{'corpus':<10} {'retriever':<20} {'recall@5':>8} {'recall@10':>9} {'mrr':>6} {'prec@5':>7} {'index_s':>8}"
    print(hdr); print("-" * len(hdr))
    for name, r, row, idx_s in rows:
        print(f"{name:<10} {r:<20} {row['recall@5']:>8} {row['recall@10']:>9} {row['mrr']:>6} "
              f"{row['precision@5']:>7} {('' if idx_s is None else f'{idx_s:.1f}'):>8}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
