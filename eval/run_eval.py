"""M1-8: run a retriever over the question set, print metrics, append a CSV row.

Output format follows docs/eval-set-httpx.md:
  run, retriever, recall@5, recall@10, precision@5, mrr, <per-mode recall@10>
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import sys
from pathlib import Path

from evallib import (
    gold_ids,
    load_questions,
    modes_of,
    open_store,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cartograph.embed import get_embedder  # noqa: E402
from cartograph.rerank import get_reranker  # noqa: E402
from cartograph.retrieve import Retriever  # noqa: E402

DEFAULT_DB = "cartograph-out/httpx.kuzu"
MODE_COLS = ["STRUCT", "MULTIHOP", "SEMANTIC", "EXACT", "CROSS", "WHY"]
PRECISION_QIDS = {1, 2, 3, 4, 5}  # small, well-defined answer sets


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def run(db: str, retriever_name: str, embedder_name: str | None, out: str | None,
        reranker_name: str | None = None, rerank_model: str | None = None,
        questions_path: Path | None = None) -> dict:
    store, nodes = open_store(db)
    embedder = get_embedder(embedder_name) if embedder_name else None
    reranker = get_reranker(reranker_name, rerank_model) if retriever_name == "rerank" else None
    retriever = Retriever(store, embedder=embedder, reranker=reranker)
    label = retriever_name
    if retriever_name in ("vector", "hybrid"):
        label = f"{retriever_name}+{retriever.embedder.name}"
    elif retriever_name == "rerank" and reranker is not None:
        label = f"rerank+{retriever.embedder.name}+{reranker.name}"

    r5, r10, mrr, p5 = [], [], [], []
    per_mode: dict[str, list[float]] = {m: [] for m in MODE_COLS}

    questions = load_questions(questions_path) if questions_path else load_questions()
    for q in questions:
        gold = gold_ids(q["anchors"], nodes)
        ranked = [nid for nid, _ in retriever.retrieve(q["question"], mode=retriever_name, k=10)]
        hit5 = recall_at_k(ranked, gold, 5)
        hit10 = recall_at_k(ranked, gold, 10)
        r5.append(hit5)
        r10.append(hit10)
        mrr.append(reciprocal_rank(ranked, gold))
        if q["id"] in PRECISION_QIDS:
            p5.append(precision_at_k(ranked, gold, 5))
        for m in modes_of(q):
            if m in per_mode:
                per_mode[m].append(hit10)

    store.close()
    row = {
        "run": dt.date.today().isoformat(),
        "retriever": label,
        "recall@5": round(_mean(r5), 3),
        "recall@10": round(_mean(r10), 3),
        "precision@5": round(_mean(p5), 3),
        "mrr": round(_mean(mrr), 3),
    }
    for m in MODE_COLS:
        row[f"{m.lower()}@10"] = round(_mean(per_mode[m]), 3)

    _print_row(row)
    if out:
        _append_csv(out, row)
    return row


def _print_row(row: dict) -> None:
    print(f"\n{row['retriever']}  (run {row['run']})")
    print(f"  recall@5={row['recall@5']}  recall@10={row['recall@10']}  "
          f"precision@5={row['precision@5']}  mrr={row['mrr']}")
    per = "  ".join(f"{m.lower()}={row[m.lower() + '@10']}" for m in MODE_COLS)
    print(f"  per-mode recall@10:  {per}")


def _append_csv(out: str, row: dict) -> None:
    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if new:
            w.writeheader()
        w.writerow(row)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=DEFAULT_DB)
    ap.add_argument("--retriever", default="hybrid",
                    choices=["vector", "graph", "lexical", "hybrid", "rerank"])
    ap.add_argument("--embedder", default=None, help="hash | ollama (default: match index)")
    ap.add_argument("--reranker", default="ollama", help="identity | lexical | ollama (rerank mode)")
    ap.add_argument("--rerank-model", default=None, help="Ollama model for reranking (e.g. gemma2:9b)")
    ap.add_argument("--out", default="eval/results.csv")
    ap.add_argument("--questions", default=None, help="path to a questions.yaml (default: the httpx set)")
    args = ap.parse_args()
    qp = Path(args.questions) if args.questions else None
    run(args.db, args.retriever, args.embedder, args.out, args.reranker, args.rerank_model, qp)
    return 0


if __name__ == "__main__":
    sys.exit(main())
