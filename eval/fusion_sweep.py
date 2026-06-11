"""Fusion parameter sweep: find (weights, rrf_k, depth) where hybrid beats vector.

The 2026-06-07 Gate-1 scorecard (real ollama embeddings, 4 corpora) showed plain
vector search beating the equal-weight RRF hybrid on every aggregate metric: two
low-precision signals (graph, lexical) outvote one high-precision signal (vector),
and rrf_k=60 flattens rank differences so consensus mediocrity beats a confident
top hit. This harness searches the weighted-fusion space for a config that wins.

Cost model: each signal's ranking is computed ONCE per question at depth 50 (the
only expensive part — one embedding call per query), then the whole grid re-fuses
cached rankings in pure CPU, so 128 configs cost the same as one eval pass.

Decision rule (printed in the verdict): a config wins only if mean recall@5 AND
mean MRR across all present corpora are >= the vector-only baseline. If nothing
wins, the honest move is to demote hybrid to vector-primary — the harness says so
rather than hiding it.

Usage: uv run python eval/fusion_sweep.py [--embedder hash|ollama] [--reindex]
                                          [--out eval/fusion_sweep.csv]
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))

from evallib import gold_ids, load_questions, open_store, recall_at_k, reciprocal_rank  # noqa: E402
from scorecard import CORPORA  # noqa: E402

from cartograph.embed import get_embedder  # noqa: E402
from cartograph.pipeline import index_path  # noqa: E402
from cartograph.retrieve import Retriever, rrf_fuse  # noqa: E402

DEPTH_MAX = 50
K = 10
GRID = [
    (wv, wg, wl, rrf_k, depth)
    for wv in (1.0, 1.5, 2.0, 3.0)
    for wg in (0.5, 1.0)
    for wl in (0.5, 1.0)
    for rrf_k in (10, 20, 40, 60)
    for depth in (20, 50)
]
CURRENT = (3.0, 0.5, 0.5, 10, 50)  # the calibrated hybrid() default (2026-06-11 sweep)


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def collect_rankings(embedder_name: str, reindex: bool) -> dict[str, list[dict]]:
    """Per corpus: one entry per question with gold ids and depth-50 rankings."""
    out: dict[str, list[dict]] = {}
    for name, src, db, questions in CORPORA:
        if not (ROOT / src).exists():
            print(f"skip {name}: source {src} not present")
            continue
        if questions and not (ROOT / questions).exists():
            print(f"skip {name}: questions {questions} not present")
            continue
        if reindex or not (ROOT / db).exists():
            t0 = time.time()
            index_path(ROOT / src, ROOT / db, embedder=get_embedder(embedder_name), overwrite=True).close()
            print(f"indexed {name} in {time.time() - t0:.1f}s")
        store, nodes = open_store(str(ROOT / db))
        retriever = Retriever(store, embedder=get_embedder(embedder_name))
        rows = []
        for q in load_questions(Path(ROOT / questions)) if questions else load_questions():
            gold = gold_ids(q["anchors"], nodes)
            if not gold:
                continue
            rows.append({
                "gold": gold,
                "vector": [i for i, _ in retriever.vector(q["question"], k=DEPTH_MAX)],
                "graph": [i for i, _ in retriever.graph(q["question"], k=DEPTH_MAX)],
                "lexical": [i for i, _ in retriever.lexical(q["question"], k=DEPTH_MAX)],
            })
        store.close()
        out[name] = rows
        print(f"cached {len(rows)} question rankings for {name}")
    return out


def score_ranked(rows: list[dict], ranked_of) -> tuple[float, float, float]:
    r5, r10, mrr = [], [], []
    for row in rows:
        ranked = ranked_of(row)
        r5.append(recall_at_k(ranked, row["gold"], 5))
        r10.append(recall_at_k(ranked, row["gold"], 10))
        mrr.append(reciprocal_rank(ranked, row["gold"]))
    return _mean(r5), _mean(r10), _mean(mrr)


def fused_ranking(row: dict, wv: float, wg: float, wl: float, rrf_k: int, depth: int) -> list[str]:
    rankings = [row["vector"][:depth], row["graph"][:depth], row["lexical"][:depth]]
    return [i for i, _ in rrf_fuse(rankings, k=K, rrf_k=rrf_k, weights=[wv, wg, wl])]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--embedder", default="hash")
    ap.add_argument("--reindex", action="store_true")
    ap.add_argument("--out", default=None, help="optional CSV of every config's scores")
    args = ap.parse_args()

    corpora = collect_rankings(args.embedder, args.reindex)
    if not corpora:
        print("no corpora present — fetch with eval/get_corpus.sh etc.")
        return 1

    # Baseline the fusion must beat: vector alone, macro-averaged across corpora.
    vec_by_corpus = {n: score_ranked(rows, lambda r: r["vector"][:K]) for n, rows in corpora.items()}
    vec_mean = tuple(_mean([v[i] for v in vec_by_corpus.values()]) for i in range(3))

    results = []
    for cfg in [CURRENT] + GRID:
        wv, wg, wl, rrf_k, depth = cfg
        by_corpus = {
            n: score_ranked(rows, lambda r: fused_ranking(r, wv, wg, wl, rrf_k, depth))
            for n, rows in corpora.items()
        }
        mean = tuple(_mean([v[i] for v in by_corpus.values()]) for i in range(3))
        # Consistency: corpora where this config matches/beats vector on r@5 AND mrr.
        wins = sum(
            1 for n in corpora
            if by_corpus[n][0] >= vec_by_corpus[n][0] and by_corpus[n][2] >= vec_by_corpus[n][2]
        )
        results.append({"cfg": cfg, "mean": mean, "wins": wins, "by_corpus": by_corpus})

    results.sort(key=lambda r: -(r["mean"][0] + r["mean"][2]))

    print(f"\nembedder={args.embedder}  corpora={list(corpora)}  questions={sum(len(r) for r in corpora.values())}")
    hdr = f"{'w_vec':>5} {'w_gr':>5} {'w_lex':>5} {'rrf_k':>5} {'depth':>5} | {'r@5':>6} {'r@10':>6} {'mrr':>6} {'wins':>4}"
    print(f"\n{'vector baseline':<31} | {vec_mean[0]:>6.3f} {vec_mean[1]:>6.3f} {vec_mean[2]:>6.3f}")
    print(hdr)
    print("-" * len(hdr))
    for r in results[:15]:
        wv, wg, wl, rrf_k, depth = r["cfg"]
        tag = "  <- current" if r["cfg"] == CURRENT else ""
        print(f"{wv:>5} {wg:>5} {wl:>5} {rrf_k:>5} {depth:>5} | "
              f"{r['mean'][0]:>6.3f} {r['mean'][1]:>6.3f} {r['mean'][2]:>6.3f} {r['wins']:>4}{tag}")
    cur = next(r for r in results if r["cfg"] == CURRENT)
    if cur not in results[:15]:
        wv, wg, wl, rrf_k, depth = cur["cfg"]
        print(f"{wv:>5} {wg:>5} {wl:>5} {rrf_k:>5} {depth:>5} | "
              f"{cur['mean'][0]:>6.3f} {cur['mean'][1]:>6.3f} {cur['mean'][2]:>6.3f} {cur['wins']:>4}  <- current")

    winners = [r for r in results if r["mean"][0] >= vec_mean[0] and r["mean"][2] >= vec_mean[2]]
    print("\n=== VERDICT ===")
    if winners:
        # Prefer the most consistent winner; break ties by mean score.
        best = max(winners, key=lambda r: (r["wins"], r["mean"][0] + r["mean"][2]))
        wv, wg, wl, rrf_k, depth = best["cfg"]
        print(f"hybrid matches/beats vector ({len(winners)}/{len(results)} configs qualify).")
        print(f"recommended defaults: weights=({wv}, {wg}, {wl}), rrf_k={rrf_k}, depth={depth}")
        print(f"  mean r@5 {best['mean'][0]:.3f} vs vector {vec_mean[0]:.3f}; "
              f"mrr {best['mean'][2]:.3f} vs {vec_mean[2]:.3f}; qualifies on {best['wins']}/{len(corpora)} corpora")
        for n, (r5, r10, mrr) in best["by_corpus"].items():
            v5, v10, vm = vec_by_corpus[n]
            print(f"  {n:<10} r@5 {r5:.3f} (vec {v5:.3f})  r@10 {r10:.3f} (vec {v10:.3f})  mrr {mrr:.3f} (vec {vm:.3f})")
        # Selection happens on the same questions that score it — flag overfitting:
        # the pick should still qualify with any single corpus held out.
        if len(corpora) > 1:
            loco_ok = True
            for held in corpora:
                rest = [n for n in corpora if n != held]
                m5 = _mean([best["by_corpus"][n][0] for n in rest])
                mm = _mean([best["by_corpus"][n][2] for n in rest])
                v5 = _mean([vec_by_corpus[n][0] for n in rest])
                vm = _mean([vec_by_corpus[n][2] for n in rest])
                if m5 < v5 or mm < vm:
                    loco_ok = False
            print("  leave-one-corpus-out: " + (
                "stable — qualifies with any corpus held out" if loco_ok else
                "UNSTABLE — the pick depends on one corpus; add a held-out corpus before baking defaults"))
    else:
        print("NO config beats vector on mean r@5 + mrr. The honest move is to demote")
        print("hybrid to vector-primary and update SPEC.md — do not ship a fusion that")
        print("loses to its own component.")

    if args.out:
        with Path(args.out).open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["w_vec", "w_graph", "w_lex", "rrf_k", "depth", "recall@5", "recall@10", "mrr", "wins"])
            for r in results:
                w.writerow([*r["cfg"], f"{r['mean'][0]:.4f}", f"{r['mean'][1]:.4f}", f"{r['mean'][2]:.4f}", r["wins"]])
        print(f"\nwrote {len(results)} rows to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
