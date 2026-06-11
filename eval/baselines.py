"""External baselines: what an agent gets WITHOUT Cartograph.

Two competitors, scored on the same questions, gold sets, and metrics as the real
retrievers, so the scorecard finally answers "compared to what?":

- ``grep``  — term search over raw source lines (what an agent does with
  grep/ripgrep): query words minus function words, case-insensitive, ranked by
  (#distinct terms matched, total matching lines) per symbol.
- ``naive-rag`` — the standard RAG recipe with no code structure: fixed
  40-line/20-stride windows over raw files, embedded with the same embedder as
  the index, cosine-ranked; a chunk's score credits every symbol it overlaps.

Both baselines read source files directly — that is the point (they simulate the
alternative); Cartograph retrieval itself never does. Graph node spans are used
ONLY to map text hits onto gold node ids for scoring (the answer key is expressed
as nodes), never as a retrieval signal. The chunk->symbol overlap mapping is
deliberately generous to the baseline.

Usage: uv run python eval/baselines.py [--embedder hash|ollama] [--out CSV]
(or via `eval/scorecard.py --baselines` for the combined table)
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))

from evallib import (  # noqa: E402
    gold_ids,
    load_questions,
    open_store,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)

from cartograph.cache import EmbeddingCache  # noqa: E402
from cartograph.embed import get_embedder  # noqa: E402
from cartograph.pipeline import _files  # noqa: E402
from cartograph.retrieve import _cosine_ranking  # noqa: E402
from run_eval import MODE_COLS, PRECISION_MODES  # noqa: E402

_WORD_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]+")
_STOP = {
    "what", "which", "where", "when", "how", "does", "do", "is", "are", "the",
    "a", "an", "of", "to", "in", "for", "on", "and", "or", "that", "this", "it",
    "with", "from", "be", "can", "as", "at", "by", "its", "their", "there",
    "if", "then", "into", "about", "not", "any", "all", "one", "two", "via",
}

CHUNK_LINES = 40
CHUNK_STRIDE = 20


def _query_terms(question: str) -> list[str]:
    return [w.lower() for w in _WORD_RE.findall(question) if w.lower() not in _STOP and len(w) > 2]


class CorpusText:
    """Raw source lines + the graph's node spans (the scoring answer key)."""

    def __init__(self, src: Path, db: str):
        self.files: dict[str, list[str]] = {}
        for suffix in (".py", ".sql", ".ts", ".tsx"):
            for f in _files(src, suffix):
                rel = f.relative_to(src.parent).as_posix()
                self.files[rel] = f.read_text(encoding="utf-8", errors="replace").splitlines()
        store, self.nodes = open_store(db)
        res = store.conn.execute(
            "MATCH (c:CodeNode) WHERE c.kind <> 'external' "
            "RETURN c.id, c.file_path, c.start_line, c.end_line, c.kind")
        self.spans: dict[str, list[tuple[int, int, str, str]]] = defaultdict(list)
        while res.has_next():
            nid, fp, sl, el, kind = res.get_next()
            self.spans[fp].append((sl or 0, el or 0, nid, kind))
        store.close()
        for fp in self.spans:
            self.spans[fp].sort(key=lambda s: (s[1] - s[0]))  # innermost first

    def node_at(self, file_path: str, line: int) -> str | None:
        """Innermost non-module symbol enclosing the line; module as fallback."""
        module = None
        for sl, el, nid, kind in self.spans.get(file_path, ()):
            if sl <= line <= el:
                if kind != "module":
                    return nid
                module = module or nid
        return module

    def nodes_overlapping(self, file_path: str, start: int, end: int) -> list[str]:
        out = []
        for sl, el, nid, _kind in self.spans.get(file_path, ()):
            if sl <= end and el >= start:
                out.append(nid)
        return out


def grep_rank(corpus: CorpusText, question: str, k: int = 10) -> list[str]:
    terms = _query_terms(question)
    if not terms:
        return []
    per_node_terms: dict[str, set[str]] = defaultdict(set)
    per_node_hits: dict[str, int] = defaultdict(int)
    for fp, lines in corpus.files.items():
        for i, line in enumerate(lines, 1):
            low = line.lower()
            matched = [t for t in terms if t in low]
            if not matched:
                continue
            nid = corpus.node_at(fp, i)
            if nid is None:
                continue
            per_node_terms[nid].update(matched)
            per_node_hits[nid] += len(matched)
    ranked = sorted(per_node_terms,
                    key=lambda n: (-len(per_node_terms[n]), -per_node_hits[n], n))
    return ranked[:k]


class NaiveRag:
    """Fixed-window chunk embeddings over raw files — structure-blind RAG."""

    def __init__(self, corpus: CorpusText, embedder, cache_dir: Path | None = None):
        self.corpus = corpus
        self.embedder = embedder
        chunks: list[tuple[str, int, int, str]] = []  # (file, start, end, text)
        for fp, lines in corpus.files.items():
            if not lines:
                continue
            for start in range(0, len(lines), CHUNK_STRIDE):
                end = min(start + CHUNK_LINES, len(lines))
                chunks.append((fp, start + 1, end, "\n".join(lines[start:end])))
                if end == len(lines):
                    break
        self.chunks = chunks
        cache = None
        if cache_dir is not None:
            cache = EmbeddingCache.for_embedder(cache_dir, f"rag.{getattr(embedder, 'name', 'hash')}")
        self.vecs: list[list[float]] = []
        pending = []
        for _, _, _, text in chunks:
            v = cache.get(text) if cache is not None else None
            self.vecs.append(v)
            if v is None:
                pending.append(text)
        if pending:
            fresh = iter(embedder.embed_batch(pending))
            for i, v in enumerate(self.vecs):
                if v is None:
                    nv = next(fresh)
                    self.vecs[i] = nv
                    if cache is not None:
                        cache.put(self.chunks[i][3], nv)
        if cache is not None:
            cache.save()

    def rank(self, question: str, k: int = 10) -> list[str]:
        qv = self.embedder.embed(question)
        ids = list(range(len(self.chunks)))
        ranked_chunks = _cosine_ranking(qv, ids, self.vecs)
        node_score: dict[str, float] = {}
        for cid, score in ranked_chunks:
            fp, start, end, _ = self.chunks[cid]
            for nid in self.corpus.nodes_overlapping(fp, start, end):
                if nid not in node_score:  # max chunk score per node
                    node_score[nid] = score
            if len(node_score) >= k * 5:
                break
        ranked = sorted(node_score, key=lambda n: (-node_score[n], n))
        return ranked[:k]


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def run_baseline(name: str, src: Path, db: str, questions_path: Path | None,
                 embedder_name: str = "hash") -> dict:
    """One scorecard-format row for a baseline over one corpus."""
    corpus = CorpusText(Path(src), db)
    questions = load_questions(questions_path) if questions_path else load_questions()
    if name == "grep":
        rank = lambda q: grep_rank(corpus, q, k=10)  # noqa: E731
        label = "grep"
    elif name == "naive-rag":
        embedder = get_embedder(embedder_name)
        rag = NaiveRag(corpus, embedder, cache_dir=Path(db).parent / "cache")
        rank = lambda q: rag.rank(q, k=10)  # noqa: E731
        label = f"naive-rag+{getattr(embedder, 'name', embedder_name)}"
    else:
        raise ValueError(f"unknown baseline {name!r}")

    r5, r10, mrr, p5 = [], [], [], []
    per_mode: dict[str, list[float]] = {m: [] for m in MODE_COLS}
    for q in questions:
        gold = gold_ids(q["anchors"], corpus.nodes)
        # zero-gold questions score 0, matching run_eval — skipping them here would
        # skew baseline-vs-retriever comparison if an anchor ever broke
        ranked = rank(q["question"])
        hit10 = recall_at_k(ranked, gold, 10)
        r5.append(recall_at_k(ranked, gold, 5))
        r10.append(hit10)
        mrr.append(reciprocal_rank(ranked, gold))
        modes = q["mode"].split("+")
        if set(modes) <= PRECISION_MODES:
            p5.append(precision_at_k(ranked, gold, 5))
        for m in modes:
            if m in per_mode:
                per_mode[m].append(hit10)
    row = {
        "retriever": label,
        "recall@5": round(_mean(r5), 3),
        "recall@10": round(_mean(r10), 3),
        "precision@5": round(_mean(p5), 3),
        "mrr": round(_mean(mrr), 3),
    }
    for m in MODE_COLS:
        row[f"{m.lower()}@10"] = round(_mean(per_mode[m]), 3) if per_mode[m] else "n/a"
    return row


def main() -> int:
    from scorecard import CORPORA

    ap = argparse.ArgumentParser()
    ap.add_argument("--embedder", default="hash")
    ap.add_argument("--baseline", default="both", choices=["grep", "naive-rag", "both"])
    args = ap.parse_args()
    names = ["grep", "naive-rag"] if args.baseline == "both" else [args.baseline]
    hdr = f"{'corpus':<10} {'baseline':<22} {'recall@5':>8} {'recall@10':>9} {'mrr':>6} {'prec@5':>7}"
    print(hdr)
    print("-" * len(hdr))
    for cname, src, db, questions in CORPORA:
        if not (ROOT / src).exists() or not (ROOT / db).exists():
            print(f"skip {cname}: corpus or db not present")
            continue
        if questions and not (ROOT / questions).exists():
            print(f"skip {cname}: questions {questions} not present")
            continue
        for b in names:
            row = run_baseline(b, ROOT / src, str(ROOT / db),
                               Path(ROOT / questions) if questions else None, args.embedder)
            print(f"{cname:<10} {row['retriever']:<22} {row['recall@5']:>8} "
                  f"{row['recall@10']:>9} {row['mrr']:>6} {row['precision@5']:>7}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
