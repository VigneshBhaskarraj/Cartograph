"""Shared eval helpers: load questions, resolve anchors, score rankings."""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

# Make `cartograph` importable regardless of install mode.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cartograph.store import Store  # noqa: E402

QUESTIONS = Path(__file__).parent / "questions.yaml"


def load_questions(path: Path = QUESTIONS) -> list[dict]:
    return yaml.safe_load(path.read_text())


def anchor_matches(anchor: str, node: dict) -> bool:
    qn = node["qualified_name"]
    return qn == anchor or qn.endswith("." + anchor) or node["name"] == anchor


def gold_ids(anchors: list[str], nodes: list[dict]) -> set[str]:
    out: set[str] = set()
    for a in anchors:
        for n in nodes:
            if anchor_matches(a, n):
                out.add(n["id"])
    return out


def modes_of(question: dict) -> list[str]:
    return [m for m in question["mode"].split("+")]


# -- metrics ------------------------------------------------------------------
def recall_at_k(ranked: list[str], gold: set[str], k: int) -> float:
    return 1.0 if gold & set(ranked[:k]) else 0.0


def precision_at_k(ranked: list[str], gold: set[str], k: int) -> float:
    if not ranked[:k]:
        return 0.0
    return len(gold & set(ranked[:k])) / float(k)


def reciprocal_rank(ranked: list[str], gold: set[str]) -> float:
    for i, nid in enumerate(ranked, 1):
        if nid in gold:
            return 1.0 / i
    return 0.0


def open_store(db: str) -> tuple[Store, list[dict]]:
    store = Store(db)
    return store, store.all_nodes_text()
