"""M1-7: metric correctness on hand-built rankings."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))

from evallib import precision_at_k, recall_at_k, reciprocal_rank  # noqa: E402


def test_recall_at_k():
    ranked = ["a", "b", "c", "d", "e"]
    assert recall_at_k(ranked, {"c"}, 5) == 1.0
    assert recall_at_k(ranked, {"c"}, 2) == 0.0
    assert recall_at_k(ranked, {"z"}, 5) == 0.0


def test_precision_at_k():
    ranked = ["a", "b", "c", "d", "e"]
    assert precision_at_k(ranked, {"a", "b"}, 5) == 2 / 5
    assert precision_at_k(ranked, {"a", "b"}, 2) == 1.0


def test_reciprocal_rank():
    assert reciprocal_rank(["a", "b", "c"], {"b"}) == 0.5
    assert reciprocal_rank(["a", "b", "c"], {"a"}) == 1.0
    assert reciprocal_rank(["a", "b", "c"], {"z"}) == 0.0
