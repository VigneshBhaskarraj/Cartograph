"""M0-9: the whole pipe on a single REAL httpx file, answered via BOTH retrievers.

Skips when the pinned corpus isn't present (run `bash eval/get_corpus.sh` first).
"""

from pathlib import Path

import pytest

from cartograph.pipeline import index_path
from cartograph.retrieve import Retriever

CORPUS_FILE = Path(__file__).resolve().parents[1] / ".corpus" / "httpx" / "_client.py"

pytestmark = pytest.mark.skipif(
    not CORPUS_FILE.exists(),
    reason="httpx corpus not fetched; run `bash eval/get_corpus.sh`",
)


def test_m0_vertical_slice_on_real_file(tmp_path):
    store = index_path(CORPUS_FILE, tmp_path / "client.kuzu", dim=256, overwrite=True)
    try:
        # The graph actually captured structure from the real file.
        counts = store.counts()
        assert counts.get("node:class", 0) >= 1
        assert counts.get("node:method", 0) >= 5
        assert counts.get("edge:CONTAINS", 0) >= 5

        r = Retriever(store)
        # Vector path finds the ClientState enum by description.
        vec = [i for i, _ in r.vector("client lifecycle state opened closed", k=10)]
        assert any("ClientState" in i for i in vec)

        # Graph path reaches the send dispatch helpers from a lexical seed.
        gr = [i for i, _ in r.graph("Client send request", k=10)]
        assert any("_send_single_request" in i or "send" in i for i in gr)
    finally:
        store.close()
