#!/usr/bin/env bash
# Turnkey local eval with REAL embeddings via Ollama (e.g. on an Apple Silicon Mac).
# One command: check Ollama -> pull model -> fetch corpus -> index -> eval all retrievers.
# Fully offline; the only network call is to Ollama on 127.0.0.1.
#
# Usage: bash eval/run_local.sh [embed-model] [httpx-version]
#   bash eval/run_local.sh                       # nomic-embed-text, httpx 0.27.2
#   bash eval/run_local.sh mxbai-embed-large     # a 1024-dim model
set -euo pipefail

MODEL="${1:-nomic-embed-text}"
HTTPX_VERSION="${2:-0.27.2}"
HOST="${OLLAMA_HOST:-http://127.0.0.1:11434}"
DB="cartograph-out/httpx-ollama.kuzu"
OUT="eval/results-ollama.csv"

echo "==> Checking Ollama at ${HOST}"
if ! curl -sf "${HOST}/api/tags" >/dev/null; then
  echo "ERROR: Ollama not reachable at ${HOST}." >&2
  echo "       Install from https://ollama.com, then start it (it runs as a service" >&2
  echo "       on macOS, or run 'ollama serve')." >&2
  exit 1
fi

echo "==> Pulling embedding model: ${MODEL}"
ollama pull "${MODEL}"

echo "==> Fetching corpus (httpx==${HTTPX_VERSION})"
bash eval/get_corpus.sh "${HTTPX_VERSION}"

export CARTOGRAPH_EMBEDDER=ollama
export CARTOGRAPH_OLLAMA_MODEL="${MODEL}"

echo "==> Indexing with real embeddings (${MODEL})"
uv run cartograph index .corpus/httpx --db "${DB}" --embedder ollama

echo "==> Confirming eval anchors resolve"
uv run python eval/resolve_anchors.py --db "${DB}" --check

echo "==> Running eval (vector / lexical / graph / hybrid)"
rm -f "${OUT}"
for R in vector lexical graph hybrid; do
  uv run python eval/run_eval.py --db "${DB}" --retriever "${R}" --embedder ollama --out "${OUT}"
done

echo
echo "==> Results (${OUT}):"
column -t -s, "${OUT}" 2>/dev/null || cat "${OUT}"
echo
echo "Compare against the offline-embedder baseline in eval/README.md."
