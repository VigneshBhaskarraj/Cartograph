#!/usr/bin/env bash
# Compare hybrid (RRF) vs the LLM-as-reranker stage, with real embeddings, locally.
# Fully offline; only talks to Ollama on 127.0.0.1.
#
# Usage: bash eval/run_rerank.sh <rerank-model> [embed-model] [httpx-version]
#   bash eval/run_rerank.sh gemma2:9b           # use a chat model you have
#   bash eval/run_rerank.sh llama3.2:3b         # smaller/faster reranker
# (run 'ollama list' to see installed models)
set -euo pipefail

RERANK_MODEL="${1:-}"
EMBED_MODEL="${2:-nomic-embed-text}"
HTTPX_VERSION="${3:-0.27.2}"
HOST="${OLLAMA_HOST:-http://127.0.0.1:11434}"
DB="cartograph-out/httpx-ollama.kuzu"
OUT="eval/results-ollama.csv"

if [ -z "${RERANK_MODEL}" ]; then
  echo "Usage: bash eval/run_rerank.sh <rerank-model> [embed-model] [httpx-version]" >&2
  echo "  Pick a chat model you have installed (run 'ollama list')." >&2
  echo "  e.g. gemma2:9b, llama3.2:3b, qwen2.5:7b" >&2
  exit 1
fi

echo "==> Checking Ollama at ${HOST}"
curl -sf "${HOST}/api/tags" >/dev/null || { echo "ERROR: Ollama not reachable at ${HOST}." >&2; exit 1; }

echo "==> Ensuring models: ${EMBED_MODEL} (embed), ${RERANK_MODEL} (rerank)"
ollama pull "${EMBED_MODEL}"
ollama show "${RERANK_MODEL}" >/dev/null 2>&1 || ollama pull "${RERANK_MODEL}"

echo "==> Corpus + index with real embeddings"
bash eval/get_corpus.sh "${HTTPX_VERSION}"
export CARTOGRAPH_EMBEDDER=ollama
export CARTOGRAPH_OLLAMA_MODEL="${EMBED_MODEL}"
export CARTOGRAPH_RERANK_MODEL="${RERANK_MODEL}"
uv run cartograph index .corpus/httpx --db "${DB}" --embedder ollama
uv run python eval/resolve_anchors.py --db "${DB}" --check

echo "==> Eval: vector / lexical / graph / hybrid / rerank(${RERANK_MODEL})"
rm -f "${OUT}"
for R in vector lexical graph hybrid; do
  uv run python eval/run_eval.py --db "${DB}" --retriever "${R}" --embedder ollama --out "${OUT}"
done
uv run python eval/run_eval.py --db "${DB}" --retriever rerank --embedder ollama \
    --reranker ollama --rerank-model "${RERANK_MODEL}" --out "${OUT}"

echo
echo "==> Results (${OUT}):"
column -t -s, "${OUT}" 2>/dev/null || cat "${OUT}"
echo
echo "Goal: rerank MRR rises above vector-only (~0.52) without losing recall@10 (~0.81)."
