#!/usr/bin/env bash
# Run ALL eval corpora with REAL embeddings (Ollama) on a local machine.
# Prereqs: Ollama running; `uv sync --extra dev --extra sql`.
# Usage: bash eval/run_local_all.sh [embed-model] [rerank-model]
#   bash eval/run_local_all.sh                              # nomic-embed-text, no rerank
#   bash eval/run_local_all.sh nomic-embed-text gemma3:12b  # + a rerank row per corpus
set -euo pipefail

MODEL="${1:-nomic-embed-text}"
RERANK_MODEL="${2:-}"
HOST="${OLLAMA_HOST:-http://127.0.0.1:11434}"
export CARTOGRAPH_EMBEDDER=ollama
export CARTOGRAPH_OLLAMA_MODEL="$MODEL"

echo "==> Checking Ollama at ${HOST}"
curl -sf "${HOST}/api/tags" >/dev/null || { echo "ERROR: Ollama not reachable at ${HOST}." >&2; exit 1; }
ollama pull "${MODEL}"
[ -n "${RERANK_MODEL}" ] && ollama pull "${RERANK_MODEL}"

run_set () {  # name, db, questions(optional)
  local name="$1" db="$2" questions="${3:-}"
  echo; echo "### ${name}"
  local out="/tmp/eval_${name}.csv"; rm -f "${out}"
  for R in vector lexical graph hybrid; do
    if [ -n "${questions}" ]; then
      uv run python eval/run_eval.py --db "${db}" --retriever "${R}" --embedder ollama --questions "${questions}" --out "${out}"
    else
      uv run python eval/run_eval.py --db "${db}" --retriever "${R}" --embedder ollama --out "${out}"
    fi
  done
  if [ -n "${RERANK_MODEL}" ]; then
    if [ -n "${questions}" ]; then
      uv run python eval/run_eval.py --db "${db}" --retriever rerank --embedder ollama \
        --reranker ollama --rerank-model "${RERANK_MODEL}" --questions "${questions}" --out "${out}"
    else
      uv run python eval/run_eval.py --db "${db}" --retriever rerank --embedder ollama \
        --reranker ollama --rerank-model "${RERANK_MODEL}" --out "${out}"
    fi
  fi
}

echo "==> httpx (code eval)"
bash eval/get_corpus.sh 0.27.2
uv run cartograph index .corpus/httpx --db cartograph-out/httpx.kuzu --embedder ollama
run_set httpx cartograph-out/httpx.kuzu

echo "==> bridge_corpus (synthetic code+schema)"
uv run cartograph index eval/bridge_corpus --db cartograph-out/bridge.kuzu --embedder ollama
run_set bridge cartograph-out/bridge.kuzu eval/bridge_questions.yaml

echo "==> ai-digest (real raw-SQL repo)"
bash eval/get_aidigest.sh
uv run cartograph index .corpus/ai-digest/src --db cartograph-out/aidigest.kuzu --embedder ollama
run_set aidigest cartograph-out/aidigest.kuzu eval/aidigest_questions.yaml

echo; echo "Done. Compare each to the offline-embedder tables in eval/README.md."
