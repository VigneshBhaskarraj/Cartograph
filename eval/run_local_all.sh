#!/usr/bin/env bash
# Run ALL eval corpora with REAL embeddings (Ollama) on a local machine.
# Prereqs: Ollama running; `uv sync --extra dev --extra sql`.
# Usage: bash eval/run_local_all.sh [embed-model]   (default: nomic-embed-text)
set -euo pipefail

MODEL="${1:-nomic-embed-text}"
HOST="${OLLAMA_HOST:-http://127.0.0.1:11434}"
export CARTOGRAPH_EMBEDDER=ollama
export CARTOGRAPH_OLLAMA_MODEL="$MODEL"

echo "==> Checking Ollama at ${HOST}"
curl -sf "${HOST}/api/tags" >/dev/null || { echo "ERROR: Ollama not reachable at ${HOST}." >&2; exit 1; }
ollama pull "${MODEL}"

run_set () {  # name, db, questions(optional)
  local name="$1" db="$2" questions="${3:-}"
  echo; echo "### ${name}"
  local qflag=(); [ -n "${questions}" ] && qflag=(--questions "${questions}")
  local out="/tmp/eval_${name}.csv"; rm -f "${out}"
  for R in vector lexical graph hybrid; do
    uv run python eval/run_eval.py --db "${db}" --retriever "${R}" --embedder ollama "${qflag[@]}" --out "${out}"
  done
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
