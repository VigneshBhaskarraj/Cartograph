# Running Cartograph locally with real embeddings (Apple Silicon / Ollama)

The default offline embedder is a feature-hash fallback — reproducible everywhere, but
it caps SEMANTIC recall because it isn't a real semantic model. On a local machine with
[Ollama](https://ollama.com) you get genuine code-aware embeddings (and, later, a local
reranker) with **zero data egress** — the only network call is to `127.0.0.1`. This is
the intended deployment target (`SPEC.md`: Apple Silicon, fully offline).

Reference machine: any M-series Mac. Embedding/reranker models are small (hundreds of
MB); 16 GB+ unified memory also leaves room for a local generative LLM later.

## Prerequisites
1. **Ollama** — install from https://ollama.com (runs as a background service on macOS).
2. **Python 3.12 + uv** — https://docs.astral.sh/uv/
3. Pull an embedding model (the script does this for you):
   ```bash
   ollama pull nomic-embed-text     # 768-dim, fast, general code/text
   # or: ollama pull mxbai-embed-large   # 1024-dim, stronger, a bit slower
   ```

## One command
```bash
uv sync --extra dev
bash eval/run_local.sh                    # nomic-embed-text + httpx 0.27.2
# bash eval/run_local.sh mxbai-embed-large   # try a 1024-dim model
```
This checks Ollama, pulls the model, fetches the pinned corpus, indexes httpx with **real
embeddings**, confirms the eval anchors, runs all four retrievers, and prints the table to
`eval/results-ollama.csv`. The vector column dimension auto-matches the model — no schema
edits needed.

## Using real embeddings outside the eval
Embeddings are baked into the Kuzu store at index time, so index **and** query must use
the same backend:
```bash
export CARTOGRAPH_EMBEDDER=ollama
export CARTOGRAPH_OLLAMA_MODEL=nomic-embed-text   # optional; this is the default
uv run cartograph index path/to/pkg --db cartograph-out/graph.kuzu --embedder ollama
uv run cartograph query "where is retry logic" --db cartograph-out/graph.kuzu --mode hybrid --embedder ollama
```

## What to expect vs the offline baseline
The offline `hash` run has SEMANTIC recall ≈ 0.43 (a non-semantic ceiling). With real
embeddings the vector leg's SEMANTIC recall should rise and pull `hybrid` up further —
verify it against the baseline table in [`eval/README.md`](../eval/README.md). That
measured delta is the green light for the M2 reranker stage.

## Notes
- **Reranker (M2 stage 2):** a cross-encoder isn't a first-class Ollama endpoint; it will
  use a small local ONNX/`fastembed` model (a new dependency, flagged before adding) or an
  LLM-as-reranker via Ollama generation. Added once real-embedding RRF plateaus.
- **Throttling:** a fanless MacBook Air handles indexing + eval easily (seconds–minute);
  only hours-long generative LLM workloads would thermally throttle.
