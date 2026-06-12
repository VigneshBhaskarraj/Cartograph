# Gate-3 agent benchmark — pilot results (2026-06-11)

**Question:** does an AI agent navigate a codebase better with Cartograph than with
grep? "Better" = same-or-higher task success with fewer tool calls (tool calls are
the agent-loop cost driver: every call burns a turn and pushes observations into
context).

## Setup

- **Tasks:** the 12 source-verified navigation tasks in `tasks.yaml` (httpx + flask:
  callers/callees, redirect machinery, semantic lookups, exact lookups, multihop
  traces). Authored by reading source, never by testing retrieval.
- **Conditions (matched surfaces, max 8 commands each):**
  - `grep`: `grep -rn` / `sed` line-range reads / `ls`-`find` over the raw corpus.
  - `cartograph`: the `cartograph` CLI only — `query` (hybrid retrieval), `node`,
    `resolve`, `calls`, `callers`, `path`. No file access at all.
- **Driver (pilot):** isolated Claude Code subagents (claude-haiku-4-5), one per
  task per condition, temperature defaults, no shared context. Per-task raw data:
  [`pilot_results.jsonl`](./pilot_results.jsonl). The offline harness
  (`run_bench.py`, any local Ollama chat model, zero egress) implements the same
  protocol shape (line-oriented TOOL/ANSWER loop, 8-command cap) for independent
  replication; the pilot's tool surface was the equivalent shell commands
  (`grep`/`sed`/`ls`-`find` vs the `cartograph` CLI).
- **Grading:** the final answer must contain a gold symbol on identifier word
  boundaries (case-sensitive for capitalized golds, so prose "cookies" can't match
  the class `Cookies`). One task's gold was widened post-hoc (task 5:
  `handle_request` accepted alongside `_send_single_request`) because the
  baseline's answer was a defensible reading — grading generosity went **to the
  baseline**.

## Results (n = 12 tasks per condition)

| condition | success | mean tool calls | median calls |
| --- | --- | --- | --- |
| grep | 12/12 (11/12 before gold widening) | **5.2** | 5 |
| cartograph | 12/12 | **3.0** | 2.5 |

- **Equal success, 42% fewer tool calls.** Cartograph answered structural questions
  (callers/inheritance) in 2–3 calls where grep needed 5–8 (search → open file →
  read context → confirm enclosing class…).
- Cartograph's wins were biggest exactly where the graph is the point: "which method
  calls X" became one `callers` call; "what class hierarchy handles Y" became one
  `query` + one `node`.
- grep stayed competitive on EXACT lookups (a name you already know is one grep away)
  — consistent with the retrieval scorecard, where lexical search wins EXACT mode.

## Honest caveats (read before quoting)

1. **n=12 with a strong driver model.** Haiku-class models grep well; both conditions
   saturate success on tasks this size. The differentiating signal at this n is
   **efficiency**, not accuracy. Expect accuracy gaps to appear with weaker local
   models (the `run_bench.py` Ollama path) and larger/multi-package corpora, where
   grep's hit lists explode — measure, don't assume.
2. **The pilot graphs were indexed with the offline `hash` embedder** — the weakest
   setting for Cartograph's `query` tool. Real deployments use Ollama embeddings,
   which strengthen exactly the semantic lookups. This biases the pilot *against*
   Cartograph.
3. Subagent token totals are dominated by per-agent harness overhead and are not a
   clean signal at this scale; `run_bench.py` measures conversation characters
   properly for the offline run.
4. Wall-clock was similar: each `uv run cartograph` call pays ~1–2 s of process
   startup the in-process MCP server doesn't pay.

## Reproduce offline (zero egress)

```bash
bash eval/get_corpus.sh 0.27.2 && bash eval/get_flask.sh
CARTOGRAPH_EMBEDDER=ollama uv run cartograph index .corpus/httpx --db cartograph-out/httpx.kuzu
CARTOGRAPH_EMBEDDER=ollama uv run cartograph index .corpus/flask/src/flask --db cartograph-out/flask.kuzu
ollama pull qwen2.5-coder:7b
uv run python eval/agent_bench/run_bench.py --tools grep --model qwen2.5-coder:7b --out eval/agent_bench/results.jsonl
uv run python eval/agent_bench/run_bench.py --tools cartograph --model qwen2.5-coder:7b --out eval/agent_bench/results.jsonl
```
