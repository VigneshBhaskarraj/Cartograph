"""Gate-3 agent benchmark: does an agent navigate code better WITH Cartograph?

Drives a local chat model (Ollama) through a constrained tool loop on the tasks in
tasks.yaml, under two conditions with comparable surfaces:

- ``--tools grep``        : grep(pattern), read(file,start,end), ls(dir) over raw
                            corpus files — the standard "agent greps the repo" loop.
- ``--tools cartograph``  : query(text), node(ref), calls(ref), callers(ref),
                            path(src,dst) over the indexed graph — no file access.

Per task it records: success (final answer names a gold symbol), tool calls used,
characters exchanged (a model-agnostic token proxy), and wall seconds. The protocol
is plain text so any chat model works, no function-calling API required:

    the model replies either  TOOL <name> <json-args>   or  ANSWER <text>

Zero egress: the only network call is the loopback Ollama chat endpoint (same
guard as embeddings). Usage:

  uv run python eval/agent_bench/run_bench.py --tools grep --model qwen2.5-coder:7b
  uv run python eval/agent_bench/run_bench.py --tools cartograph --model qwen2.5-coder:7b
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.request
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from cartograph.embed import _check_loopback  # noqa: E402
from cartograph.service import CartographService  # noqa: E402

CORPUS = {
    "httpx": (ROOT / ".corpus/httpx", ROOT / "cartograph-out/httpx.kuzu"),
    "flask": (ROOT / ".corpus/flask/src/flask", ROOT / "cartograph-out/flask.kuzu"),
}
MAX_TURNS = 10
OBS_LIMIT = 4000  # chars per tool observation


# -- tools ---------------------------------------------------------------------
class GrepTools:
    name = "grep"
    help = (
        'TOOL grep {"pattern": "regex"}            search all source files (case-insensitive)\n'
        'TOOL read {"file": "path", "start": 1, "end": 60}   read numbered lines\n'
        'TOOL ls {"dir": "."}                       list files'
    )

    def __init__(self, src: Path, db: Path):
        self.src = src
        self.files = {p.relative_to(src).as_posix(): p for p in sorted(src.rglob("*.py"))
                      if "__pycache__" not in p.parts}

    def close(self) -> None:
        pass

    def call(self, name: str, args: dict) -> str:
        if name == "grep":
            try:
                rx = re.compile(str(args.get("pattern", "")), re.IGNORECASE)
            except re.error as e:
                return f"bad pattern: {e}"
            out = []
            for rel, p in self.files.items():
                for i, line in enumerate(p.read_text(errors="replace").splitlines(), 1):
                    if rx.search(line):
                        out.append(f"{rel}:{i}: {line.strip()[:160]}")
                        if len(out) >= 30:
                            return "\n".join(out) + "\n... (truncated at 30 hits)"
            return "\n".join(out) if out else "(no matches)"
        if name == "read":
            rel = str(args.get("file", ""))
            p = self.files.get(rel)
            if p is None:
                return f"unknown file {rel!r}; use ls"
            lines = p.read_text(errors="replace").splitlines()
            start = max(1, int(args.get("start", 1)))
            end = min(len(lines), int(args.get("end", start + 59)))
            return "\n".join(f"{i}: {lines[i - 1]}" for i in range(start, end + 1))
        if name == "ls":
            return "\n".join(self.files)
        return f"unknown tool {name!r}"


class CartographTools:
    name = "cartograph"
    help = (
        'TOOL query {"text": "natural language or symbols"}   ranked symbols (hybrid retrieval)\n'
        'TOOL node {"ref": "Class.method"}                     full detail incl. docstring\n'
        'TOOL calls {"ref": "Class.method"}                    what it calls\n'
        'TOOL callers {"ref": "Class.method"}                  what calls it\n'
        'TOOL path {"src": "A", "dst": "B"}                    shortest connection'
    )

    def __init__(self, src: Path, db: Path):
        self.svc = CartographService(db)

    def close(self) -> None:
        self.svc.close()

    @staticmethod
    def _fmt(nodes: list[dict]) -> str:
        if not nodes:
            return "(none)"
        out = []
        for n in nodes[:10]:
            rel = f" <{n['relation']}:{n['direction']}>" if "relation" in n else ""
            out.append(f"[{n['kind']}] {n['qualified_name']}{rel}  ({n['file_path']}:{n['start_line']})")
        return "\n".join(out)

    def call(self, name: str, args: dict) -> str:
        if name == "query":
            return self._fmt(self.svc.query(str(args.get("text", "")), k=8))
        if name == "node":
            n = self.svc.get_node(str(args.get("ref", "")))
            if n is None:
                return "(no match — try query or a different ref)"
            doc = (n.get("docstring") or "")[:400]
            return (f"[{n['kind']}] {n['qualified_name']}  ({n['file_path']}:{n['start_line']})\n"
                    f"signature: {n['signature']}\ndocstring: {doc}")
        if name == "calls":
            return self._fmt(self.svc.calls(str(args.get("ref", ""))))
        if name == "callers":
            return self._fmt(self.svc.callers(str(args.get("ref", ""))))
        if name == "path":
            return self._fmt(self.svc.shortest_path(str(args.get("src", "")), str(args.get("dst", ""))))
        return f"unknown tool {name!r}"


# -- model backend --------------------------------------------------------------
class OllamaChat:
    def __init__(self, model: str, host: str | None = None):
        import os

        self.model = model
        self.host = (host or os.environ.get("OLLAMA_HOST") or "http://127.0.0.1:11434").rstrip("/")
        _check_loopback(self.host)

    def chat(self, messages: list[dict]) -> str:
        payload = json.dumps({"model": self.model, "messages": messages, "stream": False,
                              "options": {"temperature": 0}}).encode()
        req = urllib.request.Request(f"{self.host}/api/chat", data=payload,
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=300) as resp:
                return json.loads(resp.read())["message"]["content"]
        except OSError as e:
            raise RuntimeError(
                f"Ollama not reachable at {self.host} ({e}). Run `ollama serve` and "
                f"`ollama pull {self.model}`.") from e


_TOOL_RE = re.compile(r"TOOL\s+(\w+)\s*(\{.*?\})", re.DOTALL)
_ANSWER_RE = re.compile(r"ANSWER[:\s]+(.+)", re.DOTALL)


def system_prompt(tools) -> str:
    return (
        "You navigate a codebase to answer one question. You cannot see the code "
        "except through tools. Reply with EXACTLY ONE line per turn, either:\n"
        f"{tools.help}\n"
        "or, when you know the answer:\n"
        "ANSWER <the fully qualified symbol / name asked for>\n"
        f"You have at most {MAX_TURNS} tool calls. Be decisive; do not repeat a call."
    )


def run_task(task: dict, tools, model) -> dict:
    messages = [{"role": "system", "content": system_prompt(tools)},
                {"role": "user", "content": task["task"]}]
    chars = sum(len(m["content"]) for m in messages)
    calls, answer = 0, ""
    t0 = time.time()
    for _ in range(MAX_TURNS + 1):
        reply = model.chat(messages)
        chars += len(reply)
        messages.append({"role": "assistant", "content": reply})
        m = _ANSWER_RE.search(reply)
        if m:
            answer = m.group(1).strip()
            break
        t = _TOOL_RE.search(reply)
        if t and calls < MAX_TURNS:
            calls += 1
            try:
                args = json.loads(t.group(2))
            except json.JSONDecodeError:
                args = {}
            obs = tools.call(t.group(1), args)[:OBS_LIMIT]
            messages.append({"role": "user", "content": f"RESULT:\n{obs}"})
            chars += len(obs)
        else:
            messages.append({"role": "user", "content":
                             "Reply with exactly one TOOL line or one ANSWER line."})
    success = any(g.lower() in answer.lower() for g in task["gold"])
    return {"id": task["id"], "corpus": task["corpus"], "mode": task["mode"],
            "success": success, "tool_calls": calls, "chars": chars,
            "seconds": round(time.time() - t0, 1), "answer": answer[:200]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tools", required=True, choices=["grep", "cartograph"])
    ap.add_argument("--model", default="qwen2.5-coder:7b")
    ap.add_argument("--tasks", default=str(Path(__file__).parent / "tasks.yaml"))
    ap.add_argument("--out", default=None, help="append JSONL results here")
    args = ap.parse_args()

    tasks = yaml.safe_load(Path(args.tasks).read_text())
    model = OllamaChat(args.model)
    rows = []
    for task in tasks:
        src, db = CORPUS[task["corpus"]]
        if not src.exists() or not db.exists():
            print(f"skip task {task['id']}: corpus {task['corpus']} not indexed")
            continue
        tool_cls = GrepTools if args.tools == "grep" else CartographTools
        tools = tool_cls(src, db)
        try:
            row = run_task(task, tools, model)
        finally:
            tools.close()
        rows.append(row)
        mark = "PASS" if row["success"] else "FAIL"
        print(f"task {row['id']:>2} [{row['mode']:<8}] {mark}  calls={row['tool_calls']} "
              f"chars={row['chars']}  {row['answer'][:60]!r}")
        if args.out:
            with open(args.out, "a") as f:
                f.write(json.dumps({**row, "tools": args.tools, "model": args.model}) + "\n")

    n = len(rows)
    if n:
        ok = sum(r["success"] for r in rows)
        print(f"\n== {args.tools} / {args.model} ==")
        print(f"success {ok}/{n} ({ok / n:.0%})  "
              f"mean tool calls {sum(r['tool_calls'] for r in rows) / n:.1f}  "
              f"mean chars {sum(r['chars'] for r in rows) / n:.0f}  "
              f"mean seconds {sum(r['seconds'] for r in rows) / n:.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
