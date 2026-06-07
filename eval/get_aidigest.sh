#!/usr/bin/env bash
# Fetch the ai-digest repo (a real raw-SQL/sqlite app) for the schema-bridging eval.
# Gitignored; regenerable. Demonstrates the eval is a corpus swap, not new code.
set -euo pipefail
DEST=".corpus/ai-digest"
rm -rf "$DEST"
git clone --depth 1 https://github.com/vigneshbhaskarraj/ai-digest "$DEST" >/dev/null 2>&1
echo "ai-digest -> $DEST ($(find "$DEST/src" -name '*.py' | wc -l) python files)"
