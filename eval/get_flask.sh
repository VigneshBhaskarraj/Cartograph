#!/usr/bin/env bash
# Fetch Flask (a second real Python repo) for the multi-corpus scorecard. Gitignored.
set -euo pipefail
rm -rf .corpus/flask
git clone --depth 1 https://github.com/pallets/flask .corpus/flask >/dev/null 2>&1
echo "flask -> .corpus/flask/src/flask ($(find .corpus/flask/src -name '*.py' | wc -l) python files)"
