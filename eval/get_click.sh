#!/usr/bin/env bash
# Fetch click (pinned) — the HELD-OUT validation corpus. Its questions are written
# blind to fusion tuning: the 2026-06-11 sweep calibrated on the other four corpora,
# so click is the generalization check (especially for the corpus-sensitive mrr win).
set -euo pipefail
TAG="${1:-8.1.7}"
rm -rf .corpus/click
git clone --depth 1 --branch "$TAG" https://github.com/pallets/click .corpus/click >/dev/null 2>&1
echo "click@$TAG -> .corpus/click/src/click ($(find .corpus/click/src -name '*.py' | wc -l) python files)"
