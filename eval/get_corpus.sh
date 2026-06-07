#!/usr/bin/env bash
# Fetch the httpx package source at a pinned version into .corpus/ for the eval.
# Pinned because private helpers (_send_*) shift across releases (see eval doc).
set -euo pipefail
VERSION="${1:-0.27.2}"
DEST=".corpus"
rm -rf "$DEST/httpx" "$DEST/site"
uv pip install --no-deps --target "$DEST/site" "httpx==${VERSION}" >/dev/null
cp -r "$DEST/site/httpx" "$DEST/httpx"
echo "httpx==${VERSION} -> $DEST/httpx ($(find "$DEST/httpx" -name '*.py' | wc -l) python files)"
