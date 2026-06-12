#!/usr/bin/env bash
# Fetch spring-petclinic (pinned) — the Java eval corpus: canonical JPA entities +
# multi-dialect schema.sql, exactly the enterprise shape the Java extractor targets.
set -euo pipefail
SHA="${1:-$(cat "$(dirname "$0")/petclinic.pin")}"
rm -rf .corpus/petclinic
git clone https://github.com/spring-projects/spring-petclinic .corpus/petclinic >/dev/null 2>&1
git -C .corpus/petclinic checkout -q "$SHA"
echo "petclinic@$SHA -> .corpus/petclinic/src/main ($(find .corpus/petclinic/src/main -name '*.java' | wc -l) java files)"
