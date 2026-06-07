# Eval corpus

- **Corpus:** the `httpx` package only, pinned at **`httpx==0.27.2`**.
- **Why pinned:** private helpers (`_send_*`) and class layout shift across releases;
  the eval anchors in `questions.yaml` are confirmed against this exact version.
- **How to fetch:** `bash eval/get_corpus.sh 0.27.2` → `.corpus/httpx/` (gitignored;
  regenerable, never committed).
- **Anchors confirmed on this version** via `eval/resolve_anchors.py --check`
  (e.g. `Auth.sync_auth_flow` lives on the base class; `DigestAuth` overrides
  `auth_flow`; `event_hooks` is a `BaseClient` property).
