# kb

Homelab knowledge base search backend — Python API, MCP server, ChromaDB embed daemon, FTS5 hybrid retrieval, and mMARCO cross-encoder reranking.

> **CLI frontend:** [kb-go](https://github.com/pradeda/kb-go) — Go CLI (`kb ask`, `kb add`, `kb search`, `kb retire`) that calls this backend.

## Components

| Component | File | Port/Socket |
|-----------|------|-------------|
| Search API (v2) | `kb_v2.py` via `kb_search_api.py` | `:8050` (user unit: `kb-search-api`) |
| MCP server | `mcp_server.py` | system units `kb-mcp-sse` / `kb-mcp-http`; stdio clients |
| FastEmbed daemon | `embed_daemon.py` | `/run/kb-embed/embed.sock` |
| Compiler | `compile.py` | invoked by `kb-watcher` / `ai-kb-watcher`; owns the per-corpus mutating lock (`/tmp/kb-watcher.lock`, `/tmp/ai-kb-watcher.lock`) for every mutating mode — `watcher.sh` must not hold a shell `flock` on the same path, that deadlocks the child |
| Corpus router | `corpus-router.yml` | config for search API |
| KB Atlas | `kb_atlas.py` + `atlas_template.html` | `:3085` (user units `kb-atlas` / `kb-atlas-rebuild.timer`); source of truth in [kb-go](https://github.com/pradeda/kb-go) `runtime/`, deployed via `make install` |

## Which files are owned here, and which are installed from kb-go

This tree is the live deployment, not the source of truth for everything in it.
Nine artifacts are **installed from [kb-go](https://github.com/pradeda/kb-go)** by
`make install` in that repo (`DEPLOY_MANIFEST` is the authoritative list):

`compile.py` · `watcher.sh` · `refresh_volatile.sh` · `supersede_index.py` ·
`embed_daemon.py` · `kb_atlas.py` · `atlas_template.html` · `secret_patterns.json`
(→ `/opt/kb/`), and `kb-health-check.sh` (→ `/home/turok/scripts/`).

Editing one of those here is overwritten by the next `make install`, and
`make verify-installed` in kb-go reports `DRIFT` until then. A fix must land in
kb-go `runtime/` (or `config/`) in the same session; the deployed tree is what
must be verified live first, because services read `/opt/kb`.

Everything else in this tree is canonical **here** and has no kb-go counterpart:
`kb_v2.py`, `kb_search_api.py`, `mcp_server.py`, `gate.py`, `index_gemini.py`,
`docker-compose.yml`, `corpus-router.yml`, `v2-clients.yml`, `eval/`, `setup/`,
`prompts/`, `mkdocs.yml`. `tests/` is mirrored in both repos and is run there by
`make test`, except `tests/test_mcp_*.py`, which covers `mcp_server.py` and lives
only here (run it with `/opt/kb/venv/bin/python3 -m unittest discover -s tests -p 'test_mcp_*.py'`).

## Retrieval pipeline

```
Query
  ├── FastEmbed (nomic-embed-text-v1.5, Unix socket, ~50ms)
  ├── ChromaDB cosine top 20 semantic candidates
  ├── FTS5 top 5 lexical-only candidates (BM25, porter stemmer)
  ├── Merge: 20 semantic + up to 5 FTS5-only, backfill to 25
  ├── Cross-encoder rerank (mmarco-mMiniLMv2-L12-H384-v1, CPU)
  │     └── Tokenizer-bounded passage selection (512 token budget)
  ├── Dedup best chunk per entry
  ├── Time decay: final = relevance × max(1/(1+days/540), 0.3)
  └── Threshold 0.40 → top 5
```

## MCP tools

`mcp_server.py` exposes the corpora to agents (stdio for local Claude Code, `:9100` SSE,
`:9101` Streamable HTTP). No tool makes an LLM call.

| Tool | What it does |
|------|--------------|
| `semantic_search(query, query_alt?, query_alt_language?, scope?)` | Default search, required before agent work. `scope="both"` (default): homelab hits in full, at most 2 AI hits as a brief (source + summary, ≤700 chars). `scope="ai"`: full AI entries, for questions about models, tools, papers or techniques. `scope="homelab"`: homelab only. |
| `kb_get(reference)` | Full text of one entry, e.g. `kb_get("ai:363")`, read-only from the corpus SQLite. Use it to expand a brief. |
| `corpus_search(query, scope, top_k, …)` | Raw v2 response as JSON (grouped + merged ranking). `scope="auto"` returns 409 until the router is calibrated. |
| `add(content, title, tag, corpus?)` | Adds a note via `kb add`; `corpus="ai"` for the AI corpus. |
| `supersede(entry_id, replacement)` | Marks an obsolete entry `[SUPERSEDED]` and links its replacement. |
| `history(reference)` | Supersede lineage of one entry as JSON. |

AI entries are shortened in mixed searches because they are ~4× longer than homelab
notes (median 6.8 KB vs 1.7 KB) and rarely what a homelab task needs. On 32 homelab and
12 AI eval queries this cut output by 17% and 88% with no lost hits (KB `homelab:1076`).

## Requirements

- Python 3.13
- Three isolated venvs (no system site-packages):
  - `venv-embed` — ChromaDB, FastEmbed for `compile.py`, watchers, `kb-embed`
  - `venv-search` — Search API + CPU cross-encoder stack
  - `venv` — MCP transports
- ChromaDB at `localhost:8000` (Docker: `kb-chromadb`)
- SQLite databases: `/opt/kb/kb.db` (homelab), `/opt/ai-kb/ai-kb.db` (AI)

## Setup

See `setup/` directory for schema, scripts, and initial configuration.

## What's not in this repo

- `.env` — API keys and bearer tokens (keep local)
- `kb.db` — built from raw notes via `compile.py`
- `chroma_data/` — vector store (derived from `kb.db`)
- `raw/` — source markdown notes (backed up to NAS)
- `venv*/` — Python environments
- `fastembed_cache/` — downloaded model weights

## Eval

The `eval/` directory contains the retrieval evaluation harness: golden sets, rewrite test sets, and historical reports. See also `projects/kb-eval-snapshot/` (separate) for holdout experiments.

## License

Private.
