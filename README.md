# kb

Homelab knowledge base search backend — Python API, MCP server, ChromaDB embed daemon, FTS5 hybrid retrieval, and mMARCO cross-encoder reranking.

> **CLI frontend:** [kb-go](https://github.com/pradeda/kb-go) — Go CLI (`kb ask`, `kb add`, `kb search`, `kb retire`) that calls this backend.

## Components

| Component | File | Port/Socket |
|-----------|------|-------------|
| Search API (v2) | `kb_v2.py` via `kb_search_api.py` | `:8050` (user unit: `kb-search-api`) |
| MCP server | `mcp_server.py` | system units `kb-mcp-sse` / `kb-mcp-http`; stdio clients |
| FastEmbed daemon | `embed_daemon.py` | `/run/kb-embed/embed.sock` |
| Compiler | `compile.py` | invoked by `kb-watcher` / `ai-kb-watcher` |
| Corpus router | `corpus-router.yml` | config for search API |

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
