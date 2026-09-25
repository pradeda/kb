"""Canonical corpus identity for the KB stack.

Every Python consumer reads its corpus paths and collection names from here:
compile.py (and through it the watcher units, the Go CLI's compile subprocesses,
refresh_volatile.sh and the health wrapper), kb_atlas.py, provision_storage.py,
kb_v2.py / kb_search_api.py, mcp_server.py and gate.py. Adding a corpus or moving a
path is one edit instead of the six to nine copies the review found.

The Go CLI cannot import Python and keeps its own table in corpus.go; the two are bound
by tests/test_corpus_identity.py, so the copies cannot drift apart unnoticed - the same
arrangement the isolation variable list now has.

Deployed with the rest of runtime/ (`make install` -> /opt/kb/corpora.py); the deployed
services import it by module name, which resolves because systemd runs them with
/opt/kb as the working directory and Python puts the script's directory on sys.path.
"""
from __future__ import annotations

import copy

CORPORA: dict[str, dict[str, str]] = {
    "homelab": {
        "root": "/opt/kb",
        "db": "/opt/kb/kb.db",
        "raw": "/opt/kb/raw",
        "env": "/opt/kb/.env",
        "collection": "kb_collection",
        "wiki_index": "/opt/kb/wiki/index.md",
        "secret_patterns": "/opt/kb/secret_patterns.json",
        "quarantine_dir": "/opt/kb/quarantine",
        "quarantine_log": "/opt/kb/quarantine.log",
        "watcher_lock": "/tmp/kb-watcher.lock",
        "watcher_state": "/tmp/kb-watcher-last",
    },
    "ai": {
        "root": "/opt/ai-kb",
        "db": "/opt/ai-kb/ai-kb.db",
        "raw": "/opt/ai-kb/raw",
        "env": "/opt/ai-kb/.env",
        "collection": "ai_kb_collection",
        "wiki_index": "",
        "secret_patterns": "/opt/ai-kb/secret_patterns.json",
        "quarantine_dir": "/opt/ai-kb/quarantine",
        "quarantine_log": "/opt/ai-kb/quarantine.log",
        "watcher_lock": "/tmp/ai-kb-watcher.lock",
        "watcher_state": "/tmp/ai-kb-watcher-last",
    },
}


def profiles() -> dict[str, dict[str, str]]:
    """A deep copy for callers that mutate their copy (compile.py's isolation override)."""
    return copy.deepcopy(CORPORA)


def databases() -> dict[str, str]:
    """corpus name -> SQLite path, for readers that only need the database."""
    return {name: profile["db"] for name, profile in CORPORA.items()}


def collections() -> dict[str, str]:
    """corpus name -> Chroma collection name."""
    return {name: profile["collection"] for name, profile in CORPORA.items()}
