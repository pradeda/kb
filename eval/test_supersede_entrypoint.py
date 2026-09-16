#!/usr/bin/env python3
"""Entry-point test (locked Condition 3): drive the LOCALLY BUILT `kb` binary
against ISOLATED temp databases and assert its real CLI contract — argument
passing, JSON output, and failed exit status — without deploying anything and
without touching production storage.

The binary reaches the EDITED source, not the deployed install, via two env
overrides read by corpus.go compileArgv:
  KB_COMPILE_PYTHON -> the venv interpreter
  KB_COMPILE_PY     -> ~/projects/kb-go/runtime/compile.py (whose own
                       `import supersede_index` resolves to runtime/ too)
Data isolation uses the all-or-nothing set (KB_HOMELAB_DB, KB_AI_DB,
KB_HOMELAB_RAW, KB_AI_RAW) honored identically by corpus.go and compile.py.

Only `history` and `rebuild-supersede-index` are exercised: both return before
compile.py's embed block, so no ChromaDB / embed socket is ever contacted. We do
NOT run `kb supersede` here — it falls through to embedding, which would reach
production Chroma.

Run: /opt/kb/venv-embed/bin/python3 eval/test_supersede_entrypoint.py
(needs `go` on PATH to build the binary once).
"""
import json, os, shutil, sqlite3, subprocess, sys, tempfile

REPO = "/home/turok/projects/kb-go"
RUNTIME_COMPILE = os.path.join(REPO, "runtime", "compile.py")
PYTHON = "/opt/kb/venv-embed/bin/python3"
MARK = "SUPERSEDED — use "

ENTRIES_SCHEMA = """
CREATE TABLE entries (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  type TEXT NOT NULL DEFAULT 'note',
  content TEXT NOT NULL,
  title TEXT DEFAULT '',
  tags TEXT DEFAULT '',
  raw_path TEXT DEFAULT '',
  source TEXT DEFAULT 'test',
  compiled_at DATETIME,
  embedded_at DATETIME,
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
"""

_BINARY = None  # built once


def build_binary():
    global _BINARY
    if _BINARY:
        return _BINARY
    d = tempfile.mkdtemp(prefix="kb-entrypoint-bin-")
    out = os.path.join(d, "kb")
    subprocess.run(["go", "build", "-tags", "fts5", "-o", out, "."],
                   cwd=REPO, check=True)
    _BINARY = out
    return out


class Bed:
    """Isolated temp homelab + ai DBs and a temp raw tree. Injected purely via
    the env-override contract — production paths are never referenced."""
    def __init__(self):
        self.dir = tempfile.mkdtemp(prefix="kb-entrypoint-")
        self.homelab_raw = os.path.join(self.dir, "homelab-raw")
        self.ai_raw = os.path.join(self.dir, "ai-raw")
        os.makedirs(self.homelab_raw); os.makedirs(self.ai_raw)
        self.homelab_db = os.path.join(self.dir, "kb.db")
        self.ai_db = os.path.join(self.dir, "ai-kb.db")
        for p in (self.homelab_db, self.ai_db):
            c = sqlite3.connect(p); c.executescript(ENTRIES_SCHEMA); c.commit(); c.close()

    def add(self, corpus, eid, content, title="t"):
        c = sqlite3.connect(self.homelab_db if corpus == "homelab" else self.ai_db)
        c.execute("INSERT INTO entries(id,type,content,title,created_at) VALUES(?,?,?,?,?)",
                  (eid, "note", content, title, "2026-01-01T00:00:00"))
        c.commit(); c.close()

    def env(self, complete=True):
        e = dict(os.environ)
        e["KB_COMPILE_PYTHON"] = PYTHON
        e["KB_COMPILE_PY"] = RUNTIME_COMPILE
        e["KB_HOMELAB_DB"] = self.homelab_db
        e["KB_HOMELAB_RAW"] = self.homelab_raw
        if complete:
            e["KB_AI_DB"] = self.ai_db
            e["KB_AI_RAW"] = self.ai_raw
        else:
            e.pop("KB_AI_DB", None); e.pop("KB_AI_RAW", None)
        return e

    def run(self, *args, complete=True):
        return subprocess.run([build_binary(), *args], env=self.env(complete),
                              capture_output=True, text=True)

    def cleanup(self):
        shutil.rmtree(self.dir, ignore_errors=True)


# ── tests ──────────────────────────────────────────────────────────────────

def test_rebuild_exit0_and_json():
    b = Bed()
    try:
        b.add("homelab", 1, f"{MARK}homelab:2\n\nold"); b.add("homelab", 2, "current")
        r = b.run("rebuild-supersede-index")
        assert r.returncode == 0, r.stderr
        out = json.loads(r.stdout)                     # piped stdout -> JSON
        assert out["rebuild_supersede_index"]["edges"] == 1, out
    finally:
        b.cleanup()


def test_history_json_after_rebuild_is_complete():
    b = Bed()
    try:
        b.add("homelab", 1, f"{MARK}homelab:2\n\nold"); b.add("homelab", 2, "current")
        assert b.run("rebuild-supersede-index").returncode == 0
        r = b.run("history", "homelab:1")
        assert r.returncode == 0, r.stderr
        h = json.loads(r.stdout)                       # non-terminal -> --json auto-added
        assert set(h["nodes"]) == {"homelab:1", "homelab:2"}, h
        assert ["homelab:1", "homelab:2"] in [list(e) for e in h["edges"]]
        assert h["completeness"] == "complete", h
        assert h["chain"] == ["homelab:1", "homelab:2"], h
    finally:
        b.cleanup()


def test_history_before_rebuild_is_partial_stale():
    b = Bed()
    try:
        b.add("homelab", 1, f"{MARK}homelab:2\n\nold"); b.add("homelab", 2, "current")
        r = b.run("history", "homelab:1")              # index never built
        assert r.returncode == 0, r.stderr
        h = json.loads(r.stdout)
        assert h["index_stale"] is True and h["completeness"] == "partial", h
    finally:
        b.cleanup()


def test_cross_corpus_history_reads_ai_db():
    b = Bed()
    try:
        b.add("homelab", 1, f"{MARK}ai:5\n\nold"); b.add("ai", 5, "ai current")
        assert b.run("rebuild-supersede-index").returncode == 0
        h = json.loads(b.run("history", "ai:5").stdout)   # from ai side, find homelab predecessor
        assert set(h["nodes"]) == {"homelab:1", "ai:5"}, h
    finally:
        b.cleanup()


def test_bad_ref_fails_nonzero():
    b = Bed()
    try:
        b.add("homelab", 1, "x")
        for bad in ("garbage", "homelab:", "42", "other:7"):
            r = b.run("history", bad)
            assert r.returncode != 0, f"{bad!r} should fail: {r.stdout}"
    finally:
        b.cleanup()


def test_missing_arg_fails_nonzero():
    b = Bed()
    try:
        r = b.run("history")
        assert r.returncode != 0 and "Usage" in r.stderr, (r.returncode, r.stderr)
    finally:
        b.cleanup()


def test_incomplete_isolation_env_refuses_no_fallback():
    b = Bed()
    try:
        b.add("homelab", 1, "x")
        r = b.run("rebuild-supersede-index", complete=False)   # KB_AI_* missing
        assert r.returncode != 0, r.stdout
        assert "incomplete KB isolation env" in r.stderr, r.stderr
    finally:
        b.cleanup()


def test_readable_output_when_forced_non_json_shape():
    """Sanity: the JSON path carries a `chain`; the readable renderer (exercised
    directly in unit tests) is what a TTY would print. Here we only confirm the
    piped default is valid JSON, not the human text."""
    b = Bed()
    try:
        b.add("homelab", 1, f"{MARK}homelab:2\n\no"); b.add("homelab", 2, "c")
        b.run("rebuild-supersede-index")
        r = b.run("history", "homelab:2")
        json.loads(r.stdout)  # must parse
        assert r.stdout.lstrip().startswith("{"), r.stdout[:40]
    finally:
        b.cleanup()


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn(); print(f"PASS {fn.__name__}")
        except AssertionError as e:
            failed += 1; print(f"FAIL {fn.__name__}: {e}")
        except Exception as e:
            failed += 1; print(f"ERROR {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(fns)-failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
