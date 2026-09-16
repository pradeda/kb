#!/usr/bin/env python3
"""Integration test: the REAL implementation (compile.supersede_entry marker
writer + supersede_index edge/history/rebuild) on ISOLATED temp databases and
temp raw files. Production (/opt/kb/kb.db, /opt/ai-kb/ai-kb.db, /opt/kb/raw)
is never opened for writing — every path is injected, and we assert prod raw is
untouched.

Run: /opt/kb/venv-embed/bin/python3 eval/test_supersede_integration.py
(venv-embed matches the interpreter compile.py runs under; plain python3 also works
since we call only the sqlite/file code paths, not embedding.)"""
import os, sys, shutil, sqlite3, tempfile

# order matters: runtime/ must resolve `import compile` to the EDITED source,
# while supersede_index resolves from /opt/kb (canonical module).
sys.path.insert(0, "/opt/kb")
sys.path.insert(0, "/home/turok/projects/kb-go/runtime")  # source under test (index 0)
import supersede_index as si
import compile as kbc  # runtime/compile.py (edited); its own `import supersede_index` -> /opt/kb
assert "projects/kb-go/runtime" in kbc.__file__, f"wrong compile.py: {kbc.__file__}"

MARK = si.MARKER_PREFIX
PROD_RAW = "/opt/kb/raw"

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


class Bed:
    """Isolated test bed: temp homelab + ai DBs, temp raw tree, injected paths."""
    def __init__(self):
        self.dir = tempfile.mkdtemp(prefix="kb-supersede-it-")
        self.raw = os.path.join(self.dir, "raw", "notes")
        os.makedirs(self.raw, exist_ok=True)
        self.homelab_db = os.path.join(self.dir, "kb.db")
        self.ai_db = os.path.join(self.dir, "ai-kb.db")
        for p in (self.homelab_db, self.ai_db):
            c = sqlite3.connect(p); c.executescript(ENTRIES_SCHEMA); c.commit(); c.close()
        self.db_paths = {"homelab": self.homelab_db, "ai": self.ai_db}
        self.edges_db = self.homelab_db

    def db(self, corpus):
        return self.db_paths[corpus]

    def add(self, corpus, eid, content, title="t", tags="test", date="2026-01-01T00:00:00"):
        raw_file = os.path.join(self.raw, f"{corpus}-{eid}.md")
        with open(raw_file, "w", encoding="utf-8") as f:
            f.write(f"---\ntype: note\ntitle: {title}\ntags: {tags}\nsaved: {date}\n---\n\n{content}")
        c = sqlite3.connect(self.db(corpus))
        c.execute("INSERT INTO entries(id,type,content,title,tags,raw_path,created_at) "
                  "VALUES(?,?,?,?,?,?,?)", (eid, "note", content, title, tags, raw_file, date))
        c.commit(); c.close()

    def content(self, corpus, eid):
        c = sqlite3.connect(self.db(corpus))
        row = c.execute("SELECT content,title,embedded_at FROM entries WHERE id=?", (eid,)).fetchone()
        c.close(); return row

    def supersede(self, corpus, eid, replacement):
        """Mirror compile.main() supersede path with INJECTED paths (no prod)."""
        src = (corpus, eid)
        refs, malformed = si.parse_canonical(f"{MARK}{replacement}")
        if malformed or not refs:
            raise ValueError(f"invalid replacement {replacement!r}")
        errs = si.validate_supersede(self.edges_db, src, refs, self.db_paths)
        if errs:
            raise PermissionError("; ".join(errs))
        kbc.supersede_entry(eid, replacement, db_path=self.db(corpus))  # real marker writer
        si.apply_supersede_edges(self.edges_db, src, refs)             # real edge replace-set
        return refs

    def history(self, corpus, eid, limit=200):
        return si.history_from_stores(self.edges_db, (corpus, eid), self.db_paths, limit=limit)

    def cleanup(self):
        shutil.rmtree(self.dir, ignore_errors=True)


def occurrences(s, sub):
    return s.count(sub)


# ── tests ─────────────────────────────────────────────────────────────────────

def test_supersede_writes_marker_edge_and_history():
    b = Bed()
    try:
        b.add("homelab", 1, "radi na X"); b.add("homelab", 2, "radi na Y (novo)")
        b.supersede("homelab", 1, "homelab:2")
        content, title, emb = b.content("homelab", 1)
        assert content.startswith(f"{MARK}homelab:2"), content[:40]
        assert title.startswith("[SUPERSEDED]")
        assert emb is None  # queued for re-embed
        h = b.history("homelab", 2)
        assert set(h["nodes"]) == {"homelab:1", "homelab:2"}
        assert ["homelab:1", "homelab:2"] in [list(e) for e in h["edges"]]
        assert h["completeness"] == "partial"  # index not rebuilt after this write => stale? see next test
    finally:
        b.cleanup()


def test_rebuild_clears_stale_and_history_complete():
    b = Bed()
    try:
        b.add("homelab", 1, "x"); b.add("homelab", 2, "y")
        b.supersede("homelab", 1, "homelab:2")
        si.rebuild_from_stores(b.edges_db, b.db_paths)   # explicit rebuild => valid
        h = b.history("homelab", 1)
        assert h["index_stale"] is False and h["completeness"] == "complete"
    finally:
        b.cleanup()


def test_resupersede_replaces_edge_set_no_marker_accumulation():
    b = Bed()
    try:
        for i, c in [(1, "x"), (2, "b"), (3, "c")]:
            b.add("homelab", i, c)
        b.supersede("homelab", 1, "homelab:2")
        b.supersede("homelab", 1, "homelab:3")   # re-supersede to a different target
        content, _, _ = b.content("homelab", 1)
        assert content.startswith(f"{MARK}homelab:3"), content[:40]
        assert occurrences(content, MARK) == 1, "canonical marker must not accumulate"
        # edge set replaced: only 1->3 remains
        c = sqlite3.connect(b.edges_db)
        rows = c.execute("SELECT dst_id FROM supersede_edges WHERE src_id=1").fetchall(); c.close()
        assert [r[0] for r in rows] == [3], rows
    finally:
        b.cleanup()


def test_same_replacement_is_idempotent():
    b = Bed()
    try:
        b.add("homelab", 1, "x"); b.add("homelab", 2, "y")
        b.supersede("homelab", 1, "homelab:2")
        c1, _, _ = b.content("homelab", 1)
        b.supersede("homelab", 1, "homelab:2")   # again, same target
        c2, _, _ = b.content("homelab", 1)
        assert c1 == c2, "identical re-supersede must not change content"
    finally:
        b.cleanup()


def test_cross_corpus_supersede_and_history():
    b = Bed()
    try:
        b.add("homelab", 1, "homelab thing"); b.add("ai", 5, "ai replacement")
        b.supersede("homelab", 1, "ai:5")
        si.rebuild_from_stores(b.edges_db, b.db_paths)
        h = b.history("ai", 5)   # from the ai side, find the homelab predecessor
        assert set(h["nodes"]) == {"homelab:1", "ai:5"}
    finally:
        b.cleanup()


def test_cycle_and_self_and_missing_rejected():
    b = Bed()
    try:
        b.add("homelab", 1, "a"); b.add("homelab", 2, "b")
        b.supersede("homelab", 1, "homelab:2")
        # cycle: 2 -> 1 would close a loop
        try:
            b.supersede("homelab", 2, "homelab:1"); assert False, "cycle not rejected"
        except PermissionError as e:
            assert "cycle" in str(e)
        # self
        try:
            b.supersede("homelab", 1, "homelab:1"); assert False, "self not rejected"
        except PermissionError as e:
            assert "self" in str(e)
        # missing target
        try:
            b.supersede("homelab", 2, "homelab:999"); assert False, "missing not rejected"
        except PermissionError as e:
            assert "does not exist" in str(e)
    finally:
        b.cleanup()


def test_recover_marks_stale_partial():
    b = Bed()
    try:
        b.add("homelab", 1, "x"); b.add("homelab", 2, "y")
        b.supersede("homelab", 1, "homelab:2")
        si.rebuild_from_stores(b.edges_db, b.db_paths)      # valid
        si.mark_stale_from_recovery(b.edges_db)             # simulate recover
        h = b.history("homelab", 1)
        assert h["index_stale"] is True and h["completeness"] == "partial"
    finally:
        b.cleanup()


def test_broken_link_partial_after_target_retired():
    b = Bed()
    try:
        b.add("homelab", 1, "x"); b.add("homelab", 2, "y")
        b.supersede("homelab", 1, "homelab:2")
        si.rebuild_from_stores(b.edges_db, b.db_paths)
        # target retired out from under the edge
        c = sqlite3.connect(b.homelab_db); c.execute("DELETE FROM entries WHERE id=2"); c.commit(); c.close()
        h = b.history("homelab", 1)
        assert h["nodes"]["homelab:2"].get("missing") is True
        assert h["completeness"] == "partial" and h["broken_links"]
    finally:
        b.cleanup()


def test_health_mismatch_is_readonly():
    b = Bed()
    try:
        b.add("homelab", 1, f"{MARK}homelab:2\n\nbody"); b.add("homelab", 2, "y")
        si.rebuild_from_stores(b.edges_db, b.db_paths)      # builds table + edge, clears stale
        # simulate drift: an edge the markers imply is missing from the table
        c = sqlite3.connect(b.edges_db); c.execute("DELETE FROM supersede_edges"); c.commit(); c.close()
        before = sqlite3.connect(b.edges_db).execute("SELECT count(*) FROM supersede_edges").fetchone()[0]
        rep = si.health_mismatch(b.edges_db, b.db_paths)     # read-only
        after = sqlite3.connect(b.edges_db).execute("SELECT count(*) FROM supersede_edges").fetchone()[0]
        assert before == after == 0, "health must not write/repair"
        assert ("homelab", 1, "homelab", 2) in [tuple(x) for x in rep["missing_from_table"]]
        assert rep["consistent"] is False
    finally:
        b.cleanup()


def test_production_raw_untouched():
    """The whole suite must never write under /opt/kb/raw."""
    snap = None
    if os.path.isdir(PROD_RAW):
        snap = sorted((f, os.path.getmtime(os.path.join(dp, f)))
                      for dp, _, fs in os.walk(PROD_RAW) for f in fs)
    b = Bed()
    try:
        b.add("homelab", 1, "x"); b.add("homelab", 2, "y")
        b.supersede("homelab", 1, "homelab:2")
    finally:
        b.cleanup()
    if snap is not None:
        now = sorted((f, os.path.getmtime(os.path.join(dp, f)))
                     for dp, _, fs in os.walk(PROD_RAW) for f in fs)
        assert snap == now, "production raw tree changed — isolation breach"


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
