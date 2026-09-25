#!/usr/bin/env python3
"""Prototype (read + rebuild) of the supersede link index. Pure, storage-agnostic
logic so it can be unit-tested on an isolated SQLite before it is ported into the
Go write path (cmdSupersede) and a `kb history` / `kb rebuild-supersede-index` CLI.

Contract (locked with the user):
- Edge source of truth = the CANONICAL supersede marker: the FIRST line of an
  entry's content, exactly `SUPERSEDED — use <corpus:id>[, <corpus:id> ...]`.
  Markers embedded later in preserved history text are NOT edges.
- The edges table is a DERIVED index (single shared table, cross-corpus rows).
  It never removes drift by itself: after any recover/rebuild it must be marked
  stale; `history` then reports completeness "partial", never "complete".
- Re-supersede REPLACES the whole outgoing edge set for a source.
- Refs are strict `^(homelab|ai):\\d+$`. Validation (format, target exists, no
  self-supersede, no cycle) is shared by CLI and MCP write paths.
"""
import re
import sqlite3

MARKER_PREFIX = "SUPERSEDED — use "   # note the em-dash U+2014, matches compile.py
REF_RE = re.compile(r"^(homelab|ai):([0-9]+)$")

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS supersede_edges (
  src_corpus TEXT NOT NULL, src_id INTEGER NOT NULL,
  dst_corpus TEXT NOT NULL, dst_id INTEGER NOT NULL,
  UNIQUE(src_corpus, src_id, dst_corpus, dst_id)
);
CREATE INDEX IF NOT EXISTS idx_se_src ON supersede_edges(src_corpus, src_id);
CREATE INDEX IF NOT EXISTS idx_se_dst ON supersede_edges(dst_corpus, dst_id);
"""

STALE_KEY = "supersede_index_stale"


# ── marker parsing ────────────────────────────────────────────────────────────

def parse_canonical(content):
    """Return (refs, malformed) from the CANONICAL (first) marker line only.
    refs = list of (corpus, id:int); malformed = list of raw tokens that were not
    valid corpus:id. A non-canonical marker (not at the very start) yields ([],[])."""
    if content is None:
        return [], []
    stripped = content.lstrip()
    # canonical marker must be the first thing in the entry
    first_line = stripped.split("\n", 1)[0]
    if not first_line.startswith(MARKER_PREFIX):
        return [], []
    payload = first_line[len(MARKER_PREFIX):].strip()
    refs, malformed = [], []
    for tok in (t.strip() for t in payload.split(",")):
        if not tok:
            continue
        m = REF_RE.match(tok)
        if m:
            refs.append((m.group(1), int(m.group(2))))
        else:
            malformed.append(tok)
    return refs, malformed


# ── schema / staleness ──────────────────────────────────────────────────────

def ensure_schema(conn):
    conn.executescript(SCHEMA_SQL)
    conn.execute("CREATE TABLE IF NOT EXISTS kb_meta (key TEXT PRIMARY KEY, value TEXT)")


def set_stale(conn, stale=True):
    conn.execute(
        "INSERT INTO kb_meta(key,value) VALUES(?,?) "
        "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        (STALE_KEY, "1" if stale else "0"),
    )


def _table_exists(conn, name):
    return conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)).fetchone() is not None


def is_stale(conn):
    # read-only: a missing kb_meta table means the index was never built => stale
    if not _table_exists(conn, "kb_meta"):
        return True
    row = conn.execute("SELECT value FROM kb_meta WHERE key=?", (STALE_KEY,)).fetchone()
    if row is None:
        return True
    return row[0] != "0"


# ── rebuild (scan markers -> edges) ───────────────────────────────────────────

def rebuild_edges(conn, entries, known_ids=None):
    """Rebuild the whole edges table from entry markers.
    entries: iterable of (corpus, id, content). known_ids: optional set of
    (corpus,id) that exist, to flag dangling targets. Returns dict(stats)."""
    ensure_schema(conn)
    conn.execute("DELETE FROM supersede_edges")
    warnings = []
    n_edges = 0
    for corpus, eid, content in entries:
        refs, malformed = parse_canonical(content)
        for tok in malformed:
            warnings.append(f"{corpus}:{eid} malformed ref {tok!r}")
        for dc, di in refs:
            if (dc, di) == (corpus, eid):
                warnings.append(f"{corpus}:{eid} self-supersede ignored")
                continue
            if known_ids is not None and (dc, di) not in known_ids:
                warnings.append(f"{corpus}:{eid} -> {dc}:{di} dangling target")
            conn.execute(
                "INSERT OR IGNORE INTO supersede_edges VALUES(?,?,?,?)",
                (corpus, eid, dc, di),
            )
            n_edges += 1
    set_stale(conn, False)   # explicit rebuild clears staleness
    conn.commit()
    return {"edges": n_edges, "warnings": warnings}


# ── write-side validation + replace-set (contract prototype) ──────────────────

def _reaches(conn, start, target):
    """True if `target` is reachable from `start` following forward edges."""
    seen, stack = set(), [start]
    while stack:
        node = stack.pop()
        if node == target:
            return True
        if node in seen:
            continue
        seen.add(node)
        for row in conn.execute(
            "SELECT dst_corpus,dst_id FROM supersede_edges WHERE src_corpus=? AND src_id=?",
            node,
        ):
            stack.append((row[0], row[1]))
    return False


def validate_write(conn, src, dst_refs, exists):
    """src=(corpus,id); dst_refs=list[(corpus,id)]; exists=callable((corpus,id))->bool.
    Returns list of error strings (empty = ok). Cross-corpus aware."""
    errors = []
    if not dst_refs:
        errors.append("no replacement refs")
    for dst in dst_refs:
        if dst == src:
            errors.append(f"self-supersede {src}")
        if not exists(dst):
            errors.append(f"target {dst[0]}:{dst[1]} does not exist")
        # cycle: would dst already reach src via existing edges?
        if _reaches(conn, dst, src):
            errors.append(f"cycle: {dst[0]}:{dst[1]} already reaches {src[0]}:{src[1]}")
    return errors


def apply_write(conn, src, dst_refs):
    """Replace the ENTIRE outgoing edge set for src (re-supersede semantics)."""
    conn.execute(
        "DELETE FROM supersede_edges WHERE src_corpus=? AND src_id=?", src)
    for dst in dst_refs:
        conn.execute("INSERT OR IGNORE INTO supersede_edges VALUES(?,?,?,?)",
                     (src[0], src[1], dst[0], dst[1]))
    conn.commit()


# ── history (read-only, bidirectional) ────────────────────────────────────────

def history(conn, start, meta, limit=200):
    """Bidirectional traversal from start=(corpus,id). meta=callable((corpus,id))
    -> dict(title,date) or None. Returns {nodes, edges, warnings, truncated,
    index_stale, completeness}. visited-set guards cycles; limit -> truncated."""
    nodes, edges, warnings = {}, [], []
    seen, queue = set(), [start]
    truncated = False
    has_edges = _table_exists(conn, "supersede_edges")  # read-only: never create it here
    while queue:
        node = queue.pop(0)
        if node in seen:
            continue
        if len(seen) >= limit:
            truncated = True
            break
        seen.add(node)
        m = meta(node)
        key = f"{node[0]}:{node[1]}"
        if m is None:
            nodes[key] = {"title": None, "date": None, "missing": True}
            warnings.append(f"{key} referenced but not found")
        else:
            nodes[key] = {"title": m.get("title"), "date": m.get("date")}
        if not has_edges:
            continue
        # forward (successors) and backward (predecessors)
        for row in conn.execute(
            "SELECT dst_corpus,dst_id FROM supersede_edges WHERE src_corpus=? AND src_id=?", node):
            nxt = (row[0], row[1]); edges.append((key, f"{nxt[0]}:{nxt[1]}")); queue.append(nxt)
        for row in conn.execute(
            "SELECT src_corpus,src_id FROM supersede_edges WHERE dst_corpus=? AND dst_id=?", node):
            prv = (row[0], row[1]); edges.append((f"{prv[0]}:{prv[1]}", key)); queue.append(prv)

    stale = is_stale(conn)
    if stale:
        warnings.append("index stale — run kb rebuild-supersede-index")
    # dedupe edges preserving order
    seen_e, uniq = set(), []
    for e in edges:
        if e not in seen_e:
            seen_e.add(e); uniq.append(e)
    # broken links: an edge whose target entry is missing. This makes the
    # returned history NOT the whole connected history -> completeness partial.
    broken_links = [[s, d] for (s, d) in uniq if nodes.get(d, {}).get("missing")]
    complete = (not stale) and (not truncated) and (not broken_links)
    return {
        "start": f"{start[0]}:{start[1]}",
        "nodes": nodes,
        "edges": uniq,
        "chain": linear_chain_order(uniq, nodes),  # oldest→newest, or None if branch/merge
        "broken_links": broken_links,
        "warnings": warnings,
        "truncated": truncated,
        "index_stale": stale,
        "completeness": "complete" if complete else "partial",
    }


def linear_chain_order(edges, nodes):
    """If the edge set forms ONE simple chain (each node <=1 in and <=1 out, no
    branch/merge/cycle), return node keys oldest→newest. Otherwise None — caller
    then shows the real edges instead of faking a sequence."""
    if not edges:
        return list(nodes) if len(nodes) <= 1 else None
    succ, pred, indeg, outdeg = {}, {}, {}, {}
    keys = set(nodes)
    for s, d in edges:
        if outdeg.get(s, 0) or indeg.get(d, 0):
            return None  # a node with 2 out or 2 in -> branch/merge
        succ[s] = d; pred[d] = s
        outdeg[s] = outdeg.get(s, 0) + 1
        indeg[d] = indeg.get(d, 0) + 1
        keys.add(s); keys.add(d)
    heads = [k for k in keys if k not in pred]
    if len(heads) != 1:
        return None  # not a single chain (cycle or disjoint)
    order, node, seen = [], heads[0], set()
    while node is not None:
        if node in seen:
            return None  # cycle guard
        seen.add(node); order.append(node); node = succ.get(node)
    return order if len(order) == len(keys) else None


def render_history_text(result):
    """Human-readable rendering of a history() result. A linear chain prints
    oldest→newest with the newest marked current; a branch/merge prints the raw
    edges instead of faking a sequence. Broken links, truncation and staleness
    are always surfaced so a partial answer can never read as whole."""
    nodes = result["nodes"]

    def label(key):
        n = nodes.get(key, {})
        if n.get("missing"):
            return f"{key}  (missing)"
        title = (n.get("title") or "").strip()
        return f"{key}  {title}".rstrip()

    lines = [f"History for {result['start']}  [{result['completeness']}]"]
    chain = result.get("chain")
    if chain:
        lines.append("Chain (oldest → newest):")
        for i, key in enumerate(chain):
            lines.append(f"  {label(key)}" + ("  ← current" if i == len(chain) - 1 else ""))
    elif result["edges"]:
        lines.append("Edges (branch/merge — not a single chain):")
        lines += [f"  {s} → {d}" for s, d in result["edges"]]
    else:
        lines.append("  (no linked predecessors or successors)")
    if result.get("broken_links"):
        lines.append("Broken links (target missing):")
        lines += [f"  {s} → {d}" for s, d in result["broken_links"]]
    if result.get("truncated"):
        lines.append("! truncated at traversal limit — history incomplete")
    for w in result.get("warnings", []):
        lines.append(f"! {w}")
    return "\n".join(lines)


# ── real storage adapters ─────────────────────────────────────────────────────
# The functions above are storage-agnostic (they take a connection + callables).
# The adapters below bind them to the real two-DB layout, but take EVERY path
# explicitly (both corpus DBs AND the edges DB) so an isolated test can never
# fall through to a production database.


def _iter_all_entries(db_paths):
    for corpus, path in db_paths.items():
        conn = sqlite3.connect(path)
        try:
            for row in conn.execute("SELECT id, content FROM entries"):
                yield (corpus, row[0], row[1])
        finally:
            conn.close()


def _known_ids(db_paths):
    ids = set()
    for corpus, path in db_paths.items():
        conn = sqlite3.connect(path)
        try:
            for (i,) in conn.execute("SELECT id FROM entries"):
                ids.add((corpus, i))
        finally:
            conn.close()
    return ids


def _make_meta(db_paths):
    def meta(node):
        corpus, eid = node
        path = db_paths.get(corpus)
        if not path:
            return None
        conn = sqlite3.connect(path)
        try:
            row = conn.execute(
                "SELECT title, created_at FROM entries WHERE id=?", (eid,)).fetchone()
        finally:
            conn.close()
        return None if row is None else {"title": row[0], "date": row[1]}
    return meta


def rebuild_from_stores(edges_db, db_paths):
    """Rebuild the shared edge index from markers in ALL corpus DBs. Marks the
    index valid (not stale) only on successful completion."""
    conn = sqlite3.connect(edges_db)
    try:
        return rebuild_edges(conn, _iter_all_entries(db_paths), known_ids=_known_ids(db_paths))
    finally:
        conn.close()


def history_from_stores(edges_db, start, db_paths, limit=200):
    conn = sqlite3.connect(edges_db)
    try:
        return history(conn, start, _make_meta(db_paths), limit=limit)
    finally:
        conn.close()


def validate_supersede(edges_db, src, dst_refs, db_paths):
    """Validate only (no write): existence across BOTH corpora, no self, no cycle."""
    ids = _known_ids(db_paths)
    exists = lambda n: n in ids
    conn = sqlite3.connect(edges_db)
    try:
        ensure_schema(conn)
        return validate_write(conn, src, dst_refs, exists)
    finally:
        conn.close()


def apply_supersede_edges(edges_db, src, dst_refs):
    """Apply only (replace the outgoing edge set for src). Assumes validation
    already passed; call AFTER the marker write so a rejected supersede never
    marks the entry."""
    conn = sqlite3.connect(edges_db)
    try:
        ensure_schema(conn)
        apply_write(conn, src, dst_refs)
    finally:
        conn.close()


def mark_stale_from_recovery(edges_db):
    """Called by recover-db / recover-raw: the entries changed under the index,
    so it can no longer be trusted until an explicit rebuild."""
    conn = sqlite3.connect(edges_db)
    try:
        ensure_schema(conn)
        set_stale(conn, True)
        conn.commit()
    finally:
        conn.close()


def health_mismatch(edges_db, db_paths):
    """READ-ONLY: compare the edge table against what the markers imply. Returns
    a dict of discrepancies (missing/extra edges, stale flag). Never writes."""
    conn = sqlite3.connect(edges_db)
    try:
        # read-only: do NOT create the table; a missing table means no edges yet
        if _table_exists(conn, "supersede_edges"):
            have = {(r[0], r[1], r[2], r[3]) for r in
                    conn.execute("SELECT src_corpus,src_id,dst_corpus,dst_id FROM supersede_edges")}
        else:
            have = set()
        stale = is_stale(conn)
    finally:
        conn.close()
    want = set()
    for corpus, eid, content in _iter_all_entries(db_paths):
        refs, _ = parse_canonical(content)
        for dc, di in refs:
            if (dc, di) != (corpus, eid):
                want.add((corpus, eid, dc, di))
    return {
        "stale": stale,
        "missing_from_table": sorted(want - have),   # marker says edge, table lacks it
        "extra_in_table": sorted(have - want),        # table has edge, no marker
        "consistent": (want == have) and not stale,
    }
