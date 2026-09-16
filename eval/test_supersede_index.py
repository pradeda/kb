#!/usr/bin/env python3
"""Contract tests for the supersede link index prototype, on an ISOLATED
in-memory SQLite (never touches production). Covers the locked scenarios:
standalone / 2-version / 7-chain (enter start/mid/end) / branch / merge /
cycle / missing+invalid ref / historical-text-not-an-edge / limit-truncated /
re-supersede replaces set / recovery->stale->not-complete / cross-corpus.
Run: python3 eval/test_supersede_index.py"""
import os, sqlite3, sys
sys.path.insert(0, "/home/turok/projects/kb-go/runtime")  # canonical source (deployed to /opt/kb via make install)
import supersede_index as si

MARK = si.MARKER_PREFIX  # "SUPERSEDED — use "


def fresh():
    conn = sqlite3.connect(":memory:")
    si.ensure_schema(conn)
    return conn


def mk_entries(spec):
    """spec: dict {(corpus,id): content}. Returns (entries_iter, meta_fn, exists_fn)."""
    def meta(node):
        return {"title": f"entry {node[0]}:{node[1]}", "date": "2026-01-01"} if node in spec else None
    def exists(node):
        return node in spec
    entries = [(c, i, content) for (c, i), content in spec.items()]
    known = set(spec.keys())
    return entries, meta, exists, known


def build(conn, spec):
    entries, meta, exists, known = mk_entries(spec)
    stats = si.rebuild_edges(conn, entries, known_ids=known)
    return meta, exists, known, stats


# ── scenarios ─────────────────────────────────────────────────────────────────

def test_standalone_no_edges():
    conn = fresh()
    meta, *_ = build(conn, {("homelab", 1): "just a normal entry"})
    h = si.history(conn, ("homelab", 1), meta)
    assert h["edges"] == [] and list(h["nodes"]) == ["homelab:1"]
    assert h["completeness"] == "complete"


def test_two_version_both_entrypoints():
    conn = fresh()
    spec = {("homelab", 1): f"{MARK}homelab:2\n\nold body", ("homelab", 2): "current"}
    meta, *_ = build(conn, spec)
    for start in [("homelab", 1), ("homelab", 2)]:
        h = si.history(conn, start, meta)
        assert set(h["nodes"]) == {"homelab:1", "homelab:2"}, start
        assert ("homelab:1", "homelab:2") in h["edges"]


def test_seven_chain_same_set_from_any_entrypoint():
    conn = fresh()
    spec = {}
    for i in range(1, 7):  # 1->2->...->7
        spec[("homelab", i)] = f"{MARK}homelab:{i+1}\n\nbody"
    spec[("homelab", 7)] = "head, current"
    meta, *_ = build(conn, spec)
    expected_nodes = {f"homelab:{i}" for i in range(1, 8)}
    for start in [("homelab", 1), ("homelab", 4), ("homelab", 7)]:
        h = si.history(conn, start, meta)
        assert set(h["nodes"]) == expected_nodes, f"from {start}: {set(h['nodes'])}"
        assert len(h["edges"]) == 6


def test_branch_fanout():
    conn = fresh()
    spec = {("homelab", 1): f"{MARK}homelab:2, homelab:3\n\nbody",
            ("homelab", 2): "b", ("homelab", 3): "c"}
    meta, *_ = build(conn, spec)
    h = si.history(conn, ("homelab", 2), meta)
    assert set(h["nodes"]) == {"homelab:1", "homelab:2", "homelab:3"}
    assert ("homelab:1", "homelab:2") in h["edges"] and ("homelab:1", "homelab:3") in h["edges"]


def test_merge_fanin():
    conn = fresh()
    spec = {("homelab", 1): f"{MARK}homelab:3\n\nb", ("homelab", 2): f"{MARK}homelab:3\n\nb",
            ("homelab", 3): "current"}
    meta, *_ = build(conn, spec)
    h = si.history(conn, ("homelab", 3), meta)
    assert set(h["nodes"]) == {"homelab:1", "homelab:2", "homelab:3"}


def test_cycle_no_infinite_loop():
    conn = fresh()
    spec = {("homelab", 1): f"{MARK}homelab:2\n\nb", ("homelab", 2): f"{MARK}homelab:1\n\nb"}
    meta, *_ = build(conn, spec)
    h = si.history(conn, ("homelab", 1), meta)  # must terminate
    assert set(h["nodes"]) == {"homelab:1", "homelab:2"}


def test_invalid_and_dangling_refs():
    conn = fresh()
    # malformed ref "see 323" (not corpus:id) + dangling homelab:999
    spec = {("homelab", 1): f"{MARK}see 323, homelab:999\n\nb"}
    _, _, known, stats = build(conn, {("homelab", 1): spec[("homelab", 1)]})
    assert any("malformed" in w for w in stats["warnings"])
    assert any("dangling" in w for w in stats["warnings"])


def test_historical_marker_not_an_edge():
    conn = fresh()
    # canonical marker points to :2; a SECOND marker deep in preserved history
    # points to :999 and must NOT become an edge
    content = (f"{MARK}homelab:2\n\n--- Historical incident content ---\n\n"
               f"{MARK}homelab:999\n\nold stuff")
    spec = {("homelab", 1): content, ("homelab", 2): "current"}
    meta, _, known, stats = build(conn, spec)
    refs, _ = si.parse_canonical(content)
    assert refs == [("homelab", 2)], refs
    h = si.history(conn, ("homelab", 1), meta)
    assert "homelab:999" not in h["nodes"]


def test_limit_truncated_is_partial():
    conn = fresh()
    spec = {}
    for i in range(1, 11):
        spec[("homelab", i)] = f"{MARK}homelab:{i+1}\n\nb"
    spec[("homelab", 11)] = "head"
    meta, *_ = build(conn, spec)
    h = si.history(conn, ("homelab", 1), meta, limit=4)
    assert h["truncated"] is True
    assert h["completeness"] == "partial"


def test_resupersede_replaces_edge_set():
    conn = fresh()
    spec = {("homelab", 1): "x", ("homelab", 2): "b", ("homelab", 3): "c"}
    _, exists, *_ = build(conn, spec)
    si.apply_write(conn, ("homelab", 1), [("homelab", 2)])          # A -> B
    si.apply_write(conn, ("homelab", 1), [("homelab", 3)])          # re-supersede A -> C
    rows = conn.execute("SELECT dst_id FROM supersede_edges WHERE src_id=1").fetchall()
    assert [r[0] for r in rows] == [3], rows   # B edge gone, only C


def test_recovery_marks_stale_history_not_complete():
    conn = fresh()
    spec = {("homelab", 1): f"{MARK}homelab:2\n\nb", ("homelab", 2): "cur"}
    meta, *_ = build(conn, spec)                    # rebuild clears stale
    si.set_stale(conn, True)                        # simulate a recover/rebuild of entries
    h = si.history(conn, ("homelab", 1), meta)
    assert h["index_stale"] is True
    assert h["completeness"] == "partial"           # never "complete" on stale index
    assert any("stale" in w for w in h["warnings"])


def test_cross_corpus_edge_and_cycle():
    conn = fresh()
    spec = {("homelab", 1): f"{MARK}ai:5\n\nb", ("ai", 5): "current ai entry"}
    meta, exists, known, _ = build(conn, spec)
    # traversal from the ai side finds the homelab predecessor (cross-corpus, backward)
    h = si.history(conn, ("ai", 5), meta)
    assert set(h["nodes"]) == {"homelab:1", "ai:5"}
    # cycle-check must see cross-corpus: ai:5 -> homelab:1 would close a cycle
    errs = si.validate_write(conn, ("ai", 5), [("homelab", 1)], exists)
    assert any("cycle" in e for e in errs), errs


def test_broken_link_history_is_partial():
    conn = fresh()
    # 201 -> 999 where 999 does not exist
    spec = {("homelab", 201): f"{MARK}homelab:999\n\nb"}
    meta, *_ = build(conn, spec)
    h = si.history(conn, ("homelab", 201), meta)
    assert h["nodes"]["homelab:999"].get("missing") is True
    assert ["homelab:201", "homelab:999"] in h["broken_links"]
    assert h["completeness"] == "partial"          # missing target -> not whole history
    assert h["index_stale"] is False               # index trust is a separate axis


def test_linear_chain_order_and_branch_returns_none():
    # a clean 3-chain 1->2->3 orders oldest→newest
    edges = [("homelab:1", "homelab:2"), ("homelab:2", "homelab:3")]
    nodes = {f"homelab:{i}": {} for i in (1, 2, 3)}
    assert si.linear_chain_order(edges, nodes) == ["homelab:1", "homelab:2", "homelab:3"]
    # a branch (1->2, 1->3) is not a chain -> None (caller shows real edges)
    b_edges = [("homelab:1", "homelab:2"), ("homelab:1", "homelab:3")]
    b_nodes = {f"homelab:{i}": {} for i in (1, 2, 3)}
    assert si.linear_chain_order(b_edges, b_nodes) is None


def test_render_history_text_linear_and_branch():
    conn = fresh()
    # linear 1->2->3: readable chain oldest→newest, newest marked current
    spec = {("homelab", 1): f"{MARK}homelab:2\n\nb", ("homelab", 2): f"{MARK}homelab:3\n\nb",
            ("homelab", 3): "current"}
    meta, *_ = build(conn, spec)
    txt = si.render_history_text(si.history(conn, ("homelab", 1), meta))
    assert "Chain (oldest → newest):" in txt
    assert txt.index("homelab:1") < txt.index("homelab:2") < txt.index("homelab:3")
    assert "← current" in txt.split("homelab:3", 1)[1].split("\n", 1)[0]
    # branch 1->2, 1->3: not a chain -> raw edges, no "Chain"
    conn2 = fresh()
    spec2 = {("homelab", 1): f"{MARK}homelab:2, homelab:3\n\nb", ("homelab", 2): "b", ("homelab", 3): "c"}
    meta2, *_ = build(conn2, spec2)
    txt2 = si.render_history_text(si.history(conn2, ("homelab", 1), meta2))
    assert "Chain (oldest → newest):" not in txt2
    assert "Edges (branch/merge" in txt2


def test_render_history_text_surfaces_partial_warnings():
    conn = fresh()
    spec = {("homelab", 201): f"{MARK}homelab:999\n\nb"}  # dangling target
    meta, *_ = build(conn, spec)
    txt = si.render_history_text(si.history(conn, ("homelab", 201), meta))
    assert "[partial]" in txt
    assert "Broken links" in txt and "homelab:999" in txt


def test_validate_write_rules():
    conn = fresh()
    spec = {("homelab", 1): "a", ("homelab", 2): "b"}
    _, exists, *_ = build(conn, spec)
    assert si.validate_write(conn, ("homelab", 1), [("homelab", 1)], exists)  # self
    assert si.validate_write(conn, ("homelab", 1), [("homelab", 999)], exists)  # missing
    assert si.validate_write(conn, ("homelab", 1), [], exists)  # empty
    assert si.validate_write(conn, ("homelab", 1), [("homelab", 2)], exists) == []  # ok


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
