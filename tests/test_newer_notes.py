"""Newer-note ("updates") edges: detection, filters, freshness and the API surface.

A note that says it UPDATES/CORRECTS/EXTENDS another one is not a supersede: only
part of the older note changed, so nothing is demoted or reordered — the search
result carries one advisory line naming the newer note (homelab:1109).

Run from /opt/kb:  /opt/kb/venv-search/bin/python3 -m unittest tests.test_newer_notes
"""
from __future__ import annotations

import os
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient

import kb_v2
import newer_notes as nn
from kb_v2 import CorpusHealthV2, create_v2_app


KNOWN = {
    ("homelab", 963), ("homelab", 964), ("homelab", 965), ("homelab", 966),
    ("homelab", 992), ("homelab", 995), ("homelab", 996), ("homelab", 997),
    ("homelab", 724), ("homelab", 725), ("homelab", 726), ("homelab", 1082),
    ("homelab", 413), ("homelab", 550), ("homelab", 780), ("homelab", 882),
    ("homelab", 529), ("homelab", 1), ("homelab", 12), ("homelab", 16),
    ("homelab", 100), ("homelab", 940), ("homelab", 449), ("homelab", 562),
    ("homelab", 1099), ("ai", 5),
}


def relations(title: str, content: str = "", corpus: str = "homelab", entry_id: int = 5000):
    return nn.detect_relations(corpus, entry_id, title, content, KNOWN)


def targets(title: str, content: str = ""):
    return {target for target, _relation, _where, _form in relations(title, content)}


class DetectionTests(unittest.TestCase):
    """The shapes the corpus actually uses, and the ones that must stay silent."""

    def test_corpus_shapes_are_edges(self):
        cases = {
            "Forrix Qwen per-model load defaults (updates homelab:963)": {("homelab", 963)},
            "Pi-hole game blocking (extends #724, #725)": {("homelab", 724), ("homelab", 725)},
            "AI Ingest user agents now ai-ingest/0.14 — corrects stale UA refs in homelab:882 and 529":
                {("homelab", 882), ("homelab", 529)},
            "UpSnap NET_RAW error (supersedes homelab:562)": {("homelab", 562)},
            "Ispravka KB #449 — enrichment": {("homelab", 449)},
            "Procedure (supersedes the old-project note KB 413 for this repo)": {("homelab", 413)},
            "Zamenjuje ogradu o qlang-u iz unosa 550": {("homelab", 550)},
            "Junie DeepSeek profile: correction to entry 780": {("homelab", 780)},
            "OMP collab backlog drop — Corrects [[homelab:1099]]": {("homelab", 1099)},
        }
        for title, expected in cases.items():
            with self.subTest(title=title):
                self.assertEqual(targets(title), expected)

    def test_the_first_paragraph_of_the_content_counts_too(self):
        content = "Nastavak homelab:1082 (1-hop A/B na autorskom setu).\n\nDrugo poglavlje.\n"
        found = relations("KB 1-hop ekspanzija", content)
        self.assertEqual(
            [(target, relation, where) for target, relation, where, _form in found],
            [(("homelab", 1082), "continues", "content")],
        )

    def test_a_marker_below_the_first_paragraph_is_not_scanned(self):
        content = "Uvod bez referenci.\n\n" + "x" * 800 + "\n\nupdates homelab:963\n"
        self.assertEqual(relations("Note", content), [])

    def test_serbian_change_verbs(self):
        cases = {
            "Ispravlja unos 550": {("homelab", 550)},
            "Dopunjuje homelab:963": {("homelab", 963)},
            "Menja pravila iz unosa 780": {("homelab", 780)},
            "Nastavak na KB #965": {("homelab", 965)},
            "Zamenjuje homelab:724": {("homelab", 724)},
        }
        for title, expected in cases.items():
            with self.subTest(title=title):
                self.assertEqual(targets(title), expected)

    def test_cross_corpus_reference(self):
        found = nn.detect_relations(
            "ai", 5, "Update to the homelab Qwen setup (updates homelab:963)", "", KNOWN
        )
        self.assertEqual([target for target, *_ in found], [("homelab", 963)])

    def test_plain_mentions_are_not_edges(self):
        cases = [
            "vidi homelab:963",
            "Related: homelab:963",
            "Povezano sa homelab:963",
            "Reference list: homelab:963, homelab:965, homelab:966",
            "homelab:963 — the older note",
            "See also KB 413 and unos 550",
            "## Related\n\nhomelab:963, #724",
        ]
        for title in cases:
            with self.subTest(title=title):
                self.assertEqual(targets(title), set(), title)

    def test_bare_numbers_that_are_not_references(self):
        """Regressions: the first version of the detector read these as edges."""
        cases = [
            "CORRECTION — this entry corrects section 1 of the union plan",
            "Codex entity 9 revision 11 was superseded for display by revision 12",
            "Omar node state — RAM corrected (16 GB, 3 GiB iGPU carve-out)",
            "Lineage lookup on top of the action shipped 2026-09-15 (see KB note, id 940)",
        ]
        for title in cases:
            with self.subTest(title=title):
                self.assertEqual(targets(title), set(), title)

    def test_hash_reference_must_exist_in_the_notes_own_corpus(self):
        # ai:5 exists, homelab:5 does not: a homelab note writing #5 is not an edge.
        self.assertEqual(targets("extends #5"), set())
        self.assertEqual(targets("extends #724"), {("homelab", 724)})

    def test_upstream_issue_numbers_are_not_references(self):
        self.assertEqual(targets("fixes upstream bug #36403 (extends #724)"), {("homelab", 724)})

    def test_self_reference_is_ignored(self):
        self.assertEqual(relations("updates homelab:963", "", entry_id=963), [])

    def test_a_negated_verb_is_not_a_relation(self):
        self.assertEqual(targets("This note does not update homelab:963"), set())

    def test_a_verb_in_another_sentence_is_not_a_relation(self):
        cases = [
            "Updates were deployed. See homelab:963.",
            "The update shipped. homelab:963 describes it.",
        ]
        for title in cases:
            with self.subTest(title=title):
                self.assertEqual(targets(title), set(), title)

    def test_a_distant_verb_is_not_a_relation(self):
        title = "updates the router, the pipeline, the corpus and then homelab:963"
        self.assertEqual(targets(title), set())

    def test_only_the_first_verb_binds_a_reference(self):
        found = relations("(extends #724, #725)")
        self.assertEqual(
            [(target, relation, form) for target, relation, _where, form in found],
            [(("homelab", 724), "extends", "hash"), (("homelab", 725), "extends", "hash")],
        )


class EdgeFilterTests(unittest.TestCase):
    def entry(self, corpus, entry_id, title, created_at, content=""):
        return {
            "corpus": corpus, "id": entry_id, "title": title,
            "content": content, "created_at": created_at,
        }

    def test_older_note_pointing_at_a_newer_one_is_not_an_edge(self):
        edges = nn.build_edges([
            self.entry("homelab", 963, "Route", "2026-09-20T10:00:00"),
            self.entry("homelab", 992, "Defaults (updates homelab:963)", "2026-09-19T10:00:00"),
        ])
        self.assertEqual(edges, {})

    def test_superseded_newer_note_is_not_an_edge(self):
        edges = nn.build_edges([
            self.entry("homelab", 963, "Route", "2026-09-19T10:00:00"),
            self.entry(
                "homelab", 992, "[SUPERSEDED] Defaults (updates homelab:963)", "2026-09-20T10:00:00"
            ),
        ])
        self.assertEqual(edges, {})

    def test_dangling_target_is_dropped(self):
        edges = nn.build_edges([
            self.entry("homelab", 992, "Defaults (updates homelab:99999)", "2026-09-20T10:00:00"),
        ])
        self.assertEqual(edges, {})

    def test_at_most_three_newer_notes_newest_first(self):
        entries = [self.entry("homelab", 963, "Route", "2026-09-19T10:00:00")]
        for offset, entry_id in enumerate((990, 991, 992, 993), start=1):
            entries.append(
                self.entry(
                    "homelab", entry_id, f"Change {entry_id} (updates homelab:963)",
                    f"2026-09-2{offset}T10:00:00",
                )
            )
        edges = nn.build_edges(entries)
        self.assertEqual(
            [item["ref"] for item in edges[("homelab", 963)]],
            ["homelab:993", "homelab:992", "homelab:991"],
        )

    def test_same_created_at_falls_back_to_the_id(self):
        entries = [
            self.entry("homelab", 963, "Route", "2026-09-19T10:00:00"),
            self.entry("homelab", 992, "A (updates homelab:963)", "2026-09-20T10:00:00"),
            self.entry("homelab", 993, "B (updates homelab:963)", "2026-09-20T10:00:00"),
        ]
        edges = nn.build_edges(entries)
        self.assertEqual([item["ref"] for item in edges[("homelab", 963)]],
                         ["homelab:993", "homelab:992"])


def corpus_db(path: Path) -> Path:
    connection = sqlite3.connect(path)
    try:
        connection.execute(
            "CREATE TABLE entries (id INTEGER PRIMARY KEY, type TEXT, content TEXT, "
            "title TEXT, summary TEXT, tags TEXT, raw_path TEXT, source TEXT, "
            "compiled_at TEXT, embedded_at TEXT, created_at TEXT)"
        )
        connection.executemany(
            "INSERT INTO entries VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            [
                (963, "note", "opencode Qwen route on Forrix, 65536 ctx.", "opencode Qwen route",
                 "", "qwen", None, "telegram", None, None, "2026-09-19T10:00:00"),
                (964, "note", "Forrix GPU cache behaviour.", "Forrix GPU cache",
                 "", "gpu", None, "telegram", None, None, "2026-09-19T11:00:00"),
                (992, "note", "Per-model load defaults persist across reboot.",
                 "Forrix Qwen per-model load defaults (updates homelab:963)",
                 "", "qwen", None, "telegram", None, None, "2026-09-20T10:00:00"),
            ],
        )
        connection.commit()
    finally:
        connection.close()
    return path


def empty_corpus_db(path: Path) -> Path:
    """The ai corpus is irrelevant here; it must simply exist and hold no notes."""
    connection = sqlite3.connect(path)
    try:
        connection.execute(
            "CREATE TABLE entries (id INTEGER PRIMARY KEY, type TEXT, content TEXT, "
            "title TEXT, summary TEXT, tags TEXT, raw_path TEXT, source TEXT, "
            "compiled_at TEXT, embedded_at TEXT, created_at TEXT)"
        )
        connection.commit()
    finally:
        connection.close()
    return path


def add_entry(path: Path, entry_id: int, title: str, content: str, created_at: str) -> None:
    connection = sqlite3.connect(path)
    try:
        connection.execute(
            "INSERT INTO entries (id, type, content, title, summary, tags, source, created_at) "
            "VALUES (?,?,?,?,?,?,?,?)",
            (entry_id, "note", content, title, "", "test", "telegram", created_at),
        )
        connection.commit()
    finally:
        connection.close()


class IndexTests(unittest.TestCase):
    """Refresh follows the FTS5 mechanism: signature, throttle, keep-on-failure."""

    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.dir = Path(self.temp.name)
        self.source = corpus_db(self.dir / "homelab.db")
        self.signature_calls = 0

    def tearDown(self) -> None:
        self.temp.cleanup()

    def signature(self, path):
        """The production fingerprint: a file size/mtime pair misses an insert that
        reuses a free page, which is why the FTS5 index counts rows and max id."""
        self.signature_calls += 1
        return kb_v2._fts5_source_signature(path)

    def test_build_reads_the_edges_and_refresh_is_throttled(self):
        index = nn.NewerNotesIndex.build({"homelab": str(self.source)}, self.signature)
        self.assertEqual([item["ref"] for item in index.for_ref("homelab", 963)], ["homelab:992"])
        self.assertEqual(self.signature_calls, 1)

        index.refresh()          # the first check after a build is always allowed
        before = self.signature_calls
        index.throttle_seconds = 3600.0
        index.refresh()
        self.assertEqual(self.signature_calls, before)

    def test_a_new_edge_appears_after_the_source_changes(self):
        index = nn.NewerNotesIndex.build({"homelab": str(self.source)}, self.signature)
        index.throttle_seconds = 0.0
        self.assertEqual(index.for_ref("homelab", 964), [])

        add_entry(self.source, 994, "GPU cache revisited (updates homelab:964)",
                  "Body.", "2026-09-25T10:00:00")
        index.refresh()
        self.assertEqual([item["ref"] for item in index.for_ref("homelab", 964)], ["homelab:994"])

    def test_an_unchanged_source_is_not_rebuilt(self):
        index = nn.NewerNotesIndex.build({"homelab": str(self.source)}, self.signature)
        index.throttle_seconds = 0.0
        calls = 0
        original = nn.build_edges

        def counting(entries):
            nonlocal calls
            calls += 1
            return original(entries)

        with patch.object(nn, "build_edges", counting):
            index.refresh()
        self.assertEqual(calls, 0)

    def test_a_failed_rebuild_keeps_the_edges_and_records_the_reason(self):
        index = nn.NewerNotesIndex.build({"homelab": str(self.source)}, self.signature)
        index.throttle_seconds = 0.0
        status: dict[str, str] = {}
        add_entry(self.source, 995, "Anything", "Body.", "2026-09-25T11:00:00")
        with patch.object(nn, "load_entries", side_effect=sqlite3.OperationalError("gone")):
            index.refresh(status)
        self.assertEqual([item["ref"] for item in index.for_ref("homelab", 963)], ["homelab:992"])
        self.assertIn("OperationalError", status["newer_notes_degraded"])
        self.assertEqual([item["ref"] for item in index.for_ref("homelab", 963)], ["homelab:992"])

    def test_for_ref_returns_a_new_list_each_time(self):
        index = nn.NewerNotesIndex.build({"homelab": str(self.source)}, self.signature)
        first = index.for_ref("homelab", 963)
        first.append({"ref": "homelab:1", "title": "t", "relation": "updates"})
        self.assertEqual(len(index.for_ref("homelab", 963)), 1)


if __name__ == "__main__":
    unittest.main()
