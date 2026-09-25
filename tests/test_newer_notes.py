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


class FakeTokenizer:
    def encode(self, text, **kwargs):
        return list(range(len(text)))

    def __call__(self, text, **kwargs):
        return {"offset_mapping": [(i, i + 1) for i in range(len(text))]}

    def num_special_tokens_to_add(self, pair=True):
        return 3


class FakeReranker:
    tokenizer = FakeTokenizer()
    max_seq_length = 512

    def __init__(self, scores=None):
        self.scores = scores

    def predict(self, pairs):
        return self.scores or [2.0] * len(pairs)


CLIENTS = """clients:
  full:
    token_env: KB_V2_TOKEN_TEST_FULL
    allowed_corpora: [homelab, ai]
    allowed_scopes: [homelab, ai, both, auto]
"""

ROUTER = """router_version: corpus-router-v2-fts5-hybrid
accept_thresholds:
  homelab: 0.60
  ai: 0.60
reject_threshold: 0.40
both_margin: 0.05
dead_zone:
  lower: 0.40
  upper: 0.60
candidate_k: 25
max_distance:
  homelab: 0.60
  ai: 0.60
ai_decay:
  mode: disabled
fts5:
  enabled: true
  semantic_k: 20
  lexical_k: 5
"""


class SearchSurfaceTests(unittest.TestCase):
    """The warning reaches the API response and changes nothing else."""

    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.dir = Path(self.temp.name)
        self.source = corpus_db(self.dir / "homelab.db")
        self.ai_source = empty_corpus_db(self.dir / "ai.db")
        clients = self.dir / "clients.yml"
        clients.write_text(CLIENTS, encoding="utf-8")
        os.chmod(clients, 0o600)
        router = self.dir / "corpus-router.yml"
        router.write_text(ROUTER, encoding="utf-8")
        os.chmod(router, 0o600)
        self.environment = patch.dict(
            os.environ,
            {
                "KB_V2_CLIENTS_CONFIG": str(clients),
                "KB_CORPUS_ROUTER_CONFIG": str(router),
                "KB_V2_TOKEN_TEST_FULL": "f" * 64,
                "KB_FTS5_DIR": str(self.dir),
            },
        )
        self.environment.start()
        self.addCleanup(self.environment.stop)
        registry = patch.dict(
            kb_v2.CORPUS_REGISTRY,
            {
                "homelab": {"db_path": str(self.source), "collection": "homelab_collection"},
                "ai": {"db_path": str(self.ai_source), "collection": "ai_collection"},
            },
            clear=True,
        )
        registry.start()
        self.addCleanup(registry.stop)
        health = patch(
            "kb_v2._corpus_health",
            side_effect=lambda corpus, _ready: CorpusHealthV2(
                ready=True, collection=f"{corpus}_collection"
            ),
        )
        health.start()
        self.addCleanup(health.stop)
        # No lexical lane here: these tests are about the warning lane.
        fts5 = patch("kb_v2.Fts5Index.build", return_value=None)
        fts5.start()
        self.addCleanup(fts5.stop)
        throttle = patch.object(nn, "NEWER_NOTES_RECHECK_SECONDS", 0.0)
        throttle.start()
        self.addCleanup(throttle.stop)
        audit = patch("kb_v2._audit")
        self.audit = audit.start()
        self.addCleanup(audit.stop)
        self.headers = {"Authorization": "Bearer " + "f" * 64}

    def tearDown(self) -> None:
        self.temp.cleanup()

    def app(self, embed=None, scores=None, edges=True):
        if edges:
            return TestClient(create_v2_app(embed or Mock(return_value=[0.1]),
                                            lambda: FakeReranker(scores)))
        with patch("kb_v2.newer_notes_edges.NewerNotesIndex.build", return_value=None):
            return TestClient(create_v2_app(embed or Mock(return_value=[0.1]),
                                            lambda: FakeReranker(scores)))

    def search(self, client, query="qwen load defaults"):
        response = client.post(
            "/kb/search",
            headers=self.headers,
            json={"query": query, "scope": "homelab", "top_k": 5, "allow_degraded": False},
        )
        self.assertEqual(response.status_code, 200)
        return response.json()

    def candidates(self, ids):
        return [
            {"entry_id": entry_id, "distance": 0.1 + index / 100}
            for index, entry_id in enumerate(ids)
        ]

    def test_the_newer_note_is_reported_next_to_the_older_result(self):
        with patch("kb_v2._query_collection", side_effect=lambda corpus, *_: self.candidates([963, 964])):
            client = self.app(scores=[3.0, 2.0])
            value = self.search(client)
        results = {item["ref"]: item for item in value["corpora"]["homelab"]["results"]}
        self.assertEqual(
            results["homelab:963"]["newer_notes"],
            [{
                "ref": "homelab:992",
                "title": "Forrix Qwen per-model load defaults (updates homelab:963)",
                "relation": "updates",
            }],
        )
        self.assertEqual(results["homelab:964"]["newer_notes"], [])
        self.assertEqual(self.audit.call_args.kwargs["newer_notes_shown"], 1)

    def test_no_warning_when_the_newer_note_is_itself_in_the_results(self):
        with patch("kb_v2._query_collection", side_effect=lambda corpus, *_: self.candidates([963, 992])):
            client = self.app(scores=[3.0, 2.5])
            value = self.search(client)
        for item in value["corpora"]["homelab"]["results"]:
            self.assertEqual(item["newer_notes"], [], item["ref"])
        self.assertEqual(self.audit.call_args.kwargs["newer_notes_shown"], 0)

    def test_ranking_and_scores_are_identical_with_and_without_the_warning_lane(self):
        def strip(value):
            return [
                (item["ref"], item["distance"], item["relevance"], item["final_score"])
                for item in value["ranked"]
            ]

        with patch("kb_v2._query_collection", side_effect=lambda corpus, *_: self.candidates([963, 964])):
            with_edges = self.search(self.app(scores=[3.0, 2.0]))
            without_edges = self.search(self.app(scores=[3.0, 2.0], edges=False))
        self.assertEqual(strip(with_edges), strip(without_edges))
        self.assertEqual(with_edges["total_count"], without_edges["total_count"])
        self.assertEqual(
            [item["ref"] for item in with_edges["corpora"]["homelab"]["results"]],
            [item["ref"] for item in without_edges["corpora"]["homelab"]["results"]],
        )
        self.assertEqual(
            [(item["ref"], item["final_score"]) for item in with_edges["corpora"]["homelab"]["results"]],
            [(item["ref"], item["final_score"]) for item in without_edges["corpora"]["homelab"]["results"]],
        )
        # The warning is the only difference.
        self.assertTrue(
            any(item["newer_notes"] for item in with_edges["corpora"]["homelab"]["results"])
        )
        self.assertFalse(
            any(item["newer_notes"] for item in without_edges["corpora"]["homelab"]["results"])
        )

    def test_the_field_defaults_to_an_empty_list(self):
        with patch("kb_v2._query_collection", side_effect=lambda corpus, *_: self.candidates([964])):
            value = self.search(self.app(edges=False))
        self.assertEqual(value["corpora"]["homelab"]["results"][0]["newer_notes"], [])

    def test_a_note_added_after_startup_warns_without_a_restart(self):
        with patch("kb_v2._query_collection", side_effect=lambda corpus, *_: self.candidates([963])):
            client = self.app(scores=[3.0])
            before = self.search(client)
            self.assertEqual(
                [note["ref"] for note in before["corpora"]["homelab"]["results"][0]["newer_notes"]],
                ["homelab:992"],
            )

            add_entry(
                self.source, 993, "Forrix Qwen defaults revisited (updates homelab:963)",
                "Follow-up on the load defaults.", "2026-09-25T09:00:00",
            )
            after = self.search(client)
        self.assertEqual(
            [note["ref"] for note in after["corpora"]["homelab"]["results"][0]["newer_notes"]],
            ["homelab:993", "homelab:992"],
        )
        self.assertEqual(self.audit.call_args.kwargs["newer_notes_shown"], 2)

    def test_a_failing_edge_lane_is_reported_and_costs_no_results(self):
        with patch("kb_v2.newer_notes_edges.NewerNotesIndex.build",
                   side_effect=RuntimeError("boom")):
            with patch("kb_v2._query_collection", side_effect=lambda corpus, *_: self.candidates([963])):
                client = TestClient(create_v2_app(Mock(return_value=[0.1]),
                                                  lambda: FakeReranker([3.0])))
                value = self.search(client)
        self.assertEqual([item["ref"] for item in value["ranked"]], ["homelab:963"])
        self.assertEqual(value["ranked"][0]["newer_notes"], [])
        self.assertEqual(self.audit.call_args.kwargs["newer_notes_shown"], 0)
        self.assertIn("RuntimeError", self.audit.call_args.kwargs["newer_notes_degraded"])


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
