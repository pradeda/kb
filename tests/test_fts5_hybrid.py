"""FTS5 hybrid retrieval: index isolation, visible degradation, merge order.

The lexical index is built when the app is created and read on every search.
Two properties must hold, and neither was covered before:

  1. A process that is not the service — a test run, or any tool that imports
     ``kb_search_api`` — must never write the index file the running service
     reads. The index path therefore follows ``KB_FTS5_DIR`` (or an explicit
     ``fts5_dir``), and importing ``kb_search_api`` must build nothing at all.
  2. A missing or unreadable index must be visible: the search still answers
     without lexical candidates, but the audit carries ``fts5_degraded`` with
     the reason instead of dropping them silently.

Run from /opt/kb:  /opt/kb/venv-search/bin/python3 -m unittest tests.test_fts5_hybrid
"""
from __future__ import annotations

import os
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient

import kb_search_api
import kb_v2
from kb_v2 import CorpusHealthV2, create_v2_app


SERVICE_INDEX = Path("/tmp/kb-fts5-homelab.db")


def corpus_db(path: Path) -> Path:
    """A stand-in for a corpus database, with the columns the search path reads."""
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
                (1, "note", "Bridge networks and port publishing on Nexus.", "Docker networking", "", "docker", None, "telegram", None, None, "2026-08-01T10:00:00"),
                (2, "note", "Peer configuration and preshared keys.", "WireGuard", "VPN peers", "vpn", None, "telegram", None, None, "2026-08-02T10:00:00"),
                (3, "note", "Library paths and hardware transcoding.", "Plex", "", "plex", None, "telegram", None, None, "2026-08-03T10:00:00"),
            ],
        )
        connection.commit()
    finally:
        connection.close()
    return path


def add_entry(path, entry_id: int, title: str, content: str) -> None:
    """Insert one note the way compile.py would: a new row with a fresh created_at."""
    connection = sqlite3.connect(path)
    try:
        connection.execute(
            "INSERT INTO entries (id, type, content, title, summary, tags, source, created_at) "
            "VALUES (?,?,?,?,?,?,?,?)",
            (entry_id, "note", content, title, "", "test", "telegram", "2026-09-24T13:30:00"),
        )
        connection.commit()
    finally:
        connection.close()


def router_config(**changes) -> kb_v2.RouterConfig:
    values = {
        "router_version": kb_v2.PRECALIBRATION_ROUTER_VERSION,
        "accept_thresholds": {"homelab": 0.6, "ai": 0.6},
        "reject_threshold": 0.4,
        "both_margin": 0.05,
        "dead_zone_lower": 0.4,
        "dead_zone_upper": 0.6,
        "candidate_k": 25,
        "max_distance": {"homelab": 0.6, "ai": 0.6},
        "ai_decay_mode": "disabled",
        "fts5_enabled": True,
        "fts5_semantic_k": 20,
        "fts5_lexical_k": 5,
    }
    values.update(changes)
    return kb_v2.RouterConfig(**values)


class Fts5IndexIsolationTests(unittest.TestCase):
    """The index the service reads is not writable by anything else."""

    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.dir = Path(self.temp.name)
        self.source = corpus_db(self.dir / "source.db")

    def tearDown(self) -> None:
        self.temp.cleanup()

    def service_index_state(self):
        return SERVICE_INDEX.stat().st_mtime_ns if SERVICE_INDEX.exists() else None

    def test_index_is_written_under_kb_fts5_dir(self) -> None:
        before = self.service_index_state()
        with patch.dict(os.environ, {"KB_FTS5_DIR": str(self.dir)}):
            path = kb_v2._build_fts5_index("homelab", str(self.source))
        self.assertEqual(Path(path).parent, self.dir)
        self.assertTrue(Path(path).is_file())
        # The running service's index must not be rewritten by this process.
        self.assertEqual(self.service_index_state(), before)
        connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        try:
            rows = connection.execute(
                "SELECT entry_id, title FROM entries_fts ORDER BY entry_id"
            ).fetchall()
        finally:
            connection.close()
        self.assertEqual(
            rows,
            [(1, "Docker networking"), (2, "WireGuard"), (3, "Plex")],
        )

    def test_explicit_directory_wins_over_the_environment(self) -> None:
        explicit = self.dir / "explicit"
        with patch.dict(os.environ, {"KB_FTS5_DIR": str(self.dir / "from-env")}):
            path = kb_v2._build_fts5_index("ai", str(self.source), str(explicit))
        self.assertEqual(Path(path).parent, explicit)

    def test_without_a_directory_nothing_lands_on_the_service_path(self) -> None:
        """No env, no argument: a private directory, never the service's file."""
        before = self.service_index_state()
        with patch.dict(os.environ, {"KB_FTS5_DIR": str(self.dir / "unused")}):
            del os.environ["KB_FTS5_DIR"]
            path = kb_v2._build_fts5_index("homelab", str(self.source))
        self.assertNotEqual(Path(path), SERVICE_INDEX)
        self.assertTrue(Path(path).is_file())
        self.assertEqual(self.service_index_state(), before)

    def test_importing_kb_search_api_builds_no_index(self) -> None:
        """Import is what tools and tests do; it must not touch any index."""
        target = self.dir / "imported"
        environment = {**os.environ, "KB_FTS5_DIR": str(target)}
        completed = subprocess.run(
            [sys.executable, "-c", "import kb_search_api"],
            cwd=str(Path(__file__).parents[1]),
            env=environment,
            capture_output=True,
            text=True,
            timeout=300,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertNotIn("[fts5]", completed.stdout + completed.stderr)
        self.assertEqual(sorted(target.glob("*.db")) if target.exists() else [], [])


    def test_private_index_directory_does_not_outlive_the_process(self) -> None:
        """A caller that names no directory gets a private one - it must not leak.

        The private directory is created with mkdtemp and was never removed, so
        every importing tool and every test run left a /tmp/kb-fts5-private-*
        directory behind for the lifetime of the machine.
        """
        program = (
            "import os, kb_v2\n"
            "print(os.path.dirname(kb_v2._fts5_index_path('homelab')))\n"
        )
        environment = {key: value for key, value in os.environ.items() if key != "KB_FTS5_DIR"}
        completed = subprocess.run(
            [sys.executable, "-c", program],
            cwd=str(Path(__file__).parents[1]),
            env=environment,
            capture_output=True,
            text=True,
            timeout=300,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        private_dir = Path(completed.stdout.strip().splitlines()[-1])
        self.assertIn("kb-fts5-private-", private_dir.name)
        self.assertFalse(
            private_dir.exists(), f"{private_dir} outlived the process that created it"
        )

class Fts5DegradationTests(unittest.TestCase):
    """A broken lexical index is reported, not swallowed."""

    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.dir = Path(self.temp.name)
        self.config = router_config()

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_retrieve_corpus_records_a_broken_index(self) -> None:
        status: dict[str, str] = {}
        # Source gone and index gone: the refresh cannot fingerprint, and the query
        # cannot read — both must land in the status instead of failing the search.
        index = kb_v2.Fts5Index(
            "homelab",
            str(self.dir / "missing-source.db"),
            str(self.dir / "missing-index.db"),
            (),
        )
        with patch("kb_v2._query_collection", return_value=[]), patch(
            "kb_v2._fetch_candidates", return_value=[]
        ):
            candidates = kb_v2._retrieve_corpus(
                "homelab", [0.1], self.config, None, None, index, "docker", status
            )
        self.assertEqual(candidates, [])
        self.assertIn("OperationalError", status.get("degraded", ""))

    def test_healthy_index_records_no_degradation(self) -> None:
        status: dict[str, str] = {}
        index = kb_v2.Fts5Index.build("homelab", str(corpus_db(self.dir / "ok.db")), str(self.dir))
        with patch("kb_v2._query_collection", return_value=[]), patch(
            "kb_v2._fetch_candidates", return_value=[]
        ):
            kb_v2._retrieve_corpus(
                "homelab", [0.1], self.config, None, None, index, "docker", status
            )
        self.assertEqual(status, {})

    def test_search_audit_carries_fts5_degraded(self) -> None:
        clients = self.dir / "clients.yml"
        clients.write_text(
            """clients:
  full:
    token_env: KB_V2_TOKEN_TEST_FULL
    allowed_corpora: [homelab, ai]
    allowed_scopes: [homelab, ai, both, auto]
""",
            encoding="utf-8",
        )
        os.chmod(clients, 0o600)
        router = self.dir / "corpus-router.yml"
        router.write_text(
            """router_version: corpus-router-v2-fts5-hybrid
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
""",
            encoding="utf-8",
        )
        os.chmod(router, 0o600)
        environment = patch.dict(
            os.environ,
            {
                "KB_V2_CLIENTS_CONFIG": str(clients),
                "KB_CORPUS_ROUTER_CONFIG": str(router),
                "KB_V2_TOKEN_TEST_FULL": "f" * 64,
                "KB_FTS5_DIR": str(self.dir),
            },
        )
        environment.start()
        self.addCleanup(environment.stop)
        health = patch(
            "kb_v2._corpus_health",
            side_effect=lambda corpus, _ready: CorpusHealthV2(
                ready=True, collection=f"{corpus}_collection"
            ),
        )
        health.start()
        self.addCleanup(health.stop)
        # A stub source keeps the freshness check off the live database, and a corrupt
        # index file makes the lexical read fail without a rebuild being triggered.
        source = corpus_db(self.dir / "audit-source.db")
        broken = self.dir / "broken-index.db"
        broken.write_bytes(b"not a database")
        index = kb_v2.Fts5Index(
            "homelab", str(source), str(broken), kb_v2._fts5_source_signature(str(source))
        )
        with patch("kb_v2.Fts5Index.build", return_value=index), patch(
            "kb_v2._query_collection", return_value=[]
        ), patch("kb_v2._fetch_candidates", return_value=[]), patch(
            "kb_v2._audit"
        ) as audit:
            client = TestClient(create_v2_app(Mock(return_value=[0.1]), lambda: Mock()))
            response = client.post(
                "/kb/search",
                headers={"Authorization": "Bearer " + "f" * 64},
                json={"query": "docker", "scope": "homelab", "top_k": 5, "allow_degraded": False},
            )
        self.assertEqual(response.status_code, 200)
        fields = audit.call_args.kwargs
        self.assertEqual(fields["fts5_candidates"], 0)
        self.assertTrue(fields["fts5_degraded"]["homelab"])


class Fts5QueryTests(unittest.TestCase):
    """Terms, match expression, BM25 order and the merge rule."""

    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.dir = Path(self.temp.name)
        self.index = self.dir / "index.db"
        connection = sqlite3.connect(self.index)
        try:
            connection.execute(
                f"CREATE VIRTUAL TABLE entries_fts USING fts5("
                f"entry_id UNINDEXED, title, summary, content, "
                f"tokenize='{kb_v2.FTS5_TOKENIZER}')"
            )
            connection.executemany(
                "INSERT INTO entries_fts VALUES(?,?,?,?)",
                [
                    (7, "Plex transcoding", "", "Plex hardware transcoding on Nexus."),
                    (8, "Backups", "", "Nightly backups to the NAS over NFS."),
                    (9, "Docker networking", "", "Bridge networks and port publishing."),
                ],
            )
            connection.commit()
        finally:
            connection.close()

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_terms_drop_stopwords_and_fold_diacritics(self) -> None:
        self.assertEqual(kb_v2._fts5_terms("How do I restart the Docker container?"), ["restart", "docker", "container"])
        self.assertEqual(kb_v2._fts5_terms("greška vraća 500"), ["greska", "vraca", "500"])
        self.assertEqual(kb_v2._fts5_terms("the a of"), [])

    def test_serbian_suffix_is_stripped_for_multiword_queries(self) -> None:
        self.assertIn("konfiguracij", kb_v2._fts5_terms("kako je konfiguracijama podesen dns"))

    def test_match_expression_quotes_and_prefixes_long_words(self) -> None:
        self.assertEqual(kb_v2._fts5_match_expression([]), "")
        self.assertEqual(kb_v2._fts5_match_expression(["dns"]), '"dns"')
        self.assertEqual(
            kb_v2._fts5_match_expression(["docker", "dns"]), '"docker"* OR "dns"'
        )

    def test_query_returns_bm25_ordered_hits_within_the_limit(self) -> None:
        hits = kb_v2._fts5_query(str(self.index), "plex transcoding nexus", 10)
        self.assertEqual([hit["entry_id"] for hit in hits], [7])
        self.assertLess(hits[0]["bm25"], 0)
        self.assertEqual(kb_v2._fts5_query(str(self.index), "the a of", 10), [])
        self.assertEqual(len(kb_v2._fts5_query(str(self.index), "backups nas nfs", 1)), 1)

    def test_missing_index_raises_instead_of_returning_nothing(self) -> None:
        with self.assertRaises(sqlite3.OperationalError):
            kb_v2._fts5_query(str(self.dir / "missing.db"), "docker", 10)

    def test_merge_keeps_semantic_head_then_adds_lexical_only(self) -> None:
        config = router_config()
        semantic = [{"entry_id": entry_id, "distance": 0.1} for entry_id in range(1, 26)]
        # A lexical hit that is already a semantic candidate adds nothing.
        merged = kb_v2._merge_fts5_candidates(
            semantic, [{"entry_id": 1, "bm25": -1.0}], config
        )
        self.assertEqual([item["entry_id"] for item in merged], list(range(1, 26)))
        self.assertFalse(any(item.get("from_fts5") for item in merged))

        # Lexical-only hits follow the semantic head and push the semantic tail out.
        lexical = [
            {"entry_id": 1, "bm25": -1.0},
            {"entry_id": 30, "bm25": -2.0},
            {"entry_id": 31, "bm25": -3.0},
        ]
        merged = kb_v2._merge_fts5_candidates(semantic, lexical, config)
        self.assertEqual(
            [item["entry_id"] for item in merged],
            list(range(1, 21)) + [30, 31, 21, 22, 23],
        )
        self.assertEqual(
            [item["entry_id"] for item in merged if item.get("from_fts5")], [30, 31]
        )
        lexical_only = [item for item in merged if item.get("from_fts5")]
        self.assertTrue(
            all(item["distance"] == config.max_distance["homelab"] for item in lexical_only)
        )

    def test_merge_respects_the_lexical_limit_and_candidate_k(self) -> None:
        config = router_config()
        semantic = [{"entry_id": entry_id, "distance": 0.1} for entry_id in range(1, 26)]
        lexical = [{"entry_id": 100 + offset, "bm25": -float(offset)} for offset in range(10)]
        merged = kb_v2._merge_fts5_candidates(semantic, lexical, config)
        lexical_only = [item["entry_id"] for item in merged if item.get("from_fts5")]
        self.assertEqual(lexical_only, [100, 101, 102, 103, 104])
        self.assertEqual(len(merged), config.candidate_k)
        self.assertEqual(len({item["entry_id"] for item in merged}), len(merged))


class FakeTokenizer:
    """Character offsets suffice for short fixtures; no model download."""

    def encode(self, text, **kwargs):
        return list(range(len(text)))

    def __call__(self, text, **kwargs):
        return {"offset_mapping": [(i, i + 1) for i in range(len(text))]}

    def num_special_tokens_to_add(self, pair=True):
        return 3


class FakeReranker:
    max_seq_length = 512

    def __init__(self, score: float = 2.0) -> None:
        self.score = score
        self.tokenizer = FakeTokenizer()

    def predict(self, pairs):
        return [self.score] * len(pairs)


class Fts5FreshnessTests(unittest.TestCase):
    """The index follows the source database, not only the process start."""

    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.dir = Path(self.temp.name)
        self.source = corpus_db(self.dir / "source.db")
        self.index = kb_v2.Fts5Index.build("homelab", str(self.source), str(self.dir))
        self.status: dict[str, str] = {}
        self.throttle = patch.object(kb_v2, "FTS5_RECHECK_SECONDS", 0.0)
        self.throttle.start()
        self.addCleanup(self.throttle.stop)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def write(self, statement: str, parameters: tuple = ()) -> None:
        connection = sqlite3.connect(self.source)
        try:
            connection.execute(statement, parameters)
            connection.commit()
        finally:
            connection.close()

    def ids(self, query: str) -> list[int]:
        return [hit["entry_id"] for hit in kb_v2._fts5_query(self.index.path, query, 10)]

    def test_new_row_is_found_by_the_next_query(self) -> None:
        self.assertEqual(self.ids("beszel monitoring"), [])
        add_entry(self.source, 4, "Beszel agent", "Beszel monitoring for the Nexus host.")
        self.index.refresh(self.status)
        self.assertEqual(self.ids("beszel monitoring"), [4])
        self.assertEqual(self.status, {})

    def test_updated_row_replaces_its_lexical_content(self) -> None:
        self.assertEqual(self.ids("preshared keys"), [2])
        self.write(
            "UPDATE entries SET title=?, content=? WHERE id=?",
            ("Tailscale", "Replaced WireGuard with Tailscale on the Pi.", 2),
        )
        self.index.refresh(self.status)
        self.assertEqual(self.ids("tailscale"), [2])
        self.assertEqual(self.ids("preshared keys"), [])
        self.assertEqual(self.status, {})

    def test_deleted_row_disappears(self) -> None:
        self.assertEqual(self.ids("transcoding"), [3])
        self.write("DELETE FROM entries WHERE id=?", (3,))
        self.index.refresh(self.status)
        self.assertEqual(self.ids("transcoding"), [])
        self.assertEqual(self.status, {})

    def test_unchanged_source_does_not_rebuild(self) -> None:
        with patch("kb_v2._build_fts5_index", wraps=kb_v2._build_fts5_index) as build:
            self.index.refresh(self.status)
            self.index.refresh(self.status)
        build.assert_not_called()
        self.assertEqual(self.status, {})

    def test_throttle_skips_the_check_within_the_window(self) -> None:
        self.index.refresh(self.status)  # establishes the check window
        add_entry(self.source, 4, "Beszel agent", "Beszel monitoring for the Nexus host.")
        with patch.object(kb_v2, "FTS5_RECHECK_SECONDS", 3600.0), patch(
            "kb_v2._build_fts5_index", wraps=kb_v2._build_fts5_index
        ) as build:
            self.index.refresh(self.status)
        build.assert_not_called()
        # The old index keeps answering; the new row waits for the next window.
        self.assertEqual(self.ids("beszel monitoring"), [])

    def test_missing_index_file_is_rebuilt(self) -> None:
        os.remove(self.index.path)
        self.index.refresh(self.status)
        self.assertTrue(Path(self.index.path).is_file())
        self.assertEqual(self.ids("wireguard"), [2])
        self.assertEqual(self.status, {})

    def test_unreadable_source_keeps_the_old_index_and_records_the_reason(self) -> None:
        os.rename(self.source, str(self.source) + ".gone")
        self.index.refresh(self.status)
        self.assertIn("fingerprint failed", self.status["degraded"])
        self.assertEqual(self.ids("wireguard"), [2])

    def test_failed_rebuild_keeps_the_old_index_and_records_the_reason(self) -> None:
        add_entry(self.source, 4, "Beszel agent", "Beszel monitoring for the Nexus host.")
        with patch("kb_v2._build_fts5_index", side_effect=OSError("no space left on device")):
            self.index.refresh(self.status)
        self.assertIn("index rebuild failed", self.status["degraded"])
        self.assertIn("no space left on device", self.status["degraded"])
        self.assertEqual(self.ids("wireguard"), [2])
        self.assertEqual(self.ids("beszel monitoring"), [])

    def test_replacement_is_atomic_and_leaves_no_temporary_files(self) -> None:
        reader = sqlite3.connect(f"file:{self.index.path}?mode=ro", uri=True)
        try:
            add_entry(self.source, 4, "Beszel agent", "Beszel monitoring for the Nexus host.")
            self.index.refresh(self.status)
            # os.replace, not DROP TABLE: a reader holding the old file keeps working.
            self.assertEqual(reader.execute("SELECT count(*) FROM entries_fts").fetchone()[0], 3)
        finally:
            reader.close()
        self.assertEqual(sorted(path.name for path in self.dir.glob("*.tmp")), [])
        self.assertEqual(self.ids("beszel monitoring"), [4])


class Fts5FreshnessEndpointTests(unittest.TestCase):
    """End to end: a note added while the app runs is found by the next search."""

    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.dir = Path(self.temp.name)
        self.source = corpus_db(self.dir / "homelab.db")
        self.ai_source = corpus_db(self.dir / "ai.db")
        clients = self.dir / "clients.yml"
        clients.write_text(
            """clients:
  full:
    token_env: KB_V2_TOKEN_TEST_FULL
    allowed_corpora: [homelab, ai]
    allowed_scopes: [homelab, ai, both, auto]
""",
            encoding="utf-8",
        )
        os.chmod(clients, 0o600)
        router = self.dir / "corpus-router.yml"
        router.write_text(
            """router_version: corpus-router-v2-fts5-hybrid
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
""",
            encoding="utf-8",
        )
        os.chmod(router, 0o600)
        environment = patch.dict(
            os.environ,
            {
                "KB_V2_CLIENTS_CONFIG": str(clients),
                "KB_CORPUS_ROUTER_CONFIG": str(router),
                "KB_V2_TOKEN_TEST_FULL": "f" * 64,
                "KB_FTS5_DIR": str(self.dir),
            },
        )
        environment.start()
        self.addCleanup(environment.stop)
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
        throttle = patch.object(kb_v2, "FTS5_RECHECK_SECONDS", 0.0)
        throttle.start()
        self.addCleanup(throttle.stop)
        self.audit = patch("kb_v2._audit")
        self.audit_mock = self.audit.start()
        self.addCleanup(self.audit.stop)
        self.client = TestClient(create_v2_app(Mock(return_value=[0.1]), lambda: FakeReranker()))
        self.headers = {"Authorization": "Bearer " + "f" * 64}

    def tearDown(self) -> None:
        self.temp.cleanup()

    def search(self) -> dict:
        response = self.client.post(
            "/kb/search",
            headers=self.headers,
            json={"query": "beszel monitoring", "scope": "homelab", "top_k": 5, "allow_degraded": False},
        )
        self.assertEqual(response.status_code, 200)
        return response.json()

    def test_note_added_after_startup_is_found_without_a_restart(self) -> None:
        with patch("kb_v2._query_collection", return_value=[]):
            before = self.search()
            self.assertEqual(before["total_count"], 0)
            self.assertEqual(self.audit_mock.call_args.kwargs["fts5_candidates"], 0)

            add_entry(self.source, 4, "Beszel agent", "Beszel monitoring for the Nexus host.")

            after = self.search()
        self.assertEqual(
            [item["ref"] for item in after["corpora"]["homelab"]["results"]], ["homelab:4"]
        )
        self.assertEqual(self.audit_mock.call_args.kwargs["fts5_candidates"], 1)
        self.assertEqual(self.audit_mock.call_args.kwargs["fts5_degraded"], {})


if __name__ == "__main__":
    unittest.main()
