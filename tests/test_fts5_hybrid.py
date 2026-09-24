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
    """A stand-in for a corpus database: only the columns the index reads."""
    connection = sqlite3.connect(path)
    try:
        connection.execute(
            "CREATE TABLE entries (id INTEGER PRIMARY KEY, title TEXT, summary TEXT, content TEXT)"
        )
        connection.executemany(
            "INSERT INTO entries VALUES (?,?,?,?)",
            [
                (1, "Docker networking", "", "Bridge networks and port publishing on Nexus."),
                (2, "WireGuard", "VPN peers", "Peer configuration and preshared keys."),
                (3, "Plex", "", "Library paths and hardware transcoding."),
            ],
        )
        connection.commit()
    finally:
        connection.close()
    return path


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
        with patch("kb_v2._query_collection", return_value=[]), patch(
            "kb_v2._fetch_candidates", return_value=[]
        ):
            candidates = kb_v2._retrieve_corpus(
                "homelab",
                [0.1],
                self.config,
                None,
                None,
                str(self.dir / "missing-index.db"),
                "docker",
                status,
            )
        self.assertEqual(candidates, [])
        self.assertIn("OperationalError", status.get("degraded", ""))

    def test_healthy_index_records_no_degradation(self) -> None:
        status: dict[str, str] = {}
        index = kb_v2._build_fts5_index(
            "homelab", str(corpus_db(self.dir / "ok.db")), str(self.dir)
        )
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
        missing = str(self.dir / "missing-index.db")
        with patch("kb_v2._build_fts5_index", return_value=missing), patch(
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
        self.assertIn("OperationalError", fields["fts5_degraded"]["homelab"])


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


if __name__ == "__main__":
    unittest.main()
