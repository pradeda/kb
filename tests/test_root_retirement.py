"""The retired v1 search surface stays retired.

v1 (`/kb/search`, `/kb/websearch`) answered without authentication and had no live
caller left; the retirement of 2026-08-12 replaced it with unconditional 410
tombstones. The implementation and the `KB_V1_SEARCH_ENABLED` switch are gone, so the
surface cannot be brought back by configuration - these are the properties that must
keep holding, and they replaced tests/test_v1_contract.py (which pinned the v1
response shapes, OpenAPI projection and the switch itself).
"""
from __future__ import annotations

import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

import kb_search_api


class RootRetirementTests(unittest.TestCase):
    def root_app(self):
        """Build the root app with the lexical lane stubbed out.

        The mounted v2 app builds the FTS5 index at creation; stub it so this test
        never reads or writes a real corpus index.
        """
        with patch("kb_v2.Fts5Index.build", return_value=None):
            return kb_search_api.create_root_app()

    def test_v1_endpoints_answer_410(self) -> None:
        client = TestClient(self.root_app())
        for path in ("/kb/search", "/kb/websearch"):
            response = client.post(path, json={"query": "docker"})
            self.assertEqual(response.status_code, 410, path)
            self.assertEqual(response.json()["detail"], "KB Search v1 is retired")

    def test_the_removed_switch_cannot_re_enable_v1(self) -> None:
        """KB_V1_SEARCH_ENABLED used to gate the surface; it is not read any more."""
        with patch.dict("os.environ", {"KB_V1_SEARCH_ENABLED": "true"}):
            self.assertFalse(hasattr(kb_search_api, "V1_SEARCH_ENABLED"))
            self.assertFalse(hasattr(kb_search_api, "_parse_v1_search_enabled"))
            client = TestClient(self.root_app())
            self.assertEqual(client.post("/kb/search", json={"query": "x"}).status_code, 410)

    def test_openapi_publishes_only_the_health_route(self) -> None:
        """The tombstones and the synthesis endpoint stay out of the schema."""
        schema = self.root_app().openapi()
        self.assertEqual(sorted(schema["paths"]), ["/health"])

    def test_health_still_reports_the_reranker_state(self) -> None:
        response = TestClient(self.root_app()).get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "ok")
        self.assertIn("rerank_model", response.json())

    def test_synthesis_endpoint_requires_its_own_token(self) -> None:
        with patch.dict("os.environ", {}, clear=False):
            import os

            os.environ.pop("KB_SYNTHESIS_TOKEN", None)
            response = TestClient(self.root_app()).post(
                "/kb/synthesize/nexus-relevance",
                json={
                    "query": "docker",
                    "source_type": "video",
                    "video_title": "t",
                },
            )
        self.assertEqual(response.status_code, 503)


if __name__ == "__main__":
    unittest.main()
