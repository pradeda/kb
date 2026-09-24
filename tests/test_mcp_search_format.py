"""Tests for semantic_search result shaping and kb_get in mcp_server.

Mixed (scope=both) searches show at most MIXED_AI_HIT_LIMIT AI hits as briefs and
never shorten or drop homelab hits; scope='ai' keeps full AI text; kb_get reads
one entry read-only.

Run from /opt/kb:  /opt/kb/venv/bin/python3 -m unittest discover -s tests -p 'test_mcp_*.py'
"""
import importlib.util
import pathlib
import sqlite3
import tempfile
import unittest
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("mcp_server", ROOT / "mcp_server.py")
mcp_server = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mcp_server)

AI_CONTENT = (
    "# Title\n\nSource: [Blog](https://example.com/post)\nPublished: 2026-09-22\n\n"
    "## Summary\n\nShort summary of the article.\n\n"
    "## Detailed content brief\n\n" + "long body " * 500
)


def hit(corpus, entry_id, content="homelab body"):
    return {
        "corpus": corpus, "entry_id": entry_id, "ref": f"{corpus}:{entry_id}",
        "title": f"{corpus} {entry_id}", "content": content, "tags": "t",
    }


def payload(ranked):
    return {
        "ranked": ranked,
        "corpora": {"homelab": {"searched": True}, "ai": {"searched": True}},
    }


class MixedFormatTests(unittest.TestCase):
    def test_ai_hits_become_briefs_and_are_capped(self):
        ranked = [hit("ai", 1, AI_CONTENT), hit("homelab", 2), hit("ai", 3, AI_CONTENT),
                  hit("ai", 4, AI_CONTENT), hit("homelab", 5)]
        text = mcp_server._format_corpus_payload(payload(ranked), compact_ai=True)
        self.assertIn("[ai:1]", text)
        self.assertIn("[ai:3]", text)
        self.assertNotIn("[ai:4]", text)
        self.assertIn("1 more AI result(s) omitted", text)
        self.assertIn("Short summary of the article.", text)
        self.assertIn("Source: [Blog](https://example.com/post)", text)
        self.assertNotIn("long body", text)
        self.assertIn("kb_get('ai:1')", text)

    def test_homelab_hits_are_untouched(self):
        body = "x" * 9000
        ranked = [hit("homelab", 1, body), hit("ai", 2, AI_CONTENT), hit("homelab", 3, body)]
        text = mcp_server._format_corpus_payload(payload(ranked), compact_ai=True)
        self.assertEqual(text.count(body), 2)
        self.assertLess(text.index("[homelab:1]"), text.index("[ai:2]"))
        self.assertLess(text.index("[ai:2]"), text.index("[homelab:3]"))

    def test_brief_falls_back_to_truncated_opening(self):
        brief = mcp_server._ai_brief("no sections here " * 200)
        self.assertLessEqual(len(brief), mcp_server.AI_BRIEF_MAX_CHARS + 2)
        self.assertTrue(brief.endswith("…"))

    def test_full_rendering_without_compaction(self):
        ranked = [hit("ai", 1, AI_CONTENT), hit("ai", 2, AI_CONTENT), hit("ai", 3, AI_CONTENT)]
        text = mcp_server._format_corpus_payload(payload(ranked))
        self.assertEqual(text.count("long body"), 3 * 500)
        self.assertNotIn("omitted", text)


class ScopeTests(unittest.TestCase):
    def run_search(self, **kwargs):
        ranked = [hit("ai", 1, AI_CONTENT)]
        with mock.patch.object(mcp_server, "corpus_search", return_value=payload(ranked)) as cs:
            text = mcp_server.semantic_search("q", **kwargs)
        return cs, text

    def test_default_scope_is_both_and_compact(self):
        cs, text = self.run_search()
        self.assertEqual(cs.call_args.kwargs["scope"], "both")
        self.assertNotIn("long body", text)

    def test_ai_scope_returns_full_text(self):
        cs, text = self.run_search(scope="ai")
        self.assertEqual(cs.call_args.kwargs["scope"], "ai")
        self.assertIn("long body", text)


class KbGetTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        path = pathlib.Path(self.tmp.name) / "kb.db"
        db = sqlite3.connect(path)
        db.execute("CREATE TABLE entries (id INTEGER PRIMARY KEY, title TEXT, tags TEXT, content TEXT)")
        db.execute("INSERT INTO entries VALUES (7, 'Seven', 'a,b', 'full body')")
        db.commit()
        db.close()
        self.patch = mock.patch.dict(mcp_server.KB_DATABASES, {"ai": str(path)})
        self.patch.start()

    def tearDown(self):
        self.patch.stop()
        self.tmp.cleanup()

    def test_returns_full_entry(self):
        text = mcp_server.kb_get("ai:7")
        self.assertIn("[ai:7] Seven", text)
        self.assertIn("full body", text)

    def test_missing_and_invalid_references(self):
        self.assertIn("No entry ai:8", mcp_server.kb_get("ai:8"))
        self.assertIn("Invalid reference", mcp_server.kb_get("other:1"))
        self.assertIn("Invalid reference", mcp_server.kb_get("ai:x"))


if __name__ == "__main__":
    unittest.main()
