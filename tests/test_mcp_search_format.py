"""Tests for semantic_search result shaping and kb_get in mcp_server.

Mixed (scope=both) searches show at most MIXED_AI_HIT_LIMIT AI hits as briefs; with
cap_homelab (semantic_search always sets it) homelab hits longer than
HOMELAB_HIT_MAX_CHARS are cut to a preview plus a kb_get pointer, while hits at or below
the cap stay whole. scope='ai' keeps full AI text; kb_get reads one entry read-only.

Run from /opt/kb:  /opt/kb/venv/bin/python3 -m unittest discover -s tests -p 'test_mcp_*.py'
"""
import copy
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

# 4000 chars of space-separated words: over the homelab cap, with whitespace everywhere.
LONG_HOMELAB = "word " * 800


def hit(corpus, entry_id, content="homelab body", newer_notes=None):
    value = {
        "corpus": corpus, "entry_id": entry_id, "ref": f"{corpus}:{entry_id}",
        "title": f"{corpus} {entry_id}", "content": content, "tags": "t",
    }
    if newer_notes is not None:
        value["newer_notes"] = newer_notes
    return value


def note(ref, title, relation):
    return {"ref": ref, "title": title, "relation": relation}


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

    def test_newer_note_warning_is_rendered_below_the_result(self):
        ranked = [
            hit("homelab", 963, newer_notes=[
                note("homelab:992", "Forrix Qwen per-model load defaults (updates homelab:963)",
                     "updates")
            ]),
            hit("homelab", 964),
        ]
        text = mcp_server._format_corpus_payload(payload(ranked))
        self.assertIn(
            "\u26a0 newer note updates this: homelab:992 \u2014 Forrix Qwen per-model load "
            "defaults (updates homelab:963)",
            text,
        )
        # One line, under its own result, and never under a result without edges.
        block_963 = text.split("[homelab:964]")[0]
        self.assertEqual(block_963.count("\u26a0"), 1)
        self.assertNotIn("\u26a0", text.split("[homelab:964]")[1])

    def test_newer_note_warning_survives_compact_ai_mode(self):
        ranked = [
            hit("ai", 1, AI_CONTENT, newer_notes=[
                note("ai:2", "A later brief (updates ai:1)", "updates")
            ]),
        ]
        text = mcp_server._format_corpus_payload(payload(ranked), compact_ai=True)
        self.assertIn("\u26a0 newer note updates this: ai:2 \u2014 A later brief (updates ai:1)", text)
        self.assertIn("(brief \u2014 full text: kb_get('ai:1'))", text)
        self.assertLess(text.index("kb_get('ai:1')"), text.index("\u26a0"))

    def test_results_without_the_field_render_unchanged(self):
        ranked = [hit("homelab", 1)]
        text = mcp_server._format_corpus_payload(payload(ranked))
        self.assertNotIn("\u26a0", text)

    def test_full_rendering_without_compaction(self):
        ranked = [hit("ai", 1, AI_CONTENT), hit("ai", 2, AI_CONTENT), hit("ai", 3, AI_CONTENT)]
        text = mcp_server._format_corpus_payload(payload(ranked))
        self.assertEqual(text.count("long body"), 3 * 500)
        self.assertNotIn("omitted", text)


class HomelabCapTests(unittest.TestCase):
    def render(self, ranked, **kwargs):
        return mcp_server._format_corpus_payload(payload(ranked), **kwargs)

    def test_hits_at_or_below_the_cap_render_verbatim(self):
        cap = mcp_server.HOMELAB_HIT_MAX_CHARS
        for size in (cap - 1, cap):
            body = "a" * size
            with self.subTest(size=size):
                text = self.render([hit("homelab", 1, body)], cap_homelab=True)
                self.assertIn(body, text)
                self.assertNotIn("truncated", text)

    def test_hit_above_the_cap_is_cut_with_a_kb_get_pointer(self):
        cap = mcp_server.HOMELAB_HIT_MAX_CHARS
        body = "a" * (cap + 1)
        text = self.render([hit("homelab", 1, body)], cap_homelab=True)
        self.assertNotIn(body, text)
        preview = "a" * cap + " …"
        self.assertIn(preview, text)
        self.assertLessEqual(len(preview), cap + 2)
        self.assertIn("(truncated \u2014 full text: kb_get('homelab:1'))", text)

    def test_no_whitespace_anywhere_is_hard_cut_at_the_cap(self):
        cap = mcp_server.HOMELAB_HIT_MAX_CHARS
        text = self.render([hit("homelab", 1, "z" * 3000)], cap_homelab=True)
        self.assertIn("z" * cap + " …", text)
        self.assertNotIn("z" * (cap + 1), text)

    def test_cut_is_at_the_last_whitespace(self):
        text = self.render([hit("homelab", 1, LONG_HOMELAB)], cap_homelab=True)
        self.assertNotIn(LONG_HOMELAB, text)
        # The head is 400 full "word " groups, so the cut lands on the final space.
        self.assertIn("word " * 399 + "word …", text)

    def test_tab_is_a_cut_boundary(self):
        text = self.render([hit("homelab", 1, "a" * 1995 + "\t" + "c" * 100)], cap_homelab=True)
        self.assertIn("a" * 1995 + " …", text)
        self.assertNotIn("ccc", text)

    def test_paragraph_break_wins_when_it_is_late_in_the_cap(self):
        cap = mcp_server.HOMELAB_HIT_MAX_CHARS
        for start in (int(cap * 0.6), cap - 100):
            body = "p" * start + "\n\n" + "q" * 900
            with self.subTest(start=start):
                text = self.render([hit("homelab", 1, body)], cap_homelab=True)
                self.assertIn("p" * start + " …", text)
                self.assertNotIn("q", text)

    def test_early_paragraph_break_falls_back_to_the_last_whitespace(self):
        text = self.render([hit("homelab", 1, "p" * 500 + "\n\n" + "q" * 2500)], cap_homelab=True)
        self.assertIn("p" * 500 + "\n" + " …", text)
        self.assertNotIn("q", text)

    def test_truncated_hit_keeps_its_newer_note_warning(self):
        ranked = [
            hit("homelab", 1, LONG_HOMELAB,
                newer_notes=[note("homelab:2", "A later note (updates homelab:1)", "updates")]),
            hit("homelab", 3),
        ]
        text = self.render(ranked, cap_homelab=True)
        pointer = text.index("(truncated \u2014 full text: kb_get('homelab:1'))")
        warning = text.index("\u26a0 newer note updates this: homelab:2")
        self.assertLess(pointer, warning)
        self.assertLess(warning, text.index("[homelab:3]"))

    def test_grouped_fallback_truncates_homelab_and_keeps_ai_full(self):
        data = {
            "corpora": {
                "homelab": {"searched": True, "results": [
                    hit("homelab", 1, LONG_HOMELAB,
                        newer_notes=[note("homelab:2", "A later note", "updates")]),
                ]},
                "ai": {"searched": True, "results": [hit("ai", 9, AI_CONTENT)]},
            }
        }
        text = mcp_server._format_corpus_payload(data, cap_homelab=True)
        self.assertNotIn(LONG_HOMELAB, text)
        self.assertIn("long body", text)  # AI hits stay full text in the grouped shape
        pointer = text.index("(truncated \u2014 full text: kb_get('homelab:1'))")
        warning = text.index("\u26a0 newer note updates this: homelab:2 \u2014 A later note")
        self.assertLess(pointer, warning)
        self.assertLess(warning, text.index("[ai:9]"))

    def test_payload_is_not_mutated(self):
        ranked = [hit("homelab", 1, LONG_HOMELAB)] + [hit("ai", i, AI_CONTENT) for i in (2, 3, 4)]
        data = payload(ranked)
        before = copy.deepcopy(data)
        mcp_server._format_corpus_payload(data, compact_ai=True, cap_homelab=True)
        self.assertEqual(data, before)


class ScopeTests(unittest.TestCase):
    def run_search(self, ranked=None, **kwargs):
        ranked = [hit("ai", 1, AI_CONTENT)] if ranked is None else ranked
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

    def test_both_scope_truncates_a_long_homelab_hit(self):
        cs, text = self.run_search(ranked=[hit("homelab", 1, LONG_HOMELAB)])
        self.assertEqual(cs.call_args.kwargs["scope"], "both")
        self.assertNotIn(LONG_HOMELAB, text)
        self.assertIn("(truncated \u2014 full text: kb_get('homelab:1'))", text)

    def test_homelab_scope_truncates_a_long_homelab_hit(self):
        cs, text = self.run_search(ranked=[hit("homelab", 1, LONG_HOMELAB)], scope="homelab")
        self.assertEqual(cs.call_args.kwargs["scope"], "homelab")
        self.assertNotIn(LONG_HOMELAB, text)
        self.assertIn("(truncated \u2014 full text: kb_get('homelab:1'))", text)

    def test_homelab_scope_keeps_a_short_hit_whole(self):
        cs, text = self.run_search(ranked=[hit("homelab", 1, "short body")], scope="homelab")
        self.assertEqual(cs.call_args.kwargs["scope"], "homelab")
        self.assertIn("short body", text)
        self.assertNotIn("truncated", text)


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


class KbGetLongEntryTests(unittest.TestCase):
    """kb_get is the pointer target: it must never truncate."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.body = LONG_HOMELAB + "end"  # kb_get strips, so no trailing whitespace
        path = pathlib.Path(self.tmp.name) / "kb.db"
        db = sqlite3.connect(path)
        db.execute("CREATE TABLE entries (id INTEGER PRIMARY KEY, title TEXT, tags TEXT, content TEXT)")
        db.execute("INSERT INTO entries VALUES (7, 'Long', '', ?)", (self.body,))
        db.commit()
        db.close()
        self.patch = mock.patch.dict(mcp_server.KB_DATABASES, {"homelab": str(path)})
        self.patch.start()

    def tearDown(self):
        self.patch.stop()
        self.tmp.cleanup()

    def test_returns_the_full_long_entry(self):
        text = mcp_server.kb_get("homelab:7")
        self.assertIn("[homelab:7] Long", text)
        self.assertIn(self.body, text)
        self.assertNotIn("truncated", text)


if __name__ == "__main__":
    unittest.main()
