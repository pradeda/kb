"""Tools must be declared unstructured: no outputSchema, no duplicated structuredContent.

FastMCP auto-generates an output schema for `-> str` tools, so every call returned the text
twice — once as TextContent and once inside `structuredContent` as `{"result": "<text>"}`
(OMP renders both to the model). `structured_output=False` on the decorator keeps the
unstructured path only; the TextContent text must stay byte-identical, which is what the
golden texts below pin. Uses the real FastMCP instance, not the contract test's FakeFastMCP.

Run from /opt/kb:  /opt/kb/venv/bin/python3 -m unittest tests.test_mcp_no_structured
"""
import asyncio
import importlib.util
import os
import pathlib
import sqlite3
import tempfile
import unittest
from unittest import mock

from mcp.types import TextContent

ROOT = pathlib.Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("mcp_server", ROOT / "mcp_server.py")
mcp_server = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mcp_server)

TOOL_NAMES = {"semantic_search", "kb_get", "corpus_search", "add", "supersede", "history"}

AI_CONTENT = (
    "# Article\n\nSource: [Blog](https://example.com/post)\n\n"
    "## Summary\n\nShort summary.\n\n## Body\n\n" + "long body " * 20
)
SEARCH_PAYLOAD = {
    "ranked": [
        {
            "corpus": "homelab", "entry_id": 7, "ref": "homelab:7", "title": "homelab 7",
            "content": "short homelab body", "tags": "t",
        },
        {
            "corpus": "ai", "entry_id": 3, "ref": "ai:3", "title": "ai 3",
            "content": AI_CONTENT, "tags": "t",
        },
    ],
    "corpora": {"homelab": {"searched": True}, "ai": {"searched": True}},
}
V2_RESPONSE = {
    "corpora": {"homelab": {"searched": True, "results": []}, "ai": {"searched": False}},
    "ranked": [],
}

# Texts produced before the decorator change (captured from the live FastMCP instance).
KB_GET_TEXT = "[ai:7] Seven\nTags: a,b\nfull body"
SEMANTIC_TEXT = (
    "=== 2 result(s), best first ===\n"
    "[homelab:7] homelab 7\n"
    "Tags: t\n"
    "short homelab body\n"
    "\n"
    "[ai:3] ai 3\n"
    "Tags: t\n"
    "Source: [Blog](https://example.com/post)\n"
    "Short summary.\n"
    "(brief \u2014 full text: kb_get('ai:3'))"
)
CORPUS_TEXT = (
    "{\n"
    '  "corpora": {\n'
    '    "homelab": {\n'
    '      "searched": true,\n'
    '      "results": []\n'
    "    },\n"
    '    "ai": {\n'
    '      "searched": false\n'
    "    }\n"
    "  },\n"
    '  "ranked": []\n'
    "}"
)


def call_tool(name, arguments):
    return asyncio.run(mcp_server.mcp.call_tool(name, arguments))


def text_of(result):
    """The single TextContent text; fails loudly if the call returned a (content, structured) pair."""
    assert isinstance(result, list), f"expected a content list, got {type(result).__name__}"
    assert len(result) == 1 and isinstance(result[0], TextContent), result
    return result[0].text


class NoOutputSchemaTests(unittest.TestCase):
    def test_every_tool_is_declared_unstructured(self):
        tools = asyncio.run(mcp_server.mcp.list_tools())
        self.assertEqual({tool.name for tool in tools}, TOOL_NAMES)
        self.assertEqual(
            {tool.name: tool.outputSchema for tool in tools},
            dict.fromkeys(TOOL_NAMES),
            "an outputSchema makes FastMCP send structuredContent alongside the text",
        )


class UnstructuredCallTests(unittest.TestCase):
    """Every tool must return a plain content list — a tuple means the duplicate came back."""

    def test_kb_get(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "kb.db"
            db = sqlite3.connect(path)
            db.execute(
                "CREATE TABLE entries (id INTEGER PRIMARY KEY, title TEXT, tags TEXT, content TEXT)"
            )
            db.execute("INSERT INTO entries VALUES (7, 'Seven', 'a,b', 'full body')")
            db.commit()
            db.close()
            with mock.patch.dict(mcp_server.KB_DATABASES, {"ai": str(path)}):
                result = call_tool("kb_get", {"reference": "ai:7"})
        self.assertEqual(text_of(result), KB_GET_TEXT)

    def test_semantic_search(self):
        with mock.patch.object(mcp_server, "corpus_search", return_value=SEARCH_PAYLOAD):
            result = call_tool("semantic_search", {"query": "q"})
        self.assertEqual(text_of(result), SEMANTIC_TEXT)

    def test_corpus_search(self):
        response = mock.Mock(status_code=200, json=mock.Mock(return_value=V2_RESPONSE))
        with mock.patch.object(mcp_server, "_load_local_v2_token"), mock.patch.dict(
            os.environ, {"KB_V2_TOKEN_MCP_LOCAL": "test-token"}
        ), mock.patch.object(mcp_server.httpx, "post", return_value=response):
            result = call_tool("corpus_search", {"query": "q"})
        self.assertEqual(text_of(result), CORPUS_TEXT)

    def test_cli_backed_tools(self):
        stdout = mock.Mock(returncode=0, stdout="", stderr="")
        cli = mock.Mock(run=mock.Mock(return_value=stdout))
        with mock.patch.object(mcp_server, "subprocess", cli):
            results = {
                "add": call_tool("add", {"content": "c", "title": "t", "tag": "x"}),
                "supersede": call_tool("supersede", {"entry_id": 1, "replacement": "homelab:2"}),
                "history": call_tool("history", {"reference": "homelab:1"}),
            }
        self.assertEqual(
            {name: text_of(result) for name, result in results.items()},
            {
                "add": "Added successfully.",
                "supersede": "Superseded successfully.",
                "history": "No history.",
            },
        )


if __name__ == "__main__":
    unittest.main()
