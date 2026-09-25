#!/opt/kb/venv/bin/python3
import os
import re
import sqlite3
import sys
import subprocess
from typing import Annotated, Literal

import httpx
from pydantic import Field

from mcp.server.fastmcp import FastMCP

import corpora as corpus_identity  # single corpus-identity definition

# SSE / Streamable HTTP transport — host/port passed to FastMCP constructor
# (env vars are overridden by explicit params, so we pass them directly)
_sse_mode   = "--sse"  in sys.argv
_http_mode  = "--http" in sys.argv
_remote     = _sse_mode or _http_mode
_host       = "0.0.0.0" if _remote else "127.0.0.1"
_port       = 9101 if _http_mode else (9100 if _sse_mode else 8000)

mcp = FastMCP("kb", host=_host, port=_port, log_level="WARNING")


def _load_local_v2_token() -> None:
    if os.getenv("KB_V2_TOKEN_MCP_LOCAL"):
        return
    try:
        with open("/opt/kb/.env", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if line.startswith("KB_V2_TOKEN_MCP_LOCAL="):
                    value = line.split("=", 1)[1].strip().strip('"').strip("'")
                    if value:
                        os.environ["KB_V2_TOKEN_MCP_LOCAL"] = value
                    return
    except OSError:
        return


# In a mixed (scope=both) search, AI entries are long article briefs (median ~6.8 KB
# vs ~1.7 KB for homelab notes) and are rarely what a homelab task needs, so they are
# shown as a short brief and capped. scope="ai" or kb_get() returns the full text.
MIXED_AI_HIT_LIMIT = 2
AI_BRIEF_MAX_CHARS = 700
_AI_SUMMARY_RE = re.compile(
    r"^## (?:Summary|Quick take|Executive summary)\s*\n+(.+?)(?=\n## |\Z)", re.S | re.M
)
_SOURCE_RE = re.compile(r"^Source: *(.+)$", re.M)

# Corpus identity comes from corpora.py (same directory in the deployment).
KB_DATABASES = corpus_identity.databases()


def _ai_brief(content: str) -> str:
    """The summary section of an AI article brief, or its opening text as a fallback."""
    match = _AI_SUMMARY_RE.search(content)
    text = (match.group(1) if match else content).strip()
    if len(text) > AI_BRIEF_MAX_CHARS:
        text = text[:AI_BRIEF_MAX_CHARS].rsplit(" ", 1)[0] + " …"
    source = _SOURCE_RE.search(content)
    return f"Source: {source.group(1).strip()}\n{text}" if source else text


def _render_hits(items: list, header: str, *, compact_ai: bool = False) -> list[str]:
    lines = [header]
    for item in items:
        lines.append(f"[{item.get('ref')}] {item.get('title')}")
        tags = item.get("tags")
        if tags:
            lines.append(f"Tags: {tags}")
        content = str(item.get("content", "")).strip()
        if compact_ai and item.get("corpus") == "ai":
            lines.append(_ai_brief(content))
            lines.append(f"(brief — full text: kb_get('{item.get('ref')}'))")
        else:
            lines.append(content)
        lines.append("")
    return lines


def _format_corpus_payload(payload: dict, *, compact_ai: bool = False) -> str:
    """Render the v2 response as one cross-corpus ranking, each hit naming its corpus.

    Prefers the merged `ranked` list: the reranker scores both corpora in a single
    batch, so ordering them together is the ranking the model actually produced, and
    reading it as one list is what a caller asking a question wants. Falls back to the
    grouped shape when `ranked` is absent, so an older API stays readable.

    compact_ai keeps the ranking but shows at most MIXED_AI_HIT_LIMIT AI hits, each
    as a brief; homelab hits are never shortened or dropped.
    """
    ranked = payload.get("ranked")
    if isinstance(ranked, list) and ranked:
        not_searched = [
            corpus
            for corpus in ("homelab", "ai")
            if not (payload.get("corpora", {}).get(corpus) or {}).get("searched")
        ]
        omitted_ai = 0
        if compact_ai:
            shown, ai_seen = [], 0
            for item in ranked:
                if item.get("corpus") == "ai":
                    ai_seen += 1
                    if ai_seen > MIXED_AI_HIT_LIMIT:
                        omitted_ai += 1
                        continue
                shown.append(item)
            ranked = shown
        lines = _render_hits(
            ranked, f"=== {len(ranked)} result(s), best first ===", compact_ai=compact_ai
        )
        if omitted_ai:
            lines.append(
                f"({omitted_ai} more AI result(s) omitted — for AI/model/tool questions "
                "use semantic_search(scope='ai'))"
            )
        if not_searched:
            lines.append(f"(not searched: {', '.join(not_searched)})")
        return "\n".join(lines).strip()

    lines: list[str] = []
    for corpus in ("homelab", "ai"):
        section = payload.get("corpora", {}).get(corpus)
        if not isinstance(section, dict):
            continue
        if not section.get("searched"):
            lines.append(f"=== {corpus}: not searched ===")
            continue
        results = section.get("results") or []
        lines.extend(_render_hits(results, f"=== {corpus}: {len(results)} result(s) ==="))
    return "\n".join(lines).strip() or "No results."


@mcp.tool(
    description=(
        "MUST be called before any research, implementation, debugging, or configuration task. "
        "Searches the knowledge corpora — Homelab infrastructure and AI research — and returns "
        "corpus-qualified entries in one ranking, without an internal LLM call. "
        "scope='both' (default, for homelab tasks) returns homelab entries in full and at most "
        f"{MIXED_AI_HIT_LIMIT} AI entries as short briefs; use scope='ai' for questions about AI "
        "models, tools, papers or techniques (full AI entries), scope='homelab' to skip AI, and "
        "kb_get(reference) for the full text of one brief. When supplying query_alt, translate the "
        "same intent faithfully into the other language without adding facts, and preserve technical "
        "literals exactly; query_alt_language is required with it."
    )
)
def semantic_search(
    query: str,
    query_alt: str | None = None,
    query_alt_language: Literal["sr", "en"] | None = None,
    scope: Literal["both", "homelab", "ai"] = "both",
) -> str:
    return _format_corpus_payload(
        corpus_search(query, scope=scope, query_alt=query_alt, query_alt_language=query_alt_language),
        compact_ai=scope == "both",
    )


@mcp.tool(
    description=(
        "Return the full text of one KB entry by its reference, e.g. 'ai:363' or 'homelab:1072' — "
        "use it to expand a brief from semantic_search. Read-only, no search or LLM call."
    )
)
def kb_get(reference: str) -> str:
    corpus, _, raw_id = reference.strip().partition(":")
    if corpus not in KB_DATABASES or not raw_id.isdigit():
        return f"Invalid reference {reference!r}: expected 'homelab:<id>' or 'ai:<id>'."
    try:
        connection = sqlite3.connect(f"file:{KB_DATABASES[corpus]}?mode=ro", uri=True, timeout=5)
        try:
            row = connection.execute(
                "SELECT title, tags, content FROM entries WHERE id=?", (int(raw_id),)
            ).fetchone()
        finally:
            connection.close()
    except sqlite3.Error as exc:
        return f"KB get failed for {reference}: {exc}"
    if row is None:
        return f"No entry {reference}."
    title, tags, content = row
    lines = [f"[{corpus}:{raw_id}] {title}"]
    if tags:
        lines.append(f"Tags: {tags}")
    lines.append(str(content or "").strip())
    return "\n".join(lines)


@mcp.tool(
    description=(
        "Search the structured Homelab and AI knowledge corpora without an internal LLM call. "
        "Defaults to both corpora; narrow it with homelab or ai when the target is known. "
        "auto stays unavailable until its calibrated router is enabled. "
        "Returns corpus-qualified references and grouped results. When supplying query_alt, translate "
        "the same intent faithfully into the other language without adding facts, and preserve technical "
        "literals exactly; query_alt_language is required with it."
    )
)
def corpus_search(
    query: str,
    # Not "auto": that returns HTTP 409 until the router is calibrated, so it
    # would make every scope-less call an error.
    scope: Literal["homelab", "ai", "both", "auto"] = "both",
    top_k: Annotated[int, Field(ge=1, le=5)] = 5,
    query_alt: str | None = None,
    query_alt_language: Literal["sr", "en"] | None = None,
) -> dict:
    _load_local_v2_token()
    token = os.getenv("KB_V2_TOKEN_MCP_LOCAL")
    if not token:
        raise RuntimeError("KB corpus search is not configured")
    endpoint = os.getenv("KB_V2_SEARCH_URL", "http://127.0.0.1:8050/v2/kb/search")
    try:
        payload = {
            "query": query,
            "scope": scope,
            "top_k": top_k,
            "allow_degraded": False,
        }
        if query_alt is not None:
            payload["query_alt"] = query_alt
        if query_alt_language is not None:
            payload["query_alt_language"] = query_alt_language
        response = httpx.post(
            endpoint,
            headers={"Authorization": f"Bearer {token}"},
            json=payload,
            timeout=45,
        )
    except httpx.TimeoutException as exc:
        raise RuntimeError("KB corpus search timed out") from exc
    except httpx.HTTPError as exc:
        raise RuntimeError("KB corpus search is unavailable") from exc
    if response.status_code != 200:
        reason = "unknown"
        try:
            detail = response.json().get("detail")
            if isinstance(detail, dict):
                reason = str(detail.get("reason", "unknown"))
            elif isinstance(detail, str):
                reason = detail
        except (ValueError, AttributeError):
            pass
        raise RuntimeError(f"KB corpus search failed (HTTP {response.status_code}: {reason})")
    try:
        value = response.json()
    except ValueError as exc:
        raise RuntimeError("KB corpus search returned invalid JSON") from exc
    if not isinstance(value, dict) or "corpora" not in value:
        raise RuntimeError("KB corpus search returned an invalid response")
    return value


@mcp.tool(
    description=(
        "Add a note to a knowledge base corpus. Defaults to the homelab corpus; "
        "pass corpus='ai' to write to the AI research corpus instead. "
        "Use for documenting solutions, gotchas, config changes, or any knowledge worth preserving. "
        "Content is passed via stdin to support multi-line text safely."
    )
)
def add(
    content: str,
    title: str,
    tag: str,
    corpus: Literal["homelab", "ai"] = "homelab",
) -> str:
    try:
        result = subprocess.run(
            ["/usr/local/bin/kb", "add", "--corpus", corpus, "note", "-", title, tag],
            input=content,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except subprocess.TimeoutExpired:
        return "KB add timed out — check kb-embed/SQLite (WAL lock?)."
    if result.returncode != 0:
        return f"KB add failed: {result.stderr or result.stdout}".strip()
    return result.stdout or result.stderr or "Added successfully."


@mcp.tool(
    description=(
        "Mark an existing KB entry as superseded because a newer entry replaces a "
        "current-state fact it asserts that is no longer true (e.g. a changed domain, "
        "port, path, threshold, or a corrected diagnosis). "
        "Use ONLY when the old entry would mislead if retrieved today — NOT for adding "
        "history or a different aspect of the same topic (those should coexist as normal "
        "add). First find the old entry via semantic_search, then add the corrective "
        "entry, then call this with its reference. entry_id is the obsolete entry's numeric "
        "id; replacement is the KB reference(s) of the current entry, e.g. 'homelab:323'. "
        "corpus selects which corpus entry_id belongs to (default homelab) and must match "
        "the entry's own reference — an 'ai:<id>' entry needs corpus='ai', otherwise the "
        "call fails with 'no such entry' or hits the homelab row with the same id. "
        "The old entry keeps its id and history, its title is prefixed [SUPERSEDED], and it "
        "is demoted in search. Refuses if entry_id does not exist in that corpus."
    )
)
def supersede(
    entry_id: int,
    replacement: str,
    corpus: Literal["homelab", "ai"] = "homelab",
) -> str:
    try:
        result = subprocess.run(
            ["/usr/local/bin/kb", "supersede", "--corpus", corpus, str(entry_id), replacement],
            input="",  # pipe stdin -> non-interactive (no y/N prompt); never blocks
            capture_output=True,
            text=True,
            timeout=30,
        )
    except subprocess.TimeoutExpired:
        return "KB supersede timed out — check kb-embed/SQLite (WAL lock?)."
    if result.returncode != 0:
        return f"KB supersede failed: {result.stderr or result.stdout}".strip()
    return result.stdout or result.stderr or "Superseded successfully."


@mcp.tool(
    description=(
        "Show the supersede lineage of one KB entry: every explicitly linked "
        "predecessor and successor, in a single call, with no LLM or vector lookup. "
        "Use when prior versions of a fact matter — to see what an entry replaced or "
        "what replaced it, or to trace a chain of corrections. reference is a full "
        "'corpus:id' (e.g. 'homelab:940'); traversal is cross-corpus. Returns JSON "
        "with nodes, edges, an ordered 'chain' (oldest→newest, or null when the "
        "lineage branches/merges), and a 'completeness' field. Heed it: 'partial' "
        "with warnings/broken_links/truncated means the answer is NOT the whole "
        "history (a stale index needs 'kb rebuild-supersede-index'; a broken link "
        "means a linked entry was retired)."
    )
)
def history(reference: str) -> str:
    try:
        result = subprocess.run(
            ["/usr/local/bin/kb", "history", reference],
            input="",  # pipe stdin/stdout -> non-interactive, JSON output
            capture_output=True,
            text=True,
            timeout=15,
        )
    except subprocess.TimeoutExpired:
        return "KB history timed out — check SQLite (WAL lock?)."
    if result.returncode != 0:
        return f"KB history failed: {result.stderr or result.stdout}".strip()
    return result.stdout or result.stderr or "No history."


if __name__ == "__main__":
    if "--sse" in sys.argv:
        mcp.run(transport="sse")
    elif "--http" in sys.argv:
        mcp.run(transport="streamable-http")
    else:
        mcp.run()
