"""Query rewriter for KB retrieval evaluation.

Supports Gemini Flash (OpenRouter, default) and local Qwen (Forrix).
"""

import json
import os
import re
import time
import urllib.request
from typing import Optional

QWEN_URL = "http://192.168.1.160:1234/v1/chat/completions"
QWEN_MODEL = "qwen3.8-27b"

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
GEMINI_MODEL = "google/gemini-2.5-flash"

REWRITE_SYSTEM = """\
You are a search query optimizer for a technical knowledge base about homelab infrastructure \
(Docker, NFS, Pi-hole, WireGuard, Nexus server, Raspberry Pi, KB search pipeline, AI ingest) \
and AI research notes.

Given a natural-language question, extract the key search terms that would best retrieve \
relevant documents from an embedding-based search system.

Rules:
- Preserve entity names exactly (Pi-hole, WireGuard, ChromaDB, NAS, RPi4, Forrix, Nexus, Qwen)
- Preserve version numbers and technical identifiers
- Preserve negation and relationship constraints
- Remove question words (how, what, why, when, where, which, does, is, are, do, can)
- Remove filler words (the, a, an, to, of, for, with, on, in, it, this, that)
- Keep nouns, verbs (action words), adjectives that carry technical meaning
- Output ONLY the rewritten keywords, nothing else — no explanation, no punctuation
- If the query is already keyword-style, return it unchanged
- For Serbian queries, keep the language — rewrite in Serbian keywords"""

REWRITE_EXAMPLES = [
    ("How is NAS connected to the rest of the network?", "NAS network connection topology"),
    ("What backup strategy do we use?", "backup strategy NAS schedule"),
    ("How does the reranker work in KB search?", "KB search reranker cross-encoder pipeline"),
    ("Why did the AI ingest produce low quality entries?", "AI ingest quality filter relevance"),
    ("Kako se Plex povezuje sa NAS-om?", "Plex NAS konfiguracija povezivanje"),
]


def _build_messages(query: str) -> list[dict]:
    messages = [{"role": "system", "content": REWRITE_SYSTEM}]
    for q, a in REWRITE_EXAMPLES:
        messages.append({"role": "user", "content": q})
        messages.append({"role": "assistant", "content": a})
    messages.append({"role": "user", "content": query})
    return messages


def _clean_response(content: str) -> str:
    content = content.strip()
    if "<think>" in content:
        content = re.sub(r"<think>.*?</think>\s*", "", content, flags=re.DOTALL).strip()
    content = content.strip('"\'')
    return content


def rewrite_query(
    query: str,
    backend: str = "gemini",
    temperature: float = 0.0,
    timeout: int = 30,
) -> dict:
    """Rewrite a natural-language query to keywords.

    backend: "gemini" (OpenRouter, default) or "qwen" (local Forrix).
    Returns {rewritten, latency_ms, model, backend}.
    """
    messages = _build_messages(query)

    if backend == "gemini":
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            return {"rewritten": None, "error": "OPENROUTER_API_KEY not set", "latency_ms": 0, "model": GEMINI_MODEL, "backend": backend}
        url = OPENROUTER_URL
        model = GEMINI_MODEL
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": 100,
        }
    else:
        url = QWEN_URL
        model = QWEN_MODEL
        headers = {"Content-Type": "application/json"}
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": 2000,
        }

    started = time.perf_counter()
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            result = json.loads(resp.read())
        raw = result["choices"][0]["message"]["content"]
        rewritten = _clean_response(raw)
        latency = round((time.perf_counter() - started) * 1000, 1)
        if not rewritten:
            return {"rewritten": None, "error": "empty", "latency_ms": latency, "model": model, "backend": backend}
        return {"rewritten": rewritten, "latency_ms": latency, "model": model, "backend": backend}
    except Exception as exc:
        latency = round((time.perf_counter() - started) * 1000, 1)
        return {"rewritten": None, "error": str(exc), "latency_ms": latency, "model": model, "backend": backend}


def batch_rewrite(queries: list[str], **kwargs) -> list[dict]:
    """Rewrite multiple queries sequentially."""
    return [rewrite_query(q, **kwargs) for q in queries]


if __name__ == "__main__":
    test_queries = [
        "How is NAS connected to the rest of the network?",
        "What backup strategy do we use?",
        "How do I add a new entry to the knowledge base?",
        "Kako se Plex povezuje sa NAS-om?",
    ]
    for backend in ["gemini", "qwen"]:
        print(f"=== {backend.upper()} ===")
        for q in test_queries:
            result = rewrite_query(q, backend=backend)
            status = result['rewritten'] or f"FAIL: {result.get('error')}"
            print(f"  Q: {q}")
            print(f"  R: {status}  ({result['latency_ms']}ms)")
            print()
