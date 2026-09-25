#!/usr/bin/env python3
"""Supersede gate: given an incoming KB entry, surface existing entries that
are related enough that a blind insert risks clobbering / duplicating a fact.

The gate does NOT decide supersede-vs-coexist — that judgment is semantic and
belongs to the agent. The gate only answers "is there a related entry the agent
must look at before inserting?" (crossing = similarity >= threshold AND topic
overlap >= 1). It reuses production embedding (query embedding via the embed
daemon socket) and ChromaDB, so its similarity matches live retrieval.

Fail policy: embedding/Chroma infrastructure unavailable -> raise
GateUnavailable (caller fails OPEN with a warning). Invalid input and
programming errors propagate as ordinary exceptions (caller hard-fails).
"""
import json
import re
import socket
import httpx

EMBED_SOCKET = "/run/kb-embed/embed.sock"
CHROMA_BASE = "http://localhost:8000/api/v2/tenants/default_tenant/databases/default_database"
CHROMA_COLLECTION = "kb_collection"

# Generic/structural tags+tokens that carry no topic signal — never count as overlap.
STOP = {
    "gotcha", "fix", "bug", "homelab", "ai", "archive", "changelog", "telegram",
    "url", "note", "repo", "policy", "documentation", "dijagnoza", "superseded",
    "howto", "decision", "evergreen", "config", "security", "monitoring",
    "docker", "nexus", "rpi4", "rpi", "nexus2", "forrix", "nas", "kerrigan",
    "fix", "update", "patch", "root-cause", "diagnosis", "review",
    "github", "repo", "web", "configuration", "setup", "integration",
}

_WORD = re.compile(r"[a-z0-9][a-z0-9._-]{2,}")


class GateUnavailable(Exception):
    """Embedding or Chroma infrastructure could not be reached. Caller fails open."""


def _norm_tags(tags: str) -> set:
    parts = re.split(r"[,/;]", (tags or "").lower())
    return {p.strip() for p in parts if p.strip()}


def _topic_tokens(tags: str, title: str) -> set:
    """Deterministic topic set: normalized tags (primary) + significant title
    tokens (secondary). No model classification. Stoplist + numeric filter drop
    generic/structural terms so same-service-different-fact does not over-match
    on noise alone."""
    toks = _norm_tags(tags) | set(_WORD.findall((title or "").lower()))
    return {t for t in toks if t not in STOP and not t.isdigit() and len(t) >= 3}


def _embed(text: str) -> list:
    """Query-embed via the production embed daemon socket. Returns vector.
    Raises GateUnavailable on any socket problem or null response."""
    try:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(10)
        sock.connect(EMBED_SOCKET)
        sock.sendall((text.strip() + "\n").encode())
        chunks = []
        while True:
            try:
                data = sock.recv(4096)
                if not data:
                    break
                chunks.append(data)
                if b"\n" in data:
                    break
            except socket.timeout:
                break
        sock.close()
        line = b"".join(chunks).decode().strip()
    except OSError as e:
        raise GateUnavailable(f"embed socket: {e}") from e
    if not line or line == "null":
        raise GateUnavailable("embed daemon returned null")
    return json.loads(line)


def _chroma_query(embedding: list) -> list:
    """Full-scan Chroma query. Returns [{id,distance,title,tags}]. Raises
    GateUnavailable if Chroma cannot be reached / errors."""
    try:
        client = httpx.Client(timeout=30)
        resp = None
        collection_id = None
        for attempt in range(2):
            cresp = client.get(f"{CHROMA_BASE}/collections/{CHROMA_COLLECTION}")
            if cresp.status_code != 200:
                raise GateUnavailable(f"collection lookup {cresp.status_code}")
            collection_id = cresp.json()["id"]
            count_resp = client.get(f"{CHROMA_BASE}/collections/{collection_id}/count")
            if count_resp.status_code != 200:
                raise GateUnavailable(f"chroma count {count_resp.status_code}")
            n = count_resp.json()
            if not isinstance(n, int) or n < 1:
                return []
            resp = client.post(
                f"{CHROMA_BASE}/collections/{collection_id}/query",
                json={"query_embeddings": [embedding], "n_results": n,
                      "include": ["distances", "metadatas"]},
            )
            if resp.status_code != 404:
                break
        if resp is None or resp.status_code != 200:
            raise GateUnavailable(f"chroma query {getattr(resp, 'status_code', 'none')}")
        data = resp.json()
    except httpx.HTTPError as e:
        raise GateUnavailable(f"chroma http: {e}") from e

    if not data.get("ids") or not data["ids"][0]:
        return []
    ids_ = data["ids"][0]
    dists = data.get("distances", [[]])[0]
    metas = data.get("metadatas", [[]])[0]
    out = []
    for i, eid in enumerate(ids_):
        meta = metas[i] if i < len(metas) else {}
        out.append({
            "id": str(eid),
            "distance": float(dists[i]) if i < len(dists) else 1.0,
            "title": meta.get("title", ""),
            "tags": meta.get("tags", ""),
        })
    return out


def check_relatedness(content: str, title: str, tags: str,
                      sim_threshold: float, exclude_ids=()) -> dict:
    """Embed the incoming entry (title+content) and return existing entries that
    cross both bars: similarity (1 - cosine distance) >= sim_threshold AND
    topic overlap >= 1 shared topic token.

    Returns {"status": "ok", "threshold": t, "related": [ {id,similarity,
    topic_overlap,title,tags} ... sorted by similarity desc ]}.
    Raises GateUnavailable on infra failure (caller fails open)."""
    if sim_threshold is None:
        raise ValueError("sim_threshold is required")
    exclude = {str(x) for x in exclude_ids}
    new_topic = _topic_tokens(tags, title)
    text = f"{title}\n{content}".strip()
    if not text:
        raise ValueError("empty incoming entry")

    embedding = _embed(text)                 # GateUnavailable on infra failure
    candidates = _chroma_query(embedding)     # GateUnavailable on infra failure

    related = []
    for c in candidates:
        if c["id"] in exclude:
            continue
        sim = 1.0 - c["distance"]
        if sim < sim_threshold:
            continue
        overlap = new_topic & _topic_tokens(c["tags"], c["title"])
        if not overlap:
            continue
        related.append({
            "id": c["id"],
            "similarity": round(sim, 4),
            "topic_overlap": sorted(overlap),
            "title": c["title"],
            "tags": c["tags"],
        })
    related.sort(key=lambda r: -r["similarity"])
    return {"status": "ok", "threshold": sim_threshold, "related": related}
