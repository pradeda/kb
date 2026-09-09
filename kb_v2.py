"""Strict, authenticated multi-corpus KB read plane.

This module deliberately does not import the legacy FastAPI application.  The
caller injects the shared embedding and reranker dependencies, which keeps the
v1 route/model surface unchanged while allowing one process to serve both APIs.
"""

from __future__ import annotations

import hmac
import json
import math
import os
import re
import sqlite3
import stat
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field as dc_field
from datetime import datetime, timezone
from typing import Callable, Literal, Optional
from urllib.parse import urlparse

import httpx
import yaml
from fastapi import Depends, FastAPI, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import (
    AnyHttpUrl,
    AnyUrl,
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    field_validator,
    model_validator,
)


CorpusName = Literal["homelab", "ai"]
ScopeName = Literal["homelab", "ai", "both", "auto"]
SelectedScope = Literal["homelab", "ai", "both", "none"]

CHROMA_BASE = "http://localhost:8000/api/v2/tenants/default_tenant/databases/default_database"
CLIENT_CONFIG_PATH = "/opt/kb/v2-clients.yml"
ROUTER_CONFIG_PATH = "/opt/kb/corpus-router.yml"
PRECALIBRATION_ROUTER_VERSION = "corpus-router-v2-fts5-hybrid"
HOMELAB_DECAY_HALF_LIFE = 540.0
HOMELAB_DECAY_FLOOR = 0.30
EMBEDDING_MODEL = "nomic-ai/nomic-embed-text-v1.5"
EMBEDDING_DIMENSION = 768
COLLECTION_SCHEMA_VERSION = 1
ALTERNATE_ONLY_RANK_LIMIT = 5
FTS5_FETCH_LIMIT = 100
FTS5_TOKENIZER = "porter unicode61 remove_diacritics 2"
FTS5_BM25_WEIGHTS = (0, 3, 2, 1)

_FTS5_STOP = frozenset(
    "how what why when where which does is are do can the a an to of for with on "
    "in it this that we our i be as from and or have has did use us exact value "
    "configured assigned koji koja koje kako sta zasto kada gde je su se sa za u "
    "na od do da li mi nasa nasom nasoj nase nasu nasih".split()
)
_FTS5_SR_MARKERS = frozenset("kako sta zasto kada gde je su nasom nasoj nase koji koja koje".split())
_FTS5_SR_SUFFIXES = ("ovima", "evima", "ama", "ima", "om", "em")


def _fts5_terms(query: str) -> list[str]:
    folded = "".join(
        c for c in unicodedata.normalize("NFD", query.lower())
        if unicodedata.category(c) != "Mn"
    )
    tokens = re.findall(r"[^\W_]+(?:[-.][^\W_]+)*", folded)
    serbian = bool(set(tokens) & _FTS5_SR_MARKERS)
    out: list[str] = []
    for word in tokens:
        if word in _FTS5_STOP or len(word) < 2:
            continue
        if serbian and word.isalpha():
            for suffix in _FTS5_SR_SUFFIXES:
                if word.endswith(suffix) and len(word) - len(suffix) >= 4:
                    word = word[: -len(suffix)]
                    break
        if word not in out:
            out.append(word)
    return out


def _fts5_match_expression(word_list: list[str]) -> str:
    if not word_list:
        return ""
    parts = []
    for w in word_list:
        quoted = '"' + w + '"'
        if w.isalpha() and len(w) >= 4:
            quoted += "*"
        parts.append(quoted)
    return " OR ".join(parts)


def _build_fts5_index(corpus: str, db_path: str) -> str:
    source = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=5)
    try:
        rows = source.execute("SELECT id, title, summary, content FROM entries").fetchall()
    finally:
        source.close()
    path = f"/tmp/kb-fts5-{corpus}.db"
    db = sqlite3.connect(path)
    db.execute("DROP TABLE IF EXISTS entries_fts")
    db.execute(
        f"CREATE VIRTUAL TABLE entries_fts USING fts5("
        f"entry_id UNINDEXED, title, summary, content, "
        f"tokenize='{FTS5_TOKENIZER}')"
    )
    db.executemany("INSERT INTO entries_fts VALUES(?,?,?,?)", rows)
    db.commit()
    db.close()
    count = len(rows)
    print(f"[fts5] Built index for {corpus}: {count} entries → {path}", flush=True)
    return path


def _fts5_query(fts5_path: str, query: str, limit: int) -> list[dict]:
    word_list = _fts5_terms(query)
    expression = _fts5_match_expression(word_list)
    if not expression:
        return []
    db = sqlite3.connect(f"file:{fts5_path}?mode=ro", uri=True, timeout=5)
    try:
        rows = db.execute(
            "SELECT entry_id, bm25(entries_fts, 0, 3, 2, 1) AS score "
            "FROM entries_fts WHERE entries_fts MATCH ? ORDER BY score, entry_id LIMIT ?",
            (expression, limit),
        ).fetchall()
    except sqlite3.OperationalError:
        return []
    finally:
        db.close()
    return [{"entry_id": row[0], "bm25": row[1]} for row in rows]


CORPUS_REGISTRY = {
    "homelab": {
        "db_path": "/opt/kb/kb.db",
        "collection": "kb_collection",
    },
    "ai": {
        "db_path": "/opt/ai-kb/ai-kb.db",
        "collection": "ai_kb_collection",
    },
}

_collection_ids: dict[str, str] = {}
_bearer = HTTPBearer(auto_error=False, scheme_name="bearerAuth")


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SearchRequestV2(StrictModel):
    query: str = Field(min_length=1, max_length=1500)
    query_alt: Optional[str] = Field(default=None, max_length=1500)
    query_alt_language: Optional[Literal["sr", "en"]] = None
    scope: ScopeName
    top_k: int = Field(default=5, ge=1, le=5, strict=True)
    allow_degraded: StrictBool = False

    @field_validator("query")
    @classmethod
    def trim_nonempty_query(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("query must contain a non-whitespace character")
        return value

    @field_validator("query_alt")
    @classmethod
    def trim_nonempty_query_alt(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        value = value.strip()
        if not value:
            raise ValueError("query_alt must contain a non-whitespace character")
        return value

    @model_validator(mode="before")
    @classmethod
    def require_alternate_pair(cls, data):
        if isinstance(data, dict) and (
            (data.get("query_alt") is None) != (data.get("query_alt_language") is None)
        ):
            raise ValueError("query_alt and query_alt_language must be supplied together")
        return data


class SearchResultV2(StrictModel):
    corpus: CorpusName
    entry_id: int
    ref: str = Field(pattern=r"^(homelab|ai):[1-9][0-9]*$")
    title: str
    content: Optional[str]
    tags: Optional[str]
    public_source_url: Optional[AnyHttpUrl]
    link: AnyUrl
    distance: float
    relevance: float
    final_score: float


class CorpusResultsV2(StrictModel):
    searched: bool
    available: bool
    count: int = Field(ge=0, le=5)
    results: list[SearchResultV2] = Field(max_length=5)


class CorporaV2(StrictModel):
    homelab: CorpusResultsV2
    ai: CorpusResultsV2


class SearchResponseV2(StrictModel):
    query: str
    requested_scope: ScopeName
    selected_scope: SelectedScope
    routing_mode: Literal["explicit", "auto"]
    routing_reason: str
    needs_clarification: bool
    router_version: Optional[str]
    degraded_corpora: list[CorpusName]
    total_count: int = Field(ge=0, le=10)
    corpora: CorporaV2
    # The same results as `corpora`, merged into one cross-corpus ranking. Additive:
    # `corpora` stays the frozen grouped shape for existing consumers, and callers who
    # want a single list read this one. Each item carries its own `corpus` and `ref`.
    ranked: list[SearchResultV2] = Field(default_factory=list, max_length=10)


class CorpusHealthV2(StrictModel):
    ready: bool
    collection: str
    reason: Optional[str] = None
    # Observability, not readiness: these stay None when the corpus DB cannot be
    # read, and never flip `ready`, so a metrics problem is not an outage.
    entry_count: Optional[int] = None
    pending_embed: Optional[int] = None
    last_compile: Optional[str] = None


class HealthResponseV2(StrictModel):
    status: Literal["ok", "degraded"]
    corpora: dict[CorpusName, CorpusHealthV2]
    auto_routing_enabled: bool = False


class TraceRequestV2(StrictModel):
    query: str = Field(min_length=1, max_length=1500)
    scope: Literal["homelab", "ai", "both"] = "both"
    watch_entry_ids: list[int] = Field(default_factory=list, max_length=50)


class DualQueryRequestV2(StrictModel):
    original_query: str = Field(min_length=1, max_length=1500)
    rewritten_query: str = Field(min_length=1, max_length=1500)
    scope: Literal["homelab", "ai", "both"] = "both"
    top_k: int = Field(default=5, ge=1, le=25)


@dataclass(frozen=True)
class AuthorizedClient:
    name: str
    allowed_corpora: frozenset[str]
    allowed_scopes: frozenset[str]


@dataclass(frozen=True)
class RouterConfig:
    router_version: str
    accept_thresholds: dict[str, float]
    reject_threshold: float
    both_margin: float
    dead_zone_lower: float
    dead_zone_upper: float
    candidate_k: int
    max_distance: dict[str, float]
    ai_decay_mode: str
    ai_decay_half_life_days: Optional[float] = None
    ai_decay_floor: Optional[float] = None
    fts5_enabled: bool = False
    fts5_semantic_k: int = 20
    fts5_lexical_k: int = 5


@dataclass
class Candidate:
    corpus: str
    entry_id: int
    title: str
    content: Optional[str]
    summary: Optional[str]
    tags: Optional[str]
    source: Optional[str]
    date: Optional[str]
    distance: float
    from_primary: bool = True
    from_fts5: bool = False
    alternate_rank: Optional[int] = None
    relevance: float = 0.0
    final_score: float = 0.0


def _config_path() -> str:
    return os.getenv("KB_V2_CLIENTS_CONFIG", CLIENT_CONFIG_PATH)


def _router_config_path() -> str:
    return os.getenv("KB_CORPUS_ROUTER_CONFIG", ROUTER_CONFIG_PATH)


def _number(value, name: str, minimum: float, maximum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(f"router config {name} must be numeric")
    numeric = float(value)
    if not math.isfinite(numeric) or numeric < minimum or numeric > maximum:
        raise RuntimeError(f"router config {name} is outside [{minimum}, {maximum}]")
    return numeric


def _load_router_config() -> RouterConfig:
    path = _router_config_path()
    try:
        file_stat = os.stat(path)
        if stat.S_IMODE(file_stat.st_mode) != 0o600 or file_stat.st_uid != os.geteuid():
            raise RuntimeError("router config ownership or mode is invalid")
        with open(path, encoding="utf-8") as handle:
            document = yaml.safe_load(handle)
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise RuntimeError("router config is unavailable") from exc
    required = {
        "router_version",
        "accept_thresholds",
        "reject_threshold",
        "both_margin",
        "dead_zone",
        "candidate_k",
        "max_distance",
        "ai_decay",
        "fts5",
    }
    if not isinstance(document, dict) or set(document) != required:
        raise RuntimeError("router config shape is invalid")
    version = document["router_version"]
    if not isinstance(version, str) or not re.fullmatch(r"[a-z0-9][a-z0-9._-]{2,63}", version):
        raise RuntimeError("router config version is invalid")

    def corpus_numbers(field: str, minimum: float, maximum: float) -> dict[str, float]:
        values = document[field]
        if not isinstance(values, dict) or set(values) != set(CORPUS_REGISTRY):
            raise RuntimeError(f"router config {field} shape is invalid")
        return {
            corpus: _number(values[corpus], f"{field}.{corpus}", minimum, maximum)
            for corpus in CORPUS_REGISTRY
        }

    accept = corpus_numbers("accept_thresholds", 0.0, 1.0)
    max_distance = corpus_numbers("max_distance", 0.0, 2.0)
    reject = _number(document["reject_threshold"], "reject_threshold", 0.0, 1.0)
    margin = _number(document["both_margin"], "both_margin", 0.0, 1.0)
    dead_zone = document["dead_zone"]
    if not isinstance(dead_zone, dict) or set(dead_zone) != {"lower", "upper"}:
        raise RuntimeError("router config dead_zone shape is invalid")
    dead_lower = _number(dead_zone["lower"], "dead_zone.lower", 0.0, 1.0)
    dead_upper = _number(dead_zone["upper"], "dead_zone.upper", 0.0, 1.0)
    if reject > dead_lower or dead_lower >= dead_upper or dead_upper > min(accept.values()):
        raise RuntimeError("router config thresholds are inconsistent")
    candidate_k = document["candidate_k"]
    if isinstance(candidate_k, bool) or not isinstance(candidate_k, int) or not 1 <= candidate_k <= 100:
        raise RuntimeError("router config candidate_k must be an integer in [1, 100]")

    decay = document["ai_decay"]
    if not isinstance(decay, dict) or "mode" not in decay:
        raise RuntimeError("router config ai_decay shape is invalid")
    mode = decay["mode"]
    half_life = floor = None
    if mode == "disabled":
        if set(decay) != {"mode"}:
            raise RuntimeError("disabled AI decay cannot have parameters")
    elif mode == "rational":
        if set(decay) != {"mode", "half_life_days", "floor"}:
            raise RuntimeError("rational AI decay parameters are incomplete")
        half_life = _number(decay["half_life_days"], "ai_decay.half_life_days", 1.0, 36500.0)
        floor = _number(decay["floor"], "ai_decay.floor", 0.0, 1.0)
    else:
        raise RuntimeError("router config AI decay mode is invalid")

    fts5 = document["fts5"]
    if not isinstance(fts5, dict) or "enabled" not in fts5:
        raise RuntimeError("router config fts5 shape is invalid")
    fts5_enabled = fts5["enabled"]
    if not isinstance(fts5_enabled, bool):
        raise RuntimeError("router config fts5.enabled must be boolean")
    fts5_semantic_k = 20
    fts5_lexical_k = 5
    if fts5_enabled:
        expected_fts5_keys = {"enabled", "semantic_k", "lexical_k"}
        if set(fts5) != expected_fts5_keys:
            raise RuntimeError("router config fts5 shape is invalid when enabled")
        fts5_semantic_k = fts5["semantic_k"]
        fts5_lexical_k = fts5["lexical_k"]
        if (
            isinstance(fts5_semantic_k, bool) or not isinstance(fts5_semantic_k, int)
            or not 1 <= fts5_semantic_k < candidate_k
        ):
            raise RuntimeError("router config fts5.semantic_k is invalid")
        if (
            isinstance(fts5_lexical_k, bool) or not isinstance(fts5_lexical_k, int)
            or not 1 <= fts5_lexical_k <= candidate_k - fts5_semantic_k
        ):
            raise RuntimeError("router config fts5.lexical_k is invalid")
    elif set(fts5) != {"enabled"}:
        raise RuntimeError("disabled fts5 cannot have parameters")

    config = RouterConfig(
        router_version=version,
        accept_thresholds=accept,
        reject_threshold=reject,
        both_margin=margin,
        dead_zone_lower=dead_lower,
        dead_zone_upper=dead_upper,
        candidate_k=candidate_k,
        max_distance=max_distance,
        ai_decay_mode=mode,
        ai_decay_half_life_days=half_life,
        ai_decay_floor=floor,
        fts5_enabled=fts5_enabled,
        fts5_semantic_k=fts5_semantic_k,
        fts5_lexical_k=fts5_lexical_k,
    )
    expected_precalibration = {
        "accept_thresholds": {"homelab": 0.60, "ai": 0.60},
        "reject_threshold": 0.40,
        "both_margin": 0.05,
        "dead_zone_lower": 0.40,
        "dead_zone_upper": 0.60,
        "candidate_k": 25,
        "max_distance": {"homelab": 0.60, "ai": 0.60},
        "ai_decay_mode": "disabled",
        "fts5_enabled": True,
        "fts5_semantic_k": 20,
        "fts5_lexical_k": 5,
    }
    if config.router_version != PRECALIBRATION_ROUTER_VERSION or any(
        getattr(config, key) != value for key, value in expected_precalibration.items()
    ):
        raise RuntimeError("router version is not bound to the approved effective values")
    return config


def _load_clients() -> list[tuple[AuthorizedClient, str]]:
    """Load and fully validate the token-name allowlist; token values stay in env."""
    try:
        path = _config_path()
        file_stat = os.stat(path)
        if stat.S_IMODE(file_stat.st_mode) != 0o600 or file_stat.st_uid != os.geteuid():
            raise RuntimeError("client config ownership or mode is invalid")
        with open(path, encoding="utf-8") as handle:
            document = yaml.safe_load(handle)
    except (OSError, RuntimeError, UnicodeError, yaml.YAMLError) as exc:
        raise HTTPException(status_code=503, detail="V2 client authorization is unavailable") from exc

    if not isinstance(document, dict) or set(document) != {"clients"}:
        raise HTTPException(status_code=503, detail="V2 client authorization is invalid")
    clients = document["clients"]
    if not isinstance(clients, dict) or not clients:
        raise HTTPException(status_code=503, detail="V2 client authorization is invalid")

    resolved: list[tuple[AuthorizedClient, str]] = []
    seen_tokens: set[str] = set()
    valid_corpora = set(CORPUS_REGISTRY)
    valid_scopes = {"homelab", "ai", "both", "auto"}
    for name, raw in clients.items():
        if not isinstance(name, str) or not name or not isinstance(raw, dict):
            raise HTTPException(status_code=503, detail="V2 client authorization is invalid")
        if set(raw) != {"token_env", "allowed_corpora", "allowed_scopes"}:
            raise HTTPException(status_code=503, detail="V2 client authorization is invalid")
        token_env = raw["token_env"]
        corpora = raw["allowed_corpora"]
        scopes = raw["allowed_scopes"]
        if (
            not isinstance(token_env, str)
            or not re.fullmatch(r"KB_V2_TOKEN_[A-Z0-9_]{3,64}", token_env)
            or not isinstance(corpora, list)
            or not corpora
            or not all(isinstance(value, str) for value in corpora)
            or len(corpora) != len(set(corpora))
            or not set(corpora) <= valid_corpora
            or not isinstance(scopes, list)
            or not scopes
            or not all(isinstance(value, str) for value in scopes)
            or len(scopes) != len(set(scopes))
            or not set(scopes) <= valid_scopes
        ):
            raise HTTPException(status_code=503, detail="V2 client authorization is invalid")
        if "homelab" in scopes and "homelab" not in corpora:
            raise HTTPException(status_code=503, detail="V2 client authorization is invalid")
        if "ai" in scopes and "ai" not in corpora:
            raise HTTPException(status_code=503, detail="V2 client authorization is invalid")
        if "both" in scopes and set(corpora) != valid_corpora:
            raise HTTPException(status_code=503, detail="V2 client authorization is invalid")
        token = os.getenv(token_env)
        if (
            not token
            or not re.fullmatch(r"[A-Za-z0-9._~-]{32,256}", token)
            or token in seen_tokens
        ):
            raise HTTPException(status_code=503, detail="V2 client authorization is invalid")
        seen_tokens.add(token)
        resolved.append((AuthorizedClient(name, frozenset(corpora), frozenset(scopes)), token))
    return resolved


def authorize_v2(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer),
) -> AuthorizedClient:
    return _authorize_credentials(credentials, _load_clients())


def _authorize_credentials(
    credentials: Optional[HTTPAuthorizationCredentials],
    configured_clients: list[tuple[AuthorizedClient, str]],
) -> AuthorizedClient:
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise HTTPException(
            status_code=401,
            detail="Missing or invalid bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    presented = credentials.credentials
    if not re.fullmatch(r"[A-Za-z0-9._~-]{32,256}", presented or ""):
        raise HTTPException(
            status_code=401,
            detail="Missing or invalid bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    match: Optional[AuthorizedClient] = None
    for client, expected in configured_clients:
        if hmac.compare_digest(presented, expected):
            match = client
    if match is None:
        raise HTTPException(
            status_code=401,
            detail="Missing or invalid bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return match


def _collection_descriptor(corpus: str, force_refresh: bool = False) -> dict:
    profile = CORPUS_REGISTRY[corpus]
    response = httpx.get(
        f"{CHROMA_BASE}/collections/{profile['collection']}", timeout=10
    )
    if response.status_code != 200:
        raise RuntimeError(f"collection_lookup_http_{response.status_code}")
    descriptor = response.json()
    collection_id = descriptor.get("id")
    if not isinstance(collection_id, str) or not collection_id:
        raise RuntimeError("collection descriptor has an invalid id")
    # A name lookup is authoritative even if an old UUID still returns 200.
    _collection_ids[corpus] = collection_id
    return descriptor


def _corpus_metrics(corpus: str) -> dict:
    """Entry count, embed backlog and last compile — never raises.

    Read-only and best effort: metrics are not readiness, so a locked or
    missing database returns Nones instead of turning a healthy corpus into a
    degraded one. `last_compile` stays None until compile.py starts recording
    it; the field is published now so the dashboard mapping does not have to
    change twice.
    """
    empty = {"entry_count": None, "pending_embed": None, "last_compile": None}
    profile = CORPUS_REGISTRY[corpus]
    try:
        db = sqlite3.connect(f"file:{profile['db_path']}?mode=ro", uri=True, timeout=2)
    except sqlite3.Error:
        return empty
    try:
        total, pending = db.execute(
            "SELECT COUNT(*), SUM(embedded_at IS NULL) FROM entries"
        ).fetchone()
        last_compile = None
        try:
            row = db.execute(
                "SELECT value FROM kb_meta WHERE key='last_compile'"
            ).fetchone()
            last_compile = row[0] if row else None
        except sqlite3.Error:
            # Table arrives with the compile.py change; absent is not an error.
            pass
        return {
            "entry_count": total,
            "pending_embed": pending or 0,
            "last_compile": last_compile,
        }
    except sqlite3.Error:
        return empty
    finally:
        db.close()


def _corpus_health(corpus: str, reranker_ready: bool) -> CorpusHealthV2:
    profile = CORPUS_REGISTRY[corpus]
    try:
        db = sqlite3.connect(f"file:{profile['db_path']}?mode=ro", uri=True)
        try:
            if db.execute("PRAGMA quick_check").fetchone()[0] != "ok":
                raise RuntimeError("sqlite_integrity")
            db.execute("SELECT id FROM entries LIMIT 1").fetchall()
        finally:
            db.close()
        descriptor = _collection_descriptor(corpus)
        metadata = descriptor.get("metadata") or {}
        expected = {
            "hnsw:space": "cosine",
            "corpus": corpus,
            "embedding_model": EMBEDDING_MODEL,
            "embedding_dimension": EMBEDDING_DIMENSION,
            "schema_version": COLLECTION_SCHEMA_VERSION,
        }
        if any(metadata.get(key) != value for key, value in expected.items()):
            raise RuntimeError("collection_metadata_mismatch")
        if not metadata.get("created_at"):
            raise RuntimeError("collection_metadata_mismatch")
        space = ((descriptor.get("configuration_json") or {}).get("hnsw") or {}).get("space")
        if space != "cosine":
            raise RuntimeError("collection_metric_mismatch")
        if not reranker_ready:
            raise RuntimeError("reranker_unavailable")
        return CorpusHealthV2(
            ready=True, collection=profile["collection"], **_corpus_metrics(corpus)
        )
    except Exception as exc:
        reason = str(exc)
        if not reason or "/" in reason or "\\" in reason:
            reason = "dependency_unavailable"
        # A degraded corpus is exactly when the counts are worth seeing, so they
        # are reported here too — still best effort.
        return CorpusHealthV2(
            ready=False,
            collection=profile["collection"],
            reason=reason[:120],
            **_corpus_metrics(corpus),
        )


def _query_collection(
    corpus: str,
    embedding: list[float],
    router_config: RouterConfig,
) -> list[dict]:
    response = None
    for attempt in range(2):
        if attempt or corpus not in _collection_ids:
            _collection_descriptor(corpus, force_refresh=bool(attempt))
        cid = _collection_ids[corpus]
        count_resp = httpx.get(f"{CHROMA_BASE}/collections/{cid}/count", timeout=10)
        if count_resp.status_code != 200:
            raise RuntimeError(f"collection_count_http_{count_resp.status_code}")
        n_results = count_resp.json()
        if not isinstance(n_results, int) or n_results < 1:
            return []
        response = httpx.post(
            f"{CHROMA_BASE}/collections/{cid}/query",
            json={
                "query_embeddings": [embedding],
                "n_results": n_results,
                "include": ["distances", "documents", "metadatas"],
            },
            timeout=30,
        )
        if response.status_code != 404:
            break
    if response is None or response.status_code != 200:
        status = response.status_code if response is not None else "unavailable"
        raise RuntimeError(f"collection_query_http_{status}")
    value = response.json()
    ids_outer = value.get("ids")
    distances_outer = value.get("distances")
    if (
        not isinstance(ids_outer, list)
        or len(ids_outer) != 1
        or not isinstance(ids_outer[0], list)
        or not isinstance(distances_outer, list)
        or len(distances_outer) != 1
        or not isinstance(distances_outer[0], list)
        or len(ids_outer[0]) != len(distances_outer[0])
    ):
        raise RuntimeError("collection query response shape is invalid")
    ids = ids_outer[0]
    distances = distances_outer[0]
    candidates = []
    for index, raw_id in enumerate(ids):
        entry_id = str(raw_id)
        if not entry_id.isdigit() or int(entry_id) < 1:
            continue
        distance = distances[index]
        if isinstance(distance, bool) or not isinstance(distance, (int, float)):
            raise RuntimeError("collection query returned an invalid distance")
        distance = float(distance)
        if not math.isfinite(distance):
            raise RuntimeError("collection query returned a non-finite distance")
        if distance > router_config.max_distance[corpus]:
            continue
        candidates.append({"entry_id": int(entry_id), "distance": round(float(distance), 4)})
    candidates.sort(key=lambda item: item["distance"])
    return candidates[:router_config.candidate_k]


def _query_collection_raw(
    corpus: str,
    embedding: list[float],
) -> tuple[list[dict], int]:
    """ChromaDB full-scan without distance or candidate_k filtering. For eval trace only."""
    response = None
    for attempt in range(2):
        if attempt or corpus not in _collection_ids:
            _collection_descriptor(corpus, force_refresh=bool(attempt))
        cid = _collection_ids[corpus]
        count_resp = httpx.get(f"{CHROMA_BASE}/collections/{cid}/count", timeout=10)
        if count_resp.status_code != 200:
            raise RuntimeError(f"collection_count_http_{count_resp.status_code}")
        n_results = count_resp.json()
        if not isinstance(n_results, int) or n_results < 1:
            return [], 0
        response = httpx.post(
            f"{CHROMA_BASE}/collections/{cid}/query",
            json={
                "query_embeddings": [embedding],
                "n_results": n_results,
                "include": ["distances"],
            },
            timeout=30,
        )
        if response.status_code != 404:
            break
    if response is None or response.status_code != 200:
        status = response.status_code if response is not None else "unavailable"
        raise RuntimeError(f"collection_query_http_{status}")
    value = response.json()
    ids_outer = value.get("ids")
    distances_outer = value.get("distances")
    if (
        not isinstance(ids_outer, list)
        or len(ids_outer) != 1
        or not isinstance(distances_outer, list)
        or len(distances_outer) != 1
    ):
        raise RuntimeError("collection query response shape is invalid")
    ids = ids_outer[0]
    distances = distances_outer[0]
    candidates = []
    for index, raw_id in enumerate(ids):
        entry_id = str(raw_id)
        if not entry_id.isdigit() or int(entry_id) < 1:
            continue
        distance = float(distances[index])
        if not math.isfinite(distance):
            continue
        candidates.append({"entry_id": int(entry_id), "distance": round(distance, 4)})
    candidates.sort(key=lambda item: item["distance"])
    return candidates, n_results


def _fetch_candidates(corpus: str, raw: list[dict]) -> list[Candidate]:
    if not raw:
        return []
    ids = [item["entry_id"] for item in raw]
    placeholders = ",".join("?" for _ in ids)
    profile = CORPUS_REGISTRY[corpus]
    db = sqlite3.connect(f"file:{profile['db_path']}?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    try:
        rows = db.execute(
            f"SELECT id, title, content, summary, tags, source, created_at "
            f"FROM entries WHERE id IN ({placeholders})",
            ids,
        ).fetchall()
    finally:
        db.close()
    row_map = {row["id"]: row for row in rows}
    output = []
    for item in raw:
        row = row_map.get(item["entry_id"])
        if row is None:
            continue
        output.append(Candidate(
            corpus=corpus,
            entry_id=row["id"],
            title=row["title"] or "Untitled",
            content=row["content"] or None,
            summary=row["summary"] or None,
            tags=row["tags"] or None,
            source=row["source"] or None,
            date=row["created_at"],
            distance=item["distance"],
            from_primary=item.get("from_primary", True),
            from_fts5=item.get("from_fts5", False),
            alternate_rank=item.get("alternate_rank"),
        ))
    return output


def _merge_fts5_candidates(
    semantic: list[dict],
    fts5_raw: list[dict],
    router_config: RouterConfig,
) -> list[dict]:
    semantic_k = router_config.fts5_semantic_k
    lexical_k = router_config.fts5_lexical_k
    max_dist = max(router_config.max_distance.values())

    chosen = list(semantic[:semantic_k])
    chosen_ids = {c["entry_id"] for c in chosen}

    added = 0
    for fc in fts5_raw:
        if added >= lexical_k:
            break
        if fc["entry_id"] not in chosen_ids:
            chosen.append({
                "entry_id": fc["entry_id"],
                "distance": max_dist,
                "from_fts5": True,
            })
            chosen_ids.add(fc["entry_id"])
            added += 1

    for sc in semantic[semantic_k:]:
        if len(chosen) >= router_config.candidate_k:
            break
        if sc["entry_id"] not in chosen_ids:
            chosen.append(sc)
            chosen_ids.add(sc["entry_id"])

    return chosen


def _retrieve_corpus(
    corpus: str,
    embedding: list[float],
    router_config: RouterConfig,
    alternate_embedding: Optional[list[float]] = None,
    alternate_only_rank_limit: Optional[int] = ALTERNATE_ONLY_RANK_LIMIT,
    fts5_path: Optional[str] = None,
    query_text: Optional[str] = None,
) -> list[Candidate]:
    primary = _query_collection(corpus, embedding, router_config)

    use_fts5 = (
        router_config.fts5_enabled
        and fts5_path is not None
        and query_text is not None
    )

    if alternate_embedding is None:
        if use_fts5:
            fts5_raw = _fts5_query(fts5_path, query_text, FTS5_FETCH_LIMIT)
            primary = _merge_fts5_candidates(primary, fts5_raw, router_config)
        return _fetch_candidates(corpus, primary)

    alternate = _query_collection(corpus, alternate_embedding, router_config)
    merged = _union_candidates(primary, alternate)
    if alternate_only_rank_limit is not None:
        merged = [
            item for item in merged
            if item["from_primary"] or item["alternate_rank"] <= alternate_only_rank_limit
        ]
    if use_fts5:
        fts5_raw = _fts5_query(fts5_path, query_text, FTS5_FETCH_LIMIT)
        merged = _merge_fts5_candidates(merged, fts5_raw, router_config)
    return _fetch_candidates(corpus, merged)


def _union_candidates(primary: list[dict], alternate: list[dict]) -> list[dict]:
    """Union two same-collection candidate lists using the measured rule."""
    best: dict[int, dict] = {}
    for rank, candidate in enumerate(primary, 1):
        entry_id = candidate["entry_id"]
        best[entry_id] = {
            **candidate,
            "from_primary": True,
            "alternate_rank": None,
        }
    for rank, candidate in enumerate(alternate, 1):
        entry_id = candidate["entry_id"]
        previous = best.get(entry_id)
        if previous is None:
            best[entry_id] = {
                **candidate,
                "from_primary": False,
                "alternate_rank": rank,
            }
            continue
        previous["alternate_rank"] = rank
        if candidate["distance"] < previous["distance"]:
            previous["distance"] = candidate["distance"]
    # The eval reference stores ids as strings, so use the same canonical tie-break.
    return sorted(best.values(), key=lambda item: (item["distance"], str(item["entry_id"])))


def _reranker_passage(query: str, content: str, model) -> str:
    """Select by query-term coverage; tied coverage favors central evidence; no evidence prefers the prefix.

    Keep one model prediction per document. Do not synthesize or translate source
    text. The original query and original full candidate remain unchanged.
    """
    if not content:
        return content
    tokenizer = model.tokenizer
    limit = model.max_seq_length
    special = tokenizer.num_special_tokens_to_add(pair=True)
    query_length = len(tokenizer.encode(query, add_special_tokens=False, verbose=False))
    # Match longest-first pair truncation for queries exceeding half the budget.
    budget = limit - special - min(query_length, (limit - special + 1) // 2)
    if budget < 1:
        raise ValueError('reranker has no document token budget')
    offsets = tokenizer(content, add_special_tokens=False, return_offsets_mapping=True,
                        truncation=False, verbose=False)['offset_mapping']
    if len(offsets) <= budget:
        return content
    terms = _fts5_terms(query)
    windows = []
    for start in range(0, len(offsets), max(1, budget // 2)):
        end = min(start + budget, len(offsets))
        text = content[offsets[start][0]:offsets[end - 1][1]]
        # A slice starting inside a word may tokenize differently in isolation.
        while end > start + 1 and len(tokenizer.encode(text, add_special_tokens=False, verbose=False)) > budget:
            end -= 1
            text = content[offsets[start][0]:offsets[end - 1][1]]
        words = set(_fts5_terms(text))
        hits = {term for term in terms if any(word.startswith(term) for word in words)}
        windows.append((text, hits))
    frequencies = {term: sum(term in hits for _, hits in windows) for term in terms}
    scores = []
    for text, hits in windows:
        coverage = sum(math.log(1 + len(windows) / (1 + frequencies[term])) for term in sorted(hits))
        folded = ''.join(c for c in unicodedata.normalize('NFD', text.lower())
                         if unicodedata.category(c) != 'Mn')
        centrality = sum(max((1 - abs((match.start() + len(term) / 2) / max(1, len(folded)) - .5) * 2
                             for match in re.finditer(r'\b' + re.escape(term), folded)), default=0)
                         for term in sorted(hits))
        scores.append((coverage, centrality))
    best = max(range(len(windows)), key=lambda index: scores[index])
    return windows[best][0]


def _rerank_batch(query: str, candidates: list[Candidate], model) -> None:
    if not candidates:
        return
    pairs = [(query, _reranker_passage(query, item.content or item.summary or "", model)) for item in candidates]
    raw_scores = list(model.predict(pairs))
    if len(raw_scores) != len(candidates):
        raise RuntimeError("reranker score cardinality mismatch")
    relevance_scores = []
    for score in raw_scores:
        numeric = float(score)
        if not math.isfinite(numeric):
            raise RuntimeError("reranker returned a non-finite score")
        try:
            relevance = 1.0 / (1.0 + math.exp(-numeric))
        except OverflowError:
            relevance = 1.0 if numeric > 0 else 0.0
        relevance_scores.append(round(relevance, 4))
    for item, relevance in zip(candidates, relevance_scores):
        item.relevance = relevance


def _admit_alternate_only_candidates(
    candidates: list[Candidate], reject_threshold: float
) -> list[Candidate]:
    """Apply the sealed A1 alternate-only admission rule after one rerank pass."""
    primary_has_result = any(
        candidate.corpus == "homelab"
        and candidate.from_primary
        and candidate.relevance >= reject_threshold
        for candidate in candidates
    )
    admitted: list[Candidate] = []
    for candidate in candidates:
        if candidate.corpus != "homelab" or candidate.from_primary:
            admitted.append(candidate)
            continue
        if candidate.alternate_rank is None or candidate.alternate_rank > 5:
            continue
        if not primary_has_result and candidate.relevance < 0.94:
            continue
        admitted.append(candidate)
    return admitted


def _apply_decay(candidate: Candidate, router_config: RouterConfig) -> None:
    if candidate.corpus == "ai" and router_config.ai_decay_mode == "disabled":
        candidate.final_score = candidate.relevance
        return
    days_old = 0.0
    if candidate.date:
        try:
            created = datetime.fromisoformat(candidate.date)
            if created.tzinfo is None:
                created = created.replace(tzinfo=timezone.utc)
            days_old = max((datetime.now(timezone.utc) - created).total_seconds() / 86400.0, 0)
        except (TypeError, ValueError):
            pass
    if candidate.corpus == "ai":
        half_life = router_config.ai_decay_half_life_days
        floor = router_config.ai_decay_floor
        if half_life is None or floor is None:
            raise RuntimeError("AI decay config is incomplete")
    else:
        half_life = HOMELAB_DECAY_HALF_LIFE
        floor = HOMELAB_DECAY_FLOOR
    decay = max(1.0 / (1.0 + days_old / half_life), floor)
    candidate.final_score = round(candidate.relevance * decay, 4)


def _public_url(source: Optional[str]) -> Optional[str]:
    if not source:
        return None
    parsed = urlparse(source)
    if parsed.scheme in {"http", "https"} and parsed.netloc:
        return source
    return None


def _result(candidate: Candidate) -> SearchResultV2:
    public_source_url = _public_url(candidate.source)
    return SearchResultV2(
        corpus=candidate.corpus,
        entry_id=candidate.entry_id,
        ref=f"{candidate.corpus}:{candidate.entry_id}",
        title=candidate.title,
        content=candidate.content or candidate.summary,
        tags=candidate.tags,
        public_source_url=public_source_url,
        link=public_source_url or f"kb://{candidate.corpus}/{candidate.entry_id}",
        distance=candidate.distance,
        relevance=candidate.relevance,
        final_score=candidate.final_score,
    )


def _empty_corpora() -> dict[str, CorpusResultsV2]:
    return {
        name: CorpusResultsV2(searched=False, available=False, count=0, results=[])
        for name in CORPUS_REGISTRY
    }


_SR_DIACRITICS = frozenset("čćšžđČĆŠŽĐ")
_SR_MARKERS = frozenset({
    "kako", "sta", "šta", "zasto", "zašto", "koji", "koja", "koje", "gde", "kada",
    "je", "su", "na", "za", "sa", "ili", "nije", "ima", "sam", "treba", "moze",
    "može", "radi", "posle", "pre", "bez", "svi", "sve", "ovo", "taj", "ta",
})
_EN_MARKERS = frozenset({
    "how", "what", "why", "which", "where", "when", "the", "is", "are", "on",
    "for", "with", "or", "not", "has", "have", "does", "do", "should", "can",
    "after", "before", "without", "all", "this", "that", "it", "in", "of", "to",
})
# Technical queries are usually a service name plus one noun — no function
# words and, in practice, no diacritics either, so they used to land in
# "unknown" and understate how much is actually asked in Serbian.
_SR_TECH_MARKERS = frozenset({
    "servis", "servisi", "servisa", "greska", "greske", "mreza", "mreze",
    "putanja", "putanje", "verzija", "verzije", "provera", "izmena", "izmene",
    "datoteka", "fajl", "fajlovi", "lozinka", "dozvole", "korisnik", "pretraga",
    "upit", "upiti", "unos", "unosi", "baza", "brzina", "memorija", "kopija",
    "nadogradnja", "pokretanje", "gasenje", "brisanje",
})
# Endings that carry no English collision. "-ost" is deliberately absent:
# host, post, most, cost and lost would all match it.
_SR_TECH_SUFFIXES = ("acija", "acije", "aciju", "anje", "anja", "enje", "enja")
_SR_SUFFIX_MIN_LENGTH = 6


def _query_language(query: str) -> str:
    """Grubo odredi jezik upita za audit — u log ide oznaka, nikad tekst upita.

    Namena je raspodela za kalibraciju rutera, ne tacna klasifikacija, pa je
    heuristika namerno bez spoljnih zavisnosti: dijakritika je jednoznacan
    signal, a inace odlucuje prevaga markera. Bez signala vraca "unknown"
    umesto da pogadja.

    Uz funkcijske reci racunaju se i tehnicke imenice i srpski nastavci, jer
    upit tipa "Pi-hole WireGuard DNS konfiguracija" nema ni dijakritiku ni
    funkcijsku rec, a nesumnjivo je srpski. Rec se broji jednom bez obzira na
    to koliko pravila je uhvati.
    """
    if not query:
        return "unknown"
    if any(ch in _SR_DIACRITICS for ch in query):
        return "sr"
    words = {w.strip(".,:;?!()[]\"'").lower() for w in query.split()}
    sr_words = (words & _SR_MARKERS) | (words & _SR_TECH_MARKERS) | {
        word
        for word in words
        if len(word) >= _SR_SUFFIX_MIN_LENGTH and word.endswith(_SR_TECH_SUFFIXES)
    }
    sr_hits = len(sr_words)
    en_hits = len(words & _EN_MARKERS)
    if sr_hits > en_hits:
        return "sr"
    if en_hits > sr_hits:
        return "en"
    return "unknown"


@dataclass(frozen=True)
class ShadowRoute:
    scope: SelectedScope
    reason: str


def _shadow_route(best: dict[str, float], config: RouterConfig) -> ShadowRoute:
    """Decide what `auto` would have selected, without applying it.

    Implements plan 4.2 points 6-8 plus revision #2, on pre-decay cross-encoder
    relevance. Two boundary cases the plan leaves open are resolved here and
    given their own reasons, so calibration can see how often they fire instead
    of having them silently folded into a neighbouring branch:

    - exactly one corpus above the accept threshold but the two within the
      margin — read as point 7 ("difference below the margin") rather than
      point 6, which requires a sufficient margin;
    - one corpus in the dead zone while the other is below the reject
      threshold — kept as that corpus rather than `none`, since a candidate
      above reject is not "no strong enough candidate" in the sense of point 8.
    """
    homelab = best.get("homelab")
    ai = best.get("ai")
    if homelab is None and ai is None:
        return ShadowRoute("none", "no_candidate")

    scores = {"homelab": homelab, "ai": ai}
    above_accept = [
        name
        for name, value in scores.items()
        if value is not None and value >= config.accept_thresholds[name]
    ]
    above_reject = [
        name for name, value in scores.items() if value is not None and value >= config.reject_threshold
    ]
    if not above_reject:
        return ShadowRoute("none", "no_candidate")

    # A margin comparison needs both sides; with one corpus missing entirely
    # there is nothing to be within the margin of.
    both_present = homelab is not None and ai is not None
    within_margin = both_present and abs(homelab - ai) < config.both_margin

    if len(above_accept) == 2:
        return ShadowRoute("both", "both_strong")
    if len(above_accept) == 1:
        if within_margin:
            return ShadowRoute("both", "margin_tie")
        return ShadowRoute(above_accept[0], "single_strong")  # type: ignore[arg-type]

    in_dead_zone = [
        name
        for name, value in scores.items()
        if value is not None and config.dead_zone_lower <= value < config.dead_zone_upper
    ]
    if len(in_dead_zone) == 2 and within_margin:
        return ShadowRoute("both", "dead_zone_both")
    if len(above_reject) == 2 and within_margin:
        return ShadowRoute("both", "margin_tie")

    leader = max(above_reject, key=lambda name: scores[name])
    return ShadowRoute(leader, "single_above_reject")  # type: ignore[arg-type]


def _audit(event: str, **fields) -> None:
    safe = {"event": event, **fields}
    print("[v2-audit] " + json.dumps(safe, sort_keys=True, separators=(",", ":")), flush=True)


def create_v2_app(
    embed: Callable[[str], list[float]],
    reranker: Callable[[], object],
    union_enabled: bool = False,
    alternate_only_rank_limit: Optional[int] = ALTERNATE_ONLY_RANK_LIMIT,
) -> FastAPI:
    try:
        client_snapshot = _load_clients()
    except HTTPException:
        client_snapshot = None
    try:
        router_snapshot = _load_router_config()
    except RuntimeError:
        router_snapshot = None

    fts5_paths: dict[str, str] = {}
    if router_snapshot is not None and router_snapshot.fts5_enabled:
        for corpus_name, profile in CORPUS_REGISTRY.items():
            try:
                fts5_paths[corpus_name] = _build_fts5_index(corpus_name, profile["db_path"])
            except Exception as exc:
                print(f"[fts5] WARNING: {corpus_name}: {exc}", flush=True)

    def authorize_snapshot(
        credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer),
    ) -> AuthorizedClient:
        if client_snapshot is None:
            raise HTTPException(
                status_code=503,
                detail="V2 client authorization is unavailable",
            )
        return _authorize_credentials(credentials, client_snapshot)

    v2 = FastAPI(
        title="KB multi-corpus API",
        version="2.0.0",
        docs_url=None,
        redoc_url=None,
    )

    @v2.get("/health", response_model=HealthResponseV2)
    def health_v2(client: AuthorizedClient = Depends(authorize_snapshot)):
        model = reranker()
        router_ready = router_snapshot is not None
        corpora = {
            corpus: (
                _corpus_health(corpus, model is not None)
                if router_ready
                else CorpusHealthV2(
                    ready=False,
                    collection=CORPUS_REGISTRY[corpus]["collection"],
                    reason="router_config_unavailable",
                )
            )
            for corpus in sorted(client.allowed_corpora)
        }
        status = "ok" if all(item.ready for item in corpora.values()) else "degraded"
        return HealthResponseV2(status=status, corpora=corpora)

    @v2.post("/kb/search", response_model=SearchResponseV2)
    def search_v2(
        request: SearchRequestV2,
        client: AuthorizedClient = Depends(authorize_snapshot),
    ):
        started = time.perf_counter()
        if request.scope not in client.allowed_scopes:
            raise HTTPException(status_code=403, detail="Scope is not allowed for this client")
        if router_snapshot is None:
            raise HTTPException(
                status_code=503,
                detail={"reason": "router_config_unavailable"},
            )
        router_config = router_snapshot
        if request.scope == "auto":
            raise HTTPException(
                status_code=409,
                detail={"reason": "auto_routing_not_enabled"},
            )
        requested = [request.scope] if request.scope in CORPUS_REGISTRY else ["homelab", "ai"]
        model = reranker()
        health = {name: _corpus_health(name, model is not None) for name in requested}
        degraded = [name for name, value in health.items() if not value.ready]
        if degraded and (request.scope != "both" or not request.allow_degraded):
            _audit("search", client=client.name, scope=request.scope, status=503, degraded=degraded)
            raise HTTPException(
                status_code=503,
                detail={"reason": "required_corpus_unavailable", "corpora": degraded},
            )
        searchable = [name for name in requested if health[name].ready]
        if not searchable:
            raise HTTPException(status_code=503, detail={"reason": "all_corpora_unavailable"})

        forms_supplied = 2 if request.query_alt is not None else 1
        use_alt = (
            union_enabled
            and request.query_alt is not None
            and request.query_alt != request.query
            and "homelab" in searchable
        )
        try:
            embedding = embed(request.query)
            alternate_embedding = embed(request.query_alt) if use_alt else None
        except Exception as exc:
            _audit("search", client=client.name, scope=request.scope, status=503, failure="retrieval")
            raise HTTPException(status_code=503, detail={"reason": "retrieval_unavailable"}) from exc
        candidates = []
        retrieval_failures = []
        with ThreadPoolExecutor(max_workers=len(searchable)) as executor:
            futures = {
                executor.submit(
                    _retrieve_corpus,
                    corpus,
                    embedding,
                    router_config,
                    alternate_embedding if corpus == "homelab" else None,
                    alternate_only_rank_limit,
                    fts5_paths.get(corpus),
                    request.query,
                ): corpus
                for corpus in searchable
            }
            for future in as_completed(futures):
                corpus = futures[future]
                try:
                    candidates.extend(future.result())
                except Exception:
                    retrieval_failures.append(corpus)
        if retrieval_failures:
            if request.scope != "both" or not request.allow_degraded:
                _audit(
                    "search",
                    client=client.name,
                    scope=request.scope,
                    status=503,
                    failure="retrieval",
                    degraded=sorted(retrieval_failures),
                )
                raise HTTPException(
                    status_code=503,
                    detail={
                        "reason": "required_corpus_unavailable",
                        "corpora": sorted(retrieval_failures),
                    },
                )
            degraded.extend(name for name in retrieval_failures if name not in degraded)
            searchable = [name for name in searchable if name not in retrieval_failures]
            candidates = [item for item in candidates if item.corpus in searchable]
        if not searchable:
            raise HTTPException(status_code=503, detail={"reason": "all_corpora_unavailable"})
        degraded = [name for name in CORPUS_REGISTRY if name in degraded]
        try:
            _rerank_batch(request.query, candidates, model)
        except Exception as exc:
            _audit("search", client=client.name, scope=request.scope, status=503, failure="rerank")
            raise HTTPException(status_code=503, detail={"reason": "reranker_unavailable"}) from exc
        if use_alt:
            candidates = _admit_alternate_only_candidates(
                candidates, router_config.reject_threshold
            )

        # Best pre-decay relevance per corpus, taken before the reject filter
        # below drops the weak candidates: the shadow router has to see the
        # true best score of each corpus, including one that scores badly.
        best_relevance: dict[str, float] = {}
        for candidate in candidates:
            current = best_relevance.get(candidate.corpus)
            if current is None or candidate.relevance > current:
                best_relevance[candidate.corpus] = candidate.relevance
        # Only meaningful when both corpora were actually searched — otherwise
        # one side has no score and the comparison would be against nothing.
        shadow = (
            _shadow_route(best_relevance, router_config)
            if len(searchable) == len(CORPUS_REGISTRY)
            else None
        )

        by_corpus: dict[str, list[Candidate]] = {name: [] for name in CORPUS_REGISTRY}
        for candidate in candidates:
            if candidate.relevance < router_config.reject_threshold:
                continue
            _apply_decay(candidate, router_config)
            by_corpus[candidate.corpus].append(candidate)
        corpora = _empty_corpora()
        # Kept alongside the per-corpus grouping to build `ranked` below: the same
        # candidates, carrying the score the grouping throws away.
        scored: list[tuple[float, SearchResultV2]] = []
        for corpus in searchable:
            ranked = sorted(by_corpus[corpus], key=lambda item: -item.final_score)[:request.top_k]
            results = [_result(item) for item in ranked]
            scored.extend(zip((item.final_score for item in ranked), results))
            corpora[corpus] = CorpusResultsV2(
                searched=True,
                available=True,
                count=len(results),
                results=results,
            )
        for corpus in degraded:
            corpora[corpus] = CorpusResultsV2(
                searched=False,
                available=False,
                count=0,
                results=[],
            )

        selected: SelectedScope
        if request.scope == "both" and len(searchable) == 1:
            selected = searchable[0]  # type: ignore[assignment]
        else:
            selected = request.scope  # type: ignore[assignment]
        reason = "degraded_explicit_both" if degraded else "explicit_scope"
        response_value = SearchResponseV2(
            query=request.query,
            requested_scope=request.scope,
            selected_scope=selected,
            routing_mode="explicit",
            routing_reason=reason,
            needs_clarification=False,
            router_version=None,
            degraded_corpora=degraded,
            total_count=sum(item.count for item in corpora.values()),
            corpora=CorporaV2(**corpora),
            # Same entries as `corpora`, ordered across corpora instead of within one.
            # Legitimate because every candidate is scored by a single _rerank_batch()
            # call over both corpora, so the scores share a scale. Ties keep the
            # per-corpus order, which is registry order — deterministic either way.
            ranked=[result for _, result in sorted(scored, key=lambda pair: -pair[0])],
        )
        _audit(
            "search",
            client=client.name,
            scope=request.scope,
            selected_scope=selected,
            qlang=_query_language(request.query),
            query_alt_language=request.query_alt_language,
            forms_supplied=forms_supplied,
            forms_used=2 if use_alt else 1,
            status=200,
            degraded=degraded,
            homelab_count=corpora["homelab"].count,
            ai_count=corpora["ai"].count,
            fts5_candidates=sum(1 for c in candidates if c.from_fts5),
            elapsed_ms=round((time.perf_counter() - started) * 1000, 1),
            **(
                {
                    "shadow_scope": shadow.scope,
                    "shadow_reason": shadow.reason,
                    "shadow_homelab": round(best_relevance["homelab"], 4)
                    if "homelab" in best_relevance
                    else None,
                    "shadow_ai": round(best_relevance["ai"], 4) if "ai" in best_relevance else None,
                }
                if shadow is not None
                else {}
            ),
        )
        return response_value

    @v2.post("/kb/trace")
    def trace_v2(
        request: TraceRequestV2,
        client: AuthorizedClient = Depends(authorize_snapshot),
    ):
        """Per-stage retrieval trace for eval diagnostics."""
        started = time.perf_counter()
        if router_snapshot is None:
            raise HTTPException(status_code=503, detail="router_config_unavailable")
        rc = router_snapshot
        model = reranker()
        if model is None:
            raise HTTPException(status_code=503, detail="reranker_unavailable")

        embedding = embed(request.query)
        corpora_to_search = (
            [request.scope] if request.scope != "both" else list(CORPUS_REGISTRY)
        )
        watch_set = set(request.watch_entry_ids)

        trace_corpora = {}
        for corpus in corpora_to_search:
            raw_all, total = _query_collection_raw(corpus, embedding)
            after_distance = [
                c for c in raw_all if c["distance"] <= rc.max_distance[corpus]
            ]
            after_ck = after_distance[: rc.candidate_k]
            fetched = _fetch_candidates(corpus, after_ck)
            _rerank_batch(request.query, fetched, model)
            fetched.sort(key=lambda c: -c.relevance)
            after_reject = [
                c for c in fetched if c.relevance >= rc.reject_threshold
            ]
            for c in after_reject:
                _apply_decay(c, rc)
            after_reject.sort(key=lambda c: -c.final_score)

            after_ck_ids = {c["entry_id"] for c in after_ck}
            watched = {}
            for c in raw_all:
                if c["entry_id"] not in watch_set:
                    continue
                eid = c["entry_id"]
                dist_rank = next(
                    (i + 1 for i, d in enumerate(after_distance) if d["entry_id"] == eid),
                    None,
                )
                w = {
                    "chromadb_rank": raw_all.index(c) + 1,
                    "distance": c["distance"],
                    "passed_max_distance": c["distance"] <= rc.max_distance[corpus],
                    "distance_rank": dist_rank,
                    "passed_candidate_k": eid in after_ck_ids,
                }
                reranked = next((f for f in fetched if f.entry_id == eid), None)
                if reranked:
                    w["relevance"] = reranked.relevance
                    w["passed_reject"] = reranked.relevance >= rc.reject_threshold
                final = next((f for f in after_reject if f.entry_id == eid), None)
                if final:
                    w["final_score"] = final.final_score
                    w["final_rank"] = after_reject.index(final) + 1
                else:
                    w["eliminated_before"] = (
                        "max_distance" if not w["passed_max_distance"]
                        else "candidate_k" if not w["passed_candidate_k"]
                        else "reject_threshold" if "relevance" in w and not w.get("passed_reject")
                        else "sqlite_missing" if "relevance" not in w and w["passed_candidate_k"]
                        else "unknown"
                    )
                watched[str(eid)] = w

            def _brief(cands, limit=10):
                return [
                    {"entry_id": c["entry_id"], "distance": c["distance"]}
                    for c in cands[:limit]
                ]

            def _scored(cands, limit=25):
                return [
                    {
                        "entry_id": c.entry_id,
                        "title": c.title[:80] if c.title else "",
                        "distance": c.distance,
                        "relevance": c.relevance,
                        "final_score": c.final_score,
                    }
                    for c in cands[:limit]
                ]

            trace_corpora[corpus] = {
                "total_vectors": total,
                "stages": {
                    "chromadb_raw": {"count": len(raw_all), "top_10": _brief(raw_all)},
                    "after_max_distance": {
                        "count": len(after_distance),
                        "top_10": _brief(after_distance),
                    },
                    "after_candidate_k": {
                        "count": len(after_ck),
                        "entries": _brief(after_ck),
                    },
                    "after_rerank": {"count": len(fetched), "entries": _scored(fetched)},
                    "after_reject": {
                        "count": len(after_reject),
                        "entries": _scored(after_reject),
                    },
                },
                "watched": watched,
            }

        return {
            "query": request.query,
            "config": {
                "max_distance": rc.max_distance,
                "candidate_k": rc.candidate_k,
                "reject_threshold": rc.reject_threshold,
            },
            "corpora": trace_corpora,
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 1),
        }

    @v2.post("/kb/trace/dual")
    def trace_dual_v2(
        request: DualQueryRequestV2,
        client: AuthorizedClient = Depends(authorize_snapshot),
    ):
        """Dual-query retrieval: union candidates before rerank for query-rewriting eval."""
        started = time.perf_counter()
        if router_snapshot is None:
            raise HTTPException(status_code=503, detail="router_config_unavailable")
        rc = router_snapshot
        model = reranker()
        if model is None:
            raise HTTPException(status_code=503, detail="reranker_unavailable")

        orig_emb = embed(request.original_query)
        rewrite_emb = embed(request.rewritten_query)
        corpora_to_search = (
            [request.scope] if request.scope != "both" else list(CORPUS_REGISTRY)
        )

        orig_candidates: list[Candidate] = []
        union_candidates: list[Candidate] = []
        for corpus in corpora_to_search:
            orig_raw = _query_collection(corpus, orig_emb, rc)
            rewrite_raw = _query_collection(corpus, rewrite_emb, rc)
            merged = _union_candidates(orig_raw, rewrite_raw)
            orig_fetched = _fetch_candidates(corpus, orig_raw)
            union_fetched = _fetch_candidates(corpus, merged)
            orig_candidates.extend(orig_fetched)
            union_candidates.extend(union_fetched)

        _rerank_batch(request.original_query, orig_candidates, model)
        _rerank_batch(request.original_query, union_candidates, model)

        def _apply_pipeline(candidates):
            passed = [c for c in candidates if c.relevance >= rc.reject_threshold]
            for c in passed:
                _apply_decay(c, rc)
            passed.sort(key=lambda c: -c.final_score)
            return passed[:request.top_k]

        orig_final = _apply_pipeline(orig_candidates)
        union_final = _apply_pipeline(union_candidates)

        orig_ids = {c.entry_id for c in orig_final}
        union_ids = {c.entry_id for c in union_final}
        gained = union_ids - orig_ids
        lost = orig_ids - union_ids

        def _to_list(candidates):
            return [
                {
                    "corpus": c.corpus,
                    "entry_id": c.entry_id,
                    "ref": f"{c.corpus}:{c.entry_id}",
                    "title": c.title,
                    "distance": c.distance,
                    "relevance": c.relevance,
                    "final_score": c.final_score,
                    "from_primary": c.from_primary,
                }
                for c in candidates
            ]

        return {
            "original_query": request.original_query,
            "rewritten_query": request.rewritten_query,
            "original_results": _to_list(orig_final),
            "union_results": _to_list(union_final),
            "gained_entry_ids": sorted(gained),
            "lost_entry_ids": sorted(lost),
            "original_candidate_count": len(orig_candidates),
            "union_candidate_count": len(union_candidates),
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 1),
        }

    return v2
