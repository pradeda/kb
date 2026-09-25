#!/usr/bin/env python3
"""Newer-note edges: "a later note says it changes this one".

`supersede_index.py` covers the full replacement: the old entry carries
`SUPERSEDED — use X` and is demoted. A partial change has no marker — the author
writes it in the title or the opening line, e.g.

    Forrix Qwen per-model load defaults — 65536/1 persists across reboot (updates homelab:963)
    Pi-hole game blocking — removal and rollback procedures (extends #724, #725)
    AI Ingest user agents now ai-ingest/0.14 — corrects stale UA refs in homelab:882 and 529
    KB 1-hop ekspanzija ... ; content starts with "Nastavak homelab:1082 (1-hop A/B ...)"

Such a note must not be demoted — only part of the old note changed — but a reader
(usually an agent, which reads the top 5 and nothing else; homelab:1109) should be
told that a newer note speaks about it. This module finds those edges.

An edge Y -> X exists when, inside Y's title or Y's first paragraph, a change verb
stands at most CHANGE_VERB_GAP_WORDS words before a reference to X, with no
sentence boundary between them. Reference forms:

  * `homelab:N` / `ai:N` — always (target must exist);
  * `#N` — only when N exists in Y's own corpus (`#724` is a note, `#36403` is an
    upstream bug number);
  * `KB N` / `entry N` / `unos N` (this corpus names entries that way) — only when
    N is at least 100 and exists in Y's own corpus;
  * a bare number that continues a reference list (`homelab:882 and 529`), only
    when it exists in Y's own corpus.

Plain mentions are not edges: "see homelab:X", "related: ...", a bare list of
references. In the 1-hop A/B (homelab:1082) that kind of reference was 88 of the
89 notes it added as noise, so the verb is what makes a mention a relation.

Direction and filters: Y must be newer than X (created_at, then id as tie-break),
Y must not be `[SUPERSEDED]`, no self-edges, at most NEWER_NOTES_LIMIT Y per X,
newest first. Nothing is reordered, dropped or demoted by this module — it only
produces the warning line the API attaches to a result.
"""
from __future__ import annotations

import re
import sqlite3
import threading
import time
from typing import Callable, Iterable, Optional

# ── rules (frozen; see the module docstring and homelab:1109) ────────────────

# Change verbs, mapped to the canonical relation reported to callers. Serbian
# forms are the ones this corpus actually uses (ekavica).
CHANGE_VERBS: dict[str, str] = {
    "updates": "updates",
    "update": "updates",
    "updated": "updates",
    "corrects": "corrects",
    "correct": "corrects",
    "corrected": "corrects",
    "correction": "corrects",
    "extends": "extends",
    "extend": "extends",
    "extended": "extends",
    "supersedes": "supersedes",
    "supersede": "supersedes",
    "superseded": "supersedes",
    "replaces": "replaces",
    "replace": "replaces",
    "replaced": "replaces",
    "ispravlja": "corrects",
    "ispravka": "corrects",
    "ispravke": "corrects",
    "dopunjuje": "extends",
    "dopuna": "extends",
    "menja": "updates",
    "izmena": "updates",
    "zamenjuje": "replaces",
    "nastavak": "continues",
    "nastavlja": "continues",
}

# A negation directly before the verb means the note does NOT change the target
# ("this no longer updates homelab:X").
NEGATIONS = frozenset({"not", "never", "no", "without", "ne", "nije", "bez", "nema"})

REF_RE = re.compile(r"\b(homelab|ai):([0-9]+)\b")
HASH_REF_RE = re.compile(r"#([0-9]+)\b")
# "KB 413", "entry 780", "iz unosa 550": the ways this corpus names an entry by id
# without a corpus prefix (homelab:1082 froze the >=100 rule for the same reason).
LABEL_REF_RE = re.compile(
    r"\b(?:KB|kb|entry|entries|unos|unosa|unosu|unosom)\s*#?\s*([0-9]+)\b"
)
LABEL_REF_MIN_ID = 100
TOKEN_RE = re.compile(r"\S+")
# Separators inside a reference list: "homelab:882 and 529", "#724, #725".
LIST_SEPARATORS = frozenset({",", "and", "or", "i", "&", "/", "+"})
# Sentence ends (and line ends) that a verb may not reach across.
BREAK_RE = re.compile(r"[.!?](?=\s|$)|\n")

CHANGE_VERB_GAP_WORDS = 6
# The marker lives in the title or in the opening of the note; scanning further
# would collect ordinary mentions from the body.
CONTENT_SCAN_LINES = 6
CONTENT_SCAN_CHARS = 700
NEWER_NOTES_LIMIT = 3
NEWER_NOTES_RECHECK_SECONDS = 30.0


# ── detection ────────────────────────────────────────────────────────────────

def _bare(token: str) -> str:
    """Token stripped of surrounding punctuation/quotes, lowercased."""
    return token.strip("\"'`()[]{}<>:;,*—–-…!?.").lower()


def _tokens(segment: str) -> list[tuple[int, int, str]]:
    return [(m.start(), m.end(), m.group(0)) for m in TOKEN_RE.finditer(segment)]


def _token_index(tokens: list[tuple[int, int, str]], position: int) -> int:
    for index, (start, end, _) in enumerate(tokens):
        if start <= position < end:
            return index
    return -1


def _verb_before(tokens, index, breaks, position) -> Optional[str]:
    """Canonical relation for a change verb at most CHANGE_VERB_GAP_WORDS words
    before token `index`, or None. Never crosses a sentence/line break, never a
    negated verb."""
    gap = 0
    cursor = index - 1
    while cursor >= 0 and gap <= CHANGE_VERB_GAP_WORDS:
        start, end, raw = tokens[cursor]
        if any(start <= mark <= end for mark in breaks):
            return None
        word = _bare(raw)
        if word in CHANGE_VERBS:
            previous = _bare(tokens[cursor - 1][2]) if cursor else ""
            if previous in NEGATIONS:
                return None
            return CHANGE_VERBS[word]
        gap += 1
        cursor -= 1
    return None


def _references(
    segment: str, corpus: str, known_ids: set
) -> list[tuple[str, int, str, Optional[str]]]:
    """(target_corpus, target_id, form, relation) for every reference in a segment.

    relation is the change verb bound to that reference, or None when the
    reference is a plain mention — only bound references become edges.
    """
    tokens = _tokens(segment)
    breaks = {m.end() for m in BREAK_RE.finditer(segment)}
    found: list[tuple[int, str, int, str]] = []  # (position, corpus, id, form)

    for match in REF_RE.finditer(segment):
        found.append((match.start(), match.group(1), int(match.group(2)), "ref"))
    for match in HASH_REF_RE.finditer(segment):
        candidate = int(match.group(1))
        if (corpus, candidate) in known_ids:
            found.append((match.start(), corpus, candidate, "hash"))
    for match in LABEL_REF_RE.finditer(segment):
        candidate = int(match.group(1))
        # Labelled ids below 100 are list numbering, not references (homelab:1082).
        if candidate >= LABEL_REF_MIN_ID and (corpus, candidate) in known_ids:
            found.append((match.start(), corpus, candidate, "label"))
    found.sort()

    # A bare number that continues a resolved reference list ("homelab:882 and 529").
    # Deliberately NOT a bare number after a change verb: in this corpus that caught
    # "section 1", "revision 12" and "16 GB" as references.
    accepted = {position for position, *_ in found}
    for index, (start, _end, raw) in enumerate(tokens):
        if start in accepted:
            continue
        word = _bare(raw)
        if not word.isdigit():
            continue
        number = int(word)
        if number == 0 or (corpus, number) not in known_ids:
            continue
        previous = _bare(tokens[index - 1][2]) if index else ""
        before_previous = tokens[index - 2] if index >= 2 else None
        if (
            previous in LIST_SEPARATORS
            and before_previous is not None
            and before_previous[0] in accepted
        ):
            accepted.add(start)
            found.append((start, corpus, number, "bare"))
    found.sort()

    references: list[tuple[str, int, str, Optional[str]]] = []
    for position, target_corpus, target_id, form in found:
        index = _token_index(tokens, position)
        relation = (
            _verb_before(tokens, index, breaks, position) if index >= 0 else None
        )
        references.append((target_corpus, target_id, form, relation))
    return references


def _content_head(content: Optional[str]) -> str:
    """The first paragraph of a note (up to a blank line), capped."""
    if not content:
        return ""
    lines: list[str] = []
    for line in content.split("\n")[:CONTENT_SCAN_LINES]:
        if not line.strip():
            break
        lines.append(line)
    return "\n".join(lines)[:CONTENT_SCAN_CHARS]


def detect_relations(
    corpus: str,
    entry_id: int,
    title: Optional[str],
    content: Optional[str],
    known_ids: set,
) -> list[tuple[tuple[str, int], str, str, str]]:
    """[((target_corpus, target_id), canonical relation, where, form)] for one note.

    `where` is "title" or "content" (the first paragraph); `form` is how the
    reference was written ("ref", "hash", "bare") — kept for the edge report.
    """
    relations: list[tuple[tuple[str, int], str, str, str]] = []
    seen: set[tuple[str, int]] = set()
    for where, segment in (("title", title or ""), ("content", _content_head(content))):
        if not segment:
            continue
        for target_corpus, target_id, form, relation in _references(segment, corpus, known_ids):
            target = (target_corpus, target_id)
            if relation is None or target == (corpus, entry_id) or target in seen:
                continue
            seen.add(target)
            relations.append((target, relation, where, form))
    return relations


# ── index (in memory, refreshed like the FTS5 index) ─────────────────────────

def build_edges(entries: Iterable[dict]) -> dict[tuple[str, int], list[dict]]:
    """{(target_corpus, target_id): [{ref, title, relation, created_at, where}]}

    Only edges from a newer, not-[SUPERSEDED] note survive; at most
    NEWER_NOTES_LIMIT per target, newest first.
    """
    entries = list(entries)
    known_ids = {(item["corpus"], item["id"]) for item in entries}
    meta = {(item["corpus"], item["id"]): item for item in entries}
    edges: dict[tuple[str, int], list[dict]] = {}
    for item in entries:
        title = item.get("title") or ""
        if title.startswith("[SUPERSEDED]"):
            continue
        for target, relation, where, form in detect_relations(
            item["corpus"], item["id"], title, item.get("content"), known_ids
        ):
            other = meta.get(target)
            if other is None:
                continue
            if not _is_newer(item, other):
                continue
            edges.setdefault(target, []).append(
                {
                    "ref": f"{item['corpus']}:{item['id']}",
                    "title": title,
                    "relation": relation,
                    "created_at": item.get("created_at") or "",
                    "id": item["id"],
                    "where": where,
                    "form": form,
                }
            )
    for target, items in edges.items():
        items.sort(key=lambda item: (item["created_at"], item["id"]), reverse=True)
        del items[NEWER_NOTES_LIMIT:]
    return edges


def _is_newer(item: dict, other: dict) -> bool:
    """Y is newer than X by created_at, then id (per corpus only as a tie-break)."""
    created, other_created = item.get("created_at") or "", other.get("created_at") or ""
    if created != other_created:
        return created > other_created
    return item["id"] > other["id"]


def load_entries(db_path: str, corpus: str) -> list[dict]:
    """Read only what detection needs: id, title, the note's opening, created_at."""
    connection = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=5)
    try:
        rows = connection.execute(
            "SELECT id, title, substr(content, 1, ?), created_at FROM entries",
            (CONTENT_SCAN_CHARS,),
        ).fetchall()
    finally:
        connection.close()
    return [
        {"corpus": corpus, "id": row[0], "title": row[1], "content": row[2], "created_at": row[3]}
        for row in rows
    ]


class NewerNotesIndex:
    """Cross-corpus "newer note" edges, held in memory.

    Built from the corpus databases at app creation and refreshed with the same
    mechanism the FTS5 index uses (homelab:1078): a cheap source signature checked
    at most once per NEWER_NOTES_RECHECK_SECONDS, a non-blocking rebuild lock, and a
    failed check that keeps the current edges in place while recording the reason.
    The refresh is never per-query work: it is one signature read per window.
    """

    def __init__(self, sources: dict[str, str], signature_fn: Callable[[str], tuple],
                 throttle_seconds: Optional[float] = None) -> None:
        self.sources = dict(sources)
        self.signature_fn = signature_fn
        # Resolved at construction, so the module constant stays the single knob
        # (and a test can patch it).
        self.throttle_seconds = (
            NEWER_NOTES_RECHECK_SECONDS if throttle_seconds is None else throttle_seconds
        )
        self.edges: dict[tuple[str, int], list[dict]] = {}
        self.signatures: dict[str, tuple] = {}
        self.lock = threading.Lock()
        self.checked_at = 0.0

    @classmethod
    def build(cls, sources: dict[str, str], signature_fn: Callable[[str], tuple],
              throttle_seconds: Optional[float] = None) -> "NewerNotesIndex":
        index = cls(sources, signature_fn, throttle_seconds)
        index.rebuild()
        return index

    def rebuild(self) -> None:
        entries: list[dict] = []
        signatures: dict[str, tuple] = {}
        for corpus, path in sorted(self.sources.items()):
            entries.extend(load_entries(path, corpus))
            signatures[corpus] = self.signature_fn(path)
        self.edges = build_edges(entries)
        self.signatures = signatures

    def refresh(self, status: Optional[dict[str, str]] = None) -> None:
        now = time.monotonic()
        if now - self.checked_at < self.throttle_seconds:
            return
        if not self.lock.acquire(blocking=False):
            return
        try:
            self.checked_at = now
            try:
                signatures = {
                    corpus: self.signature_fn(path) for corpus, path in self.sources.items()
                }
            except (sqlite3.Error, OSError) as exc:
                self._record(
                    f"source fingerprint failed ({type(exc).__name__}: {exc})", status
                )
                return
            if signatures == self.signatures:
                return
            try:
                self.rebuild()
            except Exception as exc:  # keep the previous edges, say why
                self._record(f"edge rebuild failed ({type(exc).__name__}: {exc})", status)
        finally:
            self.lock.release()

    def _record(self, reason: str, status: Optional[dict[str, str]]) -> None:
        if status is not None:
            status["newer_notes_degraded"] = reason
        if _warned.get("reason") != reason:
            _warned["reason"] = reason
            print(
                f"[newer-notes] WARNING: {reason} — results carry no newer-note warnings",
                flush=True,
            )

    def for_ref(self, corpus: str, entry_id: int) -> list[dict]:
        """[{ref, title, relation}] for one result, or [] (never None)."""
        return [
            {"ref": item["ref"], "title": item["title"], "relation": item["relation"]}
            for item in self.edges.get((corpus, entry_id), ())
        ]


_warned: dict[str, str] = {}
