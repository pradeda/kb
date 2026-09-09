#!/usr/bin/env python3
# /opt/kb/compile.py

import argparse
import sqlite3, os, re, json, tempfile
from datetime import datetime, timezone
from pathlib import Path

# --- configuration ---
CORPUS_PROFILES = {
    "homelab": {
        "root": "/opt/kb",
        "db": "/opt/kb/kb.db",
        "raw": "/opt/kb/raw",
        "env": "/opt/kb/.env",
        "collection": "kb_collection",
        "wiki_index": "/opt/kb/wiki/index.md",
        "secret_patterns": "/opt/kb/secret_patterns.json",
        "quarantine_dir": "/opt/kb/quarantine",
        "quarantine_log": "/opt/kb/quarantine.log",
        "watcher_lock": "/tmp/kb-watcher.lock",
        "watcher_state": "/tmp/kb-watcher-last",
    },
    "ai": {
        "root": "/opt/ai-kb",
        "db": "/opt/ai-kb/ai-kb.db",
        "raw": "/opt/ai-kb/raw",
        "env": "/opt/ai-kb/.env",
        "collection": "ai_kb_collection",
        "wiki_index": "",
        "secret_patterns": "/opt/ai-kb/secret_patterns.json",
        "quarantine_dir": "/opt/ai-kb/quarantine",
        "quarantine_log": "/opt/ai-kb/quarantine.log",
        "watcher_lock": "/tmp/ai-kb-watcher.lock",
        "watcher_state": "/tmp/ai-kb-watcher-last",
    },
}

KB = DB = WIKI = RAW = ENV_FILE = None
SECRET_PATTERNS_FILE = QUARANTINE_DIR = QUARANTINE_LOG = None
CHROMA_COLLECTION = None
ACTIVE_CORPUS = None

EMBED_MODEL       = "nomic-ai/nomic-embed-text-v1.5"
CHROMA_HOST       = "localhost"
CHROMA_PORT       = 8000
EMBEDDING_DIMENSION = 768
COLLECTION_SCHEMA_VERSION = 1

def configure_corpus(name):
    global ACTIVE_CORPUS, KB, DB, WIKI, RAW, ENV_FILE
    global CHROMA_COLLECTION, SECRET_PATTERNS_FILE, QUARANTINE_DIR, QUARANTINE_LOG
    global _secret_rules
    try:
        profile = CORPUS_PROFILES[name]
    except KeyError as exc:
        raise ValueError(f"unknown corpus {name!r}") from exc
    ACTIVE_CORPUS = name
    KB = Path(profile["root"])
    DB = Path(profile["db"])
    RAW = Path(profile["raw"])
    ENV_FILE = Path(profile["env"])
    CHROMA_COLLECTION = profile["collection"]
    WIKI = Path(profile["wiki_index"]).parent if profile["wiki_index"] else None
    SECRET_PATTERNS_FILE = Path(profile["secret_patterns"])
    QUARANTINE_DIR = Path(profile["quarantine_dir"])
    QUARANTINE_LOG = Path(profile["quarantine_log"])
    _secret_rules = None
    return profile

def load_profile_env():
    if not ENV_FILE.exists():
        return
    for line in ENV_FILE.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Compile one allowlisted KB corpus")
    parser.add_argument("--corpus", choices=tuple(CORPUS_PROFILES), default="homelab")
    recovery = parser.add_mutually_exclusive_group()
    recovery.add_argument("--recover-db", action="store_true")
    recovery.add_argument("--recover-raw", action="store_true")
    recovery.add_argument("--health", action="store_true", help="check SQLite and Chroma invariants")
    recovery.add_argument("--retire", type=int, metavar="ID",
                          help="retire one entry from SQLite, FTS5, Chroma and raw storage")
    recovery.add_argument("--supersede", type=int, metavar="ID",
                          help="mark an existing entry as superseded and queue re-embedding")
    parser.add_argument("--replacement", metavar="REFS",
                        help="replacement KB references required by --supersede")
    args = parser.parse_args(argv)
    if (args.supersede is None) != (args.replacement is None):
        parser.error("--supersede and --replacement must be used together")
    return args

configure_corpus("homelab")

def get_db():
    return sqlite3.connect(str(DB))

def retire_entry(entry_id, collection=None, db_path=None):
    """Povuci jedan unos iz aktivnog corpusa kroz sva četiri sloja.

    Redosled je namerno takav da svaki prekid ostavi stanje koje --health
    prijavljuje i koje ponovni poziv može da dovrši:
      1. obriši Chroma vektor i potvrdi da ga stvarno nema
      2. obriši SQLite red (trigger čisti FTS5)
      3. obriši raw fajl

    Vektor ide prvi jer orphan vektor bez SQLite reda obara health, dok
    unos bez vektora samo ispada iz semantičke pretrage do ponovnog embed-a.

    Ako SQLite reda nema a vektor postoji, unos je već delimično obrisan mimo
    ovog puta (ručni DELETE) — tada se čisti samo vektor, jer je to jedini
    sloj koji je ostao i jedino stanje koje --health prijavljuje kao orphan.
    LookupError ostaje za id koji ne postoji ni u jednom sloju, da pogrešno
    otkucan broj i dalje pukne umesto da tiho „uspe".
    """
    entry_id = int(entry_id)
    target_db = Path(db_path) if db_path else DB
    connection = sqlite3.connect(str(target_db))
    try:
        row = connection.execute(
            "SELECT id, title, raw_path FROM entries WHERE id=?", (entry_id,)
        ).fetchone()

        if collection is None:
            collection = get_chroma_collection()

        if row is None:
            present = collection.get(ids=[str(entry_id)], include=[]).get("ids") or []
            if not present:
                raise LookupError(f"entry {entry_id} not found in {target_db}")
            collection.delete(ids=[str(entry_id)])
            survivors = collection.get(ids=[str(entry_id)], include=[]).get("ids") or []
            if survivors:
                raise RuntimeError(
                    f"Chroma still holds orphan vector for entry {entry_id}"
                )
            return {
                "entry_id": entry_id,
                "corpus": ACTIVE_CORPUS,
                "title": None,
                "raw_path": None,
                "raw_removed": False,
                "orphan_vector_only": True,
            }

        _, title, raw_path = row
        collection.delete(ids=[str(entry_id)])
        survivors = collection.get(ids=[str(entry_id)], include=[]).get("ids") or []
        if survivors:
            raise RuntimeError(
                f"Chroma still holds vector for entry {entry_id}; SQLite left untouched"
            )

        connection.execute("DELETE FROM entries WHERE id=?", (entry_id,))
        connection.commit()

        removed_raw = False
        if raw_path:
            try:
                Path(raw_path).unlink()
                removed_raw = True
            except FileNotFoundError:
                removed_raw = False
        return {
            "entry_id": entry_id,
            "corpus": ACTIVE_CORPUS,
            "title": title,
            "raw_path": raw_path,
            "raw_removed": removed_raw,
            "orphan_vector_only": False,
        }
    finally:
        connection.close()


def encode_frontmatter_value(value):
    """Return a JSON string literal, which is also a valid YAML scalar."""
    return json.dumps("" if value is None else str(value), ensure_ascii=False)

def decode_frontmatter_value(value):
    """Decode values written by encode_frontmatter_value; accept legacy scalars."""
    value = value.strip()
    if value.startswith('"'):
        decoded = json.loads(value)
        if not isinstance(decoded, str):
            raise ValueError(f"Expected string frontmatter value, got {type(decoded)}")
        return decoded
    return value

def supersede_entry(entry_id, replacement, db_path=None):
    """Mark one historical record obsolete while preserving its ID and history.

    SQLite/FTS and the raw source are changed as one recoverable operation. The
    row is left pending (`embedded_at=NULL`) so the normal compile path upserts
    its Chroma document before this command returns.
    """
    entry_id = int(entry_id)
    replacement = str(replacement).strip()
    if not replacement:
        raise ValueError("replacement references must not be empty")
    target_db = Path(db_path) if db_path else DB
    db = sqlite3.connect(str(target_db))
    temp_path = None
    raw_path = None
    original_raw = None
    try:
        row = db.execute(
            "SELECT type,content,title,tags,raw_path,created_at FROM entries WHERE id=?",
            (entry_id,),
        ).fetchone()
        if row is None:
            raise LookupError(f"entry {entry_id} not found in {target_db}")
        etype, content, title, tags, raw_value, created_at = row
        marker = f"SUPERSEDED — use {replacement}"
        if content.startswith(marker):
            return {"entry_id": entry_id, "replacement": replacement, "already_superseded": True}
        if not raw_value:
            raise RuntimeError(f"entry {entry_id} has no raw_path; refusing partial update")
        raw_path = Path(raw_value)
        original_raw = raw_path.read_bytes()
        new_title = title if title.startswith("[SUPERSEDED]") else f"[SUPERSEDED] {title}"
        new_content = (
            f"{marker}\n\n"
            "The operational guidance below is historical and must not be followed. "
            "Use the replacement records above for the current interpreter mapping.\n\n"
            "--- Historical incident content ---\n\n"
            f"{content}"
        )
        raw_text = (
            f"---\ntype: {etype}\n"
            f"title: {encode_frontmatter_value(new_title)}\n"
            f"tags: {encode_frontmatter_value(tags)}\n"
            f"saved: {created_at}\n---\n\n{new_content}"
        )
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=raw_path.parent, delete=False
        ) as handle:
            handle.write(raw_text)
            temp_path = Path(handle.name)
        temp_path.chmod(0o600)

        db.execute("BEGIN IMMEDIATE")
        db.execute(
            "UPDATE entries SET content=?,title=?,compiled_at=NULL,embedded_at=NULL WHERE id=?",
            (new_content, new_title, entry_id),
        )
        os.replace(temp_path, raw_path)
        temp_path = None
        try:
            db.commit()
        except Exception:
            raw_path.write_bytes(original_raw)
            raw_path.chmod(0o600)
            raise
        return {
            "entry_id": entry_id,
            "replacement": replacement,
            "already_superseded": False,
            "raw_path": str(raw_path),
        }
    except Exception:
        db.rollback()
        raise
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
        db.close()


def get_unembedded():
    db = get_db()
    rows = db.execute("""
        SELECT id, type, content, title, tags, summary, raw_path
        FROM entries WHERE embedded_at IS NULL ORDER BY created_at
    """).fetchall()
    db.close()
    return rows


def mark_embedded(ids):
    db = get_db()
    ph = ",".join("?" * len(ids))
    db.execute(f"UPDATE entries SET embedded_at=? WHERE id IN ({ph})",
               [datetime.now().isoformat()] + list(ids))
    db.commit()
    db.close()

def record_compile_run():
    """Stamp the end of a compile pass, including a pass that had nothing to do.

    max(compiled_at) over entries cannot answer "is the compiler still running":
    on an idle corpus it keeps pointing at the last real write, which looks
    healthy while the watcher may have been dead for days. This row moves on
    every pass, so a stale value means the pass itself stopped happening.

    UTC with an explicit offset, unlike the naive local timestamps elsewhere in
    this file: this one is read back by /v2/health and compared against now.
    """
    stamp = datetime.now(timezone.utc).isoformat()
    try:
        db = get_db()
        try:
            db.execute(
                "CREATE TABLE IF NOT EXISTS kb_meta("
                "key TEXT PRIMARY KEY, value TEXT NOT NULL)"
            )
            db.execute(
                "INSERT INTO kb_meta(key, value) VALUES('last_compile', ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (stamp,),
            )
            db.commit()
        finally:
            db.close()
    except sqlite3.Error as exc:
        # Bookkeeping must not fail a run that did real work.
        print(f"  [WARN] last_compile not recorded: {exc}")
        return None
    return stamp


def get_chroma_collection():
    import chromadb
    client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
    collection = client.get_collection(name=CHROMA_COLLECTION)
    validate_collection_metadata(collection.metadata)
    validate_collection_configuration(collection.configuration_json)
    return collection

def validate_collection_metadata(metadata):
    expected = {
        "hnsw:space": "cosine",
        "corpus": ACTIVE_CORPUS,
        "embedding_model": EMBED_MODEL,
        "embedding_dimension": EMBEDDING_DIMENSION,
        "schema_version": COLLECTION_SCHEMA_VERSION,
    }
    mismatches = {
        key: {"expected": value, "actual": (metadata or {}).get(key)}
        for key, value in expected.items()
        if (metadata or {}).get(key) != value
    }
    if not (metadata or {}).get("created_at"):
        mismatches["created_at"] = {"expected": "non-empty", "actual": (metadata or {}).get("created_at")}
    if mismatches:
        raise RuntimeError(
            f"{CHROMA_COLLECTION} metadata mismatch for {ACTIVE_CORPUS}: "
            f"{json.dumps(mismatches, sort_keys=True)}"
        )

def validate_collection_configuration(configuration):
    actual_space = ((configuration or {}).get("hnsw") or {}).get("space")
    if actual_space != "cosine":
        raise RuntimeError(
            f"{CHROMA_COLLECTION} configuration mismatch for {ACTIVE_CORPUS}: "
            f"expected cosine, got {actual_space!r}"
        )

_embed_model = None

def get_embed_model():
    global _embed_model
    if _embed_model is None:
        from fastembed import TextEmbedding
        _embed_model = TextEmbedding(EMBED_MODEL)
    return _embed_model

EMBED_BATCH_SIZE = 50

def embed_entries(entries):
    total = len(entries)
    print(f"\nEmbedding {total} entries to ChromaDB (FastEmbed batch)...")
    collection = get_chroma_collection()
    model = get_embed_model()
    failed = 0
    successful_ids = []

    for chunk_start in range(0, total, EMBED_BATCH_SIZE):
        chunk = entries[chunk_start:chunk_start + EMBED_BATCH_SIZE]
        ids, texts, metadatas, filtered_rows = [], [], [], []

        for row in chunk:
            entry_id, etype, content, title, tags, summary, raw_path = row
            text_parts = []
            if title:   text_parts.append(f"Title: {title}")
            if summary: text_parts.append(f"Summary: {summary}")
            if content: text_parts.append(content[:8000])
            if tags:    text_parts.append(f"Tags: {tags}")
            if not text_parts:
                print(f"  [SKIP] #{entry_id}: empty content")
                continue
            ids.append(str(entry_id))
            texts.append("\n".join(text_parts))
            filtered_rows.append(row)
            metadatas.append({
                "type": etype,
                "title": title or "",
                "tags": tags or "",
                "raw_path": raw_path or "",
            })

        if not ids:
            continue

        try:
            embeddings = [e.tolist() for e in model.passage_embed(texts)]
            collection.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas,
            )
            for entry_id, row in zip(ids, filtered_rows):
                label = row[3] or row[2][:40]
                print(f"  [OK] embed #{entry_id}: {label}")
            successful_ids.extend(ids)
        except Exception as e:
            print(f"  [ERROR] batch {chunk_start}-{chunk_start + len(chunk)}: {e}")
            failed += len(ids)

    print(f"\nEmbed summary: {len(successful_ids)} succeeded, {failed} failed out of {total}")
    if failed:
        import sys
        print("ERROR: some entries failed to embed", file=sys.stderr)
    return successful_ids, failed


def recover_db_from_raw():
    """Parse all raw .md files and rebuild SQLite database."""
    print("Recover DB: parsing raw .md files...")
    recovered = 0
    skipped = 0

    db = get_db()
    db.execute("BEGIN TRANSACTION")
    try:
        for subdir in ["notes", "urls"]:
            raw_sub = RAW / subdir
            if not raw_sub.exists():
                continue
            for md_file in sorted(raw_sub.glob("*.md")):
                try:
                    content = md_file.read_text(encoding="utf-8")
                except UnicodeDecodeError:
                    content = md_file.read_text(encoding="latin-1")
                # Parse frontmatter
                fm = {}
                if content.startswith("---"):
                    parts = content.split("---", 2)
                    if len(parts) >= 3:
                        for line in parts[1].strip().split("\n"):
                            if ":" in line:
                                k, v = line.split(":", 1)
                                fm[k.strip()] = decode_frontmatter_value(v)
                        body = parts[2].strip()
                    else:
                        body = content
                else:
                    body = content

                etype = fm.get("type", subdir.rstrip("s"))  # "notes"→"note", "urls"→"url"
                if etype not in ("note", "url"):
                    etype = "note"  # default
                title = fm.get("title", md_file.stem)
                tags = fm.get("tags", "")
                saved = fm.get("saved", datetime.fromtimestamp(md_file.stat().st_mtime).strftime("%Y-%m-%dT%H:%M:%S"))

                row = db.execute("SELECT id FROM entries WHERE raw_path = ?", (str(md_file),)).fetchone()
                if row:
                    skipped += 1
                    continue

                db.execute(
                    "INSERT INTO entries (type, content, title, tags, raw_path, source, created_at) "
                    "VALUES (?, ?, ?, ?, ?, 'recovery', ?)",
                    (etype, body, title, tags, str(md_file), saved),
                )
                recovered += 1
                print(f"  [OK] {md_file.name}")

        db.commit()
    except Exception as e:
        db.rollback()
        print(f"Error during DB recovery: {e}")
        raise
    finally:
        db.close()

    print(f"\nRecover DB done: {recovered} recovered, {skipped} skipped (already in DB)")

def recover_raw_from_db():
    """Rebuild raw .md files from SQLite database."""
    print("Recover RAW: generating .md files from SQLite...")
    recovered = 0
    skipped = 0

    db = get_db()
    rows = db.execute(
        "SELECT id, type, content, title, tags, created_at, raw_path FROM entries ORDER BY created_at"
    ).fetchall()
    db.close()

    for row in rows:
        eid, etype, content, title, tags, created, raw_path = row
        raw_file = Path(raw_path) if raw_path else None

        # Determine path
        subdir = "notes" if etype == "note" else "urls"
        date_str = created[:10] if created else "unknown"
        slug = re.sub(r"[^a-z0-9-]", "", title.lower().replace(" ", "-"))[:60] if title else f"entry-{eid}"

        use_existing = False
        if raw_file:
            try:
                use_existing = raw_file.exists()
            except OSError:
                pass  # path too long, regenerate

        if use_existing:
            skipped += 1
            continue

        if not raw_file or not use_existing:
            raw_file = RAW / subdir / f"{date_str}-{slug}.md"
        raw_file.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        raw_file.parent.chmod(0o700)

        fm = (
            f"---\ntype: {etype}\n"
            f"title: {encode_frontmatter_value(title)}\n"
            f"tags: {encode_frontmatter_value(tags)}\n"
            f"saved: {created}\n---\n\n{content}"
        )
        try:
            with raw_file.open("x", encoding="utf-8") as handle:
                handle.write(fm)
            raw_file.chmod(0o600)
        except FileExistsError:
            skipped += 1
            continue
        recovered += 1
        print(f"  [OK] {raw_file.name}")

    print(f"\nRecover RAW done: {recovered} recovered, {skipped} skipped (already exists)")

# ─── Gate 2: secret scanner (embed-time safety-net) ─────────────────────────
# Mirrors kb-go/secretscan.go. RE2-only patterns match identically in both.

_secret_rules = None

def load_secret_rules():
    global _secret_rules
    if _secret_rules is not None:
        return _secret_rules
    try:
        data = json.loads(SECRET_PATTERNS_FILE.read_text(encoding="utf-8"))
    except Exception as e:
        raise RuntimeError(
            f"secret scanner unavailable for {ACTIVE_CORPUS}: {SECRET_PATTERNS_FILE}: {e}"
        ) from e
    if data.get("version") != 1:
        raise RuntimeError(
            f"unsupported secret rules version for {ACTIVE_CORPUS}: {data.get('version')!r}"
        )
    raw_patterns = data.get("patterns")
    if not isinstance(raw_patterns, list) or not raw_patterns:
        raise RuntimeError(f"secret rules for {ACTIVE_CORPUS} must contain at least one pattern")
    pats = []
    for index, p in enumerate(raw_patterns):
        if not isinstance(p, dict) or not p.get("name") or not p.get("regex"):
            raise RuntimeError(f"secret pattern {index} for {ACTIVE_CORPUS} requires name and regex")
        if p.get("action") not in {"redact", "log"}:
            raise RuntimeError(
                f"secret pattern {p['name']!r} for {ACTIVE_CORPUS} has invalid action {p.get('action')!r}"
            )
        if p["action"] == "redact" and not p.get("placeholder"):
            raise RuntimeError(
                f"redact pattern {p['name']!r} for {ACTIVE_CORPUS} requires placeholder"
            )
        capture_group = p.get("capture_group")
        if not isinstance(capture_group, int) or capture_group < 0:
            raise RuntimeError(
                f"secret pattern {p['name']!r} for {ACTIVE_CORPUS} has invalid capture_group"
            )
        try:
            p["_re"] = re.compile(p["regex"])
        except re.error as e:
            raise RuntimeError(
                f"invalid secret pattern {p.get('name')!r} for {ACTIVE_CORPUS}: {e}"
            ) from e
        if capture_group > p["_re"].groups:
            raise RuntimeError(
                f"secret pattern {p['name']!r} capture_group {capture_group} exceeds {p['_re'].groups} groups"
            )
        pats.append(p)
    _secret_rules = {
        "patterns": pats,
        "allow": [a.lower() for a in data.get("allowlist", [])],
        "allow_sub": [a.lower() for a in data.get("allow_contains", [])],
    }
    return _secret_rules

def _shannon(s):
    if not s:
        return 0.0
    from math import log2
    data = s.encode("utf-8", "replace")
    freq = {}
    for b in data:
        freq[b] = freq.get(b, 0) + 1
    n = len(data)
    return -sum((c / n) * log2(c / n) for c in freq.values())

def _allowed(rules, value):
    v = value.strip().lower()
    if not v:
        return True
    if v in rules["allow"]:
        return True
    return any(a in v for a in rules["allow_sub"])

def sanitize_secrets(content):
    rules = load_secret_rules()
    hits = []
    for p in rules["patterns"]:
        for m in reversed(list(p["_re"].finditer(content))):
            try:
                start, end = m.span(p["capture_group"])
            except (re.error, IndexError):
                continue
            if start < 0:
                continue
            value = content[start:end]
            if _allowed(rules, value):
                continue
            if len(value) < p.get("min_len", 0):
                continue
            me = p.get("min_entropy", 0)
            if me > 0 and _shannon(value) < me:
                continue
            hits.append((p["name"], p["action"], value))
            if p["action"] == "redact":
                content = content[:start] + p["placeholder"] + content[end:]
    return content, hits

def _record_quarantine(entry_id, orig_content, raw_path, hits):
    ts = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    bak = ""
    try:
        QUARANTINE_DIR.mkdir(mode=0o700, exist_ok=True)
        bak = QUARANTINE_DIR / f"{entry_id}-{datetime.now().strftime('%Y%m%d-%H%M%S')}.orig"
        bak.write_text(orig_content or "", encoding="utf-8")
        os.chmod(bak, 0o600)
    except Exception as e:
        print(f"  [WARN] quarantine backup failed: {e}")
    # Redact the on-disk raw archive too, so rsync/NAS copies stay clean.
    if raw_path:
        try:
            rp = Path(raw_path)
            if rp.exists():
                raw = rp.read_text(encoding="utf-8")
                clean_raw, _ = sanitize_secrets(raw)
                if clean_raw != raw:
                    rp.write_text(clean_raw, encoding="utf-8")
        except Exception as e:
            print(f"  [WARN] raw redact failed for {raw_path}: {e}")
    names = ",".join(f"{h[0]}:{h[1]}" for h in hits)
    try:
        with open(QUARANTINE_LOG, "a") as f:
            f.write(f"{ts} id={entry_id} source=compile hits={names} backup={bak}\n")
        os.chmod(QUARANTINE_LOG, 0o600)
    except Exception as e:
        print(f"  [WARN] quarantine log failed: {e}")

def sanitize_unembedded(rows):
    cleaned = []
    db = get_db()
    for row in rows:
        entry_id, etype, content, title, tags, summary, raw_path = row
        new_content, ch = sanitize_secrets(content or "")
        new_title,   th = sanitize_secrets(title or "")
        new_summary, sh = sanitize_secrets(summary or "")
        redact_hits = [h for h in (ch + th + sh) if h[1] == "redact"]
        if redact_hits and (new_content != (content or "") or new_title != (title or "") or new_summary != (summary or "")):
            db.execute("UPDATE entries SET content=?, title=?, summary=? WHERE id=?",
                       (new_content, new_title, new_summary, entry_id))
            db.commit()
            _record_quarantine(entry_id, content, raw_path, redact_hits)
            print(f"  [REDACT] #{entry_id}: {', '.join(h[0] for h in redact_hits)}")
        cleaned.append((entry_id, etype, new_content, new_title, tags, new_summary, raw_path))
    db.close()
    return cleaned

def check_health():
    report = {"corpus": ACTIVE_CORPUS, "sqlite": {}, "chroma": {}}
    db = get_db()
    try:
        integrity = db.execute("PRAGMA integrity_check").fetchone()[0]
        tables = {row[0] for row in db.execute(
            "SELECT name FROM sqlite_master WHERE type IN ('table', 'trigger')"
        )}
        required = {
            "entries", "entries_fts", "entries_fts_insert",
            "entries_fts_update", "entries_fts_delete",
        }
        missing = sorted(required - tables)
        all_ids = {str(row[0]) for row in db.execute("SELECT id FROM entries")}
        embedded_ids = {
            str(row[0]) for row in db.execute(
                "SELECT id FROM entries WHERE embedded_at IS NOT NULL"
            )
        }
        report["sqlite"] = {
            "integrity": integrity,
            "missing_objects": missing,
            "entries": len(all_ids),
            "embedded": len(embedded_ids),
        }
        if integrity != "ok" or missing:
            raise RuntimeError(f"SQLite health failed: {report['sqlite']}")
    finally:
        db.close()

    collection = get_chroma_collection()
    chroma_ids = set(collection.get(include=[])["ids"])
    missing_vectors = sorted(embedded_ids - chroma_ids)
    orphan_vectors = sorted(chroma_ids - all_ids)
    if missing_vectors or orphan_vectors:
        raise RuntimeError(
            f"{ACTIVE_CORPUS} SQLite/Chroma ID mismatch: "
            f"missing={missing_vectors[:20]} orphans={orphan_vectors[:20]}"
        )
    report["chroma"] = {
        "collection": CHROMA_COLLECTION,
        "count": collection.count(),
        "metadata": collection.metadata,
        "missing_vectors": missing_vectors,
        "orphan_vectors": orphan_vectors,
    }
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))

def main(argv=None):
    args = parse_args(argv)
    configure_corpus(args.corpus)
    load_profile_env()
    supersede_report = None

    if args.health:
        check_health()
        return

    if args.retire is not None:
        report = retire_entry(args.retire)
        print(json.dumps(report, ensure_ascii=False, sort_keys=True))
        return

    if args.supersede is not None:
        supersede_report = supersede_entry(args.supersede, args.replacement)

    if args.recover_db:
        recover_db_from_raw()
        return
    if args.recover_raw:
        recover_raw_from_db()
        return

    embed_failed = 0
    unembedded = get_unembedded()
    if unembedded:
        unembedded = sanitize_unembedded(unembedded)
        embedded_ids, embed_failed = embed_entries(unembedded)
        if embedded_ids:
            mark_embedded(embedded_ids)
    if supersede_report is not None:
        print(json.dumps(supersede_report, ensure_ascii=False, sort_keys=True))

    record_compile_run()

    if embed_failed:
        raise SystemExit(1)

if __name__ == "__main__":
    main()
