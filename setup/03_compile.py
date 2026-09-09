#!/usr/bin/env python3
# /opt/kb/compile.py
# Upotreba: python3 /opt/kb/compile.py
# Env: OPENROUTER_API_KEY mora biti setovan (ili u /opt/kb/.env)

import sqlite3, os, re, sys
from datetime import datetime
from pathlib import Path

# --- konfiguracija ---
KB          = Path("/opt/kb")
DB          = KB / "kb.db"
WIKI        = KB / "wiki"
RAW         = KB / "raw"
PROMPT_FILE = KB / "prompts" / "compiler.md"
ENV_FILE    = KB / ".env"

MODEL       = "google/gemini-2.0-flash-lite-001"
BATCH_SIZE  = 5
MAX_TOKENS  = 8000

# --- učitaj .env ako postoji ---
if ENV_FILE.exists():
    for line in ENV_FILE.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

API_KEY = os.environ.get("OPENROUTER_API_KEY")
if not API_KEY:
    print("GREŠKA: OPENROUTER_API_KEY nije setovan")
    sys.exit(1)

# --- funkcije ---
def get_new_entries():
    db = sqlite3.connect(DB)
    rows = db.execute("""
        SELECT id, type, content, title, tags, raw_path, created_at
        FROM entries
        WHERE compiled_at IS NULL
        ORDER BY created_at
    """).fetchall()
    db.close()
    return rows

def mark_compiled(ids):
    db = sqlite3.connect(DB)
    placeholders = ",".join("?" * len(ids))
    db.execute(
        f"UPDATE entries SET compiled_at = ? WHERE id IN ({placeholders})",
        [datetime.now().isoformat()] + list(ids)
    )
    db.commit()
    db.close()

def read_file_safe(path):
    try:
        return Path(path).read_text(encoding="utf-8")
    except Exception:
        return ""

def build_user_message(entries, index_content):
    raws = []
    for row in entries:
        raw_path = row[5]
        content = read_file_safe(raw_path) if raw_path else row[2]
        raws.append(f"=== {raw_path or 'note'} ===\n{content}")

    return f"""Trenutni wiki/index.md:
{index_content or '(prazan — ovo su prvi unosi)'}

---
Novi unosi za kompajliranje ({len(entries)} ukupno):

{chr(10).join(raws)}
"""

def call_openrouter(system_prompt, user_message):
    import urllib.request, json

    payload = json.dumps({
        "model": MODEL,
        "max_tokens": MAX_TOKENS,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_message},
        ],
    }).encode()

    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=payload,
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read())

    return data["choices"][0]["message"]["content"]

def parse_and_write(response_text):
    # format koji agent vraca:
    # FILE: wiki/concepts/naziv.md
    # ---
    # sadrzaj
    # ---
    pattern = r"FILE:\s*(.+?)\n---\n(.*?)\n---"
    matches = re.findall(pattern, response_text, re.DOTALL)
    written = []
    for rel_path, content in matches:
        full_path = KB / rel_path.strip()
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(content.strip(), encoding="utf-8")
        written.append(rel_path.strip())
        print(f"  [OK] {rel_path.strip()}")
    return written

def main():
    entries = get_new_entries()
    if not entries:
        print("Nema novih unosa za kompajliranje.")
        return

    print(f"Ukupno novih unosa: {len(entries)}")
    system_prompt = PROMPT_FILE.read_text(encoding="utf-8")
    index_content = read_file_safe(WIKI / "index.md")

    all_compiled_ids = []

    for i in range(0, len(entries), BATCH_SIZE):
        batch = entries[i:i + BATCH_SIZE]
        batch_num = i // BATCH_SIZE + 1
        total_batches = (len(entries) + BATCH_SIZE - 1) // BATCH_SIZE
        print(f"\nBatch {batch_num}/{total_batches} ({len(batch)} unosa)...")

        user_msg = build_user_message(batch, index_content)

        try:
            response = call_openrouter(system_prompt, user_msg)
        except Exception as e:
            print(f"  GREŠKA API poziv: {e}")
            continue

        written = parse_and_write(response)

        if written:
            all_compiled_ids += [row[0] for row in batch]
            # osvezi index za sledeci batch
            index_content = read_file_safe(WIKI / "index.md")
        else:
            print("  UPOZORENJE: agent nije vratio FILE: blokove")
            print("  Prvih 300 karaktera responsa:")
            print(response[:300])

    if all_compiled_ids:
        mark_compiled(all_compiled_ids)
        print(f"\nKompajlirano: {len(all_compiled_ids)} unosa.")
    else:
        print("\nNijedan unos nije kompajliran — proveri compiler.md prompt.")

if __name__ == "__main__":
    main()
