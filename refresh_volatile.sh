#!/usr/bin/env bash
# Refresh URL entries tagged "volatile" when they are older than DAYS.

set -euo pipefail

DB="${KB_REFRESH_DB:-/opt/kb/kb.db}"
DAYS="${KB_REFRESH_DAYS:-3}"
SQLITE="${KB_REFRESH_SQLITE:-/usr/bin/sqlite3}"
CURL="${KB_REFRESH_CURL:-/usr/bin/curl}"
PY="${KB_REFRESH_PYTHON:-/opt/kb/venv-embed/bin/python}"
COMPILE="${KB_REFRESH_COMPILE:-/opt/kb/compile.py}"
ENV_FILE="${KB_REFRESH_ENV:-/opt/kb/.env}"

# shellcheck source=/dev/null
[ -r "$ENV_FILE" ] && . "$ENV_FILE"

echo "=== Volatile refresh — $(date) ==="

INVALID_IDS=$("$SQLITE" -noheader "$DB" "
    SELECT group_concat(id, ',')
    FROM entries
    WHERE type='url'
      AND tags LIKE '%volatile%'
      AND created_at < datetime('now', '-${DAYS} days')
      AND content NOT LIKE 'http://%'
      AND content NOT LIKE 'https://%';
")
if [ -n "$INVALID_IDS" ]; then
    echo "WARNING: volatile entries without a refreshable URL in content; skipped IDs: $INVALID_IDS" >&2
fi

URLS=$("$SQLITE" -separator $'\x1f' "$DB" "
    SELECT id, content, tags
    FROM entries
    WHERE type='url'
      AND tags LIKE '%volatile%'
      AND created_at < datetime('now', '-${DAYS} days')
      AND (content LIKE 'http://%' OR content LIKE 'https://%')
    ORDER BY created_at ASC;
")

if [ -z "$URLS" ]; then
    echo "No volatile URLs to refresh."
    exit 0
fi

COUNT=0
while IFS=$'\x1f' read -r id url tags; do
    echo "Refreshing #$id: $url"
    JINA_RESPONSE=$("$CURL" -fsS -H "Accept: application/json" "https://r.jina.ai/$url")

    TITLE=$(printf '%s' "$JINA_RESPONSE" | "$PY" -c '
import json, re, sys
data = json.load(sys.stdin)
raw = data.get("data", "")
match = re.search(r"^Title:\s*(.+)$", raw, re.MULTILINE)
print(match.group(1).strip() if match else "Untitled")
')
    FRESH_CONTENT=$(printf '%s' "$JINA_RESPONSE" | "$PY" -c '
import json, sys
print(json.load(sys.stdin).get("data", "")[:4000])
')

    if [ -z "$FRESH_CONTENT" ]; then
        echo "  SKIP: empty Jina content for #$id" >&2
        continue
    fi

    # This legacy job updates in place. sqlite quote() keeps arbitrary fetched
    # text out of SQL syntax; the entries_fts_update trigger refreshes FTS5.
    Q_CONTENT=$(printf '%s' "$FRESH_CONTENT" | "$PY" -c \
        'import sqlite3,sys; print(sqlite3.connect(":memory:").execute("SELECT quote(?)", (sys.stdin.read(),)).fetchone()[0])')
    Q_TITLE=$(printf '%s' "$TITLE" | "$PY" -c \
        'import sqlite3,sys; print(sqlite3.connect(":memory:").execute("SELECT quote(?)", (sys.stdin.read(),)).fetchone()[0])')
    "$SQLITE" "$DB" "
        UPDATE entries SET
            content=$Q_CONTENT,
            title=$Q_TITLE,
            summary='',
            compiled_at=NULL,
            embedded_at=NULL,
            created_at=CURRENT_TIMESTAMP
        WHERE id=$id;
    "
    COUNT=$((COUNT + 1))
    echo "  OK: $TITLE"
done <<< "$URLS"

echo "Refreshed: $COUNT URLs"
if [ "$COUNT" -gt 0 ]; then
    "$PY" "$COMPILE"
fi

echo "=== Done ==="
