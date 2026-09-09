#!/bin/bash
# /usr/local/bin/kb
# Instalacija: chmod +x /usr/local/bin/kb

DB="/opt/kb/kb.db"
RAW="/opt/kb/raw"

_ts() { date +"%Y-%m-%dT%H:%M:%S"; }
_slug() { echo "$1" | tr '[:upper:]' '[:lower:]' | sed 's/[^a-z0-9]/-/g' | sed 's/--*/-/g' | cut -c1-60; }

case "$1" in
  add)
    TYPE="$2"
    CONTENT="$3"
    TITLE="${4:-}"
    TAGS="${5:-}"
    TS=$(_ts)
    SLUG=$(_slug "${TITLE:-$CONTENT}")
    DATE=$(date +"%Y-%m-%d")

    if [ "$TYPE" = "url" ]; then
      RAW_PATH="$RAW/urls/${DATE}-${SLUG}.md"
    else
      RAW_PATH="$RAW/notes/${DATE}-${SLUG}.md"
    fi

    mkdir -p "$(dirname $RAW_PATH)"
    cat > "$RAW_PATH" <<EOF
---
type: $TYPE
content: $CONTENT
title: $TITLE
tags: $TAGS
saved: $TS
---

$CONTENT
EOF

    sqlite3 "$DB" "INSERT INTO entries (type, content, title, tags, raw_path, created_at)
        VALUES ('$TYPE', '$CONTENT', '$TITLE', '$TAGS', '$RAW_PATH', '$TS');"
    echo "OK: $RAW_PATH"
    ;;

  search)
    sqlite3 -column -header "$DB" \
      "SELECT id, type, title, content, tags, created_at
       FROM entries
       WHERE content LIKE '%$2%' OR title LIKE '%$2%' OR tags LIKE '%$2%'
       ORDER BY created_at DESC LIMIT ${3:-20};"
    ;;

  list)
    sqlite3 -column -header "$DB" \
      "SELECT id, type, title, created_at FROM entries
       ORDER BY created_at DESC LIMIT ${2:-20};"
    ;;

  pending)
    sqlite3 -column -header "$DB" \
      "SELECT id, type, title, raw_path FROM entries
       WHERE compiled_at IS NULL ORDER BY created_at;"
    ;;

  dump)
    sqlite3 "$DB" \
      "SELECT id, type, content, title, tags, raw_path, created_at
       FROM entries ORDER BY created_at DESC;"
    ;;

  *)
    echo "Upotreba: kb [add|search|list|pending|dump]"
    echo "  kb add url 'https://...' 'Naziv' 'tag1,tag2'"
    echo "  kb add note 'Tekst beleske' 'Naziv' 'tag1'"
    echo "  kb search 'docker'"
    echo "  kb list 10"
    echo "  kb pending"
    ;;
esac
