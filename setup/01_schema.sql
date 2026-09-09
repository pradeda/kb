-- /opt/kb/kb.db schema
-- Pokreni: sqlite3 /opt/kb/kb.db < /opt/kb/01_schema.sql

CREATE TABLE IF NOT EXISTS entries (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    type        TEXT NOT NULL CHECK(type IN ('url', 'note')),
    content     TEXT NOT NULL,
    title       TEXT DEFAULT '',
    summary     TEXT DEFAULT '',
    tags        TEXT DEFAULT '',
    raw_path    TEXT DEFAULT '',
    source      TEXT DEFAULT 'telegram',
    compiled_at DATETIME,
    created_at  DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE VIRTUAL TABLE IF NOT EXISTS entries_fts
    USING fts5(title, content, summary, tags, content='entries', content_rowid='id');

CREATE TRIGGER IF NOT EXISTS entries_fts_insert AFTER INSERT ON entries BEGIN
    INSERT INTO entries_fts(rowid, title, content, summary, tags)
    VALUES (new.id, new.title, new.content, new.summary, new.tags);
END;

CREATE TRIGGER IF NOT EXISTS entries_fts_update AFTER UPDATE ON entries BEGIN
    UPDATE entries_fts SET title=new.title, content=new.content,
        summary=new.summary, tags=new.tags WHERE rowid=new.id;
END;
