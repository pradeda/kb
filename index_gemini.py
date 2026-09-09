#!/usr/bin/env python3
# /opt/kb/index_gemini.py
# Indexes .gemini/*.md files into ChromaDB with chunking
# Cronjob: once daily at 3:00
# Usage: python3 /opt/kb/index_gemini.py

import os, sys, json, hashlib
from pathlib import Path
from datetime import datetime

ENV_FILE    = Path("/opt/kb/.env")
EMBED_SOCKET = "/run/kb-embed/embed.sock"
CHROMA_HOST = "localhost"
CHROMA_PORT = 8000
COLLECTION = "kb_collection"
CHUNK_SIZE  = 400   # characters per chunk
CHUNK_OVERLAP = 50  # overlap between chunks

# files to index
GEMINI_FILES = [
    Path("/home/turok/.gemini/GOTCHAS.md"),
    Path("/home/turok/.gemini/ARCHIVE.md"),
    Path("/home/turok/.gemini/SERVICES.md"),
    Path("/home/turok/.gemini/HARDWARE.md"),
]

# per-project GEMINI.md files — discovered automatically
PROJECTS_DIR = Path("/home/turok/projects")

# load .env
if ENV_FILE.exists():
    for line in ENV_FILE.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Split text into chunks by paragraph, respecting chunk_size."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    current = []
    current_len = 0

    for para in paragraphs:
        if current_len + len(para) > chunk_size and current:
            chunks.append("\n\n".join(current))
            # overlap — keep the last paragraph
            if len(current) > 1:
                current = current[-1:]
                current_len = len(current[0])
            else:
                current = []
                current_len = 0
        current.append(para)
        current_len += len(para)

    if current:
        chunks.append("\n\n".join(current))

    return chunks

def get_embedding(text):
    import socket
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
        s.settimeout(30)
        s.connect(EMBED_SOCKET)
        s.sendall((text + "\n").encode("utf-8"))
        data = b""
        while True:
            chunk = s.recv(4096)
            if not chunk:
                break
            data += chunk
            if b"\n" in data:
                break
    result = data.decode().strip()
    if result == "null":
        raise RuntimeError("kb-embed daemon returned null")
    return json.loads(result)

def file_hash(path):
    return hashlib.md5(path.read_bytes()).hexdigest()

def get_indexed_hashes(collection):
    """Fetch hashes of already-indexed files."""
    try:
        results = collection.get(where={"source": "gemini-context"}, include=["metadatas"])
        hashes = {}
        for meta in results["metadatas"]:
            fname = meta.get("file", "")
            fhash = meta.get("file_hash", "")
            if fname and fhash:
                hashes[fname] = fhash
        return hashes
    except Exception:
        return {}

def collect_files():
    files = []
    for f in GEMINI_FILES:
        if f.exists():
            files.append(f)
        else:
            print(f"  [SKIP] does not exist: {f}")

    # per-project GEMINI.md files
    if PROJECTS_DIR.exists():
        for gemini_md in PROJECTS_DIR.glob("**/GEMINI.md"):
            files.append(gemini_md)
        for gemini_md in PROJECTS_DIR.glob("**/gemini.md"):
            files.append(gemini_md)

    return files

def index_file(collection, filepath, indexed_hashes):
    fname = str(filepath)
    current_hash = file_hash(filepath)

    # skip if the file hasn't changed
    if indexed_hashes.get(fname) == current_hash:
        print(f"  [SKIP] unchanged: {filepath.name}")
        return 0

    text = filepath.read_text(encoding="utf-8")
    chunks = chunk_text(text)

    # delete stale chunks for this file
    try:
        existing = collection.get(where={"file": fname}, include=["ids"])
        if existing["ids"]:
            collection.delete(ids=existing["ids"])
            print(f"  [DEL] removed {len(existing['ids'])} stale chunks for {filepath.name}")
    except Exception as e:
        print(f"  [WARN] stale chunk cleanup: {e}")

    # index new chunks
    indexed = 0
    for i, chunk in enumerate(chunks):
        chunk_id = f"gemini_{hashlib.md5(f'{fname}_{i}'.encode()).hexdigest()[:12]}"
        try:
            embedding = get_embedding(chunk)
            collection.upsert(
                ids=[chunk_id],
                embeddings=[embedding],
                documents=[chunk],
                metadatas=[{
                    "source": "gemini-context",
                    "file": fname,
                    "file_name": filepath.name,
                    "file_hash": current_hash,
                    "chunk_index": i,
                    "indexed_at": datetime.now().isoformat(),
                }]
            )
            indexed += 1
        except Exception as e:
            print(f"  [ERROR] chunk {i} of {filepath.name}: {e}")

    print(f"  [OK] {filepath.name} → {indexed}/{len(chunks)} chunks")
    return indexed

def main():
    print(f"Indexing .gemini/ files — {datetime.now().strftime('%Y-%m-%d %H:%M')}")

    import chromadb
    client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
    collection = client.get_or_create_collection(
        name=COLLECTION,
        metadata={"hnsw:space": "cosine"}
    )

    print(f"ChromaDB collection '{COLLECTION}': {collection.count()} chunks total")

    files = collect_files()
    print(f"Files to index: {len(files)}")

    indexed_hashes = get_indexed_hashes(collection)
    total_indexed = 0

    for filepath in files:
        total_indexed += index_file(collection, filepath, indexed_hashes)

    print(f"\nDone. New chunks: {total_indexed}")
    print(f"ChromaDB total: {collection.count()} chunks")

if __name__ == "__main__":
    main()
