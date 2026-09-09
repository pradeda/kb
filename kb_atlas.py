#!/opt/kb/venv-embed/bin/python
"""KB Atlas — vizuelni pregled oba corpusa iz Chrome i SQLite-a.

Cita embeddinge iz obe Chroma kolekcije, spaja ih sa SQLite metapodacima,
projektuje 768D na 2D klasicnim MDS-om i generise samostalnu HTML stranicu.

Bez LLM poziva. `numpy` i `chromadb` dolaze iz izolovanog `venv-embed` okruzenja.

    /opt/kb/venv-embed/bin/python /opt/kb/kb_atlas.py [--out /opt/kb/atlas/index.html]
"""

import argparse
import json
import math
import os
import sqlite3
import sys
import tempfile
from collections import Counter
from datetime import datetime, timezone

import numpy as np
import chromadb

CORPORA = {
    "homelab": {"collection": "kb_collection", "db": "/opt/kb/kb.db"},
    "ai": {"collection": "ai_kb_collection", "db": "/opt/ai-kb/ai-kb.db"},
}
CHROMA_HOST = "localhost"
CHROMA_PORT = 8000
DEFAULT_OUT = "/opt/kb/atlas/index.html"

NEIGHBORS = 5
# Izvod teksta po unosu. 2400 je bilo postavljeno kad je AI corpus imao dva
# unosa; izmereno 2026-08-08: Homelab medijana 965 (13% preko granice), AI
# medijana 4772 (91% preko granice), pa se skoro svaki AI clanak sekao na pola.
# 6000 pokriva AI medijanu sa rezervom i prakticno ceo Homelab, uz rast
# generisanog fajla ispod 10%. Granica ostaje jer jedan unos ume da bude
# ekstreman — trenutni maksimum je 78.606 znakova.
PREVIEW_CHARS = 6000
# Pragovi izvedeni iz izmerene raspodele nad 487 Homelab vektora (2026-08-02),
# ne pogodjeni: medijana razdaljine svih parova 0.283, do najblizeg suseda 0.120.
# Corpus je semanticki gust jer je celi jedan domen, pa su podrazumevani
# "duplikat" pragovi iz literature previse labavi.
REDUNDANCY_MAX = 0.08   # ispod p25 razdaljina do najblizeg suseda (0.089)
ISOLATED_MIN = 0.28     # iznad p95 razdaljina do najblizeg suseda (0.199)
SEED = 20260802

# tagovi koji ne nose znacenje za labelu klastera
STOP_TAGS = {"url", "web", "note", "test", "youtube", "ai", "kb"}


def log(msg):
    print(f"[atlas] {msg}", file=sys.stderr, flush=True)


# --------------------------------------------------------------------------- ucitavanje

def load_corpus(name, spec, client):
    """Vrati listu zapisa sa embeddingom i metapodacima za jedan corpus."""
    try:
        col = client.get_collection(spec["collection"])
    except Exception as exc:
        log(f"{name}: kolekcija nedostupna ({exc}) — preskacem")
        return []

    got = col.get(include=["metadatas", "embeddings"])
    ids = got.get("ids") or []
    metas = got.get("metadatas") or []
    embs = got.get("embeddings")
    if embs is None or len(ids) == 0:
        log(f"{name}: nema vektora")
        return []

    dates, previews = {}, {}
    if os.path.exists(spec["db"]):
        try:
            conn = sqlite3.connect(f"file:{spec['db']}?mode=ro", uri=True)
            for row_id, created, summary, content in conn.execute(
                "SELECT id, created_at, summary, content FROM entries"
            ):
                key = str(row_id)
                dates[key] = created
                body = (content or "").strip()
                previews[key] = {
                    "summary": (summary or "").strip(),
                    "text": body[:PREVIEW_CHARS],
                    "truncated": len(body) > PREVIEW_CHARS,
                    "chars": len(body),
                }
            conn.close()
        except sqlite3.Error as exc:
            log(f"{name}: SQLite nedostupan ({exc}) — datumi i izvodi izostaju")

    records = []
    for i, entry_id in enumerate(ids):
        meta = metas[i] or {}
        raw_tags = (meta.get("tags") or "").strip()
        tags = [t.strip() for t in raw_tags.split(",") if t.strip()]
        prev = previews.get(str(entry_id), {})
        records.append({
            "key": f"{name}:{entry_id}",
            "corpus": name,
            "id": entry_id,
            "title": (meta.get("title") or "").strip() or f"(bez naslova) {entry_id}",
            "tags": tags,
            "type": meta.get("type") or "",
            "raw_path": meta.get("raw_path") or "",
            "created": dates.get(str(entry_id)) or "",
            "summary": prev.get("summary", ""),
            "text": prev.get("text", ""),
            "truncated": bool(prev.get("truncated")),
            "chars": prev.get("chars", 0),
            "vec": np.asarray(embs[i], dtype=np.float64),
        })
    log(f"{name}: {len(records)} vektora")
    return records


# --------------------------------------------------------------------------- geometrija

def unit(matrix):
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def cosine_distances(matrix):
    sim = np.clip(matrix @ matrix.T, -1.0, 1.0)
    dist = 1.0 - sim
    np.fill_diagonal(dist, 0.0)
    return dist


def classical_mds(dist):
    """Klasicni MDS: dvostruko centriranje pa eigendekompozicija."""
    n = dist.shape[0]
    if n < 3:
        return np.zeros((n, 2))
    sq = dist ** 2
    j = np.eye(n) - np.ones((n, n)) / n
    b = -0.5 * j @ sq @ j
    b = (b + b.T) / 2.0
    vals, vecs = np.linalg.eigh(b)
    order = np.argsort(vals)[::-1][:2]
    vals = np.clip(vals[order], 0.0, None)
    return vecs[:, order] * np.sqrt(vals)


def kmeans(matrix, k, iters=80):
    """k-means++ inicijalizacija, fiksan seed radi ponovljivosti."""
    rng = np.random.default_rng(SEED)
    n = matrix.shape[0]
    k = max(1, min(k, n))
    centers = [matrix[rng.integers(n)]]
    for _ in range(1, k):
        d = np.min(
            np.stack([np.sum((matrix - c) ** 2, axis=1) for c in centers]), axis=0
        )
        total = d.sum()
        probs = d / total if total > 0 else np.full(n, 1.0 / n)
        centers.append(matrix[rng.choice(n, p=probs)])
    centers = np.stack(centers)

    labels = np.zeros(n, dtype=int)
    for _ in range(iters):
        dists = ((matrix[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        new_labels = np.argmin(dists, axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for c in range(k):
            members = matrix[labels == c]
            if len(members):
                centers[c] = members.mean(axis=0)
    return labels, centers


def label_cluster(records, idx):
    """Labela klastera: najcesci smisleni tagovi, fallback na medoid naslov."""
    counts = Counter()
    for i in idx:
        for tag in records[i]["tags"]:
            low = tag.lower()
            if low in STOP_TAGS or len(low) < 2:
                continue
            counts[tag] += 1
    top = [t for t, _ in counts.most_common(3)]
    if top:
        return " · ".join(top)
    return records[idx[0]]["title"][:44]


# --------------------------------------------------------------------------- analiza

def build_model(records):
    vecs = unit(np.stack([r["vec"] for r in records]))
    dist = cosine_distances(vecs)
    n = len(records)
    log(f"matrica razdaljina {n}x{n}")

    coords = classical_mds(dist)
    span = np.max(np.abs(coords)) or 1.0
    coords = coords / span  # normalizovano na [-1, 1]

    k = max(4, min(16, int(round(math.sqrt(n / 2.0)))))
    labels, _ = kmeans(vecs, k)
    log(f"klastera: {len(set(labels.tolist()))}")

    # susedi
    masked = dist.copy()
    np.fill_diagonal(masked, np.inf)
    nn_order = np.argsort(masked, axis=1)[:, :NEIGHBORS]

    for i, rec in enumerate(records):
        rec["x"] = float(coords[i, 0])
        rec["y"] = float(coords[i, 1])
        rec["cluster"] = int(labels[i])
        rec["neighbors"] = [
            {"key": records[j]["key"], "title": records[j]["title"],
             "corpus": records[j]["corpus"], "d": round(float(masked[i, j]), 3)}
            for j in nn_order[i]
        ]
        rec["nearest"] = round(float(masked[i, nn_order[i][0]]), 3) if len(nn_order[i]) else 1.0

    # klasteri
    clusters = []
    for c in sorted(set(labels.tolist())):
        idx = [i for i in range(n) if labels[i] == c]
        by_corpus = Counter(records[i]["corpus"] for i in idx)
        inner = [masked[i, nn_order[i][0]] for i in idx]
        # clanovi sortirani po blizini centru klastera — najreprezentativniji prvi
        members = sorted(idx, key=lambda i: float(np.mean([dist[i, j] for j in idx])))
        clusters.append({
            "id": c,
            "label": label_cluster(records, idx),
            "size": len(idx),
            "homelab": by_corpus.get("homelab", 0),
            "ai": by_corpus.get("ai", 0),
            "cohesion": round(float(np.mean(inner)), 3),
            "members": [records[i]["key"] for i in members],
        })
    clusters.sort(key=lambda c: -c["size"])

    # redundansa
    pairs = []
    iu = np.triu_indices(n, k=1)
    for i, j in zip(*iu):
        d = dist[i, j]
        if d <= REDUNDANCY_MAX:
            pairs.append({
                "a": records[i]["title"], "ak": records[i]["key"], "ac": records[i]["corpus"],
                "b": records[j]["title"], "bk": records[j]["key"], "bc": records[j]["corpus"],
                "d": round(float(d), 3),
            })
    pairs.sort(key=lambda p: p["d"])

    isolated = sorted(
        [r for r in records if r["nearest"] >= ISOLATED_MIN],
        key=lambda r: -r["nearest"],
    )[:20]

    return {
        "generated": datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M %Z"),
        "totals": {
            "all": n,
            "homelab": sum(1 for r in records if r["corpus"] == "homelab"),
            "ai": sum(1 for r in records if r["corpus"] == "ai"),
            "clusters": len(clusters),
            "redundant": len(pairs),
            "isolated": len(isolated),
        },
        "entries": [
            {k: r[k] for k in
             ("key", "corpus", "id", "title", "tags", "type", "raw_path",
              "created", "summary", "text", "truncated", "chars",
              "x", "y", "cluster", "neighbors", "nearest")}
            for r in records
        ],
        "clusters": clusters,
        "redundant": pairs[:60],
        "isolated": [
            {"key": r["key"], "title": r["title"], "corpus": r["corpus"], "d": r["nearest"]}
            for r in isolated
        ],
    }


# --------------------------------------------------------------------------- izlaz

def render(model, template_path):
    with open(template_path, "r", encoding="utf-8") as fh:
        html = fh.read()
    payload = json.dumps(model, ensure_ascii=False, separators=(",", ":"))
    if "__ATLAS_DATA__" not in html:
        raise SystemExit("template nema __ATLAS_DATA__ placeholder")
    return html.replace("__ATLAS_DATA__", payload)


def write_atomic(path, content):
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=directory, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(content)
        os.chmod(tmp, 0o644)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description="Generise KB Atlas HTML pregled.")
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--template", default=os.path.join(here, "atlas_template.html"))
    args = ap.parse_args()

    client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
    records = []
    for name, spec in CORPORA.items():
        records.extend(load_corpus(name, spec, client))

    if len(records) < 3:
        raise SystemExit("premalo vektora za projekciju (potrebno bar 3)")

    model = build_model(records)
    write_atomic(args.out, render(model, args.template))
    t = model["totals"]
    log(f"upisano {args.out} — {t['all']} unosa, {t['clusters']} klastera, "
        f"{t['redundant']} bliskih parova, {t['isolated']} izolovanih")


if __name__ == "__main__":
    main()
