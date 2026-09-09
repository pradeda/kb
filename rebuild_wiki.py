#!/usr/bin/env python3
"""Rebuild KB wiki from raw .md files with category-based structure."""

import os, re, sys
from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime

KB = Path("/opt/kb")
RAW = KB / "raw"
WIKI = KB / "wiki2"
MIN_CONTENT_FOR_PAGE = 200  # chars
MIN_CONTENT_AT_ALL = 150  # below this, don't even include as short note

# Blacklist: entries matching these patterns are excluded from wiki entirely
TAG_BLACKLIST = {
    "session", "changelog", "archive", "volatile", "debug", "test",
    "todo", "todos", "opinion", "recenzije", "rebranding",
}
TITLE_BLACKLIST_PREFIXES = (
    "Sesija", "Session Summary", "Session summary", "CHANGELOG",
    "[AUTO]", "[RUČNO]",
)
CONTENT_SKIP_SIGNALS = (
    "git diff", "docker logs", "STDERR", "WARN ", "ERROR ",
)

# Tag → category mapping
TAG_MAP = {
    # Services
    "docker": "servisi", "traefik": "servisi", "nginx": "servisi",
    "nginx-proxy-manager": "servisi", "npm": "servisi",
    "beszel": "servisi", "monitoring": "servisi", "prometheus": "servisi",
    "grafana": "servisi", "homepage-dashboard": "servisi",
    "navidrome": "servisi", "memos": "servisi", "metube": "servisi",
    "omni-tools": "servisi", "searxng": "servisi", "ttyd": "servisi",
    "upsnap": "servisi", "portainer": "servisi", "dockge": "servisi",
    "flaresolverr": "servisi", "lanlens": "servisi", "ai-brain": "servisi",
    "picard": "servisi", "prowlarr": "servisi", "sonarr": "servisi",
    "radarr": "servisi", "bazarr": "servisi", "jackett": "servisi",
    "qbittorrent": "servisi", "swaparr": "servisi", "readmeabook": "servisi",
    "rmab": "servisi", "arr": "servisi", "arrs": "servisi", "arr-stack": "servisi",
    "karakeep": "servisi", "audiobookshelf": "servisi",
    "telegram": "servisi", "loremaster": "servisi", "bot": "servisi",
    "webhook": "servisi", "n8n": "servisi",
    # Hardver
    "nexus": "hardver", "rpi4": "hardver", "nas": "hardver",
    "forrix": "hardver", "nexus2": "hardver", "turok-pc": "hardver",
    "windows": "hardver", "hardware": "hardver", "ssd": "hardver",
    "sdcard": "hardver", "usb": "hardver", "disk": "hardver",
    "boot": "hardver", "amd": "hardver", "gpu": "hardver",
    "seagate": "hardver", "synology": "hardver", "proxmox": "hardver",
    "raspberry-pi": "hardver", "workstation": "hardver",
    # Network
    "nfs": "mreza", "dns": "mreza", "wireguard": "mreza",
    "vpn": "mreza", "protonvpn": "mreza", "firewall": "mreza",
    "fail2ban": "mreza", "ssh": "mreza", "network": "mreza",
    "adblock": "mreza", "pihole": "mreza", "pi-hole": "mreza",
    "authelia": "mreza", "auth": "mreza", "autentifikacija": "mreza",
    "proxy": "mreza", "reverse-proxy": "mreza", "ssl": "mreza",
    "hard-mount": "mreza", "protocol": "mreza",
    # AI
    "ollama": "ai", "deepseek": "ai", "openrouter": "ai",
    "agenti": "ai", "agents": "ai", "ai": "ai", "ai-agents": "ai",
    "llm": "ai", "open-webui": "ai", "qwen": "ai", "qwen3.6": "ai",
    "vllm": "ai", "embedding": "ai", "rag": "ai", "semantic-search": "ai",
    "anythingllm": "ai", "crawl4ai": "ai", "jina": "ai", "web-search": "ai",
    "whisper-deployment": "ai", "whisper-ahk-dictation": "ai",
    # Dev
    "kb": "dev", "kb-go": "dev", "kb-ask": "dev", "kb-cli": "dev",
    "opencode": "dev", "api": "dev", "python": "dev", "go": "dev",
    "rust": "dev", "mcp": "dev", "sqlite": "dev", "chromadb": "dev",
    "fts5": "dev", "cli": "dev", "gitops": "dev", "terraform": "dev",
    "kubernetes": "dev", "k3s": "dev", "iac": "dev", "infrastructure": "dev",
    "nodejs": "dev", "json": "dev", "yaml": "dev", "async": "dev",
    "asyncio": "dev", "httpx": "dev", "playwright": "dev", "scraping": "dev",
    "fastapi": "dev", "coding-principles": "dev", "devops": "dev",
    "frontend": "dev", "react": "dev", "design": "dev", "ui": "dev",
    "typescript": "dev", "dispatch": "dev",
    # Gotchas
    "gotcha": "gotchas", "bug": "gotchas", "bugfix": "gotchas",
    "fix": "gotchas", "crash": "gotchas", "incident": "gotchas",
    "debug": "gotchas", "debugging": "gotchas", "kdump": "gotchas",
    "kernel": "gotchas", "uas": "gotchas", "xhci": "gotchas",
    # Linkovi
    "url": "linkovi", "repo": "linkovi", "github": "linkovi",
    "awesome-list": "linkovi", "web": "linkovi", "research": "linkovi",
    "tutorial": "linkovi", "docs": "linkovi", "volatile": "linkovi",
    "pricing": "linkovi", "tools": "linkovi",
    # Sesije
    "session": "sesije", "architecture": "sesije", "arhitektura": "sesije",
    "decision": "sesije", "plan": "sesije", "todo": "sesije", "todos": "sesije",
    "workflow": "sesije", "documentation": "sesije", "archive": "sesije",
}

# Category metadata
CATEGORIES = {
    "servisi": {"icon": "⚙️", "title": "Servisi", "desc": "Docker servisi, monitoring, automatizacija, botovi"},
    "hardver": {"icon": "🖥️", "title": "Hardver", "desc": "Mašine, diskove, boot, periferija"},
    "mreza":  {"icon": "🌐", "title": "Mreža", "desc": "NFS, DNS, VPN, firewall, proxy"},
    "ai":     {"icon": "🧠", "title": "AI / LLM", "desc": "Modeli, embedding, RAG, agenti"},
    "dev":    {"icon": "💻", "title": "Dev / Kod", "desc": "KB infrastruktura, API, Python, Go, MCP"},
    "gotchas":{"icon": "⚠️", "title": "Gotchas", "desc": "Bugovi, crash-ovi, poznati problemi i rešenja"},
    "linkovi":{"icon": "🔖", "title": "Linkovi", "desc": "GitHub repo-ovi, članci, reference"},
    "sesije": {"icon": "📋", "title": "Sesije", "desc": "Dnevni session summary-i i planovi"},
}


def slugify(text):
    """Convert text to filesystem-safe slug."""
    # Transliterate Serbian chars
    replacements = {"č": "c", "ć": "c", "š": "s", "ž": "z", "đ": "dj", "Č": "C", "Ć": "C", "Š": "S", "Ž": "Z", "Đ": "Dj"}
    for k, v in replacements.items():
        text = text.replace(k, v)
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", text.lower()).strip("-")
    return slug[:60]


def parse_frontmatter(text):
    """Parse YAML-like frontmatter from markdown text."""
    fm = {}
    body = text
    if text.startswith("---"):
        parts = text.split("---", 2)
        if len(parts) >= 3:
            for line in parts[1].strip().split("\n"):
                if ":" in line:
                    k, v = line.split(":", 1)
                    fm[k.strip()] = v.strip()
            body = parts[2].strip()
    return fm, body


def categorize(tags_str):
    """Map comma-separated tags to best category."""
    if not tags_str:
        return "servisi"  # default
    tags = [t.strip().lower() for t in tags_str.split(",")]
    scores = Counter()
    for tag in tags:
        cat = TAG_MAP.get(tag)
        if cat:
            scores[cat] += 1
    if scores:
        return scores.most_common(1)[0][0]
    return "servisi"


def should_skip(info):
    """Return True if entry should be excluded from wiki (session, changelog, log dump, etc)."""
    title = info.get("title", "")
    tags = set(t.strip().lower() for t in info.get("tags", "").split(",") if t.strip())
    body = info.get("body", "")
    length = info.get("length", 0)

    # 1. Too short — noise
    if length < MIN_CONTENT_AT_ALL:
        return True, "prekratko"

    # 2. Tag blacklist
    if tags & TAG_BLACKLIST:
        return True, f"tag: {tags & TAG_BLACKLIST}"

    # 3. Title patterns — session summaries, changelog entries
    for prefix in TITLE_BLACKLIST_PREFIXES:
        if title.startswith(prefix):
            return True, f"naslov: {title[:50]}"

    # 4. Date-prefixed title with colon (e.g. "2026-04-29: kb-ask...")
    if re.match(r"^\d{4}-\d{2}-\d{2}:", title):
        return True, f"datum-prefix: {title[:50]}"

    # 5. GOTCHA: prefix in title — already covered by gotchas category
    if title.upper().startswith("GOTCHA"):
        return True, f"gotcha-naslov: {title[:50]}"

    # 6. Body starts with date heading (changelog-like)
    if re.match(r"^## \d{4}-\d{2}-\d{2}", body.strip()):
        return True, "datum-heading u body"

    # 7. Raw log signals — log dumps, not knowledge
    body_lower = body[:500].lower()
    for signal in CONTENT_SKIP_SIGNALS:
        if signal in body_lower:
            return True, f"signal: {signal}"

    # 8. Body is mostly code blocks or json blobs
    code_block_count = body_lower.count("```")
    if code_block_count >= 4 and length < 2000:
        return True, "mostly code/json"

    return False, ""


def format_wiki_page(title, content, date="", tags_str="", source_file=""):
    """Format a wiki page with metadata header."""
    header = f"---\ntitle: {title}\n"
    if date:
        header += f"date: {date}\n"
    if tags_str:
        header += f"tags: {tags_str}\n"
    header += f"source: {source_file}\n---\n\n"
    return header + content


def backup_existing():
    """Backup current wiki directory."""
    backup_dir = KB / f"wiki.backup.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    if WIKI.exists():
        import shutil
        shutil.move(str(WIKI), str(backup_dir))
        print(f"  Backed up to {backup_dir}")


def main():
    print("Rebuilding wiki from raw .md files...\n")

    # Backup if wiki2 exists
    if WIKI.exists():
        backup_existing()

    # Collect all entries grouped by category
    categories = defaultdict(lambda: {"pages": [], "short": [], "_skipped": []})

    for subdir in ["notes", "urls"]:
        raw_sub = RAW / subdir
        if not raw_sub.exists():
            continue
        for md_file in sorted(raw_sub.glob("*.md")):
            try:
                text = md_file.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                text = md_file.read_text(encoding="latin-1")

            fm, body = parse_frontmatter(text)
            if not body.strip():
                # Use first line of frontmatter as content hint
                body = fm.get("title", md_file.stem)

            raw_title = fm.get("title", "")
            if raw_title and raw_title.strip() and not raw_title.strip().startswith("---"):
                title = raw_title.strip()
            else:
                # Fallback: use first meaningful line from body
                title = None
                for line in body.split("\n"):
                    stripped = line.strip()
                    # Skip separators, empty lines, frontmatter markers, metadata
                    if not stripped or stripped.startswith("---") or stripped.startswith("==="):
                        continue
                    if re.match(r"^[\|\\\-—]+$", stripped):
                        continue
                    if stripped.startswith("#"):
                        title = stripped.lstrip("#").strip()
                        break
                    if re.match(r"^(Metapodaci|Metapodaci:|type:|content:|tags:|saved:)", stripped):
                        continue
                    title = stripped[:80]
                    break
                if not title:
                    title = md_file.stem.replace("-", " ").title()
            tags_str = fm.get("tags", "")
            date = fm.get("saved", fm.get("date", ""))
            etype = fm.get("type", subdir.rstrip("s"))

            cat = categorize(tags_str)

            info = {
                "slug": "",  # filled later
                "title": title,
                "body": body,
                "tags": tags_str,
                "date": date,
                "etype": etype,
                "length": len(body),
                "source": f"raw/{subdir}/{md_file.name}",
            }

            # Filter: skip session logs, changelogs, log dumps
            skip, reason = should_skip(info)
            if skip:
                categories[cat]["_skipped"].append((md_file.name, reason))
                continue

            page_content = format_wiki_page(
                title, body, date, tags_str,
                source_file=info["source"]
            )

            # Generate clean slug: use title if short, otherwise truncate
            base_slug = slugify(title) or slugify(md_file.stem)
            if len(base_slug) > 50:
                base_slug = base_slug[:50].rstrip("-")
            slug = base_slug

            info["slug"] = slug
            info["content"] = page_content

            if len(body) >= MIN_CONTENT_FOR_PAGE:
                categories[cat]["pages"].append(info)
            else:
                categories[cat]["short"].append(info)

    # Generate wiki structure
    WIKI.mkdir(parents=True, exist_ok=True)

    total_pages = 0
    cat_list = []

    for cat_slug, cat_meta in CATEGORIES.items():
        data = categories.get(cat_slug, {"pages": [], "short": [], "_skipped": []})
        pages = data["pages"]
        shorts = data["short"]
        skipped = data["_skipped"]

        cat_dir = WIKI / cat_slug
        cat_dir.mkdir(parents=True, exist_ok=True)

        # Write individual pages
        for p in pages:
            page_path = cat_dir / f"{p['slug']}.md"
            page_path.write_text(p["content"], encoding="utf-8")
            total_pages += 1

        # Write category index (_index.md or index.md)
        # MkDocs material uses index.md for section landing pages
        index_content = f"# {cat_meta['title']}\n\n{cat_meta['desc']}\n\n"

        if pages:
            index_content += "## Stranice\n\n"
            for p in sorted(pages, key=lambda x: x["title"]):
                idx_title = p["title"].replace("|", "—")[:80]
                date_str = p["date"][:10] if p["date"] else ""
                desc = p["body"][:120].replace("\n", " ").strip()
                index_content += f"- **[{idx_title}]({p['slug']}.md)** — {desc}"
                if date_str:
                    index_content += f" _{date_str}_"
                index_content += "\n"

        if shorts:
            index_content += "\n## Kratke beleške\n\n"
            # Group short entries by topic
            topic_groups = defaultdict(list)
            for s in shorts:
                key = s["title"].split(":")[0].split(" - ")[0].strip()[:40]
                topic_groups[key].append(s)

            for topic, entries in sorted(topic_groups.items()):
                if len(entries) == 1:
                    s = entries[0]
                    date_str = s["date"][:10] if s["date"] else ""
                    index_content += f"- **{s['title'][:100]}** — {s['body'][:150].strip()}"
                    if date_str:
                        index_content += f" _{date_str}_"
                    index_content += "\n"
                else:
                    index_content += f"- **{topic}** ({len(entries)}): "
                    # Merge bodies
                    bodies = [e["body"][:100].strip() for e in entries]
                    index_content += "; ".join(bodies)
                    index_content += "\n"

        index_path = cat_dir / "index.md"
        index_path.write_text(index_content, encoding="utf-8")

        # Build category entry for root index
        cat_list.append({
            "slug": cat_slug,
            "icon": cat_meta["icon"],
            "title": cat_meta["title"],
            "desc": cat_meta["desc"],
            "pages": len(pages),
            "shorts": len(shorts),
        })

    # Write root index
    total_skipped = sum(len(cat["_skipped"]) for cat in categories.values())
    root = WIKI / "index.md"
    root_content = f"""# Nexus Wiki

Dobrodošli u Nexus Wiki — strukturirani pregled celog homelab znanja.

Auto-generisano iz 222 raw `.md` beleški ({total_skipped} preskočeno — sesije, changelogs).
**{total_pages} stranica**, **{sum(len(cat['short']) for cat in categories.values())} kratkih beleški**.

---

"""

    for cat in cat_list:
        total = cat["pages"] + cat["shorts"]
        p_str = f"{cat['pages']} str." if cat["pages"] else ""
        s_str = f"{cat['shorts']} beleški" if cat["shorts"] else ""
        count = ", ".join(filter(None, [p_str, s_str]))
        root_content += f"- {cat['icon']} **[{cat['title']}]({cat['slug']}/index.md)** — {cat['desc']} ({count})\n"

    root_content += "\n---\n*Poslednje generisanje: " + datetime.now().strftime("%Y-%m-%d %H:%M") + "*\n"
    root_path = WIKI / "index.md"
    root_path.write_text(root_content, encoding="utf-8")

    # Stats
    print(f"\n{'='*50}")
    print(f"Wiki rebuild complete: {WIKI}")
    print(f"  Skipped: {total_skipped} entries (sessions, changelogs, log dumps)")
    print(f"{'='*50}")
    for cat in cat_list:
        data = categories.get(cat["slug"], {"pages": [], "short": [], "_skipped": []})
        total = cat["pages"] + cat["shorts"]
        sk = len(data["_skipped"])
        sk_str = f"  ⊘{sk}" if sk else ""
        print(f"  {cat['icon']} {cat['title']:15s}  {cat['pages']:3d} pages  {cat['shorts']:3d} notes  = {total}{sk_str}")
    print(f"\n  Total: {total_pages} pages + {sum(len(cat['short']) for cat in categories.values())} short notes  (⊘{total_skipped} skipped)")


if __name__ == "__main__":
    main()
