# KB Setup — Instrukcije za Gemini

Implementiraj kompletan Personal Knowledge Base sistem na nexusu (ThinkCentre, Debian 13,
IP: nexus, /home/turok). Korisnik šalje URL ili tekst via Telegram, sistem kompajlira wiki.

---

## Korak 1 — Direktorijumska struktura (nexus)

```bash
mkdir -p /opt/kb/{raw/urls,raw/notes,wiki/{concepts,sources,queries},prompts}
chown -R turok:turok /opt/kb
```

---

## Korak 2 — SQLite baza

Fajl: `01_schema.sql`
Pokreni:
```bash
sqlite3 /opt/kb/kb.db < /opt/kb/01_schema.sql
```

---

## Korak 3 — kb CLI

Fajl: `02_kb_cli.sh`
```bash
cp 02_kb_cli.sh /usr/local/bin/kb
chmod +x /usr/local/bin/kb
```

Test:
```bash
kb add note "test beleska" "Test" "test"
kb list
kb pending
```

---

## Korak 4 — OpenRouter API key

OpenRouter API key je dostupan u gemini.md (hashovan). Dekoduj ga i postavi:
```bash
echo "OPENROUTER_API_KEY=sk-or-..." > /opt/kb/.env
chmod 600 /opt/kb/.env
```

---

## Korak 5 — Compiler prompt

Fajl: `04_compiler.md`
```bash
cp 04_compiler.md /opt/kb/prompts/compiler.md
```

---

## Korak 6 — compile.py

Fajl: `03_compile.py`
```bash
cp 03_compile.py /opt/kb/compile.py
chmod +x /opt/kb/compile.py
```

Test sa prvim unosima:
```bash
kb add url "https://docs.docker.com/network/" "Docker networking" "docker,networking"
kb add note "Ollama radi na nexusu na portu 11434, modeli su u /root/.ollama/models" "Ollama setup" "ollama,homelab"
python3 /opt/kb/compile.py
```

Proveri output:
```bash
ls /opt/kb/wiki/concepts/
ls /opt/kb/wiki/sources/
cat /opt/kb/wiki/index.md
```

---

## Korak 7 — compile.sh wrapper

Fajl: `05_compile.sh`
```bash
cp 05_compile.sh /opt/kb/compile.sh
chmod +x /opt/kb/compile.sh
```

Cronjob (svakih 6 sati):
```bash
crontab -e
# dodaj:
0 */6 * * * /opt/kb/compile.sh >> /var/log/kb-compile.log 2>&1
```

---

## Korak 8 — MCP filesystem server (za Claude Code na Forrixu)

Na nexusu:
```bash
npm install -g @modelcontextprotocol/server-filesystem
```

Na Forrixu u WSL, fajl `~/.claude/claude.json`:
```json
{
  "mcpServers": {
    "kb-wiki": {
      "command": "ssh",
      "args": ["nexus", "npx @modelcontextprotocol/server-filesystem /opt/kb/wiki"]
    },
    "kb-sqlite": {
      "command": "ssh",
      "args": ["nexus", "npx @modelcontextprotocol/server-sqlite /opt/kb/kb.db"]
    }
  }
}
```

---

## Korak 9 — n8n workflow (Synology NAS)

Kreiraj novi workflow u n8n sa sledećim nodovima:

1. **Telegram Trigger** — Bot token iz postojećeg bota ili novi
2. **IF** — condition: `{{ $json.message.text.startsWith('http') }}`
3. **Grana URL:**
   - HTTP Request (GET `{{ $json.message.text }}`) — za fetch title iz og:title / title taga
   - SSH Execute na nexus: `kb add url '{{ $json.message.text }}' '{{ $json.title }}' ''`
4. **Grana NOTE:**
   - SSH Execute na nexus: `kb add note '{{ $json.message.text }}' '' ''`
5. **Telegram node** — Reply: `✓ Sačuvano`

SSH konekcija u n8n: host=nexus, user=turok, key iz Synology credential store.

---

## Korak 10 — Obsidian vault mount (Forrix, Windows)

Opcija A — SMB share:
Na nexusu:
```bash
# instalacija samba ako nije
apt install samba -y
# dodaj u /etc/samba/smb.conf:
[kb-wiki]
path = /opt/kb/wiki
browseable = yes
read only = no
valid users = turok
```

Na Forrixu (PowerShell kao admin):
```powershell
net use W: \\nexus\kb-wiki /persistent:yes
```

Obsidian → Open folder as vault → W:\

---

## Verifikacija kompletnog sistema

```bash
# 1. Pošalji URL via Telegram botu
# 2. Proveri da je stiglo u bazu:
kb list 1
# 3. Ručno kompajliraj:
python3 /opt/kb/compile.py
# 4. Proveri wiki:
cat /opt/kb/wiki/index.md
# 5. Otvori Obsidian — trebalo bi da se vidi novi članak
```

---

## Napomene

- `compile.py` koristi samo stdlib (urllib) — nema pip dependency
- Batch size je 5 unosa po API pozivu zbog 8K output limit Gemini Flash Lite
- `compiled_at` kolona prati šta je već kompajlirano — nema duplikata
- Compiler prompt je u `04_compiler.md` — iterišeš ga dok output ne izgleda dobro
- Za Gemini CLI pristup wikiju: `ssh nexus "cat /opt/kb/wiki/index.md"` kao context inject
