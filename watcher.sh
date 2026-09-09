#!/bin/bash
# One allowlisted implementation, run as separate systemd services per corpus.
set -u

corpus="homelab"
if [[ "${1:-}" == "--corpus" ]]; then
    if [[ $# -lt 2 ]]; then
        echo "Usage: watcher.sh [--corpus homelab|ai] [--show-profile]" >&2
        exit 2
    fi
    corpus="${2:-}"
    shift 2
fi

case "$corpus" in
    homelab)
        rawdir="/opt/kb/raw"
        compile=(/opt/kb/venv-embed/bin/python /opt/kb/compile.py)
        lock="/tmp/kb-watcher.lock"
        state="/tmp/kb-watcher-last"
        ;;
    ai)
        rawdir="/opt/ai-kb/raw"
        compile=(/opt/kb/venv-embed/bin/python /opt/kb/compile.py --corpus ai)
        lock="/tmp/ai-kb-watcher.lock"
        state="/tmp/ai-kb-watcher-last"
        ;;
    *)
        echo "unknown corpus: $corpus (allowed: homelab, ai)" >&2
        exit 2
        ;;
esac

if [[ "${1:-}" == "--show-profile" ]]; then
    printf 'corpus=%s\nraw=%s\nlock=%s\nstate=%s\ncompile=' "$corpus" "$rawdir" "$lock" "$state"
    printf '%q ' "${compile[@]}"
    printf '\n'
    exit 0
fi
if [[ $# -ne 0 ]]; then
    echo "Usage: watcher.sh [--corpus homelab|ai] [--show-profile]" >&2
    exit 2
fi

min_interval=5

# Prolaz pri startu: inotify je jedini okidac, pa unos koji je usao dok je
# watcher bio ugasen ne bi bio embedovan sve dok neki nevezan add ne probudi
# petlju. Ponedeljni reboot u 04:30 je tacno taj slucaj.
(
    flock 200
    date +%s > "$state"
    "${compile[@]}"
) 200>"$lock"

inotifywait -m -r -e close_write -e moved_to --format "%w%f" "$rawdir" 2>/dev/null |
while read -r filepath; do
    [[ "$filepath" != *.md ]] && continue

    now=$(date +%s)
    last=$(cat "$state" 2>/dev/null || echo 0)
    elapsed=$(( now - last ))
    if (( elapsed < min_interval )); then
        sleep $(( min_interval - elapsed ))
    fi

    (
        flock 200
        date +%s > "$state"
        "${compile[@]}"
    ) 200>"$lock"
done
