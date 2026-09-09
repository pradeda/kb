#!/usr/bin/env bash
# Compatibility entry point. Production logic is owned by kb-go and installed
# at /opt/kb/refresh_volatile.sh; do not duplicate it here.
set -euo pipefail
exec /opt/kb/refresh_volatile.sh "$@"
