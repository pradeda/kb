#!/bin/bash
# /opt/kb/compile.sh — wrapper za crontab
set -a
source /opt/kb/.env
set +a
/opt/kb/venv-embed/bin/python /opt/kb/compile.py
