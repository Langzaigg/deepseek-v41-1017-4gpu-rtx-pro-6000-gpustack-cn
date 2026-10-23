#!/usr/bin/env bash
set -euo pipefail
export PYTHONUTF8=1 PYTHONUNBUFFERED=1
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
exec python3 scripts/standalone.py "$@"
