#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"

cd "$ROOT"

python -m PyInstaller \
    --noconfirm \
    --clean \
    --workpath "build/macos" \
    --distpath "releases/macos" \
    "packaging/boltzmann.spec"

echo "macOS build completed."