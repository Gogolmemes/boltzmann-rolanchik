#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"

cd "$ROOT"

python -m PyInstaller \
    --noconfirm \
    --clean \
    --workpath "build/linux" \
    --distpath "releases/linux" \
    "packaging/boltzmann.spec"

echo "Linux build completed."