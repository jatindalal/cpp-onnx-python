#!/usr/bin/env bash
#
# Build manylinux wheels with the running Docker engine (Colima on macOS).
# Pass a comma-separated arch list to override the default.
#
#   scripts/linux_wheels.sh                # aarch64 + x86_64
#   scripts/linux_wheels.sh aarch64        # only the native arch
#
# aarch64 builds natively; x86_64 builds through emulation and is much slower.
# Output lands in dist/ next to the macOS wheel from scripts/release.sh.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

ARCHS="${1:-aarch64,x86_64}"
PYTHON="${PYTHON:-python3}"

command -v docker >/dev/null || { echo "docker not found on PATH"; exit 1; }
docker info >/dev/null 2>&1 || { echo "Docker engine not reachable (try 'colima start')"; exit 1; }

mkdir -p dist
"$PYTHON" -m cibuildwheel --platform linux --archs "$ARCHS" --output-dir dist .
ls -la dist
