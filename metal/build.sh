#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

xcrun -sdk macosx metal -fmodules-cache-path=.clang-module-cache -c barrier_test.metal -o barrier_test.air
xcrun -sdk macosx metallib barrier_test.air -o default.metallib
swiftc main.swift -module-cache-path ./.swift-module-cache -o barrier_test.exec -framework Metal -framework Foundation

echo "Build complete."
echo "Generate blobs with: python3 generate_blobs.py --output-dir data"
echo "Run with: ./barrier_test.exec"
