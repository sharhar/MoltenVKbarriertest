#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

xcrun -sdk macosx metal -c barrier_test.metal -o barrier_test.air
xcrun -sdk macosx metallib barrier_test.air -o default.metallib
swiftc main.swift -o barrier_test.exec -framework Metal -framework Foundation

echo "Build complete. Run with: ./barrier_test.exec"
