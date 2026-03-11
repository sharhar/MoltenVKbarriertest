#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

swiftc main.swift -module-cache-path ./.swift-module-cache -o barrier_test.exec -framework Metal -framework Foundation

echo "Build complete."
echo "Run with: ./barrier_test.exec"
