#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${1:-"$ROOT_DIR/build"}"

if [[ -z "${VULKAN_SDK:-}" ]]; then
  echo "VULKAN_SDK is not set. Install the LunarG macOS Vulkan SDK and source setup-env.sh first." >&2
  exit 2
fi

command -v cmake >/dev/null 2>&1 || {
  echo "cmake was not found on PATH." >&2
  exit 2
}

command -v glslangValidator >/dev/null 2>&1 || {
  echo "glslangValidator was not found on PATH." >&2
  exit 2
}

cmake -S "$ROOT_DIR" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release
cmake --build "$BUILD_DIR" --target repro_shaders moltenvk_barrier_repro -j

echo
echo "Build complete:"
echo "  $BUILD_DIR/moltenvk_barrier_repro"
