#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${1:-"$ROOT_DIR/build"}"
SHADER_DIR="$ROOT_DIR/shaders"
OUT_DIR="$BUILD_DIR/shaders"

mkdir -p "$OUT_DIR"

GLSLANG="${GLSLANG_VALIDATOR:-$(command -v glslangValidator || true)}"
if [[ -z "$GLSLANG" ]]; then
  echo "glslangValidator was not found on PATH." >&2
  exit 2
fi

for shader in barrier_only.comp memorybarrier_plus_barrier.comp groupmemorybarrier_plus_barrier.comp; do
  "$GLSLANG" -V --target-env vulkan1.1 "-I$SHADER_DIR" -o "$OUT_DIR/$shader.spv" "$SHADER_DIR/$shader"
done

echo "Compiled shaders into $OUT_DIR"
