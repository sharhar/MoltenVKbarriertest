#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${1:-"$ROOT_DIR/build"}"
SHADER_DIR="$BUILD_DIR/shaders"
OUT_DIR="$BUILD_DIR/spirv"

SPIRV_DIS="${SPIRV_DIS:-$(command -v spirv-dis || true)}"
if [[ -z "$SPIRV_DIS" ]]; then
  echo "spirv-dis was not found on PATH." >&2
  exit 2
fi

mkdir -p "$OUT_DIR"

for spv in "$SHADER_DIR"/*.spv; do
  base="$(basename "$spv")"
  "$SPIRV_DIS" "$spv" -o "$OUT_DIR/$base.spvasm"
done

echo "Wrote disassembly into $OUT_DIR"
