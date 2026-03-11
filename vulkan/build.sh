#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
TOOLCHAIN_ENV="${ROOT_DIR}/build/vulkan-toolchain.env"
cd "$SCRIPT_DIR"

if [[ -f "${TOOLCHAIN_ENV}" ]]; then
    # Load the managed Vulkan toolchain selection when present.
    source "${TOOLCHAIN_ENV}"
fi

GLSLANG_VALIDATOR_BIN="${VULKAN_GLSLANG_VALIDATOR:-glslangValidator}"

mkdir -p shader_dump

"${GLSLANG_VALIDATOR_BIN}" -V -S comp -DBARRIER_MODE=0 barrier_test.comp -o barrier_only.spv
"${GLSLANG_VALIDATOR_BIN}" -V -S comp -DBARRIER_MODE=1 barrier_test.comp -o memory_barrier_then_barrier.spv

read -r -a VULKAN_CFLAGS <<<"$(pkg-config --cflags vulkan)"
read -r -a VULKAN_LIBS <<<"$(pkg-config --libs vulkan)"

clang++ -std=c++17 -O2 -Wall -Wextra -pedantic \
    main.cpp \
    -o barrier_test.exec \
    "${VULKAN_CFLAGS[@]}" \
    "${VULKAN_LIBS[@]}"

echo "Build complete."
echo "GLSL compiler: ${GLSLANG_VALIDATOR_BIN}"
echo "Generate blobs with: python3 generate_blobs.py --output-dir data"
echo "Run with: ./run.sh"
