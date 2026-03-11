#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

mkdir -p shader_dump

glslangValidator -V -S comp -DBARRIER_MODE=0 barrier_test.comp -o barrier_only.spv
glslangValidator -V -S comp -DBARRIER_MODE=1 barrier_test.comp -o memory_barrier_then_barrier.spv

read -r -a VULKAN_CFLAGS <<<"$(pkg-config --cflags vulkan)"
read -r -a VULKAN_LIBS <<<"$(pkg-config --libs vulkan)"

clang++ -std=c++17 -O2 -Wall -Wextra -pedantic \
    main.cpp \
    -o barrier_test.exec \
    "${VULKAN_CFLAGS[@]}" \
    "${VULKAN_LIBS[@]}"

echo "Build complete."
echo "Generate blobs with: python3 generate_blobs.py --output-dir data"
echo "Run with: ./barrier_test.exec"
