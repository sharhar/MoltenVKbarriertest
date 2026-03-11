#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
TOOLCHAIN_ENV="${ROOT_DIR}/build/vulkan-toolchain.env"

cd "${SCRIPT_DIR}"

if [[ -f "${TOOLCHAIN_ENV}" ]]; then
    source "${TOOLCHAIN_ENV}"
fi

if [[ -n "${VULKAN_MOLTENVK_ICD_JSON:-}" ]]; then
    export VK_DRIVER_FILES="${VULKAN_MOLTENVK_ICD_JSON}"
    export VK_ICD_FILENAMES="${VULKAN_MOLTENVK_ICD_JSON}"
fi

exec ./barrier_test.exec "$@"
