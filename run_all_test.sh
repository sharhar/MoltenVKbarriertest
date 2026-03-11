#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

run_target() {
    local target_dir="$1"

    echo "=== Running ${target_dir} ==="
    (
        cd "${SCRIPT_DIR}/${target_dir}"
        ./build.sh
        ./barrier_test.exec
    )
    echo
}

run_target "vulkan"
run_target "metal"
