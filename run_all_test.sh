#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

run_target() {
    local target_dir="$1"

    echo "=== Running ${target_dir} ==="
    (
        cd "${SCRIPT_DIR}/${target_dir}"
        ./build.sh
        if [[ "${target_dir}" == "vulkan" ]]; then
            ./run.sh
        else
            ./barrier_test.exec
        fi
    )
    echo
}

run_target "vulkan"
run_target "metal"
