#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${1:-"$ROOT_DIR/build"}"
EXE="$BUILD_DIR/moltenvk_barrier_repro"
INPUT_BLOB="${INPUT_BLOB:-$ROOT_DIR/data/fft_875x125_input.bin}"
REFERENCE_BLOB="${REFERENCE_BLOB:-$ROOT_DIR/data/fft_875x125_reference.bin}"

if [[ ! -x "$EXE" ]]; then
  echo "Executable not found: $EXE" >&2
  echo "Run scripts/build_macos.sh first." >&2
  exit 2
fi

if [[ ! -f "$INPUT_BLOB" ]]; then
  echo "Input blob not found: $INPUT_BLOB" >&2
  exit 2
fi

if [[ ! -f "$REFERENCE_BLOB" ]]; then
  echo "Reference blob not found: $REFERENCE_BLOB" >&2
  exit 2
fi

COMMON_ARGS=(
  --input "$INPUT_BLOB"
  --reference "$REFERENCE_BLOB"
  --repetitions "${REPETITIONS:-50}"
  --abs-tolerance "${ABS_TOLERANCE:-1e-5}"
  --rel-tolerance "${REL_TOLERANCE:-1e-5}"
  --first-mismatch-limit "${FIRST_MISMATCH_LIMIT:-8}"
)

set +e
echo "===== barrier_only ====="
"$EXE" --shader barrier_only "${COMMON_ARGS[@]}"
baseline_status=$?

echo
echo "===== memorybarrier_plus_barrier ====="
"$EXE" --shader memorybarrier_plus_barrier "${COMMON_ARGS[@]}"
workaround_status=$?
set -e

echo
if [[ $baseline_status -eq 1 && $workaround_status -eq 0 ]]; then
  echo "Observed the expected repro signature: barrier_only failed and memorybarrier_plus_barrier passed."
  exit 1
fi

if [[ $baseline_status -eq 0 && $workaround_status -eq 0 ]]; then
  echo "Both shader variants passed. The bug did not reproduce in this run."
  exit 0
fi

echo "Unexpected outcome. baseline_status=$baseline_status workaround_status=$workaround_status"
exit 2
