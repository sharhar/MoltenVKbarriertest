#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
BUILD_DIR="${ROOT_DIR}/build"
TOOLCHAIN_DIR="${BUILD_DIR}/toolchains"
REPOS_DIR="${TOOLCHAIN_DIR}/repos"
WORKTREES_DIR="${TOOLCHAIN_DIR}/worktrees"
PAIRS_DIR="${TOOLCHAIN_DIR}/pairs"
STATE_ENV="${BUILD_DIR}/vulkan-toolchain.env"
STATE_JSON="${BUILD_DIR}/vulkan-toolchain.json"

MOLTENVK_REMOTE_URL="https://github.com/KhronosGroup/MoltenVK.git"
GLSLANG_REMOTE_URL="https://github.com/KhronosGroup/glslang.git"

usage() {
    cat <<'EOF'
Usage:
  ./select_toolchain.sh --moltenvk-ref <git-ref> [--glslang-ref <git-ref>] [--rebuild]
  ./select_toolchain.sh --show
  ./select_toolchain.sh --clear

Behavior:
  - Caches git clones and build outputs under ../build/toolchains/
  - Defaults glslang to the revision pinned by the selected MoltenVK commit
  - Persists the active selection in ../build/vulkan-toolchain.env and .json
EOF
}

fail() {
    echo "ERROR: $*" >&2
    exit 1
}

require_command() {
    local command_name="$1"
    command -v "${command_name}" >/dev/null 2>&1 || fail "Required command not found: ${command_name}"
}

ensure_dir() {
    mkdir -p "$1"
}

ensure_clone() {
    local name="$1"
    local remote_url="$2"
    local repo_dir="${REPOS_DIR}/${name}"

    ensure_dir "${REPOS_DIR}"

    if [[ ! -d "${repo_dir}/.git" ]]; then
        git clone "${remote_url}" "${repo_dir}"
    fi

    git -C "${repo_dir}" remote set-url origin "${remote_url}"

    echo "${repo_dir}"
}

resolve_commit() {
    local repo_dir="$1"
    local ref="$2"

    if git -C "${repo_dir}" rev-parse --verify "${ref}^{commit}" >/dev/null 2>&1; then
        git -C "${repo_dir}" rev-parse --verify "${ref}^{commit}"
        return 0
    fi

    git -C "${repo_dir}" fetch --tags --prune origin
    git -C "${repo_dir}" rev-parse --verify "${ref}^{commit}"
}

ensure_worktree() {
    local repo_dir="$1"
    local commit="$2"
    local worktree_dir="$3"

    ensure_dir "$(dirname "${worktree_dir}")"

    if [[ ! -e "${worktree_dir}/.git" ]]; then
        git -C "${repo_dir}" worktree add --detach "${worktree_dir}" "${commit}"
    fi
}

cmake_generator_args() {
    if command -v ninja >/dev/null 2>&1; then
        printf '%s\n' "-G" "Ninja"
    fi
}

find_glslang_validator() {
    local search_dir="$1"
    local candidate

    while IFS= read -r candidate; do
        if [[ -x "${candidate}" ]]; then
            echo "${candidate}"
            return 0
        fi
    done < <(find "${search_dir}" -type f -name glslangValidator | sort)

    return 1
}

build_glslang_validator() {
    local glslang_worktree="$1"
    local glslang_commit="$2"
    local pair_dir="$3"
    local output_bin="${pair_dir}/tools/glslangValidator"
    local build_dir="${glslang_worktree}/codex-build-${glslang_commit:0:12}"
    local generator_args=()
    local built_bin=""

    if [[ -x "${output_bin}" ]]; then
        echo "${output_bin}"
        return 0
    fi

    ensure_dir "${pair_dir}/tools"
    mapfile -t generator_args < <(cmake_generator_args)

    cmake -S "${glslang_worktree}" -B "${build_dir}" "${generator_args[@]}" -DCMAKE_BUILD_TYPE=Release
    cmake --build "${build_dir}" --target glslangValidator

    built_bin="$(find_glslang_validator "${build_dir}")" || fail "Failed to locate built glslangValidator under ${build_dir}"
    cp -f "${built_bin}" "${output_bin}"
    chmod +x "${output_bin}"

    echo "${output_bin}"
}

build_moltenvk() {
    local moltenvk_worktree="$1"
    local glslang_worktree="$2"
    local pair_dir="$3"
    local output_dylib="${pair_dir}/runtime/libMoltenVK.dylib"
    local output_icd="${pair_dir}/runtime/MoltenVK_icd.json"
    local source_dylib="${moltenvk_worktree}/Package/Latest/MoltenVK/dynamic/dylib/macOS/libMoltenVK.dylib"
    local quoted_dylib_path=""

    if [[ -f "${output_dylib}" && -f "${output_icd}" ]]; then
        printf '%s\n%s\n' "${output_dylib}" "${output_icd}"
        return 0
    fi

    ensure_dir "${pair_dir}/runtime"

    (
        cd "${moltenvk_worktree}"
        ./fetchDependencies --macos --glslang-root "${glslang_worktree}"
        make macos
    )

    [[ -f "${source_dylib}" ]] || fail "MoltenVK build finished but ${source_dylib} was not produced"

    cp -f "${source_dylib}" "${output_dylib}"
    chmod +x "${output_dylib}"

    quoted_dylib_path="$(printf '%s' "${output_dylib}" | sed 's/\\/\\\\/g; s/"/\\"/g')"
    cat > "${output_icd}" <<EOF
{
  "file_format_version": "1.0.0",
  "ICD": {
    "library_path": "${quoted_dylib_path}",
    "api_version": "1.2.0",
    "is_portability_driver": true
  }
}
EOF

    printf '%s\n%s\n' "${output_dylib}" "${output_icd}"
}

write_state_files() {
    local moltenvk_ref="$1"
    local moltenvk_commit="$2"
    local moltenvk_dylib="$3"
    local moltenvk_icd_json="$4"
    local glslang_ref="$5"
    local glslang_commit="$6"
    local glslang_validator="$7"
    local pair_dir="$8"
    local escaped_pair_dir=""
    local escaped_mvk_ref=""
    local escaped_mvk_commit=""
    local escaped_mvk_dylib=""
    local escaped_mvk_icd=""
    local escaped_gl_ref=""
    local escaped_gl_commit=""
    local escaped_gl_validator=""

    ensure_dir "${BUILD_DIR}"

    cat > "${STATE_ENV}" <<EOF
export VULKAN_MOLTENVK_REF=$(printf '%q' "${moltenvk_ref}")
export VULKAN_MOLTENVK_COMMIT=$(printf '%q' "${moltenvk_commit}")
export VULKAN_MOLTENVK_DYLIB=$(printf '%q' "${moltenvk_dylib}")
export VULKAN_MOLTENVK_ICD_JSON=$(printf '%q' "${moltenvk_icd_json}")
export VULKAN_GLSLANG_REF=$(printf '%q' "${glslang_ref}")
export VULKAN_GLSLANG_COMMIT=$(printf '%q' "${glslang_commit}")
export VULKAN_GLSLANG_VALIDATOR=$(printf '%q' "${glslang_validator}")
EOF

    escaped_pair_dir="$(printf '%s' "${pair_dir}" | sed 's/\\/\\\\/g; s/"/\\"/g')"
    escaped_mvk_ref="$(printf '%s' "${moltenvk_ref}" | sed 's/\\/\\\\/g; s/"/\\"/g')"
    escaped_mvk_commit="$(printf '%s' "${moltenvk_commit}" | sed 's/\\/\\\\/g; s/"/\\"/g')"
    escaped_mvk_dylib="$(printf '%s' "${moltenvk_dylib}" | sed 's/\\/\\\\/g; s/"/\\"/g')"
    escaped_mvk_icd="$(printf '%s' "${moltenvk_icd_json}" | sed 's/\\/\\\\/g; s/"/\\"/g')"
    escaped_gl_ref="$(printf '%s' "${glslang_ref}" | sed 's/\\/\\\\/g; s/"/\\"/g')"
    escaped_gl_commit="$(printf '%s' "${glslang_commit}" | sed 's/\\/\\\\/g; s/"/\\"/g')"
    escaped_gl_validator="$(printf '%s' "${glslang_validator}" | sed 's/\\/\\\\/g; s/"/\\"/g')"

    cat > "${STATE_JSON}" <<EOF
{
  "moltenvk": {
    "remote_url": "${MOLTENVK_REMOTE_URL}",
    "ref": "${escaped_mvk_ref}",
    "commit": "${escaped_mvk_commit}",
    "dylib": "${escaped_mvk_dylib}",
    "icd_json": "${escaped_mvk_icd}"
  },
  "glslang": {
    "remote_url": "${GLSLANG_REMOTE_URL}",
    "ref": "${escaped_gl_ref}",
    "commit": "${escaped_gl_commit}",
    "validator": "${escaped_gl_validator}"
  },
  "pair_build_root": "${escaped_pair_dir}"
}
EOF
}

show_state() {
    if [[ ! -f "${STATE_ENV}" ]]; then
        echo "No managed Vulkan toolchain is currently selected."
        return 0
    fi

    source "${STATE_ENV}"
    echo "MoltenVK ref: ${VULKAN_MOLTENVK_REF}"
    echo "MoltenVK commit: ${VULKAN_MOLTENVK_COMMIT}"
    echo "MoltenVK dylib: ${VULKAN_MOLTENVK_DYLIB}"
    echo "MoltenVK ICD: ${VULKAN_MOLTENVK_ICD_JSON}"
    echo "glslang ref: ${VULKAN_GLSLANG_REF}"
    echo "glslang commit: ${VULKAN_GLSLANG_COMMIT}"
    echo "glslangValidator: ${VULKAN_GLSLANG_VALIDATOR}"
}

clear_state() {
    rm -f "${STATE_ENV}" "${STATE_JSON}"
    echo "Cleared managed Vulkan toolchain selection."
}

MODE="select"
MOLTENVK_REF=""
GLSLANG_REF=""
REBUILD=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --moltenvk-ref)
            [[ $# -ge 2 ]] || fail "Missing value for --moltenvk-ref"
            MOLTENVK_REF="$2"
            shift 2
            ;;
        --glslang-ref)
            [[ $# -ge 2 ]] || fail "Missing value for --glslang-ref"
            GLSLANG_REF="$2"
            shift 2
            ;;
        --rebuild)
            REBUILD=1
            shift
            ;;
        --show)
            MODE="show"
            shift
            ;;
        --clear)
            MODE="clear"
            shift
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            fail "Unknown argument: $1"
            ;;
    esac
done

case "${MODE}" in
    show)
        show_state
        exit 0
        ;;
    clear)
        clear_state
        exit 0
        ;;
esac

require_command git
require_command cmake
require_command make
require_command xcodebuild

[[ -n "${MOLTENVK_REF}" ]] || fail "You must provide --moltenvk-ref"

ensure_dir "${WORKTREES_DIR}"
ensure_dir "${PAIRS_DIR}"

MOLTENVK_REPO="$(ensure_clone MoltenVK "${MOLTENVK_REMOTE_URL}")"
GLSLANG_REPO="$(ensure_clone glslang "${GLSLANG_REMOTE_URL}")"

MOLTENVK_COMMIT="$(resolve_commit "${MOLTENVK_REPO}" "${MOLTENVK_REF}")"

if [[ -z "${GLSLANG_REF}" ]]; then
    GLSLANG_REF="$(git -C "${MOLTENVK_REPO}" show "${MOLTENVK_COMMIT}:ExternalRevisions/glslang_repo_revision" | tr -d '[:space:]')"
fi

GLSLANG_COMMIT="$(resolve_commit "${GLSLANG_REPO}" "${GLSLANG_REF}")"

PAIR_KEY="moltenvk-${MOLTENVK_COMMIT:0:12}__glslang-${GLSLANG_COMMIT:0:12}"
PAIR_DIR="${PAIRS_DIR}/${PAIR_KEY}"
MOLTENVK_WORKTREE="${WORKTREES_DIR}/${PAIR_KEY}/MoltenVK"
GLSLANG_WORKTREE="${WORKTREES_DIR}/glslang-${GLSLANG_COMMIT:0:12}"
GLSLANG_BUILD_DIR="${GLSLANG_WORKTREE}/codex-build-${GLSLANG_COMMIT:0:12}"

if [[ "${REBUILD}" -eq 1 ]]; then
    if [[ -e "${MOLTENVK_WORKTREE}/.git" ]]; then
        git -C "${MOLTENVK_REPO}" worktree remove --force "${MOLTENVK_WORKTREE}" || true
    fi
    rm -rf "${PAIR_DIR}"
    rm -rf "${GLSLANG_BUILD_DIR}"
fi

ensure_worktree "${GLSLANG_REPO}" "${GLSLANG_COMMIT}" "${GLSLANG_WORKTREE}"
ensure_worktree "${MOLTENVK_REPO}" "${MOLTENVK_COMMIT}" "${MOLTENVK_WORKTREE}"

GLSLANG_VALIDATOR_PATH="$(build_glslang_validator "${GLSLANG_WORKTREE}" "${GLSLANG_COMMIT}" "${PAIR_DIR}")"
mapfile -t moltenvk_outputs < <(build_moltenvk "${MOLTENVK_WORKTREE}" "${GLSLANG_WORKTREE}" "${PAIR_DIR}")
MOLTENVK_DYLIB_PATH="${moltenvk_outputs[0]}"
MOLTENVK_ICD_JSON_PATH="${moltenvk_outputs[1]}"

write_state_files \
    "${MOLTENVK_REF}" \
    "${MOLTENVK_COMMIT}" \
    "${MOLTENVK_DYLIB_PATH}" \
    "${MOLTENVK_ICD_JSON_PATH}" \
    "${GLSLANG_REF}" \
    "${GLSLANG_COMMIT}" \
    "${GLSLANG_VALIDATOR_PATH}" \
    "${PAIR_DIR}"

echo "Selected MoltenVK ref: ${MOLTENVK_REF}"
echo "Resolved MoltenVK commit: ${MOLTENVK_COMMIT}"
echo "Selected glslang ref: ${GLSLANG_REF}"
echo "Resolved glslang commit: ${GLSLANG_COMMIT}"
echo "Managed glslangValidator: ${GLSLANG_VALIDATOR_PATH}"
echo "Managed MoltenVK ICD: ${MOLTENVK_ICD_JSON_PATH}"
