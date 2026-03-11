# MoltenVK Barrier Repro

This repository is a purpose-built Vulkan compute repro for a suspected MoltenVK/macOS shared-memory synchronization bug. It runs the same workgroup-local algorithm in two GLSL compute shader variants:

- `barrier_only`: uses `barrier()` at each shared-memory sync point
- `memorybarrier_plus_barrier`: uses `memoryBarrier(); barrier()` at the same sync points

The host computes an exact CPU reference and compares every output word. The intended bug signature is:

- `barrier_only` fails
- `memorybarrier_plus_barrier` passes

If that happens on MoltenVK/macOS, the repo provides a compact issue attachment set: shader sources, SPIR-V binaries, SPIR-V disassembly, deterministic host verification, and concise device/driver metadata.

## What This Repo Demonstrates

The test stresses `shared` memory visibility inside a 64-thread workgroup. Each lane writes eight values into a padded shared-memory layout, synchronizes, reads back a transposed cross-lane pattern, folds the reads into a checksum, and repeats for many rounds. The only intentional difference between the two primary shader variants is whether the shared-memory sync point is:

- `barrier()`
- `memoryBarrier(); barrier()`

By GLSL/Vulkan compute semantics, `barrier()` should already be sufficient for workgroup-local synchronization and visibility of `shared` memory accesses. Requiring the stronger sequence would suggest a backend or translation issue rather than expected shader semantics.

## Why MoltenVK Is Relevant

On macOS, Vulkan applications typically run through MoltenVK, the Vulkan-on-Metal portability layer from Khronos. The LunarG macOS Vulkan SDK ships the Vulkan loader and MoltenVK integration used by most macOS Vulkan setups.

Authoritative upstream references:

- [MoltenVK repository](https://github.com/KhronosGroup/MoltenVK)
- [MoltenVK runtime user guide](https://github.com/KhronosGroup/MoltenVK/blob/main/Docs/MoltenVK_Runtime_UserGuide.md)
- [LunarG macOS Vulkan SDK getting started](https://vulkan.lunarg.com/doc/view/latest/mac/getting_started.html)

## Dependencies

Required:

- macOS
- Xcode command line tools
- LunarG Vulkan SDK for macOS
- CMake
- a C++ compiler compatible with the installed Xcode toolchain

Optional but useful:

- `spirv-dis`
- `spirv-val`
- Vulkan validation layers from the SDK
- MoltenVK debug or logging environment variables if you want extra runtime diagnostics

## Setup

1. Install the current LunarG macOS Vulkan SDK from the official LunarG site.
2. Open a shell with the SDK environment configured.

Example:

```bash
export VULKAN_SDK="/path/to/vulkansdk/macOS"
source "$VULKAN_SDK/setup-env.sh"
```

3. Build the repro.

```bash
./scripts/build_macos.sh
```

That configures CMake, builds the executable, and compiles the GLSL shaders into `build/shaders/`.

## Running The Repro

Run both primary variants back to back:

```bash
./scripts/run_repro.sh
```

Run a single variant directly:

```bash
./build/moltenvk_barrier_repro --shader barrier_only --workgroups 8192 --rounds 256 --repetitions 50
./build/moltenvk_barrier_repro --shader memorybarrier_plus_barrier --workgroups 8192 --rounds 256 --repetitions 50
```

Useful command-line flags:

- `--list-devices`
- `--device <index>`
- `--shader barrier_only|memorybarrier_plus_barrier|groupmemorybarrier_plus_barrier`
- `--workgroups N`
- `--rounds N`
- `--repetitions N`
- `--verbose`
- `--dump-spirv-info`
- `--first-mismatch-limit N`
- `--shader-dir <path>`

The application exits with:

- `0` when the selected variant matches the CPU reference
- `1` when mismatches are found
- `2` for setup or runtime errors

The `run_repro.sh` wrapper exits differently on purpose:

- `1` when the repro signature is observed (`barrier_only` fails, workaround passes)
- `0` when both pass and the bug did not reproduce
- `2` for unexpected or setup failures

## Interpreting Results

- If both variants pass, the bug did not reproduce in that run.
- If `barrier_only` fails and `memorybarrier_plus_barrier` passes, the repro succeeded.
- If both fail, there is likely a broader bug, build issue, or environment problem unrelated to the narrow barrier difference.

Example output:

```text
Device: Apple M3 Max via MoltenVK
Shader variant: barrier_only
Workgroups: 8192
Shader rounds: 256
Host repetitions: 50
Mismatches: 137
Status: FAIL
```

## Design Note

This shader pattern is intentionally close to a reduced FFT-style reorder/transposition workload:

- each lane owns eight values
- writes land in a padded shared-memory layout using `idx = raw + (raw >> 4)`
- reads come from a transposed cross-lane address pattern
- repeated rounds force the backend to preserve visibility across many shared-memory reuse cycles

That structure tends to expose weak or incorrectly lowered workgroup-memory barriers more reliably than a trivial single-write/single-read example while staying compact enough for maintainers to inspect quickly.

## Capturing Extra Diagnostics

Print built-in SPIR-V barrier metadata:

```bash
./build/moltenvk_barrier_repro --shader barrier_only --dump-spirv-info
```

Disassemble generated SPIR-V:

```bash
./scripts/dump_spirv.sh
```

Compile shaders without a full rebuild:

```bash
./scripts/compile_shaders.sh
```

Useful extra diagnostics:

- enable Vulkan validation layers from the LunarG SDK
- enable MoltenVK runtime logging/debug environment variables if available in your SDK/runtime
- capture generated MSL or a Metal compute trace when testing a local MoltenVK fix

This repo does not automate MSL dumping because the exact workflow depends on the SDK/runtime setup. If you are testing MoltenVK from source, point the Vulkan loader at your custom runtime and collect the translated MSL or a Metal GPU capture alongside the repro output.

## Using A Custom MoltenVK Build

The default path is to use the MoltenVK runtime bundled by the LunarG SDK. If you are validating a local MoltenVK fix, keep the repo unchanged and override the loader/runtime using your usual macOS Vulkan environment variables or loader configuration so the same repro can run against both stock and patched MoltenVK builds.

## Bug Report Bundle

When filing an issue, attach:

- macOS version
- Apple GPU / machine model
- Xcode version
- Vulkan SDK version
- MoltenVK version
- command lines used
- output from both shader variants
- generated `.spv` files
- SPIR-V disassembly from `build/spirv/`
- generated MSL or a Metal trace if you captured one

The file at `docs/issue_template.md` is ready to paste into a GitHub issue.
