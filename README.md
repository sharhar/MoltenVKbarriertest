# MoltenVK Barrier Repro

This repository is a minimal Vulkan compute repro for a suspected MoltenVK/macOS shared-memory synchronization bug in a length-125 FFT kernel. It runs the same FFT shader in two variants:

- `barrier_only`: uses `barrier()` at every shared-memory synchronization point
- `memorybarrier_plus_barrier`: uses `memoryBarrier(); barrier()` at those same points

The host loads a fixed input blob, dispatches 875 FFTs of length 125, and compares the in-place output buffer against a reference output blob. The intended bug signature is:

- `barrier_only` fails
- `memorybarrier_plus_barrier` passes

## What This Repo Demonstrates

The shader performs 875 independent FFTs of length 125 over a contiguous axis. Each workgroup handles one FFT, with `local_size_x = 25` and five complex values per invocation. Two stages require shared-memory shuffles through `shared vec2 sdata[125]`, and the only intentional difference between the two primary variants is whether the shared-memory sync sites use:

- `barrier()`
- `memoryBarrier(); barrier()`

If `barrier_only` corrupts the FFT result on MoltenVK while the workaround passes against the same reference blob, that is a strong signal that shared-memory synchronization is being lowered incorrectly in the Vulkan-to-Metal path.

## Why MoltenVK Is Relevant

On macOS, Vulkan applications normally run through MoltenVK, the Vulkan-on-Metal portability layer from Khronos. The LunarG macOS Vulkan SDK ships the Vulkan loader and MoltenVK integration used by most macOS Vulkan setups.

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

## Blob Format

The repro uses raw binary blobs instead of generating inputs or shipping a CPU FFT implementation.

- shape: `(875, 125)`
- logical meaning: `875` FFT batches, each of length `125`
- storage: row-major contiguous complex array
- element format: two little-endian `float32` values per complex number, stored as `(real, imag)`
- total element count: `109375`
- total file size: `875000` bytes

Default filenames:

- `data/fft_875x125_input.bin`
- `data/fft_875x125_reference.bin`

## Setup

1. Install the current LunarG macOS Vulkan SDK from the official LunarG site.
2. Open a shell with the SDK environment configured.

Example:

```bash
export VULKAN_SDK="/path/to/vulkansdk/macOS"
source "$VULKAN_SDK/setup-env.sh"
```

3. Place the input and reference blobs under `data/` or pass their paths explicitly at runtime.
   To generate them with NumPy:

```bash
python3 scripts/generate_fft_blobs.py
```

4. Build the repro.

```bash
./scripts/build_macos.sh
```

That configures CMake, builds the executable, and compiles the GLSL shaders into `build/shaders/`.

## Running The Repro

Run both primary variants back to back:

```bash
./scripts/run_repro.sh
```

Or override the blobs explicitly:

```bash
INPUT_BLOB=/path/to/input.bin REFERENCE_BLOB=/path/to/reference.bin ./scripts/run_repro.sh
```

Run a single variant directly:

```bash
./build/moltenvk_barrier_repro \
  --shader barrier_only \
  --input data/fft_875x125_input.bin \
  --reference data/fft_875x125_reference.bin \
  --repetitions 50
```

Useful command-line flags:

- `--list-devices`
- `--device <index>`
- `--shader barrier_only|memorybarrier_plus_barrier|groupmemorybarrier_plus_barrier`
- `--input <path>`
- `--reference <path>`
- `--repetitions N`
- `--abs-tolerance X`
- `--rel-tolerance X`
- `--verbose`
- `--dump-spirv-info`
- `--first-mismatch-limit N`
- `--shader-dir <path>`

The application exits with:

- `0` when the selected variant matches the reference blob within tolerance
- `1` when mismatches are found
- `2` for setup or runtime errors

The `run_repro.sh` wrapper exits differently on purpose:

- `1` when the repro signature is observed (`barrier_only` fails, workaround passes)
- `0` when both pass and the bug did not reproduce
- `2` for unexpected or setup failures

## Interpreting Results

- If both variants pass, the bug did not reproduce in that run.
- If `barrier_only` fails and `memorybarrier_plus_barrier` passes, the repro succeeded.
- If both fail, there is likely a broader issue unrelated to the narrow barrier difference.

Each mismatch line prints the flattened index, FFT batch, element index inside the 125-point FFT, repetition number, expected complex value, actual complex value, and maximum component error.

## Design Note

This shader is a reduced real workload rather than a synthetic checksum kernel. It uses the exact shared-memory shuffle pattern from the failing FFT implementation:

- stage-local register butterflies
- shared-memory transposes through `sdata[125]`
- in-place global buffer IO
- three workgroup barriers in the barrier-only variant

That makes the repro much closer to the failure mode you observed while keeping the host side small and auditable.

## Capturing Extra Diagnostics

Print built-in SPIR-V barrier metadata:

```bash
./build/moltenvk_barrier_repro \
  --shader barrier_only \
  --input data/fft_875x125_input.bin \
  --reference data/fft_875x125_reference.bin \
  --dump-spirv-info
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
- enable MoltenVK runtime logging/debug environment variables if available
- capture generated MSL or a Metal compute trace when testing a local MoltenVK fix

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
- input blob and reference blob metadata
- generated `.spv` files
- SPIR-V disassembly from `build/spirv/`
- generated MSL or a Metal trace if you captured one

The file at `docs/issue_template.md` is ready to paste into a GitHub issue.
