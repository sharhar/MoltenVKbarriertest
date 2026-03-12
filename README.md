# MoltenVK FFT Barrier Repro

This repository reproduces a synchronization-related correctness bug seen only through the Vulkan-on-macOS path.

The workload is a batch FFT of length 125 over contiguous `complex64` data:

- 875 independent FFTs
- FFT length 125
- 25 threads per workgroup
- shared/threadgroup memory used for the two internal data reorders between radix-5 stages

## Current conclusion

The bug is not reproduced by native Metal.

It is reproduced by Vulkan through MoltenVK when the GLSL shader uses `barrier()` alone, and it disappears when the GLSL shader uses `memoryBarrier(); barrier()`.

The strongest narrowing result in this repo is:

1. Native Metal with the handwritten shader passes for all barrier flag variants.
2. Vulkan through MoltenVK fails with `barrier()` but passes with `memoryBarrier(); barrier()`.
3. Native Metal execution of the exact dumped MSL shaders emitted by MoltenVK also passes.

That means the bug is very likely not in the FFT algorithm itself, and not in the visible generated MSL source by itself. The failure appears to exist specifically in the Vulkan-to-Metal execution path, making MoltenVK the most likely place to file the bug first.

## Why this repo exists

At first glance, the Vulkan-generated MSL dumps look like they should explain the behavior:

- the `barrier()`-only Vulkan shader lowers to `threadgroup_barrier(mem_flags::mem_threadgroup)`
- the `memoryBarrier(); barrier()` Vulkan shader lowers to `threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup | mem_flags::mem_texture)`

However, compiling and running those dumped `.metal` files directly through a native Metal harness does not reproduce the failure. So the problem is narrower than "the generated MSL text is wrong".

The remaining plausible fault domains are things inside the active MoltenVK Vulkan path, such as:

- internal pipeline compilation details not visible in the dumped source
- MoltenVK runtime handling of synchronization semantics
- some command encoding or execution detail specific to the Vulkan path

## Repository layout

- `data/`: shared input and reference blobs
- `generate_blobs.py`: generates the shared FFT input and NumPy reference output
- `metal/`: native Metal shader and harness
- `vulkan/`: Vulkan/MoltenVK GLSL shader, SPIR-V binaries, and Vulkan harness
- `metal_from_dumps/`: native Metal harness that loads and executes the dumped MSL shaders from `vulkan/shader_dump/`

## Shared data layout

`generate_blobs.py` writes:

- input shape `(875, 125)` of `complex64`
- reference output `np.fft.fft(input_data, axis=-1).astype(np.complex64)`
- binary blob format: interleaved little-endian `float32` real/imag pairs

Each blob is `875 * 125 * 8 = 875000` bytes.

## Harnesses

### 1. Native Metal

The Metal shader is in `metal/barrier_test.metal`.

It builds three kernels that differ only in barrier flags:

- `fft_875x125_threadgroup_only`
- `fft_875x125_device_and_threadgroup`
- `fft_875x125_all_flags`

All three kernels use the same FFT arithmetic and the same three barrier sites.

Build and run:

```bash
python3 generate_blobs.py --output-dir data
cd metal
./build.sh
./barrier_test.exec
```

Expected result on the currently investigated machine:

- all three Metal variants pass

### 2. Vulkan / MoltenVK

The Vulkan shader source is `vulkan/barrier_test.comp`.

It builds two SPIR-V variants:

- `barrier()` only
- `memoryBarrier(); barrier()`

Build and run:

```bash
python3 generate_blobs.py --output-dir data
cd vulkan
./build.sh
./run.sh
```

Expected result on the currently investigated machine:

- `barrier()` only: fails
- `memoryBarrier(); barrier()`: passes

The Vulkan harness sets `MVK_CONFIG_SHADER_DUMP_DIR` so MoltenVK emits its generated Metal shaders into `vulkan/shader_dump/`.

### Selecting MoltenVK and glslang versions

The Vulkan harness can now be pointed at a specific MoltenVK git ref and the matching or overridden glslang compiler version.

Default behavior is unchanged:

- if you do nothing, `vulkan/build.sh` uses `glslangValidator` from `PATH`
- if you do nothing, `vulkan/run.sh` uses the system Vulkan loader's normal ICD discovery

Managed toolchain workflow:

```bash
cd vulkan
./select_toolchain.sh --moltenvk-ref <tag-or-commit>
./build.sh
./run.sh
```

Explicit glslang override:

```bash
cd vulkan
./select_toolchain.sh --moltenvk-ref <tag-or-commit> --glslang-ref <tag-or-commit>
./build.sh
./run.sh
```

The selector stores the active choice in `build/vulkan-toolchain.env` and `build/vulkan-toolchain.json`.
`build.sh` uses the managed `glslangValidator` when one is selected, and `run.sh` exports `VK_DRIVER_FILES` and
`VK_ICD_FILENAMES` to point the Vulkan loader at the generated MoltenVK ICD JSON for the selected build.

### 3. Native Metal From MoltenVK Shader Dumps

This harness exists specifically to answer the question: "does the failure reproduce if the dumped MSL is compiled and run directly through Metal?"

It loads all `.metal` files from `../vulkan/shader_dump/`, compiles them at runtime, runs `main0`, and validates the output against the shared FFT reference.

Build and run:

```bash
cd metal_from_dumps
./build.sh
./barrier_test.exec
```

Expected result on the currently investigated machine:

- both dumped MSL shaders pass

## Current observed results

On the currently tested machine:

- hardware: Apple M2 Pro
- macOS: 15.7.4 (Build 24G517)

Observed behavior:

- `metal/`: all variants pass
- `vulkan/`: `barrier()` fails, `memoryBarrier(); barrier()` passes
- `metal_from_dumps/`: both dumped shaders pass

This is the core evidence that the bug only exists in the Vulkan/MoltenVK path.

## Interpreting the barrier behavior

The kernel only uses shared/threadgroup memory for inter-thread communication during the FFT reorders. It does not require cross-workgroup communication, and its final buffer writes happen after those reorder stages are complete.

Because of that, the most suspicious symptom is:

- Vulkan `barrier()` alone behaves as if the workgroup-memory synchronization is insufficient in practice
- but native Metal `threadgroup_barrier(mem_flags::mem_threadgroup)` behaves correctly for equivalent code

That mismatch is exactly why this repo is useful for a MoltenVK bug report.

## Suggested bug-report summary

If you file this with MoltenVK, the short version is:

1. A GLSL compute shader for a 125-point batched FFT fails on macOS through MoltenVK when using `barrier()` alone.
2. Adding `memoryBarrier(); barrier()` makes the Vulkan path correct.
3. Native Metal does not fail, even when compiling and executing the exact `.metal` shader dumps emitted by MoltenVK.
4. Therefore the failure seems specific to the active MoltenVK Vulkan execution path, not to the FFT algorithm and not to the visible emitted MSL source alone.

## Commands

Generate the shared blobs:

```bash
python3 generate_blobs.py --output-dir data
```

Run native Metal:

```bash
cd metal
./build.sh
./barrier_test.exec
```

Run Vulkan / MoltenVK:

```bash
cd vulkan
./build.sh
./run.sh
```

Run native Metal on dumped MSL:

```bash
cd metal_from_dumps
./build.sh
./barrier_test.exec
```


```bash

python3 generate_blobs.py --output-dir data

# Optionally select using the main branch latest commit
bash vulkan/select_toolchain.sh --moltenvk-ref f79c6c5690d3ee06ec3a00d11a8b1bab4aa1d030 --glslang-ref f0bd0257c308b9a26562c1a30c4748a0219cc951

# Or the last release version: 1.4.1
# bash vulkan/select_toolchain.sh --moltenvk-ref db445ff2042d9ce348c439ad8451112f354b8d2a --glslang-ref f0bd0257c308b9a26562c1a30c4748a0219cc951

bash run_all_test.sh

spirv-dis vulkan/shader_dump/shader-cs-c958ee748730720d.spv | grep Barrier
               OpControlBarrier %uint_2 %uint_2 %uint_264
               OpControlBarrier %uint_2 %uint_2 %uint_264
               OpControlBarrier %uint_2 %uint_2 %uint_264

spirv-dis vulkan/shader_dump/shader-cs-09d44a25441d4ba7.spv | grep Barrier
               OpMemoryBarrier %uint_1 %uint_3400
               OpControlBarrier %uint_2 %uint_2 %uint_264
               OpMemoryBarrier %uint_1 %uint_3400
               OpControlBarrier %uint_2 %uint_2 %uint_264
               OpMemoryBarrier %uint_1 %uint_3400
               OpControlBarrier %uint_2 %uint_2 %uint_264

```