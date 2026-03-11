# `threadgroup_barrier(mem_flags::mem_threadgroup)` FFT Synchronization Repro

This repository is organized around a shared FFT test corpus plus backend-specific harnesses. It exercises a 125-point FFT kernel that intermittently miscomputes on macOS when the shader relies on the weaker shared-memory barrier form. The Metal backend shows the issue via `threadgroup_barrier(mem_flags::mem_threadgroup)`, and the Vulkan backend mirrors the same workload through GLSL/SPIR-V on MoltenVK.

## Context

The original issue was found through Vulkan on macOS via MoltenVK and SPIRV-Cross. The repository is arranged so multiple backends can run the same input and reference blobs and be compared directly.

The workload performs 875 independent FFTs of length 125 over contiguous complex input data. Each workgroup has 25 threads and uses shared memory for the two internal data reorders between radix-5 stages.

## Repository layout

- `data/`: shared input and reference blobs.
- `generate_blobs.py`: writes the shared input and reference blobs in the binary layout expected by backend harnesses.
- `metal/`: Metal-specific shader, host harness, and build script.
- `vulkan/`: Vulkan/MoltenVK host harness, GLSL shader, SPIR-V build script.

## Shared data layout

`generate_blobs.py` generates:

- input shape `(875, 125)` of `complex64`
- reference output `np.fft.fft(input_data, axis=-1).astype(np.complex64)`
- binary blob format: interleaved little-endian `float32` real/imag pairs

Each blob is `875 * 125 * 8 = 875000` bytes.

## Metal backend

The Metal harness builds and runs three kernels that differ only in barrier flags:

- `fft_875x125_threadgroup_only`
- `fft_875x125_device_and_threadgroup`
- `fft_875x125_all_flags`

All three kernels use the same FFT arithmetic, the same 25-thread workgroup shape, and the same three shared-memory barrier sites.

Build and run:

```bash
python3 generate_blobs.py --output-dir data
cd metal
./build.sh
./barrier_test.exec
```

Override paths and tolerances if needed:

```bash
cd metal
./barrier_test.exec \
  --input ../data/fft_875x125_input.bin \
  --reference ../data/fft_875x125_reference.bin \
  --iterations 20 \
  --abs-tol 0.005 \
  --rel-tol 0.0005
```

The harness prints:

- mismatch count per iteration
- first few mismatched FFT bins
- maximum absolute complex error
- PASS/FAIL summary across all iterations

If the bug reproduces, the expected pattern is:

- `mem_threadgroup` only: intermittent FAIL
- `mem_device | mem_threadgroup`: PASS
- `mem_device | mem_threadgroup | mem_texture`: PASS

## Vulkan backend

The Vulkan harness runs the same FFT workload through a GLSL compute shader compiled to SPIR-V and loaded from disk at runtime by the C++ executable. It tests exactly two synchronization variants:

- `barrier()`
- `memoryBarrier(); barrier()`

Before creating the Vulkan instance, the harness sets `MVK_CONFIG_SHADER_DUMP_DIR` to `vulkan/shader_dump/` by default so MoltenVK will emit the generated Metal shader sources there for inspection.

Build and run:

```bash
python3 generate_blobs.py --output-dir data
cd vulkan
./build.sh
./barrier_test.exec
```

Override paths and tolerances if needed:

```bash
cd vulkan
./barrier_test.exec \
  --input ../data/fft_875x125_input.bin \
  --reference ../data/fft_875x125_reference.bin \
  --iterations 20 \
  --abs-tol 0.005 \
  --rel-tol 0.0005 \
  --barrier-only-spv barrier_only.spv \
  --memory-and-barrier-spv memory_barrier_then_barrier.spv \
  --shader-dump-dir ./shader_dump
```

If the MoltenVK bug reproduces, the expected pattern is:

- `barrier()` only: intermittent FAIL
- `memoryBarrier(); barrier()`: PASS

## Environment to report

When sharing results, include:

- macOS version
- hardware platform (Apple Silicon or Intel + discrete GPU)
- GPU model
- Metal GPU family, if known
- Xcode version or Command Line Tools version

## Metal Shading Language reference

Apple's official Metal resources page links the current Metal Shading Language Specification PDF:

- [Metal resources](https://developer.apple.com/metal/resources/)
- [Metal Shading Language Specification PDF](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf)

The relevant part of the specification is the section on threadgroup and SIMD-group synchronization functions and the `mem_flags` table for barrier functions. Apple documents `threadgroup_barrier` as an execution and memory barrier, and documents `mem_flags::mem_threadgroup` as ordering memory operations to threadgroup memory for threads in a threadgroup. This repro checks whether that documented behavior holds for the FFT kernel's shared-memory stages.
