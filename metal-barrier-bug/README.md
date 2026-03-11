# Metal `threadgroup_barrier(mem_flags::mem_threadgroup)` Shared Memory Synchronization Bug

This repository is a minimal pure-Metal reproduction for an apparent bug in Metal's shader compiler or GPU driver. In the failing configuration, `threadgroup_barrier(mem_flags::mem_threadgroup)` does not reliably synchronize `threadgroup` memory writes between threads in the same compute workgroup. Expanding the barrier flags to include `mem_device` makes the exact same shared-memory algorithm behave correctly.

## Context

The issue was first observed through Vulkan on macOS via MoltenVK and SPIRV-Cross, but this reproduction removes both layers and exercises Metal directly. The Vulkan source pattern is a workgroup-memory `OpControlBarrier` with `AcquireRelease | WorkgroupMemory` semantics, which SPIRV-Cross translates to `threadgroup_barrier(mem_flags::mem_threadgroup)`. Per Metal's documented barrier semantics, that should be sufficient for `threadgroup` memory ordering.

## Test design

The shader dispatch uses 1024 threadgroups with 256 threads per threadgroup. Inside each threadgroup, every thread:

1. Writes a distinct `float` value into `threadgroup` memory.
2. Executes a barrier.
3. Reads a different slot written by another thread.
4. Accumulates the result.
5. Executes a second barrier before the next round.

This repeats for 8 rounds with a different shuffle offset each round. The host compares the GPU output against a CPU reference and reports exact mismatch counts.

The repository contains two kernels:

- `test_barrier_threadgroup_only`
- `test_barrier_all_flags`

The kernels are identical except for the barrier flags:

- `mem_flags::mem_threadgroup`
- `mem_flags::mem_device | mem_flags::mem_threadgroup | mem_flags::mem_texture`

## Expected behavior

Both kernels should produce identical and correct results. `mem_threadgroup` should be sufficient to synchronize `threadgroup` memory accesses inside a threadgroup.

## Actual behavior

On affected systems, the `mem_threadgroup`-only kernel intermittently returns incorrect values, indicating that some threads observe stale or uninitialized `threadgroup` memory. The broader-flags kernel acts as a workaround and produces the expected results.

If both variants pass on a given machine, that is still useful data because it suggests the issue may be hardware-, driver-, or OS-version-specific.

## Files

- `main.swift`: Metal host program that compiles pipelines, dispatches both kernels, validates results, and prints PASS/FAIL summaries.
- `barrier_test.metal`: Two compute kernels that differ only in barrier flags.
- `build.sh`: Builds the Metal library and Swift executable.

## How to build and run

```bash
cd metal-barrier-bug
./build.sh
./barrier_test
```

`build.sh` runs:

```bash
xcrun -sdk macosx metal -c barrier_test.metal -o barrier_test.air
xcrun -sdk macosx metallib barrier_test.air -o default.metallib
swiftc main.swift -o barrier_test -framework Metal -framework Foundation
```

The executable loads `./default.metallib` from the current directory rather than using an app bundle.

## Example output

```text
=== Metal threadgroup_barrier Bug Reproduction ===
Device: Apple M2 Pro
macOS: Version 14.5 (Build 23F79)

--- Test 1: threadgroup_barrier(mem_flags::mem_threadgroup) ---
  Iteration 1: 847 mismatches out of 262144 (FAIL)
  Iteration 2: 1203 mismatches out of 262144 (FAIL)
  Result: FAIL (mismatches in 10/10 iterations)

--- Test 2: threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup | mem_flags::mem_texture) ---
  Iteration 1: 0 mismatches out of 262144 (PASS)
  Iteration 2: 0 mismatches out of 262144 (PASS)
  Result: PASS (0/10 iterations had mismatches)

=== CONCLUSION ===
threadgroup_barrier with mem_threadgroup alone does NOT reliably
synchronize threadgroup memory. Adding mem_device works around the issue.
```

## Environment to report

When sharing results, include:

- macOS version
- hardware platform (Apple Silicon or Intel + discrete GPU)
- GPU model
- Metal GPU family, if known

## Metal Shading Language reference

Apple's official Metal resources page links the current Metal Shading Language Specification PDF:

- [Metal resources](https://developer.apple.com/metal/resources/)
- [Metal Shading Language Specification PDF](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf)

The relevant part of the specification is the section on threadgroup and SIMD-group synchronization functions and the `mem_flags` table for barrier functions. Apple documents `threadgroup_barrier` as an execution and memory barrier, and documents `mem_flags::mem_threadgroup` as ordering memory operations to `threadgroup` memory for threads in a threadgroup. That documented behavior is what this repro is testing.
