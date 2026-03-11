# Summary

`barrier()` alone appears insufficient for `shared` memory synchronization in this length-125 FFT compute shader on MoltenVK/macOS, while `memoryBarrier(); barrier()` produces correct output.

# Expected Behavior

Both shader variants should pass against the same reference output blob. The only intentional difference is the insertion of `memoryBarrier()` before `barrier()` at the shared-memory synchronization points.

# Actual Behavior

- `barrier_only`: [PASS/FAIL]
- `memorybarrier_plus_barrier`: [PASS/FAIL]

`barrier_only` shows mismatches against the reference FFT output blob, while the workaround variant does not.

# Environment

- macOS version:
- Machine / GPU:
- Xcode version:
- Vulkan SDK version:
- MoltenVK version:
- Vulkan loader path or SDK path:

# Reproduction Steps

1. Install the LunarG macOS Vulkan SDK.
2. Configure the SDK environment.
3. Build the repro:
   `./scripts/build_macos.sh`
4. Run both variants:
   `./scripts/run_repro.sh`

# Repro Output

## barrier_only

```text
paste output here
```

## memorybarrier_plus_barrier

```text
paste output here
```

# Attached Artifacts

- GLSL source for both variants
- `.spv` binaries
- SPIR-V disassembly
- input blob and reference output blob format details
- optional generated MSL / Metal capture

# Notes

The workaround changes only the insertion of `memoryBarrier()` before `barrier()`. The rest of the shader source, pipeline state, descriptor layout, dispatch geometry, and host verification are unchanged.
