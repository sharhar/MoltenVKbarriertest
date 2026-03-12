# Metal Pipeline Descriptor Barrier Bug Repro

This harness mirrors the CLI and reporting style of `../metal/`, but runs each dumped shader through two Metal compute pipeline creation paths:

- `Baseline`: `makeComputePipelineState(function:)`
- `Bug Trigger`: `MTLComputePipelineDescriptor` with `maxTotalThreadsPerThreadgroup` set to the shader's Vulkan local size (`25`)

The goal is to provide a minimal native Metal demo showing that the dumped `barrier()`-only MSL can pass in the baseline path and fail in the descriptor-constrained path, while the `memoryBarrier(); barrier()` variant continues to pass in both.

Build with:

```bash
./build.sh
```

Run with:

```bash
./barrier_test.exec
```
