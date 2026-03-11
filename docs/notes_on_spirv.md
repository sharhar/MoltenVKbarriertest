# SPIR-V Notes

This repo keeps three artifact layers for each shader variant:

- GLSL source in `shaders/`
- compiled SPIR-V in `build/shaders/`
- disassembly in `build/spirv/` after running `scripts/dump_spirv.sh`

The main inspection point is the synchronization sequence emitted around the FFT shared-memory shuffles.

Expected source-level difference:

- `barrier_only.comp` uses `barrier()` at the FFT shuffle sync sites
- `memorybarrier_plus_barrier.comp` uses `memoryBarrier(); barrier()`
- `groupmemorybarrier_plus_barrier.comp` uses `groupMemoryBarrier(); barrier()` for triangulation

The application also supports `--dump-spirv-info`, which prints a small built-in summary including the number of `OpControlBarrier` and `OpMemoryBarrier` instructions found in the loaded module.
