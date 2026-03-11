# Expected Behavior

This repro is checking a narrow synchronization claim: in a Vulkan GLSL compute shader, `barrier()` should be enough to make prior `shared` memory writes visible to other invocations in the same workgroup when control reaches the barrier in uniform flow.

Because both shader variants run the same workgroup-shared-memory algorithm, they should produce identical output on a conformant implementation.

# Why The Shader Pattern Is Stressful

The shader intentionally mirrors a reduced reorder/transposition workload:

- 64 lanes per workgroup
- 8 values written per lane per round
- padded shared-memory addressing with `idx = raw + (raw >> 4)`
- cross-lane reads after synchronization
- many repeated rounds with shared-memory reuse

This is still small enough to inspect, but it creates broad read-after-write dependencies across the workgroup and forces the backend to preserve shared-memory visibility repeatedly.

# Result Interpretation

- Both variants pass: no reproduction in this environment.
- `barrier_only` fails and `memorybarrier_plus_barrier` passes: strong signal of a backend/compiler/translation issue.
- Both variants fail: likely a broader problem unrelated to the precise barrier distinction.
