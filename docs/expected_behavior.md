# Expected Behavior

This repro is checking a narrow synchronization claim in a real FFT kernel: `barrier()` should be enough to make prior `shared` memory writes visible to other invocations in the same workgroup when control reaches the barrier in uniform flow.

Because both shader variants run the same FFT and are compared against the same reference output blob, they should produce identical output on a conformant implementation.

# Why The Shader Pattern Is Stressful

The shader intentionally keeps the failing shared-memory structure from the FFT workload:

- 25 lanes per workgroup
- 5 complex values per lane
- shared-memory shuffles through `sdata[125]`
- two shuffle stages plus an inter-stage barrier before shared-memory reuse
- in-place writeback to the storage buffer

This is still small enough to inspect, but it keeps the exact shuffle pattern that reproduced the bug.

# Result Interpretation

- Both variants pass: no reproduction in this environment.
- `barrier_only` fails and `memorybarrier_plus_barrier` passes: strong signal of a backend/compiler/translation issue.
- Both variants fail: likely a broader problem unrelated to the precise barrier distinction.
