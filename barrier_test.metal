#include <metal_stdlib>

using namespace metal;

#define THREADGROUP_SIZE 256
#define ROUNDS 8

inline float make_value(uint localID, uint round)
{
    return float(localID * 37u + round * 1000u + 7u);
}

kernel void test_barrier_threadgroup_only(device float *output [[buffer(0)]],
                                          uint gid [[thread_position_in_grid]],
                                          uint lid [[thread_index_in_threadgroup]])
{
    threadgroup float sharedData[THREADGROUP_SIZE];
    float accumulator = 0.0f;

    for (uint round = 0; round < ROUNDS; ++round) {
        sharedData[lid] = make_value(lid, round);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint readIndex = (lid + round + 1u) & (THREADGROUP_SIZE - 1u);
        accumulator += sharedData[readIndex];

        // Keep rounds independent so the only differing variable is barrier flags.
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    output[gid] = accumulator;
}

kernel void test_barrier_all_flags(device float *output [[buffer(0)]],
                                   uint gid [[thread_position_in_grid]],
                                   uint lid [[thread_index_in_threadgroup]])
{
    threadgroup float sharedData[THREADGROUP_SIZE];
    float accumulator = 0.0f;

    for (uint round = 0; round < ROUNDS; ++round) {
        sharedData[lid] = make_value(lid, round);
        threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup | mem_flags::mem_texture);

        uint readIndex = (lid + round + 1u) & (THREADGROUP_SIZE - 1u);
        accumulator += sharedData[readIndex];

        // Keep rounds independent so the only differing variable is barrier flags.
        threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup | mem_flags::mem_texture);
    }

    output[gid] = accumulator;
}
