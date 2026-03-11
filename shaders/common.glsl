#ifndef SHARED_SYNC
#error "SHARED_SYNC must be defined by the shader variant."
#endif

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

layout(std430, binding = 0) buffer OutputBuffer {
    uint data[];
} output_buffer;

layout(push_constant) uniform Params {
    uint rounds;
} params;

const uint VALUES_PER_THREAD = 8u;
const uint RAW_SLOTS = 64u * VALUES_PER_THREAD;
const uint PADDED_SLOTS = RAW_SLOTS + (RAW_SLOTS >> 4u);

shared uint sdata[PADDED_SLOTS];

uint padded_index(uint raw_index) {
    return raw_index + (raw_index >> 4u);
}

uint mix_bits(uint x) {
    x ^= x >> 16u;
    x *= 0x7feb352du;
    x ^= x >> 15u;
    x *= 0x846ca68bu;
    x ^= x >> 16u;
    return x;
}

void main() {
    const uint lane = gl_LocalInvocationID.x;
    const uint group = gl_WorkGroupID.x;
    uint state = mix_bits((group + 1u) * 0x9e3779b9u ^ (lane + 1u) * 0x85ebca6bu);

    for (uint round = 0u; round < params.rounds; ++round) {
        for (uint slot = 0u; slot < VALUES_PER_THREAD; ++slot) {
            const uint write_slot = (slot * 5u + round + (lane >> 4u)) & 7u;
            const uint raw = lane * VALUES_PER_THREAD + write_slot;
            const uint value = mix_bits(state
                                        ^ (round * 0x27d4eb2du)
                                        ^ (slot * 0x165667b1u)
                                        ^ (write_slot * 0x9e3779b9u));
            sdata[padded_index(raw)] = value;
        }

        SHARED_SYNC();

        uint round_accum = state ^ (round * 0x94d049bbu);
        for (uint slot = 0u; slot < VALUES_PER_THREAD; ++slot) {
            const uint src_lane = (lane * 17u + slot * 7u + round * 3u + 13u) & 63u;
            const uint src_slot = (slot * 3u + lane + round) & 7u;
            const uint observed = sdata[padded_index(src_lane * VALUES_PER_THREAD + src_slot)];
            round_accum = mix_bits(round_accum
                                   ^ observed
                                   ^ ((src_lane + 1u) * 0x85ebca6bu)
                                   ^ ((src_slot + 1u) * 0xc2b2ae35u));
        }

        state = mix_bits(round_accum ^ 0x27d4eb2du ^ lane);

        SHARED_SYNC();
    }

    output_buffer.data[gl_WorkGroupID.x * 64u + lane] = state;
}
