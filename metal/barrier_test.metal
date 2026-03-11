#pragma clang diagnostic ignored "-Wmissing-prototypes"

#include <metal_stdlib>

using namespace metal;

inline void fft125_impl(device float2 *data,
                        uint workgroupIndex,
                        uint tid,
                        threadgroup float2 *sdata,
                        mem_flags barrierFlags)
{
    uint inputBatchOffset = workgroupIndex * 125u;
    uint outputBatchOffset = 0u;
    float2 omegaRegister = float2(0.0);
    uint ioIndex = 0u;
    float2 radixRegister0 = float2(0.0);
    float2 radixRegister1 = float2(0.0);
    float2 radixRegister2 = float2(0.0);
    float2 radixRegister3 = float2(0.0);
    float2 radixRegister4 = float2(0.0);
    float2 fftReg0 = float2(0.0);
    float2 fftReg1 = float2(0.0);
    float2 fftReg2 = float2(0.0);
    float2 fftReg3 = float2(0.0);
    float2 fftReg4 = float2(0.0);

    ioIndex = tid + inputBatchOffset;
    fftReg0 = data[ioIndex];
    ioIndex = (tid + 25u) + inputBatchOffset;
    fftReg1 = data[ioIndex];
    ioIndex = (tid + 50u) + inputBatchOffset;
    fftReg2 = data[ioIndex];
    ioIndex = (tid + 75u) + inputBatchOffset;
    fftReg3 = data[ioIndex];
    ioIndex = (tid + 100u) + inputBatchOffset;
    fftReg4 = data[ioIndex];

    radixRegister0 = fftReg0;
    radixRegister0 += fftReg1;
    radixRegister0 += fftReg2;
    radixRegister0 += fftReg3;
    radixRegister0 += fftReg4;

    radixRegister1 = fftReg0;
    omegaRegister = float2(fma(fftReg1.x, 0.309017002582550048828125, (-fftReg1.y) * (-0.951056540012359619140625)),
                           fma(fftReg1.x, -0.951056540012359619140625, fftReg1.y * 0.309017002582550048828125));
    radixRegister1 += omegaRegister;
    omegaRegister = float2(fma(fftReg2.x, -0.809017002582550048828125, (-fftReg2.y) * (-0.587785243988037109375)),
                           fma(fftReg2.x, -0.587785243988037109375, fftReg2.y * (-0.809017002582550048828125)));
    radixRegister1 += omegaRegister;
    omegaRegister = float2(fma(fftReg3.x, -0.809017002582550048828125, (-fftReg3.y) * 0.587785243988037109375),
                           fma(fftReg3.x, 0.587785243988037109375, fftReg3.y * (-0.809017002582550048828125)));
    radixRegister1 += omegaRegister;
    omegaRegister = float2(fma(fftReg4.x, 0.309017002582550048828125, (-fftReg4.y) * 0.951056540012359619140625),
                           fma(fftReg4.x, 0.951056540012359619140625, fftReg4.y * 0.309017002582550048828125));
    radixRegister1 += omegaRegister;

    radixRegister2 = fftReg0;
    omegaRegister = float2(fma(fftReg1.x, -0.809017002582550048828125, (-fftReg1.y) * (-0.587785243988037109375)),
                           fma(fftReg1.x, -0.587785243988037109375, fftReg1.y * (-0.809017002582550048828125)));
    radixRegister2 += omegaRegister;
    omegaRegister = float2(fma(fftReg2.x, 0.309017002582550048828125, (-fftReg2.y) * 0.951056540012359619140625),
                           fma(fftReg2.x, 0.951056540012359619140625, fftReg2.y * 0.309017002582550048828125));
    radixRegister2 += omegaRegister;
    omegaRegister = float2(fma(fftReg3.x, 0.309017002582550048828125, (-fftReg3.y) * (-0.951056540012359619140625)),
                           fma(fftReg3.x, -0.951056540012359619140625, fftReg3.y * 0.309017002582550048828125));
    radixRegister2 += omegaRegister;
    omegaRegister = float2(fma(fftReg4.x, -0.809017002582550048828125, (-fftReg4.y) * 0.587785243988037109375),
                           fma(fftReg4.x, 0.587785243988037109375, fftReg4.y * (-0.809017002582550048828125)));
    radixRegister2 += omegaRegister;

    radixRegister3 = fftReg0;
    omegaRegister = float2(fma(fftReg1.x, -0.809017002582550048828125, (-fftReg1.y) * 0.587785243988037109375),
                           fma(fftReg1.x, 0.587785243988037109375, fftReg1.y * (-0.809017002582550048828125)));
    radixRegister3 += omegaRegister;
    omegaRegister = float2(fma(fftReg2.x, 0.309017002582550048828125, (-fftReg2.y) * (-0.951056540012359619140625)),
                           fma(fftReg2.x, -0.951056540012359619140625, fftReg2.y * 0.309017002582550048828125));
    radixRegister3 += omegaRegister;
    omegaRegister = float2(fma(fftReg3.x, 0.309017002582550048828125, (-fftReg3.y) * 0.951056540012359619140625),
                           fma(fftReg3.x, 0.951056540012359619140625, fftReg3.y * 0.309017002582550048828125));
    radixRegister3 += omegaRegister;
    omegaRegister = float2(fma(fftReg4.x, -0.809017002582550048828125, (-fftReg4.y) * (-0.587785243988037109375)),
                           fma(fftReg4.x, -0.587785243988037109375, fftReg4.y * (-0.809017002582550048828125)));
    radixRegister3 += omegaRegister;

    radixRegister4 = fftReg0;
    omegaRegister = float2(fma(fftReg1.x, 0.309017002582550048828125, (-fftReg1.y) * 0.951056540012359619140625),
                           fma(fftReg1.x, 0.951056540012359619140625, fftReg1.y * 0.309017002582550048828125));
    radixRegister4 += omegaRegister;
    omegaRegister = float2(fma(fftReg2.x, -0.809017002582550048828125, (-fftReg2.y) * 0.587785243988037109375),
                           fma(fftReg2.x, 0.587785243988037109375, fftReg2.y * (-0.809017002582550048828125)));
    radixRegister4 += omegaRegister;
    omegaRegister = float2(fma(fftReg3.x, -0.809017002582550048828125, (-fftReg3.y) * (-0.587785243988037109375)),
                           fma(fftReg3.x, -0.587785243988037109375, fftReg3.y * (-0.809017002582550048828125)));
    radixRegister4 += omegaRegister;
    omegaRegister = float2(fma(fftReg4.x, 0.309017002582550048828125, (-fftReg4.y) * (-0.951056540012359619140625)),
                           fma(fftReg4.x, -0.951056540012359619140625, fftReg4.y * 0.309017002582550048828125));
    radixRegister4 += omegaRegister;

    fftReg0 = radixRegister0;
    fftReg1 = radixRegister1;
    fftReg2 = radixRegister2;
    fftReg3 = radixRegister3;
    fftReg4 = radixRegister4;

    ioIndex = tid * 5u;
    sdata[ioIndex] = fftReg0;
    ioIndex = (tid * 5u) + 1u;
    sdata[ioIndex] = fftReg1;
    ioIndex = (tid * 5u) + 2u;
    sdata[ioIndex] = fftReg2;
    ioIndex = (tid * 5u) + 3u;
    sdata[ioIndex] = fftReg3;
    ioIndex = (tid * 5u) + 4u;
    sdata[ioIndex] = fftReg4;

    threadgroup_barrier(barrierFlags);

    ioIndex = tid;
    fftReg0 = sdata[ioIndex];
    ioIndex = tid + 25u;
    fftReg1 = sdata[ioIndex];
    ioIndex = tid + 50u;
    fftReg2 = sdata[ioIndex];
    ioIndex = tid + 75u;
    fftReg3 = sdata[ioIndex];
    ioIndex = tid + 100u;
    fftReg4 = sdata[ioIndex];

    omegaRegister.x = (-0.2513274252414703369140625) * float(tid % 5u);
    omegaRegister = float2(cos(omegaRegister.x), sin(omegaRegister.x));
    radixRegister0 = float2(fma(fftReg1.x, omegaRegister.x, (-fftReg1.y) * omegaRegister.y),
                            fma(fftReg1.x, omegaRegister.y, fftReg1.y * omegaRegister.x));
    fftReg1 = radixRegister0;
    omegaRegister.x = (-0.502654850482940673828125) * float(tid % 5u);
    omegaRegister = float2(cos(omegaRegister.x), sin(omegaRegister.x));
    radixRegister0 = float2(fma(fftReg2.x, omegaRegister.x, (-fftReg2.y) * omegaRegister.y),
                            fma(fftReg2.x, omegaRegister.y, fftReg2.y * omegaRegister.x));
    fftReg2 = radixRegister0;
    omegaRegister.x = (-0.753982245922088623046875) * float(tid % 5u);
    omegaRegister = float2(cos(omegaRegister.x), sin(omegaRegister.x));
    radixRegister0 = float2(fma(fftReg3.x, omegaRegister.x, (-fftReg3.y) * omegaRegister.y),
                            fma(fftReg3.x, omegaRegister.y, fftReg3.y * omegaRegister.x));
    fftReg3 = radixRegister0;
    omegaRegister.x = (-1.00530970096588134765625) * float(tid % 5u);
    omegaRegister = float2(cos(omegaRegister.x), sin(omegaRegister.x));
    radixRegister0 = float2(fma(fftReg4.x, omegaRegister.x, (-fftReg4.y) * omegaRegister.y),
                            fma(fftReg4.x, omegaRegister.y, fftReg4.y * omegaRegister.x));
    fftReg4 = radixRegister0;

    radixRegister0 = fftReg0;
    radixRegister0 += fftReg1;
    radixRegister0 += fftReg2;
    radixRegister0 += fftReg3;
    radixRegister0 += fftReg4;

    radixRegister1 = fftReg0;
    omegaRegister = float2(fma(fftReg1.x, 0.309017002582550048828125, (-fftReg1.y) * (-0.951056540012359619140625)),
                           fma(fftReg1.x, -0.951056540012359619140625, fftReg1.y * 0.309017002582550048828125));
    radixRegister1 += omegaRegister;
    omegaRegister = float2(fma(fftReg2.x, -0.809017002582550048828125, (-fftReg2.y) * (-0.587785243988037109375)),
                           fma(fftReg2.x, -0.587785243988037109375, fftReg2.y * (-0.809017002582550048828125)));
    radixRegister1 += omegaRegister;
    omegaRegister = float2(fma(fftReg3.x, -0.809017002582550048828125, (-fftReg3.y) * 0.587785243988037109375),
                           fma(fftReg3.x, 0.587785243988037109375, fftReg3.y * (-0.809017002582550048828125)));
    radixRegister1 += omegaRegister;
    omegaRegister = float2(fma(fftReg4.x, 0.309017002582550048828125, (-fftReg4.y) * 0.951056540012359619140625),
                           fma(fftReg4.x, 0.951056540012359619140625, fftReg4.y * 0.309017002582550048828125));
    radixRegister1 += omegaRegister;

    radixRegister2 = fftReg0;
    omegaRegister = float2(fma(fftReg1.x, -0.809017002582550048828125, (-fftReg1.y) * (-0.587785243988037109375)),
                           fma(fftReg1.x, -0.587785243988037109375, fftReg1.y * (-0.809017002582550048828125)));
    radixRegister2 += omegaRegister;
    omegaRegister = float2(fma(fftReg2.x, 0.309017002582550048828125, (-fftReg2.y) * 0.951056540012359619140625),
                           fma(fftReg2.x, 0.951056540012359619140625, fftReg2.y * 0.309017002582550048828125));
    radixRegister2 += omegaRegister;
    omegaRegister = float2(fma(fftReg3.x, 0.309017002582550048828125, (-fftReg3.y) * (-0.951056540012359619140625)),
                           fma(fftReg3.x, -0.951056540012359619140625, fftReg3.y * 0.309017002582550048828125));
    radixRegister2 += omegaRegister;
    omegaRegister = float2(fma(fftReg4.x, -0.809017002582550048828125, (-fftReg4.y) * 0.587785243988037109375),
                           fma(fftReg4.x, 0.587785243988037109375, fftReg4.y * (-0.809017002582550048828125)));
    radixRegister2 += omegaRegister;

    radixRegister3 = fftReg0;
    omegaRegister = float2(fma(fftReg1.x, -0.809017002582550048828125, (-fftReg1.y) * 0.587785243988037109375),
                           fma(fftReg1.x, 0.587785243988037109375, fftReg1.y * (-0.809017002582550048828125)));
    radixRegister3 += omegaRegister;
    omegaRegister = float2(fma(fftReg2.x, 0.309017002582550048828125, (-fftReg2.y) * (-0.951056540012359619140625)),
                           fma(fftReg2.x, -0.951056540012359619140625, fftReg2.y * 0.309017002582550048828125));
    radixRegister3 += omegaRegister;
    omegaRegister = float2(fma(fftReg3.x, 0.309017002582550048828125, (-fftReg3.y) * 0.951056540012359619140625),
                           fma(fftReg3.x, 0.951056540012359619140625, fftReg3.y * 0.309017002582550048828125));
    radixRegister3 += omegaRegister;
    omegaRegister = float2(fma(fftReg4.x, -0.809017002582550048828125, (-fftReg4.y) * (-0.587785243988037109375)),
                           fma(fftReg4.x, -0.587785243988037109375, fftReg4.y * (-0.809017002582550048828125)));
    radixRegister3 += omegaRegister;

    radixRegister4 = fftReg0;
    omegaRegister = float2(fma(fftReg1.x, 0.309017002582550048828125, (-fftReg1.y) * 0.951056540012359619140625),
                           fma(fftReg1.x, 0.951056540012359619140625, fftReg1.y * 0.309017002582550048828125));
    radixRegister4 += omegaRegister;
    omegaRegister = float2(fma(fftReg2.x, -0.809017002582550048828125, (-fftReg2.y) * 0.587785243988037109375),
                           fma(fftReg2.x, 0.587785243988037109375, fftReg2.y * (-0.809017002582550048828125)));
    radixRegister4 += omegaRegister;
    omegaRegister = float2(fma(fftReg3.x, -0.809017002582550048828125, (-fftReg3.y) * (-0.587785243988037109375)),
                           fma(fftReg3.x, -0.587785243988037109375, fftReg3.y * (-0.809017002582550048828125)));
    radixRegister4 += omegaRegister;
    omegaRegister = float2(fma(fftReg4.x, 0.309017002582550048828125, (-fftReg4.y) * (-0.951056540012359619140625)),
                           fma(fftReg4.x, -0.951056540012359619140625, fftReg4.y * 0.309017002582550048828125));
    radixRegister4 += omegaRegister;

    fftReg0 = radixRegister0;
    fftReg1 = radixRegister1;
    fftReg2 = radixRegister2;
    fftReg3 = radixRegister3;
    fftReg4 = radixRegister4;

    threadgroup_barrier(barrierFlags);

    ioIndex = (tid * 5u) - ((tid % 5u) << uint(2));
    sdata[ioIndex] = fftReg0;
    ioIndex = ((tid * 5u) - ((tid % 5u) << uint(2))) + 5u;
    sdata[ioIndex] = fftReg1;
    ioIndex = ((tid * 5u) - ((tid % 5u) << uint(2))) + 10u;
    sdata[ioIndex] = fftReg2;
    ioIndex = ((tid * 5u) - ((tid % 5u) << uint(2))) + 15u;
    sdata[ioIndex] = fftReg3;
    ioIndex = ((tid * 5u) - ((tid % 5u) << uint(2))) + 20u;
    sdata[ioIndex] = fftReg4;

    threadgroup_barrier(barrierFlags);

    ioIndex = tid;
    fftReg0 = sdata[ioIndex];
    ioIndex = tid + 25u;
    fftReg1 = sdata[ioIndex];
    ioIndex = tid + 50u;
    fftReg2 = sdata[ioIndex];
    ioIndex = tid + 75u;
    fftReg3 = sdata[ioIndex];
    ioIndex = tid + 100u;
    fftReg4 = sdata[ioIndex];

    omegaRegister.x = (-0.050265483558177947998046875) * float(tid);
    omegaRegister = float2(cos(omegaRegister.x), sin(omegaRegister.x));
    radixRegister0 = float2(fma(fftReg1.x, omegaRegister.x, (-fftReg1.y) * omegaRegister.y),
                            fma(fftReg1.x, omegaRegister.y, fftReg1.y * omegaRegister.x));
    fftReg1 = radixRegister0;
    omegaRegister.x = (-0.10053096711635589599609375) * float(tid);
    omegaRegister = float2(cos(omegaRegister.x), sin(omegaRegister.x));
    radixRegister0 = float2(fma(fftReg2.x, omegaRegister.x, (-fftReg2.y) * omegaRegister.y),
                            fma(fftReg2.x, omegaRegister.y, fftReg2.y * omegaRegister.x));
    fftReg2 = radixRegister0;
    omegaRegister.x = (-0.1507964432239532470703125) * float(tid);
    omegaRegister = float2(cos(omegaRegister.x), sin(omegaRegister.x));
    radixRegister0 = float2(fma(fftReg3.x, omegaRegister.x, (-fftReg3.y) * omegaRegister.y),
                            fma(fftReg3.x, omegaRegister.y, fftReg3.y * omegaRegister.x));
    fftReg3 = radixRegister0;
    omegaRegister.x = (-0.2010619342327117919921875) * float(tid);
    omegaRegister = float2(cos(omegaRegister.x), sin(omegaRegister.x));
    radixRegister0 = float2(fma(fftReg4.x, omegaRegister.x, (-fftReg4.y) * omegaRegister.y),
                            fma(fftReg4.x, omegaRegister.y, fftReg4.y * omegaRegister.x));
    fftReg4 = radixRegister0;

    radixRegister0 = fftReg0;
    radixRegister0 += fftReg1;
    radixRegister0 += fftReg2;
    radixRegister0 += fftReg3;
    radixRegister0 += fftReg4;

    radixRegister1 = fftReg0;
    omegaRegister = float2(fma(fftReg1.x, 0.309017002582550048828125, (-fftReg1.y) * (-0.951056540012359619140625)),
                           fma(fftReg1.x, -0.951056540012359619140625, fftReg1.y * 0.309017002582550048828125));
    radixRegister1 += omegaRegister;
    omegaRegister = float2(fma(fftReg2.x, -0.809017002582550048828125, (-fftReg2.y) * (-0.587785243988037109375)),
                           fma(fftReg2.x, -0.587785243988037109375, fftReg2.y * (-0.809017002582550048828125)));
    radixRegister1 += omegaRegister;
    omegaRegister = float2(fma(fftReg3.x, -0.809017002582550048828125, (-fftReg3.y) * 0.587785243988037109375),
                           fma(fftReg3.x, 0.587785243988037109375, fftReg3.y * (-0.809017002582550048828125)));
    radixRegister1 += omegaRegister;
    omegaRegister = float2(fma(fftReg4.x, 0.309017002582550048828125, (-fftReg4.y) * 0.951056540012359619140625),
                           fma(fftReg4.x, 0.951056540012359619140625, fftReg4.y * 0.309017002582550048828125));
    radixRegister1 += omegaRegister;

    radixRegister2 = fftReg0;
    omegaRegister = float2(fma(fftReg1.x, -0.809017002582550048828125, (-fftReg1.y) * (-0.587785243988037109375)),
                           fma(fftReg1.x, -0.587785243988037109375, fftReg1.y * (-0.809017002582550048828125)));
    radixRegister2 += omegaRegister;
    omegaRegister = float2(fma(fftReg2.x, 0.309017002582550048828125, (-fftReg2.y) * 0.951056540012359619140625),
                           fma(fftReg2.x, 0.951056540012359619140625, fftReg2.y * 0.309017002582550048828125));
    radixRegister2 += omegaRegister;
    omegaRegister = float2(fma(fftReg3.x, 0.309017002582550048828125, (-fftReg3.y) * (-0.951056540012359619140625)),
                           fma(fftReg3.x, -0.951056540012359619140625, fftReg3.y * 0.309017002582550048828125));
    radixRegister2 += omegaRegister;
    omegaRegister = float2(fma(fftReg4.x, -0.809017002582550048828125, (-fftReg4.y) * 0.587785243988037109375),
                           fma(fftReg4.x, 0.587785243988037109375, fftReg4.y * (-0.809017002582550048828125)));
    radixRegister2 += omegaRegister;

    radixRegister3 = fftReg0;
    omegaRegister = float2(fma(fftReg1.x, -0.809017002582550048828125, (-fftReg1.y) * 0.587785243988037109375),
                           fma(fftReg1.x, 0.587785243988037109375, fftReg1.y * (-0.809017002582550048828125)));
    radixRegister3 += omegaRegister;
    omegaRegister = float2(fma(fftReg2.x, 0.309017002582550048828125, (-fftReg2.y) * (-0.951056540012359619140625)),
                           fma(fftReg2.x, -0.951056540012359619140625, fftReg2.y * 0.309017002582550048828125));
    radixRegister3 += omegaRegister;
    omegaRegister = float2(fma(fftReg3.x, 0.309017002582550048828125, (-fftReg3.y) * 0.951056540012359619140625),
                           fma(fftReg3.x, 0.951056540012359619140625, fftReg3.y * 0.309017002582550048828125));
    radixRegister3 += omegaRegister;
    omegaRegister = float2(fma(fftReg4.x, -0.809017002582550048828125, (-fftReg4.y) * (-0.587785243988037109375)),
                           fma(fftReg4.x, -0.587785243988037109375, fftReg4.y * (-0.809017002582550048828125)));
    radixRegister3 += omegaRegister;

    radixRegister4 = fftReg0;
    omegaRegister = float2(fma(fftReg1.x, 0.309017002582550048828125, (-fftReg1.y) * 0.951056540012359619140625),
                           fma(fftReg1.x, 0.951056540012359619140625, fftReg1.y * 0.309017002582550048828125));
    radixRegister4 += omegaRegister;
    omegaRegister = float2(fma(fftReg2.x, -0.809017002582550048828125, (-fftReg2.y) * 0.587785243988037109375),
                           fma(fftReg2.x, 0.587785243988037109375, fftReg2.y * (-0.809017002582550048828125)));
    radixRegister4 += omegaRegister;
    omegaRegister = float2(fma(fftReg3.x, -0.809017002582550048828125, (-fftReg3.y) * (-0.587785243988037109375)),
                           fma(fftReg3.x, -0.587785243988037109375, fftReg3.y * (-0.809017002582550048828125)));
    radixRegister4 += omegaRegister;
    omegaRegister = float2(fma(fftReg4.x, 0.309017002582550048828125, (-fftReg4.y) * (-0.951056540012359619140625)),
                           fma(fftReg4.x, -0.951056540012359619140625, fftReg4.y * 0.309017002582550048828125));
    radixRegister4 += omegaRegister;

    fftReg0 = radixRegister0;
    fftReg1 = radixRegister1;
    fftReg2 = radixRegister2;
    fftReg3 = radixRegister3;
    fftReg4 = radixRegister4;

    outputBatchOffset = workgroupIndex * 125u;
    ioIndex = tid + outputBatchOffset;
    data[ioIndex] = fftReg0;
    ioIndex = (tid + 25u) + outputBatchOffset;
    data[ioIndex] = fftReg1;
    ioIndex = (tid + 50u) + outputBatchOffset;
    data[ioIndex] = fftReg2;
    ioIndex = (tid + 75u) + outputBatchOffset;
    data[ioIndex] = fftReg3;
    ioIndex = (tid + 100u) + outputBatchOffset;
    data[ioIndex] = fftReg4;
}

kernel void fft_875x125_threadgroup_only(device float2 *data [[buffer(0)]],
                                         uint3 workgroupID [[threadgroup_position_in_grid]],
                                         uint3 localID [[thread_position_in_threadgroup]])
{
    threadgroup float2 sdata[125];
    fft125_impl(data, workgroupID.x, localID.x, sdata, mem_flags::mem_threadgroup);
}

kernel void fft_875x125_device_and_threadgroup(device float2 *data [[buffer(0)]],
                                               uint3 workgroupID [[threadgroup_position_in_grid]],
                                               uint3 localID [[thread_position_in_threadgroup]])
{
    threadgroup float2 sdata[125];
    fft125_impl(data, workgroupID.x, localID.x, sdata, mem_flags::mem_device | mem_flags::mem_threadgroup);
}

kernel void fft_875x125_all_flags(device float2 *data [[buffer(0)]],
                                  uint3 workgroupID [[threadgroup_position_in_grid]],
                                  uint3 localID [[thread_position_in_threadgroup]])
{
    threadgroup float2 sdata[125];
    fft125_impl(data,
                workgroupID.x,
                localID.x,
                sdata,
                mem_flags::mem_device | mem_flags::mem_threadgroup | mem_flags::mem_texture);
}
