# MoltenVK FFT Barrier Repro

This repository reproduces a synchronization-related correctness bug seen only through the Vulkan-on-macOS path.

The workload is a batch FFT of length 125 over contiguous `complex64` data:

- 875 independent FFTs
- FFT length 125
- 25 threads per workgroup
- shared/threadgroup memory used for the two internal data reorders between radix-5 stages

## Reproduction steps

First, generate the binary input and output reference blobs using the provided Python script:

```bash
python3 generate_blobs.py --output-dir data
```

By default, the test will automatically use the installed Vulkan loader and `glslangValidator` from `PATH`, but for reproducibility and to ensure the right MoltenVK version is used, you can select a specific MoltenVK git ref and the matching glslang compiler version:

```bash
# Current commit in the main branch in MOltenVK as of 2026-03-11, and version 16.2.0 of glslang
bash vulkan/select_toolchain.sh \
  --moltenvk-ref f79c6c5690d3ee06ec3a00d11a8b1bab4aa1d030 \
  --glslang-ref f0bd0257c308b9a26562c1a30c4748a0219cc951
```

This bug was also reproduced in the latest release versions of both MoltenVK and glslang at the time of writing (1.4.1 and 16.2.0, respectively), which can be tested with:

```bash
# Latest release of MoltenVK (1.4.1) as of 2026-03-11, and version 16.2.0 of glslang
bash vulkan/select_toolchain.sh \
  --moltenvk-ref db445ff2042d9ce348c439ad8451112f354b8d2a \
  --glslang-ref f0bd0257c308b9a26562c1a30c4748a0219cc951
```

Then, build and run the Vulkan test:

```bash
bash run_all_test.sh
```

One the test ends, you can examine the SPIR-V disassembly of the two shader variants to see the difference in barrier usage:

```
$ spirv-dis vulkan/shader_dump/shader-cs-c958ee748730720d.spv | grep Barrier
               OpControlBarrier %uint_2 %uint_2 %uint_264
               OpControlBarrier %uint_2 %uint_2 %uint_264
               OpControlBarrier %uint_2 %uint_2 %uint_264

$ spirv-dis vulkan/shader_dump/shader-cs-09d44a25441d4ba7.spv | grep Barrier
               OpMemoryBarrier %uint_1 %uint_3400
               OpControlBarrier %uint_2 %uint_2 %uint_264
               OpMemoryBarrier %uint_1 %uint_3400
               OpControlBarrier %uint_2 %uint_2 %uint_264
               OpMemoryBarrier %uint_1 %uint_3400
               OpControlBarrier %uint_2 %uint_2 %uint_264
```

The `OpMemoryBarrier %uint_1 %uint_3400` instructions in the second shader correspond to the `memoryBarrier()` call in the GLSL, and the `OpControlBarrier %uint_2 %uint_2 %uint_264` instructions correspond to the `barrier()` calls. The first shader only has the control barriers, while the second shader has a memory barrier before each control barrier. Note, the scope of the barriers is the same in both shaders (`%uint_2` = `Workgroup`), but the scope of the memory barrier is `Device` in the second shader (`%uint_3400` = `Device`).

In the dumped MSL shader code, (which is identical across the current commit in the main branch and version 1.4.1 release of MoltenVK and can be found at `vulkan/shader_dump`), the `barrier()`-only shader uses `threadgroup_barrier(mem_flags::mem_threadgroup)`, while the `memoryBarrier(); barrier()` shader adds `atomic_thread_fence(mem_flags::mem_device | mem_flags::mem_threadgroup | mem_flags::mem_texture, memory_order_seq_cst, thread_scope_device);` corresponding to the `memoryBarrier();` call.

The test output will look something like the following, where the `barrier()`-only shader fails and the `memoryBarrier(); barrier()` shader passes when using the vulkan backend, but when you run the exact same dumped MSL shaders through the native Metal harness, both pass (indicating the bug is specific to the Vulkan/MoltenVK path and not to the visible generated MSL source alone):

```
=== Running vulkan ===
barrier_test.comp
barrier_test.comp
Build complete.
GLSL compiler: /Users/shaharsandhaus/MoltenVKbarriertest/build/toolchains/pairs/moltenvk-f79c6c5690d3__glslang-f0bd0257c308/tools/glslangValidator
Generate blobs with: python3 generate_blobs.py --output-dir data
Run with: ./run.sh
=== Vulkan/MoltenVK FFT Barrier Bug Reproduction ===
MoltenVK shader dump dir: /Users/shaharsandhaus/MoltenVKbarriertest/vulkan/shader_dump
[mvk-info] MoltenVK version 1.4.2, supporting Vulkan version 1.4.334.
	The following 154 Vulkan extensions are supported:
	[extensions list elided for brevity]
[mvk-info] GPU device:
	model: Apple M2 Pro
	type: Integrated
	vendorID: 0x106b
	deviceID: 0xf070208
	pipelineCacheUUID: F79C6C56-0F07-0208-0000-000100000000
	GPU memory available: 21845 MB
	GPU memory used: 0 MB
	Metal Shading Language 3.2
	supports the following GPU Features:
		GPU Family Metal 3
		GPU Family Apple 8
		GPU Family Mac 2
		Read-Write Texture Tier 2
[mvk-info] Created VkInstance for Vulkan version 1.1.334, as requested by app, with the following 0 Vulkan extensions enabled:
[mvk-info] Vulkan semaphores using MTLEvent.
[mvk-info] Descriptor sets binding resources using Metal3 argument buffers.
[mvk-info] Created VkDevice to run on GPU Apple M2 Pro with the following 1 Vulkan extensions enabled:
	VK_KHR_portability_subset v1
Managed MoltenVK ICD: /Users/shaharsandhaus/MoltenVKbarriertest/build/toolchains/pairs/moltenvk-f79c6c5690d3__glslang-f0bd0257c308/runtime/MoltenVK_icd.json
Managed glslangValidator: /Users/shaharsandhaus/MoltenVKbarriertest/build/toolchains/pairs/moltenvk-f79c6c5690d3__glslang-f0bd0257c308/tools/glslangValidator
Device: Apple M2 Pro
Vulkan API: 1.1.334
Driver version: 10402
FFT layout: 875 x 125
Threads per workgroup: 25
Iterations per variant: 10
Tolerance: abs <= 0.005, rel <= 0.0005
Input blob: ../data/fft_875x125_input.bin
Reference blob: ../data/fft_875x125_reference.bin
barrier() shader: barrier_only.spv
memoryBarrier(); barrier() shader: memory_barrier_then_barrier.spv
Expected bytes per blob: 875000

--- Test 1: barrier() ---
  Iteration 1: 455 mismatches out of 109375 (FAIL), max abs error 2.53755e+38
    fft 9, bin 24, index 1149: expected (-7.18076, 0.384117), got (-6.56776, 1.56874), abs error 1.33383
    fft 9, bin 49, index 1174: expected (7.35617, -1.21638), got (6.16393, -1.81445), abs error 1.33383
    fft 9, bin 74, index 1199: expected (14.5085, -0.161291), got (15.8246, -0.378217), abs error 1.33383
    fft 9, bin 99, index 1224: expected (-5.87513, -0.400647), got (-6.81235, 0.548417), abs error 1.33383
    fft 9, bin 124, index 1249: expected (9.17171, 9.36738), got (9.37208, 8.04868), abs error 1.33383
  Iteration 2: 495 mismatches out of 109375 (FAIL), max abs error 1.42226e+38
    fft 2, bin 24, index 274: expected (-5.23173, -6.13894), got (-2.44548, -0.75452), abs error 6.06261
    fft 2, bin 49, index 299: expected (22.528, -3.90401), got (17.109, -6.62238), abs error 6.06261
    fft 2, bin 74, index 324: expected (3.34042, -3.82987), got (9.32232, -4.81588), abs error 6.06262
    fft 2, bin 99, index 349: expected (14.0678, 13.438), got (9.80792, 17.7517), abs error 6.06262
    fft 2, bin 124, index 374: expected (28.0574, 4.89006), got (28.9682, -1.10375), abs error 6.06261
  Iteration 3: 475 mismatches out of 109375 (FAIL), max abs error 1.24247e+38
    fft 14, bin 22, index 1772: expected (6.66842, 15.7562), got (6.39273, 14.2054), abs error 1.57517
    fft 14, bin 47, index 1797: expected (20.7604, 4.38647), got (21.895, 5.4791), abs error 1.57517
    fft 14, bin 72, index 1822: expected (3.20017, 2.59572), got (1.64002, 2.37868), abs error 1.57517
    fft 14, bin 97, index 1847: expected (0.223769, -13.0917), got (1.61353, -13.8332), abs error 1.57518
    fft 14, bin 122, index 1872: expected (3.56517, -14.2263), got (2.87664, -12.8096), abs error 1.57517
  Iteration 4: 320 mismatches out of 109375 (FAIL), max abs error 3.2663e+27
    fft 2, bin 24, index 274: expected (-5.23173, -6.13894), got (-2.44548, -0.75452), abs error 6.06261
    fft 2, bin 49, index 299: expected (22.528, -3.90401), got (17.109, -6.62238), abs error 6.06261
    fft 2, bin 74, index 324: expected (3.34042, -3.82987), got (9.32232, -4.81588), abs error 6.06262
    fft 2, bin 99, index 349: expected (14.0678, 13.438), got (9.80792, 17.7517), abs error 6.06262
    fft 2, bin 124, index 374: expected (28.0574, 4.89006), got (28.9682, -1.10375), abs error 6.06261
  Iteration 5: 535 mismatches out of 109375 (FAIL), max abs error 1.39124e+28
    fft 23, bin 24, index 2899: expected (12.129, 8.87572), got (12.4665, 9.52793), abs error 0.734354
    fft 23, bin 49, index 2924: expected (-3.11814, -3.75064), got (-3.77453, -4.07991), abs error 0.734355
    fft 23, bin 74, index 2949: expected (14.2175, -1.1538), got (14.942, -1.27322), abs error 0.734351
    fft 23, bin 99, index 2974: expected (13.6292, -20.6862), got (13.1133, -20.1637), abs error 0.734354
    fft 23, bin 124, index 2999: expected (13.7507, 4.22988), got (13.861, 3.50386), abs error 0.734351
  Iteration 6: 294 mismatches out of 109375 (FAIL), max abs error 2.56674e+10
    fft 20, bin 23, index 2523: expected (6.33413, 21.0782), got (5.38079, 18.2649), abs error 2.97042
    fft 20, bin 48, index 2548: expected (-1.41286, -26.4751), got (1.01202, -24.7594), abs error 2.97042
    fft 20, bin 73, index 2573: expected (6.19791, -10.3352), got (3.22772, -10.29[mvk-info] Destroyed VkDevice on GPU Apple M2 Pro with 1 Vulkan extensions enabled.
[mvk-info] Destroyed VkPhysicalDevice for GPU Apple M2 Pro with 0 MB of GPU memory still allocated.
[mvk-info] Destroying VkInstance for Vulkan version 1.1.334 with 0 Vulkan extensions enabled.
79), abs error 2.97042
    fft 20, bin 98, index 2598: expected (-13.6511, 1.88003), got (-11.2702, 0.104001), abs error 2.97042
    fft 20, bin 123, index 2623: expected (2.70008, 23.4669), got (1.81774, 26.3033), abs error 2.97042
  Iteration 7: 650 mismatches out of 109375 (FAIL), max abs error 3.0952e+36
    fft 5, bin 4, index 629: expected (-2.4678, 2.52128), got (7.01935e+23, -2.64749e+22), abs error 7.02434e+23
    fft 5, bin 9, index 634: expected (19.7213, 17.2582), got (-3.22824e+23, -6.23857e+23), abs error 7.02434e+23
    fft 5, bin 14, index 639: expected (5.99007, 0.996526), got (-4.27031e+23, 5.57726e+23), abs error 7.02434e+23
    fft 5, bin 19, index 644: expected (-3.03586, 6.79601), got (6.86466e+23, 1.48921e+23), abs error 7.02434e+23
    fft 5, bin 24, index 649: expected (9.61623, -15.292), got (-1.57535e+23, -6.84541e+23), abs error 7.02434e+23
  Iteration 8: 512 mismatches out of 109375 (FAIL), max abs error 9.04903e+30
    fft 0, bin 20, index 20: expected (-0.67097, 21.0996), got (-0.400331, 18.9573), abs error 2.15934
    fft 0, bin 21, index 21: expected (12.0506, 3.98658), got (12.0955, 5.77404), abs error 1.78803
    fft 0, bin 24, index 24: expected (15.4574, 23.3859), got (12.8625, 18.3713), abs error 5.64628
    fft 0, bin 45, index 45: expected (5.49658, 26.6881), got (6.53685, 28.5803), abs error 2.15934
    fft 0, bin 46, index 46: expected (4.06686, 4.7141), got (2.97987, 3.29443), abs error 1.78803
  Iteration 9: 545 mismatches out of 109375 (FAIL), max abs error 2.61971e+30
    fft 2, bin 24, index 274: expected (-5.23173, -6.13894), got (-2.44548, -0.75452), abs error 6.06261
    fft 2, bin 49, index 299: expected (22.528, -3.90401), got (17.109, -6.62238), abs error 6.06261
    fft 2, bin 74, index 324: expected (3.34042, -3.82987), got (9.32232, -4.81588), abs error 6.06262
    fft 2, bin 99, index 349: expected (14.0678, 13.438), got (9.80792, 17.7517), abs error 6.06262
    fft 2, bin 124, index 374: expected (28.0574, 4.89006), got (28.9682, -1.10375), abs error 6.06261
  Iteration 10: 470 mismatches out of 109375 (FAIL), max abs error 1.88424e+33
    fft 3, bin 23, index 398: expected (-3.52483, 8.76887), got (-6.37797, 0.349308), abs error 8.88985
    fft 3, bin 24, index 399: expected (6.3577, 24.2782), got (6.1099, 23.7993), abs error 0.539181
    fft 3, bin 48, index 423: expected (7.07268, -4.82824), got (14.3298, 0.306292), abs error 8.88985
    fft 3, bin 49, index 424: expected (9.13412, 9.22396), got (9.61606, 9.46572), abs error 0.53918
    fft 3, bin 73, index 448: expected (-4.27002, 1.69342), got (-13.1592, 1.80513), abs error 8.88985
  Result: FAIL (mismatches in 10/10 iterations)

--- Test 2: memoryBarrier(); barrier() ---
  Iteration 1: 0 mismatches out of 109375 (PASS), max abs error 2.05428e-05
  Iteration 2: 0 mismatches out of 109375 (PASS), max abs error 2.05428e-05
  Iteration 3: 0 mismatches out of 109375 (PASS), max abs error 2.05428e-05
  Iteration 4: 0 mismatches out of 109375 (PASS), max abs error 2.05428e-05
  Iteration 5: 0 mismatches out of 109375 (PASS), max abs error 2.05428e-05
  Iteration 6: 0 mismatches out of 109375 (PASS), max abs error 2.05428e-05
  Iteration 7: 0 mismatches out of 109375 (PASS), max abs error 2.05428e-05
  Iteration 8: 0 mismatches out of 109375 (PASS), max abs error 2.05428e-05
  Iteration 9: 0 mismatches out of 109375 (PASS), max abs error 2.05428e-05
  Iteration 10: 0 mismatches out of 109375 (PASS), max abs error 2.05428e-05
  Result: PASS (0/10 iterations had mismatches)

=== CONCLUSION ===
The MoltenVK FFT workload reproduces a synchronization failure with barrier() alone.
Adding memoryBarrier() before barrier() is sufficient to match the NumPy reference on this machine.

=== Running metal ===
Build complete.
Run with: ./barrier_test.exec
=== Metal Dumped Shader FFT Validation ===
Device: Apple M2 Pro
macOS: Version 15.7.4 (Build 24G517)
FFT layout: 875 x 125
Threads per threadgroup: 25
Iterations per shader: 10
Tolerance: abs <= 0.005, rel <= 0.0005
Input blob: ../data/fft_875x125_input.bin
Reference blob: ../data/fft_875x125_reference.bin
Shader dump dir: ../vulkan/shader_dump
Kernel function: main0
Discovered dumped shaders: 2
Expected bytes per blob: 875000

--- Test 1: shader-cs-09d44a25441d4ba7.metal ---
  Iteration 1: 0 mismatches out of 109375 (PASS), max abs error 2.0542773e-05
  Iteration 2: 0 mismatches out of 109375 (PASS), max abs error 2.0542773e-05
  Iteration 3: 0 mismatches out of 109375 (PASS), max abs error 2.0542773e-05
  Iteration 4: 0 mismatches out of 109375 (PASS), max abs error 2.0542773e-05
  Iteration 5: 0 mismatches out of 109375 (PASS), max abs error 2.0542773e-05
  Iteration 6: 0 mismatches out of 109375 (PASS), max abs error 2.0542773e-05
  Iteration 7: 0 mismatches out of 109375 (PASS), max abs error 2.0542773e-05
  Iteration 8: 0 mismatches out of 109375 (PASS), max abs error 2.0542773e-05
  Iteration 9: 0 mismatches out of 109375 (PASS), max abs error 2.0542773e-05
  Iteration 10: 0 mismatches out of 109375 (PASS), max abs error 2.0542773e-05
  Result: PASS (0/10 iterations had mismatches)

--- Test 2: shader-cs-c958ee748730720d.metal ---
  Iteration 1: 0 mismatches out of 109375 (PASS), max abs error 2.0542773e-05
  Iteration 2: 0 mismatches out of 109375 (PASS), max abs error 2.0542773e-05
  Iteration 3: 0 mismatches out of 109375 (PASS), max abs error 2.0542773e-05
  Iteration 4: 0 mismatches out of 109375 (PASS), max abs error 2.0542773e-05
  Iteration 5: 0 mismatches out of 109375 (PASS), max abs error 2.0542773e-05
  Iteration 6: 0 mismatches out of 109375 (PASS), max abs error 2.0542773e-05
  Iteration 7: 0 mismatches out of 109375 (PASS), max abs error 2.0542773e-05
  Iteration 8: 0 mismatches out of 109375 (PASS), max abs error 2.0542773e-05
  Iteration 9: 0 mismatches out of 109375 (PASS), max abs error 2.0542773e-05
  Iteration 10: 0 mismatches out of 109375 (PASS), max abs error 2.0542773e-05
  Result: PASS (0/10 iterations had mismatches)
```