# Blob Files

Place the FFT repro blobs here unless you want to pass explicit paths on the command line.

Expected filenames:

- `fft_875x125_input.bin`
- `fft_875x125_reference.bin`

Expected format:

- row-major `(875, 125)` complex tensor
- one complex value stored as two little-endian `float32` values: `(real, imag)`
- total element count: `109375`
- total file size: `875000` bytes
