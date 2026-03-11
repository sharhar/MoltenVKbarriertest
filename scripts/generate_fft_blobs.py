#!/usr/bin/env python3
"""Generate input/reference blobs for the 875 x 125 FFT repro."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


DEFAULT_BATCHES = 875
DEFAULT_LENGTH = 125
DEFAULT_SEED = 12345


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate random complex input data and NumPy FFT reference blobs."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data"),
        help="Directory where the blobs will be written.",
    )
    parser.add_argument(
        "--input-name",
        default="fft_875x125_input.bin",
        help="Filename for the input blob.",
    )
    parser.add_argument(
        "--reference-name",
        default="fft_875x125_reference.bin",
        help="Filename for the reference FFT output blob.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Seed for NumPy's random generator.",
    )
    parser.add_argument(
        "--batches",
        type=int,
        default=DEFAULT_BATCHES,
        help="Number of FFT batches.",
    )
    parser.add_argument(
        "--length",
        type=int,
        default=DEFAULT_LENGTH,
        help="FFT length along the contiguous axis.",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="Standard deviation of the random real and imaginary components.",
    )
    return parser.parse_args()


def complex64_to_blob(array: np.ndarray) -> bytes:
    if array.dtype != np.complex64:
        raise ValueError(f"Expected complex64 array, got {array.dtype}")

    interleaved = np.empty(array.size * 2, dtype="<f4")
    flat = array.reshape(-1)
    interleaved[0::2] = flat.real.astype("<f4", copy=False)
    interleaved[1::2] = flat.imag.astype("<f4", copy=False)
    return interleaved.tobytes()


def main() -> int:
    args = parse_args()

    if args.batches <= 0 or args.length <= 0:
        raise SystemExit("--batches and --length must be positive.")

    rng = np.random.default_rng(args.seed)
    real = rng.normal(loc=0.0, scale=args.scale, size=(args.batches, args.length)).astype(
        np.float32
    )
    imag = rng.normal(loc=0.0, scale=args.scale, size=(args.batches, args.length)).astype(
        np.float32
    )
    input_data = (real + 1j * imag).astype(np.complex64)

    # Match the shader's in-place transform layout: independent FFTs along the contiguous axis.
    reference_data = np.fft.fft(input_data, axis=-1).astype(np.complex64)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    input_path = args.output_dir / args.input_name
    reference_path = args.output_dir / args.reference_name

    input_path.write_bytes(complex64_to_blob(input_data))
    reference_path.write_bytes(complex64_to_blob(reference_data))

    element_count = args.batches * args.length
    byte_count = element_count * 8
    print(f"Seed: {args.seed}")
    print(f"Shape: ({args.batches}, {args.length})")
    print(f"Elements: {element_count}")
    print(f"Bytes per blob: {byte_count}")
    print(f"Wrote input blob: {input_path}")
    print(f"Wrote reference blob: {reference_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
