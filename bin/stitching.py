#!/usr/bin/env python
# Compute affine transformation matrix

import argparse
import os
from utils.io import save_h5, load_pickle
from utils.cropping import reconstruct_image
from utils.read_metadata import get_image_file_shape


def _parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--crops",
        type=str,
        default=None,
        required=True,
        help="A list of crops",
    )
    parser.add_argument(
        "-c",
        "--crop_size",
        type=int,
        default=2000,
        required=False,
        help="Size of the crop",
    )
    parser.add_argument(
        "-o",
        "--overlap_size",
        type=int,
        default=200,
        required=False,
        help="Size of the overlap",
    )
    parser.add_argument(
        "-or",
        "--original_file",
        type=str,
        default=None,
        required=True,
        help="Padded full moving or fixed file.",
    )
    args = parser.parse_args()
    return args


def main():
    args = _parse_args()
    original_shape = get_image_file_shape(args.original_file, format='.h5')

    crops = []
    positions = []
    for crop in crops:
        crops.append(load_pickle(crop))
        x, y = map(int, crop.split("_")[1:3])
        positions.append((x, y))

    reconstructed_image = reconstruct_image(
        crops,
        positions,
        original_shape=original_shape,
        crop_size=args.crop_size,
        overlap_size=args.overlap_size,
    )

    save_h5(reconstructed_image, f"registered_{os.path.basename(args.original_file)}")


if __name__ == "__main__":
    main()
