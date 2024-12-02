#!/usr/bin/env python
# Padding

import argparse
import ast
import gc
import h5py
import os
import numpy as np
from utils.io import load_nd2, save_h5

def pad_image_to_shape(image, target_shape, constant_values=0):
    if image.shape[:2] != target_shape:
        x, y = image.shape[:2]
        w, z = target_shape

        if w < x or z < y:
            raise ValueError(
                "Target shape must be greater than or equal to the image shape."
            )

        pad_height = w - x
        pad_width = z - y

        # Distribute padding equally on both sides
        pad_top = pad_height // 2
        pad_bottom = pad_height - pad_top
        pad_left = pad_width // 2
        pad_right = pad_width - pad_left

        # Determine padding for each dimension
        if image.ndim == 2:
            # Grayscale image
            pad_widths = ((pad_top, pad_bottom), (pad_left, pad_right))
        elif image.ndim == 3:
            # Color image (e.g., RGB)
            pad_widths = ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0))
        else:
            raise ValueError("Image must be 2D or 3D array.")

        # Apply padding
        padded_image = np.pad(
            image, pad_width=pad_widths, mode="constant", constant_values=constant_values
        )

        return padded_image
    else:
        return image


def _parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--image",
        type=str,
        default=None,
        required=True,
        help="nd5 image file",
    )
    parser.add_argument(
        "-p",
        "--padding",
        type=str,
        default=None,
        required=True,
        help="Padding file containing a single line with shape in text format. E.g. (10, 10).",
    )

    args = parser.parse_args()
    return args


def main():
    args = _parse_args()
    with open(args.padding, "r") as file:
        data = file.read()

    padding_shape = ast.literal_eval(data)
    padded_image = pad_image_to_shape(
        load_nd2(args.image), 
        padding_shape
    )

    outname = str.replace(os.path.basename(args.image), "nd2", "h5")
    save_h5(padded_image, outname)

    del padded_image
    gc.collect()


if __name__ == "__main__":
    main()
