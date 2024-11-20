#!/usr/bin/env python
# Padding

import argparse
import ast
import gc
import h5py
import nd2
import os

import numpy as np


def load_nd2(file_path):
    """
    Read an ND2 file and return the image array.

    Parameters:
    file_path (str): Path to the ND2 file

    Returns:
    numpy.ndarray: Image data
    """
    with nd2.ND2File(file_path) as nd2_file:
        data = nd2_file.asarray()
        data = data.transpose((1, 2, 0))

    return data


def pad_image_to_shape(image, target_shape, constant_values=0):

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


def save_h5(data, path, chunks=None):
    # Save the NumPy array to an HDF5 file
    with h5py.File(path, "w") as hdf5_file:
        hdf5_file.create_dataset("dataset", data=data, chunks=chunks)
        hdf5_file.flush()


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
    image = load_nd2(args.image)
    padded_images = pad_image_to_shape(image, padding_shape)

    del image
    gc.collect()
    outname = str.replace(os.path.basename(args.image), "nd2", "h5")
    save_h5(padded_images, outname)


if __name__ == "__main__":
    main()
