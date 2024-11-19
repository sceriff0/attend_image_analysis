#!/usr/bin/env python
# Padding

import argparse
import h5py
import nd2
import tifffile


def get_image_file_shape(file, format=".h5"):
    """
    Get the width and height of a TIFF image without fully loading the image.

    Parameters:
        file (str): Path to the TIFF image.

    Returns:
        tuple: (width, height) of the image.
    """

    if format == "tiff" or format == ".tiff":
        with tifffile.TiffFile(file) as tiff:
            image_shape = tiff.pages[0].shape  # (height, width)
            width, height = image_shape[1], image_shape[0]  # Extract width and height

    if format == "nd2" or format == ".nd2":
        with nd2.ND2File(file) as nd2_file:
            # Access metadata about the dimensions
            shape_metadata = nd2_file.sizes  # Example: "XYCZT" or similar
            shape_metadata = dict(shape_metadata)
            width = shape_metadata.get("Y", 0)
            height = shape_metadata.get("X", 0)

    if format == ".h5" or format == "h5":
        with h5py.File(file, "r") as f:
            shape = f["dataset"].shape
            width, height = shape[0], shape[1]
            f.close()

    return width, height


def get_max_axis_value(files):

    shapes = [get_image_file_shape(file, format=".nd2") for file in files]
    max_shape = tuple(map(max, zip(*shapes)))

    return max_shape


def _parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        default=None,
        required=True,
        help="A string of nd2 files.",
    )

    args = parser.parse_args()
    return args


def main():

    args = _parse_args()
    files = args.input.split()
    max_shape = get_max_axis_value(files)

    with open("pad.txt", "w") as file:
        file.write(str(max_shape))


if __name__ == "__main__":
    main()
