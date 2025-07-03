#!/usr/bin/env python
# Padding

import argparse
import os
import logging
import tifffile
import nd2
import h5py
from utils import logging_config
# from utils.metadata_tools import get_image_file_shape

# Set up logging configuration
logging_config.setup_logging()
logger = logging.getLogger(__name__)


def get_image_file_shape(file, format=None):
    """
    Get the width and height of a TIFF image without fully loading the image.

    Parameters:
        file (str): Path to the TIFF image.

    Returns:
        tuple: (width, height) of the image.
    """
    if format is None:
        format = file.split(".")[-1]

    if format == "tiff" or format == ".tiff":
        with tifffile.TiffFile(file) as tiff:
            shape = tiff.pages[0].shape  # (height, width)

    if format == "nd2" or format == ".nd2":
        with nd2.ND2File(file) as nd2_file:
            # Access metadata about the dimensions
            shape_metadata = nd2_file.sizes  # Example: "XYCZT" or similar
            shape_metadata = dict(shape_metadata)
            shape = (
                shape_metadata.get("C", 0),
                shape_metadata.get("Y", 0),
                shape_metadata.get("X", 0),
            )

    if format == ".h5" or format == "h5":
        with h5py.File(file, "r") as f:
            shape = f["dataset"].shape
            f.close()

    return shape


def get_max_axis_value(files):
    formats = [os.path.basename(file).split(".")[1] for file in files]

    shapes = [
        (
            get_image_file_shape(file, format=format)[1],
            get_image_file_shape(file, format=format)[2],
        )
        for file, format in zip(files, formats)
    ]
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
    parser.add_argument(
        "-l",
        "--log_file",
        type=str,
        required=False,
        help="Path to log file.",
    )

    args = parser.parse_args()
    return args


def main():
    args = _parse_args()

    handler = logging.FileHandler(args.log_file)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    files = args.input.split()

    max_shape = get_max_axis_value(files)

    logger.debug(f"MAX SHAPE: {max_shape}")

    with open("pad.txt", "w") as file:
        file.write(str(max_shape))


if __name__ == "__main__":
    main()
