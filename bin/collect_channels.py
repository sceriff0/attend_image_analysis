#!/usr/bin/env python

import argparse
import os
import tifffile as tiff
import logging
import numpy as np
from utils import logging_config
from utils.io import load_nd2
from utils.io import save_h5


def extract_alpha(s):
    return "".join(c for c in s if c.isalpha())


# Set up logging configuration
logging_config.setup_logging()
logger = logging.getLogger(__name__)


def _parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p",
        "--patient_id",
        type=str,
        default=None,
        required=True,
        help="A string containing the current patient id.",
    )
    parser.add_argument(
        "-i",
        "--images",
        type=str,
        default=None,
        required=True,
        nargs="*",
        help="List of nd2 multichannel image names.",
    )
    parser.add_argument(
        "-c",
        "--channels",
        type=str,
        default=None,
        required=True,
        nargs="*",
        help="List of single channel tif files.",
    )
    parser.add_argument(
        "--is_fixed",
        type=str,
        default=None,
        required=True,
        nargs="*",
        help="List of boolean values indicating if the image is fixed or not.",
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


def sort_paths_by_marker(list_of_lists, list_of_orders):
    """
    Sorts each inner list of paths based on its corresponding marker order.

    Parameters:
    - list_of_lists: List of lists of file paths.
    - list_of_orders: List of marker orders corresponding to each inner list.

    Returns:
    - Sorted list of lists.
    """
    sorted_lists = []

    for paths, marker_order in zip(list_of_lists, list_of_orders):
        sorted_paths = sorted(
            paths,
            key=lambda x: marker_order.index(x.split(".nd2_")[-1].split(".tif")[0]),
        )
        sorted_lists.append(sorted_paths)

    return sorted_lists


def main():
    args = _parse_args()

    handler = logging.FileHandler(args.log_file)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    images = args.images
    channels = args.channels

    is_fixed = [extract_alpha(f) for f in args.is_fixed]

    unique_images = []
    is_fixed_unique = []
    for is_f, image in zip(is_fixed, images):
        base = os.path.basename(image)
        if base not in unique_images:
            is_fixed_unique.append(is_f)
            unique_images.append(base)

    unique_images = [
        (image, is_f) for image, is_f in zip(unique_images, is_fixed_unique)
    ]

    logger.debug(f"Unique images: {unique_images}")

    output_names = []
    for image, is_f in unique_images:
        image_base = os.path.basename(image)

        # Get channels in the image from the image name
        if "__" in image_base:
            channels_in_image = (
                image_base.replace("__", "-").split(".")[0].split("_")[1:][::-1]
            )
        else:
            channels_in_image = image_base.split(".")[0].split("_")[1:][::-1]

        channels_to_be_collected = []
        stacked_channels = []
        for ch in channels:
            if image_base in ch:
                channels_to_be_collected.append((ch, is_f))

        logger.debug(f"Channels in image {image_base}: {channels_in_image}")
        logger.debug(f"Channels to be collected: {channels_to_be_collected}")

        sorted_channels = sorted(
            channels_to_be_collected,
            key=lambda x: channels_in_image.index(
                x[0].split(".nd2_")[-1].split(".tif")[0]
            ),
        )

        # logger.debug(f"Sorted channels: {sorted_channels}")

        for ch, is_f in sorted_channels:
            stacked_channels.append(tiff.imread(ch))

        # logger.debug(f"Stacked channels: {stacked_channels}")

        stacked_channels = np.stack(stacked_channels, axis=0)

        stacked_channels_output_path = f"{image_base.split('.')[0]}.h5"

        output_names.append((stacked_channels_output_path, is_f))
        save_h5(stacked_channels, stacked_channels_output_path)

    # logger.debug(f"Output names: {output_names}")

    # Save csv with metadata (patient_id and output names)
    cwd = os.getcwd()
    with open(f"{args.patient_id}.csv", "w") as f:
        f.write("patient_id,image,fixed\n")
        for output_name, is_f in output_names:
            f.write(f"{args.patient_id},{os.path.join(cwd, output_name)},{is_f}\n")


if __name__ == "__main__":
    main()
