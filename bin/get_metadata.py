#!/usr/bin/env python

import os
import numpy as np
import tifffile as tiff
import argparse
import logging
from utils.metadata_tools import get_channel_list, get_metadata_nd2
from utils.io import save_pickle
from utils import logging_config

# Set up logging configuration
logging_config.setup_logging()
logger = logging.getLogger(__name__)


def create_tiff_metadata(channel_names=None, pixel_microns=[]):
    resolution = (1 / pixel_microns, 1 / pixel_microns)
    metadata = {
        "axes": "CYX",
        "PhysicalSizeX": pixel_microns,
        "PhysicalSizeY": pixel_microns,
        "PhysicalSizeXUnit": "µm",
        "PhysicalSizeYUnit": "µm",
        "Channel": {"Name": channel_names},
    }

    return resolution, metadata


def save_tiff(image, save_path, resolution, metadata):
    tiff.imwrite(
        save_path,
        image,
        resolution=resolution,
        bigtiff=True,
        ome=True,
        metadata=metadata,
    )


def are_all_alphabetic_lowercase(string):
    # Filter alphabetic characters and check if all are lowercase
    return all(char.islower() for char in string if char.isalpha())


def remove_lowercase_channels(channels):
    filtered_channels = []
    for ch in channels:
        if not are_all_alphabetic_lowercase(ch):
            filtered_channels.append(ch)
    return filtered_channels


def _parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--image_files",
        type=str,
        default=None,
        required=True,
        help="A string of nd2 files.",
    )
    parser.add_argument(
        "-pm",
        "--pixel_microns",
        type=float,
        default=0.34533768547788,
        required=False,
        help="Pixel size in microns.",
    )
    parser.add_argument(
        "-p",
        "--patient_id",
        type=str,
        default=None,
        required=True,
        help="A string containing the current patient id.",
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

    # Fixed-order list of channels
    channels_list = get_channel_list()

    image_files = args.image_files.split()

    channel_names = []
    for ch in channels_list:
        # if ch is within the name of any image_files
        if any(ch in img for img in image_files):
            channel_names.append(ch)

    if channel_names:
        resolution, metadata = create_tiff_metadata(
            channel_names=channel_names, pixel_microns=0.34533768547788
        )
        meta = (resolution, metadata)
    else:
        meta = []
    save_path = f"metadata_{args.patient_id}.pkl"
    save_pickle(meta, save_path)


if __name__ == "__main__":
    main()
