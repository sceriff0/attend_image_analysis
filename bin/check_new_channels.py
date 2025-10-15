#!/usr/bin/env python

import os
import argparse
import logging
from utils.io import save_pickle
from utils.metadata_tools import get_image_channel_names, get_channel_list
from utils import logging_config

# Set up logging configuration
logging_config.setup_logging()
logger = logging.getLogger(__name__)


def are_all_alphabetic_lowercase(string):
    # Filter alphabetic characters and check if all are lowercase
    return all(char.islower() for char in string if char.isalpha())


def remove_lowercase_channels(channels):
    filtered_channels = []
    if channels:
        for ch in channels:
            if not are_all_alphabetic_lowercase(ch):
                filtered_channels.append(ch)
    return filtered_channels


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
        "-o",
        "--ome_tiff_image",
        type=str,
        default=None,
        required=True,
        help="Image file in .ome.tiff format",
    )
    parser.add_argument(
        "-i",
        "--nd2_files",
        type=str,
        default=None,
        required=True,
        help="A string of nd2 files.",
    )
    parser.add_argument(
        "-oc",
        "--optional_channels",
        nargs="*",
        default=[],
        help="List of optional channels",
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

    nd2_files = args.nd2_files.split()
    channels_list = get_channel_list()
    channels_list.extend(args.optional_channels.split(","))
    logger.info(f"Full channel list: {channels_list}")
    save_path = f"channels_{args.patient_id}.pkl"

    channel_names = []
    for file in nd2_files:
        filename = os.path.basename(file).split(".")[0]
        channels_in_file = filename.split("_")[1:][::-1]
        if (
            not "MLH1" in channels_in_file
        ):  # If MLH1 is not in the list, the channels belong to a moving image
            channel_names.append(channels_in_file[:-1])  # Remove DAPI channel

    # Flatten channel_names
    channel_names = [item for sublist in channel_names for item in sublist]
    channel_names = remove_lowercase_channels(list(set(channel_names)))

    registered_channels = get_image_channel_names(args.ome_tiff_image)
    new_channels = remove_lowercase_channels(
        [e for e in channel_names if e not in registered_channels]
    )

    if new_channels:
        channels_to_register = sorted(
            new_channels, key=lambda x: channels_list.index(x)
        )
    else:
        channels_to_register = []

    save_pickle(channels_to_register, save_path)


if __name__ == "__main__":
    main()
