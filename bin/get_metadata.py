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


def create_tiff_metadata(src_path=None, channel_names=None, pixel_microns=[]):
    if not pixel_microns:
        metadata = get_metadata_nd2(src_path)
        pixel_microns = metadata['pixel_microns']

    resolution = (1/pixel_microns, 1/pixel_microns)
    metadata = {
        'axes': 'CYX', 
        'PhysicalSizeX': pixel_microns, 
        'PhysicalSizeY': pixel_microns, 
        'PhysicalSizeXUnit': 'µm',                             
        'PhysicalSizeYUnit': 'µm', 
        'Channel': {'Name': channel_names}
    }

    return resolution, metadata

def save_tiff(image, save_path, resolution, metadata):
    tiff.imwrite(
        save_path, 
        image, 
        resolution=resolution,
        bigtiff=True, 
        ome=True,
        metadata=metadata
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
        "-c",
        "--channels",
        type=str,
        default=None,
        required=True,
        help="String of paths to h5 files (registered images).",
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
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # Fixed-order list of channels
    channels_list = get_channel_list()

    image_files = args.image_files.split()

    nd2_files = [file for file in image_files if os.path.basename(file).split('.')[1] == 'nd2']

    channels_files = args.channels.split()

    channel_names = []
    for file in channels_files:
        filename = os.path.basename(file).split('.')[0]
        filename = filename.split('_')
        if len(filename) >= 3:
            channel_name = filename[2]
            channel_names.append(channel_name)

    if channel_names:
        channels_to_register = remove_lowercase_channels(list(set(channel_names)))
        channels_to_register = sorted(channels_to_register, key=lambda x: channels_list.index(x))

        if nd2_files:
            resolution, metadata = create_tiff_metadata(src_path=nd2_files[0], channel_names=channels_to_register)
            meta = (resolution, metadata)
        else:
            resolution, metadata = create_tiff_metadata(channel_names=channels_to_register, pixel_microns=0.34533768547788)
            meta = (resolution, metadata)
             
    else:
        meta = []

    save_path = f"metadata_{args.patient_id}.pkl"
    save_pickle(meta, save_path)

if __name__ == "__main__":
    main()