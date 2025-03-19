#!/usr/bin/env python

import argparse
import os
import tifffile as tiff
import logging
import numpy as np
from utils.cropping import reconstruct_image, image_reconstruction_loop
from utils.metadata_tools import get_image_file_shape
from utils import logging_config

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
        "--image",
        type=str,
        default=None,
        required=True,
        help="Path to nd2 multichannel image.",
    )
    parser.add_argument(
        "--crops",
        type=str,
        nargs='*',
        default=None,
        required=True,
        help="Path to tiff single channel image.",
    )
    parser.add_argument(
        "--is_fixed",
        type=str,
        default=None,
        required=True,
        help="Boolean string ('true' or 'false') indicating if the image is fixed.",
    )
    parser.add_argument(
        "-o",
        "--overlap_size",
        type=int,
        default=800,
        required=False,
        help="Size of the overlap between crops.",
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

    overlap_size = args.overlap_size

    crops_files = args.crops

    channel_name = os.path.basename(crops_files[0]).split('.')[1].split('_')[-3]

    image_name = os.path.basename(args.image)

    reconstructed_channel_file = f"{image_name}_{channel_name}.tif"

    original_shape = (get_image_file_shape(args.image)[1], get_image_file_shape(args.image)[2])

    reconstructed_channel = np.zeros(original_shape)
    for file in crops_files:
        sub = os.path.basename(file).split('.')[1].split('_')[-2:]
        position = tuple([int(pos) for pos in sub])
        crop = tiff.imread(file)

        reconstructed_channel = reconstruct_image(
            reconstructed_channel, 
            crop, 
            position, 
            original_shape, 
            overlap_size
        )

    
    tiff.imwrite(reconstructed_channel_file, reconstructed_channel)


if __name__ == "__main__":
    main()
