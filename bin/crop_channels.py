#!/usr/bin/env python

import argparse
import os
import tifffile as tiff
import logging
from utils import logging_config

# Set up logging configuration
logging_config.setup_logging()
logger = logging.getLogger(__name__)


def get_crops_positions(shape, crop_size, overlap_size):
    """
    Create overlapping crops from a 3D image.

    Args:
        image (numpy.ndarray): The input image of shape (X, Y, Z) with dtype uint16.
        crop_size (int): The size of the crops along X and Y (a x a).
        overlap_size (int): The overlap size along X and Y.

    Returns:
        list: A list of crops of shape (a, a, Z).
        list: A list of the top-left corner indices of each crop.
    """
    if len(shape) == 2:
        Y, X = shape
    elif len(shape) == 3:
        Y, X, Z = shape

    positions = []

    for start_col in range(0, X - overlap_size, crop_size - overlap_size):
        for start_row in range(0, Y - overlap_size, crop_size - overlap_size):
            # Ensure crop dimensions don't exceed the image dimensions
            end_row = min(start_row + crop_size, Y)
            end_col = min(start_col + crop_size, X)
            positions.append((start_row, end_row, start_col, end_col))

    return positions


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
        "--channel",
        type=str,
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
        "-cd",
        "--crop_size",
        type=int,
        default=2000,
        required=False,
        help="Size of the crop.",
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

#
def main():
    args = _parse_args()

    handler = logging.FileHandler(args.log_file)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    channel = tiff.imread(args.channel)
    channel_name = os.path.basename(args.channel).split('.')[0].split('_')[-1]

    shape = channel.shape
    crop_size = args.crop_size
    overlap_size = args.overlap_size

    crops_positions = get_crops_positions(shape, crop_size, overlap_size)

    for pos in crops_positions:
        outname = f"{args.patient_id}_{channel_name}_{pos[0]}_{pos[2]}.tiff"
        logger.debug(f"Crop position: {pos}")
        crop = channel[pos[0]:pos[1],pos[2]:pos[3]]

        tiff.imwrite(outname, crop)


    

if __name__ == "__main__":
    main()
