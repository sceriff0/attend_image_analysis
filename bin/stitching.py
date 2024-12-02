#!/usr/bin/env python
# Compute affine transformation matrix

import logging
import argparse
import os
import numpy as np
from utils.io import save_h5, load_pickle
from utils.cropping import reconstruct_image
from utils.read_metadata import get_image_file_shape
from utils import logging_config

# Set up logging configuration
logging_config.setup_logging()
logger = logging.getLogger(__name__)

def _parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--crops",
        type=str,
        default=None,
        required=True,
        nargs='+',
        help="A list of crops",
    )
    parser.add_argument(
        "-cs",
        "--crop_size",
        type=int,
        default=2000,
        required=False,
        help="Size of the crop",
    )
    parser.add_argument(
        "-o",
        "--overlap_size",
        type=int,
        default=200,
        required=False,
        help="Size of the overlap",
    )
    parser.add_argument(
        "-or",
        "--original_file",
        type=str,
        default=None,
        required=True,
        help="Padded full moving or fixed file.",
    )
    args = parser.parse_args()
    return args


def main():
    handler = logging.FileHandler('/hpcnfs/scratch/DIMA/chiodin/repositories/attend_image_analysis/LOG.log')
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    args = _parse_args()
    original_shape = get_image_file_shape(args.original_file, format='.h5') #+ (3,)
    crops_files = args.crops
    reconstructed_image = np.zeros(original_shape, dtype='float32')
    for crop_file in crops_files:
        crop = load_pickle(crop_file)
        logger.info(f"Loaded crop: {crop_file}")

        x, y = map(int, os.path.basename(crop_file).split("_")[1:3])
        position = (x, y)
        reconstructed_image = reconstruct_image(reconstructed_image, crop, position, original_shape, args.overlap_size)

    save_h5(reconstructed_image, f"registered_{os.path.basename(args.original_file)}")


if __name__ == "__main__":
    main()
