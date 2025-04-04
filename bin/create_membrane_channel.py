#!/usr/bin/env python


import os

os.environ["NUMBA_CACHE_DIR"] = "/tmp/numba_cache"

import numpy as np
import shutil
import argparse
import tifffile as tiff
import logging
from utils import logging_config

# Set up logging configuration
logging_config.setup_logging()
logger = logging.getLogger(__name__)

def multiply_image_and_scalar(image, s):
    return image * s


def power(image, a):
    return image**a


def gamma_correction(image, gamma):
    max_intensity = np.max(image)
    image = multiply_image_and_scalar(image, 1.0 / max_intensity)
    image = power(image, gamma)
    image = multiply_image_and_scalar(image, max_intensity)

    return image


def normalize_image(image):
    min_val = image.min(axis=(0, 1), keepdims=True)
    max_val = image.max(axis=(0, 1), keepdims=True)
    scaled_image = (image - min_val) / (max_val - min_val)

    return scaled_image


def rescale_to_uint8(image):
    # Rescale downsampled image to uint8
    min_val = image.min(axis=(1, 2), keepdims=True)
    max_val = image.max(axis=(1, 2), keepdims=True)
    image = (image - min_val) / (max_val - min_val) * 255
    image = image.astype(np.uint8)

    return image


def _parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--patient_id",
        type=str,
        default=None,
        required=True,
        help="A string containing the current patient id.",
    )
    parser.add_argument(
        "--channels",
        type=str,
        default=None,
        required=True,
        nargs='*',
        help="List of tiff single channel images.",
    )
    parser.add_argument(
        "--log_file",
        type=str,
        required=False,
        help="Path to log file.",
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = _parse_args()

    handler = logging.FileHandler(args.log_file)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    channels_files = args.channels 
    patient_id = args.patient_id

    combined_membrane_channel = []
    for file in channels_files:
        if 'VIMENTIN' in file or 'PANCK' in file:
            combined_membrane_channel.append(
                normalize_image(tiff.imread(file))
            )
        # else:
        #     shutil.copy(file, os.path.basename(file))

    combined_membrane_channel = np.stack(combined_membrane_channel)
    
    combined_membrane_channel = np.max(combined_membrane_channel, axis=0)

    combined_membrane_channel = gamma_correction(combined_membrane_channel, 0.6)

    output_file = f"{patient_id}_MEMBRANE.tiff"

    tiff.imwrite(output_file, combined_membrane_channel)






    
    
    
