#!/usr/bin/env python

import os
from skimage import morphology
from skimage.filters import gaussian
import shutil
import numpy as np
import tifffile as tiff
import argparse
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


def dapi_preprocessing(image):
    image = gamma_correction(image, 0.6)
    image = morphology.white_tophat(image, footprint=morphology.disk(50))
    image = np.array(image, dtype="float32")
    image = gaussian(image, sigma=1.0)

    return image


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
        help="List of tiff single channel images.",
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


if __name__ == "__main__":
    args = _parse_args()

    handler = logging.FileHandler(args.log_file)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    logger.debug("Checking files for segmentation preprocessing.")
    for file in args.images:
        filename = os.path.basename(file)

        if "DAPI" in filename:
            logger.info(f"Starting segmentation preprocessing on {file}")
            dapi = dapi_preprocessing(tiff.imread(file))
            logger.info(f"Saving preprocessed {filename} to preprocessed_{filename}")
            tiff.imwrite(f"prep_{filename}", dapi)
        else:
            # output_file = os.path.join(os.getcwd(), filename)
            # logger.info(f"Copying raw channel {file} to {output_file}")
            outname = f"prep_{filename}"
            logger.info(f"Copying raw channel {file} to {outname}")
            shutil.copy(file, outname)
