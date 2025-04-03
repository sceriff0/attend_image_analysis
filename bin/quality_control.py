#!/usr/bin/env python
# Compute affine transformation matrix

import logging
import argparse
import os
import numpy as np
import re
import gc
import matplotlib.pyplot as plt
import tifffile as tiff
from skimage.transform import rescale
from utils.io import load_h5
from utils.cropping import image_reconstruction_loop, get_crop_areas
from utils.metadata_tools import get_image_file_shape
from utils import logging_config

# Set up logging configuration
logging_config.setup_logging()
logger = logging.getLogger(__name__)


def normalize_image(image):
    """Normalize each channel of the image independently to [0, 255] uint8."""
    min_val = image.min(axis=(1, 2), keepdims=True)
    max_val = image.max(axis=(1, 2), keepdims=True)
    scaled_image = (image - min_val) / (max_val - min_val) * 255
    return scaled_image.astype(np.uint8)


def save_dapi_stack(
    dapi_crops_files, moving_path, fixed_path, shape, overlap_size, scale=1
):
    outname = os.path.basename(moving_path).split(".")[0]
    output_path = f"QC_{outname}.tiff".replace("padded_", "")

    reconstructed_image = np.squeeze(
        image_reconstruction_loop(dapi_crops_files, shape, overlap_size)
    ).astype("uint16")

    fixed_dapi = np.squeeze(load_h5(fixed_path, channels_to_load=-1))

    logger.info(f"Quality control - Saving {output_path}")

    # Stack images along the channel axis (c, n, m)
    dapi_stack = np.stack((reconstructed_image, fixed_dapi), axis=0)

    del reconstructed_image, fixed_dapi
    gc.collect()

    # Normalize each channel independently
    dapi_stack = normalize_image(dapi_stack)

    # Downsample each channel separately
    downsampled_image = np.array(
        [rescale(channel, scale=0.25, anti_aliasing=True) for channel in dapi_stack]
    )

    del dapi_stack
    gc.collect()

    # Rescale downsampled image to uint8
    min_val = downsampled_image.min(axis=(1, 2), keepdims=True)
    max_val = downsampled_image.max(axis=(1, 2), keepdims=True)
    downsampled_image = (downsampled_image - min_val) / (max_val - min_val) * 255
    downsampled_image = downsampled_image.astype(np.uint8)

    tiff.imwrite(output_path, downsampled_image, imagej=True)


def touch(file_path):
    # Check if the file exists
    if os.path.exists(file_path):
        # Update the access and modification times to the current time
        os.utime(file_path, None)
    else:
        # Create an empty file
        with open(file_path, "a"):
            os.utime(file_path, None)


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
        "-d",
        "--dapi_crops",
        type=str,
        default=None,
        required=True,
        nargs="+",
        help="A list of crops (DAPI channel)",
    )
    parser.add_argument(
        "-c",
        "--crops",
        type=str,
        default=None,
        required=True,
        nargs="+",
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
        "-f",
        "--fixed",
        type=str,
        default=None,
        required=True,
        help="Padded full fixed file.",
    )
    parser.add_argument(
        "-m",
        "--moving",
        type=str,
        default=None,
        required=True,
        help="Padded full moving file.",
    )
    parser.add_argument(
        "-df",
        "--downscale_factor",
        type=int,
        default=1,
        required=False,
        help="Factor for image downscaling (e.g. if downscale_factor=2, the resulting image will be half the original resolution)",
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

    original_shape = get_image_file_shape(args.moving, format=".h5")

    matches = []
    for crop_name in args.crops:
        matches.append(
            len(crop_name.split(".")[0].split("_")[-1]) == 64
        )  # Check if filename ends in a 64 characters hash

    if not all(matches):
        crops_files = args.crops
        dapi_crops_files = args.dapi_crops
        cr = load_h5(crops_files[0])

        if not isinstance(cr, int):
            shape = (original_shape[0], original_shape[1])
            save_dapi_stack(
                dapi_crops_files, args.moving, args.fixed, shape, args.overlap_size
            )
    else:
        touch(f"QCNULL_{args.patient_id}.tiff")


if __name__ == "__main__":
    main()
