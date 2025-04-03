#!/usr/bin/env python

import numpy as np
import os

os.environ["NUMBA_CACHE_DIR"] = "/tmp/numba_cache"

import argparse
import tifffile as tiff
import logging
from cellpose.utils import masks_to_outlines
from skimage.transform import rescale
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
        "--dapi_image",
        type=str,
        default=None,
        required=True,
        help="List of tiff single channel images.",
    )
    parser.add_argument(
        "--segmentation_mask",
        type=str,
        default=None,
        required=True,
        help="Tif file containing segmentation mask",
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

    dapi_image = args.dapi_image
    segmentation_mask = args.segmentation_mask

    dapi = tiff.imread(dapi_image).astype("float32")
    mask = tiff.imread(segmentation_mask).astype("float32")
    outlines = masks_to_outlines(mask)
    outlines = np.array(outlines, dtype="float32")

    output_array = np.stack((dapi, mask, outlines), axis=0).astype("float32")

    # Normalize each channel independently
    output_array = normalize_image(output_array)

    # Downsample each channel separately
    output_array = np.array(
        [rescale(channel, scale=0.25, anti_aliasing=True) for channel in output_array]
    )

    output_array = rescale_to_uint8(output_array)

    tiff.imwrite(f"QC_segmentation_{dapi_image}", output_array)
