#!/usr/bin/env python

import numpy as np
import os

# os.environ["NUMBA_CACHE_DIR"] = "/tmp/numba_cache"

import argparse
import tifffile as tiff
from cellpose.utils import masks_to_outlines
from skimage.transform import rescale


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
        "--output_dir",
        type=str,
        default=None,
        required=True,
        help="Tif file containing segmentation mask",
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = _parse_args()

    dapi_image = args.dapi_image
    membrane_image = args.membrane_image
    segmentation_mask = args.segmentation_mask


    dapi = tiff.imread(dapi_image).astype("float32")
    mask = tiff.imread(segmentation_mask).astype("float32")
    outlines = masks_to_outlines(mask)
    outlines = np.array(outlines, dtype="float32")
    

    if membrane_image is not None:
        membrane = tiff.imread(membrane_image).astype("float32")
        output_array = np.stack((dapi, membrane, mask, outlines), axis=0).astype("float32")
        output_array = normalize_image(output_array)
        output_array = np.array([
            rescale(output_array[0], scale=0.5, anti_aliasing=True),  
            rescale(output_array[1], scale=0.5, anti_aliasing=True),
            rescale(output_array[2], scale=0.5, anti_aliasing=False),  
            rescale(output_array[3], scale=0.5, anti_aliasing=False)  
        ])
        
    else:
        output_array = np.stack((dapi, mask, outlines), axis=0).astype("float32")
        output_array = normalize_image(output_array)
        output_array = np.array([
            rescale(output_array[0], scale=0.5, anti_aliasing=True), 
            rescale(output_array[1], scale=0.5, anti_aliasing=False),
            rescale(output_array[2], scale=0.5, anti_aliasing=False)
        ])

    output_array = rescale_to_uint8(output_array)
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, f"QC_segmentation_{dapi_image}")
    pixel_microns = 0.34533768547788
    tiff.imwrite(
        output_path, 
        output_array, 
        imagej=True, 
        resolution=(1/pixel_microns, 1/pixel_microns), 
        metadata={'unit': 'um', 'axes': 'CYX', 'mode': 'composite'}
    )


    