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

def save_quality_control_plot(dapi_crops_files, shape, overlap_size, fixed_path, moving_path, downscale_factor = 2):
        def save_overlay_plot(image_1, image_2, output_path):
            plt.figure(figsize=(20, 20))
            plt.imshow(image_1, cmap='Reds', alpha=0.8)
            plt.imshow(image_2, cmap='Greens', alpha=0.6)
            if not os.path.exists(output_path):
                plt.savefig(output_path, format="jpg", dpi=300)
        
        def save_single_plot(image, output_path):
            plt.figure(figsize=(20, 20))
            plt.imshow(image, cmap='Reds')
            if not os.path.exists(output_path):
                plt.savefig(output_path, format="jpg", dpi=300)

        crop_areas = get_crop_areas(shape, 4)
        downscale_factor = 1 / downscale_factor
        if downscale_factor !=1 :
            reconstructed_image = image_reconstruction_loop(dapi_crops_files, shape, overlap_size)
            reconstructed_image = rescale(
                reconstructed_image,
                scale=downscale_factor,
                anti_aliasing=True
            )

            fixed = load_h5(fixed_path, channels_to_load=-1)
            fixed = rescale(
                fixed,
                scale=downscale_factor,
                anti_aliasing=True
            )
            fixed = np.squeeze(fixed)
        else:
            reconstructed_image = image_reconstruction_loop(dapi_crops_files, shape, overlap_size)
            fixed = load_h5(fixed_path, channels_to_load=-1)
            fixed = np.squeeze(fixed)

        for area in crop_areas:
            output_path_overlay = f"registered_DAPI_overlay_{os.path.basename(moving_path).split('.')[0]}_{area[0]}_{area[1]}_{area[2]}_{area[3]}.jpg"
            output_path_single_1 = f"registered_DAPI_{os.path.basename(moving_path).split('.')[0]}_{area[0]}_{area[1]}_{area[2]}_{area[3]}.jpg"
            output_path_single_2 = f"registered_DAPI_{os.path.basename(fixed_path).split('.')[0]}_{area[0]}_{area[1]}_{area[2]}_{area[3]}.jpg"

            logger.debug(f"Saving {output_path_overlay}")
            save_overlay_plot(
                reconstructed_image[area[0]:area[1], area[2]:area[3]], 
                fixed[area[0]:area[1], area[2]:area[3]],
                output_path_overlay
            )
            logger.debug(f"Saving {output_path_single_1}")
            save_single_plot(
                reconstructed_image[area[0]:area[1], area[2]:area[3]], 
                output_path_single_1
            )
            logger.debug(f"Saving {output_path_single_2}")
            save_single_plot(
                fixed[area[0]:area[1], area[2]:area[3]], 
                output_path_single_2
            )

def save_dapi_channels_tiff(dapi_crops_files, moving_path, fixed_path, shape, overlap_size):
    reconstructed_image = np.squeeze(
        image_reconstruction_loop(dapi_crops_files, shape, overlap_size)
    )
    reconstructed_image = rescale(reconstructed_image, scale=0.25, anti_aliasing=True)
    outname = os.path.basename(moving_path).split(".")[0]
    logger.debug(f'SAVING DAPI CHANNEL (MOVING): registered_{outname}__DAPI_ONLY.tiff')
    logger.debug(f'DAPI CHANNEL SHAPE (MOVING): {reconstructed_image.shape}')
    tiff.imwrite(f'registered_DAPI_{outname}.tiff', reconstructed_image)
    del reconstructed_image
    gc.collect()

    fixed_dapi = np.squeeze(
        load_h5(fixed_path, channels_to_load=-1)
    )
    fixed_dapi = rescale(fixed_dapi, scale=0.25, anti_aliasing=True)
    outname = os.path.basename(fixed_path).split(".")[0]
    logger.debug(f'SAVING DAPI CHANNEL (FIXED): registered_channel_{outname}__DAPI_ONLY.tiff')
    logger.debug(f'DAPI CHANNEL SHAPE (FIXED): {fixed_dapi.shape}')
    if not os.path.exists(f'registered_{outname}__DAPI_ONLY.tiff'):
        tiff.imwrite(f'registered_DAPI_{outname}.tiff', fixed_dapi)
    

def touch(file_path):
    # Check if the file exists
    if os.path.exists(file_path):
        # Update the access and modification times to the current time
        os.utime(file_path, None)
    else:
        # Create an empty file
        with open(file_path, 'a'):
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
        nargs='+',
        help="A list of crops (DAPI channel)",
    )
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
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    original_shape = get_image_file_shape(args.moving, format='.h5')

    pattern = r'^registered_[a-zA-Z0-9]+_[a-fA-F0-9]{64}.h5$'

    matches = []
    for crop_name in args.crops:
        matches.append(bool(re.match(pattern, crop_name)))

    if not all(matches):
        crops_files = args.crops
        dapi_crops_files = args.dapi_crops
        cr = load_h5(crops_files[0])

        if not isinstance(cr, int):
            shape = (original_shape[0], original_shape[1], cr.shape[2])
            save_dapi_channels_tiff(dapi_crops_files, args.moving, args.fixed, shape, args.overlap_size)
            save_quality_control_plot(dapi_crops_files, shape, args.overlap_size, args.fixed, args.moving, args.downscale_factor)
    else:
        touch(f"registered_NULL_{args.patient_id}.jpg")

if __name__ == "__main__":
    main()
