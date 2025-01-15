#!/usr/bin/env python
# Compute affine transformation matrix

import logging
import argparse
import os
import numpy as np
import re
import gc
import matplotlib.pyplot as plt
from skimage.transform import rescale
from utils.io import load_h5, save_h5
from utils.cropping import reconstruct_image
from utils.metadata_tools import get_image_file_shape
from utils import logging_config

# Set up logging configuration
logging_config.setup_logging()
logger = logging.getLogger(__name__)

def are_all_alphabetic_lowercase(string):
    # Filter alphabetic characters and check if all are lowercase
    return all(char.islower() for char in string if char.isalpha())

def remove_lowercase_channels(channels):
    filtered_channels = []
    for ch in channels:
        if not are_all_alphabetic_lowercase(ch):
            filtered_channels.append(ch)
    return filtered_channels

def image_reconstruction_loop(crops_files, shape, overlap_size):
    reconstructed_image = np.zeros(shape, dtype='float32')

    for crop_file in crops_files:
        crop = load_h5(crop_file)
        logger.info(f"Loaded crop: {crop_file}")

        x, y = map(int, os.path.basename(crop_file).split("_")[1:3])
        position = (x, y)
        reconstructed_image = reconstruct_image(reconstructed_image, crop, position, (shape[0], shape[1]), overlap_size)
    
    return reconstructed_image

def save_quality_control_plot(output_path, dapi_crops_files, shape, overlap_size, fixed, downscale_factor = 0.5):
        reconstructed_image = image_reconstruction_loop(dapi_crops_files, shape, overlap_size)
        reconstructed_image = rescale(
            reconstructed_image,
            scale=downscale_factor,
            anti_aliasing=True
        )

        fixed = load_h5(fixed, channels_to_load=-1)
        fixed = rescale(
            fixed,
            scale=downscale_factor,
            anti_aliasing=True
        )
        fixed = np.squeeze(fixed)

        plt.figure(figsize=(20, 20))
        plt.imshow(reconstructed_image, cmap='Reds', alpha=0.8)
        plt.imshow(fixed, cmap='Greens', alpha=0.6) 
        plt.savefig(output_path, format="jpg", dpi=300)

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
    args = parser.parse_args()
    return args


def main():
    handler = logging.FileHandler('/hpcnfs/scratch/DIMA/chiodin/repositories/attend_image_analysis/LOG.log')
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)


    args = _parse_args()
    original_shape = get_image_file_shape(args.moving, format='.h5')

    pattern = r'^registered_[a-zA-Z0-9]+_[a-fA-F0-9]{64}.h5$'

    matches = []
    for crop_name in args.crops:
        matches.append(bool(re.match(pattern, crop_name)))

    if not all(matches):
        crops_files = args.crops
        dapi_crops_files = args.crops
        cr = load_h5(crops_files[0])

        logger.debug(f'OBJECT: {cr}')

        if not isinstance(cr, int):
            shape = (original_shape[0], original_shape[1], cr.shape[2])

            plot_output = f"registered_{os.path.basename(args.moving).split('.')[0]}_overlay.jpg"
            save_quality_control_plot(plot_output, dapi_crops_files, shape, args.overlap_size, args.fixed)

            reconstructed_image = image_reconstruction_loop(crops_files, original_shape, args.overlap_size)

            moving_channels = os.path.basename(args.moving)\
                .split('.')[0] \
                .split('_')[2:] \
                [::-1] # Select first two channels (omit DAPI) and reverse list
            
            fixed_channels = os.path.basename(args.fixed) \
                .split('.')[0] \
                .split('_')[1:] \
                [::-1] # Select all channels and reverse list
            
            moving_channels_to_export = remove_lowercase_channels(moving_channels)
            moving_channels_to_export_no_dapi = [ch for ch in moving_channels_to_export if ch != 'DAPI']
            fixed_channels_to_export = remove_lowercase_channels(fixed_channels)

            # Save moving channels
            for idx, ch in enumerate(moving_channels_to_export_no_dapi):
                save_h5(
                    np.expand_dims(reconstructed_image[:,:,idx], axis=0).astype(np.float32), 
                    f"registered_{args.patient_id}_{ch}.h5"
                )
            
            # Save fixed channels
            for idx, ch in enumerate(fixed_channels_to_export):
                image = load_h5(args.fixed, channels_to_load=idx)
                image = image.astype(np.float32)
                image = np.expand_dims(image, axis=0)
                save_h5(
                    image, 
                    f"registered_{args.patient_id}_{ch}.h5"
                )
    else:
        save_h5(
                0, 
                f"registered_{args.patient_id}.h5"
            )



if __name__ == "__main__":
    main()
