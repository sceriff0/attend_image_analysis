#!/usr/bin/env python
# Compute affine transformation matrix

import argparse
import gc
import os
import numpy as np
import logging
from utils.io import load_h5
from utils.io import save_pickle, load_pickle
from utils.cropping import reconstruct_image
from utils.mapping import compute_affine_mapping_cv2, apply_mapping
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
        "-c",
        "--channels_to_register",
        type=str,
        default=None,
        required=True,
        help="Pickle file containing list of image channels to register",
    )
    parser.add_argument(
        "-m",
        "--moving_image",
        type=str,
        default=None,
        required=True,
        help="h5 image file",
    )
    parser.add_argument(
        "-f",
        "--fixed_image",
        type=str,
        default=None,
        required=True,
        help="h5 image file",
    )
    parser.add_argument(
        "-ca",
        "--crop_size_affine",
        type=int,
        default=10000,
        required=False,
        help="Size of the crop",
    )
    parser.add_argument(
        "-oa",
        "--overlap_size_affine",
        type=int,
        default=4000,
        required=False,
        help="Size of the overlap",
    )
    parser.add_argument(
        "-cd",
        "--crop_size_diffeo",
        type=int,
        default=2000,
        required=False,
        help="Size of the crop",
    )
    parser.add_argument(
        "-od",
        "--overlap_size_diffeo",
        type=int,
        default=800,
        required=False,
        help="Size of the overlap",
    )

    args = parser.parse_args()
    return args

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
    X, Y, Z = shape
    positions = []

    for start_col in range(0, X - overlap_size, crop_size - overlap_size):
        for start_row in range(0, Y - overlap_size, crop_size - overlap_size):
            # Ensure crop dimensions don't exceed the image dimensions
            end_row = min(start_row + crop_size, Y)
            end_col = min(start_col + crop_size, X)
            positions.append((start_row, end_row, start_col, end_col))

    return positions

def save_stacked_crops(areas, fixed_image_path, moving_image_path, reconstructed_image):
    for area in areas:
        logger.debug(f"Affine - processing crop area: {area}")

        start_row, end_row, start_col, end_col = area

        fixed_crop = load_h5(fixed_image_path, loading_region=area)
        moving_crop = reconstructed_image[start_row:end_row, start_col:end_col, :]
        patient_id = os.path.basename(moving_image_path)
        output_path = f"{start_row}_{start_col}_{patient_id}.pkl"
    
        if len(np.unique(moving_crop)) != 1 and len(np.unique(fixed_crop)) != 1:
            logger.debug(f"Affine - computing transformation: {area}")
            matrix = compute_affine_mapping_cv2(
                y=fixed_crop[:,:,-1].squeeze(), 
                x=moving_crop[:,:,-1].squeeze()
            )
            logger.debug(f"Affine - computed transformation: {area}")

            logger.debug(f"Affine - saving crop: {output_path}")
            save_pickle(
                (  
                    fixed_crop,
                    apply_mapping(matrix, moving_crop, method="cv2"),
                ),
                output_path,
            )
            logger.debug(f"Affine - saved crop: {output_path}")
        else:
            logger.debug(f"Affine - saving crop: {output_path}")
            save_pickle(
                (  
                    fixed_crop,
                    moving_crop
                ),
                output_path,
            )
            logger.debug(f"Affine - saved crop: {output_path}")

def main():
    handler = logging.FileHandler('/hpcnfs/scratch/DIMA/chiodin/repositories/attend_image_analysis/LOG.log')
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    args = _parse_args()

    logger.debug(f"Fixed image: {args.fixed_image}, Moving image: {args.moving_image}")

    channels_to_register = load_pickle(args.channels_to_register)

    if channels_to_register:
        moving = load_h5(args.moving_image)
        fixed = load_h5(args.fixed_image)
        moving_shape = moving.shape

        matrix = compute_affine_mapping_cv2(
            y=fixed[:, :, -1].squeeze(), 
            x=moving[:, :, -1].squeeze()
        )

        del fixed, moving
        gc.collect()

        reconstructed_image = np.zeros(moving_shape, dtype='uint16')
        areas_affine = get_crops_positions(moving_shape, args.crop_size_affine, args.overlap_size_affine)
        for area in areas_affine:
            position = (area[0], area[2])
            crop = apply_mapping(
                matrix, 
                load_h5(args.moving_image, loading_region=area), 
                method="cv2"
            )    
            reconstructed_image = reconstruct_image(
                reconstructed_image, 
                crop, 
                position, 
                moving_shape, 
                args.overlap_size_affine
            )

        areas_diffeo = get_crops_positions(moving_shape, args.crop_size_diffeo, args.overlap_size_diffeo)
        save_stacked_crops(areas_diffeo, args.fixed_image, args.moving_image, reconstructed_image)

        del reconstructed_image
        gc.collect()

    else:
        save_pickle([], f"0_0_{args.patient_id}.pkl")


if __name__ == "__main__":
    main()

