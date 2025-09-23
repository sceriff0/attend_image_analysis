#!/usr/bin/env python
# Compute affine transformation matrix

import argparse
import gc
import os
import numpy as np
import logging
import hashlib
from utils.io import load_h5, save_h5
from utils.io import save_pickle, load_pickle
from utils.cropping import reconstruct_image
from utils.mapping import compute_affine_mapping_cv2, compute_diffeomorphic_mapping_dipy, apply_mapping
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
    parser.add_argument(
        "-l",
        "--log_file",
        type=str,
        required=False,
        help="Path to log file.",
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
    Y, X, Z = shape
    positions = []

    for start_col in range(0, X - overlap_size, crop_size - overlap_size):
        for start_row in range(0, Y - overlap_size, crop_size - overlap_size):
            # Ensure crop dimensions don't exceed the image dimensions
            end_row = min(start_row + crop_size, Y)
            end_col = min(start_col + crop_size, X)
            positions.append((start_row, end_row, start_col, end_col))

    return positions

def save_stacked_crops(areas, fixed_image_path, moving_image_path, reconstructed_image, channels_to_register, 
                       current_channels_to_register_no_dapi, moving_channels_no_dapi):
    for area in areas:
        logger.debug(f"Affine - processing crop area: {area}")

        start_row, end_row, start_col, end_col = area

        fixed_crop = load_h5(fixed_image_path, loading_region=area)
        moving_crop = reconstructed_image[start_row:end_row, start_col:end_col, :]
        patient_id = os.path.basename(moving_image_path)

        crop_name = crop_id_pos + current_channels_to_register_no_dapi[::-1]   
        crop_name = '_'.join(crop_name)
        output_path = f"registered_{crop_name}.h5"
        output_path = output_path.replace('padded_', '')
    
        crop_id_pos = [str(start_row), str(start_col)]
        output_path_dapi = f"qc_{'_'.join(crop_id_pos)}_DAPI.h5"
        output_path_dapi = output_path_dapi.replace('padded_', '')

        if len(np.unique(moving_crop[:,:,-1])) != 1 and len(np.unique(fixed_crop[:,:,-1])) != 1:
            logger.debug(f"Affine - computing transformation: {area}")
            try:
                matrix = compute_affine_mapping_cv2(
                    y=fixed_crop[:,:,-1].squeeze(),
                    x=moving_crop[:,:,-1].squeeze()
                )

                moving_crop = apply_mapping(matrix, moving_crop, method="cv2")

                if current_channels_to_register_no_dapi:
                    if any([e for e in current_channels_to_register_no_dapi if e in channels_to_register]):
                        if len(np.unique(moving_crop[:,:,-1])) != 1 and len(np.unique(fixed_crop[:,:,-1])) != 1:
                            logger.debug(f"Computing mapping: {crop_id_pos}")
                            mapping = compute_diffeomorphic_mapping_dipy(
                                y=fixed_crop[:, :, -1].squeeze(), 
                                x=moving_crop[:, :, -1].squeeze()
                            )
                        # Save registered dapi channel for quality control
                            save_h5(
                                np.squeeze(apply_mapping(mapping, moving_crop[:, :, -1])), 
                                output_path_dapi
                            )

                            logger.debug(f"Applying mapping: {crop_id_pos}")
                            registered_images = []

                            for idx, ch in enumerate(moving_channels_no_dapi):
                                if ch in current_channels_to_register_no_dapi:
                                    registered_images.append(apply_mapping(mapping, moving_crop[:, :, idx]))

                            registered_images = np.stack(registered_images, axis=-1)

                            logger.debug(f"Saving registered image: {crop_id_pos}")
                            save_h5(
                                registered_images, 
                                output_path
                            )

            except:
                #if np.mean(fixed_crop!=0) < 0.1 or np.mean(moving_crop!=0) < 0.1:
                save_pickle(
                    (
                        fixed_crop,
                        moving_crop
                    ),
                    output_path,
                )
                #else:
                #    error_message = "Error in affine transformation. Check the input image."
                #    logger.debug("Raising ValueError: %s", error_message)
                #    raise ValueError(error_message)
                
        elif len(np.unique(moving_crop[:,:,-1])) == 1 or len(np.unique(fixed_crop[:,:,-1])) == 1:
            logger.debug(f"Affine - skipping crop: {output_path}")
            save_pickle(
                (
                    fixed_crop,
                    moving_crop
                ),
                output_path,
            )
        else:
            error_message = "Error in affine transformation. Check the input image."
            logger.debug("Raising ValueError: %s", error_message)
            raise ValueError(error_message)


def main():
    args = _parse_args()

    handler = logging.FileHandler(args.log_file)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    logger.debug(f"Fixed image: {args.fixed_image}, Moving image: {args.moving_image}")

    moving_channels = os.path.basename(args.moving_image) \
        .split('.')[0] \
        .split('_')[2:][::-1] 
    
    logger.debug(f'DIFFEOMORPHIC - MOVING CHANNELS: {moving_channels}')

    moving_channels_no_dapi = [ch for ch in moving_channels if ch != 'DAPI']

    channels_to_register = load_pickle(args.channels_to_register)
    current_channels_to_register = remove_lowercase_channels(moving_channels)
    current_channels_to_register_no_dapi = [ch for ch in current_channels_to_register if ch != 'DAPI']

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
            logger.debug(f"AREA: {area}")
            position = (area[0], area[2])
            crop = apply_mapping(
                matrix, 
                load_h5(args.moving_image, loading_region=area), # uint16 
                method="cv2"
            )

            logger.debug(f"AFFINE: transformed CROP image dtype : {crop.dtype}")  
            logger.debug(f"AFFINE: transformed CROP image min and max: {np.min(crop)} , {np.max(crop)}")  
            
            crop = crop.astype('uint16')  # Ensure the crop is in uint16 format
            logger.debug(f"AFFINE: transformed CROP image dtype : {crop.dtype}")
            logger.debug(f"AFFINE: transformed CROP image min and max: {np.min(crop)} , {np.max(crop)}")  
            # logger.debug(f"AFFINE: all zeros in CROP: {np.all(crop == 0)}")

            logger.debug(f"CROP SHAPE: {crop.shape}")
            reconstructed_image = reconstruct_image(
                reconstructed_image, 
                crop, 
                position, 
                moving_shape, 
                args.overlap_size_affine
            )
            logger.debug(f"AFFINE: RECONSTRUCTED IMAGE image dtype : {reconstructed_image.dtype}")  

        areas_diffeo = get_crops_positions(moving_shape, args.crop_size_diffeo, args.overlap_size_diffeo)

        save_stacked_crops(areas_diffeo, args.fixed_image, args.moving_image, reconstructed_image, channels_to_register, 
                           current_channels_to_register_no_dapi, moving_channels_no_dapi)

        del reconstructed_image
        gc.collect()

    else:
        save_pickle([], f"0_0_{args.patient_id}.pkl")


if __name__ == "__main__":
    main()
