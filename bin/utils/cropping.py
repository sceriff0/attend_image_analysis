#!/usr/bin/env python

import logging
import os
import numpy as np
from utils import logging_config
from utils.io import load_h5

# Set up logging configuration
logging_config.setup_logging()
logger = logging.getLogger(__name__)

def create_crops(image, crop_size, overlap_size):
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
    X, Y, Z = image.shape
    crops = []
    positions = []

    for x in range(0, X - overlap_size, crop_size - overlap_size):
        for y in range(0, Y - overlap_size, crop_size - overlap_size):
            # Ensure crop dimensions don't exceed the image dimensions
            x_end = min(x + crop_size, X)
            y_end = min(y + crop_size, Y)
            crop = image[x:x_end, y:y_end, :]
            crops.append(crop)
            positions.append((x, y))

    return crops, positions

def get_crop_areas(shape, n_crops):
    if int(np.sqrt(n_crops)) != np.sqrt(n_crops):
            ValueError('Argument `n_crops` must be a number whose square root is an integer.')

    n_crops = int(np.sqrt(n_crops))
    
    # Calculate row and column step size based on n_crops
    row_step = shape[0] // n_crops
    col_step = shape[1] // n_crops

    # Generate export areas dynamically for n_crops x n_crops grid
    crop_areas = []
    for i in range(n_crops):
        for j in range(n_crops):
            top = i * row_step
            bottom = (i + 1) * row_step if i < n_crops - 1 else shape[0]
            left = j * col_step
            right = (j + 1) * col_step if j < n_crops - 1 else shape[1]
            crop_areas.append((top, bottom, left, right))

    return crop_areas

def reconstruct_image(reconstructed, crop, position, original_shape, overlap_size):
    """
    Reconstruct the original image from overlapping crops.

    Args:
        reconstructed (numpy.ndarray): The array to reconstruct the image into.
        crop (numpy.ndarray): The crop to be placed in the reconstructed image.
        position (tuple): The top-left corner indices (x, y) for the crop.
        original_shape (tuple): Shape of the original image (X, Y, Z) or (X, Y).
        overlap_size (int): The overlap size along X and Y.

    Returns:
        numpy.ndarray: The reconstructed image of dtype uint16.
    """
    x, y = position
    c_x, c_y = crop.shape[:2]
    X, Y = original_shape[:2]

    def calculate_bounds(start, crop_dim, total_dim, overlap):
        if start == 0:
            start_idx = start
            end_idx = start + crop_dim - overlap // 2
            crop_slice = slice(0, crop_dim - overlap // 2)
        elif start == total_dim - crop_dim:
            start_idx = start + overlap // 2
            end_idx = start + crop_dim
            crop_slice = slice(overlap // 2, crop_dim)
        else:
            start_idx = start + overlap // 2
            end_idx = start + crop_dim - overlap // 2
            crop_slice = slice(overlap // 2, crop_dim - overlap // 2)
        return start_idx, end_idx, crop_slice

    # Calculate bounds for y-axis
    y_start, y_end, y_slice = calculate_bounds(y, c_y, Y, overlap_size)
    # Calculate bounds for x-axis
    x_start, x_end, x_slice = calculate_bounds(x, c_x, X, overlap_size)

    # Adjust crop based on dimensionality
    if len(original_shape) == 3:  # 3D image
        crop = crop[x_slice, y_slice, :]
        reconstructed[x_start:x_end, y_start:y_end, :] += crop
    elif len(original_shape) == 2:  # 2D image
        crop = crop[x_slice, y_slice]
        reconstructed[x_start:x_end, y_start:y_end] += crop

    return reconstructed

def image_reconstruction_loop(crops_files, shape, overlap_size, dtype=None):
    if dtype is None:
        dtype = load_h5(crops_files[0], shape='YX').dtype

    logger.debug(f"Reconstruction dtype: {dtype}")

    logger.info(f"Reconstruction shape: {shape}")
    reconstructed_image = np.zeros(shape, dtype=dtype)

    for crop_file in crops_files:
        logger.info(f"Loading crop: {crop_file}")
        crop = load_h5(crop_file, shape='YX').astype(dtype)
        logger.info(f"Loaded crop: {crop_file}, Shape: {crop.shape}")

        if len(shape) > len(crop.shape):
            crop = np.expand_dims(crop, axis=2)

        x, y = map(int, os.path.basename(crop_file).split("_")[1:3])
        position = (x, y)
        reconstructed_image = reconstruct_image(reconstructed_image, crop, position, (shape[0], shape[1]), overlap_size)
    
    return reconstructed_image
