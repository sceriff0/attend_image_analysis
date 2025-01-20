#!/usr/bin/env python

import logging
from utils import logging_config

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