#!/usr/bin/env python

import numpy as np
import gc

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


def reconstruct_image(crops, positions, original_shape, crop_size, overlap_size):
    """
    Reconstruct the original image from overlapping crops.

    Args:
        crops (list): List of crops.
        positions (list): List of top-left corner indices for each crop.
        original_shape (tuple): Shape of the original image (X, Y, Z).
        crop_size (int): The size of the crops along X and Y.
        overlap_size (int): The overlap size along X and Y.

    Returns:
        numpy.ndarray: The reconstructed image of dtype uint16.
    """
    X, Y, Z = original_shape
    reconstructed = np.zeros(
        original_shape, dtype=crops[0].dtype
    )  # Use uint16 for intermediate sums

    for crop, (x, y) in zip(crops, positions):

        # Shape crops
        c_x, c_y, _ = crop.shape
        # border
        if y == 0:
            y_start = y
            y_end = y + crop.shape[1] - overlap_size // 2
            crop = crop[:, : (crop.shape[1] - overlap_size // 2), :]

        elif y == Y - c_y:  # border
            y_start = y + overlap_size // 2
            y_end = y + crop.shape[1]
            crop = crop[:, (overlap_size // 2) :, :]

        else:  # no border
            y_start = y + overlap_size // 2
            y_end = y + crop.shape[1] - overlap_size // 2
            crop = crop[:, (overlap_size // 2) : (crop.shape[1] - overlap_size // 2), :]

        # border
        if x == 0:
            x_start = x
            x_end = x + crop.shape[0] - overlap_size // 2
            crop = crop[: (crop.shape[0] - overlap_size // 2), :, :]
        elif x == X - c_x:  # border
            x_start = x + overlap_size // 2
            x_end = x + crop.shape[0]
            crop = crop[(overlap_size // 2) :, :, :]
        else:  # no border
            x_start = x + overlap_size // 2
            x_end = x + crop.shape[0] - overlap_size // 2
            crop = crop[(overlap_size // 2) : (crop.shape[0] - overlap_size // 2), :, :]
        reconstructed[x_start:x_end, y_start:y_end, :] += crop
        del crop
        gc.collect()
    # Convert back to uint16
    return reconstructed