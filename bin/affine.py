#!/usr/bin/env python
# Compute affine transformation matrix

import argparse
import ast
import gc
import h5py
import nd2
import os
import pickle

import numpy as np

import cv2


def _parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--moving_image",
        type=str,
        default=None,
        required=True,
        help="nd5 image file",
    )
    parser.add_argument(
        "-f",
        "--fixed_image",
        type=str,
        default=None,
        required=True,
        help="nd5 image file",
    )
    parser.add_argument(
        "-c",
        "--crop_size",
        type=int,
        default=8000,
        required=False,
        help="Size of the crop",
    )
    parser.add_argument(
        "-o",
        "--overlap_size",
        type=int,
        default=2000,
        required=False,
        help="Size of the overlap",
    )

    args = parser.parse_args()
    return args


def load_h5(path, channels_to_load=None):
    with h5py.File(path, "r") as hdf5_file:
        data = hdf5_file["dataset"]

        # Select channels if channels_to_load is provided
        if channels_to_load is not None:
            data = data[:, :, channels_to_load].squeeze()
        else:
            data = data[:, :, :]

    return data


def compute_affine_mapping_cv2(
    y: np.ndarray, x: np.ndarray, crop=False, crop_size=None, n_features=2000
):
    """
    Compute affine mapping using OpenCV.

    Parameters:
        y (ndarray): Reference image.
        x (ndarray): Moving image to be registered.
        crop (bool, optional): Whether to crop the images before processing. Default is True.
        crop_size (int, optional): Size of the crop. Default is 4000.
        n_features (int, optional): Maximum number of features to detect. Default is 2000.

    Returns:
        matrix (ndarray): Affine transformation matrix.
    """
    # Crop the images if specified and normalize them to 8-bit (0-255) for feature detection
    if crop and crop_size is not None:
        mid_y = np.array(y.shape) // 2
        mid_x = np.array(x.shape) // 2
        y = cv2.normalize(
            y[
                (mid_y[0] - crop_size) : (mid_y[0] + crop_size),
                (mid_y[1] - crop_size) : (mid_y[1] + crop_size),
            ],
            None,
            0,
            255,
            cv2.NORM_MINMAX,
        ).astype(np.uint8)
        x = cv2.normalize(
            x[
                (mid_x[0] - crop_size) : (mid_x[0] + crop_size),
                (mid_x[1] - crop_size) : (mid_x[1] + crop_size),
            ],
            None,
            0,
            255,
            cv2.NORM_MINMAX,
        ).astype(np.uint8)

    y = cv2.normalize(y, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    x = cv2.normalize(x, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Detect ORB keypoints and descriptors
    orb = cv2.ORB_create(fastThreshold=0, edgeThreshold=0, nfeatures=n_features)

    # Compute keypoints and descriptors for both images
    keypoints1, descriptors1 = orb.detectAndCompute(y, None)
    keypoints2, descriptors2 = orb.detectAndCompute(x, None)

    if descriptors1 is None or descriptors2 is None:
        raise ValueError("One of the descriptors is empty")

    # Convert descriptors to uint8 if they are not already in that format
    if descriptors1 is not None and descriptors1.dtype != np.uint8:
        descriptors1 = descriptors1.astype(np.uint8)

    if descriptors2 is not None and descriptors2.dtype != np.uint8:
        descriptors2 = descriptors2.astype(np.uint8)

    # Match descriptors using BFMatcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)
    matches = sorted(matches, key=lambda x: x.distance)

    # Extract location of good matches
    points1 = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    points2 = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    # Compute affine transformation matrix from matched points
    matrix, mask = cv2.estimateAffinePartial2D(points2, points1)

    return matrix


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
        original_shape, dtype=np.uint16
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


def apply_mapping(mapping, x, method="dipy"):
    """
    Apply mapping to the image.

    Parameters:
        mapping: A mapping object from either the DIPY or the OpenCV package.
        x (ndarray): 2-dimensional numpy array to transform.
        method (str, optional): Method used for mapping. Either 'cv2' or 'dipy'. Default is 'dipy'.

    Returns:
        mapped (ndarray): Transformed image as a 2D numpy array.
    """
    # Validate the method parameter
    if method not in ["cv2", "dipy"]:
        raise ValueError("Invalid method specified. Choose either 'cv2' or 'dipy'.")

    # Apply the mapping based on the selected method
    if method == "dipy":
        mapped = mapping.transform(x)
    elif method == "cv2":
        height, width = x.shape[:2]
        mapped = cv2.warpAffine(x, mapping, (width, height))

    return mapped


def save_h5(data, path, chunks=None):
    # Save the NumPy array to an HDF5 file
    with h5py.File(path, "w") as hdf5_file:
        hdf5_file.create_dataset("dataset", data=data, chunks=chunks)
        hdf5_file.flush()


def save_pickle(object, path):
    # Open a file in binary write mode
    with open(path, "wb") as file:
        # Serialize the object and write it to the file
        pickle.dump(object, file)


def create_crops2save(fixed, moving, crop_size, overlap_size, outname):
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
    X, Y, Z = fixed.shape
    assert fixed.shape == moving.shape

    crops = []
    positions = []

    for x in range(0, X - overlap_size, crop_size - overlap_size):
        for y in range(0, Y - overlap_size, crop_size - overlap_size):
            # Ensure crop dimensions don't exceed the image dimensions
            x_end = min(x + crop_size, X)
            y_end = min(y + crop_size, Y)
            save_pickle(
                (fixed[x:x_end, y:y_end, :], moving[x:x_end, y:y_end, :]),
                f"{x}_{y}_{os.path.basename(outname)}.pkl",
            )


def main():
    args = _parse_args()

    moving = load_h5(args.moving_image)
    fixed = load_h5(args.fixed_image)
    moving_shape = moving.shape

    matrix = compute_affine_mapping_cv2(
        y=fixed[:, :, 2].squeeze(), x=moving[:, :, 2].squeeze()
    )

    del fixed
    gc.collect()
    crops, positions = create_crops(moving, args.crop_size, args.overlap_size)

    del moving
    gc.collect()

    registered_crops = []

    for crop in crops:
        registered_crops.append(apply_mapping(matrix, crop, method="cv2"))

    del crops, crop
    gc.collect()

    reconstructed_image = reconstruct_image(
        registered_crops,
        positions,
        original_shape=moving_shape,
        crop_size=args.crop_size,
        overlap_size=args.overlap_size,
    )

    del registered_crops
    gc.collect()

    fixed = load_h5(args.fixed_image)
    create_crops2save(
        fixed,
        reconstructed_image,
        2000,
        200,
        outname=args.moving_image,
    )


if __name__ == "__main__":
    main()
