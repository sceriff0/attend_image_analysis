#!/usr/bin/env python
# Compute affine transformation matrix

import argparse
import gc
import h5py
import nd2
import os
import pickle

import numpy as np

import cv2

from dipy.align.imwarp import SymmetricDiffeomorphicRegistration
from dipy.align.metrics import CCMetric


def _parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--crop_image",
        type=str,
        default=None,
        required=True,
        help="pickle containing fixed and moving crops, in this order.",
    )

    args = parser.parse_args()
    return args


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


def load_pickle(path):
    # Open the file in binary read mode
    with open(path, "rb") as file:
        # Deserialize the object from the file
        loaded_data = pickle.load(file)

    return loaded_data


def save_pickle(object, path):
    # Open a file in binary write mode
    with open(path, "wb") as file:
        # Serialize the object and write it to the file
        pickle.dump(object, file)


def compute_diffeomorphic_mapping_dipy(
    y: np.ndarray, x: np.ndarray, sigma_diff=5, radius=4
):
    """
    Compute diffeomorphic mapping using DIPY.

    Parameters:
        y (ndarray): Reference image.
        x (ndarray): Moving image to be registered.
        sigma_diff (int, optional): Standard deviation for the CCMetric. Default is 20.
        radius (int, optional): Radius for the CCMetric. Default is 20.

    Returns:
        mapping: A mapping object containing the transformation information.
    """
    # Check if both images have the same shape
    if y.shape != x.shape:
        raise ValueError(
            "Reference image (y) and moving image (x) must have the same shape."
        )

    # Define the metric and create the Symmetric Diffeomorphic Registration object
    metric = CCMetric(2, sigma_diff=sigma_diff, radius=radius)
    sdr = SymmetricDiffeomorphicRegistration(metric, opt_tol=1e-03, inv_tol=0.1)

    # Perform the diffeomorphic registration using the pre-alignment from affine registration
    mapping = sdr.optimize(y, x)

    return mapping


def main():
    args = _parse_args()

    fixed, moving = load_pickle(args.crop_image)

    if len(np.unique(moving)) != 1:
        mapping = compute_diffeomorphic_mapping_dipy(
            y=fixed[:, :, 2].squeeze(), x=moving[:, :, 2].squeeze()
        )

        registered_images = []
        for ch in range(moving.shape[-1]):
            registered_images.append(apply_mapping(mapping, moving[:, :, ch]))
        registered_images = np.stack(registered_images, axis=-1)

        registered_images.astype(np.uint16)
        save_pickle(
            registered_images, f"registered_{os.path.basename(args.crop_image)}"
        )
    else:
        save_pickle(moving, f"registered_{os.path.basename(args.crop_image)}")


if __name__ == "__main__":
    main()
