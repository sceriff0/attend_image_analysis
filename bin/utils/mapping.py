#!/usr/bin/env python

import cv2
import numpy as np
import cupy as cp
from cudipy.align.imwarp import SymmetricDiffeomorphicRegistration
from cudipy.align.metrics import CCMetric


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


def compute_affine_mapping_cv2(
    y: np.ndarray, x: np.ndarray, n_features=2000
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
    y = cv2.normalize(y, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    x = cv2.normalize(x, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Detect ORB keypoints and descriptors
    orb = cv2.ORB_create(fastThreshold=0, edgeThreshold=0, nfeatures=n_features)

    # Compute keypoints and descriptors for both images
    keypoints1, descriptors1 = orb.detectAndCompute(y, None)
    keypoints2, descriptors2 = orb.detectAndCompute(x, None)

    if descriptors1 is None:
        raise ValueError("Object 'descriptors1' is None")
    elif descriptors2 is None:
        raise ValueError("Object 'descriptors2' is None")


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


def compute_diffeomorphic_mapping_dipy(
    y: np.ndarray, x: np.ndarray, sigma_diff=20, radius=20
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
        
    y_gpu = cp.asarray(y)
    x_gpu = cp.asarray(x)
    
    # Define the metric and create the Symmetric Diffeomorphic Registration object
    metric = CCMetric(2, sigma_diff=sigma_diff, radius=radius)
    sdr = SymmetricDiffeomorphicRegistration(metric, opt_tol=1e-04, inv_tol=0.01)

    # Perform the diffeomorphic registration using the pre-alignment from affine registration
    mapping = sdr.optimize(y_gpu, x_gpu)

    return mapping
