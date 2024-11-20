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
        "-c",
        "--crops",
        type=str,
        default=None,
        required=True,
        help="A list of crops",
    )
    parser.add_argument(
        "-c",
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
        "-or",
        "--original_file",
        type=str,
        default=None,
        required=True,
        help="Padded full moving or fixed file.",
    )
    args = parser.parse_args()
    return args


def load_pickle(path):
    # Open the file in binary read mode
    with open(path, "rb") as file:
        # Deserialize the object from the file
        loaded_data = pickle.load(file)

    return loaded_data


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


def save_h5(data, path, chunks=None):
    # Save the NumPy array to an HDF5 file
    with h5py.File(path, "w") as hdf5_file:
        hdf5_file.create_dataset("dataset", data=data, chunks=chunks)
        hdf5_file.flush()


def get_shape_h5file(path, format=".h5"):
    """
    Get the width and height of a h5.
    """

    with h5py.File(path, "r") as f:
        shape = f["dataset"].shape
        f.close()

    return shape


def main():
    args = _parse_args()
    crops_files = args.crops.split(" ")
    original_shape = get_shape_h5file(args.original_file)

    crops = []
    positions = []

    for crop in crops:
        crops.append(load_pickle(crop))
        x, y = map(int, crop.split("_")[1:3])
        positions.append((x, y))

    reconstructed_image = reconstruct_image(
        crops,
        positions,
        original_shape=original_shape,
        crop_size=args.crop_size,
        overlap_size=args.overlap_size,
    )

    save_h5(reconstructed_image, f"registered_{os.path.basename(args.original_file)}")


if __name__ == "__main__":
    main()
