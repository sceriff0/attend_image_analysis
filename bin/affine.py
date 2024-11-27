#!/usr/bin/env python
# Compute affine transformation matrix

import argparse
import gc
import os
from utils.io import load_h5, save_pickle
from utils.cropping import create_crops, reconstruct_image
from utils.mapping import compute_affine_mapping_cv2, apply_mapping


def _parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
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
        default=8000,
        required=False,
        help="Size of the crop",
    )
    parser.add_argument(
        "-oa",
        "--overlap_size_affine",
        type=int,
        default=2000,
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
        default=200,
        required=False,
        help="Size of the overlap",
    )

    args = parser.parse_args()
    return args


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

    for x in range(0, X - overlap_size, crop_size - overlap_size):
        for y in range(0, Y - overlap_size, crop_size - overlap_size):
            # Ensure crop dimensions don't exceed the image dimensions
            x_end = min(x + crop_size, X)
            y_end = min(y + crop_size, Y)

            # Computing local affine transformation matrix
            matrix = compute_affine_mapping_cv2(
                y=fixed[x:x_end, y:y_end, 2], x=moving[x:x_end, y:y_end, 2].squeeze()
            )
            save_pickle(
                (  
                    fixed[x:x_end, y:y_end, :],
                    # Apply local affine transformation
                    apply_mapping(matrix, moving[x:x_end, y:y_end, :], method="cv2"),
                ),
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
    
    crops, positions = create_crops(
        moving, args.crop_size_affine, args.overlap_size_affine
    )

    del moving
    gc.collect()

    for i, crop in enumerate(crops):
        crops[i] = apply_mapping(matrix, crop, method="cv2")

    reconstructed_image = reconstruct_image(
        crops,
        positions,
        original_shape=moving_shape,
        crop_size=args.crop_size_affine,
        overlap_size=args.overlap_size_affine,
    )

    del crops
    gc.collect()

    fixed = load_h5(args.fixed_image)
    create_crops2save(
        fixed,
        reconstructed_image,
        overlap_size=args.overlap_size_diffeo,
        outname=args.moving_image,
    )

    del reconstructed_image, fixed
    gc.collect()


if __name__ == "__main__":
    main()
