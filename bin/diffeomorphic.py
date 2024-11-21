#!/usr/bin/env python
# Compute diffeomorphic transformation matrix

import argparse
import os
import numpy as np
from utils.io import load_pickle, save_pickle
from utils.mapping import compute_diffeomorphic_mapping_dipy, apply_mapping

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
