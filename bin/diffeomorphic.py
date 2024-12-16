#!/usr/bin/env python
# Compute diffeomorphic transformation matrix

import argparse
import os
import numpy as np
from utils.io import load_pickle, save_h5
from utils.mapping import compute_diffeomorphic_mapping_dipy, apply_mapping

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
        "-c",
        "--crop_image",
        type=str,
        default=None,
        required=True,
        help="Pickle file containing fixed and moving crops, in this order.",
    )
    parser.add_argument(
        "-m",
        "--moving_image",
        type=str,
        default=None,
        required=True,
        help="h5 image file",
    )

    args = parser.parse_args()
    return args

def main():
    args = _parse_args()

    fixed, moving = load_pickle(args.crop_image)

    moving_channels = os.path.basename(args.moving_image) \
        .split('.')[0] \
        .split('_')[2:][::-1] 

    channels_to_register = remove_lowercase_channels(moving_channels)

    crop_id = os.path.basename(args.crop_image).split('.')[0].split('_')[0:3]
    crop_id = [str(e) for e in crop_id]
    crop_name = crop_id + channels_to_register[::-1]   
    crop_name = '_'.join(crop_name)
    output_path = f"registered_{crop_name}.h5"

    if len(np.unique(moving)) != 1:
        mapping = compute_diffeomorphic_mapping_dipy(
            y=fixed[:, :, -1].squeeze(), 
            x=moving[:, :, -1].squeeze()
        )

        registered_images = []
        for idx, ch in enumerate(moving_channels):
            if ch in channels_to_register:
                registered_images.append(apply_mapping(mapping, moving[:, :, idx]))

        registered_images = np.stack(registered_images, axis=-1)

        save_h5(
            registered_images, 
            output_path
        )
        
    else:
        moving_channels_images = []
        for idx, ch in enumerate(moving_channels):
            if ch in channels_to_register:
                moving_channels_images.append(moving[:, :, idx])

        moving_channels_images = np.stack(moving_channels_images, axis=-1)

        save_h5(
            moving_channels_images, 
            output_path
        )

if __name__ == "__main__":
    main()
