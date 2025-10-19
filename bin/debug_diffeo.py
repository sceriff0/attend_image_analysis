#!/usr/bin/env python

import argparse
import logging
import os
import pickle
import tifffile as tiff
import numpy as np
from utils import logging_config
from utils.io import load_h5, save_h5, save_pickle, load_pickle
from utils.mapping import apply_mapping

import numpy as np
from aicsimageio import AICSImage
from csbdeep.utils import normalize
from skimage import segmentation
from stardist.models import StarDist2D



# Set up logging configuration
logging_config.setup_logging()
logger = logging.getLogger(__name__)

# Read crop path, mapping and log file from command line arguments
def _parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-cr",
        "--crop",
        type=str,
        default=None,
        required=True,
        help="Pickle file containing fixed and moving crops, in this order.",
    )
    parser.add_argument(
        "-m",
        "--mapping",
        type=str,
        default=None,
        required=True,
        help="Pickle file containing diffeomorphic mapping.",
    )
    parser.add_argument(
        "-md",
        "--model-dir",
        type=str,
        default=None,
        required=True,
        help="Directory containing StarDist model.",
    )
    parser.add_argument(
        "-mn",
        "--model-name",
        type=str,
        default=None,
        required=True,
        help="Name of the StarDist model.",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        default=None,
        required=True,
        help="Directory to save debug files.",
    )
    parser.add_argument(
        "-l",   
        "--log_file",
        type=str,
        required=False,
        help="Path to log file.",
    )
    parser.add_argument(
        "-of",
        "--output_file",
        type=str,
        required=True,
        help="Path to output file containing IoU score.",
    )
    args = parser.parse_args()
    return args

def main():
    args = _parse_args()

    handler = logging.FileHandler(args.log_file)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # Load crop data and mapping
    crops = load_pickle(args.crop)
    mapping = load_pickle(args.mapping)

    fixed_crop = crops[0]
    moving_crop = crops[1]

    fixed_crop_norm = normalize(fixed_crop, 1, 99.8, axis=(0, 1))
    moving_crop_norm = normalize(moving_crop, 1, 99.8, axis=(0, 1))

    model = StarDist2D(None, name=args.model_name, basedir=args.model_dir)
    model.config.use_gpu = True

    fixed_pred, _ = model.predict_instances(fixed_crop_norm, n_tiles=(8,8), verbose=False)
    moving_pred, _ = model.predict_instances(moving_crop_norm, n_tiles=(8,8), verbose=False)

    registered_moving_pred = apply_mapping(mapping, moving_pred)

    # Compute IoU between fixed_pred and registered_moving_pred
    intersection = np.logical_and(fixed_pred > 0, registered_moving_pred > 0)
    union = np.logical_or(fixed_pred > 0, registered_moving_pred > 0)
    iou = np.sum(intersection) / np.sum(union)

    logger.debug(f'DIFFEOMORPHIC - IoU: {iou}')
    with open(os.path.join(args.output_dir, args.output_file), 'w') as f:
        f.write(f'IoU: {iou}\n')

if __name__ == "__main__":
    main()   