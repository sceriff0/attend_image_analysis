import argparse
import logging
import os
import pickle
import tifffile as tiff
import numpy as np
from utils import logging_config
from utils.io import load_h5, save_h5, save_pickle, load_pickle
from utils.mapping import apply_mapping


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

    # Apply mapping to moving crop
    registered_crop = apply_mapping(moving_crop, mapping)

    logger.info(f"Saved fixed, moving, and registered crops to {args.output_dir}")