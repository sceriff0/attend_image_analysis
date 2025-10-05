#!/usr/bin/env python

import argparse
import os
import tifffile as tiff
import logging
from utils import logging_config
from utils.io import load_nd2, load_h5


# Set up logging configuration
logging_config.setup_logging()
logger = logging.getLogger(__name__)


def _parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p",
        "--patient_id",
        type=str,
        default=None,
        required=True,
        help="A string containing the current patient id.",
    )
    parser.add_argument(
        "-i",
        "--image",
        type=str,
        default=None,
        required=True,
        help="Path to nd2 multichannel image.",
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

    # image = load_nd2(args.image)

    extension = args.image.split(".")[1]

    if extension == "nd2":
        image = load_nd2(args.image)
    elif extension == "h5":
        image = load_h5(args.image)
    elif extension == "tiff" or extension == "tif":
        image = tiff.imread(args.image)

    base = os.path.basename(args.image)

    if "__" in base:
        base = base.replace("__", "-")

    channel_names = base.split(".")[0].split("_")[1:][::-1]

    for idx, ch in enumerate(channel_names):
        tiff.imsave(f"{args.patient_id}_{ch}.tiff", image[idx, :, :])


if __name__ == "__main__":
    main()
