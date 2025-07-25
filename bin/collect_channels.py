#!/usr/bin/env python

import argparse
import os
import tifffile as tiff
import logging
import numpy as np
from utils import logging_config
from utils.io import load_nd2
from utils.io import save_h5


def extract_alpha(s):
    return "".join(c for c in s if c.isalpha())


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
        "--images",
        type=str,
        default=None,
        required=True,
        nargs="*",
        help="List of nd2 multichannel image names.",
    )
    parser.add_argument(
        "-c",
        "--channels",
        type=str,
        default=None,
        required=True,
        nargs="*",
        help="List of single channel tif files.",
    )
    parser.add_argument(
        "--is_fixed",
        type=str,
        default=None,
        required=True,
        nargs="*",
        help="List of boolean values indicating if the image is fixed or not.",
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

    images = args.images
    channels = args.channels

    is_fixed = [extract_alpha(f) for f in args.is_fixed]

    channels_metadata = [(chan, is_f) for chan, is_f in zip(channels, is_fixed)]
    
    cwd = os.getcwd()
    with open(f"{args.patient_id}.csv", "w") as f:
        f.write("patient_id,image,fixed\n")
        for output_name, is_f in channels_metadata:
            f.write(f"{args.patient_id},{os.path.join(cwd, output_name)},{is_f}\n")


if __name__ == "__main__":
    main()
