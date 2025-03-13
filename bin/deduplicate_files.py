#!/usr/bin/env python

import os
import argparse
import logging
import shutil

from utils import logging_config

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
        nargs='*',
        help="String of paths to h5 files (image channels).",
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
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    images = args.images

    logger.debug(f"Images: {images}")

    unique_files = []
    seen_basenames = []
    for file in images:
        base = os.path.basename(file)
        if not base in seen_basenames and len(base.split('_')) == 3:
            seen_basenames.append(base)
            unique_files.append(file)
    
    logger.debug(f"Unique images: {unique_files}")

    for file in unique_files:
        logger.debug(f"Copying {file} to current directory")
        shutil.copy(file, "./")

        
if __name__ == '__main__':
    main()