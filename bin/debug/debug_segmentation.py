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
        "-s",
        "--semgmentation_mask",
        type=str,
        default=None,
        required=True,
        help="npy file containing segmentation mask."
    )
    parser.add_argument(
        "-od",
        "--output_dir",
        type=str,
        default=None,
        required=True,
        help="Directory to save output files.",
    )
    parser.add_argument(
        "-log",
        "--log_file",
        type=str,
        required=False,
        help="Path to log file.",
    )
    parser.add_argument(
        "-o",
        "--output_file",
        type=str,
        required=True,
        help="Path to output file containing IoU score.",
    )

def main():
    args = _parse_args()

    handler = logging.FileHandler(args.log_file)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # Load segmentation mask
    seg_mask = np.load(args.semgmentation_mask)

    quality_score = 1

    with open(os.path.join(args.output_dir, args.output_file), 'w') as f:
        f.write(f"{quality_score}\n")