#!/usr/bin/env python

import time
import math
import numpy as np
import argparse
import os
os.environ["JAX_PLATFORM_NAME"] = "cpu"
import logging
from basicpy import BaSiC
import tifffile as tiff
from utils import logging_config
from utils.io import save_h5, save_pickle


# Set up logging configuration
logging_config.setup_logging()
logger = logging.getLogger(__name__)


def count_fovs(image_shape, fov_size, overlap=0):
    """Calculate how many FOVs are needed to cover an image with given FOV size and overlap."""
    height, width = image_shape[:2]
    fov_h, fov_w = (fov_size, fov_size) if isinstance(fov_size, (int, float)) else fov_size
    overlap_y, overlap_x = (overlap, overlap) if isinstance(overlap, (int, float)) else overlap
    
    if overlap_y >= fov_h or overlap_x >= fov_w:
        raise ValueError("Overlap cannot be >= FOV size")
    
    step_y = fov_h - overlap_y
    step_x = fov_w - overlap_x
    
    n_fovs_y = math.ceil((height - fov_h) / step_y) + 1 if height > fov_h else 1
    n_fovs_x = math.ceil((width - fov_w) / step_x) + 1 if width > fov_w else 1
    
    return n_fovs_y, n_fovs_x


def split_image_to_fovs(image, n_fovs_x, n_fovs_y):
    """Split image into FOVs with adaptive sizing to handle remainders."""
    if n_fovs_x <= 0 or n_fovs_y <= 0:
        raise ValueError("Number of FOVs must be positive")
    
    shape = image.shape
    height, width = shape[:2]
    
    # Calculate base FOV sizes and remainders
    base_w = width // n_fovs_x
    base_h = height // n_fovs_y
    remainder_x = width % n_fovs_x
    remainder_y = height % n_fovs_y
    
    # Calculate actual FOV sizes (some FOVs get +1 pixel to handle remainders)
    fov_widths = [base_w + (1 if j < remainder_x else 0) for j in range(n_fovs_x)]
    fov_heights = [base_h + (1 if i < remainder_y else 0) for i in range(n_fovs_y)]
    
    max_w = max(fov_widths)
    max_h = max(fov_heights)
    
    # Create FOV stack with padding to max dimensions
    if len(shape) == 2:
        fovs = np.zeros((n_fovs_y * n_fovs_x, max_h, max_w), dtype=image.dtype)
    else:
        fovs = np.zeros((n_fovs_y * n_fovs_x, max_h, max_w, shape[2]), dtype=image.dtype)
    
    # Extract FOVs and store position info
    positions = []
    y_start = 0
    idx = 0
    
    for i in range(n_fovs_y):
        x_start = 0
        for j in range(n_fovs_x):
            h = fov_heights[i]
            w = fov_widths[j]
            
            # Extract FOV
            if len(shape) == 2:
                fovs[idx, :h, :w] = image[y_start:y_start + h, x_start:x_start + w]
            else:
                fovs[idx, :h, :w, :] = image[y_start:y_start + h, x_start:x_start + w, :]
            
            positions.append((y_start, x_start, h, w))
            x_start += w
            idx += 1
        y_start += fov_heights[i]
    
    return fovs, positions, (max_h, max_w)


def reconstruct_image_from_fovs(fovs, positions, original_shape):
    """Reconstruct original image from FOV stack and position information."""
    if len(original_shape) == 2:
        reconstructed = np.zeros(original_shape, dtype=fovs.dtype)
    else:
        reconstructed = np.zeros(original_shape, dtype=fovs.dtype)

    for idx, (y_start, x_start, h, w) in enumerate(positions):
        if len(original_shape) == 2:
            reconstructed[y_start:y_start + h, x_start:x_start + w] = fovs[idx, :h, :w]
        else:
            reconstructed[y_start:y_start + h, x_start:x_start + w, :] = fovs[idx, :h, :w, :]

    return reconstructed


def _parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--channels",
        type=str,
        default=None,
        required=True,
        nargs="*"
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        required=True
    )
    parser.add_argument(
        "--patient_id",
        type=str,
        default=None,
        required=True,
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        required=True,
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


if __name__ == "__main__":
    args = _parse_args()

    handler = logging.FileHandler(args.log_file)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)


    files = args.channels
    image = args.image
    os.makedirs(args.output_dir, exist_ok=True)
    
    fov_size = (1950, 1950)
    autotune = True
    preprocessed = []    

    output_path = os.path.join(args.output_dir, f"preprocessed_{os.path.basename(image)}")

    ch_order = image.split(".")[0].split("_")[1:][::-1]
    sorted_files = []
    for ch in ch_order:
            sorted_files.append(files[np.where([ch in x for x in files])[0][0]])

    files = sorted_files

    for file in files:
        if "DAPI" in file:
            print(f"Skipping ${file}")
            image = tiff.imread(file)
            preprocessed.append(image)
        else:
            print(f"Processing ${file}")
            logger.info('Loading file')
            start_time = time.time()
            image = tiff.imread(file)
            end_time = time.time()
            logger.info(f"Loading took {end_time - start_time:.2f} seconds")
            tile_shape = image.shape[0], image.shape[1]
            logger.info(tile_shape)
        
            n_fovs_y, n_fovs_x = count_fovs(image.shape, fov_size)

            fov_stack, positions, max_fov_size = split_image_to_fovs(image, n_fovs_x, n_fovs_y)
    
            basic = BaSiC(get_darkfield=True)

            logger.info('Before optimization')
            logger.info(
                basic.smoothness_flatfield, basic.smoothness_darkfield, basic.sparse_cost_darkfield
            )

            autotune = False
            if autotune:
                # Autotune parameters
                logger.info('Begun optimization...')
                n_iter = 3
                start_time = time.time()
                basic.autotune(fov_stack, early_stop=True, n_iter=n_iter)
                end_time = time.time()
                logger.info(f"Optimization took {end_time - start_time:.2f} seconds")
                logger.info('After optimization')
                logger.info(
                    basic.smoothness_flatfield, basic.smoothness_darkfield, basic.sparse_cost_darkfield
                )

                logger.info('Applying transformation')
                start_time = time.time()
                transformed = basic.fit_transform(fov_stack)
                end_time = time.time()
                logger.info(f"Transformation took {end_time - start_time:.2f} seconds")
            else:
                logger.info('Applying default transformation')
                start_time = time.time()
                transformed = basic.fit_transform(fov_stack)
                end_time = time.time()
                logger.info(f"Transformation took {end_time - start_time:.2f} seconds")

            reconstructed = reconstruct_image_from_fovs(fov_stack, positions, image.shape)

            assert reconstructed.shape == image.shape, "Shape mismatch!"

            preprocessed.append(reconstructed) 

    # save_pickle(preprocessed, output_path.replace('.nd2', '.pkl'))
    preprocessed = np.stack(preprocessed, axis=0)

    logger.info("Writing out")

    if output_path.endswith('.nd2'):
        output_path = output_path.replace('.nd2', '.h5')
    
    elif output_path.endswith('.tiff'):
        output_path = output_path.replace('.tiff', '.h5')
    else:
        ValueError("Input path must end with .nd2 or .tiff")

    save_h5(preprocessed, output_path)
