#!/usr/bin/env python

import time
import math
import numpy as np
import argparse
import os
import logging
from basicpy import BaSiC
import tifffile as tiff
from utils import logging_config
from utils.io import save_h5

# Set up logging configuration
logging_config.setup_logging()
logger = logging.getLogger(__name__)

def calculate_fovs_in_tile(tile_shape, fov_size, overlap=0):
    """
    Calculate the number of FOVs that compose a tile image.
    
    Parameters:
    -----------
    tile_shape : tuple
        (height, width) or (y, x) dimensions of the tile image in pixels
    fov_size : tuple or int
        FOV dimensions. Can be:
        - tuple (fov_height, fov_width) for rectangular FOVs
        - int for square FOVs
    overlap : int or tuple, optional
        Overlap between adjacent FOVs in pixels. Can be:
        - int for same overlap in both dimensions
        - tuple (y_overlap, x_overlap) for different overlaps
        - default is 0 (no overlap)
    
    Returns:
    --------
    dict : Dictionary containing FOV counts and effective step sizes
    """
    
    tile_height, tile_width = tile_shape
    
    # Handle square FOV case
    if isinstance(fov_size, (int, float)):
        fov_height = fov_width = fov_size
    else:
        fov_height, fov_width = fov_size
    
    # Handle overlap
    if isinstance(overlap, (int, float)):
        y_overlap = x_overlap = overlap
    else:
        y_overlap, x_overlap = overlap
    
    # Calculate effective step size (FOV size minus overlap)
    y_step = fov_height - y_overlap
    x_step = fov_width - x_overlap
    
    # Ensure step size is positive
    if y_step <= 0 or x_step <= 0:
        raise ValueError("Overlap cannot be >= FOV size")
    
    # Calculate number of FOVs needed in each dimension
    # We need enough FOVs so that the last FOV covers the remaining area
    n_fovs_y = math.ceil((tile_height - fov_height) / y_step) + 1 if tile_height > fov_height else 1
    n_fovs_x = math.ceil((tile_width - fov_width) / x_step) + 1 if tile_width > fov_width else 1
    
    # Handle edge case where tile is smaller than FOV
    if tile_height <= fov_height:
        n_fovs_y = 1
    if tile_width <= fov_width:
        n_fovs_x = 1
    
    total_fovs = n_fovs_y * n_fovs_x
    
    return {
        'n_fovs_y': n_fovs_y,
        'n_fovs_x': n_fovs_x,
        'total_fovs': total_fovs,
        'fov_grid_shape': (n_fovs_y, n_fovs_x),
        'effective_step_y': y_step,
        'effective_step_x': x_step,
        'coverage_y': (n_fovs_y - 1) * y_step + fov_height,
        'coverage_x': (n_fovs_x - 1) * x_step + fov_width
    }


def crop_image_to_grid(img, crop_height, crop_width):
    img_height, img_width = img.shape[:2]
    n_crops_y = img_height // crop_height
    n_crops_x = img_width // crop_width

    crops = []

    for row in range(n_crops_y):
        # Determine column traversal direction (snakewise)
        col_range = range(n_crops_x) if row % 2 == 0 else reversed(range(n_crops_x))
        for col in col_range:
            y_start = row * crop_height
            y_end = y_start + crop_height
            x_start = col * crop_width
            x_end = x_start + crop_width
            crop = img[y_start:y_end, x_start:x_end]
            crops.append(crop)

    return np.stack(crops, axis=0)


def stitch_crops_to_image(crops, n_crops_y, n_crops_x):
    crop_height, crop_width = crops.shape[1:3]
    channels = crops.shape[3] if crops.ndim == 4 else None

    # Prepare empty canvas for output image
    if channels:
        full_image = np.zeros((n_crops_y * crop_height, n_crops_x * crop_width, channels), dtype=crops.dtype)
    else:
        full_image = np.zeros((n_crops_y * crop_height, n_crops_x * crop_width), dtype=crops.dtype)

    idx = 0
    for row in range(n_crops_y):
        col_range = range(n_crops_x) if row % 2 == 0 else reversed(range(n_crops_x))
        for col in col_range:
            y_start = row * crop_height
            y_end = y_start + crop_height
            x_start = col * crop_width
            x_end = x_start + crop_width

            full_image[y_start:y_end, x_start:x_end] = crops[idx]
            idx += 1

    return full_image


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
    # image = tiff.imread('/hpcnfs/scratch/DIMA/chiodin/tests/preprocessing_basicpy/mod_image.tiff')

    # file = '/hpcnfs/scratch/DIMA/chiodin/tests/preprocessing_basicpy/175029E_L1CAM.tiff'
    # output_path = '/hpcnfs/scratch/DIMA/chiodin/tests/preprocessing_basicpy/preprocessed_175029E_L1CAM.tiff'

    args = _parse_args()

    handler = logging.FileHandler(args.log_file)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # file = '/hpcnfs/scratch/DIMA/chiodin/tests/preprocessing_basicpy/19B1259816_SMA.tiff'
    # output_path = '/hpcnfs/scratch/DIMA/chiodin/tests/preprocessing_basicpy/preprocessed_19B1259816_SMA.tiff.tiff'

    files = args.channels
    image = args.image
    os.makedirs(args.output_dir, exist_ok=True)
    
    # fov_size = (1950, 1950)
    fov_size = (512, 512)
    scale_factor = 1.0
    autotune = True

    preprocessed = []    

    output_path = os.path.join(args.output_dir, f"preprocessed_{os.path.basename(image)}")

    for file in files:
        # output_path = os.path.join(args.output_dir, f"preprocessed_{os.path.basename(file)}")
        print('Loading file')
        start_time = time.time()
        image = tiff.imread(file)
        end_time = time.time()
        print(f"Loading took {end_time - start_time:.2f} seconds")
        tile_shape = image.shape[0], image.shape[1]
        print(tile_shape)
        

        tile_dict = calculate_fovs_in_tile(tile_shape, fov_size)

        n_fovs_y = tile_dict['n_fovs_y']
        n_fovs_x = tile_dict['n_fovs_x']

        crop_height = int((image.shape[0] // n_fovs_y // 2) // scale_factor)
        crop_width = int((image.shape[1] // n_fovs_x // 2) // scale_factor)

        cropped = crop_image_to_grid(image, crop_height=crop_height, crop_width=crop_width)

        print(f'Image shape: {image.shape}, Crop height: {crop_height}, Crop width: {crop_width}')

        basic = BaSiC(get_darkfield=True)

        print('Before optimization')
        print(
            basic.smoothness_flatfield, basic.smoothness_darkfield, basic.sparse_cost_darkfield
        )

        autotune = False
        if autotune:
            # Autotune parameters
            print('Begun optimization...')
            n_iter = 3
            start_time = time.time()
            basic.autotune(cropped, early_stop=True, n_iter=n_iter)
            end_time = time.time()
            print(f"Optimization took {end_time - start_time:.2f} seconds")
            print('After optimization')
            print(
                basic.smoothness_flatfield, basic.smoothness_darkfield, basic.sparse_cost_darkfield
            )

            print('Applying transformation')
            start_time = time.time()
            transformed = basic.fit_transform(cropped)
            end_time = time.time()
            print(f"Transformation took {end_time - start_time:.2f} seconds")
        else:
            print('Applying default transformation')
            start_time = time.time()
            transformed = basic.fit_transform(cropped)
            end_time = time.time()
            print(f"Transformation took {end_time - start_time:.2f} seconds")

        stitched = stitch_crops_to_image(transformed, n_crops_y=image.shape[0] // crop_height, n_crops_x=image.shape[1] // crop_width)

        preprocessed.append(stitched) 

    preprocessed = np.stack(preprocessed, axis=0)

    print("Writing out")
    #tiff.imwrite(output_path, preprocessed)
    save_h5(preprocessed, output_path.replace('.tiff', '.h5'))
