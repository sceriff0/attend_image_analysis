#!/usr/bin/env python3
"""
Cell Segmentation Pipeline using StarDist
=========================================

A pipeline for segmenting nuclei in microscopy images using StarDist models,
with support for large image processing through cropping and stitching.
"""

import argparse
import os
import time
from typing import List, Tuple

import numpy as np
from aicsimageio import AICSImage
from csbdeep.utils import normalize
from skimage import segmentation
from stardist.models import StarDist2D

import gc

import pickle 

def load_pickle(path):
    # Open the file in binary read mode
    with open(path, "rb") as file:
        # Deserialize the object from the file
        loaded_data = pickle.load(file)

    return loaded_data


def save_pickle(object, path):
    # Open a file in binary write mode
    with open(path, "wb") as file:
        # Serialize the object and write it to the file
        pickle.dump(object, file)



def crop_array(arr: np.ndarray, overlap_size: int) -> List[Tuple[np.ndarray, Tuple[int, int]]]:
    """
    Crop a 2D numpy array into a grid of overlapping crops.
        
    Args:
        arr: 2D numpy array
        overlap_size: Amount of overlap between crops
            
    Returns:
        List of tuples containing (crop, (row_start, col_start))
    """
    crops = []
    positions = []
    n_rows, n_cols = arr.shape
    step = max(overlap_size, 1)

    for i in range(0, n_rows - overlap_size, step):
        for j in range(0, n_cols - overlap_size, step):
            row_end = min(i + step + overlap_size, n_rows)
            col_end = min(j + step + overlap_size, n_cols)
            crop = arr[i:row_end, j:col_end]
            crops.append((crop, (i, j)))
            positions.append((i, row_end, j, col_end))

    return crops, positions


import os
from skimage import morphology
from skimage.filters import gaussian
import numpy as np
import argparse

def multiply_image_and_scalar(image, s):
    return image * s


def power(image, a):
    return image**a


def gamma_correction(image, gamma):
    max_intensity = np.max(image)
    image = multiply_image_and_scalar(image, 1.0 / max_intensity)
    image = power(image, gamma)
    image = multiply_image_and_scalar(image, max_intensity)

    return image


def dapi_preprocessing(image):
    image = gamma_correction(image, 0.6)
    image = morphology.white_tophat(image, footprint=morphology.disk(50))
    image = np.array(image, dtype="float32")
    image = gaussian(image, sigma=1.0)

    return image


class ImageProcessor:
    """Handles image cropping, stitching, and mask processing operations."""

    @staticmethod
    def dapi_preprocessing(image: np.ndarray) -> np.ndarray:
        image = gamma_correction(image, 0.6)
        image = morphology.white_tophat(image, footprint=morphology.disk(50))
        image = np.array(image, dtype="float32")
        image = gaussian(image, sigma=1.0)

        return image

    
    @staticmethod
    def crop_array(arr: np.ndarray, overlap_size: int) -> List[Tuple[np.ndarray, Tuple[int, int]]]:
        """
        Crop a 2D numpy array into a grid of overlapping crops.
        
        Args:
            arr: 2D numpy array
            overlap_size: Amount of overlap between crops
            
        Returns:
            List of tuples containing (crop, (row_start, col_start))
        """
        crops = []
        positions = []
        n_rows, n_cols = arr.shape
        step = max(overlap_size, 1)
        
        for i in range(0, n_rows - overlap_size, step):
            for j in range(0, n_cols - overlap_size, step):
                row_end = min(i + step + overlap_size, n_rows)
                col_end = min(j + step + overlap_size, n_cols)
                crop = arr[i:row_end, j:col_end]
                crops.append((crop, (i, j)))
                positions.append((i, row_end, j, col_end))
        
        return crops, positions

    @staticmethod
    def stitch_array(crops: List[Tuple[np.ndarray, Tuple[int, int]]], 
                    arr_shape: Tuple[int, int], 
                    overlap_size: int) -> np.ndarray:
        """
        Stitch together overlapping crops to recreate the original array.
        
        Args:
            crops: List of tuples (crop, (row_start, col_start))
            arr_shape: Shape of the original array (rows, cols)
            overlap_size: Amount of overlap between crops
            
        Returns:
            Reconstructed 2D numpy array
        """
        if not crops:
            return np.zeros(arr_shape, dtype=np.uint16)
            
        stitched = np.zeros(arr_shape, dtype=crops[0][0].dtype)
        positions = []
        
        for crop, (i, j) in crops:
            rows, cols = crop.shape
            
            # Determine the region to use from this crop (skip overlaps)
            row_start = overlap_size if i > 0 else 0
            col_start = overlap_size if j > 0 else 0
            
            # Calculate placement in stitched array
            i_start = i + row_start
            j_start = j + col_start
            i_end = i + rows
            j_end = j + cols
            
            stitched[i_start:i_end, j_start:j_end] = crop[row_start:rows, col_start:cols]
            positions.append((i_start, i_end, j_start, j_end))
        
        return stitched, positions

    @staticmethod
    def remap_mask_crop_values(mask: np.ndarray, offset: int) -> np.ndarray:
        """
        Remap mask values from [1,N] to [offset+1, N+offset].
        Background (0) remains 0.
        
        Args:
            mask: Input mask array
            offset: Value to add to all non-zero mask values
            
        Returns:
            Remapped mask array
        """
        if offset == 0:
            return mask
        
        remapped_mask = mask.copy()
        unique_vals = np.unique(mask[mask > 0])
        
        for val in unique_vals:
            remapped_mask[mask == val] = val + offset
        
        return remapped_mask
    
    @staticmethod
    def remap_mask_values(arr: np.ndarray) -> np.ndarray:
        """
        Remap all non-zero values in arr to their rank in ascending order,
        with 0 preserved as 0.
        """
        # fetch distinct non-zero values, in ascending order
        nonzero_vals = np.unique(arr[arr != 0])
        
        # forge the mapping: value → rank (starting at 1)
        val_to_rank = {val: i+1 for i, val in enumerate(nonzero_vals)}
        
        # apply mapping, keeping 0 as 0
        return np.vectorize(lambda x: val_to_rank.get(x, 0))(arr)

    @staticmethod
    def filter_noise(arr: np.ndarray, quantile: float = 0.01) -> np.ndarray:
        """
        Set rare grayscale values to zero based on frequency quantile.
        
        Args:
            arr: 2D numpy array of grayscale values
            quantile: Frequency quantile threshold
            
        Returns:
            Array with rare values set to zero
        """
        unique, counts = np.unique(arr.flatten(), return_counts=True)
        quantile_threshold = np.quantile(counts, quantile)
        rare_values = unique[counts <= quantile_threshold]
        
        result = arr.copy()
        result[np.isin(result, rare_values)] = 0
        
        return result


class SegmentationPipeline:
    """Main pipeline for cell segmentation using StarDist."""
    
    def __init__(self, model_path: str, model_name: str, verbose: bool = True):
        """
        Initialize the segmentation pipeline.
        
        Args:
            model_path: Path to the directory containing the model
            model_name: Name of the StarDist model
            verbose: Whether to print progress messages
        """
        self.model = StarDist2D(None, name=model_name, basedir=model_path)
        self.verbose = verbose
        self.processor = ImageProcessor()
    
    def log(self, message: str) -> None:
        """Print message if verbose mode is enabled."""
        if self.verbose:
            print(message)
    
    def load_image(self, filepath: str) -> Tuple[np.ndarray, object]:
        """
        Load image and extract pixel size information.
        
        Args:
            filepath: Path to the image file
            
        Returns:
            Tuple of (image_array, pixel_sizes)
        """
        img = AICSImage(filepath)
        image_data = img.get_image_data("YX")
        pixel_sizes = img.physical_pixel_sizes
        
        return image_data, pixel_sizes
    
    def normalize_image(self, image: np.ndarray, 
                       pmin: float = 1.0, pmax: float = 99.8) -> np.ndarray:
        """
        Normalize image intensity values.
        
        Args:
            image: Input image array
            pmin: Lower percentile for normalization
            pmax: Upper percentile for normalization
            
        Returns:
            Normalized image array
        """
        return normalize(image, pmin, pmax, axis=(0, 1))
    
    def predict_whole_image(self, image: np.ndarray) -> np.ndarray:
        """
        Perform segmentation on the entire image without cropping.
        
        Args:
            image: Input image array
            
        Returns:
            Segmentation mask
        """
        start_time = time.time()
        self.log(f'Processing entire image (shape: {image.shape})')
        
        # Predict instances on whole image
        pred, _ = self.model.predict_instances(image, verbose=False)
        
        # Expand labels
        expanded_pred = segmentation.expand_labels(pred, distance=10, spacing=1)
        
        elapsed = time.time() - start_time
        self.log(f'Processing time: {elapsed:.2f}s')
        self.log(f'Unique labels: {len(np.unique(expanded_pred))}')
        
        return expanded_pred

    def predict_crops(self, image: np.ndarray, overlap: int = 500) -> np.ndarray:
        """
        Perform segmentation on image crops and stitch results.
        
        Args:
            image: Input image array
            overlap: Overlap size between crops
            
        Returns:
            Stitched segmentation mask
        """
        crops, positions = self.processor.crop_array(image, overlap)
        expanded_preds = []
        max_value_so_far = 0
        
        for idx, (crop, pos) in enumerate(crops):
            start_time = time.time()
            self.log(f'Processing crop {idx + 1}/{len(crops)}')
            
            # Predict instances
            self.log(f'Preprocessing crop (shape: {crop.shape})')
            # crop = self.processor.dapi_preprocessing(crop)
            self.log(f'Predicting instances for crop at position {pos}')
            pred, _ = self.model.predict_instances(crop, verbose=False)
            
            # Remap values for consistent labeling across crops
            if idx > 0:
                pred = self.processor.remap_mask_crop_values(pred, max_value_so_far)
                self.log(f'  Remapped values: offset={max_value_so_far}')
            
            # Expand labels
            expanded_pred = segmentation.expand_labels(pred, distance=10, spacing=1)
            max_value_so_far = max(max_value_so_far, np.max(expanded_pred))
            
            expanded_preds.append((expanded_pred, pos))
            
            elapsed = time.time() - start_time
            self.log(f'  Processing time: {elapsed:.2f}s')
            self.log(f'  Unique labels: {len(np.unique(expanded_pred))}')
        
        # Stitch crops together
        self.log('Stitching crops...')
        stitched_mask, _ = self.processor.stitch_array(expanded_preds, image.shape, overlap)
        stitched_mask = self.processor.remap_mask_values(stitched_mask)
        
        return stitched_mask, positions

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Cell segmentation pipeline using StarDist",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python segmentation_pipeline.py --dapi-file /path/to/dapi.tif --model-dir /path/to/models/
        """
    )
    
    parser.add_argument(
        '--dapi-file', 
        type=str, 
        required=True,
        help='Path to the DAPI image file'
    )
    
    parser.add_argument(
        '--model-dir',
        type=str,
        default='/hpcnfs/scratch/DIMA/chiodin/tests/stardist_training_notebook/runs/20250514/runs/10/models/',
        help='Directory containing the StarDist model'
    )
    
    parser.add_argument(
        '--model-name',
        type=str,
        default='stardist_full_e200_lr00001_aug1_seed10_es50p0.001_rlr0.5p50',
        help='Name of the StarDist model'
    )
    
    parser.add_argument(
        '--whole_image',
        action='store_true',
        help='Process the entire image without cropping (default: use cropping)'
    )
    
    parser.add_argument(
        '--overlap',
        type=int,
        default=500,
        help='Overlap size for image cropping (default: 500, ignored if --no-crop is used)'
    )
    
    parser.add_argument(
        '--crop',
        nargs=4,
        type=int,
        metavar=('ROW_START', 'ROW_END', 'COL_START', 'COL_END'),
        help='Crop region as row_start row_end col_start col_end'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./output',
        help='Output directory for results (default: ./output)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    
    return parser.parse_args()


def main():
    """Main execution function."""
    args = parse_arguments()
    
    # Validate inputs
    if not os.path.exists(args.dapi_file):
        raise FileNotFoundError(f"DAPI file not found: {args.dapi_file}")
    
    if not os.path.exists(args.model_dir):
        raise FileNotFoundError(f"Model directory not found: {args.model_dir}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize pipeline
    pipeline = SegmentationPipeline(args.model_dir, args.model_name, args.verbose)
    
    # Load and process DAPI image
    pipeline.log(f"Loading DAPI image: {args.dapi_file}")
    dapi_image, pixel_sizes = pipeline.load_image(args.dapi_file)
    
    # Normalize image
    pipeline.log("Normalizing image...")
    dapi_normalized = pipeline.normalize_image(dapi_image)
    
    del dapi_image
    gc.collect()
    
    # Apply crop if specified
    if args.crop:
        row_start, row_end, col_start, col_end = args.crop
        image_to_process = dapi_normalized[row_start:row_end, col_start:col_end]
        pipeline.log(f"Applied crop: [{row_start}:{row_end}, {col_start}:{col_end}]")
    else:
        image_to_process = dapi_normalized
        
    pipeline.log(f"Processing image shape: {image_to_process.shape}")
    
    # Perform segmentation
    start_time = time.time()
    
    if args.whole_image:
        pipeline.log("Processing entire image without cropping...")
        segmentation_mask = pipeline.predict_whole_image(image_to_process)
        _, positions = crop_array(image_to_process, args.overlap)
    else:
        pipeline.log(f"Processing image with crops (overlap: {args.overlap})...")
        segmentation_mask, positions = pipeline.predict_crops(image_to_process, args.overlap)
    
    total_time = time.time() - start_time
    
    pipeline.log(f"Segmentation completed in {total_time:.2f}s")
    pipeline.log(f"Total unique labels: {len(np.unique(segmentation_mask))}")
    
    # Save segmentation mask
    basename = os.path.basename(args.dapi_file)
    mask_path = os.path.join(args.output_dir, f'segmentation_mask.npy')
    np.save(mask_path, segmentation_mask)
    pipeline.log(f"Segmentation mask saved to: {mask_path}")

    save_pickle(positions, os.path.join(args.output_dir, 'positions.pkl'))
    pipeline.log("Segmentation mask saved as pickle file.")


if __name__ == "__main__":
    main()
