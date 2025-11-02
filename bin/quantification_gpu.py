#!/usr/bin/env python3
# Standard library
import os
import time
import pickle
import argparse
import psutil

# Third-party libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from aicsimageio import AICSImage

# GPU libraries
import cupy as cp
import cucim.skimage as cskimage
from cucim.skimage.measure import regionprops_table as gpu_regionprops_table


def print_memory_usage(prefix=""):
    """Print CPU and GPU memory usage"""
    process = psutil.Process(os.getpid())
    cpu_mem = process.memory_info().rss / 1024**3  # in GB
    
    # GPU memory
    mempool = cp.get_default_memory_pool()
    gpu_mem = mempool.used_bytes() / 1024**3  # in GB
    gpu_total = mempool.total_bytes() / 1024**3
    
    print(f"{prefix}CPU Memory: {cpu_mem:.2f} GB | GPU Memory: {gpu_mem:.2f}/{gpu_total:.2f} GB")


def load_pickle(path):
    """Load a pickle file"""
    with open(path, "rb") as file:
        loaded_data = pickle.load(file)
    return loaded_data


def save_pickle(obj, path):
    """Save object to pickle file"""
    with open(path, "wb") as file:
        pickle.dump(obj, file)


def import_images(path):
    """Import image and metadata"""
    img = AICSImage(path)

    # Handle pixel size across versions
    try:
        pixel_microns = img.physical_pixel_sizes
    except AttributeError:
        pixel_microns = img.get_physical_pixel_size()

    # Handle dims across versions
    dims = img.dims
    #order = dims if isinstance(dims, str) else dims.order

    #shape = img.shape
    #series = img.scenesavailable_scenes

    return img, pixel_microns


def gpu_extract_features(
    segmentation_mask,
    channel_image,
    chan_name,
    size_cutoff=0,
    verbose=True
):
    """Extract features using GPU-accelerated regionprops"""
    
    if verbose:
        print(f"Processing channel: {chan_name}")
        print_memory_usage(f"  Before GPU transfer ({chan_name}): ")
    
    # Transfer to GPU
    mask_gpu = cp.asarray(segmentation_mask.squeeze())
    channel_gpu = cp.asarray(channel_image.squeeze())
    
    if verbose:
        print_memory_usage(f"  After GPU transfer ({chan_name}): ")
    
    # Get unique labels and filter by size
    labels, counts = cp.unique(mask_gpu, return_counts=True)
    valid_ids = labels[(labels != 0) & (counts > size_cutoff)]
    
    if len(valid_ids) == 0:
        return pd.DataFrame()
    
    # Filter mask to only include valid labels
    valid_ids_cpu = valid_ids.get()  # Need CPU version for isin
    mask_filtered = cp.where(cp.isin(mask_gpu, valid_ids), mask_gpu, 0)
    
    if (mask_filtered == 0).all():
        return pd.DataFrame()
    
    # Use GPU-accelerated regionprops
    props = gpu_regionprops_table(
        mask_filtered,
        properties=[
            "label", "centroid", "eccentricity", "perimeter",
            "convex_area", "area", "axis_major_length", "axis_minor_length"
        ]
    )
    
    # Convert to pandas DataFrame
    props_df = pd.DataFrame(props).set_index("label", drop=False)
    
    # Calculate mean intensities using GPU
    flat_mask = mask_filtered.ravel()
    flat_image = channel_gpu.ravel()
    
    # Use CuPy bincount for efficient computation
    #max_label = int(mask_filtered.max()) + 1
    sum_per_label = cp.bincount(flat_mask, weights=flat_image)#, minlength=max_label)
    count_per_label = cp.bincount(flat_mask)#, minlength=max_label)
    
    # Compute means
    with cp.errstate(divide='ignore', invalid='ignore'):
        means = cp.true_divide(sum_per_label, count_per_label)
        means = cp.nan_to_num(means, nan=0.0)
    
    # Extract mean values for valid labels
    mean_values = cp.array([means[int(label)] if label < len(means) else 0 
                            for label in valid_ids])
    
    # Convert to CPU and create DataFrame
    mean_values_cpu = mean_values.get()
    valid_ids_cpu = valid_ids.get()
    
    intensity_df = pd.DataFrame({chan_name: mean_values_cpu}, index=valid_ids_cpu)
    
    # Join with properties DataFrame
    df = intensity_df.join(props_df)
    df.rename(columns={"centroid-0": "y", "centroid-1": "x"}, inplace=True)
    
    # Clear GPU memory
    del mask_gpu, channel_gpu, mask_filtered, flat_mask, flat_image
    del sum_per_label, count_per_label, means, mean_values
    cp.get_default_memory_pool().free_all_blocks()
    
    return df


def extract_features_gpu(
    channels_files,
    segmentation_mask,
    output_file=None,
    size_cutoff=0,
    verbose=True,
    write=False
):
    """Main feature extraction function using GPU acceleration"""
    
    segmentation_mask = segmentation_mask.squeeze()
    results_all = []
    
    for file in channels_files:
        chan_name = os.path.basename(file).split('.')[0].split('_')[-1]
        
        if verbose:
            print(f"\n--- Processing channel: {chan_name} ---")
        
        # Load channel image
        channel_data, _ = import_images(file)
        channel_image = channel_data.get_image_data("YX")
        
        # Extract features using GPU
        df = gpu_extract_features(
            segmentation_mask,
            channel_image,
            chan_name,
            size_cutoff,
            verbose
        )
        
        if not df.empty:
            results_all.append(df)
        
        # Clear memory after each channel
        del channel_image
        cp.get_default_memory_pool().free_all_blocks()
    
    # Combine results
    if results_all:
        result_df = pd.concat(results_all, axis=1)
        result_df = result_df.loc[:, ~result_df.columns.duplicated()]
        
        if write and output_file:
            result_df.to_csv(output_file, index=False)
            if verbose:
                print(f"Saved output to: {output_file}")
        
        return result_df
    else:
        return pd.DataFrame()


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Run GPU-accelerated marker quantification for a patient.")
    parser.add_argument("--patient_id", required=True, help="Patient ID to process")
    parser.add_argument(
        "--indir", required=True, help="Input directory with registered channel images"
    )
    parser.add_argument(
        "--mask_file", required=True, help="Path to segmentation mask .npy file"
    )
    parser.add_argument(
        "--positions_file", required=True, help="Path to crop positions .pkl file"
    )
    parser.add_argument(
        "--outdir", required=True, help="Output directory to save quantification results"
    )
    return parser.parse_args()


def run_marker_quantification(
    indir, mask_file, outdir, patient_id, size_cutoff=0, verbose=True
):
    """Run the marker quantification pipeline"""
    
    if not os.path.exists(outdir):
        os.makedirs(outdir, exist_ok=True)
    
    output_file = os.path.join(
        outdir, f"{patient_id}_segmentation_markers_data_FULL.csv"
    )
    
    print(f"Loading segmentation mask from: {mask_file}")
    segmentation_mask = np.load(mask_file).squeeze()
    print(f"Mask shape: {segmentation_mask.shape}")
    
    # Get all channel files
    files = [os.path.join(indir, file) for file in os.listdir(indir) 
             if file.endswith(('.tif', '.tiff', '.ome.tif', '.ome.tiff'))]
    print(f"Found {len(files)} channel files")
    
    # Run GPU-accelerated feature extraction
    markers_data = extract_features_gpu(
        channels_files=files,
        segmentation_mask=segmentation_mask,
        output_file=output_file,
        size_cutoff=0,
        verbose=verbose,
        write=True,
    )
    
    return markers_data


def main():
    """Main function"""
    args = parse_args()
    
    # Run the pipeline
    start_time = time.time()
    
    markers_data = run_marker_quantification(
        indir=args.indir,
        mask_file=args.mask_file,
        outdir=args.outdir,
        patient_id=args.patient_id,
        size_cutoff=0,
    )
    
    end_time = time.time()
    print(f"\n=== Completed in {end_time - start_time:.2f} seconds ===")
    
    # Clear GPU memory
    mempool = cp.get_default_memory_pool()
    mempool.free_all_blocks()
    
    if not markers_data.empty:
        print(f"Processed {len(markers_data)} cells")
        print(f"Columns: {list(markers_data.columns)}")


if __name__ == "__main__":
    main()