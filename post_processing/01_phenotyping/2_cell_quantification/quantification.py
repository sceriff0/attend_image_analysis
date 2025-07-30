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
from skimage.measure import regionprops_table
from aicsimageio import AICSImage

import dask
from dask import delayed, compute, config
from dask.diagnostics import ProgressBar
from dask.threaded import get as threaded_get
from dask.multiprocessing import get as multiprocessing_get
from dask.distributed import Client, LocalCluster
from tqdm.dask import TqdmCallback


def print_memory_usage(prefix=""):
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / 1024**3  # in GB
    print(f"{prefix}Memory usage: {mem:.2f} GB")


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


def crop_array(array, crop_height, crop_width):
    """Croppeth the given 2D array into smaller parts of size (crop_height, crop_width)."""
    height, width = array.shape
    crops = []
    positions = []
    for i in range(0, height, crop_height):
        for j in range(0, width, crop_width):
            crop = array[i:i+crop_height, j:j+crop_width]
            positions.append((i, i+crop_height, j, j+crop_width))
            crops.append(crop)
    return crops, positions

def stitch_array(crops, original_shape, crop_height, crop_width):
    """Stitcheth the crops back into the original 2D array."""
    height, width = original_shape
    array = np.zeros((height, width), dtype=crops[0].dtype)

    idx = 0
    for i in range(0, height, crop_height):
        for j in range(0, width, crop_width):
            h, w = crops[idx].shape
            array[i:i+h, j:j+w] = crops[idx]
            idx += 1

    return array

def load_channel_crop(file, crop_positions):
    image = AICSImage(file).get_image_data("YX")
    return [image[pos[0]:pos[1], pos[2]:pos[3]] for pos in crop_positions]

def import_images(path):
        ### Importing DAPI channel
        img = AICSImage(path)
        pixel_microns = img.physical_pixel_sizes
        # print(f"{pixel_microns = }")
        dims = img.dims  # returns a Dimensions object
        order = img.dims.order  # returns string "TCZYX"
        shape = img.shape  # returns tuple of dimension sizes in TCZYX order
        series = img.scenes
        # print(f"{dims = }, {order = }, {shape = }, {series = }")

        return img, pixel_microns

def process_crop_from_files(mask_path, channel_path, pos, size_cutoff, chan_name, verbose):
    def log(msg):
        if verbose:
            print(msg)

    crop_mask = np.load(mask_path)
    crop_channel = np.load(channel_path)

    labels, counts = np.unique(crop_mask, return_counts=True)
    valid_ids = labels[(labels != 0) & (counts > size_cutoff)]
    if len(valid_ids) == 0 or (crop_mask == 0).all():
        return pd.DataFrame()

    mask_filtered = np.where(np.isin(crop_mask, valid_ids), crop_mask, 0)
    if (mask_filtered == 0).all():
        return pd.DataFrame()

    props = regionprops_table(
        mask_filtered,
        properties=[
            "label", "centroid", "eccentricity", "perimeter",
            "convex_area", "area", "axis_major_length", "axis_minor_length"
        ]
    )
    props_df = pd.DataFrame(props).set_index("label", drop=False)

    flat_mask = mask_filtered.ravel()
    flat_image = crop_channel.ravel()

    sum_per_label = np.bincount(flat_mask, weights=flat_image)
    count_per_label = np.bincount(flat_mask)

    with np.errstate(divide='ignore', invalid='ignore'):
        means = np.true_divide(sum_per_label, count_per_label)
        means[np.isnan(means)] = 0

    mean_values = np.array([means[label] if label < len(means) else 0 for label in valid_ids])
    intensity_df = pd.DataFrame({chan_name: mean_values}, index=valid_ids)

    df = intensity_df.join(props_df)
    df.rename(columns={"centroid-0": "y", "centroid-1": "x"}, inplace=True)
    df["y"] += pos[0]
    df["x"] += pos[2]

    # Cleaning
    os.remove(mask_path)
    os.remove(channel_path)
    
    return df


def extract_features_dask_crops(
    channels_files,
    segmentation_mask,
    output_file=None,
    size_cutoff=0,
    crop_positions=None,
    verbose=True,
    write=False
):
    segmentation_mask = segmentation_mask.squeeze()
    results_all = []

    for file in channels_files:
        chan_name = os.path.basename(file).split('.')[0].split('_')[-1]
        if verbose:
            print(f"\n--- Processing channel: {chan_name} ---")

        channel_data, _ = import_images(file)
        channel_image = channel_data.get_image_data("YX")

        print_memory_usage("Before Dask crops: ")
        tasks = []

        for idx, pos in enumerate(crop_positions):
            crop_mask = segmentation_mask[pos[0]:pos[1], pos[2]:pos[3]]
            crop_channel = channel_image[pos[0]:pos[1], pos[2]:pos[3]]

            # Save crops to temporary .npy files
            mask_path =  f"cropmask_{chan_name}_{pos[0]}.{pos[1]}.{pos[2]}.{pos[3]}.npy"
            channel_path = f"cropchan_{chan_name}_{pos[0]}.{pos[1]}.{pos[2]}.{pos[3]}.npy"
    
            if not os.path.exists(mask_path):
                np.save(mask_path, crop_mask)
            if not os.path.exists(channel_path):
                np.save(channel_path, crop_channel)

            task = delayed(process_crop_from_files)(
                mask_path, channel_path, pos, size_cutoff, chan_name, verbose
            )
            tasks.append(task)

        if verbose:
            print(f"\n--- Dask parallelisation ---")
            print(f"Number of delayed tasks: {len(tasks)}")

        print_memory_usage("Before Dask compute: ")

        with TqdmCallback(desc="Computing crops"):
            dfs = compute(*tasks)
            
        dfs = pd.concat([df for df in dfs if not df.empty], axis=0)
        results_all.append(dfs)
    
        
    if results_all:
        result_df = pd.concat(results_all, axis=1)
        result_df = result_df.loc[:,~result_df.columns.duplicated()]
        if write:
            result_df.to_csv(output_file, index=False)
            if verbose:
                print(f"Saved output to: {output_file}")
        return result_df
    else:
        return pd.DataFrame()




def load_pickle(filepath):
    with open(filepath, "rb") as f:
        return pickle.load(f)


def parse_args():
    parser = argparse.ArgumentParser(description="Run marker quantification for a patient.")
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
    indir, mask_file, positions_file, outdir, patient_id, extract_features_dask_crops
):
    if not os.path.exists(outdir):
        os.makedirs(outdir, exist_ok=True)

    output_file = os.path.join(
        outdir, f"{patient_id}_segmentation_markers_data_FULL.csv"
    )

    segmentation_mask = np.load(mask_file).squeeze()
    positions = load_pickle(positions_file)

    files = [os.path.join(indir, file) for file in os.listdir(indir)]

    markers_data = extract_features_dask_crops(
        channels_files=files,
        segmentation_mask=segmentation_mask,
        output_file=output_file,
        crop_positions=positions,
        write=True,
    )
    return markers_data


def main():
    args = parse_args()

    # Dask configuration
    cluster = LocalCluster(
        n_workers=40,
        threads_per_worker=1,
        memory_limit="5GB"
    )
    client = Client(cluster)

    run_marker_quantification(
        indir=args.indir,
        mask_file=args.mask_file,
        positions_file=args.positions_file,
        outdir=args.outdir,
        patient_id=args.patient_id,
        extract_features_dask_crops=extract_features_dask_crops,
    )


if __name__ == "__main__":
    main()

