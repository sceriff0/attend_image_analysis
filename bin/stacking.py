#!/usr/bin/env python

import os
import argparse
import numpy as np
import h5py
import gc
import tifffile as tiff
import logging
from utils.io import save_h5, load_h5, load_pickle
from utils import logging_config

logging_config.setup_logging()
logger = logging.getLogger(__name__)

def stack_channel(file_path, new_channel_data):
    """
    Append a new channel to an existing dataset in an HDF5 file without loading the entire dataset into memory.
    
    Args:
        file_path (str): Path to the HDF5 file.
        new_channel_data (numpy.ndarray): The new channel data to be added. 
                                           It should have the shape (n, m, 1) where (n, m) matches the existing dataset dimensions.
    """
    with h5py.File(file_path, 'a') as hdf_file:
        # Access the existing dataset
        dataset = hdf_file['dataset']
        
        # Check the current shape of the dataset
        current_shape = dataset.shape
        c, n, m = current_shape  # Unpack current dimensions

        dataset.resize((c + 1, n, m))
        
        # Add the new channel data to the last channel of the dataset
        dataset[-1, :, :] = new_channel_data.squeeze()  # Remove the last singleton dimension if present

def save_tiff(image, output_path, resolution, metadata):
    tiff.imwrite(output_path, 
                 image, 
                 resolution=resolution,
                 bigtiff=True, 
                 ome=True,
                 metadata=metadata
    )

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
        "-c",
        "--channels",
        type=str,
        default=None,
        required=True,
        help="String of paths to h5 files (image channels).",
    )
    parser.add_argument(
        "-m",
        "--metadata",
        type=str,
        default=None,
        required=True,
        help="Path to .pkl containing image metadata",
    )

    args = parser.parse_args()
    return args

def main():
    handler = logging.FileHandler('/hpcnfs/scratch/DIMA/chiodin/repositories/attend_image_analysis/LOG.log')
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    args = _parse_args()

    channels_files = args.channels.split()
    output_path = f"{args.patient_id}.h5"

    # Get unique channel paths
    channels = {}
    for path in channels_files:
        base = os.path.basename(path)
        if base not in channels:
            channels[base] = path

    # Get unique paths
    channels = list(channels.values())

    #### Channels stacking ####    
    for ch in channels:
        logger.info(f"Loading: {ch}")
        new_channel = load_h5(ch)
        
        if not os.path.exists(output_path):
            save_h5(new_channel, output_path)
        else:
            logger.info(f"Before stacking: file size: {os.path.getsize(output_path)} bytes")
            stack_channel(output_path, new_channel)
            logger.info(f"After stacking: file size: {os.path.getsize(output_path)} bytes") 
            
    
    #### Save stacked image as tiff
    resolution, metadata = load_pickle(args.metadata)
    stacked_image = load_h5(output_path)
    output_path_tiff = output_path.replace('h5', 'tiff')
    save_tiff(stacked_image, output_path_tiff, resolution, metadata)

    del stacked_image
    gc.collect()

 
        
if __name__ == '__main__':
    main()