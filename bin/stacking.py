#!/usr/bin/env python

import os
import argparse
import numpy as np
import h5py
import gc
import re
import tifffile as tiff
import logging
from utils.io import save_h5, load_h5, load_pickle, save_pickle
from utils.metadata_tools import get_channel_list, get_image_file_shape
from utils.io import save_h5, load_h5, load_pickle
from utils.cropping import get_crop_areas
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
    with h5py.File(file_path, "a") as hdf_file:
        # Access the existing dataset
        dataset = hdf_file["dataset"]
        # Check the current shape of the dataset
        current_shape = dataset.shape
        c, n, m = current_shape  # Unpack current dimensions
        dataset.resize((c + 1, n, m))
        # Add the new channel data to the last channel of the dataset
        dataset[-1, :, :] = (
            new_channel_data.squeeze()
        )  # Remove the last singleton dimension if present


def save_tiff(
    image, output_path, resolution=None, bigtiff=True, ome=True, metadata=None
):
    tiff.imwrite(
        output_path,
        image,
        resolution=resolution,
        bigtiff=bigtiff,
        ome=ome,
        metadata=metadata,
    )

def create_tiff_metadata(channel_names=None, pixel_microns=[]):
    resolution = (1 / pixel_microns, 1 / pixel_microns)
    metadata = {
        "axes": "CYX",
        "PhysicalSizeX": pixel_microns,
        "PhysicalSizeY": pixel_microns,
        "PhysicalSizeXUnit": "µm",
        "PhysicalSizeYUnit": "µm",
        "Channel": {"Name": channel_names},
    }

    return resolution, metadata


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
        "-n",
        "--n_crops",
        type=int,
        default=None,
        required=True,
        help="Number of image crops to export.",
    )
    parser.add_argument(
        "-oc",
        "--optional_channels",
        type=str,
        default="",
        help="List of optional channels",
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

    # Fixed-order list of channels
    channels_list = get_channel_list()
    optional_channels = args.optional_channels.split(",")
    channels_list.extend(optional_channels)

    channels_files = args.channels.split()

    channel_names = []
    output_path = f"{args.patient_id}.h5"
    for ch in channels_list:
        # if ch is within the name of any channels_files
        if any(ch in img for img in channels_files):

            curr_img = [img for img in channels_files if ch in img][0]
            channel_names.append(ch)
            new_channel = tiff.imread(curr_img)
            if not os.path.exists(output_path):
                save_h5(np.expand_dims(new_channel, axis=0), output_path)
            else:
                stack_channel(output_path, new_channel)
            del new_channel
            gc.collect()

    if channel_names:
        resolution, metadata = create_tiff_metadata(
            channel_names=channel_names, pixel_microns=0.34533768547788
        )

    #### Save stacked image as tiff
    n_crops = args.n_crops

    if n_crops == 1:
        stacked_image = load_h5(output_path)
        output_path_tiff = output_path.replace("h5", "tiff")
        save_tiff(
            image=stacked_image,
            output_path=output_path_tiff,
            resolution=resolution,
            metadata=metadata,
        )
        del stacked_image
        gc.collect()
    else:
        shape = (
            get_image_file_shape(output_path)[1],
            get_image_file_shape(output_path)[2],
        )
        export_areas = get_crop_areas(shape, n_crops)

        for area in export_areas:
            output_path_tiff = (
                f"{args.patient_id}_{area[0]}_{area[1]}_{area[2]}_{area[3]}.tiff"
            )
            logger.info(f"Processing file {output_path_tiff}.")
            if not os.path.exists(output_path_tiff):
                logger.info(f"Loading region {area} from {output_path}")
                stacked_image = load_h5(
                    output_path, loading_region=area, shape="CYX"
                ).astype(np.int32)
                logger.info(f"Region {area} loaded successfully")
                save_tiff(
                    image=stacked_image,
                    output_path=output_path_tiff,
                    resolution=resolution,
                    metadata=metadata,
                )
                logger.info(f"Saved file {output_path_tiff}.")
                del stacked_image
                gc.collect()
            else:
                logger.info(f"File {output_path_tiff} already exists.")


if __name__ == "__main__":
    main()
