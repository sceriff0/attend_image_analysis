#!/usr/bin/env python

import tifffile
import nd2
import h5py

def get_image_file_shape(file, format=".h5"):
    """
    Get the width and height of a TIFF image without fully loading the image.

    Parameters:
        file (str): Path to the TIFF image.

    Returns:
        tuple: (width, height) of the image.
    """

    if format == "tiff" or format == ".tiff":
        with tifffile.TiffFile(file) as tiff:
            image_shape = tiff.pages[0].shape  # (height, width)
            shape = (image_shape[1], image_shape[0])  # Extract width and height

    if format == "nd2" or format == ".nd2":
        with nd2.ND2File(file) as nd2_file:
            # Access metadata about the dimensions
            shape_metadata = nd2_file.sizes  # Example: "XYCZT" or similar
            shape_metadata = dict(shape_metadata)
            shape = (shape_metadata.get("Y", 0), shape_metadata.get("X", 0))

    if format == ".h5" or format == "h5":
        with h5py.File(file, "r") as f:
            shape = f["dataset"].shape
            f.close()

    return shape