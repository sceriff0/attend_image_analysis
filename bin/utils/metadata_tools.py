#!/usr/bin/env python

import tifffile
import nd2
import h5py
import tifffile
import xml.etree.ElementTree as ET
import json
import os
from nd2reader import ND2Reader


def get_channel_list():
    channels_list = [
        "phenotypes_mask",
        "DAPI",
        "PANCK",  # Membrane
        "MLH1",
        "P53",
        "ARID1A",
        "PAX2",
        "VIMENTIN",  # Membrane
        "SMA",
        "CD163",  # Membrane
        "CD14",  # Membrane
        "CD45",  # Membrane
        "CD3",  # Membrane (if CD4 or CD8)
        "CD4",  # Membrane
        "CD8",  # Membrane
        "FOXP3",
        "PD1",  # Membrane
        "PDL1",  # Membrane
        "L1CAM",  # Membrane
        "ARPC1B",
        "CD74",
        "GZMB"
    ]
    return channels_list


def get_image_file_shape(file, format=None):
    """
    Get the width and height (or shape) of TIFF, ND2, or HDF5 images 
    without fully loading them.
    """

    # if format has . before extension, remove it
    if format and format.startswith("."):
        format = format[1:]

    if format is None:
        format = file.split(".")[-1].lower()

    if format in ("tiff", "tif"):
        with tifffile.TiffFile(file) as tiff:
            h, w = tiff.pages[0].shape[:2]
            return (w, h)

    elif format == "nd2":
        with nd2.ND2File(file) as nd2_file:
            s = dict(nd2_file.sizes)
            return (s.get("C", 0), s.get("Y", 0), s.get("X", 0))

    elif format == "h5":
        with h5py.File(file, "r") as f:
            return f["dataset"].shape

    else:
        raise ValueError(f"Unsupported format: {format}")


def get_metadata_nd2(path):
    with ND2Reader(path) as data:
        # Print general metadata
        metadata = data.metadata

    return metadata


def get_metadata_tiff(file_path):
    """
    Extract and print metadata from a TIFF file.

    Args:
        file_path (str): Path to the TIFF file.
        method (str): Method to use for extraction. Options are 'pillow', 'tifffile', or 'imageio'.

    Returns:
        dict: A dictionary containing the metadata.
    """
    metadata = {}
    with tifffile.TiffFile(file_path) as tiff:
        metadata["TIFF Tags"] = {}
        for page in tiff.pages:
            for tag in page.tags.values():
                metadata["TIFF Tags"][tag.name] = tag.value

    return metadata


def get_image_channel_names(file_path):
    if os.path.exists(file_path):
        metadata = get_metadata_tiff(file_path)
        image_description = metadata["TIFF Tags"]["ImageDescription"]
        root = ET.fromstring(image_description)
        namespace = {"ome": "http://www.openmicroscopy.org/Schemas/OME/2016-06"}
        description_text = root.find(".//ome:Description", namespace).text
        description_json = json.loads(description_text)
        channel_info = description_json.get("Channel", {}).get("Name", [])
    else:
        channel_info = []

    return channel_info
