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
        "DAPI",
        "PANCK",
        "MLH1",
        "P53",
        "ARID1A",
        "PAX2",
        "VIMENTIN",
        "SMA", # Alpha-SMA
        "CD163",
        "CD14",
        "CD45",
        "CD3",
        "CD4",
        "CD8",
        "FOXP3",
        "PD1",
        "PDL1"
    ]

    return channels_list

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
        metadata['TIFF Tags'] = {}
        for page in tiff.pages:
            for tag in page.tags.values():
                metadata['TIFF Tags'][tag.name] = tag.value

    
    return metadata

def get_image_channel_names(file_path):
    if os.path.exists(file_path):
        metadata = get_metadata_tiff(file_path)
        image_description = metadata['TIFF Tags']['ImageDescription']
        root = ET.fromstring(image_description)
        namespace = {'ome': 'http://www.openmicroscopy.org/Schemas/OME/2016-06'}
        description_text = root.find('.//ome:Description', namespace).text
        description_json = json.loads(description_text)
        channel_info = description_json.get('Channel', {}).get('Name', [])
    else:
        channel_info = []

    return channel_info


