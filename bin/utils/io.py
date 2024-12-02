#!/usr/bin/env python

import h5py
import pickle
import nd2

## H5
def load_h5(path, loading_region=None, channels_to_load=None):
    with h5py.File(path, 'r') as hdf5_file:
        dataset = hdf5_file['dataset']
        
        # Select region to load if loading_region is provided
        if loading_region is not None:
            start_row, end_row, start_col, end_col = loading_region
            data = dataset[start_row:end_row, start_col:end_col, :]
        else:
            data = dataset[:, :, :]
        
        # Select channels if channels_to_load is provided
        if channels_to_load is not None:
            data = data[:, :, channels_to_load]

    return data

def save_h5(data, path, chunks=None):
    # Save the NumPy array to an HDF5 file
    with h5py.File(path, "w") as hdf5_file:
        hdf5_file.create_dataset("dataset", data=data, chunks=chunks)
        hdf5_file.flush()


## PICKLE
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

## ND2
def load_nd2(file_path):
    """
    Read an ND2 file and return the image array.

    Parameters:
    file_path (str): Path to the ND2 file

    Returns:
    numpy.ndarray: Image data
    """
    with nd2.ND2File(file_path) as nd2_file:
        data = nd2_file.asarray()
        data = data.transpose((1, 2, 0))

    return data

