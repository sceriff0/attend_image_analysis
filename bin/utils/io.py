#!/usr/bin/env python

import h5py
import pickle
import nd2


## H5
def load_h5(path, loading_region=None, channels_to_load=None, shape='YXC'):
    with h5py.File(path, 'r') as hdf5_file:
        dataset = hdf5_file['dataset']

        # Handle empty datasets
        if dataset.shape == ():
            return dataset[()]

        # Define default slicing
        slices = [slice(None), slice(None), slice(None)]
        
        if loading_region:
            start_row, end_row, start_col, end_col = loading_region
            if shape == 'YXC':
                slices[0] = slice(start_row, end_row)
                slices[1] = slice(start_col, end_col)
            elif shape == 'CYX':
                slices[1] = slice(start_row, end_row)
                slices[2] = slice(start_col, end_col)

        if channels_to_load is not None:
            if shape == 'YXC':
                slices[2] = channels_to_load
            elif shape == 'CYX':
                slices[0] = channels_to_load

        # Extract data using slices
        data = dataset[tuple(slices)]
    
    return data

# def load_h5(path, loading_region=None, channels_to_load=None, shape='YXC'):
#     with h5py.File(path, 'r') as hdf5_file:
#         dataset = hdf5_file['dataset']
# 
#         if loading_region is not None:
#                 start_row, end_row, start_col, end_col = loading_region
#         if not dataset.shape == ():
# 
#             if shape == 'YXC':
#                 # Select region to load if loading_region is provided
#                 if loading_region is not None and channels_to_load is not None:
#                     data = dataset[start_row:end_row, start_col:end_col, channels_to_load]
#                 elif loading_region is None and channels_to_load is not None:
#                     data = dataset[:, :, channels_to_load]
#                 elif loading_region is not None and channels_to_load is None:
#                     data = dataset[start_row:end_row, start_col:end_col, :]
#                 else:
#                     data = dataset[:, :, :]
#             
#             if shape == 'CYX':
#                 # Select region to load if loading_region is provided
#                 if loading_region is not None and channels_to_load is not None:
#                     data = dataset[channels_to_load, start_row:end_row, start_col:end_col]
#                 elif loading_region is None and channels_to_load is not None:
#                     data = dataset[channels_to_load, :, :]
#                 elif loading_region is not None and channels_to_load is None:
#                     data = dataset[:, start_row:end_row, start_col:end_col]
#                 else:
#                     data = dataset[:, :, :]
#         else:
#             data = dataset[()]
# 
#     return data

def save_h5(data, path, ndim=3):
    if isinstance(data, int): 
        chunks = None
        maxshape = None
    else:
        chunks = True
        maxshape = tuple([None] * ndim)

    with h5py.File(path, 'w') as hdf5_file:
        hdf5_file.create_dataset(
            'dataset', 
            data=data, 
            chunks=chunks, 
            maxshape=maxshape
        )
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

