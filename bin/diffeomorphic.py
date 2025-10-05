#!/usr/bin/env python
# Compute diffeomorphic transformation matrix

import argparse
import os
import numpy as np
import logging
import hashlib
from utils.io import load_pickle, save_h5
from utils.mapping import compute_diffeomorphic_mapping_dipy, apply_mapping
from utils import logging_config

from stardist.models import StarDist2D
from csbdeep.utils import normalize
from skimage import segmentation
from skimage import morphology
from skimage.filters import gaussian
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# Set up logging configuration
logging_config.setup_logging()
logger = logging.getLogger(__name__)

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

def are_all_alphabetic_lowercase(string):
            # Filter alphabetic characters and check if all are lowercase
            return all(char.islower() for char in string if char.isalpha())

def remove_lowercase_channels(channels):
            filtered_channels = []
            for ch in channels:
                if not are_all_alphabetic_lowercase(ch):
                    filtered_channels.append(ch)
            return filtered_channels

def _parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-ch",
        "--channels_to_register",
        type=str,
        default=None,
        required=True,
        help="Pickle file containing list of image channels to register",
    )
    parser.add_argument(
        "-cr",
        "--crop_image",
        type=str,
        default=None,
        required=True,
        help="Pickle file containing fixed and moving crops, in this order.",
    )
    parser.add_argument(
        "-m",
        "--moving_image",
        type=str,
        default=None,
        required=True,
        help="h5 image file",
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

    handler = logging.FileHandler(args.log_file)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    model = StarDist2D(None, name="stardist_full_e200_lr00001_aug1_seed10_es50p0.001_rlr0.5p50", basedir="/hpcnfs/scratch/P_DIMA_ATTEND/models/")
    model.config.use_gpu = True

    moving_channels = os.path.basename(args.moving_image) \
        .split('.')[0] \
        .split('_')[2:][::-1] 
    
    logger.debug(f'DIFFEOMORPHIC - MOVING CHANNELS: {moving_channels}')

    moving_channels_no_dapi = [ch for ch in moving_channels if ch != 'DAPI']

    channels_to_register = load_pickle(args.channels_to_register)
    current_channels_to_register = remove_lowercase_channels(moving_channels)
    current_channels_to_register_no_dapi = [ch for ch in current_channels_to_register if ch != 'DAPI']

    # if len(current_channels_to_register_no_dapi) == 3:
    #     patient_id = os.path.basename(args.crop_image).split('.')[0].split('_')[2]
    # elif len(current_channels_to_register_no_dapi) == 2:
    #     patient_id = os.path.basename(args.crop_image).split('.')[0].split('_')[1]

    patient_id = args.moving_image.split('.')[0].split('_')[1]

    crop_id_pos = os.path.basename(args.crop_image).split('.')[0].split('_')[0:3]
    crop_id_pos = [str(e) for e in crop_id_pos]
    crop_name = crop_id_pos + current_channels_to_register_no_dapi[::-1]   
    crop_name = '_'.join(crop_name)

    output_path = f"registered_{crop_name}.h5"
    output_path = output_path.replace('padded_', '')
    
    output_path_dapi = f"qc_{'_'.join(crop_id_pos)}_DAPI.h5"
    output_path_dapi = output_path_dapi.replace('padded_', '')
    
    if current_channels_to_register_no_dapi:
        if any([e for e in current_channels_to_register_no_dapi if e in channels_to_register]):
            fixed, moving = load_pickle(args.crop_image)
            if len(np.unique(moving[:,:,-1])) != 1 and len(np.unique(fixed[:,:,-1])) != 1:
                logger.debug(f"Computing mapping: {args.crop_image}")
                mapping = compute_diffeomorphic_mapping_dipy(
                    y=fixed[:, :, -1].squeeze(), 
                    x=moving[:, :, -1].squeeze()
                )

                normalized_fixed = normalize(dapi_preprocessing(fixed[:, :, -1]), 1.0, 99.8, axis=(0,1))
                pred_fixed, _ = model.predict_instances(normalized_fixed[:, :, -1], n_tiles=(8,8), verbose=False)
                expanded_pred_fixed = segmentation.expand_labels(pred_fixed, distance=10, spacing=1)

                normalized_moving = normalize(dapi_preprocessing(moving[:, :, -1]), 1.0, 99.8, axis=(0,1))
                pred_moving, _ = model.predict_instances(normalized_moving[:, :, -1], n_tiles=(8,8), verbose=False)
                expanded_pred_moving = segmentation.expand_labels(pred_moving, distance=10, spacing=1)

                registered_moving_labels = apply_mapping(mapping, expanded_pred_moving)
                iou = segmentation.compare_labels(expanded_pred_fixed, registered_moving_labels, method='jaccard')
                
                # Make debug dir inside work dir
                if not os.path.exists('debug'):
                    os.makedirs('debug')
                # Save iou to text file
                np.savetxt(f'debug/iou_{crop_name}.txt', [iou])
#                
                # Save registered dapi channel for quality control
                save_h5(
                    np.squeeze(apply_mapping(mapping, moving[:, :, -1])), 
                    output_path_dapi
                )

                logger.debug(f"Applying mapping: {args.crop_image}")
                registered_images = []

                for idx, ch in enumerate(moving_channels_no_dapi):
                    if ch in current_channels_to_register_no_dapi:
                        registered_images.append(apply_mapping(mapping, moving[:, :, idx]))

                registered_images = np.stack(registered_images, axis=-1)

                logger.debug(f"Saving registered image: {args.crop_image}")
                save_h5(
                    registered_images, 
                    output_path
                )

            elif len(np.unique(moving[:,:,-1])) == 1 or len(np.unique(fixed[:,:,-1])) == 1:
                moving_channels_images = []
                for idx, ch in enumerate(moving_channels_no_dapi):
                    if ch in current_channels_to_register_no_dapi:
                        moving_channels_images.append(moving[:, :, idx])

                moving_channels_images = np.stack(moving_channels_images, axis=-1)

                logger.debug(f"MOVING STACKED IMAGE SHAPE: {moving_channels_images.shape}")

                logger.debug(f"Saving empty crop (unregistered): {args.crop_image}")
                save_h5(
                    moving_channels_images, 
                    output_path
                )
                save_h5(
                    np.squeeze(moving[:,:,-1]), 
                    output_path_dapi
                )
        else:
            # Generate random bytes
            random_data = os.urandom(16)

            # Create a hash object using SHA256
            hash_object = hashlib.sha256(random_data)

            # Get the hexadecimal representation of the hash
            random_hash = hash_object.hexdigest()
            save_h5(
                0, 
                f"registered_0_0_{patient_id}_{random_hash}.h5"
            )
            save_h5(
                0, 
                f"qc_0_0_{patient_id}_{random_hash}.h5"
            )
    else:
        # Generate random bytes
        random_data = os.urandom(16)

        # Create a hash object using SHA256
        hash_object = hashlib.sha256(random_data)

        # Get the hexadecimal representation of the hash
        random_hash = hash_object.hexdigest()
        if not os.path.exists(f"registered_{patient_id}_{random_hash}.h5"):
            save_h5(
                0, 
                f"registered_{patient_id}_{random_hash}.h5"
            )
            save_h5(
                0, 
                f"qc_{patient_id}_{random_hash}.h5"
            )

if __name__ == "__main__":
    main()

