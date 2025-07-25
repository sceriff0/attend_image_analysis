import numpy as np
import pandas as pd
import os
import pickle 
import argparse 
from skimage.measure import regionprops_table
from aicsimageio import AICSImage

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


def extract_features(
    channels_files,
    segmentation_mask,
    output_file,
    size_cutoff=0,
    crop_positions=None,
    verbose=True
):
    """
    Extract morphological and mean intensity features from segmented nuclei.

    Parameters
    ----------
    channels_files : list of str
        Paths to images representing different channels (e.g., DAPI, GFP).
    segmentation_mask : ndarray
        2D NumPy array with labeled segmentation mask.
    output_file : str
        Path to save the output CSV.
    size_cutoff : int, optional
        Minimum number of pixels for a nucleus to be included. Default is 0.
    crop : tuple of int, optional
        (row_start, row_end, col_start, col_end) cropping window.
    verbose : bool, optional
        Whether to print progress messages. Default is True.

    Returns
    -------
    markers : pd.DataFrame
        DataFrame with morphological and intensity features for each nucleus.
    """

    def log(msg):
        if verbose:
            print(msg)

    segmentation_mask = segmentation_mask.squeeze()
    markers_df_list = []

    print('Channels files:', channels_files)

    for idx, pos in enumerate(crop_positions):
        print(f'Progress: {idx + 1}/{len(crop_positions)}')
        print('Processing crop:', pos)
        segmentation_mask_current = segmentation_mask[pos[0]:pos[1], pos[2]:pos[3]]
        log(f"Applied crop: {pos}")
    
        log(f"Segmentation mask shape: {segmentation_mask_current.shape}")
    
        # Count pixels per label
        labels, counts = np.unique(segmentation_mask_current, return_counts=True)
        valid_ids = labels[(labels != 0) & (counts > size_cutoff)]
        log(f"Valid nuclei count: {len(valid_ids)}")
    
        # Filter mask
        mask_filtered = np.where(np.isin(segmentation_mask_current, valid_ids), segmentation_mask_current, 0)

        print(f'All zero mask_filtered?: {(mask_filtered == 0).all()}')

        if (mask_filtered == 0).all():
            continue
        else:
            # Extract morphology
            log("Extracting morphological features...")
            props = regionprops_table(
                mask_filtered,
                properties=[
                    "label",
                    "centroid",
                    "eccentricity",
                    "perimeter",
                    "convex_area",
                    "area",
                    "axis_major_length",
                    "axis_minor_length",
                ]
            )
            props_df = pd.DataFrame(props).set_index("label", drop=False)
        
            # Preallocate mean intensity matrix
            mean_intensities = np.zeros((len(valid_ids), len(channels_files)), dtype=np.float32)
        
            for idx, file in enumerate(channels_files):
                chan_name = os.path.basename(file).split('.')[0].split('_')[-1]
        
                # Import image
                channel_data, _ = import_images(file)
                image = channel_data.get_image_data("YX")
        
                if crop_positions is not None:
                    image = image[pos[0]:pos[1], pos[2]:pos[3]]
        
                log(f"Quantifying channel '{chan_name}'...")
        
                # Compute mean intensity per label
                flat_mask = mask_filtered.ravel()
                flat_image = image.ravel()
        
                sum_per_label = np.bincount(flat_mask, weights=flat_image)
                count_per_label = np.bincount(flat_mask)
        
                with np.errstate(divide='ignore', invalid='ignore'):
                    means = np.true_divide(sum_per_label, count_per_label)
                    means[np.isnan(means)] = 0
        
                # Store intensities by valid label
                mean_values = np.array([means[label] if label < len(means) else 0 for label in valid_ids])
                mean_intensities[:, idx] = mean_values
    
        # Channel names and DataFrame
        chan_names = [os.path.basename(f).split('.')[0].split('_')[-1] for f in channels_files]
        intensity_df = pd.DataFrame(mean_intensities, index=valid_ids, columns=chan_names)

        print('Intensity df shape: ', intensity_df.shape)
        print('Intensity df head: ', intensity_df.head())
    
        # Combine and adjust centroids
        markers = intensity_df.join(props_df)
        markers.rename(columns={"centroid-0": "y", "centroid-1": "x"}, inplace=True)
    
        markers["y"] += pos[0]
        markers["x"] += pos[2]

        print('Markers df shape: ', markers.shape)
        markers_df_list.append(markers)

    # Concatenate all markers DataFrames
    markers_all = pd.concat(markers_df_list, axis=0).drop_duplicates(subset="label")
    
    # Save to CSV
    markers_all.to_csv(output_file, index=False)
    log(f"Saved merged features to: {output_file}")

    return markers_all




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



def main():
    parser = argparse.ArgumentParser(description='Extract markers data from image channels and segmentation mask.')
    parser.add_argument('--channels_dir', type=str, required=True, help='Directory containing channel image files.')
    parser.add_argument('--positions_file', type=str, required=True, help='Path to pickle file of crop positions.')
    parser.add_argument('--mask_file', type=str, required=True, help='Path to .npy file of segmentation mask.')
    parser.add_argument('--outdir', type=str, required=True, help='Output directory to save results.')

    args = parser.parse_args()

    print('Started processing')

    indir = args.channels_dir
    outdir = args.outdir
    positions_file = args.positions_file
    mask_file = args.mask_file

    # patient_id = os.path.basename(mask_file).split('_')[0]

    os.makedirs(outdir, exist_ok=True)

    print('Loading segmentation mask')
    segmentation_mask = np.load(mask_file)

    positions = load_pickle(positions_file)
    print('Number of crops to process', len(positions))

    files = [os.path.join(indir, file) for file in os.listdir(indir)]
    # selected_files = [file for file in files if patient_id in file]

    print('Extracting markers data')
    markers_data = extract_features(
        channels_files=files, 
        segmentation_mask=segmentation_mask, 
        output_file=os.path.join(outdir, f"segmentation_markers_data_FULL.csv"), 
        crop_positions=positions
    )

    print('Markers data shape:', markers_data.shape)
    print('Snipped markers data:')
    print(markers_data.head())
    print("Columns in markers_data:", markers_data.columns.tolist())

    # save_pickle(markers_df_list, os.path.join(outdir, f"segmentation_markers_data_list.pkl"))

    

if __name__ == '__main__':
    main()



