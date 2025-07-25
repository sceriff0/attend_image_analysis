import pandas as pd
import numpy as np
import os
import json
import tifffile as tiff
from typing import List, Union
import argparse
from scipy.stats import norm, zscore


def remove_noise(df, col_num, z_sum_thres, z_count_thres):
    """
    Removes noisy cells from the dataset based on the given thresholds.

    Parameters
    ----------
    df : DataFrame
        The input data from which noisy cells are to be removed.
    col_num : int
        The column number up to which the operation is performed.
    z_sum_thres : float
        The threshold for the sum of z-scores. Cells with a sum of z-scores greater than this threshold are considered noisy.
    z_count_thres : int
        The threshold for the count of z-scores. Cells with a count of z-scores greater than this threshold are considered noisy.

    Returns
    -------
    df_want : DataFrame
        The cleaned data with noisy cells removed.
    cc : DataFrame
        The data of the noisy cells that were removed from the original data.

    """
    df_z_1_copy = df.copy()
    df_z_1_copy["Count"] = df_z_1_copy.iloc[:, : col_num + 1].ge(0).sum(axis=1)
    df_z_1_copy["z_sum"] = df_z_1_copy.iloc[:, : col_num + 1].sum(axis=1)
    cc = df_z_1_copy[
        (df_z_1_copy["z_sum"] > z_sum_thres) | (df_z_1_copy["Count"] > z_count_thres)
    ]
    df_want = df_z_1_copy[
        ~((df_z_1_copy["z_sum"] > z_sum_thres) | (df_z_1_copy["Count"] > z_count_thres))
    ]
    percent_removed = np.round(
        1 - (df_want.shape[0] / df_z_1_copy.shape[0]), decimals=3
    )
    print(str(percent_removed * 100) + "% cells are removed.")
    df_want.drop(columns=["Count", "z_sum"], inplace=True)
    df_want.reset_index(inplace=True, drop=True)
    return df_want, cc


def format(data, list_out, list_keep, method="zscore", ArcSin_cofactor=150):
    """
    This function formats the data based on the specified method. It supports four methods: "zscore", "double_zscore", "MinMax", and "ArcSin".

    Parameters
    ----------
    data : DataFrame
        The input data to be formatted.
    list_out : list
        The list of columns to be dropped from the data.
    list_keep : list
        The list of columns to be kept in the data.
    method : str, optional
        The method to be used for normalizing the data. It can be "zscore", "double_zscore", "MinMax", or "ArcSin". By default, it is "zscore".
    ArcSin_cofactor : int, optional
        The cofactor to be used in the ArcSin transformation. By default, it is 150.

    Returns
    -------
    DataFrame
        The formatted data.

    Raises
    ------
    ValueError
        If the specified method is not supported.
    """

    list = ["zscore", "double_zscore", "MinMax", "ArcSin"]

    if method not in list:
        print("Please select methods from zscore, double_zscore, MinMax, ArcSin!")
        exit()

    ##ArcSin transformation
    if method == "ArcSin":
        # Drop column list
        list1 = [col for col in data.columns if "blank" in col]
        list_out1 = list1 + list_out

        # Drop columns not interested in
        dfin = data.drop(list_out1, axis=1)

        # save columns for later
        df_loc = dfin.loc[:, list_keep]

        # dataframe for normalization
        dfas = dfin.drop(list_keep, axis=1)

        # parameters seit in function
        # Only decrease the background if the median is higher than the background
        dfa = dfas.apply(lambda x: np.arcsinh(x / ArcSin_cofactor))

        # Add back labels for normalization type
        dfz_all = pd.concat([dfa, df_loc], axis=1, join="inner")

        return dfz_all

    ##Double Z normalization
    elif method == "double_zscore":
        # Drop column list
        list1 = [col for col in data.columns if "blank" in col]
        list_out1 = list1 + list_out

        # Drop columns not interested in
        dfin = data.drop(list_out1, axis=1)

        # save columns for later
        df_loc = dfin.loc[:, list_keep]

        # dataframe for normalization
        dfz = dfin.drop(list_keep, axis=1)

        # zscore of the column markers
        dfz1 = pd.DataFrame(
            zscore(dfz, 0), index=dfz.index, columns=[i for i in dfz.columns]
        )

        # zscore rows
        dfz2 = pd.DataFrame(
            zscore(dfz1, 1), index=dfz1.index, columns=[i for i in dfz1.columns]
        )

        # Take cumulative density function to find probability of z score across a row
        dfz3 = pd.DataFrame(
            norm.cdf(dfz2), index=dfz2.index, columns=[i for i in dfz2.columns]
        )

        # First 1-probability and then take negative logarithm so greater values demonstrate positive cell type
        dflog = dfz3.apply(lambda x: -np.log(1 - x))

        # Add back labels for normalization type
        dfz_all = pd.concat([dflog, df_loc], axis=1, join="inner")

        # print("the number of regions = "+str(len(dfz_all.region_num.unique())))

        return dfz_all

    # Min Max normalization
    elif method == "MinMax":
        # Drop column list
        list1 = [col for col in data.columns if "blank" in col]
        list_out1 = list1 + list_out

        # Drop columns not interested in
        dfin = data.drop(list_out1, axis=1)

        # save columns for later
        df_loc = dfin.loc[:, list_keep]

        # dataframe for normalization
        dfmm = dfin.drop(list_keep, axis=1)

        for col in dfmm.columns:
            max_value = dfmm[col].quantile(0.99)
            min_value = dfmm[col].quantile(0.01)
            dfmm[col].loc[dfmm[col] > max_value] = max_value
            dfmm[col].loc[dfmm[col] < min_value] = min_value
            dfmm[col] = (dfmm[col] - min_value) / (max_value - min_value)

        # Add back labels for normalization type
        dfz_all = pd.concat([dfmm, df_loc], axis=1, join="inner")

        return dfz_all

    ## Z normalization
    else:
        # Drop column list
        list1 = [col for col in data.columns if "blank" in col]
        list_out1 = list1 + list_out

        # Drop columns not interested in
        dfin = data.drop(list_out1, axis=1)

        # save columns for later
        df_loc = dfin.loc[:, list_keep]

        # dataframe for normalization
        dfz = dfin.drop(list_keep, axis=1)

        # zscore of the column markers
        dfz1 = pd.DataFrame(
            zscore(dfz, 0), index=dfz.index, columns=[i for i in dfz.columns]
        )

        # Add back labels for normalization type
        dfz_all = pd.concat([dfz1, df_loc], axis=1, join="inner")

        # print("the number of regions = "+str(len(dfz_all.region_num.unique())))

        return dfz_all


def filter_data(
    df,
    nuc_thres=1,
    size_thres=1,
    nuc_marker="DAPI",
    cell_size="area",
):
    """
    Filter data based on nuclear threshold and size threshold, and visualize the data before and after filtering.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame to be filtered.
    nuc_thres : int, optional
        The nuclear threshold, by default 1.
    size_thres : int, optional
        The size threshold, by default 1.
    nuc_marker : str, optional
        The nuclear marker, by default "DAPI".
    cell_size : str, optional
        The cell size, by default "area".

    Returns
    -------
    df_nuc : pandas.DataFrame
        The filtered DataFrame.
    """
    df_nuc = df[(df[nuc_marker] > nuc_thres) * df[cell_size] > size_thres]
    per_keep = len(df_nuc) / len(df)

    # print the percentage of cells that are kept
    print("Percentage of cells kept: ", per_keep * 100, "%")

    return df_nuc


def zcount_thres(
    dfz, col_num, cut_off=0.01
):
    """
    Determines the threshold to use for removing noises. The default cut off is the top 1%.

    Parameters
    ----------
    dfz : DataFrame
        The input data from which the threshold is to be determined.
    col_num : int
        The column number up to which the operation is performed.
    cut_off : float, optional
        The cut off percentage for the threshold. By default, it is 0.01 (1%).
    count_bin : int, optional
        The number of bins for the count histogram. By default, it is 50.
    zsum_bin : int, optional
        The number of bins for the z-score sum histogram. By default, it is 50.
    figsize : tuple, optional
        The size of the figure to be plotted. By default, it is (10, 5).

    Returns
    -------
    None
        This function doesn't return anything. It plots two histograms for 'Count' and 'Zscore sum' with the cut off line.
    """
    dfz_copy = dfz
    dfz_copy["Count"] = dfz.iloc[:, : col_num + 1].ge(0).sum(axis=1)
    dfz_copy["z_sum"] = dfz.iloc[:, : col_num + 1].sum(axis=1)

    count_threshold = dfz_copy["Count"].quantile(1 - cut_off)
    z_sum_threshold = dfz_copy["z_sum"].quantile(1 - cut_off)

    return count_threshold, z_sum_threshold


def load_default_config():
    """Load default phenotyping configuration - EXACT copy from original script."""
    return {
        'MARK_LIST': [
            ['CD163', 'CD14'],
            'CD163',
            'CD14',
            'CD45',
            'CD3',
            'CD8',
            'CD4',
            'FOXP3',
            'PANCK',
            ['VIMENTIN', 'SMA'],
            'VIMENTIN',
            'SMA',
            'L1CAM',
            'PAX2'
        ],
        'CUTOFF_LIST': [
            [0.7, 0.9], 0.7, 0.9, 0.4, 0.2, 0.4, 0.9, 1.3, 0.2, 
            [0.2, 0.2], 0.2, 0.2, 0.3, 1
        ],
        'PHENO_LIST': [
            'Macrophages',
            'M2',
            'M1', 
            'Immune',
            'T cell',
            'T helper',
            'T cytotoxic',
            'T regulatory',
            'Tumor',
            'Stroma',
            'Stroma VIM',
            'Stroma SMA',
            'L1CAM+',
            'PAX2+'
        ],
        'IMMUNE_BRANCH0': ['CD3'],
        'IMMUNE_BRANCH1': ['CD8'],
        'IMMUNE_BRANCH2': ['CD4', 'FOXP3']
    }


def load_config_from_file(config_file: str) -> dict:
    """Load configuration from JSON file."""
    with open(config_file, 'r') as f:
        return json.load(f)


def export_anndata(data: pd.DataFrame, output_path: str) -> None:
    """Export data as AnnData H5AD file (requires spacec.hf module)."""
    try:
        from spacec.hf import make_anndata
        
        adata = make_anndata(
            df_nn=data,
            col_sum=data.columns.get_loc('DAPI'),  # Last protein feature column
            nonFuncAb_list=[]  # Remove non-working antibodies from clustering
        )
        
        adata.write_h5ad(output_path)
        print(f"AnnData exported to: {output_path}")
        
    except ImportError:
        print("Warning: spacec.hf module not available. Skipping H5AD export.")
    except Exception as e:
        print(f"Error exporting AnnData: {e}")


def labels_to_phenotype(arr, phenotype_df):
    map_arr = phenotype_df[['label', 'phenotype_num']].to_numpy()
    max_val = max(map_arr[:, 0].max(), arr.max()) + 1
    lookup = np.zeros(max_val + 1, dtype=map_arr[:, 1].dtype)
    lookup[map_arr[:, 0]] = map_arr[:, 1]
    remapped_arr = np.where(arr <= max_val, lookup[arr], arr)
    return remapped_arr


def normalize_data(cell_df, output_file=None, write=False):
    nuc_marker='DAPI'
    cell_size='area'
    nuc_percentile=1.0
    size_percentile=1.0
    cutoff=0.01
    normalization_method='zscore'
    last_marker='VIMENTIN'
    
    # Column order definition
    cols_order = [
        'y', 'x', 'eccentricity', 'perimeter', 'convex_area', 'area',
        'axis_major_length', 'axis_minor_length', 'region_num', 'label',
        'ARID1A', 'CD14', 'CD163', 'CD3', 'CD4', 'CD45', 'CD8', 'FOXP3',
        'L1CAM', 'P53', 'PANCK', 'PAX2', 'PD1', 'PDL1', 'SMA', 'VIMENTIN', 'DAPI'
    ]
    
    print('Cell dataframe columns: ', cell_df.columns.tolist())
    
    # Add region_num column if it doesn't exist
    if 'region_num' not in cell_df.columns:
        cell_df['region_num'] = 1
        print("Added 'region_num' column with default value 1")

    print('Columns in cell dataframe after adding region_num: ', cell_df.columns.tolist())

    # Reorder columns
    cell_df = cell_df[cols_order]
    
    # Calculate thresholds
    nuc_thres = np.percentile(cell_df[cell_size], nuc_percentile)
    size_thres = np.percentile(cell_df[nuc_marker], size_percentile)
    
    print(f"Mean {nuc_marker} intensity: {cell_df[nuc_marker].mean():.2f}")
    print(f"Mean {cell_size}: {cell_df[cell_size].mean():.2f}")
    print(f"Using nuclear threshold ({nuc_percentile}th percentile): {nuc_thres:.2f}")
    print(f"Using size threshold ({size_percentile}th percentile): {size_thres:.2f}")
    
    # Filter data
    print("Filtering data...")
    cell_df_filt = filter_data(
        cell_df, 
        nuc_thres=nuc_thres,
        size_thres=size_thres,
        nuc_marker=nuc_marker,
        cell_size=cell_size,
    )
    
    print(f'Columns in filtered dataframe: {cell_df_filt.columns.tolist()}')
    print(f"Selecting features to normalize for each region...")
    
    # Define lists for normalization
    list_out = ['eccentricity', 'perimeter', 'convex_area', 'axis_major_length', 'axis_minor_length']
    list_keep = ['DAPI', 'x', 'y', 'area', 'region_num', 'label']
    
    # Normalize data
    print(f"Normalizing data using {normalization_method} method...")
    dfz = format(
        data=cell_df_filt, 
        list_out=list_out, 
        list_keep=list_keep, 
        method=normalization_method
    )
    
    # Get column index for last marker
    if last_marker not in dfz.columns:
        raise ValueError(f"Last marker '{last_marker}' not found in columns: {dfz.columns.tolist()}")
    
    col_num_last_marker = dfz.columns.get_loc(last_marker)
    
    print(f"Getting z-count threshold for each region...")
    print(f"Using cut-off: {cutoff}")
    
    # Calculate thresholds for noise removal
    count_threshold, z_sum_threshold = zcount_thres(
        dfz=dfz, 
        col_num=col_num_last_marker,
        cut_off=cutoff,
    )
    
    print(f"Count threshold: {count_threshold:.2f}")
    print(f"Z-sum threshold: {z_sum_threshold:.2f}")
    print("First 5 rows of normalized data:")
    print(dfz.head())
    
    # Remove noise
    print("Removing noise...")
    df_nn, _ = remove_noise(
        df=dfz, 
        col_num=col_num_last_marker,
        z_count_thres=count_threshold,
        z_sum_thres=z_sum_threshold
    )
    
    print(f"Final data shape: {df_nn.shape}")
    print(f"Saving filtered and normalized data to: {output_file}")

    if write:
        # Save results
        df_nn.to_csv(output_file, index=False)
        print("Analysis completed successfully!")

    return df_nn


def define_phenotypes(normalized_cell_df, output_file=None, write=False):

    config_file = None
    if config_file:
        print(f"Loading configuration from: {config_file}")
        config = load_config_from_file(config_file)
        # Ensure config has the right key names
        MARK_LIST = config.get('MARK_LIST', config.get('markers', []))
        CUTOFF_LIST = config.get('CUTOFF_LIST', config.get('cutoffs', []))
        PHENO_LIST = config.get('PHENO_LIST', config.get('phenotypes', []))
        IMMUNE_BRANCH0 = config.get('IMMUNE_BRANCH0', ['CD3'])
        IMMUNE_BRANCH1 = config.get('IMMUNE_BRANCH1', ['CD8'])
        IMMUNE_BRANCH2 = config.get('IMMUNE_BRANCH2', ['CD4', 'FOXP3'])
    else:
        print("Using default configuration")
        config = load_default_config()
        MARK_LIST = config['MARK_LIST']
        CUTOFF_LIST = config['CUTOFF_LIST']
        PHENO_LIST = config['PHENO_LIST']
        IMMUNE_BRANCH0 = config['IMMUNE_BRANCH0']
        IMMUNE_BRANCH1 = config['IMMUNE_BRANCH1']
        IMMUNE_BRANCH2 = config['IMMUNE_BRANCH2']
    

    
    # Phenotype classification - EXACT copy of original logic
    normalized_cell_df['phenotype'] = 'Unclassified'
    print("Starting phenotype classification...")
    
    for marker, cutoff, phenotype in zip(MARK_LIST, CUTOFF_LIST, PHENO_LIST):
        if marker not in IMMUNE_BRANCH1 and marker not in IMMUNE_BRANCH2:
            df_class = normalized_cell_df[normalized_cell_df['phenotype'] == 'Unclassified']
        elif marker in IMMUNE_BRANCH0:
            df_class = normalized_cell_df[normalized_cell_df['phenotype'] == 'Immune']
        elif marker in IMMUNE_BRANCH1:
            df_class = normalized_cell_df[normalized_cell_df['phenotype'] == 'T cell']
        elif marker in IMMUNE_BRANCH2:
            if IMMUNE_BRANCH2.index(marker) < 1:
                df_class = normalized_cell_df[normalized_cell_df['phenotype'] == 'T cell']
            else:
                df_class = normalized_cell_df[normalized_cell_df['phenotype'] == 'T helper']
        
        if isinstance(marker, list):
            m0, m1 = marker[0], marker[1]
            c0, c1 = cutoff[0], cutoff[1]
            sel = df_class[(df_class[m0] >= c0) & (df_class[m1] >= c1)]
        else:
            sel = df_class[df_class[marker] >= cutoff]
        
        normalized_cell_df.loc[sel.index, 'phenotype'] = phenotype
    
    # Add numeric labels - EXACT copy of original logic
    pheno_complete = PHENO_LIST + ['Unclassified']
    for pp, p in enumerate(pheno_complete):
        sel = normalized_cell_df[normalized_cell_df['phenotype'] == p].index
        normalized_cell_df.loc[sel, 'phenotype_num'] = pp + 1
    
    # Results - EXACT copy of original logic
    counts = normalized_cell_df.groupby('phenotype').count()['x']
    
    print("Phenotype Counts:")
    print(counts)
    
    if write:
        print(f"Saving results to {output_file}")
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        normalized_cell_df.to_csv(output_file, index=True)

    df_pheno = normalized_cell_df

    df_pheno['phenotype_num'] = df_pheno['phenotype_num'].to_numpy().astype(int)

    return df_pheno

def create_phenotype_mask(df_pheno, cell_mask, output=None, write=False):
    #df_pheno = pd.read_csv(phenotype, index_col=0, header=0)
    #df_pheno['phenotype_num'] = df_pheno['phenotype_num'].astype(int)
    #cell_mask = np.load(mask)
    
    print("Remapping cell mask to phenotypes...")
    remapped_arr = labels_to_phenotype(cell_mask, df_pheno)

    if write:
        print("Saving remapped array to:", output)
        tiff.imwrite(output, remapped_arr)

    return remapped_arr



def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Cell phenotyping analysis based on marker expression',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Required arguments
    parser.add_argument(
        '--cell_data',
        required=True,
        help='Path to input CSV file containing cell data'
    )

    parser.add_argument(
        '--segmentation_mask',
        required=True,
        help='Path to input segmentation mask file (npy)'
    )

    parser.add_argument(
        '-o', '--output_dir',
        required=True,
        help='Path to output directory where results will be saved'
    )

    return parser.parse_args()



if __name__ == "__main__":

    args = parse_arguments()
    
    desired_columns = [
    'label',
    'y',
    'x',
    'eccentricity',
    'perimeter',
    'convex_area',
    'area',
    'axis_major_length',
    'axis_minor_length',
    'PAX2',
    'CD74',
    'L1CAM',
    'CD8',
    'PANCK',
    'CD45',
    'PDL1',
    'P53',
    'PD1',
    'FOXP3',
    'CD3',
    'CD163',
    'CD4',
    'GZMB',
    'ARID1A',
    'CD14',
    'SMA',
    'DAPI',
    'VIMENTIN'
    ]


    # Parse command line arguments

    cell_data = args.cell_data
    segmentation_mask = args.segmentation_mask
    
    print(f"Loading mask: {segmentation_mask}")
    cell_mask = np.load(segmentation_mask)
    print(f"Loading quantification data: {cell_data}")
    cell_df = pd.read_csv(cell_data)
    cell_df = cell_df[desired_columns] 

    print("Normalizing data")
    normalized_cell_df = normalize_data(cell_df)
    print("Computing phenotypes")
    df_pheno = define_phenotypes(normalized_cell_df, write=True, output_file=os.path.join(args.output_dir, 'phenotypes.csv'))   
    print("Creating phenotype mask")
    pheno_mask = create_phenotype_mask(df_pheno, cell_mask, write=True, output=os.path.join(args.output_dir, 'phenotype_mask.tiff'))
