# import spacec as sp
import pandas as pd
import numpy as np
import os
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


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='SPACEC cell data analysis and normalization',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument(
        '-i', '--input-file',
        required=True,
        help='Path to input CSV file containing cell data'
    )
    
    parser.add_argument(
        '-o', '--output-file',
        required=True,
        help='Path to output CSV file for normalized data'
    )
    
    # Optional arguments
    parser.add_argument(
        '--nuc-marker',
        default='DAPI',
        help='Name of nuclear marker column'
    )
    
    parser.add_argument(
        '--cell-size',
        default='area',
        help='Name of cell size column'
    )
    
    parser.add_argument(
        '--nuc-percentile',
        type=float,
        default=1.0,
        help='Percentile for nuclear threshold (0-100)'
    )
    
    parser.add_argument(
        '--size-percentile',
        type=float,
        default=1.0,
        help='Percentile for size threshold (0-100)'
    )
    
    parser.add_argument(
        '--cutoff',
        type=float,
        default=0.01,
        help='Cut-off percentage for noise removal (0-1)'
    )
    
    parser.add_argument(
        '--normalization-method',
        choices=['zscore', 'double_zscore', 'MinMax', 'ArcSin'],
        default='zscore',
        help='Normalization method to use'
    )
    
    parser.add_argument(
        '--last-marker',
        default='VIMENTIN',
        help='Name of the last marker column for analysis'
    )
    
    return parser.parse_args()


def main():
    """Main analysis function."""
    args = parse_arguments()
    
    # Column order definition
    cols_order = [
        'y', 'x', 'eccentricity', 'perimeter', 'convex_area', 'area',
        'axis_major_length', 'axis_minor_length', 'region_num', 'label',
        'ARID1A', 'CD14', 'CD163', 'CD3', 'CD4', 'CD45', 'CD8', 'FOXP3',
        'L1CAM', 'P53', 'PANCK', 'PAX2', 'PD1', 'PDL1', 'SMA', 'VIMENTIN', 'DAPI'
    ]
    
    # Check if input file exists
    if not os.path.exists(args.input_file):
        raise FileNotFoundError(f"Input file not found: {args.input_file}")
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    
    print(f"Loading data from: {args.input_file}")
    cell_df = pd.read_csv(args.input_file, header=0)
    
    print('Cell dataframe columns: ', cell_df.columns.tolist())
    
    # Add region_num column if it doesn't exist
    if 'region_num' not in cell_df.columns:
        cell_df['region_num'] = 1
        print("Added 'region_num' column with default value 1")

    print('Columns in cell dataframe after adding region_num: ', cell_df.columns.tolist())

    # Reorder columns
    cell_df = cell_df[cols_order]
    
    # Calculate thresholds
    nuc_thres = np.percentile(cell_df[args.cell_size], args.nuc_percentile)
    size_thres = np.percentile(cell_df[args.nuc_marker], args.size_percentile)
    
    print(f"Mean {args.nuc_marker} intensity: {cell_df[args.nuc_marker].mean():.2f}")
    print(f"Mean {args.cell_size}: {cell_df[args.cell_size].mean():.2f}")
    print(f"Using nuclear threshold ({args.nuc_percentile}th percentile): {nuc_thres:.2f}")
    print(f"Using size threshold ({args.size_percentile}th percentile): {size_thres:.2f}")
    
    # Filter data
    print("Filtering data...")
    cell_df_filt = filter_data(
        cell_df, 
        nuc_thres=nuc_thres,
        size_thres=size_thres,
        nuc_marker=args.nuc_marker,
        cell_size=args.cell_size,
    )
    
    print(f'Columns in filtered dataframe: {cell_df_filt.columns.tolist()}')
    print(f"Selecting features to normalize for each region...")
    
    # Define lists for normalization
    list_out = ['eccentricity', 'perimeter', 'convex_area', 'axis_major_length', 'axis_minor_length']
    list_keep = ['DAPI', 'x', 'y', 'area', 'region_num', 'label']
    
    # Normalize data
    print(f"Normalizing data using {args.normalization_method} method...")
    dfz = format(
        data=cell_df_filt, 
        list_out=list_out, 
        list_keep=list_keep, 
        method=args.normalization_method
    )
    
    # Get column index for last marker
    if args.last_marker not in dfz.columns:
        raise ValueError(f"Last marker '{args.last_marker}' not found in columns: {dfz.columns.tolist()}")
    
    col_num_last_marker = dfz.columns.get_loc(args.last_marker)
    
    print(f"Getting z-count threshold for each region...")
    print(f"Using cut-off: {args.cutoff}")
    
    # Calculate thresholds for noise removal
    count_threshold, z_sum_threshold = zcount_thres(
        dfz=dfz, 
        col_num=col_num_last_marker,
        cut_off=args.cutoff,
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
    print(f"Saving filtered and normalized data to: {args.output_file}")
    
    # Save results
    df_nn.to_csv(args.output_file, index=False)
    print("Analysis completed successfully!")


if __name__ == "__main__":
    main()