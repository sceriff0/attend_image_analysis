import numpy as np
import pandas as pd
import argparse

def labels_to_phenotype(arr, phenotype_df):
    map_arr = phenotype_df[['label', 'phenotype_num']].to_numpy()
    max_val = max(map_arr[:, 0].max(), arr.max()) + 1
    lookup = np.zeros(max_val + 1, dtype=map_arr[:, 1].dtype)
    lookup[map_arr[:, 0]] = map_arr[:, 1]
    remapped_arr = np.where(arr <= max_val, lookup[arr], arr)
    return remapped_arr

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Remap segmentation labels to phenotype values.")
    parser.add_argument("--phenotype", required=True, help="Path to phenotype CSV file.")
    parser.add_argument("--mask", required=True, help="Path to segmentation mask .npy file.")
    parser.add_argument("--output", required=True, help="Path to save remapped output .npy file.")

    args = parser.parse_args()

    df_pheno = pd.read_csv(args.phenotype, index_col=0, header=0)
    df_pheno['phenotype_num'] = df_pheno['phenotype_num'].astype(int)
    cell_mask = np.load(args.mask)

    print("Remapping cell mask to phenotypes...")
    remapped_arr = labels_to_phenotype(cell_mask, df_pheno)

    print("Saving remapped array to:", args.output)
    np.save(args.output, remapped_arr)

