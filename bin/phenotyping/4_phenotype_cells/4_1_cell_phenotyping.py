#!/usr/bin/env python3
"""
Cell Phenotyping Analysis Script
Classifies cells into different phenotypes based on marker expression thresholds.
"""

import pandas as pd
import os
import argparse
import json
from typing import List, Union


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Cell phenotyping analysis based on marker expression',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument(
        '-i', '--input-file',
        required=True,
        help='Path to input CSV file containing normalized cell data'
    )
    
    parser.add_argument(
        '-o', '--output-file',
        required=True,
        help='Path to output CSV file for phenotyped data'
    )
    
    # Optional arguments
    parser.add_argument(
        '--config-file',
        help='Path to JSON configuration file with markers, cutoffs, and phenotypes'
    )
    
    parser.add_argument(
        '--export-h5ad',
        action='store_true',
        help='Export results as AnnData H5AD file (requires spacec.hf module)'
    )
    
    return parser.parse_args()


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


def main():
    """Main analysis function - EXACT copy of original logic."""
    args = parse_arguments()
    
    # Check if input file exists
    if not os.path.exists(args.input_file):
        raise FileNotFoundError(f"Input file not found: {args.input_file}")
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    
    # Load configuration
    if args.config_file:
        print(f"Loading configuration from: {args.config_file}")
        config = load_config_from_file(args.config_file)
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
    
    print(f"Input file: {args.input_file}")
    
    # Load data - EXACT same as original
    df_pheno = pd.read_csv(args.input_file, index_col=0, header=0)
    
    # Phenotype classification - EXACT copy of original logic
    df_pheno['phenotype'] = 'Unclassified'
    print("Starting phenotype classification...")
    
    for marker, cutoff, phenotype in zip(MARK_LIST, CUTOFF_LIST, PHENO_LIST):
        if marker not in IMMUNE_BRANCH1 and marker not in IMMUNE_BRANCH2:
            df_class = df_pheno[df_pheno['phenotype'] == 'Unclassified']
        elif marker in IMMUNE_BRANCH0:
            df_class = df_pheno[df_pheno['phenotype'] == 'Immune']
        elif marker in IMMUNE_BRANCH1:
            df_class = df_pheno[df_pheno['phenotype'] == 'T cell']
        elif marker in IMMUNE_BRANCH2:
            if IMMUNE_BRANCH2.index(marker) < 1:
                df_class = df_pheno[df_pheno['phenotype'] == 'T cell']
            else:
                df_class = df_pheno[df_pheno['phenotype'] == 'T helper']
        
        if isinstance(marker, list):
            m0, m1 = marker[0], marker[1]
            c0, c1 = cutoff[0], cutoff[1]
            sel = df_class[(df_class[m0] >= c0) & (df_class[m1] >= c1)]
        else:
            sel = df_class[df_class[marker] >= cutoff]
        
        df_pheno.loc[sel.index, 'phenotype'] = phenotype
    
    # Add numeric labels - EXACT copy of original logic
    pheno_complete = PHENO_LIST + ['Unclassified']
    for pp, p in enumerate(pheno_complete):
        sel = df_pheno[df_pheno['phenotype'] == p].index
        df_pheno.loc[sel, 'phenotype_num'] = pp + 1
    
    # Results - EXACT copy of original logic
    counts = df_pheno.groupby('phenotype').count()['x']
    
    print("Phenotype Counts:")
    print(counts)
    
    # Export H5AD if requested
    if args.export_h5ad:
        h5ad_path = args.output_file.replace('.csv', '.h5ad')
        export_anndata(df_pheno, h5ad_path)
    
    # Save results - EXACT same as original
    print(f"Saving results to {args.output_file}")
    df_pheno.to_csv(args.output_file, index=True)


if __name__ == "__main__":
    main()