#!/usr/bin/env python
"""
Extract CPM values for all unique CREs from ATAC and Histone bigwig files.
This script processes bigwig files and creates dataframes with CPM values for each modality.
Optimized with parallel processing for faster execution.
"""

import os
import re
import pandas as pd
import pyBigWig
import numpy as np
from joblib import Parallel, delayed
import warnings
warnings.filterwarnings("ignore")

# Number of parallel workers
N_WORKERS = 72

# Set up paths
PWD = os.path.dirname(os.path.abspath(__file__))
CRE_FILE = f'{PWD}/ATAC/subclass2CRE/all_unique_cCREs.txt'
CLUSTER_ANNOTATION = f'{PWD}/abc_atlas/cluster_annotation_term.csv'
ATAC_PATH = f'{PWD}/ATAC/snATACbw_bamCoverage/'
HISTONE_PATH = f'{PWD}/Histone/DNAbw/'
OUTPUT_DIR = f'{PWD}/CRE_CPM_matrices/'

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)


# Worker function to process one bigwig file for all CREs
def process_one_bigwig(args):
    """
    Process all CREs for a single bigwig file.
    Opens the bigwig once, loops through all CREs, then closes it.

    Parameters:
    -----------
    args : tuple
        (bigwig_path, celltype, cre_data, pad)
        where cre_data is a list of tuples (cre_id, chrom, start, end)

    Returns:
    --------
    tuple
        (celltype, pd.Series of CPM values)
    """
    bigwig_path, celltype, cre_data, pad = args

    # Open bigwig file once
    bw = pyBigWig.open(bigwig_path)
    total_signal = bw.header()['sumData']

    cpm_values = []
    cre_ids = []

    # Loop through all CREs sequentially
    for cre_id, chrom, start, end in cre_data:
        cre_ids.append(cre_id)
        try:
            # Extract signal sum in the padded region
            signal_sum = bw.stats(
                chrom,
                max(0, start - pad),
                end + pad,
                type='sum'
            )[0]

            # Calculate CPM (per 100k)
            if signal_sum is not None and total_signal > 0:
                cpm = signal_sum / total_signal * 1e5
            else:
                cpm = np.nan
        except:
            cpm = np.nan

        cpm_values.append(cpm)

    # Close bigwig file
    bw.close()

    # Return as series
    return (celltype, pd.Series(cpm_values, index=cre_ids, name=celltype))


# Main execution block
if __name__ == '__main__':
    print("=" * 60)
    print("CRE CPM Extraction Pipeline")
    print("=" * 60)

    # Load unique CREs
    print(f"\nLoading CREs from: {CRE_FILE}")
    with open(CRE_FILE, 'r') as f:
        cre_list = [line.strip() for line in f if line.strip()]

    print(f"Total CREs loaded: {len(cre_list):,}")

    # Parse CRE coordinates
    print("\nParsing CRE coordinates...")
    cre_info = []
    for cre in cre_list:
        if ':' in cre and '-' in cre:
            chrom = cre.split(':')[0]
            coords = re.split('[-−]', cre.split(':')[1])
            start = int(coords[0])
            end = int(coords[1])
            cre_info.append({
                'cre_id': cre,
                'chrom': chrom,
                'start': start,
                'end': end
            })

    cre_df = pd.DataFrame(cre_info)
    cre_df.set_index('cre_id', inplace=True)
    print(f"Successfully parsed {len(cre_df):,} CREs")

    # Load cluster annotation
    print(f"\nLoading cluster annotations from: {CLUSTER_ANNOTATION}")
    cluster_annotation_term = pd.read_csv(CLUSTER_ANNOTATION, index_col=0)
    cluster_annotation_term['subclass'] = cluster_annotation_term['subclass'].str.replace('/', '-')
    print(f"Loaded annotations for {len(cluster_annotation_term)} cell types")

    # Process bigwig files
    print("\n" + "=" * 60)
    print("Processing bigwig files...")
    print(f"Using {N_WORKERS} parallel workers")
    print("=" * 60)

    # Prepare CRE data once (list of tuples)
    cre_data = [(idx, row['chrom'], row['start'], row['end'])
                for idx, row in cre_df.iterrows()]

    for bigwig_path, modality in zip([ATAC_PATH, HISTONE_PATH],
                                     [['ATAC'], ['H3K27ac', 'H3K9me3', 'H3K4me1', 'H3K27me3']]):

        print(f"\nProcessing directory: {bigwig_path}")
        bigwig_files = os.listdir(bigwig_path)

        for mod in modality:
            print(f"\n  Modality: {mod}")

            # Determine the pattern based on modality
            if mod == 'H3K9me3':
                pattern = f'{mod}.e100.bs100.sm1000.bw'
            else:
                pattern = f'{mod}.e100.bs100.sm300.bw'

            # Find matching files and cell types
            matching_files = []
            celltypes = []

            for f in bigwig_files:
                if f.endswith(pattern):
                    subclass_number = int(f.split('_')[0])
                    if subclass_number in cluster_annotation_term['subclass_number'].values:
                        celltype = cluster_annotation_term.loc[
                            cluster_annotation_term['subclass_number'] == subclass_number,
                            'subclass'
                        ].values[0]
                        matching_files.append(f)
                        celltypes.append(celltype)

            print(f"    Found {len(matching_files)} bigwig files")

            if len(matching_files) == 0:
                print(f"    No files found for {mod}, skipping...")
                continue

            # Prepare tasks for parallel processing - one task per bigwig file
            tasks = []
            for f, celltype in zip(matching_files, celltypes):
                full_path = os.path.join(bigwig_path, f)
                tasks.append((full_path, celltype, cre_data, 500))

            # Process all bigwig files in parallel using joblib
            print(f"    Processing {len(tasks)} bigwig files in parallel...")
            results = Parallel(n_jobs=N_WORKERS, verbose=10)(
                delayed(process_one_bigwig)(task) for task in tasks
            )

            # Collect results into dataframe
            mod_df = pd.DataFrame(index=cre_df.index, columns=celltypes)
            for celltype, cpm_series in results:
                mod_df[celltype] = cpm_series

            # Save the dataframe
            output_file = f'{OUTPUT_DIR}/{mod}_cpm_peak_pad_500_Bysubclass.csv'
            print(f"    Saving to: {output_file}")
            mod_df.to_csv(output_file)
            print(f"    Saved dataframe with shape: {mod_df.shape}")

            # Print summary statistics
            non_nan_count = mod_df.notna().sum().sum()
            total_count = mod_df.size
            print(f"    Non-NaN values: {non_nan_count:,} / {total_count:,} ({non_nan_count/total_count*100:.1f}%)")

    print("\n" + "=" * 60)
    print("Pipeline completed successfully!")
    print("=" * 60)
    print(f"\nOutput files saved to: {OUTPUT_DIR}")
