# %%
import os
import sys
from pathlib import Path
# import functions from get
PWD = Path(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PWD)
from gcell._settings import get_setting
from get_preprocess_utils import (
    add_atpm,
    add_exp,
    download_motif,
    get_motif,
    query_motif,
)
import pandas as pd
# %%
annotation_dir = Path(f'{PWD}/Data/annotation')
motif_bed_url = "https://resources.altius.org/~jvierstra/projects/motif-clustering/releases/v1.0/hg38.archetype_motifs.v1.0.bed.gz"
motif_bed_index_url = "https://resources.altius.org/~jvierstra/projects/motif-clustering/releases/v1.0/hg38.archetype_motifs.v1.0.bed.gz.tbi"
if (
    motif_bed_url
    and motif_bed_index_url
    and not (
        (annotation_dir / "mm10.archetype_motifs.v1.0.bed.gz").exists()
        or (annotation_dir / "mm10.archetype_motifs.v1.0.bed.gz.tbi").exists()
    )
):
    download_motif(motif_bed_url, motif_bed_index_url, motif_dir=annotation_dir)
    motif_bed = str(annotation_dir / "mm10.archetype_motifs.v1.0.bed.gz")
else:
    motif_bed = str(annotation_dir / "mm10.archetype_motifs.v1.0.bed.gz")
# %%
peak_bed = "Data/CRE.bed" 
peaks_motif = query_motif(peak_bed, motif_bed)
get_motif_output = get_motif(peak_bed, peaks_motif, assembly='mm10')
# %%
def create_peak_motif(peak_motif_bed, output_zarr, peak_bed):
    """
    Create a peak motif zarr file from a peak motif bed file.

    This function reads a peak motif bed file, pivots the data, and saves it to a zarr file.
    The zarr file contains three datasets: 'data', 'peak_names', 'motif_names', and 'accessibility'.
    The 'data' dataset is a sparse matrix containing the peak motif data.
    The 'peak_names' dataset contains the peak names.
    The 'motif_names' dataset contains the motif names.

    Args:
        peak_motif_bed (str): Path to the peak motif bed file.
        output_zarr (str): Path to the output zarr file.
    """
    # Read the peak motif bed file
    peak_motif = pd.read_csv(
        peak_motif_bed,
        sep="\t",
        header=None,
        names=["Chromosome", "Start", "End", "Motif_cluster", "Score"],
    )

    # Pivot the data
    peak_motif_pivoted = peak_motif.pivot_table(
        index=["Chromosome", "Start", "End"],
        columns="Motif_cluster",
        values="Score",
        fill_value=0,
    )
    peak_motif_pivoted.reset_index(inplace=True)

    # Create the 'Name' column
    peak_motif_pivoted["Name"] = peak_motif_pivoted.apply(
        lambda x: f'{x["Chromosome"]}:{x["Start"]}-{x["End"]}', axis=1
    )
    peak_motif_pivoted = peak_motif_pivoted.drop(columns=["Chromosome", "Start", "End"])
    # Read the original peak bed file
    original_peaks = pd.read_csv(
        peak_bed, sep="\t", header=None, names=["Chromosome", "Start", "End", "Score"]
    )
    # exclude chrM and chrY
    original_peaks = original_peaks[~original_peaks.Chromosome.isin(["chrM", "chrY"])]
    original_peaks["Name"] = original_peaks.apply(
        lambda x: f'{x["Chromosome"]}:{x["Start"]}-{x["End"]}', axis=1
    )

    # Merge the pivoted data with the original peaks
    merged_data = pd.merge(original_peaks, peak_motif_pivoted, on="Name", how="left")

    # Fill NaN values with 0 for motif columns
    motif_columns = [
        col
        for col in merged_data.columns
        if col not in ["Chromosome", "Start", "End", "Score", "Name"]
    ]
    merged_data[motif_columns] = merged_data[motif_columns].fillna(0)
    return merged_data
# %%
motif_mat = create_peak_motif(get_motif_output, "pbmc10k_multiome.zarr", peak_bed)
# %%
motif_mat.to_csv('results/CRE_motif.csv', index=False)
# %%
