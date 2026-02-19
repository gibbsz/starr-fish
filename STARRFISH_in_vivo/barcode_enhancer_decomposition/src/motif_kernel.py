"""Motif score-based distance and kernel computation.

Supports two workflows:
1. Load a pre-computed sequence × motif matrix from CSV (e.g., from Vierstra
   coordinate-based scanning).
2. Scan raw DNA sequences against a motif database using FIMO to create the
   matrix on-the-fly.  This works for both barcodes and enhancers.
"""

import os
import subprocess
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform


def load_motif_matrix(motif_csv, cre_ids=None):
    """Load a CRE × motif score matrix from CSV.

    Parameters
    ----------
    motif_csv : str
        Path to CSV with CRE IDs as index and motif clusters as columns.
        Extra columns (Chromosome, Start, End, Score, Name) are dropped.
    cre_ids : list of str, optional
        If given, subset and reorder to these CRE IDs.

    Returns
    -------
    motif_mat : pd.DataFrame
        Shape (n_CREs, n_motifs), indexed by CRE ID.
    """
    df = pd.read_csv(motif_csv, index_col=0)

    # Drop non-motif metadata columns
    drop_cols = [c for c in ["Chromosome", "Start", "End", "Score", "Name"]
                 if c in df.columns]
    df = df.drop(columns=drop_cols)

    if cre_ids is not None:
        common = [c for c in cre_ids if c in df.index]
        df = df.loc[common]

    # Fill any remaining NaN with 0
    df = df.fillna(0.0)

    print(f"Loaded motif matrix: {df.shape[0]} CREs x {df.shape[1]} motifs")
    return df


def motif_distance_matrix(motif_mat):
    """Compute pairwise cosine distance from motif score vectors.

    distance(i, j) = 1 - cos(v_i, v_j)

    Parameters
    ----------
    motif_mat : pd.DataFrame or np.ndarray
        Shape (n_CREs, n_motifs).

    Returns
    -------
    D : np.ndarray, shape (n_CREs, n_CREs)
    """
    X = motif_mat.values if isinstance(motif_mat, pd.DataFrame) else motif_mat
    D = squareform(pdist(X, metric="cosine"))
    # Replace NaN (from zero vectors) with max distance
    D = np.nan_to_num(D, nan=1.0)
    return D


def motif_kernel_matrix(motif_mat, center=True):
    """Compute cosine similarity kernel from motif score vectors.

    K(i, j) = dot(v_i, v_j) / (||v_i|| * ||v_j||)

    Parameters
    ----------
    motif_mat : pd.DataFrame or np.ndarray
        Shape (n_CREs, n_motifs).
    center : bool
        If True, center the kernel matrix.

    Returns
    -------
    K : np.ndarray, shape (n_CREs, n_CREs)
    """
    from .sequence_kernels import center_kernel

    X = motif_mat.values if isinstance(motif_mat, pd.DataFrame) else motif_mat
    X = X.astype(np.float64)

    # Normalize rows to unit vectors
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    X_normed = X / norms

    K = X_normed @ X_normed.T
    np.fill_diagonal(K, 1.0)

    if center:
        K = center_kernel(K)
    return K


def fimo_motif_matrix(sequences, seq_ids, motif_db, fimo_bin="fimo",
                      pval_threshold=1e-4, output_dir=None):
    """Create a sequence × motif score matrix via FIMO scanning.

    Works on raw DNA sequences (no genomic coordinates needed), so it can
    be used for both barcodes and enhancers.

    Parameters
    ----------
    sequences : list of str
        DNA sequences to scan.
    seq_ids : list of str
        Identifiers for each sequence (e.g., CRE IDs).
    motif_db : str
        Path to MEME-format motif database.
    fimo_bin : str
        Path to FIMO binary.
    pval_threshold : float
        FIMO p-value threshold.
    output_dir : str, optional
        Directory to store intermediate FIMO output.  Uses a temp dir if None.

    Returns
    -------
    motif_mat : pd.DataFrame
        Shape (n_sequences, n_motifs), indexed by seq_id.
    """
    import tempfile
    from io import StringIO

    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="fimo_")
    os.makedirs(output_dir, exist_ok=True)

    # Write FASTA
    fasta_path = os.path.join(output_dir, "sequences.fa")
    with open(fasta_path, "w") as f:
        for sid, seq in zip(seq_ids, sequences):
            f.write(f">{sid}\n{seq.upper()}\n")

    # Run FIMO
    fimo_out = os.path.join(output_dir, "fimo_out")
    cmd = [fimo_bin, "--thresh", str(pval_threshold), "--oc", fimo_out,
           motif_db, fasta_path]
    print(f"Running FIMO (p < {pval_threshold}) on {len(sequences)} sequences...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"FIMO failed (rc={result.returncode}): {result.stderr[:500]}")

    # Parse output (FIMO 5.x → fimo.tsv)
    tsv_path = os.path.join(fimo_out, "fimo.tsv")
    txt_path = os.path.join(fimo_out, "fimo.txt")
    hit_path = tsv_path if os.path.exists(tsv_path) else txt_path

    if not os.path.exists(hit_path):
        print("  No FIMO output file found — returning zero matrix")
        motif_mat = pd.DataFrame(0.0, index=seq_ids, columns=["no_hits"])
        return motif_mat

    with open(hit_path) as f:
        lines = [line for line in f if not line.startswith("##")]
    if not lines:
        motif_mat = pd.DataFrame(0.0, index=seq_ids, columns=["no_hits"])
        return motif_mat
    lines[0] = lines[0].lstrip("#")
    hits = pd.read_csv(StringIO("".join(lines)), sep="\t")
    hits = hits.dropna(how="all")

    # Standardise column names across FIMO versions
    col_map = {"pattern name": "motif_id", "sequence name": "sequence_name"}
    hits = hits.rename(columns=col_map)

    if len(hits) == 0:
        motif_mat = pd.DataFrame(0.0, index=seq_ids, columns=["no_hits"])
        return motif_mat

    # Pivot: sequence × motif, aggregating by count of hits
    pivot = hits.pivot_table(
        index="sequence_name", columns="motif_id",
        values="score", aggfunc="sum", fill_value=0.0,
    )

    # Ensure all input sequences are present
    motif_mat = pivot.reindex(seq_ids, fill_value=0.0)
    motif_mat.index.name = None
    print(f"  FIMO motif matrix: {motif_mat.shape[0]} sequences x "
          f"{motif_mat.shape[1]} motifs, {len(hits)} total hits")
    return motif_mat


def run_motif_scan(bed_file, motif_bed, output_csv, assembly="mm10"):
    """Run the motif scanning pipeline on CRE coordinates.

    Requires: tabix, bedtools on PATH.

    Parameters
    ----------
    bed_file : str
        BED file with CRE coordinates (chr, start, end, [optional cols]).
    motif_bed : str
        Path to Vierstra motif BED.gz file (with .tbi index).
    output_csv : str
        Path to save the CRE × motif score matrix.
    assembly : str
        Genome assembly ('mm10' or 'hg38').

    Returns
    -------
    motif_mat : pd.DataFrame
    """
    import tempfile

    # Step 1: query_motif using tabix
    query_bed = output_csv.replace(".csv", ".query_motif.bed")
    print(f"Running tabix query against motif database...")
    with open(query_bed, "w") as f:
        subprocess.run(
            ["tabix", motif_bed, "-R", bed_file],
            stdout=f, check=True,
        )
    print(f"  Query result: {query_bed}")

    # Step 2: get_motif using bedtools intersect + groupby
    get_motif_bed = output_csv.replace(".csv", ".get_motif.bed")
    print(f"Running bedtools intersect and groupby...")

    # Read chromosomes from query result
    query_df = pd.read_csv(query_bed, sep="\t", header=None, usecols=[0])
    chroms = query_df[0].unique()

    with open(get_motif_bed, "w") as outf:
        for chrom in sorted(chroms):
            if "random" in chrom or chrom == "chrY":
                continue
            with tempfile.NamedTemporaryFile(mode="w", suffix=".bed", delete=False) as peak_tmp, \
                 tempfile.NamedTemporaryFile(mode="w", suffix=".bed", delete=False) as motif_tmp:
                # Filter peak and motif files by chromosome
                peak_df = pd.read_csv(bed_file, sep="\t", header=None)
                peak_chr = peak_df[peak_df[0] == chrom]
                peak_chr.to_csv(peak_tmp.name, sep="\t", header=False, index=False)

                query_full = pd.read_csv(query_bed, sep="\t", header=None)
                motif_chr = query_full[query_full[0] == chrom]
                motif_chr.to_csv(motif_tmp.name, sep="\t", header=False, index=False)

                # bedtools intersect -> sort -> groupby
                cmd = (
                    f"bedtools intersect -a {peak_tmp.name} -b {motif_tmp.name} -wa -wb "
                    f"| cut -f1,2,3,7,8 "
                    f"| sort -k1,1 -k2,2n -k3,3n -k4,4 "
                    f"| bedtools groupby -g 1,2,3,4 -c 5 -o sum"
                )
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
                outf.write(result.stdout)

                os.unlink(peak_tmp.name)
                os.unlink(motif_tmp.name)

    print(f"  Aggregated result: {get_motif_bed}")

    # Step 3: create CRE × motif matrix
    motif_mat = _create_peak_motif(get_motif_bed, bed_file)
    motif_mat.to_csv(output_csv)
    print(f"  Saved motif matrix: {output_csv}")

    return motif_mat


def _create_peak_motif(get_motif_bed, peak_bed):
    """Create a CRE × motif matrix from the get_motif output."""
    peak_motif = pd.read_csv(
        get_motif_bed, sep="\t", header=None,
        names=["Chromosome", "Start", "End", "Motif_cluster", "Score"],
    )

    peak_motif_pivoted = peak_motif.pivot_table(
        index=["Chromosome", "Start", "End"],
        columns="Motif_cluster",
        values="Score",
        fill_value=0,
    )
    peak_motif_pivoted.reset_index(inplace=True)
    peak_motif_pivoted["Name"] = peak_motif_pivoted.apply(
        lambda x: f'{x["Chromosome"]}:{x["Start"]}-{x["End"]}', axis=1
    )
    peak_motif_pivoted = peak_motif_pivoted.drop(
        columns=["Chromosome", "Start", "End"]
    )

    # Read original peak BED to get CRE IDs
    original_peaks = pd.read_csv(peak_bed, sep="\t", header=None)
    ncols = original_peaks.shape[1]

    if ncols >= 5:
        # BED has: chr, start, end, class/score, CRE_ID
        original_peaks.columns = (
            ["Chromosome", "Start", "End"]
            + [f"col{i}" for i in range(3, ncols - 1)]
            + ["CRE_ID"]
        )
    elif ncols >= 4:
        original_peaks.columns = ["Chromosome", "Start", "End", "CRE_ID"]
    else:
        # No CRE IDs — generate them
        original_peaks.columns = ["Chromosome", "Start", "End"][:ncols]
        original_peaks["CRE_ID"] = [
            f"CRE{i+1:03d}" for i in range(len(original_peaks))
        ]

    original_peaks["Name"] = original_peaks.apply(
        lambda x: f'{x["Chromosome"]}:{x["Start"]}-{x["End"]}', axis=1
    )

    # Merge
    merged = pd.merge(original_peaks[["Name", "CRE_ID"]],
                       peak_motif_pivoted, on="Name", how="left")
    merged = merged.set_index("CRE_ID").drop(columns=["Name"])
    merged = merged.fillna(0.0)

    return merged
