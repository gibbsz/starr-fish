"""Load real data and save results."""

import os
import numpy as np
import pandas as pd
from tqdm import tqdm


def reverse_complement(seq):
    """Return the reverse complement of a DNA sequence."""
    comp = str.maketrans("ACGTacgt", "TGCAtgca")
    return seq.translate(comp)[::-1]


def load_activity_matrix(path):
    """Load the activity matrix from CSV, TSV, or H5AD.

    The real data has shape (cell_types x CREs).  We transpose so that
    rows = constructs and columns = cell types, matching the spec.

    Returns
    -------
    activity : pd.DataFrame
        Shape (n_constructs, n_celltypes).
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        df = pd.read_csv(path, index_col=0)
    elif ext in (".tsv", ".txt"):
        df = pd.read_csv(path, index_col=0, sep="\t")
    elif ext in (".h5ad", ".h5"):
        import anndata
        adata = anndata.read_h5ad(path)
        df = pd.DataFrame(
            adata.X if not hasattr(adata.X, "toarray") else adata.X.toarray(),
            index=adata.obs_names,
            columns=adata.var_names,
        )
    else:
        raise ValueError(f"Unsupported file extension: {ext}")

    # Transpose: rows become CREs, columns become cell types
    activity = df.T
    print(f"Loaded activity matrix: {activity.shape[0]} constructs x "
          f"{activity.shape[1]} cell types")
    return activity


def load_sequences(xlsx_path, sheet_name, library_filter=None):
    """Load barcode and enhancer sequences from the Supplementary Table.

    Parameters
    ----------
    xlsx_path : str
        Path to the Excel file.
    sheet_name : str
        Sheet name.
    library_filter : str or None
        If given, filter to this Library/experiment value (e.g. '400CRE/in vivo').

    Returns
    -------
    seq_df : pd.DataFrame
        Columns: 'enhancer_seq', 'barcode_seq'.
        Indexed by enhancer_id (e.g. 'CRE001').
    """
    df = pd.read_excel(xlsx_path, sheet_name=sheet_name, header=1)

    if library_filter is not None:
        df = df[df["Library/experiment"] == library_filter].copy()

    seq_df = pd.DataFrame({
        "enhancer_id": df["Enhancer ID"].values,
        "enhancer_seq": df["Enhancer sequence (5'-3')"].str.upper().values,
        "barcode_seq": df["Barcode sequence (5'-3')"].str.upper().values,
    })
    seq_df = seq_df.set_index("enhancer_id")
    print(f"Loaded {len(seq_df)} sequences (library={library_filter})")
    return seq_df


def align_data(activity, seq_df):
    """Align activity matrix rows with sequence dataframe rows by CRE ID.

    Returns activity and seq_df with matching, ordered indices.
    """
    common = activity.index.intersection(seq_df.index)
    common = sorted(common, key=lambda x: int(x.replace("CRE", "")))
    activity = activity.loc[common]
    seq_df = seq_df.loc[common]
    print(f"Aligned data: {len(common)} constructs in common")
    return activity, seq_df


def filter_celltypes_by_count(activity, celltype_counts_csv, min_cells=1000):
    """Filter activity matrix columns to cell types with >= min_cells.

    Parameters
    ----------
    activity : pd.DataFrame
        Shape (n_constructs, n_celltypes).  Columns are cell-type names.
    celltype_counts_csv : str
        CSV with columns 'subclass' and 'count'.
    min_cells : int
        Minimum number of cells required to keep a cell type.

    Returns
    -------
    activity : pd.DataFrame
        Filtered to cell types with sufficient cells.
    """
    counts = pd.read_csv(celltype_counts_csv)
    keep = set(counts.loc[counts["count"] >= min_cells, "subclass"])
    cols_before = activity.columns.tolist()
    cols_keep = [c for c in cols_before if c in keep]
    activity = activity[cols_keep]
    print(f"Cell-type filter (>= {min_cells} cells): "
          f"{len(cols_keep)}/{len(cols_before)} cell types retained")
    return activity


def correlation_distance_matrix(activity):
    """Compute pairwise correlation distance with NaN-aware handling.

    For each pair of CREs, computes 1 - Pearson r using only cell types
    where both CREs have non-NaN values (pairwise complete observations).

    Parameters
    ----------
    activity : pd.DataFrame or np.ndarray
        Shape (n_constructs, n_celltypes). May contain NaN.

    Returns
    -------
    D : np.ndarray, shape (n_constructs, n_constructs)
        Correlation distance matrix. Values in [0, 2].
    """
    if isinstance(activity, pd.DataFrame):
        X = activity.values
    else:
        X = activity

    n = X.shape[0]
    D = np.zeros((n, n))

    for i in tqdm(range(n), desc="Correlation distance"):
        for j in range(i + 1, n):
            # Find shared non-NaN positions
            mask = ~np.isnan(X[i]) & ~np.isnan(X[j])
            n_shared = mask.sum()

            if n_shared < 5:
                # Too few shared observations; set to maximum distance
                D[i, j] = 1.0
                D[j, i] = 1.0
                continue

            xi = X[i, mask]
            xj = X[j, mask]

            # Check for zero variance
            if np.std(xi) < 1e-10 or np.std(xj) < 1e-10:
                D[i, j] = 1.0
                D[j, i] = 1.0
                continue

            r = np.corrcoef(xi, xj)[0, 1]
            d = 1.0 - r
            D[i, j] = d
            D[j, i] = d

    return D


def save_results(df, path, name="results"):
    """Save a DataFrame to CSV."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path)
    print(f"Saved {name} to {path}")
