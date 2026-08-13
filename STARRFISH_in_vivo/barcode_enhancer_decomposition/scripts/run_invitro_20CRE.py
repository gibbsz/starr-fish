#!/usr/bin/env python3
"""Run barcode-enhancer decomposition on 20CRE/in vitro data.

Activity matrix: from GLM fitting of July and Sept experiments.
Sequences: from Supplementary Table 6, filtered to "20CRE/in vitro".

Both barcode and enhancer distances/kernels use the same method (kmer or
motif) to avoid introducing bias in the decomposition.
"""

import sys
import os
import json
import argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform

from config import (
    SEQUENCES_XLSX, SEQUENCES_SHEET,
    RESULTS_DIR, METHOD, KMER_K, KMER_KS, BOTH_STRANDS,
    MANTEL_PERMUTATIONS, FDR_ALPHA,
    FIMO_BIN, MOTIF_DB_MOUSE, FIMO_PVAL,
)
from src.data_io import load_activity_matrix, load_sequences, align_data, save_results
from src.sequence_kernels import (
    compute_distance_matrix, compute_kernel, kmer_feature_matrix,
    distance_from_feature_matrix, kernel_from_feature_matrix,
)
from src.motif_kernel import (
    load_motif_matrix, motif_distance_matrix, motif_kernel_matrix,
    fimo_motif_matrix,
)
from src.mantel import run_partial_mantel
from src.variance_decomp import run_variance_decomposition
from src.plotting import plot_distance_scatter, plot_all_variance


def apply_transform(activity_df, method):
    """Apply phenotype transform column-wise to activity matrix."""
    if method == 'none':
        return activity_df
    elif method == 'log':
        return np.log1p(activity_df)
    elif method == 'rank':
        from scipy.stats import norm
        def rank_int(x):
            mask = np.isfinite(x)
            out = x.copy()
            n = mask.sum()
            ranks = pd.Series(x[mask]).rank()
            out[mask] = norm.ppf((ranks - 0.5) / n)
            return out
        return activity_df.apply(rank_int, axis=0)


# ── Paths ──────────────────────────────────────────────────────────────────
_INVITRO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "..", "..", "..", "STARRFISH_in_vitro")
ACTIVITY_CSV = os.path.join(_INVITRO_ROOT, "results", "activity_matrix_20CRE.csv")
LIBRARY_FILTER = "20CRE/in vitro"
SAVE_DIR = os.path.join(RESULTS_DIR, "invitro_20CRE")

# Pre-computed motif matrices for in vitro CREs
INVITRO_BARCODE_MOTIF_CSV = os.path.join(SAVE_DIR, "barcode_motif.csv")
INVITRO_ENHANCER_MOTIF_CSV = os.path.join(SAVE_DIR, "enhancer_motif.csv")


def main():
    parser = argparse.ArgumentParser(description="20CRE/in vitro decomposition")
    parser.add_argument("--method", default=METHOD,
                        choices=["kmer", "motif", "motif+kmer"],
                        help="Method for BOTH barcode and enhancer")
    parser.add_argument("--motif-db", default=MOTIF_DB_MOUSE,
                        help="MEME-format motif database (for motif method)")
    parser.add_argument("--fimo-bin", default=FIMO_BIN)
    parser.add_argument("--fimo-pval", type=float, default=FIMO_PVAL)
    parser.add_argument("--barcode-motif-csv", default=INVITRO_BARCODE_MOTIF_CSV,
                        help="Pre-computed barcode motif matrix (optional)")
    parser.add_argument("--enhancer-motif-csv", default=INVITRO_ENHANCER_MOTIF_CSV,
                        help="Pre-computed enhancer motif matrix (optional)")
    parser.add_argument("--transform", choices=["none", "log", "rank"], default="none",
                        help="Phenotype transform before fitting: log (log1p), rank (inverse normal)")
    args = parser.parse_args()

    os.makedirs(SAVE_DIR, exist_ok=True)

    # -- Load data -----------------------------------------------------------
    print("Loading data...")
    activity = load_activity_matrix(ACTIVITY_CSV)
    activity = apply_transform(activity, args.transform)
    if args.transform != 'none':
        print(f"  Applied {args.transform} transform to activity matrix.")
    seq_df = load_sequences(SEQUENCES_XLSX, SEQUENCES_SHEET, LIBRARY_FILTER)
    activity, seq_df = align_data(activity, seq_df)

    barcode_seqs = seq_df["barcode_seq"].tolist()
    enhancer_seqs = seq_df["enhancer_seq"].tolist()
    cre_ids = activity.index.tolist()

    # Activity distance: use Euclidean (only 2 experiments)
    n_celltypes = activity.shape[1]
    print(f"Number of conditions: {n_celltypes}")
    print("Using Euclidean distance for activity (too few conditions for correlation)")
    D_activity = squareform(pdist(activity.values, metric="euclidean"))

    if args.method == "kmer":
        ks = KMER_KS
        ks_label = ",".join(str(k) for k in ks)
        print(f"\nUsing combined multi-k-mer (k={ks_label})")

        # -- Part 1: Partial Mantel test -----------------------------------
        print(f"\n--- PARTIAL MANTEL TEST (k={ks_label}) ---")
        print(f"Computing barcode distance matrix (k-mer, k={ks_label})...")
        D_barcode = compute_distance_matrix(
            barcode_seqs, method="kmer", ks=ks, both_strands=BOTH_STRANDS,
        )
        print(f"Computing enhancer distance matrix (k-mer, k={ks_label})...")
        D_enhancer = compute_distance_matrix(
            enhancer_seqs, method="kmer", ks=ks, both_strands=BOTH_STRANDS,
        )

        # Save distance matrices
        np.save(os.path.join(SAVE_DIR, "D_activity.npy"), D_activity)
        np.save(os.path.join(SAVE_DIR, "D_barcode.npy"), D_barcode)
        np.save(os.path.join(SAVE_DIR, "D_enhancer.npy"), D_enhancer)
        print("Distance matrices saved.")

        # Partial Mantel
        print()
        mantel_results = run_partial_mantel(
            D_activity, D_barcode, D_enhancer,
            n_permutations=MANTEL_PERMUTATIONS,
        )

        # Distance scatter plots
        plot_distance_scatter(D_activity, D_barcode, D_enhancer,
                              save_dir=SAVE_DIR)

        # Save Mantel results
        with open(os.path.join(SAVE_DIR, "mantel_results.json"), "w") as f:
            json.dump(mantel_results, f, indent=2)

        # -- Part 2: Variance decomposition --------------------------------
        print(f"\n--- VARIANCE DECOMPOSITION (k={ks_label}) ---")
        print(f"Computing barcode kernel (k-mer, k={ks_label})...")
        K_barcode = compute_kernel(
            barcode_seqs, method="kmer", ks=ks,
            both_strands=BOTH_STRANDS, center=False,
        )
        print(f"Computing enhancer kernel (k-mer, k={ks_label})...")
        K_enhancer = compute_kernel(
            enhancer_seqs, method="kmer", ks=ks,
            both_strands=BOTH_STRANDS, center=False,
        )

        np.save(os.path.join(SAVE_DIR, "K_barcode.npy"), K_barcode)
        np.save(os.path.join(SAVE_DIR, "K_enhancer.npy"), K_enhancer)
        print("Kernel matrices saved.")

        print("\nFitting REML models...")
        results_df, blup_df = run_variance_decomposition(
            activity, K_barcode, K_enhancer,
            fdr_alpha=FDR_ALPHA,
        )

        # Save results
        save_results(results_df,
                     os.path.join(SAVE_DIR, "variance_decomp_results.csv"),
                     name="variance decomposition results")
        blup_path = os.path.join(SAVE_DIR, "blup.csv")
        blup_df.to_csv(blup_path, index=False)
        print(f"Saved BLUP scores to {blup_path}")

        # Plots
        plot_all_variance(results_df, fdr_alpha=FDR_ALPHA, save_dir=SAVE_DIR)

        print(f"Results saved to: {SAVE_DIR}")

    elif args.method == "motif":
        print("\n" + "=" * 60)
        print("PARTIAL MANTEL TEST")
        print("=" * 60)

        # Barcode motif matrix
        if os.path.exists(args.barcode_motif_csv):
            print(f"Loading barcode motif matrix from {args.barcode_motif_csv}")
            barcode_motif = load_motif_matrix(args.barcode_motif_csv, cre_ids=cre_ids)
        else:
            print("Computing barcode motif matrix via FIMO...")
            barcode_motif = fimo_motif_matrix(
                barcode_seqs, cre_ids, args.motif_db,
                fimo_bin=args.fimo_bin, pval_threshold=args.fimo_pval,
                output_dir=os.path.join(SAVE_DIR, "fimo_barcode"),
            )
            barcode_motif.to_csv(args.barcode_motif_csv)
            print(f"  Saved barcode motif matrix to {args.barcode_motif_csv}")

        # Enhancer motif matrix
        if os.path.exists(args.enhancer_motif_csv):
            print(f"Loading enhancer motif matrix from {args.enhancer_motif_csv}")
            enhancer_motif = load_motif_matrix(args.enhancer_motif_csv, cre_ids=cre_ids)
        else:
            print("Computing enhancer motif matrix via FIMO...")
            enhancer_motif = fimo_motif_matrix(
                enhancer_seqs, cre_ids, args.motif_db,
                fimo_bin=args.fimo_bin, pval_threshold=args.fimo_pval,
                output_dir=os.path.join(SAVE_DIR, "fimo_enhancer"),
            )
            enhancer_motif.to_csv(args.enhancer_motif_csv)
            print(f"  Saved enhancer motif matrix to {args.enhancer_motif_csv}")

        # Align CREs
        common = sorted(
            set(cre_ids) & set(barcode_motif.index) & set(enhancer_motif.index),
            key=lambda x: int(x.replace("CRE", "")),
        )
        if len(common) < len(cre_ids):
            print(f"  Warning: using {len(common)}/{len(cre_ids)} CREs with motif data")
            activity = activity.loc[common]
            D_activity = squareform(pdist(activity.values, metric="euclidean"))

        D_barcode = motif_distance_matrix(barcode_motif.loc[common])
        D_enhancer = motif_distance_matrix(enhancer_motif.loc[common])

        # Save distance matrices
        np.save(os.path.join(SAVE_DIR, "D_activity.npy"), D_activity)
        np.save(os.path.join(SAVE_DIR, "D_barcode.npy"), D_barcode)
        np.save(os.path.join(SAVE_DIR, "D_enhancer.npy"), D_enhancer)
        print("Distance matrices saved.")

        # Partial Mantel
        print()
        mantel_results = run_partial_mantel(
            D_activity, D_barcode, D_enhancer,
            n_permutations=MANTEL_PERMUTATIONS,
        )

        # Distance scatter plots
        plot_distance_scatter(D_activity, D_barcode, D_enhancer, save_dir=SAVE_DIR)

        # Save Mantel results
        with open(os.path.join(SAVE_DIR, "mantel_results.json"), "w") as f:
            json.dump(mantel_results, f, indent=2)

        # Variance decomposition
        print("\n" + "=" * 60)
        print("VARIANCE DECOMPOSITION")
        print("=" * 60)

        print("Computing barcode kernel (motif scores)...")
        K_barcode = motif_kernel_matrix(barcode_motif.loc[common], center=False)
        print("Computing enhancer kernel (motif scores)...")
        K_enhancer = motif_kernel_matrix(enhancer_motif.loc[common], center=False)

        np.save(os.path.join(SAVE_DIR, "K_barcode.npy"), K_barcode)
        np.save(os.path.join(SAVE_DIR, "K_enhancer.npy"), K_enhancer)
        print("Kernel matrices saved.")

        print("\nFitting REML models...")
        results_df, blup_df = run_variance_decomposition(
            activity, K_barcode, K_enhancer,
            fdr_alpha=FDR_ALPHA,
        )

        # Save results
        save_results(results_df,
                     os.path.join(SAVE_DIR, "variance_decomp_results.csv"),
                     name="variance decomposition results")
        blup_path = os.path.join(SAVE_DIR, "blup.csv")
        blup_df.to_csv(blup_path, index=False)
        print(f"Saved BLUP scores to {blup_path}")

        # Plots
        plot_all_variance(results_df, fdr_alpha=FDR_ALPHA, save_dir=SAVE_DIR)

    elif args.method == "motif+kmer":
        ks = KMER_KS
        ks_label = ",".join(str(k) for k in ks)
        print(f"\nUsing combined motif + k-mer (motif + k={ks_label})")

        # -- Part 1: Partial Mantel test -----------------------------------
        print(f"\n--- PARTIAL MANTEL TEST (motif + k={ks_label}) ---")

        # -- Barcode: get k-mer features ---
        print(f"Computing barcode k-mer features (k={ks_label})...")
        barcode_kmer_vecs, _ = kmer_feature_matrix(
            barcode_seqs, ks=ks, both_strands=BOTH_STRANDS,
        )

        # -- Barcode: get motif matrix ---
        if os.path.exists(args.barcode_motif_csv):
            print(f"Loading barcode motif matrix from {args.barcode_motif_csv}")
            barcode_motif = load_motif_matrix(args.barcode_motif_csv, cre_ids=cre_ids)
        else:
            print("Computing barcode motif matrix via FIMO...")
            barcode_motif = fimo_motif_matrix(
                barcode_seqs, cre_ids, args.motif_db,
                fimo_bin=args.fimo_bin, pval_threshold=args.fimo_pval,
                output_dir=os.path.join(SAVE_DIR, "fimo_barcode"),
            )
            barcode_motif.to_csv(args.barcode_motif_csv)
            print(f"  Saved barcode motif matrix to {args.barcode_motif_csv}")

        barcode_motif_vecs = barcode_motif.values / (
            np.linalg.norm(barcode_motif.values, axis=1, keepdims=True) + 1e-10
        )

        # -- Barcode: concatenate and compute distance ---
        barcode_combined = np.concatenate(
            [barcode_kmer_vecs, barcode_motif_vecs], axis=1
        )
        D_barcode = distance_from_feature_matrix(barcode_combined)

        # -- Enhancer: get k-mer features ---
        print(f"Computing enhancer k-mer features (k={ks_label})...")
        enhancer_kmer_vecs, _ = kmer_feature_matrix(
            enhancer_seqs, ks=ks, both_strands=BOTH_STRANDS,
        )

        # -- Enhancer: get motif matrix ---
        if os.path.exists(args.enhancer_motif_csv):
            print(f"Loading enhancer motif matrix from {args.enhancer_motif_csv}")
            enhancer_motif = load_motif_matrix(args.enhancer_motif_csv, cre_ids=cre_ids)
        else:
            print("Computing enhancer motif matrix via FIMO...")
            enhancer_motif = fimo_motif_matrix(
                enhancer_seqs, cre_ids, args.motif_db,
                fimo_bin=args.fimo_bin, pval_threshold=args.fimo_pval,
                output_dir=os.path.join(SAVE_DIR, "fimo_enhancer"),
            )
            enhancer_motif.to_csv(args.enhancer_motif_csv)
            print(f"  Saved enhancer motif matrix to {args.enhancer_motif_csv}")

        enhancer_motif_vecs = enhancer_motif.values / (
            np.linalg.norm(enhancer_motif.values, axis=1, keepdims=True) + 1e-10
        )

        # -- Enhancer: concatenate and compute distance ---
        enhancer_combined = np.concatenate(
            [enhancer_kmer_vecs, enhancer_motif_vecs], axis=1
        )
        D_enhancer = distance_from_feature_matrix(enhancer_combined)

        # Align CREs (if motif data is sparse)
        common = sorted(
            set(cre_ids) & set(barcode_motif.index) & set(enhancer_motif.index),
            key=lambda x: int(x.replace("CRE", "")),
        )
        if len(common) < len(cre_ids):
            print(f"  Warning: using {len(common)}/{len(cre_ids)} CREs with motif data")
            activity = activity.loc[common]
            D_activity = squareform(pdist(activity.values, metric="euclidean"))

        # Save distance matrices
        np.save(os.path.join(SAVE_DIR, "D_activity.npy"), D_activity)
        np.save(os.path.join(SAVE_DIR, "D_barcode.npy"), D_barcode)
        np.save(os.path.join(SAVE_DIR, "D_enhancer.npy"), D_enhancer)
        print("Distance matrices saved.")

        # Partial Mantel
        print()
        mantel_results = run_partial_mantel(
            D_activity, D_barcode, D_enhancer,
            n_permutations=MANTEL_PERMUTATIONS,
        )

        # Distance scatter plots
        plot_distance_scatter(D_activity, D_barcode, D_enhancer,
                              save_dir=SAVE_DIR)

        # Save Mantel results
        with open(os.path.join(SAVE_DIR, "mantel_results.json"), "w") as f:
            json.dump(mantel_results, f, indent=2)

        # -- Part 2: Variance decomposition --------------------------------
        print(f"\n--- VARIANCE DECOMPOSITION (motif + k={ks_label}) ---")

        # Compute kernels from concatenated features
        print(f"Computing barcode kernel (motif + k-mer)...")
        barcode_combined_norms = np.linalg.norm(barcode_combined, axis=1, keepdims=True)
        barcode_combined_norms = np.maximum(barcode_combined_norms, 1e-10)
        barcode_combined_normed = barcode_combined / barcode_combined_norms
        K_barcode = kernel_from_feature_matrix(barcode_combined_normed, center=False)

        print(f"Computing enhancer kernel (motif + k-mer)...")
        enhancer_combined_norms = np.linalg.norm(enhancer_combined, axis=1, keepdims=True)
        enhancer_combined_norms = np.maximum(enhancer_combined_norms, 1e-10)
        enhancer_combined_normed = enhancer_combined / enhancer_combined_norms
        K_enhancer = kernel_from_feature_matrix(enhancer_combined_normed, center=False)

        np.save(os.path.join(SAVE_DIR, "K_barcode.npy"), K_barcode)
        np.save(os.path.join(SAVE_DIR, "K_enhancer.npy"), K_enhancer)
        print("Kernel matrices saved.")

        print("\nFitting REML models...")
        results_df, blup_df = run_variance_decomposition(
            activity, K_barcode, K_enhancer,
            fdr_alpha=FDR_ALPHA,
        )

        # Save results
        save_results(results_df,
                     os.path.join(SAVE_DIR, "variance_decomp_results.csv"),
                     name="variance decomposition results")
        blup_path = os.path.join(SAVE_DIR, "blup.csv")
        blup_df.to_csv(blup_path, index=False)
        print(f"Saved BLUP scores to {blup_path}")

        # Plots
        plot_all_variance(results_df, fdr_alpha=FDR_ALPHA, save_dir=SAVE_DIR)

    print(f"\nAll results saved to: {SAVE_DIR}")


if __name__ == "__main__":
    main()
