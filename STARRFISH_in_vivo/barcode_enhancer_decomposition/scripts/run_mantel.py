#!/usr/bin/env python3
"""Run partial Mantel test on real data.

Both barcode and enhancer distances use the same method (kmer or motif)
to avoid introducing bias in the decomposition.
"""

import sys
import os
import argparse
import json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from config import (
    ACTIVITY_CSV, SEQUENCES_XLSX, SEQUENCES_SHEET, LIBRARY_FILTER,
    RESULTS_DIR, METHOD, KMER_K, KMER_KS, MANTEL_PERMUTATIONS,
    FIMO_BIN, MOTIF_DB_MOUSE, FIMO_PVAL,
    ENHANCER_MOTIF_CSV, BARCODE_MOTIF_CSV,
    CELLTYPE_COUNTS_CSV, MIN_CELLS,
)
from src.data_io import (
    load_activity_matrix, load_sequences, align_data,
    correlation_distance_matrix, filter_celltypes_by_count,
)
from src.sequence_kernels import (
    compute_distance_matrix, kmer_feature_matrix, distance_from_feature_matrix,
)
from src.motif_kernel import (
    load_motif_matrix, motif_distance_matrix, fimo_motif_matrix,
)
from src.mantel import run_partial_mantel
from src.plotting import plot_distance_scatter


def main():
    parser = argparse.ArgumentParser(description="Partial Mantel test")
    parser.add_argument("--activity", default=ACTIVITY_CSV,
                        help="Path to activity matrix")
    parser.add_argument("--sequences", default=SEQUENCES_XLSX,
                        help="Path to sequences Excel file")
    parser.add_argument("--sheet", default=SEQUENCES_SHEET)
    parser.add_argument("--library", default=LIBRARY_FILTER)
    parser.add_argument("--method", default=METHOD,
                        choices=["kmer", "motif", "motif+kmer"],
                        help="Distance method for BOTH barcode and enhancer")
    parser.add_argument("--kmer-ks", type=int, nargs="+", default=KMER_KS,
                        help="k-mer sizes to evaluate (default: 6 10 14)")
    parser.add_argument("--no-both-strands", action="store_true")
    parser.add_argument("--motif-db", default=MOTIF_DB_MOUSE,
                        help="MEME-format motif database (for motif method)")
    parser.add_argument("--fimo-bin", default=FIMO_BIN)
    parser.add_argument("--fimo-pval", type=float, default=FIMO_PVAL)
    parser.add_argument("--barcode-motif-csv", default=BARCODE_MOTIF_CSV,
                        help="Pre-computed barcode motif matrix (optional)")
    parser.add_argument("--enhancer-motif-csv", default=ENHANCER_MOTIF_CSV,
                        help="Pre-computed enhancer motif matrix (optional)")
    parser.add_argument("--celltype-counts", default=CELLTYPE_COUNTS_CSV,
                        help="CSV with subclass,count columns")
    parser.add_argument("--min-cells", type=int, default=MIN_CELLS,
                        help="Minimum cells per cell type")
    parser.add_argument("--permutations", type=int, default=MANTEL_PERMUTATIONS)
    parser.add_argument("--outdir", default=os.path.join(RESULTS_DIR, "mantel"))
    args = parser.parse_args()

    both_strands = not args.no_both_strands
    os.makedirs(args.outdir, exist_ok=True)

    # -- Load data -----------------------------------------------------------
    print("Loading data...")
    activity = load_activity_matrix(args.activity)

    # Filter to cell types with sufficient cells
    if args.celltype_counts and os.path.exists(args.celltype_counts):
        activity = filter_celltypes_by_count(
            activity, args.celltype_counts, min_cells=args.min_cells)

    seq_df = load_sequences(args.sequences, args.sheet, args.library)
    activity, seq_df = align_data(activity, seq_df)

    barcode_seqs = seq_df["barcode_seq"].tolist()
    enhancer_seqs = seq_df["enhancer_seq"].tolist()
    cre_ids = activity.index.tolist()

    # -- Compute distance matrices -------------------------------------------
    print("\nComputing activity distance matrix (NaN-aware)...")
    D_activity = correlation_distance_matrix(activity)

    if args.method == "kmer":
        ks = args.kmer_ks
        ks_label = ",".join(str(k) for k in ks)
        print(f"\nUsing combined multi-k-mer distance (k={ks_label})")

        print(f"Computing barcode distance matrix (k-mer, k={ks_label})...")
        D_barcode = compute_distance_matrix(
            barcode_seqs, method="kmer", ks=ks,
            both_strands=both_strands,
        )
        print(f"Computing enhancer distance matrix (k-mer, k={ks_label})...")
        D_enhancer = compute_distance_matrix(
            enhancer_seqs, method="kmer", ks=ks,
            both_strands=both_strands,
        )

        # Save distance matrices
        np.save(os.path.join(args.outdir, "D_activity.npy"), D_activity)
        np.save(os.path.join(args.outdir, "D_barcode.npy"), D_barcode)
        np.save(os.path.join(args.outdir, "D_enhancer.npy"), D_enhancer)
        print("Distance matrices saved.")

        # Partial Mantel tests
        print()
        mantel_results = run_partial_mantel(
            D_activity, D_barcode, D_enhancer,
            n_permutations=args.permutations,
        )

        # Plots
        plot_distance_scatter(D_activity, D_barcode, D_enhancer,
                              save_dir=args.outdir)

        # Save summary
        with open(os.path.join(args.outdir, "mantel_results.json"), "w") as f:
            json.dump(mantel_results, f, indent=2)
        print(f"Results saved to: {args.outdir}")

    elif args.method == "motif":
        # Barcode motif matrix
        if os.path.exists(args.barcode_motif_csv):
            print(f"Loading barcode motif matrix from {args.barcode_motif_csv}")
            barcode_motif = load_motif_matrix(args.barcode_motif_csv, cre_ids=cre_ids)
        else:
            print("Computing barcode motif matrix via FIMO...")
            barcode_motif = fimo_motif_matrix(
                barcode_seqs, cre_ids, args.motif_db,
                fimo_bin=args.fimo_bin, pval_threshold=args.fimo_pval,
                output_dir=os.path.join(args.outdir, "fimo_barcode"),
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
                output_dir=os.path.join(args.outdir, "fimo_enhancer"),
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
            D_activity = correlation_distance_matrix(activity)

        D_barcode = motif_distance_matrix(barcode_motif.loc[common])
        D_enhancer = motif_distance_matrix(enhancer_motif.loc[common])

        # Save distance matrices
        np.save(os.path.join(args.outdir, "D_activity.npy"), D_activity)
        np.save(os.path.join(args.outdir, "D_barcode.npy"), D_barcode)
        np.save(os.path.join(args.outdir, "D_enhancer.npy"), D_enhancer)
        print("Distance matrices saved.")

        # Partial Mantel tests
        print()
        mantel_results = run_partial_mantel(
            D_activity, D_barcode, D_enhancer,
            n_permutations=args.permutations,
        )

        # Plots
        plot_distance_scatter(D_activity, D_barcode, D_enhancer, save_dir=args.outdir)

        # Save summary
        with open(os.path.join(args.outdir, "mantel_results.json"), "w") as f:
            json.dump(mantel_results, f, indent=2)
        print(f"\nResults saved to: {args.outdir}")

    elif args.method == "motif+kmer":
        ks = args.kmer_ks
        ks_label = ",".join(str(k) for k in ks)
        print(f"\nUsing combined motif + k-mer distance (motif + k={ks_label})")

        # -- Barcode: get k-mer features ---
        print(f"Computing barcode k-mer features (k={ks_label})...")
        barcode_kmer_vecs, _ = kmer_feature_matrix(
            barcode_seqs, ks=ks, both_strands=both_strands,
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
                output_dir=os.path.join(args.outdir, "fimo_barcode"),
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
            enhancer_seqs, ks=ks, both_strands=both_strands,
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
                output_dir=os.path.join(args.outdir, "fimo_enhancer"),
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
            D_activity = correlation_distance_matrix(activity)

        # Save distance matrices
        np.save(os.path.join(args.outdir, "D_activity.npy"), D_activity)
        np.save(os.path.join(args.outdir, "D_barcode.npy"), D_barcode)
        np.save(os.path.join(args.outdir, "D_enhancer.npy"), D_enhancer)
        print("Distance matrices saved.")

        # Partial Mantel tests
        print()
        mantel_results = run_partial_mantel(
            D_activity, D_barcode, D_enhancer,
            n_permutations=args.permutations,
        )

        # Plots
        plot_distance_scatter(D_activity, D_barcode, D_enhancer, save_dir=args.outdir)

        # Save summary
        with open(os.path.join(args.outdir, "mantel_results.json"), "w") as f:
            json.dump(mantel_results, f, indent=2)
        print(f"\nResults saved to: {args.outdir}")


if __name__ == "__main__":
    main()
