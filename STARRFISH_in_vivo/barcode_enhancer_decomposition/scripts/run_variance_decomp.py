#!/usr/bin/env python3
"""Run kernel variance decomposition on real data.

Both barcode and enhancer kernels use the same method (kmer or motif)
to avoid introducing bias in the decomposition.
"""

import sys
import os
import argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from config import (
    ACTIVITY_CSV, SEQUENCES_XLSX, SEQUENCES_SHEET, LIBRARY_FILTER,
    RESULTS_DIR, METHOD, KMER_K, KMER_KS, KERNEL_TYPE, BOTH_STRANDS,
    REML_MAX_ITER, REML_TOL, FDR_ALPHA, N_BOOTSTRAP, N_JOBS,
    FIMO_BIN, MOTIF_DB_MOUSE, FIMO_PVAL,
    ENHANCER_MOTIF_CSV, BARCODE_MOTIF_CSV,
    CELLTYPE_COUNTS_CSV, MIN_CELLS,
)
from src.data_io import (
    load_activity_matrix, load_sequences, align_data, save_results,
    filter_celltypes_by_count,
)
from src.sequence_kernels import compute_kernel, kmer_feature_matrix, kernel_from_feature_matrix
from src.motif_kernel import (
    load_motif_matrix, motif_kernel_matrix, fimo_motif_matrix,
)
from src.variance_decomp import run_variance_decomposition
from src.plotting import plot_all_variance


def main():
    parser = argparse.ArgumentParser(description="Kernel variance decomposition")
    parser.add_argument("--activity", default=ACTIVITY_CSV,
                        help="Path to activity matrix")
    parser.add_argument("--sequences", default=SEQUENCES_XLSX,
                        help="Path to sequences Excel file")
    parser.add_argument("--sheet", default=SEQUENCES_SHEET)
    parser.add_argument("--library", default=LIBRARY_FILTER)
    parser.add_argument("--method", default=METHOD,
                        choices=["kmer", "motif", "motif+kmer"],
                        help="Kernel method for BOTH barcode and enhancer")
    parser.add_argument("--kmer-ks", type=int, nargs="+", default=KMER_KS,
                        help="k-mer sizes to evaluate (default: 6 10 14)")
    parser.add_argument("--kernel", default=KERNEL_TYPE,
                        choices=["kmer", "rbf"],
                        help="Kernel type (only used when method=kmer)")
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
    parser.add_argument("--max-iter", type=int, default=REML_MAX_ITER)
    parser.add_argument("--fdr", type=float, default=FDR_ALPHA)
    parser.add_argument("--n-bootstrap", type=int, default=N_BOOTSTRAP,
                        help="Number of bootstrap iterations for CI estimation")
    parser.add_argument("--n-jobs", type=int, default=N_JOBS,
                        help="Number of parallel jobs for bootstrap (-1 = all cores)")
    parser.add_argument("--outdir",
                        default=os.path.join(RESULTS_DIR, "variance_decomp"))
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

    # -- Compute kernels (same method for both) ------------------------------
    if args.method == "kmer":
        ks = args.kmer_ks
        ks_label = ",".join(str(k) for k in ks)
        print(f"\nUsing combined multi-k-mer kernel (k={ks_label})")

        print(f"Computing barcode kernel (k-mer, k={ks_label})...")
        K_barcode = compute_kernel(
            barcode_seqs, method=args.kernel, ks=ks,
            both_strands=both_strands, center=False,
        )
        print(f"Computing enhancer kernel (k-mer, k={ks_label})...")
        K_enhancer = compute_kernel(
            enhancer_seqs, method=args.kernel, ks=ks,
            both_strands=both_strands, center=False,
        )

        # Save kernels
        np.save(os.path.join(args.outdir, "K_barcode.npy"), K_barcode)
        np.save(os.path.join(args.outdir, "K_enhancer.npy"), K_enhancer)
        print("Kernel matrices saved.")

        # Variance decomposition
        print("\nFitting REML models...")
        results_df, blup_df = run_variance_decomposition(
            activity, K_barcode, K_enhancer,
            max_iter=args.max_iter, fdr_alpha=args.fdr,
            n_boot=args.n_bootstrap, n_jobs=args.n_jobs,
        )

        # Save results
        save_results(results_df,
                     os.path.join(args.outdir, "variance_decomp_results.csv"),
                     name="variance decomposition results")
        blup_path = os.path.join(args.outdir, "blup.csv")
        blup_df.to_csv(blup_path, index=False)
        print(f"Saved BLUP scores to {blup_path}")

        # Plots
        plot_all_variance(results_df, fdr_alpha=args.fdr, save_dir=args.outdir)

        print(f"Results saved to: {args.outdir}")

    elif args.method == "motif":
        # Barcode motif matrix
        if os.path.exists(args.barcode_motif_csv):
            print(f"\nLoading barcode motif matrix from {args.barcode_motif_csv}")
            barcode_motif = load_motif_matrix(args.barcode_motif_csv, cre_ids=cre_ids)
        else:
            print("\nComputing barcode motif matrix via FIMO...")
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

        K_barcode = motif_kernel_matrix(barcode_motif.loc[common], center=False)
        K_enhancer = motif_kernel_matrix(enhancer_motif.loc[common], center=False)

        # Save kernels
        np.save(os.path.join(args.outdir, "K_barcode.npy"), K_barcode)
        np.save(os.path.join(args.outdir, "K_enhancer.npy"), K_enhancer)
        print("Kernel matrices saved.")

        # Variance decomposition
        print("\nFitting REML models...")
        results_df, blup_df = run_variance_decomposition(
            activity, K_barcode, K_enhancer,
            max_iter=args.max_iter, fdr_alpha=args.fdr,
            n_boot=args.n_bootstrap, n_jobs=args.n_jobs,
        )

        # Save results
        save_results(results_df, os.path.join(args.outdir, "variance_decomp_results.csv"),
                     name="variance decomposition results")
        blup_path = os.path.join(args.outdir, "blup.csv")
        blup_df.to_csv(blup_path, index=False)
        print(f"Saved BLUP scores to {blup_path}")

        # Plots
        plot_all_variance(results_df, fdr_alpha=args.fdr, save_dir=args.outdir)

        print(f"\nAll results saved to: {args.outdir}")

    elif args.method == "motif+kmer":
        ks = args.kmer_ks
        ks_label = ",".join(str(k) for k in ks)
        print(f"\nUsing combined motif + k-mer kernel (motif + k={ks_label})")

        # -- Load both motif matrices first ---
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

        # -- Find common CREs and filter all data before computing features ---
        common = sorted(
            set(cre_ids) & set(barcode_motif.index) & set(enhancer_motif.index),
            key=lambda x: int(x.replace("CRE", "")),
        )
        if len(common) < len(cre_ids):
            print(f"  Warning: using {len(common)}/{len(cre_ids)} CREs with motif data")
            common_set = set(common)
            common_indices = [i for i, cid in enumerate(cre_ids) if cid in common_set]
            activity = activity.loc[common]
            barcode_seqs = [barcode_seqs[i] for i in common_indices]
            enhancer_seqs = [enhancer_seqs[i] for i in common_indices]
            cre_ids = common
            barcode_motif = barcode_motif.loc[common]
            enhancer_motif = enhancer_motif.loc[common]

        # -- Barcode: k-mer + motif features ---
        print(f"Computing barcode k-mer features (k={ks_label})...")
        barcode_kmer_vecs, _ = kmer_feature_matrix(
            barcode_seqs, ks=ks, both_strands=args.no_both_strands == False,
        )
        barcode_motif_vecs = barcode_motif.values / (
            np.linalg.norm(barcode_motif.values, axis=1, keepdims=True) + 1e-10
        )
        barcode_combined = np.concatenate([barcode_kmer_vecs, barcode_motif_vecs], axis=1)
        barcode_combined_norms = np.maximum(
            np.linalg.norm(barcode_combined, axis=1, keepdims=True), 1e-10
        )
        K_barcode = kernel_from_feature_matrix(
            barcode_combined / barcode_combined_norms, center=False
        )

        # -- Enhancer: k-mer + motif features ---
        print(f"Computing enhancer k-mer features (k={ks_label})...")
        enhancer_kmer_vecs, _ = kmer_feature_matrix(
            enhancer_seqs, ks=ks, both_strands=args.no_both_strands == False,
        )
        enhancer_motif_vecs = enhancer_motif.values / (
            np.linalg.norm(enhancer_motif.values, axis=1, keepdims=True) + 1e-10
        )
        enhancer_combined = np.concatenate([enhancer_kmer_vecs, enhancer_motif_vecs], axis=1)
        enhancer_combined_norms = np.maximum(
            np.linalg.norm(enhancer_combined, axis=1, keepdims=True), 1e-10
        )
        K_enhancer = kernel_from_feature_matrix(
            enhancer_combined / enhancer_combined_norms, center=False
        )

        # Save kernels
        np.save(os.path.join(args.outdir, "K_barcode.npy"), K_barcode)
        np.save(os.path.join(args.outdir, "K_enhancer.npy"), K_enhancer)
        print("Kernel matrices saved.")

        # Variance decomposition
        print("\nFitting REML models...")
        results_df, blup_df = run_variance_decomposition(
            activity, K_barcode, K_enhancer,
            max_iter=args.max_iter, fdr_alpha=args.fdr,
            n_boot=args.n_bootstrap, n_jobs=args.n_jobs,
        )

        # Save results
        save_results(results_df, os.path.join(args.outdir, "variance_decomp_results.csv"),
                     name="variance decomposition results")
        blup_path = os.path.join(args.outdir, "blup.csv")
        blup_df.to_csv(blup_path, index=False)
        print(f"Saved BLUP scores to {blup_path}")

        # Plots
        plot_all_variance(results_df, fdr_alpha=args.fdr, save_dir=args.outdir)

        print(f"\nAll results saved to: {args.outdir}")


if __name__ == "__main__":
    main()
