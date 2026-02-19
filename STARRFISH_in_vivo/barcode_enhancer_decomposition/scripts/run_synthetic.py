#!/usr/bin/env python3
"""End-to-end validation on synthetic data.

Generates synthetic data, runs both the partial Mantel test and
variance decomposition, and compares results to known ground truth.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from scipy.spatial.distance import pdist, squareform

from config import (
    RESULTS_DIR, SYNTH_N_CONSTRUCTS, SYNTH_N_CELLTYPES,
    SYNTH_BARCODE_LEN, SYNTH_ENHANCER_LEN,
    SYNTH_SIGMA2_BARCODE, SYNTH_SIGMA2_ENHANCER, SYNTH_SIGMA2_NOISE,
    SYNTH_N_MOTIFS_BARCODE, SYNTH_N_MOTIFS_ENHANCER,
    KMER_K, MANTEL_PERMUTATIONS, FDR_ALPHA,
)
from src.synthetic import generate_synthetic_data
from src.sequence_kernels import compute_distance_matrix, compute_kernel
from src.mantel import run_partial_mantel
from src.variance_decomp import run_variance_decomposition
from src.plotting import plot_distance_scatter, plot_all_variance
from src.data_io import save_results


def main():
    save_dir = os.path.join(RESULTS_DIR, "synthetic")
    os.makedirs(save_dir, exist_ok=True)

    # -- 1. Generate synthetic data ------------------------------------------
    print("\n" + "=" * 60)
    print("GENERATING SYNTHETIC DATA")
    print("=" * 60)
    activity, barcode_seqs, enhancer_seqs, ground_truth = generate_synthetic_data(
        n_constructs=SYNTH_N_CONSTRUCTS,
        n_celltypes=SYNTH_N_CELLTYPES,
        barcode_len=SYNTH_BARCODE_LEN,
        enhancer_len=SYNTH_ENHANCER_LEN,
        sigma2_barcode_true=SYNTH_SIGMA2_BARCODE,
        sigma2_enhancer_true=SYNTH_SIGMA2_ENHANCER,
        sigma2_noise_true=SYNTH_SIGMA2_NOISE,
        n_motifs_barcode=SYNTH_N_MOTIFS_BARCODE,
        n_motifs_enhancer=SYNTH_N_MOTIFS_ENHANCER,
    )

    # -- 2. Compute distance matrices ----------------------------------------
    print("\n" + "=" * 60)
    print("COMPUTING DISTANCE MATRICES")
    print("=" * 60)

    # Activity distance (correlation distance) — no NaN in synthetic data
    D_activity = squareform(pdist(activity.values, metric="correlation"))
    print(f"D_activity: shape {D_activity.shape}")

    # Sequence distances (both strands, k-mer Jaccard)
    D_barcode = compute_distance_matrix(barcode_seqs, method="kmer", k=KMER_K,
                                         both_strands=True)
    print(f"D_barcode: shape {D_barcode.shape}")

    D_enhancer = compute_distance_matrix(enhancer_seqs, method="kmer", k=KMER_K,
                                          both_strands=True)
    print(f"D_enhancer: shape {D_enhancer.shape}")

    # -- 3. Partial Mantel tests ---------------------------------------------
    print()
    mantel_results = run_partial_mantel(
        D_activity, D_barcode, D_enhancer,
        n_permutations=MANTEL_PERMUTATIONS,
    )

    # Distance scatter plots
    plot_distance_scatter(D_activity, D_barcode, D_enhancer, save_dir=save_dir)

    # -- 4. Variance decomposition -------------------------------------------
    print("\n" + "=" * 60)
    print("COMPUTING KERNELS FOR VARIANCE DECOMPOSITION")
    print("=" * 60)

    # Un-centered kernels — run_variance_decomposition will center per subset
    K_barcode = compute_kernel(barcode_seqs, method="kmer", k=KMER_K,
                               both_strands=True, center=False)
    K_enhancer = compute_kernel(enhancer_seqs, method="kmer", k=KMER_K,
                                both_strands=True, center=False)

    print("\nFitting REML models...")
    vd_results, blup_df = run_variance_decomposition(
        activity, K_barcode, K_enhancer,
        fdr_alpha=FDR_ALPHA,
    )
    save_results(vd_results, os.path.join(save_dir, "variance_decomp_results.csv"),
                 name="variance decomposition results")
    blup_path = os.path.join(save_dir, "blup.csv")
    blup_df.to_csv(blup_path, index=False)
    print(f"Saved BLUP scores to {blup_path}")

    # Plots
    plot_all_variance(vd_results, fdr_alpha=FDR_ALPHA, save_dir=save_dir)

    # -- 5. Compare to ground truth ------------------------------------------
    print("\n" + "=" * 60)
    print("GROUND TRUTH COMPARISON")
    print("=" * 60)
    print(f"True proportions:  barcode={ground_truth['prop_barcode']:.3f}, "
          f"enhancer={ground_truth['prop_enhancer']:.3f}, "
          f"noise={ground_truth['prop_noise']:.3f}")
    valid = vd_results.dropna(subset=["prop_barcode"])
    print(f"Estimated (median): barcode={valid['prop_barcode'].median():.3f}, "
          f"enhancer={valid['prop_enhancer'].median():.3f}, "
          f"noise={valid['prop_noise'].median():.3f}")
    print(f"Estimated (mean):   barcode={valid['prop_barcode'].mean():.3f}, "
          f"enhancer={valid['prop_enhancer'].mean():.3f}, "
          f"noise={valid['prop_noise'].mean():.3f}")

    print(f"\nMantel results: barcode r={mantel_results['barcode_r']:.4f} "
          f"(p={mantel_results['barcode_p']:.4f}), "
          f"enhancer r={mantel_results['enhancer_r']:.4f} "
          f"(p={mantel_results['enhancer_p']:.4f})")

    print("\nDone! Results saved to:", save_dir)


if __name__ == "__main__":
    main()
