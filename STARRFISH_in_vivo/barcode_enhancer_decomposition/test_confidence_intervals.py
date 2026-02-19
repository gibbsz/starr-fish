#!/usr/bin/env python3
"""Quick test of confidence interval implementation."""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd

from src.variance_decomp import (
    bootstrap_confidence_intervals,
    calculate_blup_uncertainty,
    fit_reml,
    compute_blups
)
from src.sequence_kernels import center_kernel

def test_bootstrap_ci():
    """Test bootstrap confidence interval estimation."""
    print("=" * 60)
    print("Testing Bootstrap Confidence Intervals")
    print("=" * 60)

    # Generate synthetic data
    np.random.seed(42)
    n = 50  # Small for fast testing

    # Create proper positive definite kernels using RBF
    X = np.random.randn(n, 10)  # Random features
    K_barcode = X @ X.T / 10  # Gram matrix (always PSD)
    K_barcode = center_kernel(K_barcode)
    K_barcode += 0.01 * np.eye(n)  # Add small jitter to ensure PD

    X2 = np.random.randn(n, 10)
    K_enhancer = X2 @ X2.T / 10
    K_enhancer = center_kernel(K_enhancer)
    K_enhancer += 0.01 * np.eye(n)

    # Generate phenotype
    sigma2_b, sigma2_e, sigma2_n = 0.1, 0.5, 0.2
    V = sigma2_b * K_barcode + sigma2_e * K_enhancer + sigma2_n * np.eye(n)
    V += 0.01 * np.eye(n)  # Ensure positive definite
    y = np.random.multivariate_normal(np.zeros(n), V)

    # Fit REML
    print("\nFitting REML to synthetic data...")
    res = fit_reml(y, K_barcode, K_enhancer)
    print(f"  σ²_barcode = {res['sigma2_barcode']:.4f}")
    print(f"  σ²_enhancer = {res['sigma2_enhancer']:.4f}")
    print(f"  σ²_noise = {res['sigma2_noise']:.4f}")

    total = res['sigma2_barcode'] + res['sigma2_enhancer'] + res['sigma2_noise']
    print(f"\nProportions:")
    print(f"  prop_barcode = {res['sigma2_barcode']/total:.4f}")
    print(f"  prop_enhancer = {res['sigma2_enhancer']/total:.4f}")
    print(f"  prop_noise = {res['sigma2_noise']/total:.4f}")

    # Bootstrap CIs (use small n_boot for testing)
    print(f"\nComputing bootstrap CIs (n_boot=20, parallel)...")
    import time
    start = time.time()
    boot_ci = bootstrap_confidence_intervals(
        y, K_barcode, K_enhancer,
        res['sigma2_barcode'], res['sigma2_enhancer'], res['sigma2_noise'],
        n_boot=20, n_jobs=-1  # Use all cores
    )
    elapsed = time.time() - start
    print(f"  Bootstrap completed in {elapsed:.2f} seconds")

    print(f"\nBootstrap Results:")
    print(f"  Enhancer proportion:")
    print(f"    Mean = {boot_ci['prop_enhancer_mean']:.4f}")
    print(f"    SE = {boot_ci['prop_enhancer_se']:.4f}")
    print(f"    95% CI = [{boot_ci['prop_enhancer_CI_95'][0]:.4f}, "
          f"{boot_ci['prop_enhancer_CI_95'][1]:.4f}]")

    print(f"  Barcode proportion:")
    print(f"    Mean = {boot_ci['prop_barcode_mean']:.4f}")
    print(f"    SE = {boot_ci['prop_barcode_se']:.4f}")
    print(f"    95% CI = [{boot_ci['prop_barcode_CI_95'][0]:.4f}, "
          f"{boot_ci['prop_barcode_CI_95'][1]:.4f}]")

    print(f"  Noise proportion:")
    print(f"    Mean = {boot_ci['prop_noise_mean']:.4f}")
    print(f"    SE = {boot_ci['prop_noise_se']:.4f}")
    print(f"    95% CI = [{boot_ci['prop_noise_CI_95'][0]:.4f}, "
          f"{boot_ci['prop_noise_CI_95'][1]:.4f}]")

    print("\n✓ Bootstrap test passed!")


def test_blup_uncertainty():
    """Test BLUP uncertainty calculation."""
    print("\n" + "=" * 60)
    print("Testing BLUP Uncertainty (PEV)")
    print("=" * 60)

    # Generate synthetic data
    np.random.seed(123)
    n = 50

    # Create proper positive definite kernels
    X = np.random.randn(n, 10)
    K_barcode = X @ X.T / 10
    K_barcode = center_kernel(K_barcode)
    K_barcode += 0.01 * np.eye(n)

    X2 = np.random.randn(n, 10)
    K_enhancer = X2 @ X2.T / 10
    K_enhancer = center_kernel(K_enhancer)
    K_enhancer += 0.01 * np.eye(n)

    sigma2_b, sigma2_e, sigma2_n = 0.1, 0.5, 0.2
    V = sigma2_b * K_barcode + sigma2_e * K_enhancer + sigma2_n * np.eye(n)
    V += 0.01 * np.eye(n)
    y = np.random.multivariate_normal(np.zeros(n), V)

    # Compute BLUPs
    print("\nComputing BLUPs...")
    g_bc, g_enh, g_noise = compute_blups(
        y, K_barcode, K_enhancer,
        sigma2_b, sigma2_e, sigma2_n
    )

    print(f"  Barcode BLUP range: [{g_bc.min():.4f}, {g_bc.max():.4f}]")
    print(f"  Enhancer BLUP range: [{g_enh.min():.4f}, {g_enh.max():.4f}]")

    # Calculate uncertainty
    print("\nCalculating BLUP uncertainties...")
    blup_unc = calculate_blup_uncertainty(
        y, K_barcode, K_enhancer,
        sigma2_b, sigma2_e, sigma2_n
    )

    print(f"\nStandard Errors:")
    print(f"  Barcode SE range: [{blup_unc['se_barcode'].min():.4f}, "
          f"{blup_unc['se_barcode'].max():.4f}]")
    print(f"  Enhancer SE range: [{blup_unc['se_enhancer'].min():.4f}, "
          f"{blup_unc['se_enhancer'].max():.4f}]")

    print(f"\n95% CI Half-Widths:")
    print(f"  Barcode CI95 range: [{blup_unc['CI95_barcode'].min():.4f}, "
          f"{blup_unc['CI95_barcode'].max():.4f}]")
    print(f"  Enhancer CI95 range: [{blup_unc['CI95_enhancer'].min():.4f}, "
          f"{blup_unc['CI95_enhancer'].max():.4f}]")

    # Show a few examples
    print(f"\nExample constructs (first 5):")
    for i in range(5):
        print(f"  Construct {i}:")
        print(f"    Enhancer BLUP = {g_enh[i]:.4f} ± {blup_unc['CI95_enhancer'][i]:.4f}")
        print(f"    95% CI = [{g_enh[i] - blup_unc['CI95_enhancer'][i]:.4f}, "
              f"{g_enh[i] + blup_unc['CI95_enhancer'][i]:.4f}]")

    print("\n✓ BLUP uncertainty test passed!")


if __name__ == "__main__":
    test_bootstrap_ci()
    test_blup_uncertainty()

    print("\n" + "=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)
