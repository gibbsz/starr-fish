"""Partial Mantel test implementation."""

import numpy as np
from scipy.spatial.distance import squareform
from scipy.stats import pearsonr


def _upper_tri(D):
    """Extract the upper triangle of a square matrix as a 1D vector."""
    return squareform(D, checks=False)


def _residuals(x, z):
    """Regress x on z and return residuals (OLS)."""
    z = z.copy()
    # Add intercept
    Z = np.column_stack([np.ones(len(z)), z])
    beta = np.linalg.lstsq(Z, x, rcond=None)[0]
    return x - Z @ beta


def mantel_test(D1, D2, n_permutations=9999, seed=42):
    """Standard Mantel test: Pearson correlation between distance matrices.

    Parameters
    ----------
    D1, D2 : np.ndarray
        Square distance matrices of the same size.
    n_permutations : int
        Number of permutations.

    Returns
    -------
    r_obs : float
        Observed Pearson correlation.
    p_value : float
        Permutation p-value (proportion of permuted r >= observed r).
    """
    rng = np.random.default_rng(seed)
    n = D1.shape[0]
    d1 = _upper_tri(D1)
    d2 = _upper_tri(D2)

    r_obs = np.corrcoef(d1, d2)[0, 1]

    count = 0
    for _ in range(n_permutations):
        perm = rng.permutation(n)
        D2_perm = D2[np.ix_(perm, perm)]
        d2_perm = _upper_tri(D2_perm)
        r_perm = np.corrcoef(d1, d2_perm)[0, 1]
        if r_perm >= r_obs:
            count += 1

    p_value = (count + 1) / (n_permutations + 1)
    return r_obs, p_value


def partial_mantel_test(D_y, D_x, D_z, n_permutations=9999, seed=42):
    """Partial Mantel test: correlation between D_y and D_x after removing D_z.

    Tests whether D_y is associated with D_x after partialing out D_z.

    Steps:
    1. Flatten upper triangles of all three matrices.
    2. Regress D_y on D_z → residuals_y.
    3. Regress D_x on D_z → residuals_x.
    4. Correlate residuals.
    5. Permute rows/columns of D_x and repeat for significance.

    Parameters
    ----------
    D_y : np.ndarray
        Response distance matrix (e.g. activity).
    D_x : np.ndarray
        Predictor distance matrix (e.g. barcode).
    D_z : np.ndarray
        Covariate distance matrix to partial out (e.g. enhancer).
    n_permutations : int
    seed : int

    Returns
    -------
    r_obs : float
        Partial Mantel r.
    p_value : float
    """
    rng = np.random.default_rng(seed)
    n = D_y.shape[0]

    dy = _upper_tri(D_y)
    dx = _upper_tri(D_x)
    dz = _upper_tri(D_z)

    # Partial out D_z
    res_y = _residuals(dy, dz)
    res_x = _residuals(dx, dz)
    r_obs = np.corrcoef(res_y, res_x)[0, 1]

    count = 0
    for _ in range(n_permutations):
        perm = rng.permutation(n)
        D_x_perm = D_x[np.ix_(perm, perm)]
        dx_perm = _upper_tri(D_x_perm)
        res_x_perm = _residuals(dx_perm, dz)
        r_perm = np.corrcoef(res_y, res_x_perm)[0, 1]
        if r_perm >= r_obs:
            count += 1

    p_value = (count + 1) / (n_permutations + 1)
    return r_obs, p_value


def run_partial_mantel(D_activity, D_barcode, D_enhancer,
                       n_permutations=9999, seed=42):
    """Run both partial Mantel tests and print summary.

    Test 1: corr(D_activity, D_barcode | D_enhancer)
    Test 2: corr(D_activity, D_enhancer | D_barcode)

    Returns
    -------
    results : dict
        Keys: 'barcode_r', 'barcode_p', 'enhancer_r', 'enhancer_p'.
    """
    print("=" * 60)
    print("PARTIAL MANTEL TESTS")
    print("=" * 60)
    print(f"Permutations: {n_permutations}")
    print()

    # Test 1: barcode effect after controlling for enhancer
    print("Test 1: corr(D_activity, D_barcode | D_enhancer)")
    r_bc, p_bc = partial_mantel_test(
        D_activity, D_barcode, D_enhancer,
        n_permutations=n_permutations, seed=seed,
    )
    print(f"  Partial Mantel r = {r_bc:.4f}, p = {p_bc:.4f}")
    sig_bc = "SIGNIFICANT" if p_bc < 0.05 else "not significant"
    print(f"  → Barcode effect: {sig_bc} (p {'<' if p_bc < 0.05 else '>'} 0.05)")
    print()

    # Test 2: enhancer effect after controlling for barcode
    print("Test 2: corr(D_activity, D_enhancer | D_barcode)")
    r_enh, p_enh = partial_mantel_test(
        D_activity, D_enhancer, D_barcode,
        n_permutations=n_permutations, seed=seed + 1,
    )
    print(f"  Partial Mantel r = {r_enh:.4f}, p = {p_enh:.4f}")
    sig_enh = "SIGNIFICANT" if p_enh < 0.05 else "not significant"
    print(f"  → Enhancer effect: {sig_enh} (p {'<' if p_enh < 0.05 else '>'} 0.05)")
    print()

    # Summary
    print("-" * 60)
    if p_enh < 0.05 and p_bc >= 0.05:
        print("SUMMARY: Enhancer dominates activity variation; barcode effect "
              "not significant after controlling for enhancer similarity.")
    elif p_enh < 0.05 and p_bc < 0.05:
        print("SUMMARY: Both enhancer and barcode show significant associations "
              "with activity. Enhancer effect is likely stronger.")
    elif p_enh >= 0.05 and p_bc < 0.05:
        print("SUMMARY: Surprisingly, barcode effect is significant but enhancer "
              "is not. Investigate data quality.")
    else:
        print("SUMMARY: Neither barcode nor enhancer shows significant association "
              "with activity after mutual adjustment.")
    print("=" * 60)

    return {
        "barcode_r": r_bc, "barcode_p": p_bc,
        "enhancer_r": r_enh, "enhancer_p": p_enh,
    }
