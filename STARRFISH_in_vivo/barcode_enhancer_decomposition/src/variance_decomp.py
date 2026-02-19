"""Kernel variance decomposition via REML (optimized implementation)."""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.linalg import cho_factor, cho_solve
from scipy.stats import chi2
from tqdm import tqdm
from joblib import Parallel, delayed

from .sequence_kernels import center_kernel


def kernel_correlation(K1, K2):
    """Compute Pearson correlation between off-diagonal elements of two kernels.

    This measures the collinearity between two kernel matrices. High correlation
    (> 0.8) indicates the kernels are too similar, making it difficult to
    distinguish between the two variance components.

    Parameters
    ----------
    K1, K2 : np.ndarray
        Kernel matrices (n x n).

    Returns
    -------
    float
        Pearson correlation coefficient between off-diagonal elements.
    """
    n = K1.shape[0]
    # Extract upper triangular off-diagonal elements
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    k1_offdiag = K1[mask]
    k2_offdiag = K2[mask]

    # Compute Pearson correlation
    corr = np.corrcoef(k1_offdiag, k2_offdiag)[0, 1]
    return corr


def check_kernel_structure(K):
    """Check if kernel has meaningful structure via eigenvalue spectrum CV.

    Uses Coefficient of Variation (CV) of eigenvalues to detect flat spectrum.
    If K ≈ I (identity matrix), all eigenvalues ≈ 1, so CV ≈ 0 (flat).
    If K has structure (clusters/gradients), eigenvalues vary, so CV > 0.1.

    Parameters
    ----------
    K : np.ndarray
        Kernel matrix (n x n), centered.

    Returns
    -------
    dict
        'eigenvalue_cv': Coefficient of variation of eigenvalues
        'has_structure': Boolean, True if CV > 0.1 (structured)
                        False if CV ≤ 0.1 (flat, use ridge regression instead)
    """
    # Compute eigenvalues
    eigvals = np.linalg.eigvalsh(K)
    eigvals = np.maximum(eigvals, 0)  # Remove small negative values from numerical error

    # Filter out near-zero eigenvalues
    eigvals_nonzero = eigvals[eigvals > 1e-10]

    if len(eigvals_nonzero) < 2:
        # Not enough eigenvalues to compute CV
        return {'eigenvalue_cv': 0.0, 'has_structure': False}

    # Coefficient of Variation: CV = std(λ) / mean(λ)
    mean_eig = np.mean(eigvals_nonzero)
    std_eig = np.std(eigvals_nonzero)

    if mean_eig > 1e-10:
        cv = std_eig / mean_eig
    else:
        cv = 0.0

    # Decision: has structure if CV > 0.1 (10%)
    # CV ≤ 0.1 means eigenvalues are all similar (flat spectrum, K ≈ I)
    has_structure = cv > 0.1

    return {
        'eigenvalue_cv': cv,
        'has_structure': has_structure,
    }


def qq_correlation(residuals):
    """Compute QQ plot correlation for residual normality check.

    Measures how well residuals follow a normal distribution by correlating
    observed quantiles with theoretical normal quantiles.

    Parameters
    ----------
    residuals : np.ndarray
        Model residuals.

    Returns
    -------
    float
        Pearson correlation between theoretical and observed quantiles.
        Values close to 1.0 indicate good normality. < 0.95 suggests
        transformation may be needed.
    """
    from scipy import stats

    # Remove NaN/Inf
    res = residuals[np.isfinite(residuals)]
    if len(res) < 3:
        return np.nan

    # Standardize residuals
    res_std = (res - np.mean(res)) / (np.std(res) + 1e-10)

    # Sort residuals
    res_sorted = np.sort(res_std)

    # Theoretical quantiles from standard normal
    n = len(res_sorted)
    theoretical_quantiles = stats.norm.ppf(np.arange(1, n + 1) / (n + 1))

    # Pearson correlation
    corr = np.corrcoef(theoretical_quantiles, res_sorted)[0, 1]
    return corr


def _reml_negloglik(log_theta, y, X, K_barcode, K_enhancer, n, p):
    """Negative restricted log-likelihood for the multi-kernel LMM.

    Optimized: uses cho_factor/cho_solve instead of full matrix inverse.
    """
    s2_bc, s2_enh, s2_noise = np.exp(log_theta)

    V = s2_bc * K_barcode + s2_enh * K_enhancer + s2_noise * np.eye(n)
    V.flat[::n + 1] += 1e-6  # jitter on diagonal (in-place)

    try:
        L_factor = cho_factor(V)
    except np.linalg.LinAlgError:
        return 1e10

    # log|V| via Cholesky
    logdet_V = 2.0 * np.sum(np.log(np.diag(L_factor[0])))

    # V^{-1} X and V^{-1} y via cho_solve (O(n^2) each, not O(n^3))
    Vinv_X = cho_solve(L_factor, X)       # (n, p)
    Vinv_y = cho_solve(L_factor, y)       # (n,)

    # X^T V^{-1} X  (p x p, here 1x1)
    XtVinvX = X.T @ Vinv_X               # (p, p)
    sign, logdet_XtVinvX = np.linalg.slogdet(XtVinvX)
    if sign <= 0:
        return 1e10

    # P y = V^{-1} y - V^{-1} X (X^T V^{-1} X)^{-1} X^T V^{-1} y
    XtVinv_y = X.T @ Vinv_y              # (p,)
    XtVinvX_inv_XtVinv_y = np.linalg.solve(XtVinvX, XtVinv_y)  # (p,)
    Py = Vinv_y - Vinv_X @ XtVinvX_inv_XtVinv_y  # (n,)

    yPy = y @ Py

    nll = 0.5 * ((n - p) * np.log(2 * np.pi) + logdet_V + logdet_XtVinvX + yPy)
    return nll


def _reml_negloglik_reduced(log_theta, y, X, K_enhancer, n, p):
    """Negative REML log-likelihood for reduced model (no barcode kernel)."""
    s2_enh, s2_noise = np.exp(log_theta)
    V = s2_enh * K_enhancer + s2_noise * np.eye(n)
    V.flat[::n + 1] += 1e-6

    try:
        L_factor = cho_factor(V)
    except np.linalg.LinAlgError:
        return 1e10

    logdet_V = 2.0 * np.sum(np.log(np.diag(L_factor[0])))

    Vinv_X = cho_solve(L_factor, X)
    Vinv_y = cho_solve(L_factor, y)

    XtVinvX = X.T @ Vinv_X
    sign, logdet_XtVinvX = np.linalg.slogdet(XtVinvX)
    if sign <= 0:
        return 1e10

    XtVinv_y = X.T @ Vinv_y
    XtVinvX_inv_XtVinv_y = np.linalg.solve(XtVinvX, XtVinv_y)
    Py = Vinv_y - Vinv_X @ XtVinvX_inv_XtVinv_y

    yPy = y @ Py

    nll = 0.5 * ((n - p) * np.log(2 * np.pi) + logdet_V + logdet_XtVinvX + yPy)
    return nll


def compute_blups(y, K_barcode, K_enhancer,
                  sigma2_barcode, sigma2_enhancer, sigma2_noise):
    """Compute BLUPs (Best Linear Unbiased Predictions) for random effects.

    Given REML-estimated variance components and kernel matrices, compute
    the individual barcode artifact scores, enhancer biological activity
    scores, and noise residuals for each construct.

    The model decomposes each observation as:
        y_i = mu + g_barcode_i + g_enhancer_i + epsilon_i

    BLUPs are computed as:
        g_barcode  = sigma2_B * K_B * V^{-1} * (y - X * beta_GLS)
        g_enhancer = sigma2_E * K_E * V^{-1} * (y - X * beta_GLS)
        epsilon    = y - mu - g_barcode - g_enhancer
    where V = sigma2_B * K_B + sigma2_E * K_E + sigma2_noise * I
    and beta_GLS = (X' V^{-1} X)^{-1} X' V^{-1} y.

    Parameters
    ----------
    y : np.ndarray, shape (n,)
        Phenotype vector (activity for one cell type).
    K_barcode, K_enhancer : np.ndarray, shape (n, n)
        Centered kernel matrices.
    sigma2_barcode, sigma2_enhancer, sigma2_noise : float
        REML-estimated variance components.

    Returns
    -------
    g_barcode : np.ndarray, shape (n,)
        BLUP of barcode effects (artifact scores).
    g_enhancer : np.ndarray, shape (n,)
        BLUP of enhancer effects (biological activity scores).
    g_noise : np.ndarray, shape (n,)
        Noise residuals (activity that fits neither pattern).
    """
    n = len(y)
    X = np.ones((n, 1))

    # Total covariance
    V = (sigma2_barcode * K_barcode
         + sigma2_enhancer * K_enhancer
         + sigma2_noise * np.eye(n))
    V.flat[::n + 1] += 1e-6  # jitter for numerical stability

    L_factor = cho_factor(V)
    Vinv_y = cho_solve(L_factor, y)
    Vinv_X = cho_solve(L_factor, X)

    # GLS fixed effect: beta = (X'V^{-1}X)^{-1} X'V^{-1}y
    beta = np.linalg.solve(X.T @ Vinv_X, X.T @ Vinv_y)  # (1,)
    mu = (X @ beta).ravel()

    # Residual with GLS mean removed
    e = y - mu
    Vinv_e = cho_solve(L_factor, e)

    # BLUPs: project residual through each kernel
    g_barcode = sigma2_barcode * (K_barcode @ Vinv_e)
    g_enhancer = sigma2_enhancer * (K_enhancer @ Vinv_e)

    # Noise: residual activity that fits neither barcode nor enhancer pattern
    g_noise = y - mu - g_barcode - g_enhancer

    return g_barcode.ravel(), g_enhancer.ravel(), g_noise.ravel()


def _bootstrap_iteration(seed, n, mu_hat, V_est, K_barcode, K_enhancer, max_iter):
    """Single bootstrap iteration (for parallelization).

    Returns tuple of (prop_barcode, prop_enhancer, prop_noise) or None if failed.
    """
    np.random.seed(seed)
    try:
        # 1. Generate synthetic data from estimated distribution
        y_sim = np.random.multivariate_normal(
            mean=np.full(n, mu_hat), cov=V_est
        )

        # 2. Refit the model
        res = fit_reml(y_sim, K_barcode, K_enhancer, max_iter=max_iter)

        # 3. Calculate proportions for this simulation
        s2_b = res['sigma2_barcode']
        s2_e = res['sigma2_enhancer']
        s2_n = res['sigma2_noise']
        total = s2_b + s2_e + s2_n

        if total > 1e-15:
            return (s2_b / total, s2_e / total, s2_n / total)
        else:
            return None

    except Exception:
        # Skip failed fits
        return None


def bootstrap_confidence_intervals(
    y_real, K_barcode, K_enhancer,
    sigma2_b_est, sigma2_e_est, sigma2_n_est,
    n_boot=100, max_iter=50, n_jobs=-1, random_seed=42
):
    """Estimate confidence intervals for variance proportions using parametric bootstrap.

    Parameters
    ----------
    y_real : np.ndarray
        Observed phenotype vector.
    K_barcode, K_enhancer : np.ndarray
        Centered kernel matrices.
    sigma2_b_est, sigma2_e_est, sigma2_n_est : float
        Estimated variance components from real data.
    n_boot : int
        Number of bootstrap iterations.
    max_iter : int
        Maximum iterations for each bootstrap REML fit (use fewer for speed).
    n_jobs : int
        Number of parallel jobs. -1 uses all available cores.
    random_seed : int
        Random seed for reproducibility.

    Returns
    -------
    dict with keys:
        'prop_barcode_mean', 'prop_barcode_se', 'prop_barcode_CI_95',
        'prop_enhancer_mean', 'prop_enhancer_se', 'prop_enhancer_CI_95',
        'prop_noise_mean', 'prop_noise_se', 'prop_noise_CI_95'
    """
    n = len(y_real)

    # Pre-compute covariance matrix for simulation
    V_est = (sigma2_b_est * K_barcode
             + sigma2_e_est * K_enhancer
             + sigma2_n_est * np.eye(n))
    V_est.flat[::n + 1] += 1e-6  # ensure positive definite

    mu_hat = np.mean(y_real)

    # Generate different seeds for each bootstrap iteration
    rng = np.random.RandomState(random_seed)
    seeds = rng.randint(0, 2**31 - 1, size=n_boot)

    # Run bootstrap iterations in parallel
    results = Parallel(n_jobs=n_jobs, backend='loky')(
        delayed(_bootstrap_iteration)(
            seed, n, mu_hat, V_est, K_barcode, K_enhancer, max_iter
        )
        for seed in seeds
    )

    # Filter out failed iterations
    results = [r for r in results if r is not None]

    if len(results) < 10:
        # Not enough successful bootstraps
        return {
            'prop_barcode_mean': np.nan, 'prop_barcode_se': np.nan,
            'prop_barcode_CI_95': (np.nan, np.nan),
            'prop_enhancer_mean': np.nan, 'prop_enhancer_se': np.nan,
            'prop_enhancer_CI_95': (np.nan, np.nan),
            'prop_noise_mean': np.nan, 'prop_noise_se': np.nan,
            'prop_noise_CI_95': (np.nan, np.nan),
        }

    # Unpack results
    prop_barcode_boots = np.array([r[0] for r in results])
    prop_enhancer_boots = np.array([r[1] for r in results])
    prop_noise_boots = np.array([r[2] for r in results])

    return {
        'prop_barcode_mean': np.mean(prop_barcode_boots),
        'prop_barcode_se': np.std(prop_barcode_boots),
        'prop_barcode_CI_95': tuple(np.percentile(prop_barcode_boots, [2.5, 97.5])),
        'prop_enhancer_mean': np.mean(prop_enhancer_boots),
        'prop_enhancer_se': np.std(prop_enhancer_boots),
        'prop_enhancer_CI_95': tuple(np.percentile(prop_enhancer_boots, [2.5, 97.5])),
        'prop_noise_mean': np.mean(prop_noise_boots),
        'prop_noise_se': np.std(prop_noise_boots),
        'prop_noise_CI_95': tuple(np.percentile(prop_noise_boots, [2.5, 97.5])),
    }


def calculate_blup_uncertainty(y, K_barcode, K_enhancer,
                                 sigma2_b, sigma2_e, sigma2_n):
    """Calculate standard errors for BLUPs using Prediction Error Variance (PEV).

    The PEV quantifies uncertainty in BLUP predictions. Barcodes/enhancers that
    are highly similar to others (high kernel connectivity) will have lower PEV
    (more confident predictions), while isolated sequences have higher PEV.

    Formula: PEV(g) = G - G * P * G
    where P = V^{-1} - V^{-1} X (X' V^{-1} X)^{-1} X' V^{-1}

    Parameters
    ----------
    y : np.ndarray
        Phenotype vector.
    K_barcode, K_enhancer : np.ndarray
        Centered kernel matrices.
    sigma2_b, sigma2_e, sigma2_n : float
        Estimated variance components.

    Returns
    -------
    dict with keys:
        'se_barcode': np.ndarray, standard errors for barcode BLUPs
        'se_enhancer': np.ndarray, standard errors for enhancer BLUPs
        'CI95_barcode': np.ndarray, 95% CI half-width for barcode BLUPs
        'CI95_enhancer': np.ndarray, 95% CI half-width for enhancer BLUPs
    """
    n = len(y)
    X = np.ones((n, 1))

    # 1. Reconstruct V
    V = (sigma2_b * K_barcode
         + sigma2_e * K_enhancer
         + sigma2_n * np.eye(n))
    V.flat[::n + 1] += 1e-6  # jitter for stability

    # 2. Calculate V^{-1}
    try:
        V_inv = np.linalg.inv(V)
    except np.linalg.LinAlgError:
        V_inv = np.linalg.pinv(V)

    # 3. Construct the P matrix
    # P = V^{-1} - V^{-1} X (X' V^{-1} X)^{-1} X' V^{-1}
    XtVinv = X.T @ V_inv
    XtVinvX = XtVinv @ X
    XtVinvX_inv = np.linalg.inv(XtVinvX)

    Correction = V_inv @ X @ XtVinvX_inv @ XtVinv
    P = V_inv - Correction

    # 4. Calculate PEV for Enhancers
    # PEV_E = G_E - G_E * P * G_E
    # We only need diagonal elements for standard errors
    G_e = sigma2_e * K_enhancer

    # Efficient diagonal calculation: diag(G_e @ P @ G_e)
    Ge_P = G_e @ P
    term2_diag_e = np.einsum('ij,ji->i', Ge_P, G_e)
    prior_diag_e = np.diag(G_e)

    pev_enhancer = prior_diag_e - term2_diag_e
    pev_enhancer = np.maximum(pev_enhancer, 0)  # clip numerical noise
    se_enhancer = np.sqrt(pev_enhancer)

    # 5. Calculate PEV for Barcodes
    G_b = sigma2_b * K_barcode
    Gb_P = G_b @ P
    term2_diag_b = np.einsum('ij,ji->i', Gb_P, G_b)
    prior_diag_b = np.diag(G_b)

    pev_barcode = prior_diag_b - term2_diag_b
    pev_barcode = np.maximum(pev_barcode, 0)
    se_barcode = np.sqrt(pev_barcode)

    return {
        "se_enhancer": se_enhancer,
        "se_barcode": se_barcode,
        "CI95_enhancer": 1.96 * se_enhancer,
        "CI95_barcode": 1.96 * se_barcode,
    }


def fit_reml(y, K_barcode, K_enhancer, max_iter=200, tol=1e-5):
    """Fit the two-kernel LMM via REML for a single phenotype vector.

    K_barcode and K_enhancer should already be centered.
    """
    n = len(y)
    p = 1
    X = np.ones((n, 1))
    var_y = np.var(y)
    if not np.isfinite(var_y) or var_y < 1e-12:
        var_y = 1.0
    init_var = max(var_y / 3, 1e-4)

    # Full model
    log_theta0 = np.log(np.array([init_var, init_var, init_var]))
    res_full = minimize(
        _reml_negloglik, log_theta0,
        args=(y, X, K_barcode, K_enhancer, n, p),
        method="L-BFGS-B",
        options={"maxiter": max_iter, "ftol": tol},
    )
    s2_bc, s2_enh, s2_noise = np.exp(res_full.x)
    nll_full = res_full.fun

    # Reduced model (no barcode)
    log_theta0_red = np.log(np.array([init_var, init_var]))
    res_red = minimize(
        _reml_negloglik_reduced, log_theta0_red,
        args=(y, X, K_enhancer, n, p),
        method="L-BFGS-B",
        options={"maxiter": max_iter, "ftol": tol},
    )
    nll_reduced = res_red.fun

    # LRT statistic
    lrt = 2 * (nll_reduced - nll_full)
    lrt = max(lrt, 0.0)

    return {
        "sigma2_barcode": s2_bc,
        "sigma2_enhancer": s2_enh,
        "sigma2_noise": s2_noise,
        "nll_full": nll_full,
        "nll_reduced": nll_reduced,
        "LRT_stat": lrt,
        "converged": res_full.success,
    }


def lrt_pvalue_boundary(lrt_stat):
    """P-value for LRT on the boundary (0.5*chi2(0) + 0.5*chi2(1) mixture)."""
    if lrt_stat <= 0:
        return 1.0
    return 0.5 * chi2.sf(lrt_stat, df=1)


def fdr_correction(pvalues, alpha=0.05):
    """Benjamini-Hochberg FDR correction. Returns adjusted p-values."""
    pvals = np.asarray(pvalues, dtype=float)
    n = len(pvals)
    order = np.argsort(pvals)
    ranks = np.empty(n, dtype=int)
    ranks[order] = np.arange(1, n + 1)

    adjusted = pvals * n / ranks
    adjusted = np.minimum(adjusted, 1.0)
    # Ensure monotonicity (step-up)
    sorted_idx = np.argsort(ranks)[::-1]
    for i in range(1, n):
        idx = sorted_idx[i]
        idx_prev = sorted_idx[i - 1]
        if adjusted[idx] > adjusted[idx_prev]:
            adjusted[idx] = adjusted[idx_prev]

    return adjusted


def run_variance_decomposition(activity, K_barcode, K_enhancer,
                                max_iter=200, tol=1e-5, fdr_alpha=0.05,
                                n_boot=100, n_jobs=-1):
    """Run variance decomposition for all cell types with BLUP estimation and confidence intervals.

    Handles NaN values by subsetting to non-NaN CREs per cell type and
    re-centering the kernel matrices on the subset.

    After estimating variance components via REML, computes BLUPs (Best
    Linear Unbiased Predictions) for each construct, decomposing each
    observation into barcode artifact and enhancer biological activity scores.

    Also computes bootstrap confidence intervals for variance proportions
    and prediction error variance (PEV) based standard errors for BLUPs.

    Parameters
    ----------
    activity : pd.DataFrame or np.ndarray
        Shape (n_constructs, n_celltypes). May contain NaN.
    K_barcode, K_enhancer : np.ndarray
        Un-centered kernel matrices, shape (n_constructs, n_constructs).
        These will be subset and centered per cell type.
    max_iter : int
    tol : float
    fdr_alpha : float
    n_boot : int
        Number of bootstrap iterations for CI estimation (default: 100).
    n_jobs : int
        Number of parallel jobs for bootstrap. -1 uses all cores (default: -1).

    Returns
    -------
    results_df : pd.DataFrame
        One row per cell type with variance components, test statistics,
        and bootstrap confidence intervals for proportions.
    blup_df : pd.DataFrame
        Long-format table with columns: construct, cell_type,
        blup_barcode, blup_enhancer, blup_noise, prop_barcode, prop_enhancer, prop_noise,
        se_barcode, se_enhancer, CI95_barcode, CI95_enhancer.
        Proportions are based on absolute BLUP values:
        prop_barcode = |g_bc| / (|g_bc| + |g_enh| + |g_noise|),
        prop_enhancer = |g_enh| / (|g_bc| + |g_enh| + |g_noise|),
        prop_noise = |g_noise| / (|g_bc| + |g_enh| + |g_noise|).
    """
    if isinstance(activity, pd.DataFrame):
        celltypes = activity.columns.tolist()
        construct_ids = activity.index.tolist()
        Y = activity.values
    else:
        Y = activity
        celltypes = [f"CellType{j}" for j in range(Y.shape[1])]
        construct_ids = [f"Construct{i}" for i in range(Y.shape[0])]

    n_constructs, n_celltypes = Y.shape
    records = []

    # Initialize BLUP arrays (NaN by default)
    blup_bc_array = np.full((n_constructs, n_celltypes + 1), np.nan)
    blup_enh_array = np.full((n_constructs, n_celltypes + 1), np.nan)
    blup_noise_array = np.full((n_constructs, n_celltypes + 1), np.nan)

    # Initialize BLUP uncertainty arrays (NaN by default)
    se_bc_array = np.full((n_constructs, n_celltypes + 1), np.nan)
    se_enh_array = np.full((n_constructs, n_celltypes + 1), np.nan)
    ci95_bc_array = np.full((n_constructs, n_celltypes + 1), np.nan)
    ci95_enh_array = np.full((n_constructs, n_celltypes + 1), np.nan)

    _nan_record = lambda ct, nv, kcorr=np.nan, k_bc_struct=True, k_enh_struct=True, k_bc_cv=np.nan, k_enh_cv=np.nan, qq_corr=np.nan: {
        "cell_type": ct, "n_valid": int(nv),
        "kernel_correlation": kcorr,
        "barcode_kernel_has_structure": k_bc_struct,
        "barcode_eigenvalue_cv": k_bc_cv,
        "enhancer_kernel_has_structure": k_enh_struct,
        "enhancer_eigenvalue_cv": k_enh_cv,
        "residual_qq_correlation": qq_corr,
        "sigma2_barcode": np.nan, "sigma2_enhancer": np.nan,
        "sigma2_noise": np.nan, "prop_barcode": np.nan,
        "prop_enhancer": np.nan, "prop_noise": np.nan,
        "LRT_stat": np.nan, "p_value": np.nan,
        "prop_barcode_CI_low": np.nan, "prop_barcode_CI_high": np.nan,
        "prop_barcode_CI95": np.nan,
        "prop_enhancer_CI_low": np.nan, "prop_enhancer_CI_high": np.nan,
        "prop_enhancer_CI95": np.nan,
        "prop_noise_CI_low": np.nan, "prop_noise_CI_high": np.nan,
        "prop_noise_CI95": np.nan,
    }

    for j in tqdm(range(n_celltypes), desc="Variance decomposition"):
        y = Y[:, j].copy()

        # Find non-NaN / non-Inf entries
        mask = np.isfinite(y)
        n_valid = mask.sum()

        if n_valid < 10:
            records.append(_nan_record(celltypes[j], n_valid))
            continue

        # Subset to valid CREs
        y_sub = y[mask]

        # Skip if zero variance (all identical values)
        if np.var(y_sub) < 1e-12:
            print(f"  Warning: skipping {celltypes[j]} (zero variance)")
            records.append(_nan_record(celltypes[j], n_valid))
            continue

        # Subset and re-center kernel matrices
        K_bc_sub = center_kernel(K_barcode[np.ix_(mask, mask)])
        K_enh_sub = center_kernel(K_enhancer[np.ix_(mask, mask)])

        # Phase 1: Pre-flight checks
        # 1. Check kernel structure (PCA eigenvalue spectrum)
        bc_structure = check_kernel_structure(K_bc_sub)
        enh_structure = check_kernel_structure(K_enh_sub)

        if not bc_structure['has_structure']:
            print(f"  WARNING: Barcode kernel for {celltypes[j]} has no structure (flat spectrum)")
            print(f"           Eigenvalue CV={bc_structure['eigenvalue_cv']:.3f} ≤ 0.1")
            print(f"           Kernel ≈ Identity. Use ridge regression instead of kernel.")

        if not enh_structure['has_structure']:
            print(f"  WARNING: Enhancer kernel for {celltypes[j]} has no structure (flat spectrum)")
            print(f"           Eigenvalue CV={enh_structure['eigenvalue_cv']:.3f} ≤ 0.1")
            print(f"           Kernel ≈ Identity. Use ridge regression instead.")

        # 2. Compute kernel correlation (collinearity check)
        k_corr = kernel_correlation(K_bc_sub, K_enh_sub)
        if k_corr > 0.8:
            print(f"  ✗ FAILED: Kernel correlation for {celltypes[j]}: r={k_corr:.3f} > 0.8")
            print(f"           Barcode and enhancer kernels are too similar (collinear).")
            print(f"           Model cannot distinguish biology from artifacts.")
            print(f"           SKIPPING this cell type.")
            records.append(_nan_record(celltypes[j], n_valid, k_corr,
                                        bc_structure['has_structure'],
                                        enh_structure['has_structure'],
                                        bc_structure['eigenvalue_cv'],
                                        enh_structure['eigenvalue_cv']))
            continue  # Skip REML fitting for this cell type

        # Phase 2: Fit REML model
        try:
            res = fit_reml(y_sub, K_bc_sub, K_enh_sub,
                           max_iter=max_iter, tol=tol)
        except Exception as e:
            print(f"  Warning: REML failed for {celltypes[j]}: {e}")
            records.append(_nan_record(celltypes[j], n_valid, k_corr,
                                        bc_structure['has_structure'],
                                        enh_structure['has_structure'],
                                        bc_structure['eigenvalue_cv'],
                                        enh_structure['eigenvalue_cv']))
            continue

        total = res["sigma2_barcode"] + res["sigma2_enhancer"] + res["sigma2_noise"]
        p_val = lrt_pvalue_boundary(res["LRT_stat"])

        # Phase 2: Residual diagnostics (post-fit safety check)
        # Compute residuals: y - (mu + g_barcode + g_enhancer)
        n_sub = len(y_sub)
        X_sub = np.ones((n_sub, 1))
        V_sub = (res["sigma2_barcode"] * K_bc_sub
                 + res["sigma2_enhancer"] * K_enh_sub
                 + res["sigma2_noise"] * np.eye(n_sub))
        V_sub.flat[::n_sub + 1] += 1e-6

        from scipy.linalg import cho_factor, cho_solve
        L_factor = cho_factor(V_sub)
        Vinv_y = cho_solve(L_factor, y_sub)
        Vinv_X = cho_solve(L_factor, X_sub)
        beta = np.linalg.solve(X_sub.T @ Vinv_X, X_sub.T @ Vinv_y)
        mu_fitted = (X_sub @ beta).ravel()

        # BLUPs for residual calculation
        e_sub = y_sub - mu_fitted
        Vinv_e = cho_solve(L_factor, e_sub)
        g_bc_temp = res["sigma2_barcode"] * (K_bc_sub @ Vinv_e)
        g_enh_temp = res["sigma2_enhancer"] * (K_enh_sub @ Vinv_e)

        # Residuals
        residuals = y_sub - mu_fitted - g_bc_temp - g_enh_temp

        # QQ correlation check
        qq_corr = qq_correlation(residuals)
        if qq_corr < 0.95 and np.isfinite(qq_corr):
            print(f"  WARNING: Poor residual normality for {celltypes[j]}: QQ r={qq_corr:.3f}")
            print(f"           Consider log-transform or rank-based INT transformation.")
        elif qq_corr < 0.98 and np.isfinite(qq_corr):
            print(f"  Note: Moderate residual non-normality for {celltypes[j]}: QQ r={qq_corr:.3f}")

        # Bootstrap confidence intervals for proportions
        print(f"    Computing bootstrap CIs for {celltypes[j]}...")
        boot_ci = bootstrap_confidence_intervals(
            y_sub, K_bc_sub, K_enh_sub,
            res["sigma2_barcode"], res["sigma2_enhancer"], res["sigma2_noise"],
            n_boot=n_boot, n_jobs=n_jobs
        )

        # Calculate CI95 half-widths
        prop_bc_ci95 = (boot_ci['prop_barcode_CI_95'][1] - boot_ci['prop_barcode_CI_95'][0]) / 2
        prop_enh_ci95 = (boot_ci['prop_enhancer_CI_95'][1] - boot_ci['prop_enhancer_CI_95'][0]) / 2
        prop_noise_ci95 = (boot_ci['prop_noise_CI_95'][1] - boot_ci['prop_noise_CI_95'][0]) / 2

        records.append({
            "cell_type": celltypes[j],
            "n_valid": int(n_valid),
            "kernel_correlation": k_corr,
            "barcode_kernel_has_structure": bc_structure['has_structure'],
            "barcode_eigenvalue_cv": bc_structure['eigenvalue_cv'],
            "enhancer_kernel_has_structure": enh_structure['has_structure'],
            "enhancer_eigenvalue_cv": enh_structure['eigenvalue_cv'],
            "residual_qq_correlation": qq_corr,
            "sigma2_barcode": res["sigma2_barcode"],
            "sigma2_enhancer": res["sigma2_enhancer"],
            "sigma2_noise": res["sigma2_noise"],
            "prop_barcode": res["sigma2_barcode"] / total if total > 0 else 0,
            "prop_enhancer": res["sigma2_enhancer"] / total if total > 0 else 0,
            "prop_noise": res["sigma2_noise"] / total if total > 0 else 0,
            "LRT_stat": res["LRT_stat"],
            "p_value": p_val,
            "prop_barcode_CI_low": boot_ci['prop_barcode_CI_95'][0],
            "prop_barcode_CI_high": boot_ci['prop_barcode_CI_95'][1],
            "prop_barcode_CI95": prop_bc_ci95,
            "prop_enhancer_CI_low": boot_ci['prop_enhancer_CI_95'][0],
            "prop_enhancer_CI_high": boot_ci['prop_enhancer_CI_95'][1],
            "prop_enhancer_CI95": prop_enh_ci95,
            "prop_noise_CI_low": boot_ci['prop_noise_CI_95'][0],
            "prop_noise_CI_high": boot_ci['prop_noise_CI_95'][1],
            "prop_noise_CI95": prop_noise_ci95,
        })

        # Compute BLUPs for this cell type
        try:
            g_bc, g_enh, g_noise = compute_blups(
                y_sub, K_bc_sub, K_enh_sub,
                res["sigma2_barcode"], res["sigma2_enhancer"], res["sigma2_noise"],
            )
            blup_bc_array[mask, j] = g_bc
            blup_enh_array[mask, j] = g_enh
            blup_noise_array[mask, j] = g_noise

            # Compute BLUP uncertainty (PEV-based standard errors)
            blup_unc = calculate_blup_uncertainty(
                y_sub, K_bc_sub, K_enh_sub,
                res["sigma2_barcode"], res["sigma2_enhancer"], res["sigma2_noise"]
            )
            se_bc_array[mask, j] = blup_unc['se_barcode']
            se_enh_array[mask, j] = blup_unc['se_enhancer']
            ci95_bc_array[mask, j] = blup_unc['CI95_barcode']
            ci95_enh_array[mask, j] = blup_unc['CI95_enhancer']

        except Exception as e:
            print(f"  Warning: BLUP computation failed for {celltypes[j]}: {e}")

    # -- All cell types combined (Average across cell types) ----------
    print("  Fitting combined phenotype (Average across all cell types)...")
    # For each CRE: average activity across cell types, skip NaN values
    # If more than half of cell types have NaN, set to NaN
    n_cell_types = Y.shape[1]
    nan_threshold = n_cell_types / 2

    y_all = np.full(Y.shape[0], np.nan)
    for i in range(Y.shape[0]):
        n_nans = np.sum(~np.isfinite(Y[i, :]))
        if n_nans <= nan_threshold:
            y_all[i] = np.nanmean(Y[i, :])

    mask_all = np.isfinite(y_all)
    n_valid_all = mask_all.sum()

    if n_valid_all < 10:
        records.append(_nan_record("All_celltypes", n_valid_all))
    else:
        K_bc_all = center_kernel(K_barcode[np.ix_(mask_all, mask_all)])
        K_enh_all = center_kernel(K_enhancer[np.ix_(mask_all, mask_all)])

        # Pre-flight checks for combined phenotype
        bc_structure_all = check_kernel_structure(K_bc_all)
        enh_structure_all = check_kernel_structure(K_enh_all)

        if not bc_structure_all['has_structure']:
            print(f"  WARNING: Barcode kernel for All_celltypes has no structure")
        if not enh_structure_all['has_structure']:
            print(f"  WARNING: Enhancer kernel for All_celltypes has no structure")

        k_corr_all = kernel_correlation(K_bc_all, K_enh_all)
        if k_corr_all > 0.8:
            print(f"  ✗ FAILED: Kernel correlation for All_celltypes: r={k_corr_all:.3f} > 0.8")
            print(f"           Barcode and enhancer kernels are too similar (collinear).")
            print(f"           SKIPPING All_celltypes.")
            records.append(_nan_record("All_celltypes", n_valid_all, k_corr_all,
                                        bc_structure_all['has_structure'],
                                        enh_structure_all['has_structure'],
                                        bc_structure_all['eigenvalue_cv'],
                                        enh_structure_all['eigenvalue_cv']))
            res_all = None
        else:
            try:
                res_all = fit_reml(y_all[mask_all], K_bc_all, K_enh_all,
                                   max_iter=max_iter, tol=tol)
            except Exception as e:
                print(f"  Warning: REML failed for All_celltypes: {e}")
                records.append(_nan_record("All_celltypes", n_valid_all, k_corr_all,
                                            bc_structure_all['has_structure'],
                                            enh_structure_all['has_structure'],
                                            bc_structure_all['eigenvalue_cv'],
                                            enh_structure_all['eigenvalue_cv']))
                res_all = None

        if res_all is not None:
            total_all = (res_all["sigma2_barcode"] + res_all["sigma2_enhancer"]
                         + res_all["sigma2_noise"])
            p_val_all = lrt_pvalue_boundary(res_all["LRT_stat"])

            # Residual diagnostics for combined phenotype
            n_all = np.sum(mask_all)
            X_all = np.ones((n_all, 1))
            V_all = (res_all["sigma2_barcode"] * K_bc_all
                     + res_all["sigma2_enhancer"] * K_enh_all
                     + res_all["sigma2_noise"] * np.eye(n_all))
            V_all.flat[::n_all + 1] += 1e-6

            L_factor_all = cho_factor(V_all)
            y_all_valid = y_all[mask_all]
            Vinv_y_all = cho_solve(L_factor_all, y_all_valid)
            Vinv_X_all = cho_solve(L_factor_all, X_all)
            beta_all = np.linalg.solve(X_all.T @ Vinv_X_all, X_all.T @ Vinv_y_all)
            mu_fitted_all = (X_all @ beta_all).ravel()

            e_all = y_all_valid - mu_fitted_all
            Vinv_e_all = cho_solve(L_factor_all, e_all)
            g_bc_all_temp = res_all["sigma2_barcode"] * (K_bc_all @ Vinv_e_all)
            g_enh_all_temp = res_all["sigma2_enhancer"] * (K_enh_all @ Vinv_e_all)

            residuals_all = y_all_valid - mu_fitted_all - g_bc_all_temp - g_enh_all_temp
            qq_corr_all = qq_correlation(residuals_all)

            if qq_corr_all < 0.95 and np.isfinite(qq_corr_all):
                print(f"  WARNING: Poor residual normality for All_celltypes: QQ r={qq_corr_all:.3f}")

            # Bootstrap confidence intervals for combined phenotype
            print(f"    Computing bootstrap CIs for All_celltypes...")
            boot_ci_all = bootstrap_confidence_intervals(
                y_all_valid, K_bc_all, K_enh_all,
                res_all["sigma2_barcode"], res_all["sigma2_enhancer"],
                res_all["sigma2_noise"],
                n_boot=n_boot
            )

            # Calculate CI95 half-widths for All_celltypes
            prop_bc_ci95_all = (boot_ci_all['prop_barcode_CI_95'][1] - boot_ci_all['prop_barcode_CI_95'][0]) / 2
            prop_enh_ci95_all = (boot_ci_all['prop_enhancer_CI_95'][1] - boot_ci_all['prop_enhancer_CI_95'][0]) / 2
            prop_noise_ci95_all = (boot_ci_all['prop_noise_CI_95'][1] - boot_ci_all['prop_noise_CI_95'][0]) / 2

            records.append({
                "cell_type": "All_celltypes",
                "n_valid": int(n_valid_all),
                "kernel_correlation": k_corr_all,
                "barcode_kernel_has_structure": bc_structure_all['has_structure'],
                "barcode_eigenvalue_cv": bc_structure_all['eigenvalue_cv'],
                "enhancer_kernel_has_structure": enh_structure_all['has_structure'],
                "enhancer_eigenvalue_cv": enh_structure_all['eigenvalue_cv'],
                "residual_qq_correlation": qq_corr_all,
                "sigma2_barcode": res_all["sigma2_barcode"],
                "sigma2_enhancer": res_all["sigma2_enhancer"],
                "sigma2_noise": res_all["sigma2_noise"],
                "prop_barcode": res_all["sigma2_barcode"] / total_all if total_all > 0 else 0,
                "prop_enhancer": res_all["sigma2_enhancer"] / total_all if total_all > 0 else 0,
                "prop_noise": res_all["sigma2_noise"] / total_all if total_all > 0 else 0,
                "LRT_stat": res_all["LRT_stat"],
                "p_value": p_val_all,
                "prop_barcode_CI_low": boot_ci_all['prop_barcode_CI_95'][0],
                "prop_barcode_CI_high": boot_ci_all['prop_barcode_CI_95'][1],
                "prop_barcode_CI95": prop_bc_ci95_all,
                "prop_enhancer_CI_low": boot_ci_all['prop_enhancer_CI_95'][0],
                "prop_enhancer_CI_high": boot_ci_all['prop_enhancer_CI_95'][1],
                "prop_enhancer_CI95": prop_enh_ci95_all,
                "prop_noise_CI_low": boot_ci_all['prop_noise_CI_95'][0],
                "prop_noise_CI_high": boot_ci_all['prop_noise_CI_95'][1],
                "prop_noise_CI95": prop_noise_ci95_all,
            })

            # BLUPs for combined phenotype
            try:
                g_bc_all, g_enh_all, g_noise_all = compute_blups(
                    y_all, K_bc_all, K_enh_all,
                    res_all["sigma2_barcode"], res_all["sigma2_enhancer"],
                    res_all["sigma2_noise"],
                )
                blup_bc_array[mask_all, -1] = g_bc_all
                blup_enh_array[mask_all, -1] = g_enh_all
                blup_noise_array[mask_all, -1] = g_noise_all

                # Compute BLUP uncertainty for combined phenotype
                blup_unc_all = calculate_blup_uncertainty(
                    y_all, K_bc_all, K_enh_all,
                    res_all["sigma2_barcode"], res_all["sigma2_enhancer"],
                    res_all["sigma2_noise"]
                )
                se_bc_array[mask_all, -1] = blup_unc_all['se_barcode']
                se_enh_array[mask_all, -1] = blup_unc_all['se_enhancer']
                ci95_bc_array[mask_all, -1] = blup_unc_all['CI95_barcode']
                ci95_enh_array[mask_all, -1] = blup_unc_all['CI95_enhancer']

            except Exception as e:
                print(f"  Warning: BLUP computation failed for All_celltypes: {e}")

    results_df = pd.DataFrame(records)

    # FDR correction
    valid = ~results_df["p_value"].isna()
    results_df["p_adj"] = np.nan
    if valid.any():
        results_df.loc[valid, "p_adj"] = fdr_correction(
            results_df.loc[valid, "p_value"].values, alpha=fdr_alpha
        )

    # Build BLUP DataFrames
    # Build combined long-format BLUP table with proportions and uncertainties
    blup_columns = celltypes + ["All_celltypes"]
    blup_records = []
    for j, ct in enumerate(blup_columns):
        col_idx = j if j < n_celltypes else -1
        for i, cid in enumerate(construct_ids):
            g_bc = blup_bc_array[i, col_idx if col_idx != -1 else n_celltypes]
            g_enh = blup_enh_array[i, col_idx if col_idx != -1 else n_celltypes]
            g_noise = blup_noise_array[i, col_idx if col_idx != -1 else n_celltypes]
            se_bc = se_bc_array[i, col_idx if col_idx != -1 else n_celltypes]
            se_enh = se_enh_array[i, col_idx if col_idx != -1 else n_celltypes]
            ci95_bc = ci95_bc_array[i, col_idx if col_idx != -1 else n_celltypes]
            ci95_enh = ci95_enh_array[i, col_idx if col_idx != -1 else n_celltypes]

            if not (np.isfinite(g_bc) and np.isfinite(g_enh) and np.isfinite(g_noise)):
                continue
            abs_total = abs(g_bc) + abs(g_enh) + abs(g_noise)
            blup_records.append({
                "construct": cid,
                "cell_type": ct,
                "blup_barcode": g_bc,
                "blup_enhancer": g_enh,
                "blup_noise": g_noise,
                "se_barcode": se_bc,
                "se_enhancer": se_enh,
                "CI95_barcode": ci95_bc,
                "CI95_enhancer": ci95_enh,
                "prop_barcode": abs(g_bc) / abs_total if abs_total > 1e-15 else 1.0/3,
                "prop_enhancer": abs(g_enh) / abs_total if abs_total > 1e-15 else 1.0/3,
                "prop_noise": abs(g_noise) / abs_total if abs_total > 1e-15 else 1.0/3,
            })
    blup_df = pd.DataFrame(blup_records)

    # Summary
    print("\n" + "=" * 60)
    print("VARIANCE DECOMPOSITION SUMMARY")
    print("=" * 60)
    print(f"Total cell types: {len(results_df)}")

    # Valid = successfully fitted (not NaN)
    valid_df = results_df.dropna(subset=["prop_barcode"])
    print(f"Successfully fitted: {len(valid_df)}")
    print(f"Failed/Skipped: {len(results_df) - len(valid_df)}")

    # Diagnostic summaries
    print(f"\nPHASE 1: PRE-FLIGHT DIAGNOSTICS")
    print("-" * 60)

    # Kernel structure check
    if 'barcode_kernel_has_structure' in valid_df.columns:
        n_bc_no_struct = (~valid_df['barcode_kernel_has_structure']).sum()
        n_enh_no_struct = (~valid_df['enhancer_kernel_has_structure']).sum()
        if n_bc_no_struct > 0:
            print(f"  ⚠ {n_bc_no_struct} cell type(s): barcode kernel has no structure")
        if n_enh_no_struct > 0:
            print(f"  ⚠ {n_enh_no_struct} cell type(s): enhancer kernel has no structure")
        if n_bc_no_struct == 0 and n_enh_no_struct == 0:
            print(f"  ✓ All kernels have meaningful structure")

    # Kernel correlation summary
    if 'kernel_correlation' in results_df.columns:
        all_corr = results_df['kernel_correlation'].dropna()
        n_failed = (all_corr >= 0.8).sum()
        n_passed = (all_corr < 0.8).sum()

        print(f"\nKernel correlation (identifiability check, threshold=0.8):")
        if n_failed > 0:
            print(f"  ✗ FAILED: {n_failed} cell type(s) with r ≥ 0.8 (SKIPPED)")
            failed_cts = results_df[results_df['kernel_correlation'] >= 0.8]['cell_type'].tolist()
            for ct in failed_cts:
                corr_val = results_df[results_df['cell_type'] == ct]['kernel_correlation'].values[0]
                print(f"      {ct}: r={corr_val:.3f}")
        print(f"  ✓ PASSED: {n_passed} cell type(s) with r < 0.8")

        if n_passed > 0:
            passed_corr = all_corr[all_corr < 0.8]
            print(f"      Mean = {passed_corr.mean():.4f}")
            print(f"      Range = [{passed_corr.min():.4f}, {passed_corr.max():.4f}]")

    # Residual QQ correlation summary
    print(f"\nPHASE 2: POST-FIT DIAGNOSTICS")
    print("-" * 60)
    if 'residual_qq_correlation' in valid_df.columns and valid_df['residual_qq_correlation'].notna().any():
        qq_vals = valid_df['residual_qq_correlation'].dropna()
        print(f"Residual normality (QQ correlation):")
        print(f"  Mean = {qq_vals.mean():.4f}")
        print(f"  Range = [{qq_vals.min():.4f}, {qq_vals.max():.4f}]")
        n_poor = (qq_vals < 0.95).sum()
        n_mod = ((qq_vals >= 0.95) & (qq_vals < 0.98)).sum()
        if n_poor > 0:
            print(f"  ⚠ {n_poor} cell type(s) with poor normality (QQ r<0.95) - consider transformation")
        if n_mod > 0:
            print(f"  Note: {n_mod} cell type(s) with moderate non-normality (0.95≤QQ r<0.98)")
        if n_poor == 0 and n_mod == 0:
            print(f"  ✓ All residuals show good normality")

    print(f"\nVARIANCE COMPONENT ESTIMATES")
    print("-" * 60)

    print(f"\nProportion of variance explained by barcode:")
    print(f"  Mean  = {valid_df['prop_barcode'].mean():.4f}")
    print(f"  Median = {valid_df['prop_barcode'].median():.4f}")
    print(f"Proportion of variance explained by enhancer:")
    print(f"  Mean  = {valid_df['prop_enhancer'].mean():.4f}")
    print(f"  Median = {valid_df['prop_enhancer'].median():.4f}")
    print(f"Proportion noise:")
    print(f"  Mean  = {valid_df['prop_noise'].mean():.4f}")
    print(f"  Median = {valid_df['prop_noise'].median():.4f}")
    n_sig = (valid_df["p_adj"] < fdr_alpha).sum()
    print(f"\nCell types with significant barcode effect (FDR < {fdr_alpha}): "
          f"{n_sig}/{len(valid_df)}")
    print("=" * 60)

    # BLUP summary
    if len(blup_df) > 0:
        print("\nBLUP SUMMARY (per-construct scores)")
        print("-" * 40)
        print(f"Barcode artifact scores: "
              f"range [{blup_df['blup_barcode'].min():.4f}, "
              f"{blup_df['blup_barcode'].max():.4f}]")
        print(f"Enhancer activity scores: "
              f"range [{blup_df['blup_enhancer'].min():.4f}, "
              f"{blup_df['blup_enhancer'].max():.4f}]")
        print(f"Noise residual scores: "
              f"range [{blup_df['blup_noise'].min():.4f}, "
              f"{blup_df['blup_noise'].max():.4f}]")
        # Per-cell-type mean proportions
        ct_means = blup_df.groupby("cell_type")[["prop_barcode", "prop_enhancer", "prop_noise"]].mean()
        print(f"\nMean proportion per cell type:")
        for ct in ct_means.index:
            print(f"  {ct}: barcode={ct_means.loc[ct, 'prop_barcode']:.4f}, "
                  f"enhancer={ct_means.loc[ct, 'prop_enhancer']:.4f}, "
                  f"noise={ct_means.loc[ct, 'prop_noise']:.4f}")
    else:
        print("\nNo valid BLUPs computed.")
    print("=" * 60)

    return results_df, blup_df
