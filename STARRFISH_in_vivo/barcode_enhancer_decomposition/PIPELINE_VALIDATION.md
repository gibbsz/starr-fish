# Variance Decomposition Pipeline with Validation

## Overview

This pipeline implements a rigorous 3-phase approach to decompose variance in STARR-FISH data into biological signal (enhancer activity) and technical artifacts (barcode effects), with comprehensive validation at each step.

## Complete Pipeline Structure

### Phase 1: Pre-Flight Safety Checks
**Goal**: Ensure the data meets mathematical assumptions before running the complex model.

#### 1.1 Check Kernel Structure (Eigenvalue Spectrum Flatness)
**Purpose**: Verify that kernels contain meaningful structure, not just random noise.

**Method**: Coefficient of Variation (CV) of eigenvalues
```
CV = std(eigenvalues) / mean(eigenvalues)
```

**Interpretation**:
- If K ≈ I (identity matrix), all eigenvalues ≈ 1 → std ≈ 0 → **CV ≈ 0** (flat)
- If K has structure (clusters/gradients), eigenvalues vary → **CV > 0.1** (structured)

**Threshold**:
- **CV ≤ 0.1** (10%): **FLAT** - Matrix is effectively Identity
- **CV > 0.1**: **STRUCTURED** - Clusters/gradients exist in sequence space

**Risk**: If kernel has no structure (CV ≤ 0.1), it provides no information beyond random noise. Using it wastes computation and produces unstable estimates.

**Action on failure** (CV ≤ 0.1):
- **WARNING** printed (but pipeline continues)
- Recommendation: Use standard ridge regression (g ~ N(0, σ² I)) instead
  - Much faster (no kernel matrix operations)
  - More numerically stable
  - Produces identical results when K ≈ I

#### 1.2 Check Identifiability (Kernel Correlation)
**Purpose**: Ensure barcode and enhancer kernels are distinct enough to be mathematically separable.

**Method**:
```python
def kernel_correlation(K1, K2):
    # Extract off-diagonal elements
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    k1_offdiag = K1[mask]
    k2_offdiag = K2[mask]
    return np.corrcoef(k1_offdiag, k2_offdiag)[0, 1]
```

**Thresholds**:
- **r < 0.5**: Excellent - kernels are highly distinct
- **0.5 ≤ r < 0.8**: Good - kernels are sufficiently different
- **r ≥ 0.8**: **FAILED** - Collinearity problem (HARD CUTOFF)

**Risk**: High correlation (r ≥ 0.8) means the model cannot distinguish between "barcode effect" and "enhancer effect." Variance estimates will bounce wildly between the two.

**Action on failure**:
- **If r ≥ 0.8**: **STOP** - Skip REML fitting for this cell type
- Mark all results as NaN
- Print failure message
- User must:
  1. Use different kernel parameters (e.g., different k-mer sizes)
  2. Switch to motif-based kernels instead of sequence-based
  3. Or accept that this cell type cannot be analyzed with current kernels

### Phase 2: Core Model Fitting & Testing

#### 2.1 Fit Linear Mixed Models (LMM) via REML
**Models**:
- **Full**: `y = μ + g_barcode + g_enhancer + ε`
- **Reduced**: `y = μ + g_enhancer + ε` (testing if barcode adds value)

**Method**: Restricted Maximum Likelihood (REML) using L-BFGS-B optimization

**Output**: Variance components σ²_barcode, σ²_enhancer, σ²_noise

#### 2.2 Significance Testing (Likelihood Ratio Test)
**Test statistic**: `LRT = 2 × (LogLik_full - LogLik_reduced)`

**P-value**: 50:50 mixture of χ²(0) and χ²(1) (boundary case)

**Multiple testing correction**: Benjamini-Hochberg FDR across cell types

**Threshold**: FDR < 0.05 (default, configurable)

#### 2.3 Residual Diagnostics (Post-Fit Safety)
**Purpose**: Verify that model assumptions (normality of residuals) hold.

**Method**: QQ plot correlation
```python
def qq_correlation(residuals):
    res_std = (residuals - mean) / std
    res_sorted = np.sort(res_std)
    theoretical_quantiles = norm.ppf(np.arange(1, n+1) / (n+1))
    return np.corrcoef(theoretical_quantiles, res_sorted)[0, 1]
```

**Thresholds**:
- **r_QQ > 0.98**: Excellent normality
- **0.95 ≤ r_QQ ≤ 0.98**: Acceptable normality
- **r_QQ < 0.95**: Poor normality

**Risk**: Non-normal residuals suggest the linear model is misspecified or the phenotype needs transformation.

**Action on failure**: If r_QQ < 0.95, consider:
1. Log-transform the phenotype: `y → log(y + 1)`
2. Rank-based inverse normal transformation (rank-INT)
3. Re-fit the model with transformed data

### Phase 3: Decomposition & Uncertainty

#### 3.1 Compute BLUPs (Best Linear Unbiased Predictions)
**Formula**:
```
g_barcode  = σ²_b × K_b × V⁻¹ × (y - μ)
g_enhancer = σ²_e × K_e × V⁻¹ × (y - μ)
g_noise    = y - μ - g_barcode - g_enhancer
```

**Output**: Three scores per construct per cell type
- **g_barcode**: Artifact score (technical bias)
- **g_enhancer**: Biological activity score
- **g_noise**: Unexplained variance

#### 3.2 BLUP Uncertainty (Prediction Error Variance)
**Formula**: `PEV(g) = G - G × P × G`

where `P = V⁻¹ - V⁻¹X(X'V⁻¹X)⁻¹X'V⁻¹`

**Output**: Standard errors and 95% CI for each BLUP

**Interpretation**:
- Constructs with similar sequences (high kernel connectivity) → smaller SE
- Unique/isolated constructs → larger SE

#### 3.3 Bootstrap Confidence Intervals (Parametric)
**Method**: Simulate from fitted model, refit, calculate distribution

**Implementation**: Parallelized with joblib (10-20x speedup)

**Output**: 95% CI for variance proportions (prop_barcode, prop_enhancer, prop_noise)

## Output Files

### `variance_decomp_results.csv`
One row per cell type with columns:

**Identifiers**:
- `cell_type`: Cell type name
- `n_valid`: Number of valid constructs

**Phase 1 Diagnostics**:
- `barcode_kernel_has_structure`: Boolean, True if CV > 0.1
- `barcode_eigenvalue_cv`: Coefficient of variation of eigenvalues (threshold=0.1)
- `enhancer_kernel_has_structure`: Boolean, True if CV > 0.1
- `enhancer_eigenvalue_cv`: Coefficient of variation of eigenvalues (threshold=0.1)
- `kernel_correlation`: Pearson r between kernels (threshold=0.8)

**Phase 2 Diagnostics**:
- `residual_qq_correlation`: QQ plot r for normality check (threshold=0.95)

**Variance Components**:
- `sigma2_barcode`, `sigma2_enhancer`, `sigma2_noise`: Raw variance estimates
- `prop_barcode`, `prop_enhancer`, `prop_noise`: Normalized proportions (sum to 1)

**Confidence Intervals**:
- `prop_*_CI_low`, `prop_*_CI_high`: 95% CI bounds
- `prop_*_CI95`: CI half-width (for ± notation)

**Statistical Tests**:
- `LRT_stat`: Likelihood ratio test statistic
- `p_value`: Raw p-value
- `p_adj`: FDR-adjusted p-value

### `blup.csv`
Long-format table with one row per construct per cell type:

**Identifiers**:
- `construct`: Construct ID
- `cell_type`: Cell type name

**BLUP Scores**:
- `blup_barcode`: Artifact score
- `blup_enhancer`: Biological activity score
- `blup_noise`: Residual noise

**Proportions** (based on absolute values):
- `prop_barcode = |g_bc| / (|g_bc| + |g_enh| + |g_noise|)`
- `prop_enhancer`, `prop_noise`: Similarly computed

**Uncertainty**:
- `se_barcode`, `se_enhancer`: Standard errors
- `CI95_barcode`, `CI95_enhancer`: 95% CI half-widths

## Interpretation Guide

### When to Trust Results

**✓ Good scenario** (all checks pass):
- Both kernels have structure
- Kernel correlation < 0.8
- Residual QQ correlation > 0.95
- → Variance estimates are reliable

**⚠ Moderate concern** (some warnings):
- Kernel correlation 0.8-0.9
- OR Residual QQ 0.95-0.98
- → Results are acceptable but interpret with caution

**✗ Poor scenario** (critical failures):
- Kernel has no structure
- Kernel correlation > 0.9
- OR Residual QQ < 0.95
- → Results are unreliable, transformation or different kernel needed

### Example Workflow

```python
# 1. Run pipeline
results_df, blup_df = run_variance_decomposition(
    activity, K_barcode, K_enhancer,
    n_boot=100, n_jobs=-1
)

# 2. Check diagnostics
print(results_df[['cell_type', 'kernel_correlation',
                   'residual_qq_correlation',
                   'barcode_kernel_has_structure']])

# 3. Filter to reliable cell types
reliable = results_df[
    (results_df['kernel_correlation'] < 0.8) &
    (results_df['residual_qq_correlation'] > 0.95) &
    (results_df['barcode_kernel_has_structure']) &
    (results_df['enhancer_kernel_has_structure'])
]

# 4. Examine variance proportions with CIs
print(reliable[['cell_type', 'prop_enhancer', 'prop_enhancer_CI95']])
```

## Command Line Usage

```bash
# Run full pipeline with all diagnostics
python scripts/run_variance_decomp.py \
    --n-bootstrap 100 \
    --n-jobs -1 \
    --outdir results/variance_decomp

# Results will include:
# - variance_decomp_results.csv (with all diagnostics)
# - blup.csv (with uncertainties)
# - Diagnostic plots
```

## Performance Notes

With parallelization (n_jobs=-1):
- Kernel structure check: ~0.1s per cell type
- Kernel correlation: ~0.01s per cell type
- REML fitting: ~2-5s per cell type
- Residual diagnostics: ~0.1s per cell type
- Bootstrap (100 iterations): ~3-6s per cell type (with all cores)
- BLUP uncertainty (PEV): ~0.5s per cell type

**Total**: ~10-15 seconds per cell type with all diagnostics and 100 bootstrap iterations

## References

1. **Kernel structure**: Effective rank and spectral analysis (Shawe-Taylor & Cristianini, 2004)
2. **Collinearity**: VIF and correlation diagnostics (Belsley et al., 1980)
3. **REML**: Patterson & Thompson (1971), Restricted Maximum Likelihood
4. **Boundary LRT**: Self & Liang (1987), χ² mixture distribution
5. **BLUP**: Henderson (1975), Best Linear Unbiased Prediction
6. **PEV**: Henderson (1975), Prediction error variance
7. **Bootstrap**: Efron & Tibshirani (1993), Parametric bootstrap
8. **QQ diagnostics**: Wilk & Gnanadesikan (1968), QQ plots for normality
