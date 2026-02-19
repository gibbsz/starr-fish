# Barcode vs Enhancer Activity Decomposition

## Problem Statement

We have 400 AAV constructs, each containing a unique barcode sequence and a unique enhancer sequence. Activity of each construct is measured across 300 cell types. Because each barcode is perfectly paired with one enhancer, their effects are confounded. We want to determine whether barcode sequences contribute significantly to observed activity variation.

**Key insight**: While barcode and enhancer effects are statistically confounded in the experimental design, we can use **sequence similarity** as a proxy for functional similarity to break the confounding. If two AAVs have similar barcodes but dissimilar enhancers, shared activity patterns are likely barcode-driven.

## Input Data

The code should accept the following inputs (provide flexible loading — CSV, TSV, or AnnData/H5AD):

1. **Activity matrix**: shape `(400, 300)` — rows are AAV constructs, columns are cell types. Values are quantitative activity measurements (e.g., log-normalized expression or accessibility).
2. **Barcode sequences**: a list/series of 400 DNA barcode sequences, indexed to match the activity matrix rows.
3. **Enhancer sequences**: a list/series of 400 DNA enhancer sequences, indexed to match the activity matrix rows.

For development/testing, generate **synthetic data** with known ground truth (see Synthetic Data section below).

## Approach 1: Partial Mantel Test

### Goal

Test whether pairwise similarity in activity profiles is associated with pairwise barcode sequence similarity, after controlling for enhancer sequence similarity (and vice versa).

### Steps

1. **Compute pairwise distance matrices** (all 400 × 400):
   - `D_activity`: Correlation distance (1 - Pearson r) across the 300-dimensional activity vectors.
   - `D_barcode`: Sequence distance between barcodes. Use **Levenshtein (edit) distance**, normalized by the longer sequence length. Also implement an option for **k-mer distance** (e.g., Jaccard distance on the set of k-mers, with k=4 or k=6).
   - `D_enhancer`: Same as above but for enhancer sequences.

2. **Run partial Mantel tests** using permutation (N=9999 permutations):
   - Test 1: `corr(D_activity, D_barcode | D_enhancer)` — is barcode distance associated with activity distance after partialing out enhancer distance?
   - Test 2: `corr(D_activity, D_enhancer | D_barcode)` — is enhancer distance associated with activity distance after partialing out barcode distance?

3. **Implementation details**:
   - Use the Pearson correlation variant of the Mantel test.
   - For the partial Mantel, regress both `D_activity` and `D_barcode` on `D_enhancer` (using the flattened upper triangle of each matrix), then correlate the residuals. Permutation is done by permuting rows/columns of one distance matrix simultaneously.
   - Use `skbio.stats.distance.mantel` if available, or implement from scratch using `scipy` + `numpy`.

4. **Output**:
   - Mantel r statistic and p-value for both tests.
   - A summary print statement interpreting which factor (barcode vs enhancer) shows significant association.
   - Scatter plots of flattened distance matrix pairs (D_activity vs D_barcode, D_activity vs D_enhancer) with regression lines.

## Approach 2: Kernel Variance Decomposition (Linear Mixed Model)

### Goal

Decompose the variance in activity for each cell type into components attributable to barcode sequence similarity vs. enhancer sequence similarity, using kernel/relatedness matrices analogous to GCTA/GREML in genetics.

### Steps

1. **Construct kernel (similarity) matrices** (400 × 400):
   - `K_barcode`: Barcode sequence similarity kernel. Options to implement:
     - **Normalized k-mer kernel**: For each sequence, compute k-mer frequency vector (k=4 to k=6), then K_ij = dot(v_i, v_j) / (||v_i|| * ||v_j||). Normalize so diagonal is 1.
     - **Edit distance RBF kernel**: K_ij = exp(-d_ij^2 / (2 * sigma^2)) where d_ij is normalized Levenshtein distance. Use median distance as sigma (median heuristic).
   - `K_enhancer`: Same kernels but computed on enhancer sequences.
   - Both kernels should be centered: K_centered = H @ K @ H where H = I - 11^T/n.

2. **Fit linear mixed model per cell type**:
   For each of the 300 cell types, the model is:

   ```
   y_j = mu * 1 + g_barcode + g_enhancer + epsilon
   ```

   where:
   - `y_j` is the 400-vector of activity values in cell type j
   - `g_barcode ~ N(0, sigma2_barcode * K_barcode)`
   - `g_enhancer ~ N(0, sigma2_enhancer * K_enhancer)`
   - `epsilon ~ N(0, sigma2_noise * I)`

   Estimate `sigma2_barcode`, `sigma2_enhancer`, `sigma2_noise` via **REML** (restricted maximum likelihood).

3. **Significance testing**:
   - For each cell type, test H0: `sigma2_barcode = 0` using a likelihood ratio test (LRT) comparing the full model to a reduced model without the barcode kernel.
   - The LRT statistic follows a 0.5 * chi2(0) + 0.5 * chi2(1) mixture distribution under H0 (because the parameter is on the boundary). Adjust p-values accordingly, or use permutation.
   - Apply FDR correction (Benjamini-Hochberg) across the 300 cell types.

4. **Implementation guidance**:
   - Use `numpy` for kernel construction and matrix operations.
   - For the LMM fitting, options (in order of preference):
     - `glimix_core` or `limix` (purpose-built for this)
     - `statsmodels.regression.mixed_linear_model` (may need to pass custom covariance)
     - Manual REML optimization using `scipy.optimize.minimize` on the restricted log-likelihood. The log-likelihood for the multi-kernel model with parameters theta = (sigma2_barcode, sigma2_enhancer, sigma2_noise) is:
       ```
       V = sigma2_barcode * K_barcode + sigma2_enhancer * K_enhancer + sigma2_noise * I
       log L_REML = -0.5 * [(n-p)*log(2*pi) + log|V| + log|X^T V^-1 X| + y^T P y]
       ```
       where P = V^-1 - V^-1 X (X^T V^-1 X)^-1 X^T V^-1, and X = column of ones (intercept).
     - Constrain variance components >= 0 during optimization (use L-BFGS-B with bounds).
   - If `limix`/`glimix_core` are not easily installable, implement the manual REML approach. It's ~50 lines of core code.

5. **Output**:
   - A dataframe with columns: `cell_type`, `sigma2_barcode`, `sigma2_enhancer`, `sigma2_noise`, `prop_barcode` (= sigma2_barcode / total), `prop_enhancer`, `LRT_stat`, `p_value`, `p_adj` (FDR).
   - Summary statistics: median and mean proportion of variance explained by barcodes vs enhancers across cell types.
   - Histogram of `prop_barcode` across 300 cell types.
   - Volcano-style plot: prop_barcode (x-axis) vs -log10(p_adj) (y-axis).
   - Stacked bar chart or violin plot comparing variance component proportions.

## Synthetic Data Generation

Create a function `generate_synthetic_data()` that produces test data with known ground truth to validate both approaches.

### Parameters

- `n_constructs = 400`
- `n_celltypes = 300`
- `barcode_len = 20` (bp)
- `enhancer_len = 200` (bp)
- `sigma2_barcode_true = 0.1` (barcode variance component, keep small)
- `sigma2_enhancer_true = 1.0` (enhancer variance component, dominant)
- `sigma2_noise_true = 0.2`
- `n_motifs_barcode = 3` (number of latent motifs driving barcode activity)
- `n_motifs_enhancer = 10` (number of latent motifs driving enhancer activity)

### Generation procedure

1. Generate random DNA sequences for barcodes and enhancers.
2. For barcodes: embed a small number of random short motifs (4-6bp) into a random subset of sequences. Barcode activity = weighted count of motif occurrences (cell-type invariant, i.e., same weight vector across all cell types, maybe with small cell-type noise).
3. For enhancers: embed longer motifs (6-10bp). Enhancer activity = motif counts × cell-type-specific weight matrix (300 weight vectors, one per cell type, drawn from a low-rank structure to simulate TF expression patterns).
4. Total activity = barcode_activity + enhancer_activity + noise.
5. Return the activity matrix, sequences, and the true variance components for validation.

## Code Structure

```
barcode_enhancer_decomposition/
├── README.md
├── requirements.txt
├── config.py                  # Default parameters, paths
├── data/
│   └── synthetic/             # Generated test data saved here
├── src/
│   ├── __init__.py
│   ├── data_io.py             # Load real data (CSV/TSV/H5AD), save results
│   ├── synthetic.py           # generate_synthetic_data()
│   ├── sequence_kernels.py    # Levenshtein, k-mer distance/kernel, RBF kernel
│   ├── mantel.py              # Partial Mantel test implementation
│   ├── variance_decomp.py     # Kernel LMM, REML, LRT
│   └── plotting.py            # All visualization functions
├── scripts/
│   ├── run_synthetic.py       # End-to-end on synthetic data
│   ├── run_mantel.py          # Run partial Mantel on real/synthetic data
│   └── run_variance_decomp.py # Run kernel variance decomposition
└── results/                   # Output figures and tables
```

## Requirements

```
numpy
scipy
pandas
matplotlib
seaborn
scikit-bio          # optional, for Mantel test
Levenshtein         # python-Levenshtein, for fast edit distance
statsmodels         # optional
tqdm                # progress bars for 300 cell-type loop
```

## Execution Order

1. Run `scripts/run_synthetic.py` first to generate synthetic data and validate both methods recover known ground truth.
2. Then adapt `scripts/run_mantel.py` and `scripts/run_variance_decomp.py` for real data by pointing to appropriate input files.

## Expected Validation on Synthetic Data

- The partial Mantel test should show a significant (p < 0.05) association between D_activity and D_enhancer after partialing out D_barcode, and a weaker (but potentially still significant, depending on sigma2_barcode_true) association for barcode.
- The variance decomposition should recover approximately the true variance proportions: ~77% enhancer (1.0/1.3), ~8% barcode (0.1/1.3), ~15% noise (0.2/1.3) with the default parameters.

## Notes for Implementation

- Sequence kernel computation on 400 sequences is fast. For k-mer kernel, use vectorized counting (e.g., `collections.Counter` or even a simple sliding window).
- The REML optimization is the trickiest part. Key numerical considerations:
  - Use Cholesky decomposition for `V` inversion (more stable than direct inverse).
  - If `V` becomes near-singular, add a small jitter (1e-6 * I).
  - Initialize optimization at reasonable starting values (e.g., sample variance / 3 for each component).
  - Log-transform variance parameters during optimization to enforce positivity, or use bounded optimization.
- For the 300 cell-type loop in variance decomposition, the kernels only need to be computed once. Cache them.
- Consider adding a **combined summary** that runs both approaches and presents results side by side for easy comparison.
