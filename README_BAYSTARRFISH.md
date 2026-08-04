# BAYSTARRFISH

A STARR-FISH in a bay. `BAY` carries the Bayes.

Bayesian hierarchical inference of cis-regulatory element (cCRE) activity from
single-cell [STARR-FISH](README.md) data — an imaging-based reporter assay that
measures the transcriptional capacity of enhancer reporter constructs in a
multiplexed, spatially resolved manner at single-molecule resolution.

---

## The problem

Each AAV carries a candidate cCRE driving a reporter plus a barcode. Per cell and
per barcode you observe two transcript counts — **T7** (constitutive, calibrates
infection) and **cCRE** (enhancer-driven). Neither is a DNA measurement.

The number of AAV genomes `k` that infected a cell for a given barcode is
**unobserved**, and infection is very rare: 99.2% of T7 entries and 99.85% of
cCRE entries are zero. A fold change over that background is dominated by which
cells happened to be infected, not by enhancer strength.

## The model

Treat `k` as a latent variable and integrate it out.

```
latent copies   k_ij      ~ Poisson(lambda_sj),  lambda_sj = rho_s * a_j
T7 channel      t7_ij  | k ~ NB2(mean = k * beta_t7,     disp = phi_t7)
cCRE channel    cre_ij | k ~ NB2(mean = k * gamma_sj,    disp = phi_cre)
```

with, for cell type `s` and cCRE `j`:

- `k = 0` forcing **both** channels to exactly zero — a point mass, not a
  negative binomial with mean zero;
- optional **zero-inflated measurement dropout** per channel, applied only where
  `k > 0` (an uninfected cell is already a zero, so attributing it to dropout
  would double-count);
- a two-level **class → subclass** hierarchy on the infection rate `rho` and,
  optionally, on the activity `gamma`;
- an informative **nanopore library prior** on the per-cCRE abundance `a`, mean-
  centred to fix the otherwise unidentifiable shared scale with `rho`.

`k` is marginalised analytically by `logsumexp` over a truncated grid `0..kmax`,
so the target is a smooth continuous-parameter model that NumPyro fits by SVI
(production) or NUTS (calibration).

**Scalability.** The 408,621 × 400 observation array collapses to weighted unique
`(group, cre, t7, cre)` rows: the all-zero pairs share an identical marginal
within a cell type, so they become a single weighted row. The likelihood enters
as one `numpyro.factor`, not a plate.

## Install

```bash
pip install -e .            # core
pip install -e '.[gpu]'     # + CUDA 12 wheels for fitting
pip install -e '.[dev]'     # + pytest, statsmodels (test references)
pip install -e '.[repro]'   # exact pins of the published fits
```

`envs/baystarrfish-gpu.yml` and `envs/baystarrfish-gpu.lock.txt` reproduce the
environment that produced every published fit: python 3.11.15, jax 0.10.1 with
CUDA 12.9, numpyro 0.21.0.

Double precision is mandatory — the marginal mixes `~1e-6` rates with `-inf`
point masses inside a `logsumexp`, and in float32 the gradients go `nan`.
Importing `baystarrfish` sets `JAX_ENABLE_X64=1` before JAX is imported at all.

## Use

```python
import baystarrfish as bsf

data = bsf.data.CountData.from_h5ad(section="all", negative_control_mode="ordinary")

fit = bsf.fit(
    **data.to_run_kwargs(),
    level="subclass",
    channel="joint",
    infection_model="copy_number_dropout",
    activity_model="direct",
    num_steps=30_000,
    num_posterior=1_000,
)

bsf.io.write_fit(fit, "results/bayesian", tag="subclass_joint_copy_number_dropout_svi",
                 data=data)
```

Calling activity, from the posterior draws:

```python
from baystarrfish.io import load_posterior_samples
from baystarrfish.stats import negative_control_test

post = load_posterior_samples("results/bayesian", sites=["log_gamma"])
calls = negative_control_test(post["log_gamma"], ..., t7_threshold=50.0,
                              effect_threshold=0.0, method="joint+dropout")
```

`negative_control_test` contrasts each target against the mean of the negative
controls **inside each posterior draw**, so the reference's own uncertainty is
subtracted rather than treated as a fixed offset, then reports `p_right` (the
posterior tail probability) and its BH-adjusted `q_right`.

## Recovering the copy number: how many AAV genomes reached each cell

`k_ij` is what the model is built around and the one thing it never samples — it
is marginalised so inference stays in continuous parameters. It is recoverable
exactly afterwards, because the grid the fit summed over renormalises into a
distribution:

```python
from baystarrfish.data import CountData
from baystarrfish.inference import infer_copy_number_from_fit

data = CountData.from_h5ad(section="all", negative_control_mode="ordinary")
copies = infer_copy_number_from_fit(data, "results/bayesian_full_posterior", return_sd=True)

copies.copies            # (n_cells, n_cre) float32: E[k_ij | t7_ij, cre_ij]
copies.sd                # same shape: posterior sd
copies.total_per_cell()  # expected genomes per cell, summed over cCREs
copies.to_frame()        # labelled DataFrame (dense — 1.3 GB at full size)
copies.write_npz("results/copies.npz")
```

or from the command line, which is what you want for the full matrix:

```bash
python -m baystarrfish copy-number \
    --fit-dir results/ablation/bayesian_full_posterior \
    --out results/copy_number.npz --with-sd
```

`kmax`, the cell-type granularity and the infection model are read from the fit's
manifest, so the reconstruction cannot silently disagree with the posterior it
came from.

**This is not the counts.** A cell reading `t7 = 0, cre = 0` is not known to be
uninfected — under dropout, or simply a low per-copy rate, an infected cell often
reads zero, so those pairs get a nonzero baseline that depends on the cell type's
infection rate and that cCRE's library abundance. A cell reading `t7 = 12` did
not receive twelve genomes. And `copies` is a posterior **mean over an integer**:
`0.02` means "almost certainly zero copies", not a fiftieth of a virus.
Thresholding it is not an infection call — `P(k >= 1)` is a different quantity.

**It needs `log_rho` and `log_a`, which the production fit did not save.** The
copy number depends on infection and abundance, not just activity, and those
sites cannot be reconstructed after the fact. `results/bayesian/` stores only
`log_gamma`; `results/ablation/bayesian_full_posterior/` has everything. To make
a dropout fit usable here, refit asking for them — three sites, not `all`, which
would also write the 500 MB `alpha`/`delta`/`eta` blocks:

```bash
python revision/bayesian_vs_fold_change/code/run_bayes.py \
    --posterior-sites log_gamma log_rho log_a   # ...otherwise unchanged
```

**Sanity-check the scale before trusting absolute copy numbers.** `k` and
`beta_t7` are only weakly separated by the likelihood — "many copies, tiny
per-copy rate" fits about as well as "few copies, large rate", and the
mean-centering of `log a` fixes the `rho`–`a` scale but not this one. On the
`bayesian_full_posterior` ablation the fit sits at `beta_t7 = 0.03`, i.e. one
genome yields 0.03 T7 transcripts, which lets `rho * a` reach 39 while the data
stays 99.2% zeros; `E[k]` there is ~5 copies per cell with no counts to show for
it. Ratios and rankings across cells are far more robust than the absolute level.
`infer_copy_number` warns when `rho * a` approaches `kmax`, where the truncation
additionally biases `E[k]` low.

**Cost.** `P(k | obs)` depends on the observation only through
`(cell type, cCRE, t7, cre)`, so the 99.85% all-zero pairs collapse to one
baseline per (cell type, cCRE) and the rest to ~10⁶ unique patterns instead of
1.6 × 10⁸ evaluations. The result is exact, not sampled or approximated.

Even so, the all-zero baseline alone is 328 × 389 = 128k patterns, and that cost
is paid whatever the cell count — the work scales with the *posterior*, not the
data. Runtime is linear in the number of draws, so `--max-draws 200` is the main
lever; a posterior **mean** over 200 evenly-spaced draws is within Monte Carlo
error of one over 1,000. Run it on a compute node, not a login node.

## Layout

```
baystarrfish/
  model/       priors, collapsed statistics, likelihood, sampling blocks,
               the twelve model functions, the registry, the forward sampler
  inference/   moment initialisation, SVI/NUTS, summaries, posterior predictive
               checks, the run_model / run_decoupled_model drivers, and the
               latent copy-number reconstruction (posterior_k, copy_number)
  data/        AnnData -> arrays: paths, label standardisation, blacklisting,
               negative-control modes, nanopore library prior, CountData
  stats/       BH-FDR and the negative-control contrast
  io/          write_fit / read_fit / load_gamma / load_posterior_samples
  simulate/    simulation-recovery test
```

`model/` and `inference/` need NumPyro. `data/`, `stats/` and `io/` deliberately
do not, so the plotting environment can reload inputs and read fits without the
inference stack.

Three infection families, selected by `infection_model`: `copy_number`,
`copy_number_dropout` (production), and `binary` — a shared Bernoulli infection
gate with probability `1 - exp(-rho_s a_j)` instead of a copy count. Two activity
parameterisations, selected by `activity_model`: `hierarchical`
(`log_gamma = alpha_j + eta_cj + delta_sj`) and `direct` (exchangeable over the
whole subclass × cCRE matrix, used for the production fit).

## Priors

| site | prior | meaning |
|---|---|---|
| `mu_rho` | `Normal(-6, 2)` | global log infection baseline |
| `sigma_u`, `sigma_w` | `HalfNormal(1)` | class- and subclass-level infection sd |
| `tau_a` | `HalfNormal(0.5)` | slack around the centred log nanopore abundance |
| `beta_t7` | `LogNormal(0, 1)` | T7 counts per infected copy |
| `phi_t7`, `phi_cre` | `HalfNormal(5)` | NB2 dispersions |
| `mu_alpha` | `Normal(0, 3)` | activity baseline (hierarchical) |
| `sigma_alpha` | `HalfNormal(2)` | per-cCRE baseline sd |
| `sigma_eta`, `sigma_delta` | `HalfNormal(1)` | class- and subclass-level activity sd |
| `mu_gamma`, `sigma_gamma` | `Normal(0, 3)`, `HalfNormal(2)` | activity (direct) |
| `p_drop_t7`, `p_drop_cre` | `Beta(1, 9)` | measurement dropout |

All non-centred (`*_raw ~ Normal(0, 1)`, scaled by `sigma_*`). Sensitivity to the
dropout prior is reported in
[`revision/bayesian_vs_fold_change/Bayesian_priors.md`](revision/bayesian_vs_fold_change/Bayesian_priors.md).

## Reading the output

`<tag>_gamma.csv` carries a **`prior_dominated`** flag marking pairs with no
double-positive (T7 > 0 **and** cCRE > 0) support. Those posteriors are the
prior, not evidence; they are not activity estimates and must not be read as
such. The same eligibility logic drives the T7 filter in
`negative_control_test`.

## Tests

```bash
pytest tests -m 'not slow'    # 127 tests, ~55 s
pytest tests -m slow          # end-to-end SVI recovery, ~60 s
python -m baystarrfish recovery --classes 3 --cres 5 --cells 400 --steps 4000
```

The suite checks the model against outside references rather than against itself:
the negative binomial against `scipy.stats.nbinom`, the marginal against an
explicit sum over the latent grid, BH-FDR against `statsmodels`, and the NumPy
copy-number posterior against the JAX likelihood.

`testpaths = ["tests"]` in `pyproject.toml` is load-bearing:
`revision/bayesian_vs_fold_change/code/` contains six files named `test_*.py`
that are analysis entry points launching multi-hour jobs, not tests.

## Citation

Gibbs, Z., Zhong, G. et al. (2026). STARRFISH [under submission].
Data: <https://doi.org/10.5061/dryad.3bk3j9m0p>
