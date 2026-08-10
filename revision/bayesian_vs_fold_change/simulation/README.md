# Joint-dropout simulation and recovery test

This folder generates simulated STARR-FISH count data from the fitted
subclass-level Bayesian joint model with zero-inflated dropout, checks that the
simulated data resemble the real data, then fits the model back to the simulated
data and compares estimates with the known simulation truth.

The default truth source is:

`revision/bayesian_vs_fold_change/results/ablation/bayesian_joint_dropout/`

The current production fit saved `log_gamma` draws and scalar draws, but not
`log_rho` or `log_a` draws. When those infection posterior draws are absent,
`simulate_joint_dropout.py` reconstructs infection rates as
`rho_mean * exp(centered log1p(nanopore library size))` and records that fallback
in `truth_manifest.json`. If the fit is rerun with
`--posterior-sites log_gamma log_rho log_a`, the simulator will use the direct
posterior mean of `exp(log_rho + log_a)` instead.

## Outputs

By default all generated files are written under:

`revision/bayesian_vs_fold_change/simulation/results/`

- `joint_dropout_simulated/simulated_joint_dropout.h5ad`: simulated count object
  with the same cells, obs annotations, and canonical cCRE columns as the real
  input. Blacklisted cCRE columns are retained as zeros so the standard fitting
  scripts apply their usual blacklist and recover the same fitted cCRE set.
- `joint_dropout_simulated/truth_parameters.npz`: simulation truth arrays.
- `joint_dropout_simulated/truth_gamma.csv`, `truth_infection.csv`,
  `truth_rho.csv`, `truth_scalars.csv`: tidy truth tables.
- `joint_dropout_stats/`: real-vs-simulated summary tables and diagnostic plots.
- `joint_dropout_fit/`: Bayesian fit on the simulated H5AD.
- `joint_dropout_recovery/`: estimation-vs-truth tables, metrics, and plots.

## Local smoke test

Use a small cell/cCRE subset before submitting production jobs:

```bash
/gpfs/commons/home/guojiezhong/miniconda3/envs/bayes-jax/bin/python \
  revision/bayesian_vs_fold_change/simulation/simulate_joint_dropout.py \
  --max-cells 1000 \
  --max-cres 30 \
  --outdir revision/bayesian_vs_fold_change/simulation/results/smoke_simulated

/gpfs/commons/home/guojiezhong/miniconda3/envs/bayes-jax/bin/python \
  revision/bayesian_vs_fold_change/simulation/compare_simulated_to_real.py \
  --sim-h5ad revision/bayesian_vs_fold_change/simulation/results/smoke_simulated/simulated_joint_dropout.h5ad \
  --max-cells 1000 \
  --max-cres 30 \
  --outdir revision/bayesian_vs_fold_change/simulation/results/smoke_stats
```

## Production pipeline

Submit the CPU simulation and real/simulated statistic comparison:

```bash
sbatch revision/bayesian_vs_fold_change/simulation/submit_simulate_and_compare.slurm
```

Fit the joint-dropout Bayesian model to the simulated data:

```bash
sbatch revision/bayesian_vs_fold_change/simulation/submit_fit_joint_dropout.slurm
```

Evaluate parameter recovery:

```bash
sbatch revision/bayesian_vs_fold_change/simulation/submit_evaluate_recovery.slurm
```

Or submit all three with Slurm dependencies:

```bash
bash revision/bayesian_vs_fold_change/simulation/submit_pipeline.sh
```

Optional fold-change/bootstrap comparison can be run against the simulated H5AD
with the existing bootstrap script:

```bash
sbatch revision/run_Bootstrap/submit_bootstrap.slurm \
  --h5ad revision/bayesian_vs_fold_change/simulation/results/joint_dropout_simulated/simulated_joint_dropout.h5ad \
  --outdir revision/bayesian_vs_fold_change/simulation/results/bootstrap_on_simulated
```
