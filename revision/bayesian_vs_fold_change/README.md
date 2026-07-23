# Bootstrap versus Bayesian activity analysis

This directory analyzes the full 5/28 BRBB500gn dataset:

`revision/Data/scdata_5_28_2025_BRBB500gn_final_CRE_T7CRE_NEW.h5ad`

The old sec1/sec2 results were removed. All new generated files live below
`results/`.

## Methods

The input contains 408,621 cells, 500 RNA features, and 400 cCREs with
`CRE_info` metadata. The `CRE` and `T7CRE` matrices contain 20 additional
unmapped columns; the workflow deliberately keeps only the 400 canonical
`CRE001`–`CRE400` columns backed by `CRE_info`.

Both analyses use cleaned Allen class/subclass names, the manuscript cCRE
blacklist (CRE001, CRE061, CRE143, plus barcode mismatch >20%), and the ten
`CRE_info` negative controls.

1. `code/run_bootstrap.py` runs the manuscript Fig. 4 average bootstrap:
   10,000 within-subclass resamples, subclass-level T7 normalization,
   self-cCRE centering, testing against the negative-control mean, a
   five-positive-cell filter, and BH-adjusted right/left/two-sided q-values.
2. `code/run_bayes.py` fits the production joint+dropout model at subclass
   resolution with direct activity estimation, SVI (30,000 steps), and 1,000
   posterior draws. The seven annotated negative controls are retained as
   ordinary cCREs; downstream tests compare each target with their draw-wise
   mean activity. This run is stored in `results/bayesian/`.
3. `code/run_bayes_decoupled.py` fits an ablation two-stage Bayesian model.
   Stage 1 uses
   T7 counts only to estimate subclass/cCRE infection rates with one global T7
   dropout rate. Stage 2 conditions on the T7 infection posterior to infer cCRE
   activity with a separate global cCRE dropout rate; T7 and cCRE no longer
   share a cell-level latent copy number.
4. `code/run_bayes_bootstrap_metacells.py` fits a meta-cell ablation after
   replacing cells with fixed-size meta-cells. Each meta-cell is the sum of a
   with-replacement sample drawn within one subclass only; `--bootstrap-size`
   controls cells per meta-cell and defaults to 100.
5. `code/plot_results.py` puts both methods on the same natural-log effect
   scale, writes merged/per-subclass tables, and generates activity heatmaps,
   an activity-concordance hexbin, significant-call overlap, per-subclass
   concordance, and per-cCRE significant-call plots.

## Section reproducibility

The H5AD contains both physical sections. Cell IDs encode the exact split used
by the legacy section objects:

- section 1: `Conv_zscan2_*` (187,816 cells)
- section 2: `Conv_zscan1_*` (220,805 cells)

`code/run_bayes.py` accepts `--section sec1|sec2`.
`code/submit_sections.sh` fits the selected joint+dropout model independently
in each section with the seven ordinary controls plus the pooled pseudo-control.
`code/plot_section_reproducibility.py` excludes the pooled pseudo-control and
produces only:

- a sec1/sec2 correlation of posterior-mean `log_gamma` after subtracting the
  draw-wise mean activity of all seven ordinary negative controls;
- violin distributions of sec1/sec2 Spearman correlations within each cell
  type across cCREs and across cell types for each cCRE;
- concordance of one-sided tests against the draw-wise mean `log_gamma` of all
  seven ordinary negative controls;
- per-cell-type reproducibility, defined as cCREs with the same call in both
  sections (both significant or both insignificant) divided by cCREs with
  T7 >= 50 in both sections.

The workflow first defines one shared pair universe: a subclass–cCRE pair is
retained only when target T7 >= 50 in both sections and the combined
seven-control T7 >= 50 in both sections. This identical pair set is used for
the activity correlation, each section's BH correction, and call concordance.

## Run

Submit both compute stages and the dependent plotting stage:

```bash
bash revision/bayesian_vs_fold_change/code/submit_all.sh
```

Submit the two section-specific joint+dropout Bayesian fits on GPU nodes and
the dependent section plotting job:

```bash
bash revision/bayesian_vs_fold_change/code/submit_sections.sh
```

The bootstrap requests a 1 TB big-memory node. The Bayesian fit requests one
GPU and 96 GB host memory. Plotting starts only if both compute jobs succeed.

Submit the bootstrapped-meta-cell Bayesian fit on one GPU:

```bash
sbatch revision/bayesian_vs_fold_change/code/submit_bayesian_bootstrap_metacells.slurm
```

The meta-cell fit uses `Kmax=500` by default. Override the cells per meta-cell,
number of meta-cell replicates per subclass, or truncation level with
`BOOTSTRAP_SIZE=200`, `BOOTSTRAP_NUMBER=200`, or `KMAX=300` in the submission
environment, or pass matching CLI flags after the Slurm script path.

Submit the decoupled T7/cCRE Bayesian fit on one GPU:

```bash
sbatch revision/bayesian_vs_fold_change/code/submit_bayesian_decoupled.slurm
```

Override the stage-specific SVI budgets or infection-posterior quadrature with
`STEPS_T7=40000`, `STEPS_CRE=40000`, `NUM_POSTERIOR=1500`, or
`INFECTION_QUADRATURE_POINTS=9` in the submission environment.

For local smoke tests, both compute scripts accept `--max-cells` and
`--max-cres`; these options are never used by the production Slurm scripts.

## Outputs

- `results/bootstrap/`: all three raw bootstrap arrays, axes/configuration,
  activity estimates, detection/filter matrices, calibrated effects, and
  q-values.
- `results/bayesian/`: the production direct-activity joint+dropout Bayesian
  run used for the mean-of-seven-negative-controls precision–recall result.
  It contains posterior summaries, evidence audit, ELBO losses, posterior
  predictive checks, diagnostics, `log_gamma` draws, and the run manifest.
- `results/ablation/`: all alternative Bayesian models, prior variants,
  component/full-posterior runs, meta-cell runs, and exploratory comparison
  outputs. Alternative section fits are under `results/ablation/sections/`.
- `results/tables/`: merged pair-level comparison, per-subclass metrics,
  per-cCRE call counts, and overall summary.
- `results/figures/`: PDF and PNG versions of all final plots.
- `results/logs/`: Slurm stdout/stderr.
- `results/sections/sec1/bayesian/` and the matching `sec2/` path:
  independent selected-model fits for each physical section.
- `results/section_reproducibility/`: the Bayesian activity-correlation and
  mean-negative-control test-concordance figures and supporting tables.
