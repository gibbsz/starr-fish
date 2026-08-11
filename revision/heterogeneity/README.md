# Subclass heterogeneity via random and annotated subgroup splitting

This analysis probes activity-estimate stability by splitting the highest-T7
subclasses into random subgroups and asking how far the per-subgroup estimates
recover the intact-subclass estimate under the Bayesian joint+dropout model.
The parallel annotated-supertype analysis asks the same question using the
biological `supertype_name` labels already present in the input h5ad.

The original heterogeneity code and results were moved to
`revision/archive/heterogeneity_old/`.

## Design

1. **Top subclasses** — the 10 subclasses with the largest `total_t7` in
   `revision/Data/subclass_total_t7_counts.csv` (names standardized to match the
   model's `subclass` labels: Allen numeric prefix dropped, `/`→`-`).
2. **Split** — within each top subclass, the combined sec1+sec2 cells are
   randomly partitioned into 5 subgroups (`<subclass>_group_<i>`) with a fixed
   seed (`relabel.SPLIT_SEED`, independent RNG stream per subclass). Only the
   `subclass` label changes, so the five subgroups stay nested under the same
   parent `class`. All other subclasses are kept intact and merged back in.
3. **Fit** — `run_bayes_split.py` fits the split labels with the production
   joint CRE/T7 copy-number+dropout model: direct activity, ordinary annotated
   negative controls, default Beta(1, 9) dropout priors, SVI 30k steps, and
   1000 posterior draws.
4. **Agreement metric** — the **intact-subclass** Bayesian estimate is reused
   from `revision/bayesian_vs_fold_change/results/bayesian` (no recompute).
   For every cCRE and each of the 10 selected cell types,
   `make_heterogeneity_plots.py` compares:
   - x = negative-control-centred activity in the intact cell type;
   - y = the unweighted mean activity across its 5 random cell subsets.
   Each panel is one cell type and contains all fitted cCREs as points; vertical
   bars show the SD across the 5 subset estimates. Panels also include the y=x
   line, Lin's concordance correlation coefficient (CCC), and mean absolute
   error (MAE), and are ordered from largest to smallest original cell count.
   Bootstrap is not used.

## Annotated-supertype design

The parallel workflow uses the same input h5ad, top-ten subclasses, intact
Bayesian reference, model configuration, calibration, and cCRE set. For cells
in each target subclass, the Allen numeric prefix is removed from
`obs["supertype_name"]` and that standardized label becomes the model's
subclass label. Cells outside the ten targets retain their original subclass.

The ten targets contain 5, 1, 6, 14, 5, 8, 5, 9, 4, and 6 annotated
supertypes, respectively (63 total). Every annotated supertype is retained,
including the smallest five-cell group. The plotted y-value is the
**cell-count-weighted mean** of the supertype activity estimates. This makes the
composition of the comparison match the intact cell type and prevents a
five-cell supertype from contributing as much to the agreement point as a
large supertype.

The vertical bars deliberately remain the **unweighted sample SD across
supertypes**. They describe between-supertype biological heterogeneity, not
posterior uncertainty or uncertainty in the weighted mean. Each panel reports
the median of these supertype SDs as a compact heterogeneity summary. The
unweighted mean is also retained in the output table, which makes it possible
to contrast a typical annotated supertype with the abundance-weighted intact
population. `Endo NN` has one annotated supertype, so its points and agreement
metrics are shown without SD bars and its supertype SD is unavailable.

To measure heterogeneity directly, every unordered pair of annotated
supertypes is compared **within its parent cell type**. Lin's CCC is calculated
between the two inferred activity vectors across all 389 fitted,
non-blacklisted cCREs. Each pair contributes once to its parent distribution;
lower CCC indicates more divergent inferred activity profiles. The pair table
retains both supertype cell counts because low CCC involving a very small group
may reflect greater estimation noise as well as biological heterogeneity.
`Endo NN` has no pairwise value because one supertype cannot form a pair.

## Run

```bash
bash revision/heterogeneity/code/submit_all.sh
```

Submits the Bayesian fit (1 GPU, 96 GB) and its dependent plotting job
(`afterok`). To run the plot alone once the fit exists:

```bash
sbatch revision/heterogeneity/code/submit_plots.slurm
```

Local smoke tests: the fit script accepts `--max-cells` / `--max-cres`.

`make_heterogeneity_plots.py` accepts:

- `--calibration negctrl_only` (default): negative-control centering only,
  retaining a common per-cCRE scale across the two independent fits.
- `--ncols 5`: control the cell-type panel layout.

To submit the independent annotated-supertype fit and its dependent plot:

```bash
bash revision/heterogeneity/code/submit_supertype_all.sh
```

The underlying fit command uses `run_bayes_split.py --grouping supertype`.
The default remains `--grouping random`, so existing commands and output paths
retain their original behavior.

To run the same annotated-supertype comparison with the manuscript bootstrap
estimator instead of the Bayesian model:

```bash
bash revision/heterogeneity/code/submit_supertype_bootstrap_all.sh
```

This runs 10,000 bootstrap iterations with 62 workers, writes the fit under
`results/supertype/bootstrap/`, and then reuses the intact-subclass bootstrap
from `revision/Bootstrap_OldData`. The plot applies negative-control-only
centering to both runs, matching the Bayesian figure's cross-fit calibration.

## Outputs (`results/`)

- `split/bayesian/` — full Bayesian fit results on the split labels.
- `tables/bayesian_subset_vs_whole.csv` — intact activity, mean subset
  activity, subset SD, and differences for every cell-type/cCRE pair.
- `tables/bayesian_subset_vs_whole_summary.csv` — per-cell-type and overall CCC,
  Pearson correlation, mean error, MAE, and RMSE.
- `raw/combined_activity_bayesian.csv` and
  `raw/split_activity_bayesian.csv` — activity matrices actually compared.
- `figures/bayesian_subset_mean_vs_whole.{pdf,png}` — one panel per cell type,
  with all fitted cCREs as points, subset-SD bars, and a diagonal y=x line.
- `logs/` — Slurm stdout/stderr.

## Annotated-supertype outputs (`results/supertype/`)

- `bayesian/` — Bayesian fit using the 63 annotated supertype labels, including
  `cell_group_assignment.csv`, `subgroup_cell_counts.csv`, and a manifest with
  the complete parent-to-supertype mapping.
- `tables/bayesian_supertype_vs_whole.csv` — intact activity, cell-count-weighted
  and unweighted mean supertype activity, unweighted supertype SD,
  contributing-supertype count, and both weighted and unweighted differences
  for every cell-type/cCRE pair.
- `tables/bayesian_supertype_vs_whole_summary.csv` — per-cell-type and overall
  weighted-agreement metrics plus median/mean supertype SD.
- `tables/bayesian_supertype_pairwise_ccc.csv` — one row per within-parent
  supertype pair, with CCC, Pearson correlation, MAE, RMSE, and both supertype
  cell counts.
- `tables/bayesian_supertype_pairwise_ccc_summary.csv` — pairwise-CCC
  distribution summaries for each parent cell type.
- `raw/combined_activity_bayesian.csv` and
  `raw/supertype_activity_bayesian.csv` — activity matrices actually compared.
- `figures/bayesian_supertype_mean_vs_whole.{pdf,png}` — one panel per target
  cell type, with cell-count-weighted cCRE means, unweighted supertype-SD bars
  where defined, and y=x.
- `figures/bayesian_supertype_pairwise_ccc.{pdf,png}` — within-parent pairwise
  CCC distributions; each red point is an annotated-supertype pair. The blue
  diamond for each parent is the CCC of its cell-count-weighted supertype mean
  versus the intact whole-cell-type activity, providing the aggregate-agreement
  baseline from the mean-vs-whole figure.
- `figures/bayesian_supertype_pairwise_ccc_vs_min_cells.{pdf,png}` — pairwise
  CCC against the smaller supertype's cell count on a log x-axis, colored by
  parent cell type and annotated with the overall Spearman association.
- `bootstrap/` — annotated-supertype manuscript bootstrap fit, including the
  10,000-iteration activity array and the same subgroup mapping metadata.
- `tables/bootstrap_supertype_vs_whole.csv` and its `_summary.csv` companion —
  bootstrap weighted-agreement data and supertype-heterogeneity metrics.
- `figures/bootstrap_supertype_mean_vs_whole.{pdf,png}` — the bootstrap version
  of the annotated-supertype agreement figure.
- `figures/bootstrap_supertype_pairwise_ccc.{pdf,png}` — the corresponding
  bootstrap within-parent pairwise-CCC distributions.
- `figures/bootstrap_supertype_pairwise_ccc_vs_min_cells.{pdf,png}` — bootstrap
  pairwise CCC against minimum pair cell support.
