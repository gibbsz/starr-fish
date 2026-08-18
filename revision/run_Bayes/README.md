# run_Bayes

Fits the posterior for each dataset and exports the canonical tables every
activity figure reads.

## Fitting

- `run_bayes.py`: the SVI fit. `submit_new_bayesian.slurm` runs it for the new
  low-dose dataset into `../Bayes_NewData/bayesian/`; the original run's
  posterior lives in `../Bayes_OldData/bayesian/`.

## Activity matrices (the figure inputs)

- `export_activity_matrix.py`: reduces one posterior to the tables below.
- `activity_matrix_io.py`: owns the file-name convention and the readers, so the
  writer and the plotting scripts cannot drift apart.
- `submit_activity_matrices.slurm`: runs the export for both datasets, for the
  mean-control null and the mean+1SD null.

Each dataset gets, in its own `tables/` directory:

| file | content |
|---|---|
| `subclass_cre_activity_matrix.csv.gz` | subclass x cCRE, **the value the heatmaps colour**: posterior mean target `log_gamma` minus the posterior mean of the seven ordinary negative controls |
| `subclass_cre_target_t7_matrix.csv.gz` | subclass x cCRE target T7 totals — the source of every T7 mask |
| `subclass_cre_negative_control_activity_matrix.csv.gz` | subclass x 7 controls, posterior-mean `log_gamma`, for the control-spread strip |
| `subclass_cre_significance.csv.gz` | one row per pair: `p_right`, `q_right_t7_ge50`, the effect interval, T7 and cell counts |
| `subclass_cre_matrix_manifest.json` | provenance, shapes, and the cross-check against the previously shipped test table |

`subclass_cre_mean_plus_1sd_*` are the same products under the stricter
mean+1SD control reference.

The matrices are **unfiltered**: every fitted target pair is exported, `NaN`
only where the model has no pair. Figures apply their own T7 mask from the T7
matrix, which is why one export serves every threshold.

`p_right` is universe-free and is stored as computed. BH `q` is not — it depends
on which pairs are in the family — so `q_right_t7_ge50` is BH over *this*
dataset's T7 >= 50 pairs and is `NaN` outside it. A figure plotting a different
family (the shared original/new universe, for instance) must re-run BH over that
family from `p_right`; `plot_origin_vs_new_heatmap.py` does exactly that and is
the only place where `q` is derived rather than read.

## Consumers

- `../origin_vs_new/code/plot_origin_vs_new_heatmap.py` — both panels, both
  datasets, shared universe.
- `../bayesian_vs_fold_change/code/plot_method_activity_heatmap.py` via
  `--activity-matrix-dir` — the published `joint_dropout_activity_heatmap`
  figures.
