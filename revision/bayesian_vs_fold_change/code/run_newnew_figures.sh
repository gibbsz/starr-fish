#!/bin/bash
set -euo pipefail

REPO=/gpfs/commons/groups/ren_lab/guojiezhong/starr-fish
PYTHON=${PYTHON:-/gpfs/commons/home/guojiezhong/miniconda3/envs/scvi/bin/python}

cd "$REPO"

run_step() {
  local script="$1"
  shift || true
  echo "[figures] start $script $(date --iso-8601=seconds)"
  "$PYTHON" "revision/bayesian_vs_fold_change/code/$script" "$@"
  echo "[figures] done  $script $(date --iso-8601=seconds)"
}

run_step recompute_bootstrap_prior_mask_inputs.py
run_step plot_results.py
run_step figure_work/plot_activity_heatmaps.py
run_step plot_method_activity_heatmap.py
run_step plot_method_activity_correlation.py
run_step figure_work/plot_metacell_activity_correlation.py
run_step figure_work/plot_dropout_prior_activity_correlation.py
run_step compare_dropout_prior_experiments.py
run_step plot_activity_atac_correlation.py
run_step figure_work/plot_metacell_low_mode_pair_count_diagnostics.py
run_step compute_t7_filter_negative_control_stats.py
# plot_t7_filter_precision_recall.py reads the LOO empirical-FDR and max-control
# test tables. Both are produced here rather than assumed to exist: the copies on
# disk predate the "direct_activity" stem these scripts now default to, so the
# figure step cannot find them.
run_step test_individual_negative_control_loo_empirical_fdr.py
run_step figure_work/test_max_negative_control_activity.py
# The mean-of-controls T7 series tables feed the curated precision-recall figure.
# Produced here so they track the current fits rather than whatever was on disk.
run_step figure_work/test_mean_negative_control_activity_threshold_series.py
run_step figure_work/test_bootstrap_mean_negative_control_activity_threshold_series.py
# The ablation sweep (Joint, Decoupled, +dropout variants, Metacell, LOO) is
# exploratory and keeps its own stem. The curated mean-of-controls figure under
# the plain stem comes from figure_final/submit_t7_filter_precision_recall.slurm.
run_step plot_t7_filter_precision_recall.py --stem method_activity_t7_filter_all_methods
run_step figure_work/plot_stripe_count_diagnostics.py
run_step figure_work/plot_decoupled_joint_offdiag_diagnostics.py
run_step figure_work/plot_decoupled_joint_single_cell_mismatch.py
run_step figure_work/plot_marked_decoupled_joint_outliers.py
run_step figure_work/plot_posterior_vs_t7.py
run_step figure_work/plot_percell_aav_hist.py
run_step figure_work/plot_percell_perccre_aav_hist.py
run_step figure_work/plot_posterior_k_hist.py

# Curate the manuscript set last, once every producer has written to work/.
run_step figure_final/collect_final_figures.py
