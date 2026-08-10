#!/bin/bash
set -euo pipefail

REPO=/gpfs/commons/groups/ren_lab/guojiezhong/starr-fish
PYTHON=${PYTHON:-/gpfs/commons/home/guojiezhong/miniconda3/envs/scvi/bin/python}

cd "$REPO"

run_step() {
  local script="$1"
  shift || true
  echo "[figures-no-sections] start $script $(date --iso-8601=seconds)"
  "$PYTHON" "revision/bayesian_vs_fold_change/code/$script" "$@"
  echo "[figures-no-sections] done  $script $(date --iso-8601=seconds)"
}

run_step recompute_bootstrap_prior_mask_inputs.py
run_step plot_results.py
run_step plot_method_activity_heatmap.py
run_step plot_method_activity_correlation.py
run_step plot_metacell_activity_correlation.py
run_step plot_dropout_prior_activity_correlation.py
run_step compare_dropout_prior_experiments.py
run_step plot_activity_atac_correlation.py
run_step plot_metacell_low_mode_pair_count_diagnostics.py
run_step compute_t7_filter_negative_control_stats.py
run_step plot_t7_filter_precision_recall.py
run_step plot_stripe_count_diagnostics.py
run_step plot_decoupled_joint_offdiag_diagnostics.py
run_step plot_decoupled_joint_single_cell_mismatch.py
run_step plot_marked_decoupled_joint_outliers.py
run_step plot_posterior_vs_t7.py
run_step plot_percell_aav_hist.py
run_step plot_percell_perccre_aav_hist.py
run_step plot_posterior_k_hist.py
