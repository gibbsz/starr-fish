#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/../../.." && pwd)"
RESULTS="$REPO/revision/bayesian_vs_fold_change/results"
# The bootstrap driver moved out to its own folder; the ablation submitters below
# are still siblings of this script.
BOOTSTRAP_CODE="$REPO/revision/run_Bootstrap"
LOG_DIR="$RESULTS/logs"
JOB_TABLE="$LOG_DIR/newnew_fit_jobs.tsv"

mkdir -p "$LOG_DIR"
printf "label\tjob_id\n" > "$JOB_TABLE"

submit() {
  local label="$1"
  shift
  local job_id
  job_id="$(sbatch --parsable "$@")"
  printf "%s\t%s\n" "$label" "$job_id" | tee -a "$JOB_TABLE"
}

submit bootstrap \
  "$BOOTSTRAP_CODE/submit_bootstrap.slurm"

submit joint \
  "$SCRIPT_DIR/submit_bayesian.slurm"

submit joint_dropout \
  "$SCRIPT_DIR/submit_bayesian.slurm" \
  --infection-model copy_number_dropout \
  --dropout-prior-label default_beta_1_9

submit joint_dropout_moderate \
  "$SCRIPT_DIR/submit_bayesian.slurm" \
  --infection-model copy_number_dropout \
  --dropout-prior-label moderate_beta_2_5 \
  --p-drop-t7-alpha 2 \
  --p-drop-t7-beta 5 \
  --p-drop-cre-alpha 2 \
  --p-drop-cre-beta 5 \
  --outdir "$RESULTS/ablation/bayesian_joint_dropout_moderate"

submit joint_dropout_high \
  "$SCRIPT_DIR/submit_bayesian.slurm" \
  --infection-model copy_number_dropout \
  --dropout-prior-label high_beta_5_5 \
  --p-drop-t7-alpha 5 \
  --p-drop-t7-beta 5 \
  --p-drop-cre-alpha 5 \
  --p-drop-cre-beta 5 \
  --outdir "$RESULTS/ablation/bayesian_joint_dropout_high"

submit joint_dropout_strongly_high \
  "$SCRIPT_DIR/submit_bayesian.slurm" \
  --infection-model copy_number_dropout \
  --dropout-prior-label strongly_high_beta_8_2 \
  --p-drop-t7-alpha 8 \
  --p-drop-t7-beta 2 \
  --p-drop-cre-alpha 8 \
  --p-drop-cre-beta 2 \
  --outdir "$RESULTS/ablation/bayesian_joint_dropout_strongly_high"

submit decoupled \
  "$SCRIPT_DIR/submit_bayesian_decoupled.slurm"

submit decoupled_no_dropout \
  "$SCRIPT_DIR/submit_bayesian_decoupled.slurm" \
  --dropout-model none

submit decoupled_dropout_moderate \
  "$SCRIPT_DIR/submit_bayesian_decoupled.slurm" \
  --dropout-prior-label moderate_beta_2_5 \
  --p-drop-t7-alpha 2 \
  --p-drop-t7-beta 5 \
  --p-drop-cre-alpha 2 \
  --p-drop-cre-beta 5 \
  --outdir "$RESULTS/ablation/bayesian_decoupled_dropout_moderate"

submit decoupled_dropout_high \
  "$SCRIPT_DIR/submit_bayesian_decoupled.slurm" \
  --dropout-prior-label high_beta_5_5 \
  --p-drop-t7-alpha 5 \
  --p-drop-t7-beta 5 \
  --p-drop-cre-alpha 5 \
  --p-drop-cre-beta 5 \
  --outdir "$RESULTS/ablation/bayesian_decoupled_dropout_high"

submit decoupled_dropout_strongly_high \
  "$SCRIPT_DIR/submit_bayesian_decoupled.slurm" \
  --dropout-prior-label strongly_high_beta_8_2 \
  --p-drop-t7-alpha 8 \
  --p-drop-t7-beta 2 \
  --p-drop-cre-alpha 8 \
  --p-drop-cre-beta 2 \
  --outdir "$RESULTS/ablation/bayesian_decoupled_dropout_strongly_high"

submit joint_full_posterior \
  "$SCRIPT_DIR/submit_bayesian.slurm" \
  --outdir "$RESULTS/ablation/bayesian_full_posterior" \
  --posterior-sites all

submit bayesian_bootstrap_metacells \
  "$SCRIPT_DIR/submit_bayesian_bootstrap_metacells.slurm"
