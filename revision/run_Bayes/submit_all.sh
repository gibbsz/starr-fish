#!/bin/bash
set -euo pipefail

# The three stages no longer live in one directory: the bootstrap driver is in
# run_Bootstrap/, this bayesian driver is in run_Bayes/, and the plotting jobs
# stayed with the analysis they belong to.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REVISION="$(cd "$HERE/.." && pwd)"
ANALYSIS_DIR="$REVISION/bayesian_vs_fold_change"
ANALYSIS_CODE="$ANALYSIS_DIR/code"
BOOTSTRAP_CODE="$REVISION/run_Bootstrap"
mkdir -p "$ANALYSIS_DIR/results/logs"

bootstrap_job="$(sbatch --parsable "$BOOTSTRAP_CODE/submit_bootstrap.slurm")"
bayesian_job="$(
  sbatch --parsable "$ANALYSIS_CODE/submit_bayesian.slurm" \
    --infection-model copy_number_dropout \
    --negative-control-mode ordinary \
    --activity-model direct \
    --outdir "$REVISION/Bayes_OldData/bayesian"
)"
plots_job="$(
  sbatch --parsable \
    --dependency="afterok:${bootstrap_job}:${bayesian_job}" \
    "$ANALYSIS_CODE/submit_plots.slurm"
)"

printf 'bootstrap_job=%s\nbayesian_job=%s\nplots_job=%s\n' \
  "$bootstrap_job" "$bayesian_job" "$plots_job"
