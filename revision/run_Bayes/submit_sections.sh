#!/bin/bash
set -euo pipefail

# Per-section fits stay under bayesian_vs_fold_change/results/sections/; only the
# whole-dataset run writes to Bayes_OldData/. The submitter and the section
# plotting job both live with that analysis.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REVISION="$(cd "$HERE/.." && pwd)"
ANALYSIS_DIR="$REVISION/bayesian_vs_fold_change"
ANALYSIS_CODE="$ANALYSIS_DIR/code"
mkdir -p "$ANALYSIS_DIR/results/logs"

bayesian_sec1="$(
  sbatch --parsable "$ANALYSIS_CODE/submit_bayesian.slurm" \
    --section sec1 \
    --infection-model copy_number_dropout \
    --negative-control-mode ordinary-and-pooled \
    --activity-model hierarchical \
    --outdir "$ANALYSIS_DIR/results/sections/sec1/bayesian"
)"
bayesian_sec2="$(
  sbatch --parsable "$ANALYSIS_CODE/submit_bayesian.slurm" \
    --section sec2 \
    --infection-model copy_number_dropout \
    --negative-control-mode ordinary-and-pooled \
    --activity-model hierarchical \
    --outdir "$ANALYSIS_DIR/results/sections/sec2/bayesian"
)"
plots_job="$(
  sbatch --parsable \
    --dependency="afterok:${bayesian_sec1}:${bayesian_sec2}" \
    "$ANALYSIS_CODE/submit_section_plots.slurm"
)"

printf 'bayesian_sec1=%s\nbayesian_sec2=%s\nplots_job=%s\n' \
  "$bayesian_sec1" "$bayesian_sec2" "$plots_job"
