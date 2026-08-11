#!/bin/bash
# Submit the split-subclass Bayesian fit, then its dependent agreement plot.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ANALYSIS_DIR="$(cd "$HERE/.." && pwd)"
mkdir -p "$ANALYSIS_DIR/results/logs"

bayesian_job="$(sbatch --parsable "$HERE/submit_bayesian.slurm")"
plots_job="$(
  sbatch --parsable \
    --dependency="afterok:${bayesian_job}" \
    "$HERE/submit_plots.slurm"
)"

printf 'bayesian_job=%s\nplots_job=%s\n' "$bayesian_job" "$plots_job"
