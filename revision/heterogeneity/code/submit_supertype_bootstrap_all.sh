#!/bin/bash
# Submit the annotated-supertype bootstrap fit, then its agreement plot.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ANALYSIS_DIR="$(cd "$HERE/.." && pwd)"
mkdir -p "$ANALYSIS_DIR/results/logs"

bootstrap_job="$(sbatch --parsable "$HERE/submit_supertype_bootstrap.slurm")"
plots_job="$(
  sbatch --parsable \
    --dependency="afterok:${bootstrap_job}" \
    "$HERE/submit_supertype_bootstrap_plots.slurm"
)"

printf 'bootstrap_job=%s\nplots_job=%s\n' "$bootstrap_job" "$plots_job"
