#!/bin/bash
#SBATCH --job-name=brbb_t7_filter_pr_final
#SBATCH --partition=cpu
#SBATCH --account=brlab
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --output=/gpfs/commons/groups/ren_lab/guojiezhong/starr-fish/revision/bayesian_vs_fold_change/results/logs/%x_%j.out
#SBATCH --error=/gpfs/commons/groups/ren_lab/guojiezhong/starr-fish/revision/bayesian_vs_fold_change/results/logs/%x_%j.err
#
# The manuscript precision-recall figure: two methods, both tested against the
# draw-wise MEAN of the seven ordinary negative controls.
#
#   Joint+dropout mean controls   (--include-mean-control)
#   Bootstrap mean controls       (--include-bootstrap-mean-control)
#
# --methods with no values drops the ablation arms (Joint, Decoupled,
# Joint+dropout, Decoupled+dropout, Bootstrap-vs-pooled). --skip-loo drops the
# leave-one-out empirical-FDR arm, whose reference is the pooled negative-control
# null rather than the control mean. What remains is the mean-of-controls
# comparison only.
#
# --common-pairs evaluates both methods on the intersection of their tested pairs
# and recomputes BH there, so the two curves are directly comparable.
#
# The ablation sweep is still produced by run_newnew_figures.sh under the
# method_activity_t7_filter_all_methods stem, so nothing is lost -- it just stops
# occupying the curated stem.
#
#   sbatch code/figure_final/submit_t7_filter_precision_recall.slurm

set -euo pipefail

REPO=/gpfs/commons/groups/ren_lab/guojiezhong/starr-fish
ANALYSIS="$REPO/revision/bayesian_vs_fold_change"
PYTHON=${PYTHON:-/gpfs/commons/home/guojiezhong/miniconda3/envs/scvi/bin/python}

export MPLBACKEND=Agg
export MPLCONFIGDIR="${TMPDIR:-/tmp}/matplotlib-${SLURM_JOB_ID:-local}"
export XDG_CACHE_HOME="${TMPDIR:-/tmp}/xdg-${SLURM_JOB_ID:-local}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

mkdir -p "$MPLCONFIGDIR" "$XDG_CACHE_HOME" "$ANALYSIS/results/logs"

cd "$REPO"

# Writes, under the default method_activity_t7_filter stem:
#   results/figures/work/method_activity_t7_filter_precision_recall.pdf
#   results/figures/work/method_activity_t7_filter_pr_curves_t7_ge50.pdf
# both of which are collected into results/figures/final/.
"$PYTHON" revision/bayesian_vs_fold_change/code/plot_t7_filter_precision_recall.py \
  --methods \
  --include-mean-control \
  --include-bootstrap-mean-control \
  --common-pairs \
  --skip-loo \
  --pr-threshold 50 \
  --q-cutoff 0.05
