#!/bin/bash
#SBATCH --job-name=sfv8_origin_new_stats
#SBATCH --partition=cpu
#SBATCH --account=brlab
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/gpfs/commons/groups/ren_lab/guojiezhong/starr-fish/revision/origin_vs_new/results/logs/%x_%j.out
#SBATCH --error=/gpfs/commons/groups/ren_lab/guojiezhong/starr-fish/revision/origin_vs_new/results/logs/%x_%j.err

set -euo pipefail

REPO=/gpfs/commons/groups/ren_lab/guojiezhong/starr-fish
PYTHON=/gpfs/commons/home/guojiezhong/miniconda3/envs/scvi/bin/python
ANALYSIS="$REPO/revision/origin_vs_new"
NEW_RESULTS="$REPO/revision/Bayes_NewData"
NEW_H5AD="$REPO/revision/Data/scdata_07_29_2026_SFv8_low_dose_final_CRE_T7.h5ad"
NEW_BAYES="$NEW_RESULTS/bayesian"
NEW_TABLES="$NEW_RESULTS/tables"
NEW_FIGURES="$NEW_RESULTS/figures"

export MPLBACKEND=Agg
export MPLCONFIGDIR="${TMPDIR:-/tmp}/mpl-${SLURM_JOB_ID}"
export XDG_CACHE_HOME="${TMPDIR:-/tmp}/xdg-${SLURM_JOB_ID}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

mkdir -p \
  "$MPLCONFIGDIR" \
  "$XDG_CACHE_HOME" \
  "$NEW_TABLES" \
  "$NEW_FIGURES" \
  "$ANALYSIS/results/comparison/tables" \
  "$ANALYSIS/results/comparison/figures" \
  "$ANALYSIS/results/logs"

cd "$REPO"
echo "start=$(date --iso-8601=seconds) host=$(hostname)"

"$PYTHON" revision/bayesian_vs_fold_change/code/test_mean_negative_control_activity.py \
  --bayes-dir "$NEW_BAYES" \
  --h5ad "$NEW_H5AD" \
  --t7-threshold 50 \
  --effect-threshold 0 \
  --q-cutoff 0.05 \
  --tables-dir "$NEW_TABLES" \
  --figures-dir "$NEW_FIGURES" \
  --stem new_mean_negative_control_tests_t7_ge50

"$PYTHON" "$ANALYSIS/code/compare_origin_vs_new.py"
echo "end=$(date --iso-8601=seconds)"
