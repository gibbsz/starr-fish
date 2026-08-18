#!/bin/bash
#SBATCH --job-name=sfv8_activity_matrices
#SBATCH --partition=cpu
#SBATCH --account=brlab
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=/gpfs/commons/groups/ren_lab/guojiezhong/starr-fish/revision/run_Bayes/logs/%x_%j.out
#SBATCH --error=/gpfs/commons/groups/ren_lab/guojiezhong/starr-fish/revision/run_Bayes/logs/%x_%j.err

set -euo pipefail

REPO=/gpfs/commons/groups/ren_lab/guojiezhong/starr-fish
PYTHON=${PYTHON:-/gpfs/commons/home/guojiezhong/miniconda3/envs/scvi/bin/python}
EXPORT="$REPO/revision/run_Bayes/export_activity_matrix.py"
OLD_DATA="$REPO/revision/Bayes_OldData"
NEW_DATA="$REPO/revision/Bayes_NewData"
OLD_H5AD="$REPO/revision/Data/scdata_5_28_2025_BRBB500gn_final_CRE_T7CRE_NEWNEW.h5ad"
NEW_H5AD="$REPO/revision/Data/scdata_07_29_2026_SFv8_low_dose_final_CRE_T7.h5ad"
OLD_REFERENCE="$REPO/revision/bayesian_vs_fold_change/results/tables/joint_dropout_direct_activity_mean_negative_control_tests_t7_ge50.csv.gz"
NEW_REFERENCE="$NEW_DATA/tables/new_mean_negative_control_tests_t7_ge50.csv.gz"

export MPLBACKEND=Agg
export XDG_CACHE_HOME="${TMPDIR:-/tmp}/xdg-${SLURM_JOB_ID:-local}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

mkdir -p "$XDG_CACHE_HOME" "$OLD_DATA/tables" "$NEW_DATA/tables" "$REPO/revision/run_Bayes/logs"

cd "$REPO"
echo "start=$(date --iso-8601=seconds) host=$(hostname)"

# Mean-control reference: the family the published activity heatmaps plot.
"$PYTHON" "$EXPORT" \
  --bayes-dir "$OLD_DATA/bayesian" \
  --h5ad "$OLD_H5AD" \
  --outdir "$OLD_DATA/tables" \
  --q-t7-threshold 50 \
  --check-against "$OLD_REFERENCE"

"$PYTHON" "$EXPORT" \
  --bayes-dir "$NEW_DATA/bayesian" \
  --h5ad "$NEW_H5AD" \
  --outdir "$NEW_DATA/tables" \
  --q-t7-threshold 50 \
  --check-against "$NEW_REFERENCE"

# Mean+1SD reference: same activity matrix, stricter null, suffixed stem.
"$PYTHON" "$EXPORT" \
  --bayes-dir "$OLD_DATA/bayesian" \
  --h5ad "$OLD_H5AD" \
  --outdir "$OLD_DATA/tables" \
  --q-t7-threshold 50 \
  --control-sd-multiplier 1

"$PYTHON" "$EXPORT" \
  --bayes-dir "$NEW_DATA/bayesian" \
  --h5ad "$NEW_H5AD" \
  --outdir "$NEW_DATA/tables" \
  --q-t7-threshold 50 \
  --control-sd-multiplier 1

echo "end=$(date --iso-8601=seconds)"
