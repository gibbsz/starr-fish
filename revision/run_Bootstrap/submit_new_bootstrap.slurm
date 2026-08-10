#!/bin/bash
#SBATCH --job-name=sfv8_new_bootstrap
#SBATCH --partition=cpu
#SBATCH --account=brlab
#SBATCH --cpus-per-task=14
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --output=/gpfs/commons/groups/ren_lab/guojiezhong/starr-fish/revision/run_Bootstrap/logs/%x_%j.out
#SBATCH --error=/gpfs/commons/groups/ren_lab/guojiezhong/starr-fish/revision/run_Bootstrap/logs/%x_%j.err

set -euo pipefail

REPO=/gpfs/commons/groups/ren_lab/guojiezhong/starr-fish
LOGS="$REPO/revision/run_Bootstrap/logs"
PYTHON=/gpfs/commons/home/guojiezhong/miniconda3/envs/scvi/bin/python
H5AD="$REPO/revision/Data/scdata_07_29_2026_SFv8_low_dose_final_CRE_T7.h5ad"
OUTDIR="$REPO/revision/Bootstrap_NewData"
ORIGIN_BOOTSTRAP="$REPO/revision/Bootstrap_OldData"

export MPLBACKEND=Agg
export MPLCONFIGDIR="${TMPDIR:-/tmp}/matplotlib-${SLURM_JOB_ID}"
export NUMBA_CACHE_DIR="${TMPDIR:-/tmp}/numba-${SLURM_JOB_ID}"
export XDG_CACHE_HOME="${TMPDIR:-/tmp}/xdg-${SLURM_JOB_ID}"
export JOBLIB_TEMP_FOLDER="${TMPDIR:-/tmp}/joblib-${SLURM_JOB_ID}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

mkdir -p \
  "$MPLCONFIGDIR" \
  "$NUMBA_CACHE_DIR" \
  "$XDG_CACHE_HOME" \
  "$JOBLIB_TEMP_FOLDER" \
  "$OUTDIR" \
  "$LOGS"

cd "$REPO"
echo "start=$(date --iso-8601=seconds) host=$(hostname)"
"$PYTHON" revision/run_Bootstrap/run_bootstrap.py \
  --h5ad "$H5AD" \
  --outdir "$OUTDIR" \
  --bootstrap-number 10000 \
  --n-jobs 12 \
  --min-detected-cells 5 \
  --seed 0

cmp "$ORIGIN_BOOTSTRAP/cre_blacklist.csv" "$OUTDIR/cre_blacklist.csv"
cmp "$ORIGIN_BOOTSTRAP/negative_controls.csv" "$OUTDIR/negative_controls.csv"
echo "end=$(date --iso-8601=seconds)"
