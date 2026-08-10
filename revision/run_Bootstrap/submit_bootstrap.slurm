#!/bin/bash
#SBATCH --job-name=brbb_bootstrap
#SBATCH --partition=bigmem
#SBATCH --account=brlab
#SBATCH --cpus-per-task=64
#SBATCH --mem=1000G
#SBATCH --time=48:00:00
#SBATCH --output=/gpfs/commons/groups/ren_lab/guojiezhong/starr-fish/revision/run_Bootstrap/logs/%x_%j.out
#SBATCH --error=/gpfs/commons/groups/ren_lab/guojiezhong/starr-fish/revision/run_Bootstrap/logs/%x_%j.err

set -euo pipefail

REPO=/gpfs/commons/groups/ren_lab/guojiezhong/starr-fish
LOGS="$REPO/revision/run_Bootstrap/logs"
PYTHON=/gpfs/commons/home/guojiezhong/miniconda3/envs/scvi/bin/python

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
  "$LOGS"

cd "$REPO"
echo "start=$(date --iso-8601=seconds) host=$(hostname)"
"$PYTHON" revision/run_Bootstrap/run_bootstrap.py \
  --bootstrap-number 10000 \
  --n-jobs 62 \
  --min-detected-cells 5 \
  "$@"
echo "end=$(date --iso-8601=seconds)"
