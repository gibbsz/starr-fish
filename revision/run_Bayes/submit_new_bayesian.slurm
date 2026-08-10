#!/bin/bash
#SBATCH --job-name=sfv8_new_bayes
#SBATCH --partition=gpu
#SBATCH --account=brlab
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=30:00:00
#SBATCH --output=/gpfs/commons/groups/ren_lab/guojiezhong/starr-fish/revision/run_Bayes/logs/%x_%j.out
#SBATCH --error=/gpfs/commons/groups/ren_lab/guojiezhong/starr-fish/revision/run_Bayes/logs/%x_%j.err

set -euo pipefail

REPO=/gpfs/commons/groups/ren_lab/guojiezhong/starr-fish
LOGS="$REPO/revision/run_Bayes/logs"
PYTHON=/gpfs/commons/home/guojiezhong/miniconda3/envs/bayes-jax/bin/python
H5AD="$REPO/revision/Data/scdata_07_29_2026_SFv8_low_dose_final_CRE_T7.h5ad"
OUTDIR="$REPO/revision/Bayes_NewData/bayesian"
ORIGIN_BAYES="$REPO/revision/bayesian_vs_fold_change/results/bayesian"

export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export MPLBACKEND=Agg
export XDG_CACHE_HOME="${TMPDIR:-/tmp}/xdg-${SLURM_JOB_ID}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

mkdir -p \
  "$XDG_CACHE_HOME" \
  "$OUTDIR" \
  "$LOGS"

cd "$REPO"
echo "start=$(date --iso-8601=seconds) host=$(hostname)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
"$PYTHON" -c "import jax; print(jax.devices()); assert any(d.platform == 'gpu' for d in jax.devices())"

"$PYTHON" revision/run_Bayes/run_bayes.py \
  --h5ad "$H5AD" \
  --outdir "$OUTDIR" \
  --level subclass \
  --channel joint \
  --infection-model copy_number_dropout \
  --activity-model direct \
  --negative-control-mode ordinary \
  --method svi \
  --kmax 60 \
  --steps 30000 \
  --lr 0.005 \
  --guide AutoNormal \
  --num-posterior 1000 \
  --posterior-sites log_gamma \
  --seed 0

cmp "$ORIGIN_BAYES/cre_blacklist.csv" "$OUTDIR/cre_blacklist.csv"
cmp "$ORIGIN_BAYES/negative_controls.csv" "$OUTDIR/negative_controls.csv"
echo "end=$(date --iso-8601=seconds)"
