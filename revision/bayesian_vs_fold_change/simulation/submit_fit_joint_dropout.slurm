#!/bin/bash
#SBATCH --job-name=brbb_sim_fit_joint
#SBATCH --partition=gpu
#SBATCH --account=brlab
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=30:00:00
#SBATCH --output=/gpfs/commons/groups/ren_lab/guojiezhong/starr-fish/revision/bayesian_vs_fold_change/simulation/results/logs/%x_%j.out
#SBATCH --error=/gpfs/commons/groups/ren_lab/guojiezhong/starr-fish/revision/bayesian_vs_fold_change/simulation/results/logs/%x_%j.err

set -euo pipefail

REPO=/gpfs/commons/groups/ren_lab/guojiezhong/starr-fish
PYTHON=/gpfs/commons/home/guojiezhong/miniconda3/envs/bayes-jax/bin/python
SIM_DIR="$REPO/revision/bayesian_vs_fold_change/simulation"
SIM_OUT="${SIM_OUT:-$SIM_DIR/results/joint_dropout_simulated}"
FIT_OUT="${FIT_OUT:-$SIM_DIR/results/joint_dropout_fit}"

export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export MPLBACKEND=Agg
export XDG_CACHE_HOME="${TMPDIR:-/tmp}/xdg-${SLURM_JOB_ID}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

mkdir -p "$XDG_CACHE_HOME" "$SIM_DIR/results/logs"

cd "$REPO"
echo "start=$(date --iso-8601=seconds) host=$(hostname)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
"$PYTHON" -c "import jax; print(jax.devices()); assert any(d.platform == 'gpu' for d in jax.devices())"
"$PYTHON" revision/bayesian_vs_fold_change/code/run_bayes.py \
  --h5ad "$SIM_OUT/simulated_joint_dropout.h5ad" \
  --outdir "$FIT_OUT" \
  --level subclass \
  --channel joint \
  --infection-model copy_number_dropout \
  --method svi \
  --steps "${STEPS:-30000}" \
  --num-posterior "${NUM_POSTERIOR:-1000}" \
  --posterior-sites log_gamma log_rho log_a \
  "$@"
echo "end=$(date --iso-8601=seconds)"
