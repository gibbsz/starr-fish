#!/bin/bash
#SBATCH --job-name=brbb_simulate_joint
#SBATCH --partition=bigmem
#SBATCH --account=brlab
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=/gpfs/commons/groups/ren_lab/guojiezhong/starr-fish/revision/bayesian_vs_fold_change/simulation/results/logs/%x_%j.out
#SBATCH --error=/gpfs/commons/groups/ren_lab/guojiezhong/starr-fish/revision/bayesian_vs_fold_change/simulation/results/logs/%x_%j.err

set -euo pipefail

REPO=/gpfs/commons/groups/ren_lab/guojiezhong/starr-fish
PYTHON=/gpfs/commons/home/guojiezhong/miniconda3/envs/bayes-jax/bin/python
SIM_DIR="$REPO/revision/bayesian_vs_fold_change/simulation"
SIM_OUT="${SIM_OUT:-$SIM_DIR/results/joint_dropout_simulated}"
STATS_OUT="${STATS_OUT:-$SIM_DIR/results/joint_dropout_stats}"

export MPLBACKEND=Agg
export XDG_CACHE_HOME="${TMPDIR:-/tmp}/xdg-${SLURM_JOB_ID}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

mkdir -p "$XDG_CACHE_HOME" "$SIM_DIR/results/logs"

cd "$REPO"
echo "start=$(date --iso-8601=seconds) host=$(hostname)"
"$PYTHON" "$SIM_DIR/simulate_joint_dropout.py" \
  --outdir "$SIM_OUT" \
  "$@"
"$PYTHON" "$SIM_DIR/compare_simulated_to_real.py" \
  --sim-h5ad "$SIM_OUT/simulated_joint_dropout.h5ad" \
  --outdir "$STATS_OUT"
echo "end=$(date --iso-8601=seconds)"
