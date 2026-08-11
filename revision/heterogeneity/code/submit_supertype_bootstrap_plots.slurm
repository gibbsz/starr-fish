#!/bin/bash
#SBATCH --job-name=het_supertype_boot_plot
#SBATCH --partition=cpu
#SBATCH --account=brlab
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/gpfs/commons/groups/ren_lab/guojiezhong/starr-fish/revision/heterogeneity/results/logs/%x_%j.out
#SBATCH --error=/gpfs/commons/groups/ren_lab/guojiezhong/starr-fish/revision/heterogeneity/results/logs/%x_%j.err

set -euo pipefail

REPO=/gpfs/commons/groups/ren_lab/guojiezhong/starr-fish
PYTHON=/gpfs/commons/home/guojiezhong/miniconda3/envs/scvi/bin/python
RESULTS="$REPO/revision/heterogeneity/results/supertype"

export MPLBACKEND=Agg
export MPLCONFIGDIR="${TMPDIR:-/tmp}/matplotlib-${SLURM_JOB_ID}"
export XDG_CACHE_HOME="${TMPDIR:-/tmp}/xdg-${SLURM_JOB_ID}"

mkdir -p \
  "$MPLCONFIGDIR" \
  "$XDG_CACHE_HOME" \
  "$REPO/revision/heterogeneity/results/logs"

cd "$REPO"
echo "start=$(date --iso-8601=seconds) host=$(hostname)"
"$PYTHON" revision/heterogeneity/code/make_heterogeneity_plots.py \
  --model bootstrap \
  --combined-bootstrap-dir "$REPO/revision/Bootstrap_OldData" \
  --split-bootstrap-dir "$RESULTS/bootstrap" \
  --outdir "$RESULTS" \
  "$@"
echo "end=$(date --iso-8601=seconds)"
