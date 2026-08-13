#!/bin/bash
# Whole-dataset TF motif analysis for joint-dropout mean-control results.
#
# Usage:
#   sbatch revision/TFMotif/code/submit_tf_motif_analysis.slurm
#   sbatch revision/TFMotif/code/submit_tf_motif_analysis.slurm --q-cutoff 0.01

#SBATCH --job-name=tfmotif
#SBATCH --account=brlab
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/gpfs/commons/groups/ren_lab/guojiezhong/starr-fish/revision/TFMotif/logs/slurm-%j.out
#SBATCH --error=/gpfs/commons/groups/ren_lab/guojiezhong/starr-fish/revision/TFMotif/logs/slurm-%j.err

set -euo pipefail

REPO=/gpfs/commons/groups/ren_lab/guojiezhong/starr-fish
PYTHON=/gpfs/commons/home/guojiezhong/miniconda3/envs/bayes-jax/bin/python

mkdir -p "${REPO}/revision/TFMotif/logs"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/matplotlib-${SLURM_JOB_ID:-tfmotif}"
mkdir -p "$MPLCONFIGDIR"

cd "$REPO"
"$PYTHON" revision/TFMotif/code/run_tf_motif_analysis.py "$@"
