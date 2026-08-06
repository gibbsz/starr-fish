#!/bin/bash
#SBATCH --job-name=sfv8_copy_number
#SBATCH --partition=cpu
#SBATCH --account=brlab
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --output=/gpfs/commons/groups/ren_lab/guojiezhong/starr-fish/revision/origin_vs_new/results/logs/copy_number_%x_%j.out
#SBATCH --error=/gpfs/commons/groups/ren_lab/guojiezhong/starr-fish/revision/origin_vs_new/results/logs/copy_number_%x_%j.err
#
# Posterior mean latent AAV copy number E[k_ij | t7_ij, cre_ij] for every
# (cell, cCRE) pair, from a fit that saved log_rho and log_a.
#
#   sbatch --job-name=cn_origin submit_copy_number_matrix.slurm origin
#   sbatch --job-name=cn_new    submit_copy_number_matrix.slurm new
#
# Writes <outdir>/copy_number.npz (mean + sd, float32) and
# <outdir>/copy_number_copies.csv.gz (the mean, cells x cCREs).
#
# CPU only, on purpose: this streams ~160M pairs in chunks and never touches a
# GPU. Cost scales with the posterior, not the data -- the all-zero baseline
# alone is n_cell_types x n_cre patterns -- so the posterior is thinned to 200
# evenly-spaced draws, which is within Monte Carlo error of the full 1,000 for a
# posterior *mean*.

set -euo pipefail

DATASET="${1:?usage: submit_copy_number_matrix.slurm <origin|new>}"

REPO=/gpfs/commons/groups/ren_lab/guojiezhong/starr-fish
PYTHON=/gpfs/commons/home/guojiezhong/miniconda3/envs/scvi/bin/python
ANALYSIS="$REPO/revision/origin_vs_new"
MAX_DRAWS=${MAX_DRAWS:-200}

case "$DATASET" in
  origin)
    FIT="$ANALYSIS/results/origin/bayesian_copy_number"
    H5AD="$REPO/revision/Data/scdata_5_28_2025_BRBB500gn_final_CRE_T7CRE_NEWNEW.h5ad"
    OUTDIR="$ANALYSIS/results/origin/copy_number"
    ;;
  new)
    FIT="$ANALYSIS/results/new/bayesian_copy_number"
    H5AD="$REPO/revision/Data/scdata_07_29_2026_SFv8_low_dose_final_CRE_T7.h5ad"
    OUTDIR="$ANALYSIS/results/new/copy_number"
    ;;
  *)
    echo "unknown dataset '$DATASET' (expected 'origin' or 'new')" >&2
    exit 2
    ;;
esac

if [[ -e "$OUTDIR/copy_number_copies.csv.gz" ]]; then
  echo "refusing to overwrite $OUTDIR/copy_number_copies.csv.gz" >&2
  exit 3
fi

export MPLBACKEND=Agg
export XDG_CACHE_HOME="${TMPDIR:-/tmp}/xdg-${SLURM_JOB_ID:-local}"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4

mkdir -p "$XDG_CACHE_HOME" "$OUTDIR" "$ANALYSIS/results/logs"

cd "$REPO"
echo "dataset=$DATASET"
echo "fit=$FIT"
echo "outdir=$OUTDIR"
echo "max_draws=$MAX_DRAWS"
echo "start=$(date --iso-8601=seconds) host=$(hostname)"

"$PYTHON" -m baystarrfish copy-number \
  --fit-dir "$FIT" \
  --h5ad "$H5AD" \
  --out "$OUTDIR/copy_number.npz" \
  --csv "$OUTDIR/copy_number" \
  --csv-which copies \
  --csv-decimals 5 \
  --with-sd \
  --max-draws "$MAX_DRAWS"

ls -la "$OUTDIR"
echo "end=$(date --iso-8601=seconds)"
