#!/bin/bash
#SBATCH --job-name=sfv8_ccre_spatial
#SBATCH --partition=cpu
#SBATCH --account=brlab
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=4:00:00
#SBATCH --output=/gpfs/commons/groups/ren_lab/guojiezhong/starr-fish/revision/run_Bayes/logs/ccre_spatial_%x_%j.out
#SBATCH --error=/gpfs/commons/groups/ren_lab/guojiezhong/starr-fish/revision/run_Bayes/logs/ccre_spatial_%x_%j.err
#
# One normalised-activity spatial map per cCRE in the published t7_ge50 heatmap.
#
#   sbatch submit_ccre_spatial_sweep.slurm            # old dataset, every cCRE
#   sbatch submit_ccre_spatial_sweep.slurm --limit 5  # smoke test
#
# Two stages, gated by set -e: the unit suite first, so a broken
# highlight/exclude path aborts in ~3 minutes instead of after a two-hour sweep,
# then the sweep itself. Extra arguments are forwarded to the sweep.
#
# CPU only. The copy-number reconstruction runs once over the ~219 columns the
# heatmap needs (of 389) and then every figure is drawn from it, so the cost is
# one posterior pass (~15 min at 200 draws) plus ~10-20 s per figure at 408,621
# cells. 209 of the heatmap's 212 cCREs have at least one cell type that is both
# defined and has a control reference; the other 3 are skipped and logged.
#
# Figures already on disk are skipped unless --overwrite, so a job that hits the
# wall clock can simply be resubmitted. Measured on the first full run: 5.9 GB
# peak RSS, ~15 min for the posterior pass, then ~2.5 s per figure -- 123 of 209
# cCREs in the first 5 minutes of rendering.

set -euo pipefail

REPO=/gpfs/commons/groups/ren_lab/guojiezhong/starr-fish
LOGS="$REPO/revision/run_Bayes/logs"
PYTHON=/gpfs/commons/home/guojiezhong/miniconda3/envs/scvi/bin/python
SWEEP="$REPO/revision/Bayes_OldData/visualization/plot_all_ccre_spatial.py"
OUTDIR="$REPO/revision/Bayes_OldData/visualization/figures"

export MPLBACKEND=Agg
export XDG_CACHE_HOME="${TMPDIR:-/tmp}/xdg-${SLURM_JOB_ID:-local}"
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8

mkdir -p "$XDG_CACHE_HOME" "$OUTDIR" "$LOGS"

cd "$REPO"
echo "start=$(date --iso-8601=seconds) host=$(hostname)"
echo "outdir=$OUTDIR"
echo "extra args: $*"

echo "--- stage 1/2: unit suite ---"
"$PYTHON" -m pytest tests -q -m 'not slow'

echo "--- stage 2/2: cCRE sweep ---"
"$PYTHON" "$SWEEP" --outdir "$OUTDIR" "$@"

echo "figures: $(find "$OUTDIR" -name '*.png' | wc -l) png, $(find "$OUTDIR" -name '*.pdf' | wc -l) pdf"
echo "end=$(date --iso-8601=seconds)"
