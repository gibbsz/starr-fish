#!/bin/bash
#SBATCH --job-name=sf_stabilised_density
#SBATCH --partition=cpu
#SBATCH --account=brlab
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --output=/gpfs/commons/groups/ren_lab/guojiezhong/starr-fish/revision/spatial_density/results/logs/%x_%j.out
#SBATCH --error=/gpfs/commons/groups/ren_lab/guojiezhong/starr-fish/revision/spatial_density/results/logs/%x_%j.err
#
# Density-stabilised activity maps for every cCRE. Visualisation only -- no
# permutations, no p-values, so this is minutes of compute rather than hours.
#
# R = D/N is already density-unbiased (a weighted mean, not a sum). What is
# corrected here is the residual density *noise*: the sampling variance of R
# falls with local effective sample size, so thin areas throw extreme values by
# chance. activity_density.stabilise removes that in closed form by
# empirical-Bayes shrinkage against the exact per-pixel variance the geometry
# already carries.
#
# One pass per bandwidth, because the correction's size depends on it. Measured
# on CRE138 sec2: mean shrinkage weight 0.825 at h=50 rising to 0.994 at h=400,
# pulling the peak from 5.17x baseline to 3.79x at h=50 but leaving it untouched
# at h=400. Small bandwidths are where this matters.
#
# Panels 3 and 4 are relative to each cCRE's own section baseline, which is what
# the shrinkage targets -- so a pixel with no local evidence sits at 1.0 and
# reads as "nothing to say" instead of as whatever the section average happens to
# be. Absolute baselines vary hugely between cCREs (11.7x the negative-control
# level for CRE138, 0.33x for CRE174) and are recorded as baseline_mean.
#
# Cell density is normalised out. Cell-type composition is NOT: activity differs
# by subclass and subclasses are spatially organised, so structure that survives
# may still be cell-type anatomy. Nothing here is a significance claim.

set -euo pipefail

REPO=/gpfs/commons/groups/ren_lab/guojiezhong/starr-fish
WORKFLOW="$REPO/revision/spatial_density"
CODE="$WORKFLOW/code"
PYTHON=/gpfs/commons/home/guojiezhong/miniconda3/envs/scvi/bin/python

BANDWIDTHS=${BANDWIDTHS:-"100 200 400"}
MIN_T7=${MIN_T7:-50}
TRANSFORM=${TRANSFORM:-log}
JOBS=${JOBS:-${SLURM_CPUS_PER_TASK:-8}}
T7_TOTALS="$WORKFLOW/results/subclass_cre_t7_totals.csv.gz"

export MPLBACKEND=Agg
export MPLCONFIGDIR="${TMPDIR:-/tmp}/mpl-${SLURM_JOB_ID:-local}"
export XDG_CACHE_HOME="${TMPDIR:-/tmp}/xdg-${SLURM_JOB_ID:-local}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

mkdir -p "$MPLCONFIGDIR" "$XDG_CACHE_HOME" "$WORKFLOW/results/logs"
cd "$REPO"
echo "start=$(date --iso-8601=seconds) host=$(hostname) jobs=$JOBS min_t7=$MIN_T7"

if [ ! -f "$T7_TOTALS" ]; then
    echo "=== building T7 totals $(date --iso-8601=seconds) ==="
    "$PYTHON" "$CODE/make_t7_totals.py"
fi

for H in $BANDWIDTHS; do
    OUTDIR="$WORKFLOW/results/stabilised/h${H}"
    FIGDIR="$WORKFLOW/figures/stabilised/h${H}"
    mkdir -p "$OUTDIR" "$FIGDIR"
    echo "=== h=${H} $(date --iso-8601=seconds) ==="
    "$PYTHON" "$CODE/plot_stabilised_density.py" \
        --bandwidth "$H" \
        --min-t7 "$MIN_T7" \
        --t7-totals "$T7_TOTALS" \
        --mark-transform "$TRANSFORM" \
        --jobs "$JOBS" \
        --outdir "$OUTDIR" \
        --figdir "$FIGDIR" \
        --overwrite
done

echo "end=$(date --iso-8601=seconds)"
