#!/bin/bash
#SBATCH --job-name=sf_activity_density
#SBATCH --partition=cpu
#SBATCH --account=brlab
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=16:00:00
#SBATCH --output=/gpfs/commons/groups/ren_lab/guojiezhong/starr-fish/revision/spatial_density/results/logs/%x_%j.out
#SBATCH --error=/gpfs/commons/groups/ren_lab/guojiezhong/starr-fish/revision/spatial_density/results/logs/%x_%j.err
#
# Spatial activity-density hotspots, both nulls, end to end.
#
# Per null: screen + delineate, characterise, figures. Then one calibration run
# on shuffled marks. Everything lands under results/<null><TAG_SUFFIX>/ and
# figures/<null><TAG_SUFFIX>/, so a MIN_T7 sweep never overwrites a previous run.
#
# MIN_T7 -- the correction that matters. A (subclass, cCRE) pair carrying almost
# no T7 signal has no estimable activity: the per-cell posterior falls back on
# the prior and returns a near-constant subclass-level value. The first run
# (MIN_T7=0) included those cells and the result was dominated by them --
# 18 of 20 within_subclass regions landed on one olfactory-bulb patch driven by
# `OB-out Frmd7 Gaba`, whose T7 totals for the implicated cCREs were 44, 6, 8
# and 1. A negative control appeared among the hits, the permutation threshold
# sat near 54 instead of the ~4 theory predicts, and nothing was detectable
# below h=400. All four symptoms are the same cause: prior-driven cells forming
# permutation blocks that cannot randomise. MIN_T7=50 (the threshold the
# upstream fit uses) removes them and drops max_z from ~55 to ~30.
#
# Run BOTH nulls and read them together:
#
#   global           activity reassigned to any cell -- the literal
#                    "randomly assign the estimates" expectation. Ignores that
#                    cell types are spatially organised and differ in activity,
#                    so at MIN_T7=0 it flagged 778/778 pairs including all 14
#                    negative-control pairs. Read a region here as "this area is
#                    active", not "specifically active".
#
#   within_subclass  activity reassigned only among cells of the same subclass,
#                    holding composition fixed. Regions mean "more active than
#                    the same cell types are elsewhere".
#
# Measured cost (16 cores, 999 permutations, 3 bandwidths, 389 cCREs x 2
# sections): screening ~9 min per null; delineation scales with how many pairs
# pass, so a saturated global null dominates at a few hours. Peak RSS is the
# 451 MB activity matrix shared copy-on-write plus ~400 MB of retained permuted
# surfaces per region worker.

set -euo pipefail

REPO=/gpfs/commons/groups/ren_lab/guojiezhong/starr-fish
WORKFLOW="$REPO/revision/spatial_density"
CODE="$WORKFLOW/code"
PYTHON=/gpfs/commons/home/guojiezhong/miniconda3/envs/scvi/bin/python

PERMUTATIONS=${PERMUTATIONS:-999}
BOOTSTRAP=${BOOTSTRAP:-200}
MAX_REGIONS=${MAX_REGIONS:-8}
BANDWIDTHS=${BANDWIDTHS:-"100 200 400"}
JOBS=${JOBS:-${SLURM_CPUS_PER_TASK:-8}}
REGION_JOBS=${REGION_JOBS:-$JOBS}
NULLS=${NULLS:-"within_subclass global"}

# Per-(subclass, cCRE) T7 admissibility. MIN_T7=0 reproduces the first run;
# MIN_T7=50 matches the threshold the upstream fit uses. Results land in a
# separate TAG-suffixed folder either way, so runs never overwrite each other.
MIN_T7=${MIN_T7:-50}
T7_TOTALS="$WORKFLOW/results/subclass_cre_t7_totals.csv.gz"
if [ "${MIN_T7%.*}" -gt 0 ] 2>/dev/null; then
    TAG_SUFFIX="_t7ge${MIN_T7%.*}"
else
    TAG_SUFFIX=""
fi

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

for NULL in $NULLS; do
    TAG="${NULL}${TAG_SUFFIX}"
    OUTDIR="$WORKFLOW/results/$TAG"
    FIGDIR="$WORKFLOW/figures/$TAG"
    mkdir -p "$OUTDIR/regions" "$OUTDIR/surfaces" "$FIGDIR"

    echo "=== [$TAG] screen + delineate $(date --iso-8601=seconds) ==="
    "$PYTHON" "$CODE/run_activity_density_hotspots.py" \
        --outdir "$OUTDIR" \
        --null "$NULL" \
        --min-t7 "$MIN_T7" \
        --t7-totals "$T7_TOTALS" \
        --bandwidths $BANDWIDTHS \
        --permutations "$PERMUTATIONS" \
        --bootstrap "$BOOTSTRAP" \
        --max-regions "$MAX_REGIONS" \
        --jobs "$JOBS" \
        --region-jobs "$REGION_JOBS"

    echo "=== [$TAG] characterise $(date --iso-8601=seconds) ==="
    "$PYTHON" "$CODE/characterise_regions.py" --results "$OUTDIR"

    echo "=== [$TAG] figures $(date --iso-8601=seconds) ==="
    "$PYTHON" "$CODE/plot_activity_density_hotspots.py" \
        --results "$OUTDIR" --figdir "$FIGDIR"
done

echo "=== calibration check $(date --iso-8601=seconds) ==="
# The marks are shuffled once up front, so nothing is left to find. p_fwer must
# come out roughly uniform and no region may be emitted; if this stage reports
# regions, the standardisation or the tissue mask is wrong and the real runs
# above cannot be trusted.
"$PYTHON" "$CODE/run_activity_density_hotspots.py" \
    --outdir "$WORKFLOW/results/calibration${TAG_SUFFIX}" \
    --permute-observed \
    --min-t7 "$MIN_T7" \
    --t7-totals "$T7_TOTALS" \
    --bandwidths $BANDWIDTHS \
    --permutations "$PERMUTATIONS" \
    --jobs "$JOBS" \
    --skip-regions

echo "end=$(date --iso-8601=seconds)"
