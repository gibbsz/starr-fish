#!/bin/bash
#SBATCH --job-name=sfv8_bayes_cn
#SBATCH --partition=gpu
#SBATCH --account=brlab
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=/gpfs/commons/groups/ren_lab/guojiezhong/starr-fish/revision/run_Bayes/logs/bayes_cn_%x_%j.out
#SBATCH --error=/gpfs/commons/groups/ren_lab/guojiezhong/starr-fish/revision/run_Bayes/logs/bayes_cn_%x_%j.err
#
# Refit the production joint+dropout model on the origin and new datasets,
# saving log_rho and log_a alongside log_gamma so the latent AAV copy number is
# recoverable (baystarrfish.inference.copy_number). The superseded fits stored
# log_gamma alone and those sites cannot be rebuilt after the fact.
#
# Every other setting replicates those fits exactly -- same model, same data,
# same seed -- so only the saved posterior sites differ. The activity summaries
# agree to ~1e-13 relative, which is why these fits have since replaced the
# log_gamma-only ones under revision/Bayes_{Old,New}Data/bayesian/.
#
#   sbatch --job-name=bayes_cn_old submit_bayesian_copy_number.slurm old
#   sbatch --job-name=bayes_cn_new submit_bayesian_copy_number.slurm new
#
# Measured: 12m46s (408,621 cells) and 10m19s (353,034), 6.4 GB peak RSS, on an
# L40S. The collapsed representation is ~300k rows, so 30,000 SVI steps is
# minutes; the resource request is sized from that, not from a guessed ceiling.
#
# This is now a reproduction recipe: both target directories already hold a
# completed fit, so the guard below refuses to start. Move the existing fit aside
# if you really mean to refit.

set -euo pipefail

DATASET="${1:?usage: submit_bayesian_copy_number.slurm <old|new>}"

REPO=/gpfs/commons/groups/ren_lab/guojiezhong/starr-fish
LOGS="$REPO/revision/run_Bayes/logs"
PYTHON=/gpfs/commons/home/guojiezhong/miniconda3/envs/bayes-jax/bin/python
OLD_RESULTS="$REPO/revision/Bayes_OldData"
NEW_RESULTS="$REPO/revision/Bayes_NewData"

# REFERENCE is an independent fit whose cCRE axis this one must reproduce. It is
# empty where no such fit survives -- the log_gamma-only fits these replaced were
# deleted once the refits took over the canonical bayesian/ name.
case "$DATASET" in
  old|origin)
    H5AD="$REPO/revision/Data/scdata_5_28_2025_BRBB500gn_final_CRE_T7CRE_NEWNEW.h5ad"
    OUTDIR="$OLD_RESULTS/bayesian"
    REFERENCE="$REPO/revision/bayesian_vs_fold_change/results/bayesian"
    ;;
  new)
    H5AD="$REPO/revision/Data/scdata_07_29_2026_SFv8_low_dose_final_CRE_T7.h5ad"
    OUTDIR="$NEW_RESULTS/bayesian"
    REFERENCE=""
    ;;
  *)
    echo "unknown dataset '$DATASET' (expected 'old' or 'new')" >&2
    exit 2
    ;;
esac

# Never clobber a completed fit. Cheap to recompute (~13 min) but the outputs
# are cited downstream, so replacing one should be a deliberate act.
if [[ -e "$OUTDIR/run_manifest.json" ]]; then
  echo "refusing to overwrite the completed fit in $OUTDIR" >&2
  echo "move or delete it first if you really mean to refit" >&2
  exit 3
fi

export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export MPLBACKEND=Agg
export XDG_CACHE_HOME="${TMPDIR:-/tmp}/xdg-${SLURM_JOB_ID:-local}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

mkdir -p "$XDG_CACHE_HOME" "$OUTDIR" "$LOGS"

cd "$REPO"
echo "dataset=$DATASET"
echo "input=$H5AD"
echo "outdir=$OUTDIR"
echo "start=$(date --iso-8601=seconds) host=$(hostname)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
"$PYTHON" -c "import jax; print(jax.devices()); assert any(d.platform == 'gpu' for d in jax.devices())"
"$PYTHON" -c "import baystarrfish; print('baystarrfish', baystarrfish.__version__)"

# Settings replicate the existing fits exactly. --posterior-sites is stated
# explicitly rather than relying on the package default, so this script keeps
# reproducing the same output if that default ever changes again.
"$PYTHON" revision/run_Bayes/run_bayes.py \
  --h5ad "$H5AD" \
  --outdir "$OUTDIR" \
  --section all \
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
  --posterior-sites log_gamma log_rho log_a \
  --seed 0

# The cCRE axis must match the fit this one is meant to be comparable with.
if [[ -n "$REFERENCE" ]]; then
  cmp "$REFERENCE/cre_blacklist.csv" "$OUTDIR/cre_blacklist.csv"
  cmp "$REFERENCE/negative_controls.csv" "$OUTDIR/negative_controls.csv"
else
  echo "no independent reference fit for '$DATASET'; skipping cCRE axis check"
fi

"$PYTHON" - "$OUTDIR" <<'PY'
import json, sys
import numpy as np
from pathlib import Path

root = Path(sys.argv[1])
tag = json.loads((root / "run_manifest.json").read_text())["tag"]
with np.load(root / f"{tag}_posterior_samples.npz", allow_pickle=True) as handle:
    sites = sorted(set(handle.files) - {"group_names", "cre_names"})
print("saved posterior sites:", sites)
missing = [s for s in ("log_gamma", "log_rho", "log_a") if s not in sites]
if missing:
    raise SystemExit(f"FAILED: copy-number sites missing from the fit: {missing}")
print("copy-number reconstruction is possible from this fit")
PY

echo "end=$(date --iso-8601=seconds)"
