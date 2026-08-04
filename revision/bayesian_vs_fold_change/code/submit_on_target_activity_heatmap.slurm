#!/bin/bash
#SBATCH --job-name=brbb_on_target_heatmap
#SBATCH --partition=cpu
#SBATCH --account=brlab
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=2:00:00
#SBATCH --output=/gpfs/commons/groups/ren_lab/guojiezhong/starr-fish/revision/bayesian_vs_fold_change/results/logs/%x_%j.out
#SBATCH --error=/gpfs/commons/groups/ren_lab/guojiezhong/starr-fish/revision/bayesian_vs_fold_change/results/logs/%x_%j.err

set -euo pipefail

REPO=/gpfs/commons/groups/ren_lab/guojiezhong/starr-fish
ANALYSIS="$REPO/revision/bayesian_vs_fold_change"
PYTHON=${PYTHON:-/gpfs/commons/home/guojiezhong/miniconda3/envs/scvi/bin/python}

export MPLBACKEND=Agg
export MPLCONFIGDIR="${TMPDIR:-/tmp}/matplotlib-${SLURM_JOB_ID:-local}"
export XDG_CACHE_HOME="${TMPDIR:-/tmp}/xdg-${SLURM_JOB_ID:-local}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

mkdir -p "$MPLCONFIGDIR" "$XDG_CACHE_HOME" "$ANALYSIS/results/logs"

cd "$REPO"

# On-target subset of joint_dropout_activity_heatmap_t7_ge50_joint_plus_dropout.pdf:
# keep cCREs with >=1 ATAC-positive significant subclass, then keep those subclasses.
"$PYTHON" revision/bayesian_vs_fold_change/code/plot_method_activity_heatmap.py \
  --methods joint_dropout \
  --joint-dropout-bayesian-dir "$ANALYSIS/results/bayesian" \
  --activity-calibration none \
  --activity-centering none \
  --center-by-mean-negative-controls \
  --append-individual-negative-controls \
  --cmap coolwarm \
  --t7-threshold 50 \
  --filter-variants t7_ge_threshold \
  --significance-tests "$ANALYSIS/results/tables/joint_dropout_direct_activity_mean_negative_control_tests_t7_series.csv.gz" \
  --significance-q-cutoff 0.05 \
  --atac-peaks "$REPO/STARRFISH_in_vivo/Data/cre_atac_peaks.csv" \
  --restrict-to-on-target \
  --stem joint_dropout_activity_heatmap_on_target
