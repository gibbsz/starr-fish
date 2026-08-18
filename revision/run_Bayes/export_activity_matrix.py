#!/usr/bin/env python3
"""Export one dataset's posterior as the canonical activity matrix and test table.

Every activity heatmap in the revision plots the same quantity: the posterior
mean target ``log_gamma`` minus the posterior mean of the seven ordinary
negative controls. This script computes it once per dataset, on the full
unfiltered universe, and writes it next to the posterior it came from so the
plotting scripts only have to read and mask it:

    <dataset>/tables/<stem>_activity_matrix.csv.gz
        subclass x cCRE, the value the heatmap colours
    <dataset>/tables/<stem>_target_t7_matrix.csv.gz
        subclass x cCRE target T7 totals, the source of every T7 mask
    <dataset>/tables/<stem>_negative_control_activity_matrix.csv.gz
        subclass x 7 controls, posterior-mean log_gamma, for the control-spread strip
    <dataset>/tables/<stem>_significance.csv.gz
        one row per pair: p_right, the own-universe BH q, and the effect interval
    <dataset>/tables/<stem>_matrix_manifest.json
        provenance

``p_right`` is universe-free, so it is stored as computed. BH ``q`` is not: it
depends on which pairs are in the family, so the shipped ``q_right_t7_ge<k>``
column is BH over this dataset's own T7 >= k pairs, and any figure plotting a
different universe -- a shared original/new universe, for instance -- must re-run
BH over that universe from ``p_right``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
BVFC_CODE = REPO_ROOT / "revision" / "bayesian_vs_fold_change" / "code"
for path in (HERE, BVFC_CODE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from baystarrfish.stats import bh_fdr, negative_control_test  # noqa: E402
from test_individual_negative_control_loo_empirical_fdr import (  # noqa: E402
    POOLED_NAME,
    load_grouped_t7,
)
from activity_matrix_io import (  # noqa: E402
    ACTIVITY_COLUMN,
    DEFAULT_STEM,
    matrix_paths,
    stem_for,
)

N_ORDINARY_CONTROLS = 7
SIGNIFICANCE_COLUMNS = (
    "group",
    "cre",
    "class",
    "n_cells",
    "target_t7_total",
    "negative_control_t7_total",
    "n_negative_controls",
    "activity_mean",
    "mean_negative_control_activity_mean",
    "negative_control_activity_sd_mean",
    "control_reference_activity_mean",
    ACTIVITY_COLUMN,
    "effect_vs_control_reference_mean",
    "effect_vs_control_reference_lo90",
    "effect_vs_control_reference_hi90",
    "posterior_probability_above_control_reference",
    "p_right",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bayes-dir",
        type=Path,
        required=True,
        help="Posterior directory holding *_posterior_samples.npz and run_manifest.json.",
    )
    parser.add_argument(
        "--h5ad",
        type=Path,
        required=True,
        help="Dataset the posterior was fitted on; supplies the T7 totals.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=None,
        help="Destination (default: the posterior directory's sibling tables/).",
    )
    parser.add_argument("--stem", default=DEFAULT_STEM)
    parser.add_argument(
        "--q-t7-threshold",
        type=float,
        default=50.0,
        help="Own-dataset universe used for the shipped BH q column.",
    )
    parser.add_argument(
        "--control-sd-multiplier",
        type=float,
        default=0.0,
        help=(
            "Raise the control reference by k draw-wise SDs. 0 is the mean-control "
            "family; 1 writes the mean+1SD test table under a suffixed stem."
        ),
    )
    parser.add_argument("--q-cutoff", type=float, default=0.05)
    parser.add_argument(
        "--check-against",
        type=Path,
        default=None,
        help="Existing per-pair test table to cross-check the exported values against.",
    )
    return parser.parse_args()


def read_single_column(path: Path) -> list[str]:
    if not path.exists():
        return []
    return pd.read_csv(path).iloc[:, 0].astype(str).tolist()


def load_posterior(bayes_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, Path]:
    manifest_path = bayes_dir / "run_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing run manifest: {manifest_path}")
    tag = json.loads(manifest_path.read_text())["tag"]
    posterior_path = bayes_dir / f"{tag}_posterior_samples.npz"
    if not posterior_path.exists():
        raise FileNotFoundError(f"Missing posterior samples: {posterior_path}")
    with np.load(posterior_path, allow_pickle=True) as posterior:
        groups = posterior["group_names"].astype(str)
        all_cre = posterior["cre_names"].astype(str)
        ordinary = all_cre != POOLED_NAME
        log_gamma = posterior["log_gamma"][:, :, ordinary].astype(np.float32)
    return log_gamma, groups, all_cre[ordinary], posterior_path


def compute_full_tests(
    bayes_dir: Path, h5ad: Path, control_sd_multiplier: float
) -> tuple[pd.DataFrame, pd.DataFrame, Path]:
    """Run the mean-negative-control test with no T7 filter at all."""
    log_gamma, groups, cre_names, posterior_path = load_posterior(bayes_dir)
    controls = read_single_column(bayes_dir / "negative_controls.csv")
    blacklist = set(read_single_column(bayes_dir / "cre_blacklist.csv"))

    control_indices = np.flatnonzero(np.isin(cre_names, controls))
    if len(control_indices) != N_ORDINARY_CONTROLS:
        raise ValueError(
            f"Expected {N_ORDINARY_CONTROLS} ordinary negative controls; "
            f"found {len(control_indices)} in {bayes_dir}"
        )
    target_indices = np.flatnonzero(
        ~np.isin(cre_names, controls) & ~np.isin(cre_names, list(blacklist))
    )
    t7_totals, group_classes, group_cell_counts = load_grouped_t7(
        h5ad, groups, cre_names
    )
    tests = negative_control_test(
        log_gamma,
        groups,
        cre_names,
        target_indices,
        control_indices,
        t7_totals,
        group_classes,
        group_cell_counts,
        0.0,
        0.0,
        None,
        "Joint+dropout mean controls (unfiltered export)",
        control_sd_multiplier,
    )
    tests[ACTIVITY_COLUMN] = (
        tests["activity_mean"].astype(float)
        - tests["mean_negative_control_activity_mean"].astype(float)
    )
    control_activity = pd.DataFrame(
        log_gamma[:, :, control_indices].astype(np.float64).mean(axis=0),
        index=pd.Index(groups, name="subclass"),
        columns=cre_names[control_indices],
    ).sort_index(axis=0).sort_index(axis=1)
    return tests, control_activity, posterior_path


def add_own_universe_q(
    tests: pd.DataFrame, t7_threshold: float, q_cutoff: float
) -> tuple[pd.DataFrame, str, str, int]:
    """BH over this dataset's own T7 >= threshold pairs; NaN outside it."""
    token = f"{t7_threshold:g}".replace(".", "p")
    q_column = f"q_right_t7_ge{token}"
    call_column = f"significant_q_t7_ge{token}"
    eligible = tests["target_t7_total"].astype(float).ge(t7_threshold) & tests[
        "negative_control_t7_total"
    ].astype(float).ge(t7_threshold)
    tests[q_column] = np.nan
    if eligible.any():
        tests.loc[eligible, q_column] = bh_fdr(
            tests.loc[eligible, "p_right"].to_numpy(float)
        )
    tests[call_column] = tests[q_column].le(q_cutoff).fillna(False)
    return tests, q_column, call_column, int(eligible.sum())


def to_matrix(tests: pd.DataFrame, column: str) -> pd.DataFrame:
    matrix = tests.pivot(index="group", columns="cre", values=column)
    matrix.index.name = "subclass"
    matrix.columns.name = None
    return matrix.sort_index(axis=0).sort_index(axis=1)


def cross_check(tests: pd.DataFrame, reference: Path) -> dict[str, object]:
    other = pd.read_csv(reference)
    other["centered"] = (
        other["activity_mean"].astype(float)
        - other["mean_negative_control_activity_mean"].astype(float)
    )
    merged = tests.merge(
        other[["group", "cre", "centered", "p_right"]].rename(
            columns={"p_right": "p_right_reference"}
        ),
        on=["group", "cre"],
        how="inner",
    )
    if merged.empty:
        raise ValueError(f"No shared pairs with {reference}")
    return {
        "reference": str(reference),
        "shared_pairs": int(len(merged)),
        "max_abs_activity_difference": float(
            np.abs(merged[ACTIVITY_COLUMN] - merged["centered"]).max()
        ),
        "max_abs_p_right_difference": float(
            np.abs(merged["p_right"] - merged["p_right_reference"]).max()
        ),
    }


def main() -> None:
    args = parse_args()
    outdir = args.outdir or args.bayes_dir.parent / "tables"
    outdir.mkdir(parents=True, exist_ok=True)
    paths = matrix_paths(outdir, args.stem, args.control_sd_multiplier)

    tests, control_activity, posterior_path = compute_full_tests(
        args.bayes_dir, args.h5ad, args.control_sd_multiplier
    )
    tests, q_column, call_column, n_eligible = add_own_universe_q(
        tests, args.q_t7_threshold, args.q_cutoff
    )

    activity = to_matrix(tests, ACTIVITY_COLUMN)
    activity.to_csv(paths["activity"], float_format="%.10g")
    to_matrix(tests, "target_t7_total").to_csv(paths["target_t7"], float_format="%.10g")
    control_activity.to_csv(paths["negative_control_activity"], float_format="%.10g")

    columns = [
        column for column in SIGNIFICANCE_COLUMNS if column in tests.columns
    ] + [q_column, call_column]
    (
        tests[columns]
        .rename(columns={"group": "subclass"})
        .sort_values(["subclass", "cre"], kind="stable")
        .to_csv(paths["significance"], index=False, float_format="%.10g")
    )

    manifest = {
        "stem": stem_for(args.stem, args.control_sd_multiplier),
        "bayes_dir": str(args.bayes_dir.resolve()),
        "posterior": str(posterior_path.resolve()),
        "h5ad": str(args.h5ad.resolve()),
        "activity_definition": (
            "posterior mean target log_gamma minus the posterior mean of the seven "
            "ordinary negative controls; alpha is not subtracted"
        ),
        "control_sd_multiplier": float(args.control_sd_multiplier),
        "t7_filter": "none; every fitted target pair is exported",
        "q_column": q_column,
        "q_universe": (
            "BH over this dataset's pairs with target and pooled-control T7 >= "
            f"{args.q_t7_threshold:g}; NaN outside that universe"
        ),
        "q_cutoff": float(args.q_cutoff),
        "p_right_definition": "posterior fraction of draw-wise contrasts <= 0",
        "shape": {
            "subclasses": int(activity.shape[0]),
            "cres": int(activity.shape[1]),
            "pairs": int(len(tests)),
            "pairs_in_q_universe": n_eligible,
        },
        "outputs": {key: str(value) for key, value in paths.items() if key != "manifest"},
    }
    if args.check_against is not None:
        manifest["cross_check"] = cross_check(tests, args.check_against)
    paths["manifest"].write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
