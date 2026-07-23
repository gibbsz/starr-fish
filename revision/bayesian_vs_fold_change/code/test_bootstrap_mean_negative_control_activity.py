#!/usr/bin/env python3
"""Test bootstrap cCRE activity against the replicate-wise mean controls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analysis_utils import ANALYSIS_DIR, DEFAULT_H5AD, write_json
from compute_t7_filter_negative_control_stats import (
    aligned_t7_totals,
    bh_fdr,
    read_negative_controls,
)
from plot_method_activity_correlation import read_cre_blacklist


METHOD = "Bootstrap mean controls"
FILTERED_METHOD = "Bootstrap mean controls (control T7>=50)"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bootstrap-dir",
        type=Path,
        default=ANALYSIS_DIR / "results" / "bootstrap",
    )
    parser.add_argument("--h5ad", type=Path, default=DEFAULT_H5AD)
    parser.add_argument("--t7-threshold", type=float, default=50.0)
    parser.add_argument("--q-cutoff", type=float, default=0.05)
    parser.add_argument(
        "--individual-control-t7-threshold",
        type=float,
        default=None,
        help="Average only controls meeting this cell-type-specific T7 threshold.",
    )
    parser.add_argument("--chunk-size", type=int, default=250)
    parser.add_argument(
        "--tables-dir", type=Path, default=ANALYSIS_DIR / "results" / "tables"
    )
    parser.add_argument(
        "--figures-dir", type=Path, default=ANALYSIS_DIR / "results" / "figures"
    )
    parser.add_argument("--stem", default="bootstrap_mean_negative_control_tests")
    return parser.parse_args()


def compute_statistics(
    activity_array: np.ndarray,
    negative_mask: np.ndarray,
    control_include: np.ndarray,
    chunk_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_boot, n_groups, n_cres = activity_array.shape
    activity_sum = np.zeros((n_groups, n_cres), dtype=np.float64)
    activity_count = np.zeros((n_groups, n_cres), dtype=np.int64)
    effect_sum = np.zeros((n_groups, n_cres), dtype=np.float64)
    effect_count = np.zeros((n_groups, n_cres), dtype=np.int64)
    less_equal_count = np.zeros((n_groups, n_cres), dtype=np.int64)
    controls_used_sum = np.zeros(n_groups, dtype=np.float64)
    reference_count = np.zeros(n_groups, dtype=np.int64)

    for start in range(0, n_boot, chunk_size):
        stop = min(start + chunk_size, n_boot)
        with np.errstate(divide="ignore", invalid="ignore"):
            logged = np.log(
                activity_array[start:stop].astype(np.float64, copy=False)
            )
        finite = np.isfinite(logged)
        activity_sum += np.where(finite, logged, 0.0).sum(axis=0)
        activity_count += finite.sum(axis=0)

        control_logged = logged[:, :, negative_mask]
        control_finite = np.isfinite(control_logged) & control_include[None, :, :]
        control_count = control_finite.sum(axis=2)
        control_sum = np.where(control_finite, control_logged, 0.0).sum(axis=2)
        reference = np.divide(
            control_sum,
            control_count,
            out=np.full(control_sum.shape, np.nan, dtype=np.float64),
            where=control_count > 0,
        )
        reference_finite = np.isfinite(reference)
        controls_used_sum += np.where(
            reference_finite, control_count, 0
        ).sum(axis=0)
        reference_count += reference_finite.sum(axis=0)

        testable = finite & reference_finite[:, :, None]
        effect = logged - reference[:, :, None]
        effect_sum += np.where(testable, effect, 0.0).sum(axis=0)
        effect_count += testable.sum(axis=0)
        less_equal_count += ((effect <= 0.0) & testable).sum(axis=0)

    activity_mean = np.divide(
        activity_sum,
        activity_count,
        out=np.full(activity_sum.shape, np.nan),
        where=activity_count > 0,
    )
    effect_mean = np.divide(
        effect_sum,
        effect_count,
        out=np.full(effect_sum.shape, np.nan),
        where=effect_count > 0,
    )
    p_right = np.divide(
        less_equal_count,
        effect_count,
        out=np.full(effect_sum.shape, np.nan),
        where=effect_count > 0,
    )
    mean_controls_used = np.divide(
        controls_used_sum,
        reference_count,
        out=np.full(controls_used_sum.shape, np.nan),
        where=reference_count > 0,
    )
    return (
        activity_mean,
        effect_mean,
        p_right,
        effect_count,
        mean_controls_used,
        reference_count,
    )


def main() -> None:
    args = parse_args()
    args.tables_dir.mkdir(parents=True, exist_ok=True)
    args.figures_dir.mkdir(parents=True, exist_ok=True)
    axes = json.loads((args.bootstrap_dir / "bootstrap_axes.json").read_text())
    groups = pd.Index(axes["subclasses"], dtype=str)
    cres = pd.Index(axes["cres"], dtype=str)
    negative_controls = read_negative_controls(args.bootstrap_dir)
    negative_mask = cres.isin(negative_controls)
    if int(negative_mask.sum()) != 7:
        raise ValueError(f"Expected seven negative controls; found {negative_mask.sum()}")
    blacklist = read_cre_blacklist(args.bootstrap_dir)

    activity_array = np.load(
        args.bootstrap_dir / "celltype_activity_array.npy", mmap_mode="r"
    )
    if activity_array.shape[1:] != (len(groups), len(cres)):
        raise ValueError("Bootstrap activity array does not match saved axes")
    pair_t7 = aligned_t7_totals(args.h5ad, groups, cres)
    control_pair_t7 = pair_t7.loc[:, negative_controls].to_numpy(float)
    if args.individual_control_t7_threshold is None:
        control_include = np.ones(control_pair_t7.shape, dtype=bool)
        method = METHOD
    else:
        control_include = control_pair_t7 >= args.individual_control_t7_threshold
        method = FILTERED_METHOD
    (
        activity_mean,
        effect_mean,
        p_right,
        n_testable,
        mean_controls_used,
        reference_count,
    ) = compute_statistics(
        activity_array, negative_mask, control_include, args.chunk_size
    )

    control_t7 = pair_t7.loc[:, negative_controls].sum(axis=1).to_numpy(float)
    n_controls_passing_t7 = control_include.sum(axis=1)
    group_grid, cre_grid = np.meshgrid(
        groups.to_numpy(str), cres.to_numpy(str), indexing="ij"
    )
    is_negative = np.isin(cre_grid.ravel(), negative_controls)
    is_blacklisted = np.isin(cre_grid.ravel(), list(blacklist))
    target_t7 = pair_t7.to_numpy(float).ravel()
    control_t7_flat = np.repeat(control_t7, len(cres))
    valid = (
        (target_t7 >= args.t7_threshold)
        & (control_t7_flat >= args.t7_threshold)
        & ~is_negative
        & ~is_blacklisted
        & np.isfinite(p_right.ravel())
    )
    tests = pd.DataFrame(
        {
            "t7_threshold": float(args.t7_threshold),
            "method": method,
            "group": group_grid.ravel()[valid],
            "cre": cre_grid.ravel()[valid],
            "target_t7_total": target_t7[valid],
            "negative_control_t7_total": control_t7_flat[valid],
            "activity_mean": activity_mean.ravel()[valid],
            "effect_vs_mean_control_mean": effect_mean.ravel()[valid],
            "posterior_probability_above_mean_control": 1.0
            - p_right.ravel()[valid],
            "p_right": p_right.ravel()[valid],
            "n_testable_bootstraps": n_testable.ravel()[valid],
            "mean_controls_used_per_reference": np.repeat(
                mean_controls_used, len(cres)
            )[valid],
            "n_controls_passing_t7": np.repeat(
                n_controls_passing_t7, len(cres)
            )[valid],
            "n_bootstraps_with_control_reference": np.repeat(
                reference_count, len(cres)
            )[valid],
        }
    )
    tests["q_right"] = bh_fdr(tests["p_right"].to_numpy(float))
    tests["significant_q"] = tests["q_right"].le(args.q_cutoff)

    table_path = args.tables_dir / f"{args.stem}.csv.gz"
    tests.to_csv(table_path, index=False)
    significant = tests.loc[tests["significant_q"]]
    manifest_path = args.figures_dir / f"{args.stem}_manifest.json"
    write_json(
        manifest_path,
        {
            "method": method,
            "bootstrap_dir": str(args.bootstrap_dir),
            "negative_controls": negative_controls,
            "activity_definition": "log bootstrap activity; alpha is not subtracted",
            "contrast_definition": (
                "target log activity minus the arithmetic mean of finite log activities "
                "among the cell-type-specific retained ordinary controls within each "
                "bootstrap replicate"
            ),
            "p_right_definition": "fraction of valid bootstrap contrasts <= 0",
            "multiple_testing": "BH across all eligible target-cell-type pairs",
            "t7_threshold": args.t7_threshold,
            "individual_control_t7_threshold": args.individual_control_t7_threshold,
            "minimum_retained_controls": 1,
            "q_cutoff": args.q_cutoff,
            "n_bootstraps": int(activity_array.shape[0]),
            "counts": {
                "eligible_tests": int(len(tests)),
                "significant_tests": int(len(significant)),
                "significant_ccres": int(significant["cre"].nunique()),
                "significant_cell_types": int(significant["group"].nunique()),
            },
            "outputs": {"tests": str(table_path), "manifest": str(manifest_path)},
        },
    )
    print(
        json.dumps(
            {
                "eligible_tests": int(len(tests)),
                "significant_tests": int(len(significant)),
                "minimum_q": float(tests["q_right"].min()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
