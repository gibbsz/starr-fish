#!/usr/bin/env python3
"""Test raw cCRE activity against the draw-wise maximum ordinary control."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analysis_utils import ANALYSIS_DIR, DEFAULT_H5AD, write_json
from plot_method_activity_correlation import read_cre_blacklist
from baystarrfish.stats import bh_fdr
from test_individual_negative_control_loo_empirical_fdr import (
    POOLED_NAME,
    load_grouped_t7,
)


METHOD = "Joint+dropout max control"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bayes-dir",
        type=Path,
        default=ANALYSIS_DIR / "results" / "bayesian",
    )
    parser.add_argument("--h5ad", type=Path, default=DEFAULT_H5AD)
    parser.add_argument("--t7-threshold", type=float, default=50.0)
    parser.add_argument("--effect-threshold", type=float, default=0.0)
    parser.add_argument("--q-cutoff", type=float, default=0.05)
    parser.add_argument(
        "--tables-dir", type=Path, default=ANALYSIS_DIR / "results" / "tables"
    )
    parser.add_argument(
        "--figures-dir", type=Path, default=ANALYSIS_DIR / "results" / "figures"
    )
    parser.add_argument(
        "--stem", default="joint_dropout_direct_activity_max_negative_control_tests"
    )
    return parser.parse_args()




def compute_tests(
    log_gamma: np.ndarray,
    groups: np.ndarray,
    cre_names: np.ndarray,
    target_indices: np.ndarray,
    control_indices: np.ndarray,
    t7_totals: np.ndarray,
    group_classes: np.ndarray,
    group_cell_counts: np.ndarray,
    t7_threshold: float,
    effect_threshold: float,
) -> pd.DataFrame:
    records = []
    for group_idx, group in enumerate(groups):
        control_draws = log_gamma[:, group_idx, control_indices].astype(
            np.float64, copy=False
        )
        max_control_draws = control_draws.max(axis=1)
        control_t7_total = float(t7_totals[group_idx, control_indices].sum())
        eligible = (
            (t7_totals[group_idx, target_indices] >= t7_threshold)
            & (control_t7_total >= t7_threshold)
        )
        selected_indices = target_indices[eligible]
        if len(selected_indices) == 0:
            continue

        target_draws = log_gamma[:, group_idx, selected_indices].astype(
            np.float64, copy=False
        )
        contrasts = target_draws - max_control_draws[:, None] - effect_threshold
        target_mean = target_draws.mean(axis=0)
        max_control_mean = float(max_control_draws.mean())
        contrast_mean = contrasts.mean(axis=0)
        contrast_lo, contrast_hi = np.quantile(contrasts, [0.05, 0.95], axis=0)
        posterior_probability = (contrasts > 0.0).mean(axis=0)
        p_right = (contrasts <= 0.0).mean(axis=0)

        records.append(
            pd.DataFrame(
                {
                    "t7_threshold": float(t7_threshold),
                    "method": METHOD,
                    "group": group,
                    "class": group_classes[group_idx],
                    "cre": cre_names[selected_indices],
                    "n_cells": int(group_cell_counts[group_idx]),
                    "target_t7_total": t7_totals[group_idx, selected_indices],
                    "negative_control_t7_total": control_t7_total,
                    "n_negative_controls": len(control_indices),
                    "activity_mean": target_mean,
                    "max_negative_control_activity_mean": max_control_mean,
                    "effect_vs_max_control_mean": contrast_mean,
                    "effect_vs_max_control_lo90": contrast_lo,
                    "effect_vs_max_control_hi90": contrast_hi,
                    "posterior_probability_above_max_control": posterior_probability,
                    "p_right": p_right,
                }
            )
        )
    if not records:
        raise ValueError("No cCRE-cell-type pairs passed the T7 filters")
    output = pd.concat(records, ignore_index=True)
    output["q_right"] = bh_fdr(output["p_right"].to_numpy(float))
    return output


def main() -> None:
    args = parse_args()
    args.tables_dir.mkdir(parents=True, exist_ok=True)
    args.figures_dir.mkdir(parents=True, exist_ok=True)
    run_manifest = json.loads((args.bayes_dir / "run_manifest.json").read_text())
    posterior_path = args.bayes_dir / f"{run_manifest['tag']}_posterior_samples.npz"
    negative_controls = pd.read_csv(args.bayes_dir / "negative_controls.csv").iloc[
        :, 0
    ].astype(str).tolist()
    blacklist = read_cre_blacklist(args.bayes_dir)

    with np.load(posterior_path, allow_pickle=True) as posterior:
        groups = posterior["group_names"].astype(str)
        all_cre_names = posterior["cre_names"].astype(str)
        ordinary_mask = all_cre_names != POOLED_NAME
        cre_names = all_cre_names[ordinary_mask]
        log_gamma = posterior["log_gamma"][:, :, ordinary_mask].astype(np.float32)

    control_indices = np.flatnonzero(np.isin(cre_names, negative_controls))
    if len(control_indices) != 7:
        raise ValueError(f"Expected seven ordinary negative controls; found {len(control_indices)}")
    target_indices = np.flatnonzero(
        ~np.isin(cre_names, negative_controls) & ~np.isin(cre_names, list(blacklist))
    )
    t7_totals, group_classes, group_cell_counts = load_grouped_t7(
        args.h5ad, groups, cre_names
    )
    tests = compute_tests(
        log_gamma,
        groups,
        cre_names,
        target_indices,
        control_indices,
        t7_totals,
        group_classes,
        group_cell_counts,
        args.t7_threshold,
        args.effect_threshold,
    )
    tests["significant_q"] = tests["q_right"].le(args.q_cutoff)

    table_path = args.tables_dir / f"{args.stem}.csv.gz"
    tests.to_csv(table_path, index=False)
    significant = tests.loc[tests["significant_q"]]
    manifest_path = args.figures_dir / f"{args.stem}_manifest.json"
    write_json(
        manifest_path,
        {
            "method": METHOD,
            "model": "Joint+dropout ordinary-and-pooled negative controls",
            "bayes_dir": str(args.bayes_dir),
            "posterior": str(posterior_path),
            "negative_controls": negative_controls,
            "activity_definition": "raw posterior log_gamma; alpha is not subtracted",
            "contrast_definition": (
                "target log_gamma minus the maximum of seven ordinary negative-control "
                "log_gamma values within each posterior draw"
            ),
            "p_right_definition": "posterior fraction of draw-wise contrasts <= 0",
            "multiple_testing": "BH across all eligible target-cell-type pairs",
            "t7_filter": (
                f"target T7 >= {args.t7_threshold:g} and total T7 across seven controls "
                f">= {args.t7_threshold:g}"
            ),
            "q_cutoff": args.q_cutoff,
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
