#!/usr/bin/env python3
"""Derive stricter-T7 Bayesian and bootstrap activity-concordance plots."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from compare_bootstrap_origin_vs_new import plot_activity as plot_bootstrap_activity
from compare_origin_vs_new import plot_activity as plot_bayesian_activity
from compare_origin_vs_new import safe_correlations


HERE = Path(__file__).resolve()
ANALYSIS_DIR = HERE.parent.parent
DEFAULT_COMPARISON_DIR = ANALYSIS_DIR / "results" / "comparison"
COUNT_COLUMNS = [
    "origin_target_t7_total",
    "new_target_t7_total",
    "origin_negative_control_t7_total",
    "new_negative_control_t7_total",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison-dir", type=Path, default=DEFAULT_COMPARISON_DIR)
    parser.add_argument("--source-threshold", type=float, default=50)
    parser.add_argument("--t7-threshold", type=float, default=100)
    return parser.parse_args()


def token(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def filter_pairs(path: Path, threshold: float) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = set(COUNT_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing count columns: {sorted(missing)}")
    keep = frame[COUNT_COLUMNS].ge(threshold).all(axis=1)
    return frame.loc[keep].copy()


def select_controls(path: Path, groups: set[str]) -> pd.DataFrame:
    frame = pd.read_csv(path)
    return frame.loc[frame["group"].astype(str).isin(groups)].copy()


def correlation(frame: pd.DataFrame, x: str, y: str) -> dict[str, float | None]:
    pearson, spearman = safe_correlations(frame[x], frame[y])
    return {
        "pearson": None if not np.isfinite(pearson) else float(pearson),
        "spearman": None if not np.isfinite(spearman) else float(spearman),
    }


def main() -> None:
    args = parse_args()
    if args.t7_threshold < args.source_threshold:
        raise ValueError("The requested threshold must be at least the source threshold")

    comparison_dir = args.comparison_dir
    tables_dir = comparison_dir / "tables"
    figures_dir = comparison_dir / "figures"
    bootstrap_dir = comparison_dir / "bootstrap"
    bootstrap_tables = bootstrap_dir / "tables"
    bootstrap_figures = bootstrap_dir / "figures"
    output_token = token(args.t7_threshold)
    source_token = token(args.source_threshold)

    bayesian = filter_pairs(
        tables_dir / f"overlap_t7_ge{source_token}_pair_comparison.csv.gz",
        args.t7_threshold,
    )
    bayesian_groups = set(bayesian["group"].astype(str))
    bayesian_controls = select_controls(
        tables_dir / f"overlap_t7_ge{source_token}_negative_control_activity.csv",
        bayesian_groups,
    )
    bayesian_path = tables_dir / f"overlap_t7_ge{output_token}_pair_comparison.csv.gz"
    bayesian_control_path = (
        tables_dir / f"overlap_t7_ge{output_token}_negative_control_activity.csv"
    )
    bayesian.to_csv(bayesian_path, index=False)
    bayesian_controls.to_csv(bayesian_control_path, index=False)
    plot_bayesian_activity(
        bayesian,
        bayesian_controls,
        figures_dir,
        t7_threshold=args.t7_threshold,
    )

    bootstrap = filter_pairs(
        bootstrap_tables
        / f"overlap_t7_ge{source_token}_bootstrap_pair_comparison.csv.gz",
        args.t7_threshold,
    )
    bootstrap_groups = set(bootstrap["group"].astype(str))
    bootstrap_controls = select_controls(
        bootstrap_tables
        / f"overlap_t7_ge{source_token}_bootstrap_negative_control_activity.csv",
        bootstrap_groups,
    )
    bootstrap_path = (
        bootstrap_tables
        / f"overlap_t7_ge{output_token}_bootstrap_pair_comparison.csv.gz"
    )
    bootstrap_control_path = (
        bootstrap_tables
        / f"overlap_t7_ge{output_token}_bootstrap_negative_control_activity.csv"
    )
    bootstrap.to_csv(bootstrap_path, index=False)
    bootstrap_controls.to_csv(bootstrap_control_path, index=False)
    plot_bootstrap_activity(
        bootstrap,
        bootstrap_controls,
        bootstrap_figures,
        t7_threshold=args.t7_threshold,
    )

    finite_bootstrap_controls = bootstrap_controls[
        ["origin_centered_log_activity_mean", "new_centered_log_activity_mean"]
    ].notna().all(axis=1)
    manifest = {
        "source_t7_threshold": args.source_threshold,
        "t7_threshold": args.t7_threshold,
        "filter": "all origin/new target/control T7 totals meet threshold",
        "bayesian": {
            "pairs": len(bayesian),
            "cell_types": len(bayesian_groups),
            "ccres": bayesian["cre"].nunique(),
            "negative_control_points": len(bayesian_controls),
            **correlation(
                bayesian,
                "origin_effect_vs_mean_control_mean",
                "new_effect_vs_mean_control_mean",
            ),
        },
        "bootstrap": {
            "pairs": len(bootstrap),
            "cell_types": len(bootstrap_groups),
            "ccres": bootstrap["cre"].nunique(),
            "negative_control_points": len(bootstrap_controls),
            "finite_negative_control_points": int(finite_bootstrap_controls.sum()),
            **correlation(
                bootstrap,
                "origin_effect_vs_mean_control_mean",
                "new_effect_vs_mean_control_mean",
            ),
        },
    }
    manifest_path = tables_dir / f"t7_ge{output_token}_activity_concordance_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
