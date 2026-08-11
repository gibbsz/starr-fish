#!/usr/bin/env python3
"""Compare recomputed origin/new statistical tests at a stricter T7 cutoff."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from compare_origin_vs_new import bh_fdr, call_metrics, safe_correlations


HERE = Path(__file__).resolve()
ANALYSIS_DIR = HERE.parent.parent
REPO_ROOT = ANALYSIS_DIR.parents[1]
ORIGIN_RESULTS = REPO_ROOT / "revision" / "bayesian_vs_fold_change" / "results"
NEW_RESULTS = REPO_ROOT / "revision" / "Bayes_NewData"
KEY = ["group", "cre"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--t7-threshold", type=float, default=100)
    parser.add_argument("--q-cutoff", type=float, default=0.05)
    parser.add_argument(
        "--origin-bayesian",
        type=Path,
        default=ORIGIN_RESULTS
        / "tables"
        / "joint_dropout_direct_activity_mean_negative_control_tests_t7_series.csv.gz",
    )
    parser.add_argument(
        "--new-bayesian",
        type=Path,
        default=NEW_RESULTS
        / "tables"
        / "new_mean_negative_control_tests_t7_ge100.csv.gz",
    )
    parser.add_argument(
        "--origin-bootstrap",
        type=Path,
        default=ORIGIN_RESULTS
        / "tables"
        / "bootstrap_mean_negative_control_tests_t7_series.csv.gz",
    )
    parser.add_argument(
        "--new-bootstrap",
        type=Path,
        default=NEW_RESULTS
        / "tables"
        / "new_bootstrap_mean_negative_control_tests_t7_ge100.csv.gz",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ANALYSIS_DIR / "results" / "comparison",
    )
    return parser.parse_args()


def token(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def read_threshold(path: Path, threshold: float) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = set(KEY + ["t7_threshold", "p_right", "q_right", "significant_q"])
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    frame = frame.loc[
        np.isclose(frame["t7_threshold"].to_numpy(float), threshold)
    ].copy()
    if frame.empty:
        raise ValueError(f"{path} has no T7 >= {threshold:g} tests")
    if frame.duplicated(KEY).any():
        raise ValueError(f"{path} has duplicate cell-type/cCRE tests")
    frame["significant_q"] = frame["q_right"].le(0.05)
    return frame


def compare(
    origin: pd.DataFrame,
    new: pd.DataFrame,
    q_cutoff: float,
) -> tuple[pd.DataFrame, dict]:
    columns = [
        column
        for column in origin.columns.intersection(new.columns)
        if column not in KEY
    ]
    shared = origin[KEY + columns].rename(
        columns={column: f"origin_{column}" for column in columns}
    ).merge(
        new[KEY + columns].rename(
            columns={column: f"new_{column}" for column in columns}
        ),
        on=KEY,
        how="inner",
        validate="one_to_one",
    )
    shared["origin_q_shared_universe"] = bh_fdr(shared["origin_p_right"])
    shared["new_q_shared_universe"] = bh_fdr(shared["new_p_right"])
    shared["origin_significant_shared_q"] = shared[
        "origin_q_shared_universe"
    ].le(q_cutoff)
    shared["new_significant_shared_q"] = shared["new_q_shared_universe"].le(
        q_cutoff
    )
    metrics = call_metrics(
        shared["origin_significant_shared_q"],
        shared["new_significant_shared_q"],
    )
    pearson, spearman = safe_correlations(
        shared["origin_effect_vs_mean_control_mean"],
        shared["new_effect_vs_mean_control_mean"],
    )
    summary = {
        "origin_eligible_tests": len(origin),
        "origin_significant_tests": int(origin["q_right"].le(q_cutoff).sum()),
        "new_eligible_tests": len(new),
        "new_significant_tests": int(new["q_right"].le(q_cutoff).sum()),
        "shared_tests": len(shared),
        "shared_cell_types": shared["group"].nunique(),
        "shared_ccres": shared["cre"].nunique(),
        "activity_pearson": pearson,
        "activity_spearman": spearman,
        **metrics,
    }
    return shared, summary


def plot_calls(
    shared: pd.DataFrame,
    summary: dict,
    figures_dir: Path,
    stem: str,
    title_prefix: str,
    threshold: float,
    q_cutoff: float,
) -> None:
    values = [
        summary["origin_only_significant"],
        summary["both_significant"],
        summary["new_only_significant"],
    ]
    matrix = np.asarray(
        [
            [summary["neither_significant"], summary["new_only_significant"]],
            [summary["origin_only_significant"], summary["both_significant"]],
        ]
    )
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.4))
    axes[0].bar(
        ["Original only", "Both", "New only"],
        values,
        color=["#4477AA", "#6E6E6E", "#CC6677"],
    )
    axes[0].set_ylabel("Significant cell-type–cCRE pairs")
    axes[0].set_title(f"{title_prefix}; BH q ≤ {q_cutoff:g}")
    for index, value in enumerate(values):
        axes[0].text(index, value, f"{value:,}", ha="center", va="bottom")

    image = axes[1].imshow(matrix, cmap="Blues")
    axes[1].set_xticks([0, 1], ["New not sig.", "New sig."])
    axes[1].set_yticks([0, 1], ["Origin not sig.", "Origin sig."])
    axes[1].set_title(
        f"Shared T7≥{threshold:g} tests: n={len(shared):,}\n"
        f"Concordance={summary['call_concordance']:.3f}; "
        f"significant Jaccard={summary['significant_jaccard']:.3f}"
    )
    for row in range(2):
        for column in range(2):
            axes[1].text(
                column,
                row,
                f"{matrix[row, column]:,}",
                ha="center",
                va="center",
                color="white" if matrix[row, column] > matrix.max() / 2 else "black",
            )
    fig.colorbar(image, ax=axes[1], fraction=0.046, pad=0.04)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(figures_dir / f"{stem}.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    threshold_token = token(args.t7_threshold)
    configs = {
        "bayesian": {
            "origin": args.origin_bayesian,
            "new": args.new_bayesian,
            "tables": args.output_dir / "tables",
            "figures": args.output_dir / "figures",
            "stem": f"overlap_t7_ge{threshold_token}_significant_call_concordance",
            "title": "Bayesian shared test universe",
        },
        "bootstrap": {
            "origin": args.origin_bootstrap,
            "new": args.new_bootstrap,
            "tables": args.output_dir / "bootstrap" / "tables",
            "figures": args.output_dir / "bootstrap" / "figures",
            "stem": f"overlap_t7_ge{threshold_token}_bootstrap_calls",
            "title": "Bootstrap shared test universe",
        },
    }
    summaries = {
        "t7_threshold": args.t7_threshold,
        "q_cutoff": args.q_cutoff,
        "multiple_testing": "BH recomputed separately per run and on shared universe",
    }
    for name, config in configs.items():
        config["tables"].mkdir(parents=True, exist_ok=True)
        config["figures"].mkdir(parents=True, exist_ok=True)
        origin = read_threshold(config["origin"], args.t7_threshold)
        new = read_threshold(config["new"], args.t7_threshold)
        shared, summary = compare(origin, new, args.q_cutoff)
        shared.to_csv(
            config["tables"]
            / f"overlap_t7_ge{threshold_token}_{name}_statistical_tests.csv.gz",
            index=False,
        )
        plot_calls(
            shared,
            summary,
            config["figures"],
            config["stem"],
            config["title"],
            args.t7_threshold,
            args.q_cutoff,
        )
        summaries[name] = summary

    summary_path = (
        args.output_dir
        / "tables"
        / f"overlap_t7_ge{threshold_token}_statistical_test_summary.json"
    )
    summary_path.write_text(json.dumps(summaries, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summaries, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
