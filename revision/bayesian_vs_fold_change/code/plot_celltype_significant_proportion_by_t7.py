#!/usr/bin/env python3
"""Plot per-cell-type significant cCRE proportions across T7 filters."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from analysis_utils import ANALYSIS_DIR, write_json
from plot_t7_filter_precision_recall import METHOD_COLORS, METHODS, restrict_to_common_pairs


DEFAULT_METHODS = tuple(method for method in METHODS if method != "Metacell Bayesian")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tests",
        type=Path,
        default=ANALYSIS_DIR
        / "results"
        / "tables"
        / "method_activity_t7_filter_negative_control_posterior_alpha_subtracted_with_unfiltered_tests.csv.gz",
    )
    parser.add_argument(
        "--figures-dir", type=Path, default=ANALYSIS_DIR / "results" / "figures"
    )
    parser.add_argument(
        "--tables-dir", type=Path, default=ANALYSIS_DIR / "results" / "tables"
    )
    parser.add_argument("--q-cutoff", type=float, default=0.05)
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=METHODS,
        default=list(DEFAULT_METHODS),
    )
    parser.add_argument(
        "--stem",
        default=(
            "method_activity_t7_filter_posterior_alpha_subtracted_common_pairs_"
            "celltype_significant_proportion"
        ),
    )
    return parser.parse_args()


def threshold_label(threshold: float) -> str:
    return "No T7 filter" if np.isclose(threshold, 0.0) else f"T7 >= {threshold:g}"


def summarize_cell_types(
    tests: pd.DataFrame,
    methods: tuple[str, ...],
    q_cutoff: float,
) -> pd.DataFrame:
    frame = tests.copy()
    frame["significant"] = frame["q_right"].le(q_cutoff)
    summary = (
        frame.groupby(["t7_threshold", "group", "method"], sort=False)
        .agg(
            tested_cres=("cre", "size"),
            significant_cres=("significant", "sum"),
        )
        .reset_index()
    )
    summary["significant_proportion"] = (
        summary["significant_cres"] / summary["tested_cres"]
    )
    summary["q_cutoff"] = q_cutoff
    summary["method"] = pd.Categorical(
        summary["method"], categories=list(methods), ordered=True
    )
    return summary.sort_values(["t7_threshold", "group", "method"])


def plot_proportions(
    summary: pd.DataFrame,
    group_order: list[str],
    methods: tuple[str, ...],
    output_pdf: Path,
    output_png: Path,
) -> None:
    thresholds = sorted(summary["t7_threshold"].astype(float).unique())
    max_groups = max(
        summary.loc[np.isclose(summary["t7_threshold"], threshold), "group"].nunique()
        for threshold in thresholds
    )
    width = max(45.0, max_groups * 0.22)
    height = 4.1 * len(thresholds) + 1.5
    sns.set_theme(context="paper", style="whitegrid")
    fig, axes = plt.subplots(
        len(thresholds),
        1,
        figsize=(width, height),
        sharey=True,
        constrained_layout=True,
        squeeze=False,
    )
    axes = axes[:, 0]
    bar_width = 0.86 / len(methods)

    for ax, threshold in zip(axes, thresholds):
        panel = summary[np.isclose(summary["t7_threshold"], threshold)]
        present_groups = set(panel["group"].astype(str))
        panel_groups = [group for group in group_order if group in present_groups]
        centers = np.arange(len(panel_groups), dtype=float)
        for method_idx, method in enumerate(methods):
            method_data = (
                panel[panel["method"].astype(str).eq(method)]
                .set_index("group")
                .reindex(panel_groups)
            )
            offset = (method_idx - (len(methods) - 1) / 2.0) * bar_width
            x = centers + offset
            bars = ax.bar(
                x,
                method_data["significant_proportion"],
                width=bar_width,
                color=METHOD_COLORS[method],
                alpha=0.9,
                linewidth=0,
                rasterized=True,
                label=method,
            )
            labels = [
                f"{int(significant)}/{int(tested)}"
                for significant, tested in zip(
                    method_data["significant_cres"], method_data["tested_cres"]
                )
            ]
            ax.bar_label(
                bars,
                labels=labels,
                padding=1,
                rotation=90,
                fontsize=3.1,
                color="#222222",
            )
        ax.set_ylim(0.0, 1.13)
        ax.set_xlim(-0.55, len(panel_groups) - 0.45)
        ax.set_yticks(np.linspace(0.0, 1.0, 6))
        ax.set_title(
            f"{threshold_label(float(threshold))} ({len(panel_groups)} cell types)",
            loc="left",
            fontsize=9,
        )
        ax.set_ylabel("Significant proportion")
        ax.set_xticks(centers, panel_groups, rotation=90)
        ax.tick_params(axis="x", labelsize=4.0, pad=1)
        ax.set_xlabel("Cell type")
        ax.grid(axis="x", visible=False)
        sns.despine(ax=ax)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        title="Method",
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=len(methods),
        frameon=False,
    )
    fig.suptitle(
        "Proportion of tested cCREs significantly above the negative-control posterior "
        f"(q <= {summary['q_cutoff'].iloc[0]:g})",
        fontsize=12,
    )
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_pdf, bbox_inches="tight")
    fig.savefig(output_png, bbox_inches="tight", dpi=110)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.figures_dir.mkdir(parents=True, exist_ok=True)
    args.tables_dir.mkdir(parents=True, exist_ok=True)
    methods = tuple(dict.fromkeys(args.methods))
    usecols = ["t7_threshold", "method", "group", "cre", "p_right", "q_right"]
    tests = pd.read_csv(args.tests, usecols=usecols)
    tests["method"] = tests["method"].astype(str)
    tests = tests[tests["method"].isin(methods)].copy()

    first_method = tests["method"].eq(methods[0])
    no_filter = np.isclose(tests["t7_threshold"].to_numpy(float), 0.0)
    group_order = list(dict.fromkeys(tests.loc[first_method & no_filter, "group"].astype(str)))
    if not group_order:
        raise ValueError("The tests table does not contain a threshold-0 no-filter panel")

    evaluated, common_counts = restrict_to_common_pairs(tests, methods)
    summary = summarize_cell_types(evaluated, methods, args.q_cutoff)

    summary_path = args.tables_dir / f"{args.stem}_summary.csv"
    evaluated_path = args.tables_dir / f"{args.stem}_evaluated_tests.csv.gz"
    pdf_path = args.figures_dir / f"{args.stem}.pdf"
    png_path = args.figures_dir / f"{args.stem}.png"
    summary.to_csv(summary_path, index=False)
    evaluated.to_csv(evaluated_path, index=False)
    plot_proportions(summary, group_order, methods, pdf_path, png_path)

    write_json(
        args.figures_dir / f"{args.stem}_manifest.json",
        {
            "tests": str(args.tests),
            "evaluated_tests": str(evaluated_path),
            "summary": str(summary_path),
            "figure_pdf": str(pdf_path),
            "figure_png": str(png_path),
            "methods": list(methods),
            "q_cutoff": args.q_cutoff,
            "thresholds": sorted(summary["t7_threshold"].astype(float).unique()),
            "n_cell_types": len(group_order),
            "n_cell_types_by_t7_threshold": {
                str(float(threshold)): int(
                    summary.loc[
                        np.isclose(summary["t7_threshold"], threshold), "group"
                    ].nunique()
                )
                for threshold in sorted(
                    summary["t7_threshold"].astype(float).unique()
                )
            },
            "common_pair_counts_by_t7_threshold": common_counts,
            "definition": (
                "Within each T7 threshold and cell type, number of cCREs with common-pair "
                "BH q_right <= q_cutoff divided by the number of common tested cCREs. "
                "Threshold 0 is labeled no T7 filter; cell types with no common tested "
                "cCREs at a threshold are omitted from that panel."
            ),
        },
    )


if __name__ == "__main__":
    main()
