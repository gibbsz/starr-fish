#!/usr/bin/env python3
"""Visualize raw counts behind horizontal/vertical correlation-plot stripes."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from analysis_utils import ANALYSIS_DIR, DEFAULT_H5AD, LIBSIZE_CSV, log, write_json
import plot_method_activity_correlation as pm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bootstrap-dir", type=Path, default=ANALYSIS_DIR / "results" / "bootstrap"
    )
    parser.add_argument(
        "--old-bayesian-dir",
        type=Path,
        default=ANALYSIS_DIR / "results" / "ablation" / "bayesian_joint",
    )
    parser.add_argument(
        "--new-bayesian-dir",
        type=Path,
        default=None,
        help="Legacy alias for the decoupled+dropout Bayesian directory.",
    )
    parser.add_argument(
        "--decoupled-bayesian-dir",
        type=Path,
        default=None,
        help="Bayesian decoupled directory without dropout.",
    )
    parser.add_argument(
        "--joint-dropout-bayesian-dir",
        type=Path,
        default=None,
        help="Bayesian joint directory with zero-inflated dropout.",
    )
    parser.add_argument(
        "--decoupled-dropout-bayesian-dir",
        type=Path,
        default=None,
        help="Bayesian decoupled directory with zero-inflated dropout.",
    )
    parser.add_argument(
        "--figures-dir", type=Path, default=ANALYSIS_DIR / "results" / "figures"
    )
    parser.add_argument("--h5ad", type=Path, default=DEFAULT_H5AD)
    parser.add_argument("--activity-calibration", choices=["none"], default="none")
    parser.add_argument("--bootstrap-log-chunk-size", type=int, default=250)
    parser.add_argument("--t7-threshold", type=float, default=100.0)
    parser.add_argument("--t7-thresholds", type=float, nargs="+", default=[10.0, 50.0, 100.0])
    parser.add_argument("--cell-count-threshold", type=int, default=1000)
    parser.add_argument("--nanopore-threshold", type=float, default=1000.0)
    parser.add_argument("--libsize-csv", type=Path, default=LIBSIZE_CSV)
    parser.add_argument("--bayesian-method", default="Decoupled")
    parser.add_argument("--horizontal-center", type=float, default=-0.5)
    parser.add_argument("--horizontal-halfwidth", type=float, default=0.05)
    parser.add_argument("--vertical-center", type=float, default=0.0)
    parser.add_argument("--vertical-halfwidth", type=float, default=0.05)
    parser.add_argument(
        "--off-diagonal-min-abs-diff",
        type=float,
        default=1.0,
        help=(
            "Keep stripe pairs only when abs(Bootstrap - bayesian_method) is at "
            "least this many log units."
        ),
    )
    parser.add_argument("--background-sample", type=int, default=30_000)
    parser.add_argument("--stem", default="method_activity_stripe_count_diagnostics")
    return parser.parse_args()


def count_axis_ticks(max_count: float) -> list[int]:
    candidates = [0, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000]
    return [tick for tick in candidates if tick <= max_count]


def add_count_ticks(ax, max_x: float, max_y: float) -> None:
    xticks = count_axis_ticks(max_x)
    yticks = count_axis_ticks(max_y)
    ax.set_xticks(np.log10(np.asarray(xticks, dtype=float) + 1.0))
    ax.set_xticklabels([str(tick) for tick in xticks])
    ax.set_yticks(np.log10(np.asarray(yticks, dtype=float) + 1.0))
    ax.set_yticklabels([str(tick) for tick in yticks])


def build_data(args: argparse.Namespace) -> tuple[pd.DataFrame, dict]:
    methods = pm.CORRELATION_METHODS
    matrices, pair_t7, pair_cre, cell_counts, _, metadata = pm.prepare_base(args, methods)
    wide = pm.stack_methods(matrices, methods)
    total_t7 = pm.pair_count_series(pair_t7, "total_t7").reindex(wide.index)
    total_ccre = pm.pair_count_series(pair_cre, "total_ccre").reindex(wide.index)
    group_index = wide.index.get_level_values("group").astype(str)
    n_cells = pd.Series(group_index, index=wide.index).map(cell_counts.astype(float))
    data = wide.copy()
    data["total_t7"] = total_t7
    data["total_ccre"] = total_ccre
    data["n_cells"] = n_cells
    data = data.reset_index()
    return data, metadata


def selected_summary(selected: pd.DataFrame) -> dict:
    keys = ["n_cells", "total_t7", "total_ccre", "Bootstrap", "Point log(cCRE/T7)"]
    summary = {"n_pairs": int(len(selected))}
    for key in keys:
        values = pd.to_numeric(selected[key], errors="coerce")
        summary[key] = {
            "median": float(values.median()) if values.notna().any() else np.nan,
            "q25": float(values.quantile(0.25)) if values.notna().any() else np.nan,
            "q75": float(values.quantile(0.75)) if values.notna().any() else np.nan,
        }
    return summary


def plot_diagnostic(
    data: pd.DataFrame,
    selected: pd.DataFrame,
    args: argparse.Namespace,
    *,
    title: str,
    output: Path,
    band_kind: str,
) -> dict:
    sns.set_theme(context="paper", style="white")
    rng = np.random.default_rng(0)
    finite_context = data[["Bootstrap", args.bayesian_method]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(finite_context) > args.background_sample:
        finite_context = finite_context.sample(args.background_sample, random_state=0)

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(15.8, 4.8),
        constrained_layout=True,
        gridspec_kw={"width_ratios": [1.1, 1.25, 0.95]},
    )

    ax = axes[0]
    ax.scatter(
        finite_context["Bootstrap"],
        finite_context[args.bayesian_method],
        s=1.2,
        color="0.75",
        alpha=0.08,
        linewidths=0,
        rasterized=True,
    )
    ax.scatter(
        selected["Bootstrap"],
        selected[args.bayesian_method],
        s=4,
        color="#d62728",
        alpha=0.45,
        linewidths=0,
        rasterized=True,
    )
    if band_kind == "horizontal":
        lo = args.horizontal_center - args.horizontal_halfwidth
        hi = args.horizontal_center + args.horizontal_halfwidth
        ax.axhspan(lo, hi, color="#d62728", alpha=0.12, linewidth=0)
    else:
        lo = args.vertical_center - args.vertical_halfwidth
        hi = args.vertical_center + args.vertical_halfwidth
        ax.axvspan(lo, hi, color="#d62728", alpha=0.12, linewidth=0)
    ax.set_xlabel("Bootstrap mean log activity")
    ax.set_ylabel(args.bayesian_method)
    ax.set_title("Selected stripe location")

    ax = axes[1]
    count_norm = mcolors.LogNorm(
        vmin=max(1.0, float(selected["n_cells"].min())),
        vmax=max(2.0, float(selected["n_cells"].max())),
    )
    x = np.log10(selected["total_t7"].to_numpy(float) + 1.0)
    y = np.log10(selected["total_ccre"].to_numpy(float) + 1.0)
    jitter = rng.normal(0.0, 0.012, size=(len(selected), 2))
    sc = ax.scatter(
        x + jitter[:, 0],
        y + jitter[:, 1],
        c=selected["n_cells"].to_numpy(float),
        norm=count_norm,
        cmap="viridis",
        s=7,
        alpha=0.45,
        linewidths=0,
        rasterized=True,
    )
    max_count = max(float(selected["total_t7"].max()), float(selected["total_ccre"].max()), 10.0)
    grid = np.linspace(1, max_count, 200)
    for ratio, color, label in [(1.0, "black", "cCRE=T7"), (0.2, "0.45", "cCRE=0.2*T7"), (5.0, "0.45", "cCRE=5*T7")]:
        yy = ratio * grid
        valid = yy <= max_count
        ax.plot(np.log10(grid[valid] + 1), np.log10(yy[valid] + 1), color=color, lw=0.7, alpha=0.55)
        if valid.any():
            ax.text(
                np.log10(grid[valid][-1] + 1),
                np.log10(yy[valid][-1] + 1),
                label,
                fontsize=6,
                color=color,
                ha="right",
                va="bottom",
            )
    add_count_ticks(ax, max_count, max_count)
    ax.set_xlabel("Total T7 counts, log10(count + 1)")
    ax.set_ylabel("Total cCRE counts, log10(count + 1)")
    ax.set_title("Raw pair counts; color = cells in subclass")
    cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)
    cbar.set_label("n cells")

    ax = axes[2]
    long = selected[["n_cells", "total_t7", "total_ccre"]].rename(
        columns={
            "n_cells": "n cells",
            "total_t7": "total T7",
            "total_ccre": "total cCRE",
        }
    )
    long = long.melt(var_name="quantity", value_name="value")
    long["log10_value_plus1"] = np.log10(pd.to_numeric(long["value"], errors="coerce") + 1.0)
    sns.boxplot(
        data=long,
        x="quantity",
        y="log10_value_plus1",
        color="white",
        showfliers=False,
        linewidth=0.8,
        ax=ax,
    )
    sns.stripplot(
        data=long,
        x="quantity",
        y="log10_value_plus1",
        color="#2f6f8f",
        alpha=0.12,
        size=1.3,
        jitter=0.25,
        ax=ax,
        rasterized=True,
    )
    ax.set_xlabel("")
    ax.set_ylabel("log10(value + 1)")
    ax.set_title("Count and cell-count distributions")

    summary = selected_summary(selected)
    fig.suptitle(
        f"{title}; n={len(selected):,} pairs | "
        f"median cells={summary['n_cells']['median']:.0f}, "
        f"T7={summary['total_t7']['median']:.0f}, "
        f"cCRE={summary['total_ccre']['median']:.0f}",
        fontsize=12,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    return {"output": str(output), **summary}


def main() -> None:
    args = parse_args()
    args.figures_dir.mkdir(parents=True, exist_ok=True)
    data, metadata = build_data(args)
    finite = data[["Bootstrap", args.bayesian_method]].replace([np.inf, -np.inf], np.nan)
    finite_mask = finite.notna().all(axis=1)

    horizontal = data[
        finite_mask
        & data[args.bayesian_method].between(
            args.horizontal_center - args.horizontal_halfwidth,
            args.horizontal_center + args.horizontal_halfwidth,
        )
        & (data["Bootstrap"] - data[args.bayesian_method]).abs().ge(
            args.off_diagonal_min_abs_diff
        )
    ].copy()
    vertical = data[
        finite_mask
        & data["Bootstrap"].between(
            args.vertical_center - args.vertical_halfwidth,
            args.vertical_center + args.vertical_halfwidth,
        )
        & (data["Bootstrap"] - data[args.bayesian_method]).abs().ge(
            args.off_diagonal_min_abs_diff
        )
    ].copy()

    outputs = {
        "horizontal_bayesian_minus0p5": plot_diagnostic(
            data,
            horizontal,
            args,
            title=(
                f"Horizontal stripe: {args.bayesian_method} near "
                f"{args.horizontal_center:g}"
            ),
            output=args.figures_dir / f"{args.stem}_horizontal_bayesian_minus0p5.pdf",
            band_kind="horizontal",
        ),
        "vertical_bootstrap_0": plot_diagnostic(
            data,
            vertical,
            args,
            title=f"Vertical strip: Bootstrap near {args.vertical_center:g}",
            output=args.figures_dir / f"{args.stem}_vertical_bootstrap_0.pdf",
            band_kind="vertical",
        ),
    }
    horizontal.to_csv(
        args.figures_dir / f"{args.stem}_horizontal_bayesian_minus0p5_pairs.csv",
        index=False,
    )
    vertical.to_csv(
        args.figures_dir / f"{args.stem}_vertical_bootstrap_0_pairs.csv",
        index=False,
    )
    write_json(
        args.figures_dir / f"{args.stem}_manifest.json",
        {
            "bayesian_method": args.bayesian_method,
            "horizontal_center": args.horizontal_center,
            "horizontal_halfwidth": args.horizontal_halfwidth,
            "vertical_center": args.vertical_center,
            "vertical_halfwidth": args.vertical_halfwidth,
            "off_diagonal_min_abs_diff": args.off_diagonal_min_abs_diff,
            "off_diagonal_rule": (
                "abs(Bootstrap - bayesian_method) >= off_diagonal_min_abs_diff"
            ),
            "metadata": metadata,
            "outputs": outputs,
        },
    )
    log("[stripe count diagnostics] wrote horizontal and vertical stripe count plots")


if __name__ == "__main__":
    main()
