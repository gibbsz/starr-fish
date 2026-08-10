#!/usr/bin/env python3
"""Diagnose off-diagonal Bayesian decoupled-vs-joint activity pairs."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# The shared analysis layer (analysis_utils and the plot_* modules that other
# scripts import) stays in the parent code/ directory.
import sys as _sys
from pathlib import Path as _Path
_CODE_DIR = _Path(__file__).resolve().parent.parent
if str(_CODE_DIR) not in _sys.path:
    _sys.path.insert(0, str(_CODE_DIR))

from analysis_utils import (
    ANALYSIS_DIR,
    DEFAULT_H5AD,
    FIGURES_WORK,
    LIBSIZE_CSV,
    OLD_DATA_BOOTSTRAP,
    log,
    write_json,
)
import plot_method_activity_correlation as pm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bootstrap-dir", type=Path, default=OLD_DATA_BOOTSTRAP
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
        "--figures-dir", type=Path, default=FIGURES_WORK
    )
    parser.add_argument("--h5ad", type=Path, default=DEFAULT_H5AD)
    parser.add_argument("--activity-calibration", choices=["none"], default="none")
    parser.add_argument("--bootstrap-log-chunk-size", type=int, default=250)
    parser.add_argument("--t7-threshold", type=float, default=100.0)
    parser.add_argument("--t7-thresholds", type=float, nargs="+", default=[10.0, 50.0, 100.0])
    parser.add_argument("--cell-count-threshold", type=int, default=1000)
    parser.add_argument("--nanopore-threshold", type=float, default=1000.0)
    parser.add_argument("--libsize-csv", type=Path, default=LIBSIZE_CSV)
    parser.add_argument("--joint-method", default="Joint")
    parser.add_argument("--decoupled-method", default="Decoupled")
    parser.add_argument("--delta-threshold", type=float, default=1.5)
    parser.add_argument("--background-sample", type=int, default=30_000)
    parser.add_argument("--stem", default="method_activity_decoupled_gt_joint_diagnostics")
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
    data["delta_decoupled_minus_joint"] = data[args.decoupled_method] - data[args.joint_method]
    data = data.reset_index()
    return data, metadata


def selected_summary(selected: pd.DataFrame) -> dict:
    keys = [
        "delta_decoupled_minus_joint",
        "n_cells",
        "total_t7",
        "total_ccre",
        "Bootstrap",
        "Point log(cCRE/T7)",
    ]
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
    output: Path,
) -> dict:
    sns.set_theme(context="paper", style="white")
    rng = np.random.default_rng(0)
    finite_context = (
        data[[args.joint_method, args.decoupled_method]]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )
    if len(finite_context) > args.background_sample:
        finite_context = finite_context.sample(args.background_sample, random_state=0)

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 11.2), constrained_layout=True)
    axes = axes.ravel()

    ax = axes[0]
    ax.scatter(
        finite_context[args.joint_method],
        finite_context[args.decoupled_method],
        s=1.2,
        color="0.72",
        alpha=0.08,
        linewidths=0,
        rasterized=True,
    )
    ax.scatter(
        selected[args.joint_method],
        selected[args.decoupled_method],
        s=5,
        color="#d62728",
        alpha=0.5,
        linewidths=0,
        rasterized=True,
    )
    finite_all = data[[args.joint_method, args.decoupled_method]].replace([np.inf, -np.inf], np.nan).dropna()
    lo = float(np.nanpercentile(finite_all.to_numpy(float), 0.5))
    hi = float(np.nanpercentile(finite_all.to_numpy(float), 99.5))
    ax.plot([lo, hi], [lo, hi], color="black", lw=0.8, alpha=0.7)
    ax.plot(
        [lo, hi - args.delta_threshold],
        [lo + args.delta_threshold, hi],
        color="#d62728",
        lw=0.8,
        alpha=0.8,
    )
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel(args.joint_method)
    ax.set_ylabel(args.decoupled_method)
    ax.set_title("Selected off-diagonal points")

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
        s=9,
        alpha=0.55,
        linewidths=0,
        rasterized=True,
    )
    max_count = max(float(selected["total_t7"].max()), float(selected["total_ccre"].max()), 10.0)
    grid = np.linspace(1, max_count, 200)
    for ratio, color, label in [
        (1.0, "black", "cCRE=T7"),
        (0.2, "0.45", "cCRE=0.2*T7"),
        (5.0, "0.45", "cCRE=5*T7"),
    ]:
        yy = ratio * grid
        valid = yy <= max_count
        ax.plot(
            np.log10(grid[valid] + 1),
            np.log10(yy[valid] + 1),
            color=color,
            lw=0.7,
            alpha=0.55,
        )
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
    finite_point = selected[["Point log(cCRE/T7)", "delta_decoupled_minus_joint", "total_t7"]].dropna()
    positive_t7 = finite_point["total_t7"].gt(0)
    if positive_t7.any():
        norm = mcolors.LogNorm(
            vmin=1,
            vmax=max(10.0, float(finite_point.loc[positive_t7, "total_t7"].max())),
        )
        ax.scatter(
            finite_point.loc[positive_t7, "Point log(cCRE/T7)"],
            finite_point.loc[positive_t7, "delta_decoupled_minus_joint"],
            c=finite_point.loc[positive_t7, "total_t7"],
            cmap="coolwarm_r",
            norm=norm,
            s=9,
            alpha=0.55,
            linewidths=0,
            rasterized=True,
        )
    zero_t7 = ~positive_t7
    if zero_t7.any():
        ax.scatter(
            finite_point.loc[zero_t7, "Point log(cCRE/T7)"],
            finite_point.loc[zero_t7, "delta_decoupled_minus_joint"],
            color="#7b3294",
            s=9,
            alpha=0.35,
            linewidths=0,
            rasterized=True,
        )
    ax.axhline(args.delta_threshold, color="#d62728", lw=0.8, alpha=0.8)
    ax.axvline(0, color="black", lw=0.6, alpha=0.5)
    ax.set_xlabel("Point log(total cCRE / total T7)")
    ax.set_ylabel(f"{args.decoupled_method} - {args.joint_method}")
    ax.set_title("Raw ratio versus Bayesian discrepancy")

    ax = axes[3]
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
        alpha=0.16,
        size=1.7,
        jitter=0.25,
        ax=ax,
        rasterized=True,
    )
    ax.set_xlabel("")
    ax.set_ylabel("log10(value + 1)")
    ax.set_title("Count and cell-count distributions")

    summary = selected_summary(selected)
    fig.suptitle(
        f"{args.decoupled_method} much larger than {args.joint_method}: "
        f"delta >= {args.delta_threshold:g}; n={len(selected):,} pairs | "
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
    finite = (
        data[[args.joint_method, args.decoupled_method, "delta_decoupled_minus_joint"]]
        .replace([np.inf, -np.inf], np.nan)
        .notna()
        .all(axis=1)
    )
    selected = data[finite & data["delta_decoupled_minus_joint"].ge(args.delta_threshold)].copy()
    selected = selected.sort_values("delta_decoupled_minus_joint", ascending=False)

    output = args.figures_dir / f"{args.stem}.pdf"
    summary = plot_diagnostic(data, selected, args, output)
    table_path = args.figures_dir / f"{args.stem}_pairs.csv"
    selected.to_csv(table_path, index=False)
    manifest = {
        "joint_method": args.joint_method,
        "decoupled_method": args.decoupled_method,
        "delta_threshold": args.delta_threshold,
        "selection_rule": "decoupled_method - joint_method >= delta_threshold",
        "metadata": metadata,
        "output": summary,
        "pairs_csv": str(table_path),
    }
    write_json(args.figures_dir / f"{args.stem}_manifest.json", manifest)
    log(
        "[decoupled joint diagnostics] wrote "
        f"{output} with {len(selected):,} selected pairs"
    )


if __name__ == "__main__":
    main()
