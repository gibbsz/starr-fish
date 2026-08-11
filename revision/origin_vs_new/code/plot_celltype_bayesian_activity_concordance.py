#!/usr/bin/env python3
"""Plot origin-versus-new Bayesian activity for one cell type."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


HERE = Path(__file__).resolve()
ANALYSIS_DIR = HERE.parent.parent
DEFAULT_COMPARISON_DIR = ANALYSIS_DIR / "results" / "comparison"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cell-type", default="Endo NN")
    parser.add_argument("--t7-threshold", type=float, default=100)
    parser.add_argument("--comparison-dir", type=Path, default=DEFAULT_COMPARISON_DIR)
    return parser.parse_args()


def token(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def slug(value: str) -> str:
    return "_".join(value.lower().replace("/", " ").split())


def main() -> None:
    args = parse_args()
    threshold_token = token(args.t7_threshold)
    tables_dir = args.comparison_dir / "tables"
    figures_dir = args.comparison_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    targets = pd.read_csv(
        tables_dir
        / f"overlap_t7_ge{threshold_token}_bayesian_statistical_tests.csv.gz"
    )
    targets = targets.loc[targets["group"].astype(str).eq(args.cell_type)].copy()
    controls = pd.read_csv(
        tables_dir / f"overlap_t7_ge{threshold_token}_negative_control_activity.csv"
    )
    controls = controls.loc[controls["group"].astype(str).eq(args.cell_type)].copy()
    if targets.empty:
        raise ValueError(f"No shared targets found for {args.cell_type!r}")
    if len(controls) != 7:
        raise ValueError(
            f"Expected seven negative controls for {args.cell_type!r}; found {len(controls)}"
        )

    target_x = targets["origin_activity_mean"].to_numpy(float)
    target_y = targets["new_activity_mean"].to_numpy(float)
    control_x = controls["origin_raw_activity_mean"].to_numpy(float)
    control_y = controls["new_raw_activity_mean"].to_numpy(float)
    origin_mean = float(control_x.mean())
    new_mean = float(control_y.mean())
    origin_sd = float(control_x.std(ddof=1))
    new_sd = float(control_y.std(ddof=1))
    pearson = float(pearsonr(target_x, target_y).statistic)
    spearman = float(spearmanr(target_x, target_y).statistic)

    values = np.concatenate([target_x, target_y, control_x, control_y])
    lower, upper = np.nanmin(values), np.nanmax(values)
    padding = max((upper - lower) * 0.06, 0.15)
    limits = (lower - padding, upper + padding)

    fig, ax = plt.subplots(figsize=(7.4, 6.8))
    ax.axvspan(
        origin_mean - origin_sd,
        origin_mean + origin_sd,
        color="#E69F00",
        alpha=0.10,
        zorder=0,
        label="Negative-control mean ± 1 SD",
    )
    ax.axhspan(
        new_mean - new_sd,
        new_mean + new_sd,
        color="#E69F00",
        alpha=0.10,
        zorder=0,
    )
    ax.plot(limits, limits, linestyle=":", linewidth=1.1, color="0.35", zorder=1)
    ax.axvline(
        origin_mean,
        linestyle="--",
        linewidth=1.5,
        color="#D55E00",
        zorder=2,
    )
    ax.axhline(
        new_mean,
        linestyle="--",
        linewidth=1.5,
        color="#D55E00",
        zorder=2,
    )
    ax.scatter(
        target_x,
        target_y,
        s=35,
        alpha=0.62,
        color="#2F6F8F",
        edgecolors="white",
        linewidths=0.35,
        zorder=3,
        label=f"Shared T7≥{args.t7_threshold:g} targets (n={len(targets)})",
    )
    ax.scatter(
        control_x,
        control_y,
        s=75,
        alpha=0.95,
        color="#D55E00",
        edgecolors="white",
        linewidths=0.7,
        zorder=4,
        label="Negative controls (n=7)",
    )
    for row in controls.itertuples(index=False):
        ax.annotate(
            str(row.cre),
            (row.origin_raw_activity_mean, row.new_raw_activity_mean),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=7,
            color="#9C3D00",
            zorder=5,
        )

    ax.set_xlim(limits)
    ax.set_ylim(limits)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Original Bayesian activity")
    ax.set_ylabel("New Bayesian activity")
    ax.set_title(
        f"{args.cell_type}: Bayesian activity, origin versus new\n"
        f"T7 ≥ {args.t7_threshold:g}; targets Pearson r={pearson:.3f}, "
        f"Spearman ρ={spearman:.3f}"
    )
    ax.text(
        0.02,
        0.98,
        "Negative controls\n"
        f"Origin: mean={origin_mean:.2f}, SD={origin_sd:.2f}\n"
        f"New: mean={new_mean:.2f}, SD={new_sd:.2f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.5,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.88},
    )
    ax.legend(loc="lower right", frameon=True, fontsize=8)
    fig.tight_layout()

    stem = (
        f"{slug(args.cell_type)}_bayesian_activity_origin_vs_new_"
        f"t7_ge{threshold_token}"
    )
    for suffix in ("png", "pdf"):
        fig.savefig(figures_dir / f"{stem}.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(
        f"cell_type={args.cell_type}; targets={len(targets)}; "
        f"origin_control_mean={origin_mean:.6f}; origin_control_sd={origin_sd:.6f}; "
        f"new_control_mean={new_mean:.6f}; new_control_sd={new_sd:.6f}; "
        f"pearson={pearson:.6f}; spearman={spearman:.6f}"
    )


if __name__ == "__main__":
    main()
