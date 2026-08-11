#!/usr/bin/env python3
"""Plot Bayesian and bootstrap origin-versus-new activity for one cell type."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


HERE = Path(__file__).resolve()
ANALYSIS_DIR = HERE.parent.parent
REPO_ROOT = ANALYSIS_DIR.parents[1]
DEFAULT_COMPARISON_DIR = ANALYSIS_DIR / "results" / "comparison"
DEFAULT_ORIGIN_BOOTSTRAP = REPO_ROOT / "revision" / "Bootstrap_OldData"
DEFAULT_NEW_BOOTSTRAP = REPO_ROOT / "revision" / "Bootstrap_NewData"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cell-type", default="Endo NN")
    parser.add_argument("--t7-threshold", type=float, default=100)
    parser.add_argument("--comparison-dir", type=Path, default=DEFAULT_COMPARISON_DIR)
    parser.add_argument("--origin-bootstrap", type=Path, default=DEFAULT_ORIGIN_BOOTSTRAP)
    parser.add_argument("--new-bootstrap", type=Path, default=DEFAULT_NEW_BOOTSTRAP)
    parser.add_argument("--chunk-size", type=int, default=250)
    return parser.parse_args()


def token(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def slug(value: str) -> str:
    return "_".join(value.lower().replace("/", " ").split())


def bootstrap_control_means(
    bootstrap_dir: Path,
    cell_type: str,
    controls: list[str],
    chunk_size: int,
) -> np.ndarray:
    axes = json.loads((bootstrap_dir / "bootstrap_axes.json").read_text())
    groups = pd.Index(axes["subclasses"], dtype=str)
    cres = pd.Index(axes["cres"], dtype=str)
    group_index = groups.get_indexer([cell_type])[0]
    control_indices = cres.get_indexer(controls)
    if group_index < 0:
        raise ValueError(f"{cell_type!r} is absent from {bootstrap_dir}")
    if (control_indices < 0).any():
        missing = np.asarray(controls)[control_indices < 0].tolist()
        raise ValueError(f"Controls absent from {bootstrap_dir}: {missing}")

    activity = np.load(bootstrap_dir / "celltype_activity_array.npy", mmap_mode="r")
    sums = np.zeros(len(controls), dtype=np.float64)
    counts = np.zeros(len(controls), dtype=np.int64)
    for start in range(0, activity.shape[0], chunk_size):
        stop = min(start + chunk_size, activity.shape[0])
        with np.errstate(divide="ignore", invalid="ignore"):
            logged = np.log(
                activity[start:stop, group_index, control_indices].astype(
                    np.float64, copy=False
                )
            )
        finite = np.isfinite(logged)
        sums += np.where(finite, logged, 0.0).sum(axis=0)
        counts += finite.sum(axis=0)
    return np.divide(
        sums,
        counts,
        out=np.full(len(controls), np.nan),
        where=counts > 0,
    )


def plot_panel(
    ax: plt.Axes,
    targets: pd.DataFrame,
    control_names: list[str],
    control_x: np.ndarray,
    control_y: np.ndarray,
    method: str,
    threshold: float,
) -> dict[str, float]:
    target_x = targets["origin_activity_mean"].to_numpy(float)
    target_y = targets["new_activity_mean"].to_numpy(float)
    origin_mean = float(np.nanmean(control_x))
    new_mean = float(np.nanmean(control_y))
    origin_sd = float(np.nanstd(control_x, ddof=1))
    new_sd = float(np.nanstd(control_y, ddof=1))
    pearson = float(pearsonr(target_x, target_y).statistic)
    spearman = float(spearmanr(target_x, target_y).statistic)

    values = np.concatenate([target_x, target_y, control_x, control_y])
    lower, upper = np.nanmin(values), np.nanmax(values)
    padding = max((upper - lower) * 0.06, 0.15)
    limits = (lower - padding, upper + padding)
    ax.axvspan(
        origin_mean - origin_sd,
        origin_mean + origin_sd,
        color="#E69F00",
        alpha=0.10,
        zorder=0,
        label="Control mean ± 1 SD",
    )
    ax.axhspan(
        new_mean - new_sd,
        new_mean + new_sd,
        color="#E69F00",
        alpha=0.10,
        zorder=0,
    )
    ax.plot(limits, limits, linestyle=":", linewidth=1.1, color="0.35", zorder=1)
    ax.axvline(origin_mean, linestyle="--", linewidth=1.5, color="#D55E00", zorder=2)
    ax.axhline(new_mean, linestyle="--", linewidth=1.5, color="#D55E00", zorder=2)
    ax.scatter(
        target_x,
        target_y,
        s=34,
        alpha=0.62,
        color="#2F6F8F",
        edgecolors="white",
        linewidths=0.35,
        zorder=3,
        label=f"Shared T7≥{threshold:g} targets (n={len(targets)})",
    )
    ax.scatter(
        control_x,
        control_y,
        s=70,
        alpha=0.95,
        color="#D55E00",
        edgecolors="white",
        linewidths=0.7,
        zorder=4,
        label="Negative controls (n=7)",
    )
    for name, x_value, y_value in zip(control_names, control_x, control_y):
        ax.annotate(
            name,
            (x_value, y_value),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=6.5,
            color="#9C3D00",
            zorder=5,
        )
    ax.set_xlim(limits)
    ax.set_ylim(limits)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(f"Original {method} activity")
    ax.set_ylabel(f"New {method} activity")
    ax.set_title(
        f"{method}\nPearson r={pearson:.3f}; Spearman ρ={spearman:.3f}"
    )
    ax.text(
        0.02,
        0.98,
        f"Controls\nOrigin: {origin_mean:.2f} ± {origin_sd:.2f}\n"
        f"New: {new_mean:.2f} ± {new_sd:.2f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.88},
    )
    ax.legend(loc="lower right", frameon=True, fontsize=7.5)
    return {
        "target_count": len(targets),
        "origin_control_mean": origin_mean,
        "origin_control_sd": origin_sd,
        "new_control_mean": new_mean,
        "new_control_sd": new_sd,
        "pearson": pearson,
        "spearman": spearman,
    }


def main() -> None:
    args = parse_args()
    threshold_token = token(args.t7_threshold)
    comparison_dir = args.comparison_dir
    bayesian_targets = pd.read_csv(
        comparison_dir
        / "tables"
        / f"overlap_t7_ge{threshold_token}_bayesian_statistical_tests.csv.gz"
    )
    bayesian_targets = bayesian_targets.loc[
        bayesian_targets["group"].astype(str).eq(args.cell_type)
    ].copy()
    bayesian_controls = pd.read_csv(
        comparison_dir
        / "tables"
        / f"overlap_t7_ge{threshold_token}_negative_control_activity.csv"
    )
    bayesian_controls = bayesian_controls.loc[
        bayesian_controls["group"].astype(str).eq(args.cell_type)
    ].copy()
    bootstrap_targets = pd.read_csv(
        comparison_dir
        / "bootstrap"
        / "tables"
        / f"overlap_t7_ge{threshold_token}_bootstrap_statistical_tests.csv.gz"
    )
    bootstrap_targets = bootstrap_targets.loc[
        bootstrap_targets["group"].astype(str).eq(args.cell_type)
    ].copy()
    if bayesian_targets.empty or bootstrap_targets.empty:
        raise ValueError(f"Missing shared targets for {args.cell_type!r}")
    if len(bayesian_controls) != 7:
        raise ValueError(f"Expected seven Bayesian controls; found {len(bayesian_controls)}")

    control_names = bayesian_controls["cre"].astype(str).tolist()
    origin_bootstrap_controls = bootstrap_control_means(
        args.origin_bootstrap, args.cell_type, control_names, args.chunk_size
    )
    new_bootstrap_controls = bootstrap_control_means(
        args.new_bootstrap, args.cell_type, control_names, args.chunk_size
    )

    fig, axes = plt.subplots(1, 2, figsize=(14.2, 6.5))
    bayesian_stats = plot_panel(
        axes[0],
        bayesian_targets,
        control_names,
        bayesian_controls["origin_raw_activity_mean"].to_numpy(float),
        bayesian_controls["new_raw_activity_mean"].to_numpy(float),
        "Bayesian",
        args.t7_threshold,
    )
    bootstrap_stats = plot_panel(
        axes[1],
        bootstrap_targets,
        control_names,
        origin_bootstrap_controls,
        new_bootstrap_controls,
        "bootstrap",
        args.t7_threshold,
    )
    fig.suptitle(
        f"{args.cell_type}: origin versus new activity; T7 ≥ {args.t7_threshold:g}",
        fontsize=16,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    figures_dir = comparison_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    stem = (
        f"{slug(args.cell_type)}_bayesian_and_bootstrap_activity_origin_vs_new_"
        f"t7_ge{threshold_token}"
    )
    for suffix in ("png", "pdf"):
        fig.savefig(figures_dir / f"{stem}.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(json.dumps({"bayesian": bayesian_stats, "bootstrap": bootstrap_stats}, indent=2))


if __name__ == "__main__":
    main()
