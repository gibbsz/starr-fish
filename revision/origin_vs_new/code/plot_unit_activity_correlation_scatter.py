#!/usr/bin/env python3
"""Scatter within-unit original-versus-new activity correlation for cCREs and cell types.

Every cCRE-cell-type pair in the shared overlap universe contributes to exactly
one point per panel: the left panel holds one point per cCRE (correlation taken
across its cell types) and the right panel one point per cell type (correlation
taken across its cCREs). The y axis is the within-unit Pearson correlation of
negative-control-centered posterior activity between the two experiments, and
the x axis is the number of overlap-filtered pairs supporting that unit.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Final

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from scipy.stats import pearsonr, spearmanr


HERE: Final[Path] = Path(__file__).resolve()
ANALYSIS_DIR: Final[Path] = HERE.parent.parent
DEFAULT_COMPARISON_DIR: Final[Path] = ANALYSIS_DIR / "results" / "comparison"
ORIGIN_ACTIVITY: Final[str] = "origin_effect_vs_mean_control_mean"
NEW_ACTIVITY: Final[str] = "new_effect_vs_mean_control_mean"
KEY: Final[list[str]] = ["group", "cre"]
UNIT_AXES: Final[tuple[tuple[str, str, str], ...]] = (
    ("cre", "cCRE", "cell types"),
    ("group", "cell type", "cCREs"),
)
SUPPORT_CMAP: Final[LinearSegmentedColormap] = LinearSegmentedColormap.from_list(
    "t7_support",
    plt.get_cmap("Blues")(np.linspace(0.30, 1.0, 256)),
)
REQUIRED_COLUMNS: Final[tuple[str, ...]] = (
    "group",
    "cre",
    ORIGIN_ACTIVITY,
    NEW_ACTIVITY,
    "origin_target_t7_total",
    "new_target_t7_total",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--t7-threshold", type=float, default=50)
    parser.add_argument(
        "--comparison-dir", type=Path, default=DEFAULT_COMPARISON_DIR
    )
    parser.add_argument(
        "--minimum-unit-pairs",
        type=int,
        default=10,
        help="Minimum overlap-filtered pairs required to retain a unit.",
    )
    parser.add_argument(
        "--label-count",
        type=int,
        default=3,
        help="Units labelled at each extreme of the correlation axis per panel.",
    )
    return parser.parse_args()


def token(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def lins_ccc(x: np.ndarray, y: np.ndarray) -> float:
    """Return Lin's concordance correlation coefficient using population moments."""
    if x.size < 2:
        return float("nan")
    x_mean = float(x.mean())
    y_mean = float(y.mean())
    covariance = float(np.mean((x - x_mean) * (y - y_mean)))
    denominator = float(x.var() + y.var() + (x_mean - y_mean) ** 2)
    if denominator == 0.0:
        return 1.0 if np.array_equal(x, y) else float("nan")
    return 2.0 * covariance / denominator


def read_pairs(comparison_dir: Path, t7_threshold: float) -> pd.DataFrame:
    path = (
        comparison_dir
        / "tables"
        / f"overlap_t7_ge{token(t7_threshold)}_pair_comparison.csv.gz"
    )
    if not path.exists():
        raise FileNotFoundError(f"Missing overlap pair table: {path}")
    pairs = pd.read_csv(path)
    missing = [column for column in REQUIRED_COLUMNS if column not in pairs.columns]
    if missing:
        raise ValueError(f"{path} lacks required columns: {missing}")
    if pairs.duplicated(subset=KEY).any():
        raise ValueError(f"{path} contains duplicated group/cre keys")
    pairs = pairs.loc[
        np.isfinite(pairs[ORIGIN_ACTIVITY]) & np.isfinite(pairs[NEW_ACTIVITY])
    ].copy()
    if pairs.empty:
        raise ValueError(f"{path} has no pairs with finite activity in both fits")
    pairs["min_target_t7_total"] = pairs[
        ["origin_target_t7_total", "new_target_t7_total"]
    ].min(axis=1)
    return pairs


def unit_metrics(
    pairs: pd.DataFrame, unit_column: str, minimum_pairs: int
) -> pd.DataFrame:
    """Return one row per unit with within-unit old/new concordance metrics."""
    records: list[dict[str, object]] = []
    for unit, frame in pairs.groupby(unit_column, sort=True):
        if len(frame) < minimum_pairs:
            continue
        x = frame[ORIGIN_ACTIVITY].to_numpy(np.float64)
        y = frame[NEW_ACTIVITY].to_numpy(np.float64)
        constant = x.std() == 0.0 or y.std() == 0.0
        pearson = None if constant else pearsonr(x, y)
        records.append(
            {
                "unit_axis": unit_column,
                "unit": str(unit),
                "n_supported_pairs": int(len(frame)),
                "pearson_r": float("nan") if pearson is None else float(pearson.statistic),
                "pearson_p": float("nan") if pearson is None else float(pearson.pvalue),
                "spearman_rho": float("nan")
                if constant
                else float(spearmanr(x, y).statistic),
                "lins_ccc": lins_ccc(x, y),
                "origin_mean_activity": float(x.mean()),
                "new_mean_activity": float(y.mean()),
                "mean_change_new_minus_origin": float((y - x).mean()),
                "median_min_target_t7_total": float(
                    frame["min_target_t7_total"].median()
                ),
            }
        )
    metrics = pd.DataFrame.from_records(records)
    if metrics.empty:
        raise ValueError(
            f"No {unit_column} retained at >= {minimum_pairs} overlap-filtered pairs"
        )
    return metrics


def pooled_correlations(pairs: pd.DataFrame) -> tuple[float, float]:
    x = pairs[ORIGIN_ACTIVITY].to_numpy(np.float64)
    y = pairs[NEW_ACTIVITY].to_numpy(np.float64)
    return float(pearsonr(x, y).statistic), float(spearmanr(x, y).statistic)


def annotate_extremes(
    ax: plt.Axes, metrics: pd.DataFrame, label_count: int
) -> None:
    """Label the least and most concordant units, decluttered along y."""
    if label_count <= 0:
        return
    finite = metrics.loc[np.isfinite(metrics["pearson_r"])].sort_values("pearson_r")
    if finite.empty:
        return
    selected = pd.concat(
        [finite.head(label_count), finite.tail(label_count)]
    ).drop_duplicates(subset="unit")
    x_low, x_high = ax.get_xlim()
    y_low, y_high = ax.get_ylim()
    midpoint = 0.5 * (x_low + x_high)
    minimum_separation = 0.055 * (y_high - y_low)
    x_gap = 0.02 * (x_high - x_low)
    last_placed: dict[bool, float] = {}
    for _, row in selected.iterrows():
        x_value = float(row["n_supported_pairs"])
        y_value = float(row["pearson_r"])
        to_left = x_value > midpoint
        placed = max(
            y_value, last_placed.get(to_left, -np.inf) + minimum_separation
        )
        last_placed[to_left] = placed
        ax.annotate(
            str(row["unit"]),
            xy=(x_value, y_value),
            xytext=(x_value - x_gap if to_left else x_value + x_gap, placed),
            textcoords="data",
            ha="right" if to_left else "left",
            va="center",
            fontsize=6.5,
            color="0.25",
            zorder=5,
            arrowprops={
                "arrowstyle": "-",
                "color": "0.65",
                "linewidth": 0.4,
                "shrinkA": 0.5,
                "shrinkB": 2.0,
            },
        )


def plot_units(
    panels: dict[str, pd.DataFrame],
    pairs: pd.DataFrame,
    output_stem: Path,
    *,
    t7_threshold: float,
    minimum_pairs: int,
    label_count: int,
) -> None:
    pooled_pearson, pooled_spearman = pooled_correlations(pairs)
    support = pd.concat(
        [frame["median_min_target_t7_total"] for frame in panels.values()]
    ).to_numpy(np.float64)
    norm = LogNorm(vmin=max(float(support.min()), 1.0), vmax=float(support.max()))

    fig, axes = plt.subplots(
        1, 2, figsize=(12.4, 5.6), sharey=True, layout="constrained"
    )
    scatter = None
    for ax, (unit_column, unit_label, across_label) in zip(axes, UNIT_AXES):
        metrics = panels[unit_column]
        counts = metrics["n_supported_pairs"].to_numpy(float)
        span = max(float(counts.max() - counts.min()), 1.0)
        ax.set_xlim(counts.min() - 0.12 * span, counts.max() + 0.12 * span)
        ax.set_ylim(-1.05, 1.05)
        ax.axhline(0.0, color="0.55", linewidth=0.8, linestyle=":", zorder=1)
        ax.axhline(
            pooled_pearson,
            color="#CC6677",
            linewidth=1.4,
            linestyle="--",
            zorder=2,
            label=f"Pooled pair-level r = {pooled_pearson:.3f}",
        )
        finite = metrics.loc[np.isfinite(metrics["pearson_r"]), "pearson_r"]
        ax.axhline(
            float(finite.mean()),
            color="0.35",
            linewidth=1.4,
            linestyle="-.",
            zorder=2,
            label=f"Mean within-{unit_label} r = {finite.mean():.3f}",
        )
        scatter = ax.scatter(
            counts,
            metrics["pearson_r"].to_numpy(float),
            c=metrics["median_min_target_t7_total"].to_numpy(float),
            cmap=SUPPORT_CMAP,
            norm=norm,
            s=46,
            edgecolors="white",
            linewidths=0.5,
            zorder=3,
        )
        annotate_extremes(ax, metrics, label_count)
        ax.set_xlabel(f"Overlap-filtered pairs per {unit_label}")
        ax.set_title(
            f"One point per {unit_label} (n = {len(metrics):,}); "
            f"r taken across its {across_label}\n"
            f"median r = {finite.median():.3f}",
            fontsize=9.5,
        )
        ax.grid(color="0.90", linewidth=0.7, zorder=0)
        ax.legend(frameon=False, fontsize=8, loc="lower left")
        ax.spines[["top", "right"]].set_visible(False)

    axes[0].set_ylabel("Within-unit Pearson r (original vs new low-dose activity)")
    if scatter is not None:
        colorbar = fig.colorbar(scatter, ax=axes, fraction=0.030, pad=0.015)
        colorbar.set_label("Median per-pair min(original, new) target T7", fontsize=8)
        colorbar.ax.tick_params(labelsize=7.5)
    fig.suptitle(
        "Original versus new low-dose activity concordance per cCRE and per cell type\n"
        f"Overlap filter: T7 ≥ {t7_threshold:g} in both datasets "
        f"({len(pairs):,} cCRE–cell-type pairs; pooled Spearman ρ = "
        f"{pooled_spearman:.3f}); units require ≥ {minimum_pairs} pairs",
        fontsize=10.5,
        y=1.10,
    )
    for suffix in ("png", "pdf"):
        fig.savefig(output_stem.with_suffix(f".{suffix}"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.minimum_unit_pairs < 3:
        raise ValueError("--minimum-unit-pairs must be at least 3")
    tables_dir = args.comparison_dir / "tables"
    figures_dir = args.comparison_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    pairs = read_pairs(args.comparison_dir, args.t7_threshold)
    panels = {
        unit_column: unit_metrics(pairs, unit_column, args.minimum_unit_pairs)
        for unit_column, _, _ in UNIT_AXES
    }

    stem = f"unit_activity_correlation_scatter_t7_ge{token(args.t7_threshold)}"
    plot_units(
        panels,
        pairs,
        figures_dir / stem,
        t7_threshold=args.t7_threshold,
        minimum_pairs=args.minimum_unit_pairs,
        label_count=args.label_count,
    )

    values = pd.concat(panels.values(), ignore_index=True)
    values_path = tables_dir / f"{stem}_values.csv"
    values.to_csv(values_path, index=False)

    pooled_pearson, pooled_spearman = pooled_correlations(pairs)
    manifest = {
        "figure_stem": stem,
        "source_table": str(
            tables_dir
            / f"overlap_t7_ge{token(args.t7_threshold)}_pair_comparison.csv.gz"
        ),
        "values_table": str(values_path),
        "t7_threshold": float(args.t7_threshold),
        "minimum_unit_pairs": int(args.minimum_unit_pairs),
        "n_pairs": int(len(pairs)),
        "pooled_pearson_r": pooled_pearson,
        "pooled_spearman_rho": pooled_spearman,
        "panels": {
            unit_column: {
                "n_units": int(len(frame)),
                "n_pairs_in_retained_units": int(frame["n_supported_pairs"].sum()),
                "mean_pearson_r": float(frame["pearson_r"].mean()),
                "median_pearson_r": float(frame["pearson_r"].median()),
                "std_pearson_r": float(frame["pearson_r"].std(ddof=1))
                if len(frame) >= 2
                else float("nan"),
                "mean_spearman_rho": float(frame["spearman_rho"].mean()),
                "mean_lins_ccc": float(frame["lins_ccc"].mean()),
                "n_units_with_negative_r": int((frame["pearson_r"] < 0).sum()),
                "n_units_with_nonfinite_r": int(
                    (~np.isfinite(frame["pearson_r"])).sum()
                ),
            }
            for unit_column, frame in panels.items()
        },
    }
    (tables_dir / f"{stem}_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
