#!/usr/bin/env python3
"""Plot cCRE versus T7 counts for persistent metacell lower-mode subclasses."""

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
from sklearn.mixture import GaussianMixture

from analysis_utils import ANALYSIS_DIR, DEFAULT_H5AD, LIBSIZE_CSV, log, write_json
import plot_method_activity_correlation as pm


META_METHOD = "Metacell Bayesian"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--correlations",
        type=Path,
        default=ANALYSIS_DIR
        / "results"
        / "tables"
        / "method_activity_atac_correlation_by_subclass_t7_filters.csv",
    )
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
        default=ANALYSIS_DIR
        / "results"
        / "ablation"
        / "bayesian_decoupled_no_dropout",
    )
    parser.add_argument(
        "--joint-dropout-bayesian-dir",
        type=Path,
        default=ANALYSIS_DIR / "results" / "ablation" / "bayesian_joint_dropout",
    )
    parser.add_argument(
        "--decoupled-dropout-bayesian-dir",
        type=Path,
        default=None,
        help="Bayesian decoupled directory with zero-inflated dropout.",
    )
    parser.add_argument(
        "--metacell-bayesian-dir",
        type=Path,
        default=ANALYSIS_DIR
        / "results"
        / "ablation"
        / "bayesian_bootstrap_metacells_size100_number100",
    )
    parser.add_argument("--h5ad", type=Path, default=DEFAULT_H5AD)
    parser.add_argument("--libsize-csv", type=Path, default=LIBSIZE_CSV)
    parser.add_argument("--bootstrap-log-chunk-size", type=int, default=250)
    parser.add_argument("--activity-calibration", choices=["none"], default="none")
    parser.add_argument(
        "--t7-thresholds",
        type=float,
        nargs="+",
        default=[5.0, 10.0, 20.0, 50.0, 100.0],
    )
    parser.add_argument(
        "--min-lower-filters",
        type=int,
        default=3,
        help="Keep subclasses assigned to the lower mode in at least this many T7 filters.",
    )
    parser.add_argument(
        "--min-plot-t7",
        type=float,
        default=5.0,
        help="Plot cCRE-subclass pairs with total T7 at least this value.",
    )
    parser.add_argument("--max-groups", type=int, default=60)
    parser.add_argument(
        "--figures-dir", type=Path, default=ANALYSIS_DIR / "results" / "figures"
    )
    parser.add_argument(
        "--stem", default="method_activity_metacell_persistent_lower_count_diagnostics"
    )
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


def assign_modes(correlations: pd.DataFrame, thresholds: list[float]) -> pd.DataFrame:
    meta = correlations[
        correlations["method"].eq(META_METHOD)
        & correlations["t7_threshold"].isin(thresholds)
        & np.isfinite(correlations["spearman_atac_cpm"].to_numpy(float))
    ].copy()
    frames = []
    for threshold, frame in meta.groupby("t7_threshold", sort=True):
        x = frame[["spearman_atac_cpm"]].to_numpy(float)
        if len(frame) < 10 or np.nanstd(x) == 0:
            out = frame.copy()
            out["mode"] = "single"
            out["lower_mode_probability"] = np.nan
            out["lower_mode_mean"] = np.nan
            out["upper_mode_mean"] = np.nan
            frames.append(out)
            continue
        model = GaussianMixture(n_components=2, random_state=0).fit(x)
        means = model.means_.ravel()
        lower_component = int(np.argmin(means))
        labels = model.predict(x)
        out = frame.copy()
        out["mode"] = np.where(labels == lower_component, "lower", "upper")
        out["lower_mode_probability"] = model.predict_proba(x)[:, lower_component]
        out["lower_mode_mean"] = float(means[lower_component])
        out["upper_mode_mean"] = float(means[np.argmax(means)])
        frames.append(out)
    return pd.concat(frames, ignore_index=True)


def persistent_subclasses(modes: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    lower = modes[modes["mode"].eq("lower")].copy()
    ranking = (
        lower.groupby("group")
        .agg(
            lower_filters=("t7_threshold", "nunique"),
            median_lower_rho=("spearman_atac_cpm", "median"),
            min_lower_rho=("spearman_atac_cpm", "min"),
            median_lower_probability=("lower_mode_probability", "median"),
            lower_thresholds=(
                "t7_threshold",
                lambda values: ",".join(f"{float(value):g}" for value in sorted(values)),
            ),
        )
        .sort_values(["lower_filters", "median_lower_rho"], ascending=[False, True])
    )
    selected = ranking[ranking["lower_filters"].ge(args.min_lower_filters)]
    if selected.empty:
        selected = ranking.head(args.max_groups)
    else:
        selected = selected.head(args.max_groups)
    return selected


def persistent_pair_table(
    persistent: pd.DataFrame,
    pair_t7: pd.DataFrame,
    pair_cre: pd.DataFrame,
    cell_counts: pd.Series,
    args: argparse.Namespace,
) -> pd.DataFrame:
    records = []
    for group, row in persistent.iterrows():
        if group not in pair_t7.index:
            continue
        t7 = pair_t7.loc[group].astype(float)
        cre = pair_cre.loc[group].astype(float)
        keep = t7.ge(args.min_plot_t7) & (np.isfinite(t7.to_numpy(float))) & (
            np.isfinite(cre.to_numpy(float))
        )
        for cre_name in t7.index[keep.to_numpy()]:
            records.append(
                {
                    "group": str(group),
                    "cre": str(cre_name),
                    "total_t7": float(t7.loc[cre_name]),
                    "total_ccre": float(cre.loc[cre_name]),
                    "n_cells": float(cell_counts.get(group, np.nan)),
                    "lower_filters": int(row["lower_filters"]),
                    "median_lower_rho": float(row["median_lower_rho"]),
                    "min_lower_rho": float(row["min_lower_rho"]),
                    "median_lower_probability": float(row["median_lower_probability"]),
                    "lower_thresholds": str(row["lower_thresholds"]),
                }
            )
    return pd.DataFrame(records)


def plot_cre_vs_t7(
    pairs: pd.DataFrame,
    persistent: pd.DataFrame,
    thresholds: list[float],
    args: argparse.Namespace,
) -> dict:
    output = args.figures_dir / f"{args.stem}_cre_vs_t7.pdf"
    if pairs.empty:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.text(0.5, 0.5, "No persistent lower-mode pairs", ha="center", va="center")
        ax.set_axis_off()
        fig.savefig(output, bbox_inches="tight")
        plt.close(fig)
        return {"output": str(output), "n_pairs": 0, "n_groups": int(len(persistent))}

    sns.set_theme(context="paper", style="white")
    fig, ax = plt.subplots(figsize=(7.2, 6.2), constrained_layout=True)

    x = np.log10(pairs["total_t7"].to_numpy(float) + 1.0)
    y = np.log10(pairs["total_ccre"].to_numpy(float) + 1.0)
    zero_cre = pairs["total_ccre"].to_numpy(float) <= 0
    nonzero = ~zero_cre
    if nonzero.any():
        norm = mcolors.BoundaryNorm(
            np.arange(
                pairs["lower_filters"].min() - 0.5,
                pairs["lower_filters"].max() + 1.5,
                1.0,
            ),
            ncolors=256,
        )
        sc = ax.scatter(
            x[nonzero],
            y[nonzero],
            c=pairs.loc[nonzero, "lower_filters"].to_numpy(float),
            cmap="viridis",
            norm=norm,
            s=7,
            alpha=0.35,
            linewidths=0,
            rasterized=True,
        )
        cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)
        cbar.set_label("Number of T7 filters assigned lower mode")
    if zero_cre.any():
        ax.scatter(
            x[zero_cre],
            y[zero_cre],
            color="#6a3d9a",
            s=7,
            alpha=0.35,
            linewidths=0,
            rasterized=True,
            label="cCRE count = 0",
        )

    max_count = max(float(pairs["total_t7"].max()), float(pairs["total_ccre"].max()), 10.0)
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
            lw=0.75,
            alpha=0.6,
        )
        if valid.any():
            ax.text(
                np.log10(grid[valid][-1] + 1),
                np.log10(yy[valid][-1] + 1),
                label,
                fontsize=7,
                color=color,
                ha="right",
                va="bottom",
            )
    for threshold in thresholds:
        if threshold >= args.min_plot_t7 and threshold <= max_count:
            ax.axvline(
                np.log10(threshold + 1.0),
                color="0.78",
                linewidth=0.55,
                linestyle="--",
                zorder=0,
            )
            ax.text(
                np.log10(threshold + 1.0),
                ax.get_ylim()[1],
                f"{threshold:g}",
                ha="center",
                va="top",
                fontsize=6,
                color="0.35",
            )

    add_count_ticks(ax, max_count, max_count)
    ax.set_xlabel("Total T7 counts, log10(count + 1)")
    ax.set_ylabel("Total cCRE counts, log10(count + 1)")
    ax.set_title(
        "Persistent lower-mode subclasses: cCRE counts versus T7 counts\n"
        f"{len(persistent):,} subclasses lower-mode in >= {args.min_lower_filters} T7 filters; "
        f"{len(pairs):,} cCRE-subclass pairs with T7 >= {args.min_plot_t7:g}"
    )
    if zero_cre.any():
        ax.legend(frameon=False, loc="lower right")
    sns.despine(fig=fig, ax=ax)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    return {
        "output": str(output),
        "n_pairs": int(len(pairs)),
        "n_groups": int(len(persistent)),
        "n_zero_ccre_pairs": int(zero_cre.sum()),
        "median_t7": float(pairs["total_t7"].median()),
        "median_ccre": float(pairs["total_ccre"].median()),
        "median_cells": float(pairs["n_cells"].median()),
    }


def main() -> None:
    args = parse_args()
    args.figures_dir.mkdir(parents=True, exist_ok=True)
    thresholds = sorted({float(threshold) for threshold in args.t7_thresholds})

    log("[persistent lower count diagnostics] assigning lower modes")
    correlations = pd.read_csv(args.correlations)
    modes = assign_modes(correlations, thresholds)
    persistent = persistent_subclasses(modes, args)

    log("[persistent lower count diagnostics] loading count matrices")
    matrices, pair_t7, pair_cre, cell_counts, _nanopore_counts, metadata = pm.prepare_base(
        args, (META_METHOD,)
    )
    pair_t7 = pair_t7.reindex(
        index=matrices[META_METHOD].index.astype(str),
        columns=matrices[META_METHOD].columns.astype(str),
    ).fillna(0.0)
    pair_cre = pair_cre.reindex(index=pair_t7.index, columns=pair_t7.columns).fillna(0.0)

    pairs = persistent_pair_table(persistent, pair_t7, pair_cre, cell_counts, args)
    pair_csv = args.figures_dir / f"{args.stem}_cre_vs_t7_pairs.csv"
    subclass_csv = args.figures_dir / f"{args.stem}_persistent_subclasses.csv"
    pairs.to_csv(pair_csv, index=False)
    persistent.to_csv(subclass_csv)

    log("[persistent lower count diagnostics] drawing cCRE-vs-T7 plot")
    plot_summary = plot_cre_vs_t7(pairs, persistent, thresholds, args)
    write_json(
        args.figures_dir / f"{args.stem}_manifest.json",
        {
            "diagnostic": (
                "persistent metacell lower-mode subclasses; plot shows only total "
                "cCRE counts versus total T7 counts"
            ),
            "mode_source": str(args.correlations),
            "mode_model": (
                "GaussianMixture(n_components=2) fit separately per T7 threshold "
                "to Metacell Bayesian spearman_atac_cpm; ATAC is used only to "
                "define the lower-mode subclasses and is not plotted"
            ),
            "thresholds": thresholds,
            "min_lower_filters": int(args.min_lower_filters),
            "min_plot_t7": float(args.min_plot_t7),
            "persistent_subclasses": str(subclass_csv),
            "pair_table": str(pair_csv),
            "plot": plot_summary,
            "metadata": metadata,
        },
    )
    log(f"[persistent lower count diagnostics] wrote {plot_summary['output']}")


if __name__ == "__main__":
    main()
