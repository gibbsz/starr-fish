#!/usr/bin/env python3
"""Pair-level count diagnostics for the lower metacell Bayesian activity mode."""

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


BOOTSTRAP_METHOD = "Bootstrap"
JOINT_METHOD = "Joint"
META_METHOD = "Metacell Bayesian"
METHODS = (BOOTSTRAP_METHOD, JOINT_METHOD, META_METHOD)


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
        "--min-lower-probability",
        type=float,
        default=0.5,
        help="Keep pairs with posterior probability of lower GMM component at least this value.",
    )
    parser.add_argument("--background-sample", type=int, default=30_000)
    parser.add_argument("--selected-sample", type=int, default=60_000)
    parser.add_argument(
        "--figures-dir", type=Path, default=FIGURES_WORK
    )
    parser.add_argument(
        "--stem", default="method_activity_metacell_low_mode_pair_count_diagnostics"
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


def build_pair_table(args: argparse.Namespace) -> tuple[pd.DataFrame, dict]:
    matrices, pair_t7, pair_cre, cell_counts, _nanopore_counts, metadata = pm.prepare_base(
        args, METHODS
    )
    wide = pm.stack_methods(matrices, METHODS)
    total_t7 = pm.pair_count_series(pair_t7, "total_t7").reindex(wide.index)
    total_ccre = pm.pair_count_series(pair_cre, "total_ccre").reindex(wide.index)
    groups = wide.index.get_level_values("group").astype(str)
    n_cells = pd.Series(groups, index=wide.index).map(cell_counts.astype(float))
    data = wide.copy()
    data["total_t7"] = total_t7
    data["total_ccre"] = total_ccre
    data["n_cells"] = n_cells
    data = data.reset_index()
    return data, metadata


def assign_low_mode(data: pd.DataFrame, min_probability: float) -> tuple[pd.DataFrame, dict]:
    finite = np.isfinite(data[META_METHOD].to_numpy(float))
    x = data.loc[finite, [META_METHOD]].to_numpy(float)
    if len(x) < 10 or np.nanstd(x) == 0:
        raise ValueError("Not enough finite metacell activity values for a two-mode fit")
    model = GaussianMixture(n_components=2, random_state=0).fit(x)
    means = model.means_.ravel()
    lower_component = int(np.argmin(means))
    upper_component = int(np.argmax(means))
    probabilities = np.full(len(data), np.nan, dtype=float)
    labels = np.full(len(data), "unassigned", dtype=object)
    finite_probs = model.predict_proba(x)[:, lower_component]
    finite_labels = np.where(model.predict(x) == lower_component, "lower", "upper")
    probabilities[finite] = finite_probs
    labels[finite] = finite_labels
    out = data.copy()
    out["metacell_activity_mode"] = labels
    out["metacell_lower_mode_probability"] = probabilities
    out["selected_low_mode"] = (
        out["metacell_activity_mode"].eq("lower")
        & out["metacell_lower_mode_probability"].ge(min_probability)
    )
    metadata = {
        "gmm_component_means": {
            "lower": float(means[lower_component]),
            "upper": float(means[upper_component]),
        },
        "gmm_component_weights": {
            "lower": float(model.weights_[lower_component]),
            "upper": float(model.weights_[upper_component]),
        },
        "n_finite_metacell_pairs": int(finite.sum()),
        "n_selected_low_mode_pairs": int(out["selected_low_mode"].sum()),
        "min_lower_probability": float(min_probability),
    }
    return out, metadata


def summary_stats(selected: pd.DataFrame) -> dict:
    keys = [
        "n_cells",
        "total_t7",
        "total_ccre",
        BOOTSTRAP_METHOD,
        JOINT_METHOD,
        META_METHOD,
    ]
    summary = {"n_pairs": int(len(selected))}
    for key in keys:
        values = pd.to_numeric(selected[key], errors="coerce")
        summary[key] = {
            "median": float(values.median()) if values.notna().any() else np.nan,
            "q25": float(values.quantile(0.25)) if values.notna().any() else np.nan,
            "q75": float(values.quantile(0.75)) if values.notna().any() else np.nan,
        }
    summary["n_zero_t7_pairs"] = int(selected["total_t7"].le(0).sum())
    summary["n_zero_ccre_pairs"] = int(selected["total_ccre"].le(0).sum())
    return summary


def sampled(frame: pd.DataFrame, n: int, seed: int = 0) -> pd.DataFrame:
    if len(frame) <= n:
        return frame
    return frame.sample(n, random_state=seed)


def plot_diagnostic(
    data: pd.DataFrame,
    gmm_metadata: dict,
    args: argparse.Namespace,
) -> dict:
    sns.set_theme(context="paper", style="white")
    rng = np.random.default_rng(0)
    selected = data[data["selected_low_mode"]].copy()
    finite_context = data[[BOOTSTRAP_METHOD, META_METHOD]].replace(
        [np.inf, -np.inf], np.nan
    )
    finite_context = data.loc[finite_context.notna().all(axis=1)].copy()
    finite_context = sampled(finite_context, args.background_sample)
    selected_context = selected[
        np.isfinite(selected[BOOTSTRAP_METHOD].to_numpy(float))
        & np.isfinite(selected[META_METHOD].to_numpy(float))
    ].copy()
    selected_context = sampled(selected_context, args.selected_sample, seed=1)
    selected_for_counts = sampled(selected, args.selected_sample, seed=2)

    output = args.figures_dir / f"{args.stem}.pdf"
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(15.8, 4.8),
        constrained_layout=True,
        gridspec_kw={"width_ratios": [1.1, 1.25, 0.95]},
    )

    ax = axes[0]
    ax.scatter(
        finite_context[BOOTSTRAP_METHOD],
        finite_context[META_METHOD],
        s=1.2,
        color="0.75",
        alpha=0.08,
        linewidths=0,
        rasterized=True,
    )
    ax.scatter(
        selected_context[BOOTSTRAP_METHOD],
        selected_context[META_METHOD],
        s=4,
        color="#d62728",
        alpha=0.42,
        linewidths=0,
        rasterized=True,
    )
    lower_mean = gmm_metadata["gmm_component_means"]["lower"]
    upper_mean = gmm_metadata["gmm_component_means"]["upper"]
    ax.axhline(lower_mean, color="#d62728", linewidth=1.1, alpha=0.85)
    ax.axhline(upper_mean, color="#4c78a8", linewidth=1.1, alpha=0.85)
    ax.set_xlabel("Bootstrap mean log activity")
    ax.set_ylabel("Metacell Bayesian activity")
    ax.set_title("Selected low-mode location")

    ax = axes[1]
    zero_ccre = selected_for_counts["total_ccre"].to_numpy(float) <= 0
    nonzero = ~zero_ccre
    x = np.log10(selected_for_counts["total_t7"].to_numpy(float) + 1.0)
    y = np.log10(selected_for_counts["total_ccre"].to_numpy(float) + 1.0)
    jitter = rng.normal(0.0, 0.012, size=(len(selected_for_counts), 2))
    if nonzero.any():
        count_norm = mcolors.LogNorm(
            vmin=max(1.0, float(selected_for_counts.loc[nonzero, "n_cells"].min())),
            vmax=max(2.0, float(selected_for_counts.loc[nonzero, "n_cells"].max())),
        )
        sc = ax.scatter(
            x[nonzero] + jitter[nonzero, 0],
            y[nonzero] + jitter[nonzero, 1],
            c=selected_for_counts.loc[nonzero, "n_cells"].to_numpy(float),
            norm=count_norm,
            cmap="viridis",
            s=7,
            alpha=0.42,
            linewidths=0,
            rasterized=True,
        )
        cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)
        cbar.set_label("n cells")
    if zero_ccre.any():
        ax.scatter(
            x[zero_ccre] + jitter[zero_ccre, 0],
            y[zero_ccre] + jitter[zero_ccre, 1],
            color="#6a3d9a",
            s=7,
            alpha=0.32,
            linewidths=0,
            rasterized=True,
            label="cCRE count = 0",
        )
        ax.legend(frameon=False, loc="lower right")
    max_count = max(
        float(selected["total_t7"].max()),
        float(selected["total_ccre"].max()),
        10.0,
    )
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
    ax.set_title("Raw low-mode pair counts")

    ax = axes[2]
    long = selected[["n_cells", "total_t7", "total_ccre"]].rename(
        columns={
            "n_cells": "n cells",
            "total_t7": "total T7",
            "total_ccre": "total cCRE",
        }
    )
    long = long.melt(var_name="quantity", value_name="value")
    long = sampled(long, args.selected_sample, seed=3)
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
    ax.set_title("Low-mode count and cell-count distributions")

    summary = summary_stats(selected)
    fig.suptitle(
        "Metacell Bayesian low activity-mode count diagnostics; "
        f"n={len(selected):,} cCRE-celltype pairs | "
        f"mean={lower_mean:.2f} vs upper mean={upper_mean:.2f} | "
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
    log("[metacell low-mode pair count diagnostics] loading activity and count matrices")
    data, metadata = build_pair_table(args)
    data, gmm_metadata = assign_low_mode(data, args.min_lower_probability)
    pairs_path = args.figures_dir / f"{args.stem}_pairs.csv"
    data[data["selected_low_mode"]].to_csv(pairs_path, index=False)
    log("[metacell low-mode pair count diagnostics] drawing plot")
    output_summary = plot_diagnostic(data, gmm_metadata, args)
    write_json(
        args.figures_dir / f"{args.stem}_manifest.json",
        {
            "diagnostic": (
                "count-diagnostics style plot for cCRE-celltype pairs assigned "
                "to the lower mode of metacell Bayesian activity values"
            ),
            "mode_model": (
                "GaussianMixture(n_components=2) fit to finite Metacell Bayesian "
                "activity values across cCRE-celltype pairs"
            ),
            "pair_table": str(pairs_path),
            "mode_metadata": gmm_metadata,
            "plot": output_summary,
            "metadata": metadata,
        },
    )
    log(f"[metacell low-mode pair count diagnostics] wrote {output_summary['output']}")


if __name__ == "__main__":
    main()
