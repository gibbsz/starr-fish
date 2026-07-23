#!/usr/bin/env python3
"""Diagnose the lower mode in metacell Bayesian ATAC correlations."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import spearmanr
from sklearn.mixture import GaussianMixture

from analysis_utils import ANALYSIS_DIR, DEFAULT_H5AD, STARRFISH_ROOT, log, write_json
from plot_activity_atac_correlation import (
    METHODS,
    METHOD_COLORS,
    METHOD_LABELS,
    read_atac_cpm_subset,
    read_cre_to_peak,
    threshold_suffix,
    load_activity,
)
from plot_method_activity_correlation import pair_count_totals


META_METHOD = "Metacell Bayesian"
COMPARE_METHODS = ("Bootstrap", "Joint", META_METHOD)
MODE_COLORS = {"lower": "#b2182b", "upper": "#2166ac"}


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
    parser.add_argument(
        "--atac-cpm",
        type=Path,
        default=STARRFISH_ROOT / "Data" / "ATAC" / "cpm_peakBysubclass.csv",
    )
    parser.add_argument(
        "--cre-info",
        type=Path,
        default=ANALYSIS_DIR
        / "results"
        / "ablation"
        / "bayesian_joint"
        / "cre_info.csv",
    )
    parser.add_argument("--h5ad", type=Path, default=DEFAULT_H5AD)
    parser.add_argument("--bootstrap-log-chunk-size", type=int, default=250)
    parser.add_argument(
        "--figures-dir", type=Path, default=ANALYSIS_DIR / "results" / "figures"
    )
    parser.add_argument(
        "--tables-dir", type=Path, default=ANALYSIS_DIR / "results" / "tables"
    )
    parser.add_argument("--atac-chunk-size", type=int, default=100_000)
    parser.add_argument(
        "--t7-thresholds",
        type=float,
        nargs="+",
        default=[5.0, 10.0, 20.0, 50.0, 100.0],
    )
    parser.add_argument("--example-threshold", type=float, default=50.0)
    parser.add_argument("--max-example-groups", type=int, default=4)
    parser.add_argument(
        "--activity-calibration",
        choices=["calibrated", "none"],
        default="none",
    )
    parser.add_argument(
        "--stem", default="method_activity_atac_metacell_bimodality_diagnostics"
    )
    return parser.parse_args()


def assign_metacell_modes(
    correlations: pd.DataFrame,
    thresholds: list[float],
) -> pd.DataFrame:
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
        upper_component = int(np.argmax(means))
        labels = model.predict(x)
        lower_probability = model.predict_proba(x)[:, lower_component]
        out = frame.copy()
        out["mode"] = np.where(labels == lower_component, "lower", "upper")
        out["lower_mode_probability"] = lower_probability
        out["lower_mode_mean"] = float(means[lower_component])
        out["upper_mode_mean"] = float(means[upper_component])
        out["lower_mode_weight"] = float(model.weights_[lower_component])
        out["upper_mode_weight"] = float(model.weights_[upper_component])
        frames.append(out)
    return pd.concat(frames, ignore_index=True)


def _finite_spearman(x: pd.Series, y: pd.Series) -> float:
    x_values = pd.to_numeric(x, errors="coerce").to_numpy(float)
    y_values = pd.to_numeric(y, errors="coerce").to_numpy(float)
    finite = np.isfinite(x_values) & np.isfinite(y_values)
    if finite.sum() < 3:
        return np.nan
    if np.nanstd(x_values[finite]) == 0 or np.nanstd(y_values[finite]) == 0:
        return np.nan
    return float(spearmanr(x_values[finite], y_values[finite]).statistic)


def load_matched_inputs(args: argparse.Namespace) -> tuple[
    dict[str, pd.DataFrame], pd.DataFrame, pd.DataFrame, pd.DataFrame
]:
    activity, _ = load_activity(args)
    all_cres = pd.Index(sorted(set().union(*(set(m.columns) for m in activity.values()))))
    cre_to_peak = read_cre_to_peak(args.cre_info, all_cres)
    atac, _ = read_atac_cpm_subset(
        args.atac_cpm,
        cre_to_peak,
        pd.Index(next(iter(activity.values())).index.astype(str)),
        args.atac_chunk_size,
    )
    common_groups = pd.Index(next(iter(activity.values())).index.astype(str)).intersection(
        atac.index.astype(str)
    )
    common_cres = all_cres.intersection(atac.columns.astype(str))
    activity = {
        method: matrix.reindex(index=common_groups, columns=common_cres)
        for method, matrix in activity.items()
    }
    atac = atac.reindex(index=common_groups, columns=common_cres)
    pair_t7, pair_cre = pair_count_totals(args.h5ad, common_groups, common_cres)
    pair_t7 = pair_t7.reindex(index=common_groups, columns=common_cres).fillna(0.0)
    pair_cre = pair_cre.reindex(index=common_groups, columns=common_cres).fillna(0.0)
    return activity, atac, pair_t7, pair_cre


def diagnosis_table(
    mode_assignments: pd.DataFrame,
    correlations: pd.DataFrame,
    activity: dict[str, pd.DataFrame],
    atac: pd.DataFrame,
    pair_t7: pd.DataFrame,
    pair_cre: pd.DataFrame,
    args: argparse.Namespace,
) -> pd.DataFrame:
    rho_wide = (
        correlations.pivot_table(
            index=["t7_threshold", "group"],
            columns="method",
            values="spearman_atac_cpm",
            aggfunc="first",
        )
        .reset_index()
        .rename_axis(columns=None)
    )
    n_pairs_wide = (
        correlations.pivot_table(
            index=["t7_threshold", "group"],
            columns="method",
            values="n_pairs",
            aggfunc="first",
        )
        .reset_index()
        .rename_axis(columns=None)
    )
    n_pairs_wide = n_pairs_wide.rename(
        columns={method: f"{method}_n_pairs" for method in METHODS if method in n_pairs_wide}
    )
    table = mode_assignments[
        [
            "t7_threshold",
            "group",
            "mode",
            "lower_mode_probability",
            "lower_mode_mean",
            "upper_mode_mean",
            "spearman_atac_cpm",
            "n_pairs",
        ]
    ].rename(
        columns={
            "spearman_atac_cpm": "metacell_spearman_atac_cpm",
            "n_pairs": "metacell_n_pairs",
        }
    )
    table = table.merge(rho_wide, on=["t7_threshold", "group"], how="left")
    table = table.merge(n_pairs_wide, on=["t7_threshold", "group"], how="left")

    source_counts_path = args.metacell_bayesian_dir / "source_subclass_cell_counts.csv"
    if source_counts_path.exists():
        source_counts = pd.read_csv(source_counts_path).rename(
            columns={"subclass": "group", "n_source_cells": "source_cells"}
        )
        table = table.merge(source_counts, on="group", how="left")
    else:
        table["source_cells"] = np.nan

    metric_rows = []
    common_cres = pair_t7.columns.astype(str)
    for row in table[["t7_threshold", "group"]].itertuples(index=False):
        threshold = float(row.t7_threshold)
        group = str(row.group)
        if group not in pair_t7.index:
            metric_rows.append({"t7_threshold": threshold, "group": group})
            continue
        keep = (
            pair_t7.loc[group, common_cres].ge(threshold)
            & np.isfinite(atac.loc[group, common_cres].to_numpy(float))
            & np.isfinite(activity[META_METHOD].loc[group, common_cres].to_numpy(float))
        )
        cres = common_cres[keep.to_numpy()]
        t7_values = pair_t7.loc[group, cres].to_numpy(float)
        cre_values = pair_cre.loc[group, cres].to_numpy(float)
        atac_values = atac.loc[group, cres].to_numpy(float)
        metrics = {
            "t7_threshold": threshold,
            "group": group,
            "n_diagnosis_pairs": int(len(cres)),
            "sum_t7": float(np.nansum(t7_values)) if len(cres) else np.nan,
            "median_t7": float(np.nanmedian(t7_values)) if len(cres) else np.nan,
            "sum_cre": float(np.nansum(cre_values)) if len(cres) else np.nan,
            "median_cre": float(np.nanmedian(cre_values)) if len(cres) else np.nan,
            "median_atac_cpm": float(np.nanmedian(atac_values)) if len(cres) else np.nan,
            "activity_rho_metacell_joint": _finite_spearman(
                activity[META_METHOD].loc[group, cres],
                activity["Joint"].loc[group, cres],
            ),
            "activity_rho_metacell_bootstrap": _finite_spearman(
                activity[META_METHOD].loc[group, cres],
                activity["Bootstrap"].loc[group, cres],
            ),
            "activity_rho_joint_bootstrap": _finite_spearman(
                activity["Joint"].loc[group, cres],
                activity["Bootstrap"].loc[group, cres],
            ),
        }
        metric_rows.append(metrics)
    table = table.merge(
        pd.DataFrame(metric_rows), on=["t7_threshold", "group"], how="left"
    )
    return table


def plot_mode_histograms(diagnosis: pd.DataFrame, thresholds: list[float]) -> plt.Figure:
    sns.set_theme(context="paper", style="white")
    fig, axes = plt.subplots(
        1,
        len(thresholds),
        figsize=(3.3 * len(thresholds), 3.2),
        constrained_layout=True,
        sharey=True,
    )
    axes = np.ravel(axes)
    for ax, threshold in zip(axes, thresholds):
        frame = diagnosis[np.isclose(diagnosis["t7_threshold"], threshold)].copy()
        sns.histplot(
            data=frame,
            x="metacell_spearman_atac_cpm",
            hue="mode",
            hue_order=["lower", "upper"],
            palette=MODE_COLORS,
            bins=28,
            element="step",
            stat="count",
            common_norm=False,
            ax=ax,
        )
        lower_mean = frame["lower_mode_mean"].dropna()
        upper_mean = frame["upper_mode_mean"].dropna()
        if not lower_mean.empty:
            ax.axvline(lower_mean.iloc[0], color=MODE_COLORS["lower"], linewidth=1.2)
        if not upper_mean.empty:
            ax.axvline(upper_mean.iloc[0], color=MODE_COLORS["upper"], linewidth=1.2)
        counts = frame["mode"].value_counts()
        ax.set_title(
            f"T7 >= {threshold:g}\n"
            f"lower={int(counts.get('lower', 0))}, upper={int(counts.get('upper', 0))}"
        )
        ax.set_xlabel("Metacell Spearman rho")
        ax.axvline(0, color="black", linewidth=0.7, alpha=0.5)
    axes[0].set_ylabel("Subclasses")
    sns.despine(fig=fig)
    return fig


def plot_method_rho_scatter(
    diagnosis: pd.DataFrame,
    thresholds: list[float],
) -> plt.Figure:
    sns.set_theme(context="paper", style="white")
    x_methods = ("Joint", "Bootstrap")
    fig, axes = plt.subplots(
        len(thresholds),
        len(x_methods),
        figsize=(8.2, 2.6 * len(thresholds)),
        constrained_layout=True,
        squeeze=False,
    )
    for row, threshold in enumerate(thresholds):
        frame = diagnosis[np.isclose(diagnosis["t7_threshold"], threshold)].copy()
        for col, x_method in enumerate(x_methods):
            ax = axes[row, col]
            pair = frame[
                np.isfinite(frame[x_method].to_numpy(float))
                & np.isfinite(frame[META_METHOD].to_numpy(float))
            ].copy()
            sizes = np.clip(pair["metacell_n_pairs"].to_numpy(float), 10, 140)
            ax.scatter(
                pair[x_method],
                pair[META_METHOD],
                c=pair["mode"].map(MODE_COLORS),
                s=sizes,
                alpha=0.72,
                linewidths=0,
                rasterized=True,
            )
            lo = np.nanpercentile(
                np.concatenate([pair[x_method].to_numpy(float), pair[META_METHOD].to_numpy(float)]),
                1,
            )
            hi = np.nanpercentile(
                np.concatenate([pair[x_method].to_numpy(float), pair[META_METHOD].to_numpy(float)]),
                99,
            )
            pad = max((hi - lo) * 0.08, 0.05)
            lo, hi = max(-1, lo - pad), min(1, hi + pad)
            ax.plot([lo, hi], [lo, hi], color="black", linewidth=0.7, alpha=0.55)
            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)
            pearson = pair[x_method].corr(pair[META_METHOD], method="pearson")
            spearman = pair[x_method].corr(pair[META_METHOD], method="spearman")
            ax.text(
                0.04,
                0.96,
                f"r={pearson:.3f}\nrho={spearman:.3f}\nn={len(pair):,}",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=8,
                bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
            )
            if row == len(thresholds) - 1:
                ax.set_xlabel(f"{x_method} ATAC rho")
            else:
                ax.set_xlabel("")
            ax.set_ylabel("Metacell ATAC rho" if col == 0 else "")
            if col == 0:
                ax.set_title(f"T7 >= {threshold:g}")
            else:
                ax.set_title("")
    handles = [
        plt.Line2D([0], [0], marker="o", linestyle="", color=color, label=mode)
        for mode, color in MODE_COLORS.items()
    ]
    fig.legend(handles=handles, title="Metacell mode", loc="upper right", frameon=False)
    sns.despine(fig=fig)
    return fig


def plot_metric_boxplots(diagnosis: pd.DataFrame) -> plt.Figure:
    sns.set_theme(context="paper", style="white")
    metrics = [
        ("metacell_n_pairs", "ATAC cCRE pairs"),
        ("source_cells", "Source cells"),
        ("median_t7", "Median T7 per pair"),
        ("median_cre", "Median cCRE per pair"),
        ("median_atac_cpm", "Median ATAC CPM"),
        ("activity_rho_metacell_joint", "Metacell vs Joint activity rho"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 7.0), constrained_layout=True)
    axes = axes.ravel()
    for ax, (metric, label) in zip(axes, metrics):
        frame = diagnosis[
            diagnosis["mode"].isin(["lower", "upper"])
            & np.isfinite(diagnosis[metric].to_numpy(float))
        ].copy()
        sns.boxplot(
            data=frame,
            x="t7_threshold",
            y=metric,
            hue="mode",
            hue_order=["lower", "upper"],
            palette=MODE_COLORS,
            showfliers=False,
            ax=ax,
        )
        sns.stripplot(
            data=frame,
            x="t7_threshold",
            y=metric,
            hue="mode",
            hue_order=["lower", "upper"],
            palette=MODE_COLORS,
            dodge=True,
            size=1.6,
            alpha=0.25,
            linewidth=0,
            legend=False,
            rasterized=True,
            ax=ax,
        )
        if metric not in {"activity_rho_metacell_joint"}:
            ax.set_yscale("symlog", linthresh=1)
        ax.set_xlabel("T7 filter")
        ax.set_ylabel(label)
        if ax.get_legend() is not None:
            ax.get_legend().remove()
    handles = [
        plt.Line2D([0], [0], marker="s", linestyle="", color=color, label=mode)
        for mode, color in MODE_COLORS.items()
    ]
    fig.legend(handles=handles, title="Metacell mode", loc="upper center", ncol=2, frameon=False)
    sns.despine(fig=fig)
    return fig


def plot_persistent_lower_heatmap(
    diagnosis: pd.DataFrame,
    thresholds: list[float],
    max_groups: int = 40,
) -> plt.Figure:
    lower = diagnosis[diagnosis["mode"].eq("lower")].copy()
    if lower.empty:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.text(0.5, 0.5, "No lower-mode subclasses", ha="center", va="center")
        ax.set_axis_off()
        return fig
    ranking = (
        lower.groupby("group")
        .agg(
            lower_filters=("t7_threshold", "nunique"),
            median_lower_rho=("metacell_spearman_atac_cpm", "median"),
            min_lower_rho=("metacell_spearman_atac_cpm", "min"),
        )
        .sort_values(["lower_filters", "median_lower_rho"], ascending=[False, True])
        .head(max_groups)
    )
    selected = ranking.index
    matrix = (
        diagnosis[diagnosis["group"].isin(selected)]
        .pivot(index="group", columns="t7_threshold", values="metacell_spearman_atac_cpm")
        .reindex(index=selected, columns=thresholds)
    )
    mode_matrix = (
        diagnosis[diagnosis["group"].isin(selected)]
        .assign(is_lower=lambda x: x["mode"].eq("lower").astype(float))
        .pivot(index="group", columns="t7_threshold", values="is_lower")
        .reindex(index=selected, columns=thresholds)
    )
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(11.5, max(5.5, 0.17 * len(selected))),
        constrained_layout=True,
        gridspec_kw={"width_ratios": [2.4, 0.8]},
    )
    sns.heatmap(
        matrix,
        cmap="RdBu_r",
        center=0,
        vmin=-0.6,
        vmax=0.6,
        linewidths=0.2,
        linecolor="white",
        cbar_kws={"label": "Metacell ATAC Spearman rho"},
        ax=axes[0],
    )
    axes[0].set_xlabel("T7 filter")
    axes[0].set_ylabel("Subclass")
    axes[0].set_title("Subclasses repeatedly assigned to the lower mode")
    sns.heatmap(
        mode_matrix,
        cmap=sns.color_palette(["#f0f0f0", MODE_COLORS["lower"]], as_cmap=True),
        vmin=0,
        vmax=1,
        linewidths=0.2,
        linecolor="white",
        cbar=False,
        ax=axes[1],
    )
    axes[1].set_xlabel("T7 filter")
    axes[1].set_ylabel("")
    axes[1].set_yticklabels([])
    axes[1].set_title("Lower mode")
    return fig


def example_groups(diagnosis: pd.DataFrame, threshold: float, max_groups: int) -> list[str]:
    frame = diagnosis[
        np.isclose(diagnosis["t7_threshold"], threshold) & diagnosis["mode"].eq("lower")
    ].copy()
    if frame.empty:
        frame = diagnosis[diagnosis["mode"].eq("lower")].copy()
    return (
        frame.sort_values(["metacell_spearman_atac_cpm", "metacell_n_pairs"])
        .head(max_groups)["group"]
        .astype(str)
        .tolist()
    )


def plot_example_pair_scatter(
    groups: list[str],
    threshold: float,
    activity: dict[str, pd.DataFrame],
    atac: pd.DataFrame,
    pair_t7: pd.DataFrame,
) -> plt.Figure:
    sns.set_theme(context="paper", style="white")
    if not groups:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.text(0.5, 0.5, "No example lower-mode subclasses", ha="center", va="center")
        ax.set_axis_off()
        return fig
    fig, axes = plt.subplots(
        len(groups),
        len(COMPARE_METHODS),
        figsize=(4.0 * len(COMPARE_METHODS), 2.8 * len(groups)),
        constrained_layout=True,
        squeeze=False,
    )
    for row, group in enumerate(groups):
        common_cres = pair_t7.columns.astype(str)
        keep = (
            pair_t7.loc[group, common_cres].ge(threshold)
            & np.isfinite(atac.loc[group, common_cres].to_numpy(float))
        )
        cres = common_cres[keep.to_numpy()]
        x = np.log1p(atac.loc[group, cres].to_numpy(float))
        t7 = pair_t7.loc[group, cres].to_numpy(float)
        for col, method in enumerate(COMPARE_METHODS):
            ax = axes[row, col]
            y = activity[method].loc[group, cres].to_numpy(float)
            finite = np.isfinite(x) & np.isfinite(y)
            if finite.sum() >= 3:
                rho = spearmanr(y[finite], np.expm1(x[finite])).statistic
            else:
                rho = np.nan
            ax.scatter(
                x[finite],
                y[finite],
                c=np.log10(np.maximum(t7[finite], 1.0)),
                cmap="coolwarm",
                s=16,
                alpha=0.75,
                linewidths=0,
                rasterized=True,
            )
            ax.set_title(f"{METHOD_LABELS[method]}\nrho={rho:.3f}, n={finite.sum():,}")
            if row == len(groups) - 1:
                ax.set_xlabel("log1p(ATAC CPM)")
            else:
                ax.set_xlabel("")
            if col == 0:
                ax.set_ylabel(f"{group}\nActivity")
            else:
                ax.set_ylabel("")
    return fig


def mode_summary(diagnosis: pd.DataFrame) -> pd.DataFrame:
    rows = []
    metrics = [
        "metacell_spearman_atac_cpm",
        "Bootstrap",
        "Joint",
        "source_cells",
        "metacell_n_pairs",
        "median_t7",
        "median_cre",
        "median_atac_cpm",
        "activity_rho_metacell_joint",
    ]
    for (threshold, mode), frame in diagnosis.groupby(["t7_threshold", "mode"], sort=True):
        row = {"t7_threshold": threshold, "mode": mode, "n_subclasses": int(len(frame))}
        for metric in metrics:
            values = pd.to_numeric(frame[metric], errors="coerce")
            row[f"{metric}_median"] = float(values.median()) if values.notna().any() else np.nan
            row[f"{metric}_mean"] = float(values.mean()) if values.notna().any() else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    args.figures_dir.mkdir(parents=True, exist_ok=True)
    args.tables_dir.mkdir(parents=True, exist_ok=True)
    thresholds = sorted({float(threshold) for threshold in args.t7_thresholds})

    log("[metacell ATAC diagnostics] reading existing ATAC correlation table")
    correlations = pd.read_csv(args.correlations)
    mode_assignments = assign_metacell_modes(correlations, thresholds)

    log("[metacell ATAC diagnostics] loading activity, ATAC, and count matrices")
    activity, atac, pair_t7, pair_cre = load_matched_inputs(args)

    log("[metacell ATAC diagnostics] computing diagnosis metrics")
    diagnosis = diagnosis_table(
        mode_assignments,
        correlations,
        activity,
        atac,
        pair_t7,
        pair_cre,
        args,
    )
    diagnosis_path = args.tables_dir / f"{args.stem}.csv"
    summary_path = args.tables_dir / f"{args.stem}_summary.csv"
    diagnosis.to_csv(diagnosis_path, index=False)
    summary = mode_summary(diagnosis)
    summary.to_csv(summary_path, index=False)

    log("[metacell ATAC diagnostics] drawing diagnosis PDF")
    output = args.figures_dir / f"{args.stem}.pdf"
    examples = example_groups(diagnosis, args.example_threshold, args.max_example_groups)
    with PdfPages(output) as pdf:
        for fig in (
            plot_mode_histograms(diagnosis, thresholds),
            plot_method_rho_scatter(diagnosis, thresholds),
            plot_metric_boxplots(diagnosis),
            plot_persistent_lower_heatmap(diagnosis, thresholds),
            plot_example_pair_scatter(
                examples,
                float(args.example_threshold),
                activity,
                atac,
                pair_t7,
            ),
        ):
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

    write_json(
        args.figures_dir / f"{args.stem}_manifest.json",
        {
            "diagnostic": "two-component Gaussian mixture diagnosis of metacell Bayesian ATAC Spearman rho distribution",
            "correlations": str(args.correlations),
            "diagnosis_table": str(diagnosis_path),
            "summary_table": str(summary_path),
            "figure": str(output),
            "mode_model": "GaussianMixture(n_components=2) fit separately per T7 threshold to Metacell Bayesian spearman_atac_cpm",
            "example_threshold": float(args.example_threshold),
            "example_groups": examples,
            "thresholds": thresholds,
        },
    )
    log(f"[metacell ATAC diagnostics] wrote {output}")


if __name__ == "__main__":
    main()
