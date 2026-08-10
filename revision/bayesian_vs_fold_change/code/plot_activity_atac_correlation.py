#!/usr/bin/env python3
"""Correlate full-run activity estimates with subclass ATAC CPM."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr, spearmanr

from analysis_utils import (
    ANALYSIS_DIR,
    DEFAULT_H5AD,
    FIGURES_WORK,
    OLD_DATA_BOOTSTRAP,
    STARRFISH_ROOT,
    log,
    write_json,
)
from plot_method_activity_correlation import (
    blacklisted_cres_for_methods,
    load_corrected_activity,
    method_roots,
    pair_count_totals,
)
from plot_method_activity_heatmap import combined_axes


METHODS = (
    "Bootstrap",
    "Joint",
    "Decoupled",
    "Joint+dropout",
    "Decoupled+dropout",
    "Metacell Bayesian",
)
METHOD_LABELS = {
    "Bootstrap": "Bootstrap",
    "Joint": "Joint",
    "Decoupled": "Decoupled",
    "Joint+dropout": "Joint+dropout",
    "Decoupled+dropout": "Decoupled+dropout",
    "Metacell Bayesian": "Metacell Bayesian",
}
METHOD_COLORS = {
    "Bootstrap": "#f58518",
    "Joint": "#4c78a8",
    "Decoupled": "#54a24b",
    "Joint+dropout": "#b279a2",
    "Decoupled+dropout": "#e45756",
    "Metacell Bayesian": "#72b7b2",
}


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
        help="Bayesian joint directory fit on bootstrap metacells.",
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
    parser.add_argument(
        "--figures-dir", type=Path, default=FIGURES_WORK
    )
    parser.add_argument(
        "--tables-dir", type=Path, default=ANALYSIS_DIR / "results" / "tables"
    )
    parser.add_argument(
        "--activity-calibration",
        choices=["calibrated", "none"],
        default="none",
    )
    parser.add_argument(
        "--activity-centering",
        choices=["none", "posterior-alpha"],
        default="none",
        help="Use existing activity or posterior mean log_gamma - alpha.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=METHODS,
        default=list(METHODS),
    )
    parser.add_argument("--h5ad", type=Path, default=DEFAULT_H5AD)
    parser.add_argument("--bootstrap-log-chunk-size", type=int, default=250)
    parser.add_argument("--min-pairs", type=int, default=10)
    parser.add_argument(
        "--t7-thresholds",
        type=float,
        nargs="+",
        default=[5.0, 10.0, 20.0, 50.0, 100.0],
        help="Subclass-cCRE total T7 count filters. Uses >= threshold.",
    )
    parser.add_argument("--atac-chunk-size", type=int, default=100_000)
    parser.add_argument("--stem", default="method_activity_atac_correlation")
    return parser.parse_args()


def normalize_subclass_name(name: str) -> str:
    return str(name).replace("_", " ")


def read_cre_to_peak(path: Path, cres: pd.Index) -> pd.Series:
    cre_info = pd.read_csv(path)
    if "cre" not in cre_info.columns or "enh" not in cre_info.columns:
        raise ValueError(f"{path} must contain 'cre' and 'enh' columns")
    cre_to_peak = cre_info.assign(cre=cre_info["cre"].astype(str)).set_index("cre")[
        "enh"
    ]
    cre_to_peak = cre_to_peak.astype(str).reindex(cres.astype(str)).dropna()
    return cre_to_peak


def read_atac_cpm_subset(
    path: Path,
    cre_to_peak: pd.Series,
    groups: pd.Index,
    chunk_size: int,
) -> tuple[pd.DataFrame, dict]:
    header = pd.read_csv(path, nrows=0)
    index_column = header.columns[0]
    activity_groups = set(groups.astype(str))
    atac_to_activity = {
        col: normalize_subclass_name(col)
        for col in header.columns[1:].astype(str)
        if normalize_subclass_name(col) in activity_groups
    }
    usecols = [index_column, *atac_to_activity.keys()]
    needed_peaks = set(cre_to_peak.astype(str))
    chunks = []
    for chunk in pd.read_csv(
        path,
        usecols=usecols,
        index_col=0,
        chunksize=chunk_size,
    ):
        chunk.index = chunk.index.astype(str)
        selected = chunk.index.isin(needed_peaks)
        if selected.any():
            chunks.append(chunk.loc[selected])
    if chunks:
        atac_by_peak = pd.concat(chunks, axis=0)
        atac_by_peak = atac_by_peak[~atac_by_peak.index.duplicated(keep="first")]
    else:
        atac_by_peak = pd.DataFrame(columns=atac_to_activity.keys())
    atac_by_peak = atac_by_peak.rename(columns=atac_to_activity)
    atac_by_peak = atac_by_peak.apply(pd.to_numeric, errors="coerce")

    matched_cre_to_peak = cre_to_peak[cre_to_peak.isin(atac_by_peak.index)]
    atac_by_cre = atac_by_peak.reindex(matched_cre_to_peak.to_numpy())
    atac_by_cre.index = matched_cre_to_peak.index
    atac_by_cre = atac_by_cre.T
    atac_by_cre.index = atac_by_cre.index.astype(str)
    atac_by_cre.columns = atac_by_cre.columns.astype(str)
    metadata = {
        "atac_columns_total": int(len(header.columns) - 1),
        "atac_columns_matched_to_activity_groups": int(len(atac_to_activity)),
        "requested_cre_peaks": int(len(cre_to_peak)),
        "matched_cre_peaks": int(len(matched_cre_to_peak)),
        "unmatched_cre_count": int(len(cre_to_peak) - len(matched_cre_to_peak)),
        "chunk_size": int(chunk_size),
    }
    return atac_by_cre, metadata


def finite_correlation(
    x: pd.Series,
    y: pd.Series,
    min_pairs: int,
) -> dict:
    x_values = pd.to_numeric(x, errors="coerce").to_numpy(float)
    y_values = pd.to_numeric(y, errors="coerce").to_numpy(float)
    finite = np.isfinite(x_values) & np.isfinite(y_values)
    x_values = x_values[finite]
    y_values = y_values[finite]
    if len(x_values) < min_pairs:
        return {
            "n_pairs": int(len(x_values)),
            "spearman_atac_cpm": np.nan,
            "pearson_log1p_atac_cpm": np.nan,
        }
    log_atac = np.log1p(y_values)
    if np.nanstd(x_values) == 0 or np.nanstd(y_values) == 0:
        spearman = np.nan
    else:
        spearman = float(spearmanr(x_values, y_values).statistic)
    if np.nanstd(x_values) == 0 or np.nanstd(log_atac) == 0:
        pearson = np.nan
    else:
        pearson = float(pearsonr(x_values, log_atac).statistic)
    return {
        "n_pairs": int(len(x_values)),
        "spearman_atac_cpm": spearman,
        "pearson_log1p_atac_cpm": pearson,
    }


def posterior_alpha_activity(root: Path) -> tuple[pd.DataFrame, Path]:
    manifest = json.loads((root / "run_manifest.json").read_text())
    posterior_path = root / f"{manifest['tag']}_posterior_samples.npz"
    with np.load(posterior_path, allow_pickle=True) as posterior:
        required = {"log_gamma", "alpha", "group_names", "cre_names"}
        missing = required.difference(posterior.files)
        if missing:
            raise ValueError(
                f"{posterior_path} is missing posterior-alpha sites: {sorted(missing)}"
            )
        log_gamma = posterior["log_gamma"]
        alpha = posterior["alpha"]
        if alpha.shape != (log_gamma.shape[0], log_gamma.shape[2]):
            raise ValueError(
                f"alpha shape {alpha.shape} does not match log_gamma shape "
                f"{log_gamma.shape} in {posterior_path}"
            )
        values = log_gamma.mean(axis=0, dtype=np.float64) - alpha.mean(
            axis=0, dtype=np.float64
        )[None, :]
        groups = posterior["group_names"].astype(str)
        cres = posterior["cre_names"].astype(str)
    return pd.DataFrame(values, index=groups, columns=cres), posterior_path


def load_activity(
    args: argparse.Namespace,
    methods: tuple[str, ...],
) -> tuple[dict[str, pd.DataFrame], dict]:
    posterior_sources = {}
    if args.activity_centering == "posterior-alpha":
        raw = {}
        corrections = {}
        if "Bootstrap" in methods:
            bootstrap, _ = load_corrected_activity(args, ("Bootstrap",))
            raw["Bootstrap"] = bootstrap["Bootstrap"]
        roots = method_roots(args)
        for method in methods:
            if method == "Bootstrap":
                continue
            raw[method], posterior_path = posterior_alpha_activity(roots[method])
            posterior_sources[method] = str(posterior_path)
    else:
        raw, corrections = load_corrected_activity(args, methods)
    rows, columns = combined_axes(raw)
    blacklist, blacklist_sources = blacklisted_cres_for_methods(args, methods)
    columns = pd.Index(
        [cre for cre in columns.astype(str) if cre not in blacklist],
        dtype=str,
    )
    activity = {
        method: raw[method].reindex(index=rows, columns=columns)
        for method in methods
    }
    metadata = {
        "bayesian_activity_scale": (
            "posterior mean(log_gamma - alpha)"
            if args.activity_centering == "posterior-alpha"
            else "log_gamma - mean_log_beta_t7"
        ),
        "activity_centering": args.activity_centering,
        "mean_log_beta_t7_corrections": corrections,
        "posterior_sources": posterior_sources,
        "blacklisted_cres_removed": int(len(blacklist)),
        "blacklist_sources": blacklist_sources,
        "methods": list(methods),
    }
    return activity, metadata


def activity_atac_correlations(
    activity: dict[str, pd.DataFrame],
    atac: pd.DataFrame,
    min_pairs: int,
    methods: tuple[str, ...],
    *,
    pair_mask: pd.DataFrame | None = None,
    t7_threshold: float | None = None,
) -> pd.DataFrame:
    rows = []
    group_order = pd.Index(next(iter(activity.values())).index.astype(str))
    for method in methods:
        matrix = activity[method]
        common_groups = group_order.intersection(atac.index.astype(str))
        common_cres = matrix.columns.astype(str).intersection(atac.columns.astype(str))
        for group in common_groups:
            cres = common_cres
            if pair_mask is not None:
                keep = (
                    pair_mask.reindex(index=[group], columns=common_cres, fill_value=False)
                    .iloc[0]
                    .astype(bool)
                )
                cres = common_cres[keep.to_numpy()]
            stats = finite_correlation(
                matrix.loc[group, cres],
                atac.loc[group, cres],
                min_pairs,
            )
            rows.append(
                {
                    "t7_threshold": t7_threshold,
                    "method": method,
                    "group": group,
                    **stats,
                }
            )
    return pd.DataFrame(rows)


def long_activity_atac_pairs(
    activity: dict[str, pd.DataFrame],
    atac: pd.DataFrame,
) -> pd.DataFrame:
    frames = []
    for method, matrix in activity.items():
        common_groups = matrix.index.astype(str).intersection(atac.index.astype(str))
        common_cres = matrix.columns.astype(str).intersection(atac.columns.astype(str))
        activity_stack = (
            matrix.reindex(index=common_groups, columns=common_cres)
            .rename_axis(index="group", columns="cre")
            .stack(future_stack=True)
            .rename("activity")
        )
        atac_stack = (
            atac.reindex(index=common_groups, columns=common_cres)
            .rename_axis(index="group", columns="cre")
            .stack(future_stack=True)
            .rename("atac_cpm")
        )
        frame = pd.concat([activity_stack, atac_stack], axis=1).reset_index()
        frame["method"] = method
        frame = frame[
            np.isfinite(frame["activity"].to_numpy(float))
            & np.isfinite(frame["atac_cpm"].to_numpy(float))
        ]
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def plot_correlation_figure(
    correlations: pd.DataFrame,
    output: Path,
    methods: tuple[str, ...] = METHODS,
) -> None:
    sns.set_theme(context="paper", style="white")
    valid = correlations[np.isfinite(correlations["spearman_atac_cpm"])].copy()
    if valid.empty:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.text(0.5, 0.5, "No finite ATAC correlations", ha="center", va="center")
        ax.set_axis_off()
        fig.savefig(output, bbox_inches="tight")
        plt.close(fig)
        return
    group_order = (
        valid.groupby("group", sort=False)["spearman_atac_cpm"]
        .median()
        .reindex(correlations["group"].drop_duplicates())
        .dropna()
        .index
    )
    matrix = (
        correlations.pivot(index="method", columns="group", values="spearman_atac_cpm")
        .reindex(index=methods, columns=group_order)
    )
    width = max(16, 0.08 * len(group_order))
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(width, 7.5),
        gridspec_kw={"height_ratios": [1.3, 1.0]},
        constrained_layout=True,
    )
    sns.heatmap(
        matrix,
        ax=axes[0],
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        xticklabels=True,
        yticklabels=[METHOD_LABELS[m] for m in methods],
        cbar_kws={"label": "Spearman rho(activity, ATAC CPM)"},
    )
    axes[0].set_xlabel("Subclass")
    axes[0].set_ylabel("")
    axes[0].set_title("Per-subclass activity correlation with ATAC CPM")
    axes[0].tick_params(axis="x", rotation=90, labelsize=4)
    axes[0].tick_params(axis="y", labelsize=8)

    sns.boxplot(
        data=valid,
        x="method",
        y="spearman_atac_cpm",
        hue="method",
        order=list(methods),
        hue_order=list(methods),
        palette=METHOD_COLORS,
        showfliers=False,
        legend=False,
        ax=axes[1],
    )
    sns.stripplot(
        data=valid,
        x="method",
        y="spearman_atac_cpm",
        order=list(METHODS),
        color="black",
        size=1.8,
        alpha=0.25,
        jitter=0.25,
        rasterized=True,
        ax=axes[1],
    )
    axes[1].axhline(0, color="black", linewidth=0.7, alpha=0.5)
    axes[1].set_xticks(np.arange(len(methods)))
    axes[1].set_xticklabels([METHOD_LABELS[m] for m in methods], rotation=20, ha="right")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Spearman rho")
    axes[1].set_title("Distribution across matched subclasses")
    sns.despine(fig=fig)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def threshold_suffix(threshold: float) -> str:
    threshold = float(threshold)
    if threshold.is_integer():
        return str(int(threshold))
    return str(threshold).replace(".", "p")


def barplot_figure(
    correlations: pd.DataFrame,
    threshold: float,
    min_pairs: int,
    methods: tuple[str, ...],
) -> plt.Figure:
    sns.set_theme(context="paper", style="white")
    frame = correlations[
        np.isclose(correlations["t7_threshold"].to_numpy(float), float(threshold))
    ].copy()
    frame = frame[np.isfinite(frame["spearman_atac_cpm"].to_numpy(float))]
    finite_groups = frame["group"].drop_duplicates()
    group_order = [
        group
        for group in correlations["group"].drop_duplicates().astype(str)
        if group in set(finite_groups.astype(str))
    ]
    if not group_order:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.text(
            0.5,
            0.5,
            f"No finite correlations for T7 >= {threshold:g}",
            ha="center",
            va="center",
        )
        ax.set_axis_off()
        return fig

    width = max(18.0, 0.13 * len(group_order))
    fig, ax = plt.subplots(figsize=(width, 5.5), constrained_layout=True)
    sns.barplot(
        data=frame,
        x="group",
        y="spearman_atac_cpm",
        hue="method",
        order=group_order,
        hue_order=list(methods),
        palette=METHOD_COLORS,
        errorbar=None,
        ax=ax,
    )
    ax.axhline(0, color="black", linewidth=0.7, alpha=0.6)
    ax.set_ylim(-1, 1)
    ax.set_xlabel("Subclass")
    ax.set_ylabel("Spearman rho(activity, ATAC CPM)")
    ax.set_title(
        f"ATAC correlation by subclass, T7 >= {threshold:g} "
        f"(min {min_pairs} cCREs per bar)"
    )
    ax.tick_params(axis="x", rotation=90, labelsize=4)
    ax.tick_params(axis="y", labelsize=7)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles,
        [METHOD_LABELS.get(label, label) for label in labels],
        title="Method",
        loc="upper center",
        bbox_to_anchor=(0.5, 1.18),
        ncol=len(methods),
        frameon=False,
    )
    sns.despine(fig=fig, ax=ax)
    return fig


def plot_threshold_barplots(
    correlations: pd.DataFrame,
    thresholds: list[float],
    figures_dir: Path,
    stem: str,
    min_pairs: int,
    methods: tuple[str, ...],
) -> dict[str, str]:
    figures_dir.mkdir(parents=True, exist_ok=True)
    outputs = {}
    combined_output = figures_dir / f"{stem}.pdf"
    with PdfPages(combined_output) as pdf:
        for threshold in thresholds:
            fig = barplot_figure(correlations, threshold, min_pairs, methods)
            key = f"t7_ge{threshold_suffix(threshold)}"
            output = figures_dir / f"{stem}_{key}.pdf"
            fig.savefig(output, bbox_inches="tight")
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
            outputs[key] = str(output)
    outputs["combined"] = str(combined_output)
    return outputs


def rho_axis_limit(values: pd.Series) -> tuple[float, float]:
    finite = pd.to_numeric(values, errors="coerce").to_numpy(float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return -1.0, 1.0
    lo, hi = np.percentile(finite, [1, 99])
    lo = min(float(lo), 0.0)
    hi = max(float(hi), 0.0)
    pad = max((hi - lo) * 0.06, 0.05)
    return max(-1.0, lo - pad), min(1.0, hi + pad)


def scatter_matrix_figure(
    correlations: pd.DataFrame,
    threshold: float,
    methods: tuple[str, ...],
    correlation_column: str = "spearman_atac_cpm",
    signal_label: str = "ATAC",
) -> plt.Figure:
    sns.set_theme(context="paper", style="white")
    frame = correlations[
        np.isclose(correlations["t7_threshold"].to_numpy(float), float(threshold))
    ].copy()
    wide = frame.pivot(index="group", columns="method", values=correlation_column)
    wide = wide.reindex(columns=methods)
    counts_wide = frame.pivot(index="group", columns="method", values="n_pairs")
    counts_wide = counts_wide.reindex(index=wide.index, columns=methods)
    finite_counts = counts_wide.where(np.isfinite(wide.to_numpy(float))).to_numpy(float)
    finite_counts = finite_counts[np.isfinite(finite_counts) & (finite_counts > 0)]
    if finite_counts.size:
        vmin = max(1.0, float(np.nanmin(finite_counts)))
        vmax = float(np.nanmax(finite_counts))
        if vmin >= vmax:
            vmax = vmin + 1.0
    else:
        vmin, vmax = 1.0, 10.0
    count_norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)
    count_cmap = "coolwarm"
    count_mappable = plt.cm.ScalarMappable(norm=count_norm, cmap=count_cmap)
    count_mappable.set_array([])
    limits = {method: rho_axis_limit(wide[method]) for method in methods}
    matrix_size = max(12.5, 2.55 * len(methods))
    fig, axes = plt.subplots(
        len(methods),
        len(methods),
        figsize=(matrix_size, matrix_size),
        constrained_layout=True,
        squeeze=False,
    )
    for row, y_method in enumerate(methods):
        for col, x_method in enumerate(methods):
            ax = axes[row, col]
            ax.set_xlim(limits[x_method])
            if row == col:
                values = pd.to_numeric(wide[x_method], errors="coerce")
                values = values[np.isfinite(values.to_numpy(float))]
                sns.histplot(values, bins=28, color=METHOD_COLORS[x_method], ax=ax)
                ax.text(
                    0.04,
                    0.94,
                    f"n={len(values):,}",
                    transform=ax.transAxes,
                    ha="left",
                    va="top",
                    fontsize=8,
                )
            else:
                pair = wide[[x_method, y_method]].dropna()
                pair_counts = (
                    counts_wide[[x_method, y_method]]
                    .reindex(pair.index)
                    .min(axis=1)
                    .to_numpy(float)
                )
                ax.scatter(
                    pair[x_method],
                    pair[y_method],
                    c=pair_counts,
                    cmap=count_cmap,
                    norm=count_norm,
                    s=12,
                    alpha=0.72,
                    linewidths=0,
                )
                ax.set_ylim(limits[y_method])
                lo = max(limits[x_method][0], limits[y_method][0])
                hi = min(limits[x_method][1], limits[y_method][1])
                if lo < hi:
                    ax.plot([lo, hi], [lo, hi], color="black", linewidth=0.7, alpha=0.55)
                if len(pair) > 1:
                    pearson = pair[x_method].corr(pair[y_method], method="pearson")
                    spearman = pair[x_method].corr(pair[y_method], method="spearman")
                else:
                    pearson = np.nan
                    spearman = np.nan
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
            if row == len(methods) - 1:
                ax.set_xlabel(METHOD_LABELS[x_method])
            else:
                ax.set_xlabel("")
                ax.set_xticklabels([])
            if col == 0:
                ax.set_ylabel(METHOD_LABELS[y_method])
            else:
                ax.set_ylabel("")
                ax.set_yticklabels([])
    fig.suptitle(
        f"{signal_label} Spearman rho model comparison, T7 >= {threshold:g}\n"
        "Each dot is one subclass; color is cCRE pairs used; black line is y = x.",
        fontsize=12,
    )
    colorbar = fig.colorbar(
        count_mappable,
        ax=axes,
        fraction=0.018,
        pad=0.01,
    )
    colorbar.set_label("cCRE pairs used for rho (min of compared methods)")
    sns.despine(fig=fig)
    return fig


def plot_threshold_scatter_matrices(
    correlations: pd.DataFrame,
    thresholds: list[float],
    figures_dir: Path,
    stem: str,
    methods: tuple[str, ...],
    correlation_column: str = "spearman_atac_cpm",
    signal_label: str = "ATAC",
) -> dict[str, str]:
    figures_dir.mkdir(parents=True, exist_ok=True)
    outputs = {}
    combined_output = figures_dir / f"{stem}_spearman_scatter.pdf"
    with PdfPages(combined_output) as pdf:
        for threshold in thresholds:
            fig = scatter_matrix_figure(
                correlations,
                threshold,
                methods,
                correlation_column=correlation_column,
                signal_label=signal_label,
            )
            key = f"t7_ge{threshold_suffix(threshold)}"
            output = figures_dir / f"{stem}_spearman_scatter_{key}.pdf"
            fig.savefig(output, bbox_inches="tight")
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
            outputs[key] = str(output)
    outputs["combined"] = str(combined_output)
    return outputs


def plot_pooled_scatter(
    pairs: pd.DataFrame,
    output: Path,
    methods: tuple[str, ...] = METHODS,
) -> pd.DataFrame:
    sns.set_theme(context="paper", style="white")
    stats = []
    n_cols = 3
    n_rows = int(np.ceil(len(methods) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 3.75 * n_rows), constrained_layout=True)
    axes = axes.ravel()
    for ax, method in zip(axes, methods):
        frame = pairs[pairs["method"].eq(method)].copy()
        frame["log1p_atac_cpm"] = np.log1p(frame["atac_cpm"].to_numpy(float))
        if frame.empty:
            ax.text(0.5, 0.5, "No finite pairs", ha="center", va="center")
            ax.set_axis_off()
            continue
        finite = (
            np.isfinite(frame["activity"].to_numpy(float))
            & np.isfinite(frame["log1p_atac_cpm"].to_numpy(float))
        )
        rho = (
            float(
                spearmanr(
                    frame.loc[finite, "activity"],
                    frame.loc[finite, "atac_cpm"],
                ).statistic
            )
            if finite.sum() >= 3
            else np.nan
        )
        stats.append({"method": method, "n_pairs": int(finite.sum()), "spearman": rho})
        image = ax.hexbin(
            frame["activity"],
            frame["log1p_atac_cpm"],
            gridsize=70,
            mincnt=1,
            bins="log",
            cmap="viridis",
            rasterized=True,
        )
        ax.set_title(f"{METHOD_LABELS[method]}\nrho={rho:.3f}, n={finite.sum():,}")
        ax.set_xlabel("Activity estimate")
        ax.set_ylabel("log1p(ATAC CPM)")
        fig.colorbar(image, ax=ax, label="log10(pair count)")
    for ax in axes[len(methods) :]:
        ax.set_axis_off()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    return pd.DataFrame(stats)


def main() -> None:
    args = parse_args()
    args.figures_dir.mkdir(parents=True, exist_ok=True)
    args.tables_dir.mkdir(parents=True, exist_ok=True)

    methods = tuple(dict.fromkeys(args.methods))
    log("[activity ATAC] loading activity estimates")
    activity, metadata = load_activity(args, methods)
    all_cres = pd.Index(sorted(set().union(*(set(m.columns) for m in activity.values()))))
    cre_to_peak = read_cre_to_peak(args.cre_info, all_cres)

    log("[activity ATAC] loading matched ATAC CPM rows")
    atac, atac_metadata = read_atac_cpm_subset(
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

    log("[activity ATAC] loading subclass-cCRE T7 totals for filters")
    pair_t7, _ = pair_count_totals(args.h5ad, common_groups, common_cres)
    pair_t7 = pair_t7.reindex(index=common_groups, columns=common_cres).fillna(0.0)

    thresholds = sorted({float(threshold) for threshold in args.t7_thresholds})
    correlations_by_threshold = []
    log("[activity ATAC] computing per-subclass correlations by T7 filter")
    for threshold in thresholds:
        correlations_by_threshold.append(
            activity_atac_correlations(
                activity,
                atac,
                args.min_pairs,
                methods,
                pair_mask=pair_t7.ge(threshold),
                t7_threshold=threshold,
            )
        )
    correlations = pd.concat(correlations_by_threshold, ignore_index=True)
    correlations.to_csv(
        args.tables_dir / f"{args.stem}_by_subclass_t7_filters.csv", index=False
    )
    finite_correlations = correlations[
        np.isfinite(correlations["spearman_atac_cpm"].to_numpy(float))
    ]
    summary = (
        finite_correlations.groupby(["t7_threshold", "method"], sort=False)
        .agg(
            n_subclasses=("group", "nunique"),
            median_spearman=("spearman_atac_cpm", "median"),
            mean_spearman=("spearman_atac_cpm", "mean"),
            median_pearson_log1p=("pearson_log1p_atac_cpm", "median"),
            median_pairs=("n_pairs", "median"),
            mean_pairs=("n_pairs", "mean"),
        )
        .reset_index()
    )
    threshold_order = {threshold: idx for idx, threshold in enumerate(thresholds)}
    method_order = {method: idx for idx, method in enumerate(methods)}
    summary = (
        summary.assign(
            threshold_order=summary["t7_threshold"].map(threshold_order),
            method_order=summary["method"].map(method_order),
        )
        .sort_values(["threshold_order", "method_order"])
        .drop(columns=["threshold_order", "method_order"])
    )
    summary.to_csv(args.tables_dir / f"{args.stem}_summary.csv", index=False)

    barplot_outputs = plot_threshold_barplots(
        correlations,
        thresholds,
        args.figures_dir,
        args.stem,
        args.min_pairs,
        methods,
    )
    scatter_outputs = plot_threshold_scatter_matrices(
        correlations,
        thresholds,
        args.figures_dir,
        args.stem,
        methods,
    )

    manifest = {
        **metadata,
        **atac_metadata,
        "atac_cpm": str(args.atac_cpm),
        "cre_info": str(args.cre_info),
        "h5ad": str(args.h5ad),
        "activity_calibration": args.activity_calibration,
        "min_pairs": int(args.min_pairs),
        "t7_filter": "subclass-cCRE total T7 count >= threshold",
        "t7_thresholds": thresholds,
        "matched_subclasses": int(len(common_groups)),
        "matched_cres": int(len(common_cres)),
        "barplot_outputs": barplot_outputs,
        "spearman_scatter_outputs": scatter_outputs,
        "main_figure": barplot_outputs["combined"],
        "by_subclass_table": str(
            args.tables_dir / f"{args.stem}_by_subclass_t7_filters.csv"
        ),
        "summary_table": str(args.tables_dir / f"{args.stem}_summary.csv"),
        "correlation_definitions": {
            "spearman_atac_cpm": (
                "Spearman correlation across cCREs within subclass after the T7 "
                "filter, using activity estimate and raw ATAC CPM"
            ),
            "pearson_log1p_atac_cpm": (
                "Pearson correlation across cCREs within subclass after the T7 "
                "filter, using activity estimate and log1p(ATAC CPM)"
            ),
        },
    }
    write_json(args.figures_dir / f"{args.stem}_manifest.json", manifest)
    log(
        "[activity ATAC] wrote "
        f"{barplot_outputs['combined']} and {len(correlations):,} correlation rows"
    )


if __name__ == "__main__":
    main()
