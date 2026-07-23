#!/usr/bin/env python3
"""Count diagnostics for the lower metacell Bayesian ATAC-correlation mode."""

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

from analysis_utils import ANALYSIS_DIR, DEFAULT_H5AD, LIBSIZE_CSV, STARRFISH_ROOT, log, write_json
import plot_method_activity_correlation as pm
from plot_activity_atac_correlation import read_atac_cpm_subset, read_cre_to_peak


META_METHOD = "Metacell Bayesian"
JOINT_METHOD = "Joint"
MODE_COLORS = {"lower": "#d62728", "upper": "#4c78a8"}


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
    parser.add_argument("--libsize-csv", type=Path, default=LIBSIZE_CSV)
    parser.add_argument("--bootstrap-log-chunk-size", type=int, default=250)
    parser.add_argument("--activity-calibration", choices=["none"], default="none")
    parser.add_argument("--atac-chunk-size", type=int, default=100_000)
    parser.add_argument(
        "--t7-thresholds",
        type=float,
        nargs="+",
        default=[5.0, 10.0, 20.0, 50.0, 100.0],
    )
    parser.add_argument("--background-sample", type=int, default=30_000)
    parser.add_argument("--max-persistent-groups", type=int, default=45)
    parser.add_argument(
        "--figures-dir", type=Path, default=ANALYSIS_DIR / "results" / "figures"
    )
    parser.add_argument(
        "--stem", default="method_activity_atac_metacell_count_diagnostics"
    )
    return parser.parse_args()


def threshold_suffix(threshold: float) -> str:
    threshold = float(threshold)
    if threshold.is_integer():
        return str(int(threshold))
    return str(threshold).replace(".", "p")


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


def matched_atac(
    args: argparse.Namespace,
    groups: pd.Index,
    cres: pd.Index,
) -> pd.DataFrame:
    cre_to_peak = read_cre_to_peak(args.cre_info, cres)
    atac, _ = read_atac_cpm_subset(
        args.atac_cpm,
        cre_to_peak,
        groups,
        args.atac_chunk_size,
    )
    return atac.reindex(index=groups.astype(str), columns=cres.astype(str))


def subclass_rho_table(correlations: pd.DataFrame, modes: pd.DataFrame) -> pd.DataFrame:
    wide = (
        correlations.pivot_table(
            index=["t7_threshold", "group"],
            columns="method",
            values="spearman_atac_cpm",
            aggfunc="first",
        )
        .reset_index()
        .rename_axis(columns=None)
    )
    keep = [
        "t7_threshold",
        "group",
        "mode",
        "lower_mode_probability",
        "lower_mode_mean",
        "upper_mode_mean",
        "n_pairs",
    ]
    return modes[keep].merge(wide, on=["t7_threshold", "group"], how="left")


def pair_rows_for_mode(
    threshold: float,
    mode: str,
    rho_table: pd.DataFrame,
    pair_t7: pd.DataFrame,
    pair_cre: pd.DataFrame,
    cell_counts: pd.Series,
    atac: pd.DataFrame,
    metacell_activity: pd.DataFrame,
) -> pd.DataFrame:
    selected_groups = (
        rho_table[
            np.isclose(rho_table["t7_threshold"].to_numpy(float), float(threshold))
            & rho_table["mode"].eq(mode)
        ]["group"]
        .astype(str)
        .tolist()
    )
    records = []
    common_cres = pd.Index(pair_t7.columns.astype(str)).intersection(atac.columns.astype(str))
    for group in selected_groups:
        if group not in pair_t7.index or group not in atac.index:
            continue
        keep = (
            pair_t7.loc[group, common_cres].ge(threshold)
            & np.isfinite(atac.loc[group, common_cres].to_numpy(float))
            & np.isfinite(metacell_activity.loc[group, common_cres].to_numpy(float))
        )
        for cre in common_cres[keep.to_numpy()]:
            records.append(
                {
                    "t7_threshold": threshold,
                    "mode": mode,
                    "group": group,
                    "cre": str(cre),
                    "total_t7": float(pair_t7.loc[group, cre]),
                    "total_ccre": float(pair_cre.loc[group, cre]),
                    "n_cells": float(cell_counts.get(group, np.nan)),
                    "atac_cpm": float(atac.loc[group, cre]),
                    "metacell_activity": float(metacell_activity.loc[group, cre]),
                }
            )
    return pd.DataFrame(records)


def selected_summary(selected: pd.DataFrame) -> dict:
    keys = ["n_cells", "total_t7", "total_ccre", "atac_cpm"]
    summary = {"n_pairs": int(len(selected))}
    for key in keys:
        values = pd.to_numeric(selected[key], errors="coerce")
        summary[key] = {
            "median": float(values.median()) if values.notna().any() else np.nan,
            "q25": float(values.quantile(0.25)) if values.notna().any() else np.nan,
            "q75": float(values.quantile(0.75)) if values.notna().any() else np.nan,
        }
    return summary


def plot_count_diagnostic(
    threshold: float,
    rho_table: pd.DataFrame,
    pair_data: pd.DataFrame,
    args: argparse.Namespace,
) -> dict:
    sns.set_theme(context="paper", style="white")
    rng = np.random.default_rng(0)
    threshold_rho = rho_table[
        np.isclose(rho_table["t7_threshold"].to_numpy(float), float(threshold))
    ].copy()
    selected = pair_data[pair_data["mode"].eq("lower")].copy()
    comparison = pair_data[pair_data["mode"].isin(["lower", "upper"])].copy()

    output = (
        args.figures_dir
        / f"{args.stem}_lower_mode_t7_ge{threshold_suffix(threshold)}.pdf"
    )
    if selected.empty:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.text(0.5, 0.5, f"No lower-mode pairs for T7 >= {threshold:g}", ha="center", va="center")
        ax.set_axis_off()
        fig.savefig(output, bbox_inches="tight")
        plt.close(fig)
        return {"output": str(output), "n_pairs": 0}

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(16.2, 4.8),
        constrained_layout=True,
        gridspec_kw={"width_ratios": [1.1, 1.25, 1.05]},
    )

    ax = axes[0]
    finite = threshold_rho[[JOINT_METHOD, META_METHOD]].replace([np.inf, -np.inf], np.nan).dropna()
    ax.scatter(
        finite[JOINT_METHOD],
        finite[META_METHOD],
        s=12,
        color="0.72",
        alpha=0.55,
        linewidths=0,
        rasterized=True,
    )
    lower = threshold_rho[threshold_rho["mode"].eq("lower")].dropna(
        subset=[JOINT_METHOD, META_METHOD]
    )
    ax.scatter(
        lower[JOINT_METHOD],
        lower[META_METHOD],
        s=18,
        color=MODE_COLORS["lower"],
        alpha=0.8,
        linewidths=0,
        rasterized=True,
    )
    values = np.concatenate(
        [
            finite[JOINT_METHOD].to_numpy(float),
            finite[META_METHOD].to_numpy(float),
        ]
    )
    lo, hi = np.nanpercentile(values, [1, 99])
    pad = max((hi - lo) * 0.08, 0.05)
    lo, hi = max(-1.0, lo - pad), min(1.0, hi + pad)
    ax.plot([lo, hi], [lo, hi], color="black", linewidth=0.7, alpha=0.6)
    ax.axhline(0, color="black", linewidth=0.6, alpha=0.35)
    ax.axvline(0, color="black", linewidth=0.6, alpha=0.35)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Joint Bayesian ATAC Spearman rho")
    ax.set_ylabel("Metacell Bayesian ATAC Spearman rho")
    ax.set_title(
        f"Lower-mode subclasses, T7 >= {threshold:g}\n"
        f"n={len(lower):,} of {len(threshold_rho):,}"
    )

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
        alpha=0.42,
        linewidths=0,
        rasterized=True,
    )
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
    ax.set_title("Raw lower-mode pair counts; color = cells in subclass")
    cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)
    cbar.set_label("n cells")

    ax = axes[2]
    long = comparison[["mode", "n_cells", "total_t7", "total_ccre"]].rename(
        columns={
            "n_cells": "n cells",
            "total_t7": "total T7",
            "total_ccre": "total cCRE",
        }
    )
    long = long.melt(id_vars="mode", var_name="quantity", value_name="value")
    if len(long) > args.background_sample:
        long = long.sample(args.background_sample, random_state=0)
    long["log10_value_plus1"] = np.log10(pd.to_numeric(long["value"], errors="coerce") + 1.0)
    sns.boxplot(
        data=long,
        x="quantity",
        y="log10_value_plus1",
        hue="mode",
        hue_order=["lower", "upper"],
        palette=MODE_COLORS,
        showfliers=False,
        linewidth=0.8,
        ax=ax,
    )
    sns.stripplot(
        data=long,
        x="quantity",
        y="log10_value_plus1",
        hue="mode",
        hue_order=["lower", "upper"],
        palette=MODE_COLORS,
        dodge=True,
        alpha=0.12,
        size=1.2,
        linewidth=0,
        legend=False,
        rasterized=True,
        ax=ax,
    )
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[:2], labels[:2], title="Mode", frameon=False)
    ax.set_xlabel("")
    ax.set_ylabel("log10(value + 1)")
    ax.set_title("Lower vs upper pair-level count distributions")

    summary = selected_summary(selected)
    fig.suptitle(
        f"Metacell Bayesian lower ATAC-rho mode count diagnostics, T7 >= {threshold:g}; "
        f"n={len(selected):,} lower-mode cCRE-subclass pairs | "
        f"median cells={summary['n_cells']['median']:.0f}, "
        f"T7={summary['total_t7']['median']:.0f}, "
        f"cCRE={summary['total_ccre']['median']:.0f}",
        fontsize=12,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    return {"output": str(output), **summary}


def plot_persistent_subclasses(
    rho_table: pd.DataFrame,
    thresholds: list[float],
    args: argparse.Namespace,
) -> dict:
    lower = rho_table[rho_table["mode"].eq("lower")].copy()
    output = args.figures_dir / f"{args.stem}_persistent_lower_mode_subclasses.pdf"
    csv_output = args.figures_dir / f"{args.stem}_persistent_lower_mode_subclasses.csv"
    if lower.empty:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.text(0.5, 0.5, "No lower-mode subclasses", ha="center", va="center")
        ax.set_axis_off()
        fig.savefig(output, bbox_inches="tight")
        plt.close(fig)
        return {"output": str(output), "csv": str(csv_output), "n_subclasses": 0}

    ranking = (
        lower.groupby("group")
        .agg(
            lower_filters=("t7_threshold", "nunique"),
            median_lower_rho=(META_METHOD, "median"),
            min_lower_rho=(META_METHOD, "min"),
        )
        .sort_values(["lower_filters", "median_lower_rho"], ascending=[False, True])
        .head(args.max_persistent_groups)
    )
    selected = ranking.index
    ranking.to_csv(csv_output)

    rho_matrix = (
        rho_table[rho_table["group"].isin(selected)]
        .pivot(index="group", columns="t7_threshold", values=META_METHOD)
        .reindex(index=selected, columns=thresholds)
    )
    mode_matrix = (
        rho_table[rho_table["group"].isin(selected)]
        .assign(lower_mode=lambda x: x["mode"].eq("lower").astype(float))
        .pivot(index="group", columns="t7_threshold", values="lower_mode")
        .reindex(index=selected, columns=thresholds)
    )
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(11.8, max(5.5, 0.18 * len(selected))),
        constrained_layout=True,
        gridspec_kw={"width_ratios": [2.4, 0.85]},
    )
    sns.heatmap(
        rho_matrix,
        cmap="RdBu_r",
        center=0,
        vmin=-0.6,
        vmax=0.6,
        linewidths=0.25,
        linecolor="white",
        cbar_kws={"label": "Metacell ATAC Spearman rho"},
        ax=axes[0],
    )
    axes[0].set_xlabel("T7 filter")
    axes[0].set_ylabel("Subclass")
    axes[0].set_title("Persistent lower-mode subclasses")
    sns.heatmap(
        mode_matrix,
        cmap=sns.color_palette(["#f0f0f0", MODE_COLORS["lower"]], as_cmap=True),
        vmin=0,
        vmax=1,
        linewidths=0.25,
        linecolor="white",
        cbar=False,
        ax=axes[1],
    )
    axes[1].set_xlabel("T7 filter")
    axes[1].set_ylabel("")
    axes[1].set_yticklabels([])
    axes[1].set_title("Lower mode")
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    return {
        "output": str(output),
        "csv": str(csv_output),
        "n_subclasses": int(len(ranking)),
    }


def main() -> None:
    args = parse_args()
    args.figures_dir.mkdir(parents=True, exist_ok=True)
    thresholds = sorted({float(threshold) for threshold in args.t7_thresholds})

    log("[metacell count diagnostics] loading ATAC correlation table")
    correlations = pd.read_csv(args.correlations)
    modes = assign_modes(correlations, thresholds)
    rho_table = subclass_rho_table(correlations, modes)

    log("[metacell count diagnostics] loading pair counts")
    methods = (JOINT_METHOD, META_METHOD)
    matrices, pair_t7, pair_cre, cell_counts, _nanopore_counts, metadata = pm.prepare_base(
        args, methods
    )
    groups = pd.Index(pair_t7.index.astype(str))
    cres = pd.Index(pair_t7.columns.astype(str))
    atac = matched_atac(args, groups, cres)

    pair_outputs = {}
    pair_csv_outputs = {}
    for threshold in thresholds:
        log(f"[metacell count diagnostics] plotting T7 >= {threshold:g}")
        frames = []
        for mode in ("lower", "upper"):
            frames.append(
                pair_rows_for_mode(
                    threshold,
                    mode,
                    rho_table,
                    pair_t7,
                    pair_cre,
                    cell_counts,
                    atac,
                    matrices[META_METHOD],
                )
            )
        pair_data = pd.concat(frames, ignore_index=True)
        suffix = threshold_suffix(threshold)
        pair_csv = args.figures_dir / f"{args.stem}_lower_mode_t7_ge{suffix}_pairs.csv"
        pair_data[pair_data["mode"].eq("lower")].to_csv(pair_csv, index=False)
        pair_csv_outputs[f"t7_ge{suffix}"] = str(pair_csv)
        pair_outputs[f"t7_ge{suffix}"] = plot_count_diagnostic(
            threshold, rho_table, pair_data, args
        )

    persistent = plot_persistent_subclasses(rho_table, thresholds, args)
    manifest = {
        "diagnostic": "count-diagnostics style plots for metacell Bayesian lower ATAC Spearman mode",
        "mode_model": (
            "GaussianMixture(n_components=2) fit separately per T7 threshold to "
            "Metacell Bayesian spearman_atac_cpm"
        ),
        "correlations": str(args.correlations),
        "thresholds": thresholds,
        "outputs": pair_outputs,
        "pair_csv_outputs": pair_csv_outputs,
        "persistent_lower_mode_subclasses": persistent,
        "metadata": metadata,
    }
    write_json(args.figures_dir / f"{args.stem}_manifest.json", manifest)
    log("[metacell count diagnostics] wrote count diagnostics and persistent subclass plot")


if __name__ == "__main__":
    main()
