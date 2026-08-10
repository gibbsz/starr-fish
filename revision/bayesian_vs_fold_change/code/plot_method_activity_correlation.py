#!/usr/bin/env python3
"""Plot activity correlation matrices among full-run methods."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from analysis_utils import (
    ANALYSIS_DIR,
    DEFAULT_H5AD,
    FIGURES_WORK,
    LIBSIZE_CSV,
    OLD_DATA_BOOTSTRAP,
    log,
    read_and_prepare_adata,
    write_json,
)
from plot_method_activity_heatmap import (
    combined_axes,
    read_cre_blacklist,
    trim_empty_axes,
)
from plot_section_reproducibility import bayesian_base, bootstrap_base


METHODS = ("Bayesian decoupled", "Bayesian joint", "Bootstrap")
POINT_METHOD = "Point log(cCRE/T7)"
CORRELATION_METHODS = (
    "Bootstrap",
    POINT_METHOD,
    "Joint",
    "Decoupled",
    "Joint+dropout",
    "Decoupled+dropout",
)


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
    parser.add_argument(
        "--activity-calibration",
        choices=["calibrated", "none"],
        default="none",
    )
    parser.add_argument("--bootstrap-log-chunk-size", type=int, default=250)
    parser.add_argument("--t7-threshold", type=float, default=100.0)
    parser.add_argument(
        "--t7-thresholds",
        type=float,
        nargs="+",
        default=[10.0, 50.0, 100.0],
        help="Subclass-cCRE total T7 thresholds to plot as t7_gt* variants.",
    )
    parser.add_argument("--cell-count-threshold", type=int, default=1000)
    parser.add_argument("--nanopore-threshold", type=float, default=1000.0)
    parser.add_argument("--libsize-csv", type=Path, default=LIBSIZE_CSV)
    parser.add_argument("--stem", default="method_activity_correlation")
    return parser.parse_args()


def discover_bayes_tag(root: Path) -> str:
    return json.loads((root / "run_manifest.json").read_text())["tag"]


def mean_log_beta_t7(root: Path) -> float:
    tag = discover_bayes_tag(root)
    with np.load(root / f"{tag}_scalar_samples.npz", allow_pickle=True) as samples:
        beta_t7 = np.asarray(samples["beta_t7"], dtype=float).reshape(-1)
    return float(np.log(beta_t7).mean())


def default_bayesian_root(name: str) -> Path:
    return ANALYSIS_DIR / "results" / "ablation" / name


def method_roots(args: argparse.Namespace) -> dict[str, Path]:
    joint = getattr(args, "old_bayesian_dir", default_bayesian_root("bayesian_joint"))
    legacy_decoupled = (
        getattr(args, "new_bayesian_dir", None)
        or default_bayesian_root("bayesian_decoupled")
    )
    decoupled = (
        getattr(args, "decoupled_bayesian_dir", None)
        or default_bayesian_root("bayesian_decoupled_no_dropout")
    )
    joint_dropout = (
        getattr(args, "joint_dropout_bayesian_dir", None)
        or default_bayesian_root("bayesian_joint_dropout")
    )
    decoupled_dropout = (
        getattr(args, "decoupled_dropout_bayesian_dir", None)
        or legacy_decoupled
    )
    metacell = (
        getattr(args, "metacell_bayesian_dir", None)
        or default_bayesian_root("bayesian_bootstrap_metacells_size100_number100")
    )
    return {
        "Bootstrap": getattr(args, "bootstrap_dir", OLD_DATA_BOOTSTRAP),
        "Bayesian decoupled": legacy_decoupled,
        "Bayesian joint": joint,
        "Joint": joint,
        "Decoupled": decoupled,
        "Joint+dropout": joint_dropout,
        "Decoupled+dropout": decoupled_dropout,
        "Metacell Bayesian": metacell,
    }


def load_corrected_activity(
    args: argparse.Namespace,
    methods: tuple[str, ...] = METHODS,
) -> tuple[dict[str, pd.DataFrame], dict[str, float]]:
    roots = method_roots(args)
    raw = {}
    corrections = {}
    for method in methods:
        if method == POINT_METHOD:
            continue
        root = roots[method]
        if method == "Bootstrap":
            raw[method] = bootstrap_base(root, args)[0]
        else:
            raw[method] = bayesian_base(root, args)[0]
            corrections[method] = mean_log_beta_t7(root)
            raw[method] = raw[method] - corrections[method]
    return raw, corrections


def blacklisted_cres_for_methods(
    args: argparse.Namespace, methods: tuple[str, ...]
) -> tuple[set[str], dict[str, list[str]]]:
    roots = method_roots(args)
    sources = {
        method: sorted(read_cre_blacklist(roots[method]))
        for method in methods
        if method != POINT_METHOD
    }
    blacklist = set().union(*(set(cres) for cres in sources.values()))
    return blacklist, sources


def finite_series(values: pd.Series) -> pd.Series:
    values = pd.to_numeric(values, errors="coerce")
    return values[np.isfinite(values.to_numpy(float))]


def axis_limit(values: pd.Series) -> tuple[float, float]:
    finite = finite_series(values).to_numpy(float)
    if finite.size == 0:
        return 0.0, 1.0
    lo, hi = np.percentile(finite, [0.5, 99.5])
    if not np.isfinite(lo) or not np.isfinite(hi):
        return 0.0, 1.0
    if lo == hi:
        width = max(abs(lo) * 0.05, 0.5)
        return float(lo - width), float(hi + width)
    pad = (hi - lo) * 0.03
    return float(lo - pad), float(hi + pad)


def stack_methods(
    matrices: dict[str, pd.DataFrame], methods: tuple[str, ...] = METHODS
) -> pd.DataFrame:
    series = []
    for method in methods:
        matrix = matrices[method].copy()
        index = pd.MultiIndex.from_product(
            [matrix.index.astype(str), matrix.columns.astype(str)],
            names=["group", "cre"],
        )
        values = pd.Series(
            matrix.to_numpy(float).ravel(),
            index=index,
            name=method,
        )
        series.append(values)
    return pd.concat(series, axis=1)


def correlation_row(
    variant: str,
    x_method: str,
    y_method: str,
    wide: pd.DataFrame,
) -> dict:
    pair = wide[[x_method, y_method]].replace([np.inf, -np.inf], np.nan).dropna()
    return {
        "variant": variant,
        "x_method": x_method,
        "y_method": y_method,
        "n_pairs": int(len(pair)),
        "pearson": float(pair[x_method].corr(pair[y_method], method="pearson"))
        if len(pair) > 1
        else np.nan,
        "spearman": float(pair[x_method].corr(pair[y_method], method="spearman"))
        if len(pair) > 1
        else np.nan,
    }


def scatter_style_for_variant(variant: str) -> dict:
    return {
        "s": 2,
        "alpha": 0.08,
        "linewidths": 0,
        "color": "#2f6f8f",
        "rasterized": not variant.startswith("t7_gt"),
    }


def plot_scatter_matrix(
    wide: pd.DataFrame,
    variant: str,
    filter_label: str,
    output: Path,
    methods: tuple[str, ...] = METHODS,
) -> list[dict]:
    sns.set_theme(context="paper", style="white")
    limits = {method: axis_limit(wide[method]) for method in methods}
    scatter_style = scatter_style_for_variant(variant)
    fig, axes = plt.subplots(
        len(methods),
        len(methods),
        figsize=(max(13.8, 2.7 * len(methods)), max(13.8, 2.7 * len(methods))),
        constrained_layout=True,
        squeeze=False,
    )
    correlations = []
    for row, y_method in enumerate(methods):
        for col, x_method in enumerate(methods):
            ax = axes[row, col]
            ax.set_xlim(limits[x_method])
            if row == col:
                values = finite_series(wide[x_method])
                sns.histplot(
                    values,
                    bins=80,
                    color="#4c78a8",
                    edgecolor=None,
                    ax=ax,
                )
                ax.text(
                    0.03,
                    0.95,
                    f"n={len(values):,}",
                    transform=ax.transAxes,
                    ha="left",
                    va="top",
                    fontsize=8,
                )
            else:
                stats = correlation_row(variant, x_method, y_method, wide)
                correlations.append(stats)
                pair = (
                    wide[[x_method, y_method]]
                    .replace([np.inf, -np.inf], np.nan)
                    .dropna()
                )
                ax.scatter(
                    pair[x_method],
                    pair[y_method],
                    **scatter_style,
                )
                ax.set_ylim(limits[y_method])
                lo = max(limits[x_method][0], limits[y_method][0])
                hi = min(limits[x_method][1], limits[y_method][1])
                if lo < hi:
                    ax.plot([lo, hi], [lo, hi], color="black", linewidth=0.6, alpha=0.5)
                ax.text(
                    0.03,
                    0.97,
                    "r={pearson:.3f}\nrho={spearman:.3f}\nn={n_pairs:,}".format(
                        **stats
                    ),
                    transform=ax.transAxes,
                    ha="left",
                    va="top",
                    fontsize=8,
                    bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
                )
            if row == len(methods) - 1:
                ax.set_xlabel(x_method)
            else:
                ax.set_xlabel("")
                ax.set_xticklabels([])
            if col == 0:
                ax.set_ylabel(y_method)
            else:
                ax.set_ylabel("")
                ax.set_yticklabels([])
    fig.suptitle(
        f"Method activity correlations ({filter_label})\n"
        "Bayesian values use log_gamma - E[log(beta_t7)]; "
        "point estimate is log(total cCRE / total T7); "
        "black line is y = x; axes show 0.5-99.5 percentile ranges.",
        fontsize=12,
    )
    sns.despine(fig=fig)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    return correlations


def pair_count_totals(
    h5ad: Path,
    candidate_index: pd.Index,
    candidate_columns: pd.Index,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    adata = read_and_prepare_adata(h5ad)
    totals = {}
    for label, obsm_key in (("t7", "T7CRE"), ("cre", "CRE")):
        counts = adata.obsm[obsm_key].copy()
        counts.index = adata.obs["subclass"].astype(str).to_numpy()
        grouped = counts.groupby(level=0, sort=False).sum()
        grouped.index = grouped.index.astype(str)
        grouped.columns = grouped.columns.astype(str)
        totals[label] = grouped.reindex(
            index=candidate_index.astype(str),
            columns=candidate_columns.astype(str),
            fill_value=0.0,
        ).fillna(0.0)
    return totals["t7"], totals["cre"]


def point_log_cre_t7(pair_cre: pd.DataFrame, pair_t7: pd.DataFrame) -> pd.DataFrame:
    cre = pair_cre.to_numpy(float)
    t7 = pair_t7.to_numpy(float)
    ratio = np.divide(
        cre,
        t7,
        out=np.full(cre.shape, np.nan, dtype=float),
        where=(cre > 0) & (t7 > 0),
    )
    return pd.DataFrame(np.log(ratio), index=pair_cre.index, columns=pair_cre.columns)


def pair_count_series(pair_counts: pd.DataFrame, name: str) -> pd.Series:
    index = pd.MultiIndex.from_product(
        [pair_counts.index.astype(str), pair_counts.columns.astype(str)],
        names=["group", "cre"],
    )
    return pd.Series(pair_counts.to_numpy(float).ravel(), index=index, name=name)


def plot_count_colored_scatter_matrix(
    wide: pd.DataFrame,
    total_counts: pd.Series,
    variant: str,
    filter_label: str,
    output: Path,
    methods: tuple[str, ...],
    count_label: str,
    color_by: str,
    center_count: float = 10.0,
    colorbar_ticks: tuple[float, ...] = (5, 10, 20, 50, 100),
) -> dict:
    sns.set_theme(context="paper", style="white")
    limits = {method: axis_limit(wide[method]) for method in methods}
    total_counts = pd.to_numeric(total_counts.reindex(wide.index), errors="coerce")
    positive_counts = total_counts[
        np.isfinite(total_counts.to_numpy(float)) & total_counts.gt(0).to_numpy()
    ]
    cmap = "coolwarm_r"
    vmax_floor = max(float(max(colorbar_ticks)), center_count * 1.1)
    vmax = float(max(vmax_floor, positive_counts.max())) if len(positive_counts) else vmax_floor
    norm = mcolors.TwoSlopeNorm(
        vmin=np.log10(1.0),
        vcenter=np.log10(center_count),
        vmax=np.log10(vmax),
    )
    ticks = [tick for tick in colorbar_ticks if tick <= vmax]
    tick_positions = [np.log10(tick) for tick in ticks]
    zero_color = "#7b3294"

    fig, axes = plt.subplots(
        len(methods),
        len(methods),
        figsize=(max(14.6, 2.9 * len(methods)), max(13.8, 2.7 * len(methods))),
        constrained_layout=True,
        squeeze=False,
    )
    mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array([])
    n_zero_or_missing = 0
    n_positive = 0
    for row, y_method in enumerate(methods):
        for col, x_method in enumerate(methods):
            ax = axes[row, col]
            ax.set_xlim(limits[x_method])
            if row == col:
                values = finite_series(wide[x_method])
                sns.histplot(
                    values,
                    bins=80,
                    color="#4c78a8",
                    edgecolor=None,
                    ax=ax,
                )
                ax.text(
                    0.03,
                    0.95,
                    f"n={len(values):,}",
                    transform=ax.transAxes,
                    ha="left",
                    va="top",
                    fontsize=8,
                )
            else:
                stats = correlation_row(variant, x_method, y_method, wide)
                pair = (
                    wide[[x_method, y_method]]
                    .replace([np.inf, -np.inf], np.nan)
                    .dropna()
                )
                pair_counts = total_counts.reindex(pair.index)
                pair_count_array = pair_counts.to_numpy(float)
                valid_counts = np.isfinite(pair_count_array)
                positive = valid_counts & (pair_count_array > 0)
                background = ~positive
                n_positive += int(positive.sum())
                n_zero_or_missing += int(background.sum())
                if background.any():
                    ax.scatter(
                        pair[x_method].to_numpy()[background],
                        pair[y_method].to_numpy()[background],
                        s=1.2,
                        alpha=0.08,
                        linewidths=0,
                        color=zero_color,
                        rasterized=True,
                    )
                if positive.any():
                    ax.scatter(
                        pair[x_method].to_numpy()[positive],
                        pair[y_method].to_numpy()[positive],
                        c=np.log10(pair_count_array[positive]),
                        cmap=cmap,
                        norm=norm,
                        s=2,
                        alpha=0.16,
                        linewidths=0,
                        rasterized=True,
                    )
                ax.set_ylim(limits[y_method])
                lo = max(limits[x_method][0], limits[y_method][0])
                hi = min(limits[x_method][1], limits[y_method][1])
                if lo < hi:
                    ax.plot([lo, hi], [lo, hi], color="black", linewidth=0.6, alpha=0.5)
                ax.text(
                    0.03,
                    0.97,
                    "r={pearson:.3f}\nrho={spearman:.3f}\nn={n_pairs:,}".format(
                        **stats
                    ),
                    transform=ax.transAxes,
                    ha="left",
                    va="top",
                    fontsize=8,
                    bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
                )
            if row == len(methods) - 1:
                ax.set_xlabel(x_method)
            else:
                ax.set_xlabel("")
                ax.set_xticklabels([])
            if col == 0:
                ax.set_ylabel(y_method)
            else:
                ax.set_ylabel("")
                ax.set_yticklabels([])
    colorbar = fig.colorbar(
        mappable,
        ax=axes,
        ticks=tick_positions,
        fraction=0.018,
        pad=0.01,
    )
    colorbar.set_label(
        f"{count_label} in subclass-cCRE pair (log scale; center={center_count:g})"
    )
    colorbar.ax.set_yticklabels([str(tick) for tick in ticks])
    fig.suptitle(
        f"Method activity correlations ({filter_label})\n"
        f"Off-diagonal points colored by {count_label}; purple means count = 0. "
        "Black line is y = x.",
        fontsize=12,
    )
    sns.despine(fig=fig)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    return {
        "variant": variant,
        "output": str(output),
        "color_by": color_by,
        "colormap": "coolwarm_r",
        "zero_count_color": zero_color,
        "color_norm": f"TwoSlopeNorm(log10 count, vcenter=log10({center_count:g}))",
        "colorbar_ticks": ticks,
        "center_count": center_count,
        "positive_points_across_offdiagonal_panels": n_positive,
        "zero_or_missing_points_across_offdiagonal_panels": n_zero_or_missing,
    }


def read_subclass_cell_counts(root: Path) -> pd.Series:
    counts = pd.read_csv(root / "subclass_cell_counts.csv")
    return counts.set_index(counts["subclass"].astype(str))["n_cells"].astype(int)


def row_filter_mask(
    index: pd.Index,
    columns: pd.Index,
    keep_rows: pd.Series,
) -> pd.DataFrame:
    keep_rows = keep_rows.reindex(index.astype(str), fill_value=False).astype(bool)
    return pd.DataFrame(
        np.repeat(keep_rows.to_numpy()[:, None], len(columns), axis=1),
        index=index,
        columns=columns,
    )


def column_filter_mask(
    index: pd.Index,
    columns: pd.Index,
    keep_columns: pd.Series,
) -> pd.DataFrame:
    keep_columns = keep_columns.reindex(columns.astype(str), fill_value=False).astype(bool)
    return pd.DataFrame(
        np.repeat(keep_columns.to_numpy()[None, :], len(index), axis=0),
        index=index,
        columns=columns,
    )


def read_nanopore_counts(path: Path) -> pd.Series:
    counts = pd.read_csv(path, index_col=0)
    if "counts" in counts.columns:
        values = counts["counts"]
    elif counts.shape[1] == 1:
        values = counts.iloc[:, 0]
    else:
        raise ValueError(f"{path} must contain a counts column")
    values.index = values.index.astype(str)
    return values.astype(float)


def prepare_base(
    args: argparse.Namespace,
    methods: tuple[str, ...] = METHODS,
) -> tuple[
    dict[str, pd.DataFrame],
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.Series,
    dict,
]:
    fitted_methods = tuple(method for method in methods if method != POINT_METHOD)
    raw, corrections = load_corrected_activity(args, fitted_methods)
    rows, columns = combined_axes(raw)
    blacklist, blacklist_sources = blacklisted_cres_for_methods(args, methods)
    columns = pd.Index(
        [cre for cre in columns.astype(str) if cre not in blacklist],
        dtype=str,
    )
    matrices = {
        method: raw[method].reindex(index=rows, columns=columns)
        for method in fitted_methods
    }
    pair_t7, pair_cre = pair_count_totals(args.h5ad, rows, columns)
    if POINT_METHOD in methods:
        matrices[POINT_METHOD] = point_log_cre_t7(pair_cre, pair_t7)
    cell_counts = read_subclass_cell_counts(args.bootstrap_dir)
    nanopore_counts = read_nanopore_counts(args.libsize_csv)
    metadata = {
        "bayesian_activity_scale": "log_gamma - mean_log_beta_t7",
        "black_line": "identity line y = x, not a regression line",
        "mean_log_beta_t7_corrections": corrections,
        "blacklisted_cres_removed": int(len(blacklist)),
        "blacklist_sources": blacklist_sources,
        "point_estimate_activity": (
            "Point log(cCRE/T7) = log(total CRE counts / total T7 counts); "
            "pairs with total CRE <= 0 or total T7 <= 0 are set to NaN"
        ),
        "bayesian_roots": {
            method: str(root)
            for method, root in method_roots(args).items()
            if method in methods and method != "Bootstrap"
        },
        "rows_before_variant_trimming": int(len(rows)),
        "columns_after_blacklist": int(len(columns)),
        "nanopore_counts_source": str(args.libsize_csv),
    }
    return matrices, pair_t7, pair_cre, cell_counts, nanopore_counts, metadata


def plot_correlations(args: argparse.Namespace) -> dict:
    methods = CORRELATION_METHODS
    matrices, pair_t7, pair_cre, cell_counts, nanopore_counts, metadata = prepare_base(
        args, methods
    )
    high_cell_rows = cell_counts.gt(args.cell_count_threshold)
    high_nanopore_columns = nanopore_counts.gt(args.nanopore_threshold)
    variants = {
        "complete": {
            "filter_label": "complete, blacklist cCREs removed",
            "filter_kind": "complete",
            "mask": pd.DataFrame(True, index=pair_t7.index, columns=pair_t7.columns),
        },
        "cellgt1000": {
            "filter_label": (
                f"cell types with n_cells > {args.cell_count_threshold:,}"
            ),
            "filter_kind": "cell_count",
            "mask": row_filter_mask(pair_t7.index, pair_t7.columns, high_cell_rows),
        },
        "t7nanoporegt1000": {
            "filter_label": (
                f"cCRE nanopore read counts > {args.nanopore_threshold:g}"
            ),
            "filter_kind": "nanopore_count",
            "mask": column_filter_mask(
                pair_t7.index, pair_t7.columns, high_nanopore_columns
            ),
        },
    }
    t7_thresholds = sorted({float(args.t7_threshold), *map(float, args.t7_thresholds)})
    t7_variants = {
        f"t7_gt{threshold:g}": {
            "filter_label": f"subclass-cCRE total T7 > {threshold:g}",
            "filter_kind": "pair_t7",
            "threshold": threshold,
            "mask": pair_t7.gt(threshold),
        }
        for threshold in t7_thresholds
    }
    variants = {
        "complete": variants["complete"],
        **t7_variants,
        "cellgt1000": variants["cellgt1000"],
        "t7nanoporegt1000": variants["t7nanoporegt1000"],
    }
    outputs = {}
    colored_outputs = {}
    variant_summary = {}
    correlations = []
    for variant, spec in variants.items():
        masked = {method: matrix.where(spec["mask"]) for method, matrix in matrices.items()}
        trimmed, rows, columns, present = trim_empty_axes(masked)
        wide = stack_methods(trimmed, methods)
        output = args.figures_dir / f"{args.stem}_{variant}.pdf"
        correlations.extend(
            plot_scatter_matrix(wide, variant, spec["filter_label"], output, methods)
        )
        outputs[variant] = str(output)
        variant_summary[variant] = {
            "rows": int(len(rows)),
            "columns": int(len(columns)),
            "finite_pairs_any_method": int(present.to_numpy(bool).sum()),
            "passing_filter_pairs": int(spec["mask"].to_numpy(bool).sum()),
            "filter_kind": spec["filter_kind"],
            "output": str(output),
        }
        if variant == "complete":
            t7_colored_output = (
                args.figures_dir / f"{args.stem}_{variant}_t7_colored.pdf"
            )
            cre_colored_output = (
                args.figures_dir / f"{args.stem}_{variant}_ccre_colored.pdf"
            )
            colored_outputs[f"{variant}_t7"] = plot_count_colored_scatter_matrix(
                wide,
                pair_count_series(
                    pair_t7.reindex(index=rows, columns=columns), "total_t7"
                ),
                variant,
                spec["filter_label"],
                t7_colored_output,
                methods,
                "Total T7 count",
                "subclass_cCRE_total_t7",
            )
            colored_outputs[f"{variant}_ccre"] = plot_count_colored_scatter_matrix(
                wide,
                pair_count_series(
                    pair_cre.reindex(index=rows, columns=columns), "total_ccre"
                ),
                variant,
                spec["filter_label"],
                cre_colored_output,
                methods,
                "Total cCRE count",
                "subclass_cCRE_total_ccre",
                center_count=5.0,
                colorbar_ticks=(1, 2, 3, 4),
            )

    summary = {
        **metadata,
        "activity_calibration": args.activity_calibration,
        "methods": list(methods),
        "outputs": outputs,
        "colored_outputs": colored_outputs,
        "variants": variant_summary,
        "correlations": correlations,
        "t7_threshold": args.t7_threshold,
        "t7_thresholds": t7_thresholds,
        "cell_count_threshold": args.cell_count_threshold,
        "nanopore_threshold": args.nanopore_threshold,
        "cell_types_above_threshold": int(high_cell_rows.sum()),
        "cres_above_nanopore_threshold": int(
            high_nanopore_columns.reindex(pair_t7.columns.astype(str), fill_value=False)
            .astype(bool)
            .sum()
        ),
        "pair_t7_source": "sum of H5AD T7CRE counts by subclass-cCRE pair",
        "pair_ccre_source": "sum of H5AD CRE counts by subclass-cCRE pair",
    }
    write_json(args.figures_dir / f"{args.stem}_manifest.json", summary)
    return summary


def main() -> None:
    args = parse_args()
    args.figures_dir.mkdir(parents=True, exist_ok=True)
    summary = plot_correlations(args)
    log(
        "[method activity correlation] wrote "
        f"{len(summary['outputs'])} scatter-matrix PDFs"
    )


if __name__ == "__main__":
    main()
