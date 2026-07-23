#!/usr/bin/env python3
"""Mark decoupled-greater-than-joint Bayesian outliers across correlation plots."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from analysis_utils import ANALYSIS_DIR, DEFAULT_H5AD, LIBSIZE_CSV, log, write_json
from plot_method_activity_correlation import (
    METHODS,
    axis_limit,
    column_filter_mask,
    finite_series,
    prepare_base,
    read_nanopore_counts,
    row_filter_mask,
)
from plot_method_activity_heatmap import trim_empty_axes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
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
        default=ANALYSIS_DIR / "results" / "ablation" / "bayesian_decoupled",
    )
    parser.add_argument(
        "--figures-dir", type=Path, default=ANALYSIS_DIR / "results" / "figures"
    )
    parser.add_argument("--h5ad", type=Path, default=DEFAULT_H5AD)
    parser.add_argument("--activity-calibration", choices=["none"], default="none")
    parser.add_argument("--bootstrap-log-chunk-size", type=int, default=250)
    parser.add_argument("--t7-threshold", type=float, default=100.0)
    parser.add_argument("--cell-count-threshold", type=int, default=1000)
    parser.add_argument("--nanopore-threshold", type=float, default=1000.0)
    parser.add_argument("--libsize-csv", type=Path, default=LIBSIZE_CSV)
    parser.add_argument("--delta-threshold", type=float, default=1.5)
    parser.add_argument("--strict-cre-min", type=int, default=5)
    parser.add_argument("--strict-t7-max", type=int, default=1)
    parser.add_argument(
        "--stem",
        default="method_activity_correlation_marked_decoupled_gt_joint",
    )
    return parser.parse_args()


def discover_bayes_tag(root: Path) -> str:
    return json.loads((root / "run_manifest.json").read_text())["tag"]


def stack_methods(matrices: dict[str, pd.DataFrame]) -> pd.DataFrame:
    series = []
    for method in METHODS:
        matrix = matrices[method]
        index = pd.MultiIndex.from_product(
            [matrix.index.astype(str), matrix.columns.astype(str)],
            names=["group", "cre"],
        )
        series.append(
            pd.Series(matrix.to_numpy(float).ravel(), index=index, name=method)
        )
    return pd.concat(series, axis=1)


def decoupled_evidence(root: Path) -> pd.DataFrame:
    tag = discover_bayes_tag(root)
    with np.load(root / f"{tag}_posterior_samples.npz", allow_pickle=True) as posterior:
        groups = pd.Index(posterior["group_names"].astype(str))
        cres = pd.Index(posterior["cre_names"].astype(str))
    evidence = pd.read_csv(root / f"{tag}_evidence_per_pair.csv")
    evidence["group"] = groups[evidence["group"].to_numpy(dtype=int)]
    evidence["cre"] = cres[evidence["cre"].to_numpy(dtype=int)]
    return evidence


def marked_pair_table(
    matrices: dict[str, pd.DataFrame],
    evidence: pd.DataFrame,
    delta_threshold: float,
    strict_cre_min: int,
    strict_t7_max: int,
) -> pd.DataFrame:
    delta = matrices["Bayesian decoupled"] - matrices["Bayesian joint"]
    table = (
        delta.stack(dropna=False)
        .rename("delta_decoupled_minus_joint")
        .reset_index()
        .rename(columns={"level_0": "group", "level_1": "cre"})
    )
    table["marked"] = table["delta_decoupled_minus_joint"].gt(delta_threshold)
    for method in METHODS:
        values = (
            matrices[method]
            .stack(dropna=False)
            .rename(method)
            .reset_index()
            .rename(columns={"level_0": "group", "level_1": "cre"})
        )
        table = table.merge(values, on=["group", "cre"], how="left")
    table = table.merge(evidence, on=["group", "cre"], how="left")
    table["cre_pos_gt_t7_pos"] = table["n_cre_pos"] > table["n_t7_pos"]
    table["high_cre_low_t7_strict"] = (
        table["n_cre_pos"].ge(strict_cre_min) & table["n_t7_pos"].le(strict_t7_max)
    )
    table["marked_high_cre_low_t7_strict"] = (
        table["marked"] & table["high_cre_low_t7_strict"]
    )
    return table.sort_values("delta_decoupled_minus_joint", ascending=False)


def pair_boolean_series(table: pd.DataFrame, column: str) -> pd.Series:
    pairs = pd.MultiIndex.from_frame(table[["group", "cre"]])
    return pd.Series(table[column].to_numpy(bool), index=pairs, name=column)


def correlation_stats(
    variant: str, x_method: str, y_method: str, wide: pd.DataFrame
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


def plot_marked_matrix(
    wide: pd.DataFrame,
    marked: pd.Series,
    high_cre_low_t7: pd.Series,
    variant: str,
    filter_label: str,
    output: Path,
) -> list[dict]:
    sns.set_theme(context="paper", style="white")
    marked = marked.reindex(wide.index, fill_value=False).astype(bool)
    high_cre_low_t7 = high_cre_low_t7.reindex(wide.index, fill_value=False).astype(bool)
    limits = {method: axis_limit(wide[method]) for method in METHODS}
    fig, axes = plt.subplots(
        len(METHODS),
        len(METHODS),
        figsize=(9.4, 9.4),
        constrained_layout=True,
        squeeze=False,
    )
    correlations = []
    for row, y_method in enumerate(METHODS):
        for col, x_method in enumerate(METHODS):
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
                stats = correlation_stats(variant, x_method, y_method, wide)
                correlations.append(stats)
                pair = (
                    wide[[x_method, y_method]]
                    .replace([np.inf, -np.inf], np.nan)
                    .dropna()
                )
                pair_marked = marked.reindex(pair.index, fill_value=False)
                pair_strict = high_cre_low_t7.reindex(pair.index, fill_value=False)
                background = pair[~pair_marked]
                highlighted = pair[pair_marked]
                strict = pair[pair_strict]
                ax.scatter(
                    background[x_method],
                    background[y_method],
                    s=1.4,
                    alpha=0.05,
                    linewidths=0,
                    color="#2f6f8f",
                    rasterized=True,
                )
                if not highlighted.empty:
                    ax.scatter(
                        highlighted[x_method],
                        highlighted[y_method],
                        s=8,
                        alpha=0.55,
                        linewidths=0,
                        color="#d62728",
                        rasterized=True,
                        label="Decoupled - joint > threshold",
                    )
                if not strict.empty:
                    ax.scatter(
                        strict[x_method],
                        strict[y_method],
                        s=18,
                        alpha=0.9,
                        linewidths=0.5,
                        edgecolor="black",
                        facecolor="#ffbf00",
                        marker="^",
                        rasterized=True,
                        label="Marked high cCRE / low T7",
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
            if row == len(METHODS) - 1:
                ax.set_xlabel(x_method)
            else:
                ax.set_xlabel("")
                ax.set_xticklabels([])
            if col == 0:
                ax.set_ylabel(y_method)
            else:
                ax.set_ylabel("")
                ax.set_yticklabels([])
    handles, labels = axes[0, 1].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(0.995, 0.995))
    fig.suptitle(
        f"Marked decoupled > joint outliers ({filter_label})\n"
        "Bayesian values use log_gamma - E[log(beta_t7)]; black line is y = x",
        fontsize=12,
    )
    sns.despine(fig=fig)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    return correlations


def plot_marked(args: argparse.Namespace) -> dict:
    matrices, pair_t7, _pair_cre, cell_counts, nanopore_counts, metadata = prepare_base(args)
    evidence = decoupled_evidence(args.new_bayesian_dir)
    table = marked_pair_table(
        matrices,
        evidence,
        args.delta_threshold,
        args.strict_cre_min,
        args.strict_t7_max,
    )
    marked = pair_boolean_series(table, "marked")
    high_cre_low_t7 = pair_boolean_series(table, "marked_high_cre_low_t7_strict")
    pair_csv = args.figures_dir / f"{args.stem}_pairs.csv"
    table.to_csv(pair_csv, index=False)

    high_cell_rows = cell_counts.gt(args.cell_count_threshold)
    high_nanopore_columns = nanopore_counts.gt(args.nanopore_threshold)
    variants = {
        "complete": {
            "filter_label": "complete, blacklist cCREs removed",
            "filter_kind": "complete",
            "mask": pd.DataFrame(True, index=pair_t7.index, columns=pair_t7.columns),
        },
        "t7_gt100": {
            "filter_label": f"subclass-cCRE total T7 > {args.t7_threshold:g}",
            "filter_kind": "pair_t7",
            "mask": pair_t7.gt(args.t7_threshold),
        },
        "cellgt1000": {
            "filter_label": f"cell types with n_cells > {args.cell_count_threshold:,}",
            "filter_kind": "cell_count",
            "mask": row_filter_mask(pair_t7.index, pair_t7.columns, high_cell_rows),
        },
        "t7nanoporegt1000": {
            "filter_label": f"cCRE nanopore read counts > {args.nanopore_threshold:g}",
            "filter_kind": "nanopore_count",
            "mask": column_filter_mask(
                pair_t7.index,
                pair_t7.columns,
                nanopore_counts.gt(args.nanopore_threshold),
            ),
        },
    }
    outputs = {}
    summaries = {}
    correlations = []
    for variant, spec in variants.items():
        masked_matrices = {
            method: matrix.where(spec["mask"]) for method, matrix in matrices.items()
        }
        trimmed, rows, columns, present = trim_empty_axes(masked_matrices)
        wide = stack_methods(trimmed)
        output = args.figures_dir / f"{args.stem}_{variant}.pdf"
        correlations.extend(
            plot_marked_matrix(
                wide, marked, high_cre_low_t7, variant, spec["filter_label"], output
            )
        )
        variant_marked = marked.reindex(wide.index, fill_value=False).astype(bool)
        variant_strict = high_cre_low_t7.reindex(wide.index, fill_value=False).astype(bool)
        outputs[variant] = str(output)
        summaries[variant] = {
            "rows": int(len(rows)),
            "columns": int(len(columns)),
            "finite_pairs_any_method": int(present.to_numpy(bool).sum()),
            "marked_pairs_present": int(variant_marked.sum()),
            "marked_high_cre_low_t7_pairs_present": int(variant_strict.sum()),
            "filter_kind": spec["filter_kind"],
        }
    marked_table = table[table["marked"]]
    summary = {
        **metadata,
        "outputs": outputs,
        "marked_pair_table": str(pair_csv),
        "variants": summaries,
        "correlations": correlations,
        "delta_threshold": args.delta_threshold,
        "strict_high_cre_low_t7_rule": {
            "n_cre_pos_min": args.strict_cre_min,
            "n_t7_pos_max": args.strict_t7_max,
        },
        "n_marked_pairs_total": int(table["marked"].sum()),
        "n_marked_cre_pos_gt_t7_pos": int(
            marked_table["cre_pos_gt_t7_pos"].sum()
        ),
        "n_marked_high_cre_low_t7_strict": int(
            marked_table["high_cre_low_t7_strict"].sum()
        ),
        "top_marked_pairs": marked_table.head(50).to_dict(orient="records"),
        "black_line": "identity line y = x, not a regression line",
        "bayesian_activity_scale": "log_gamma - mean_log_beta_t7",
    }
    write_json(args.figures_dir / f"{args.stem}_manifest.json", summary)
    return summary


def main() -> None:
    args = parse_args()
    args.figures_dir.mkdir(parents=True, exist_ok=True)
    summary = plot_marked(args)
    log(
        "[marked decoupled-joint outliers] wrote "
        f"{len(summary['outputs'])} marked scatter-matrix PDFs"
    )


if __name__ == "__main__":
    main()
