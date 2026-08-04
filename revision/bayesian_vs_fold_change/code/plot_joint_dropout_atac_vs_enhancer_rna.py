#!/usr/bin/env python3
"""Compare Joint+dropout correlations with ATAC and enhancer RNA.

Each point is one cell subclass. Within each subclass, both correlations use
the exact same cCREs: cCREs present in the Joint+dropout activity, ATAC, and
enhancer-RNA matrices; finite in all three; and passing the subclass-cCRE T7
threshold. The x and y values are Spearman correlations across those common
cCREs between activity and ATAC CPM or mean unstranded enhancer-RNA RPKM.
"""

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
from scipy.stats import pearsonr, spearmanr

from analysis_utils import (
    ANALYSIS_DIR,
    DEFAULT_H5AD,
    REVISION_DATA,
    STARRFISH_DATA,
    STARRFISH_ROOT,
    write_json,
)
from plot_activity_atac_correlation import (
    load_activity,
    read_atac_cpm_subset,
    read_cre_to_peak,
    threshold_suffix,
)
from plot_activity_enhancer_rna_correlation import read_enhancer_rna
from plot_method_activity_correlation import pair_count_totals


METHOD = "Joint+dropout"
ATAC_COLUMN = "spearman_atac_cpm"
RNA_COLUMN = "spearman_enhancer_rna_rpkm"
DEFAULT_STEM = "joint_dropout_atac_vs_enhancer_rna_correlation"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--enhancer-rna",
        type=Path,
        default=REVISION_DATA / "Enhancer_RNA.csv",
    )
    parser.add_argument(
        "--subclass-annotation",
        type=Path,
        default=STARRFISH_DATA / "abc_atlas" / "cluster_annotation_term.csv",
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
        default=ANALYSIS_DIR
        / "results"
        / "ablation"
        / "bayesian_joint_dropout",
    )
    parser.add_argument(
        "--decoupled-dropout-bayesian-dir",
        type=Path,
        default=None,
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
        "--bootstrap-dir",
        type=Path,
        default=ANALYSIS_DIR / "results" / "bootstrap",
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
    )
    parser.add_argument("--bootstrap-log-chunk-size", type=int, default=250)
    parser.add_argument("--atac-chunk-size", type=int, default=100_000)
    parser.add_argument("--h5ad", type=Path, default=DEFAULT_H5AD)
    parser.add_argument("--t7-threshold", type=float, default=50.0)
    parser.add_argument("--min-pairs", type=int, default=10)
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=ANALYSIS_DIR / "results" / "figures",
    )
    parser.add_argument(
        "--tables-dir",
        type=Path,
        default=ANALYSIS_DIR / "results" / "tables",
    )
    parser.add_argument("--stem", default=DEFAULT_STEM)
    args = parser.parse_args()
    if args.t7_threshold < 0:
        parser.error("--t7-threshold must be non-negative")
    if args.min_pairs < 2:
        parser.error("--min-pairs must be at least 2")
    return args


def build_comparison(args: argparse.Namespace) -> tuple[pd.DataFrame, dict]:
    activity_by_method, activity_metadata = load_activity(args, (METHOD,))
    activity = activity_by_method[METHOD]
    activity.index = activity.index.astype(str)
    activity.columns = activity.columns.astype(str)

    all_groups = pd.Index(activity.index, dtype=str)
    all_cres = pd.Index(activity.columns, dtype=str)
    cre_to_peak = read_cre_to_peak(args.cre_info, all_cres)
    atac, atac_metadata = read_atac_cpm_subset(
        args.atac_cpm,
        cre_to_peak,
        all_groups,
        args.atac_chunk_size,
    )
    enhancer_rna, enhancer_rna_metadata = read_enhancer_rna(
        args.enhancer_rna,
        args.subclass_annotation,
        all_groups,
    )

    common_groups = (
        all_groups.intersection(atac.index.astype(str))
        .intersection(enhancer_rna.index.astype(str))
    )
    common_cres = (
        all_cres.intersection(atac.columns.astype(str))
        .intersection(enhancer_rna.columns.astype(str))
    )
    activity = activity.reindex(index=common_groups, columns=common_cres)
    atac = atac.reindex(index=common_groups, columns=common_cres)
    enhancer_rna = enhancer_rna.reindex(
        index=common_groups,
        columns=common_cres,
    )
    pair_t7, _ = pair_count_totals(args.h5ad, common_groups, common_cres)
    pair_t7 = pair_t7.reindex(
        index=common_groups,
        columns=common_cres,
    ).fillna(0.0)

    rows = []
    for group in common_groups:
        activity_values = activity.loc[group].to_numpy(float)
        atac_values = atac.loc[group].to_numpy(float)
        enhancer_rna_values = enhancer_rna.loc[group].to_numpy(float)
        t7_values = pair_t7.loc[group].to_numpy(float)
        keep = (
            (t7_values >= args.t7_threshold)
            & np.isfinite(activity_values)
            & np.isfinite(atac_values)
            & np.isfinite(enhancer_rna_values)
        )
        n_common_cres = int(keep.sum())
        if n_common_cres < args.min_pairs:
            continue

        selected_activity = activity_values[keep]
        selected_atac = atac_values[keep]
        selected_enhancer_rna = enhancer_rna_values[keep]
        if (
            np.std(selected_activity) == 0
            or np.std(selected_atac) == 0
            or np.std(selected_enhancer_rna) == 0
        ):
            continue
        rows.append(
            {
                "group": str(group),
                "n_common_cCREs": n_common_cres,
                ATAC_COLUMN: float(
                    spearmanr(selected_activity, selected_atac).statistic
                ),
                RNA_COLUMN: float(
                    spearmanr(
                        selected_activity,
                        selected_enhancer_rna,
                    ).statistic
                ),
            }
        )

    comparison = pd.DataFrame(rows).sort_values("group").reset_index(drop=True)
    metadata = {
        **activity_metadata,
        **atac_metadata,
        **enhancer_rna_metadata,
        "matched_subclasses_before_pair_filter": int(len(common_groups)),
        "matched_cres_before_pair_filter": int(len(common_cres)),
        "common_cCRE_definition": (
            "Intersection of Joint+dropout activity, ATAC, and enhancer-RNA "
            "cCREs; within each subclass, T7 >= threshold and finite in all "
            "three matrices"
        ),
    }
    return comparison, metadata


def common_axis_limits(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    combined = np.concatenate([x, y])
    lo = min(float(np.nanmin(combined)), 0.0)
    hi = max(float(np.nanmax(combined)), 0.0)
    padding = max(0.07 * (hi - lo), 0.04)
    return max(-1.0, lo - padding), min(1.0, hi + padding)


def plot_comparison(
    comparison: pd.DataFrame,
    threshold: float,
    output_pdf: Path,
    output_png: Path,
) -> dict:
    if len(comparison) < 2:
        raise ValueError("Fewer than two cell types have finite correlations")

    x = comparison[ATAC_COLUMN].to_numpy(float)
    y = comparison[RNA_COLUMN].to_numpy(float)
    pearson = float(pearsonr(x, y).statistic)
    spearman = float(spearmanr(x, y).statistic)
    pair_counts = comparison["n_common_cCREs"].to_numpy(float)
    vmin = float(pair_counts.min())
    vmax = float(pair_counts.max())
    if vmin >= vmax:
        vmax = vmin + 1.0
    count_norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)

    sns.set_theme(context="talk", style="white")
    fig, ax = plt.subplots(figsize=(8.2, 7.2), constrained_layout=True)
    scatter = ax.scatter(
        x,
        y,
        c=pair_counts,
        cmap="coolwarm",
        norm=count_norm,
        s=62,
        alpha=0.86,
        edgecolor="white",
        linewidth=0.55,
        zorder=3,
    )
    lo, hi = common_axis_limits(x, y)
    ax.plot(
        [lo, hi],
        [lo, hi],
        color="0.35",
        linestyle="--",
        linewidth=1.0,
        label="y = x",
        zorder=1,
    )
    ax.axhline(0, color="0.7", linestyle=":", linewidth=0.9, zorder=0)
    ax.axvline(0, color="0.7", linestyle=":", linewidth=0.9, zorder=0)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"Spearman $\rho$(Joint+dropout activity, ATAC CPM)")
    ax.set_ylabel(
        r"Spearman $\rho$(Joint+dropout activity, enhancer RNA RPKM)"
    )
    ax.set_title(
        f"ATAC versus enhancer-RNA correlation by cell type, T7 >= {threshold:g}\n"
        "Both axes use the same cCREs within each cell subclass",
        loc="left",
        fontsize=14,
    )
    ax.text(
        0.04,
        0.96,
        f"Pearson r = {pearson:.3f}\n"
        f"Spearman $\\rho$ = {spearman:.3f}\n"
        f"n = {len(comparison)} cell types",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": "white",
            "edgecolor": "0.8",
            "alpha": 0.9,
        },
    )
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    colorbar = fig.colorbar(scatter, ax=ax, pad=0.02)
    colorbar.set_label("Common cCREs used for both correlations")
    sns.despine(fig=fig, ax=ax)

    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_pdf, bbox_inches="tight")
    fig.savefig(output_png, bbox_inches="tight", dpi=220)
    plt.close(fig)
    return {
        "pearson_across_cell_types": pearson,
        "spearman_across_cell_types": spearman,
        "n_cell_types": int(len(comparison)),
        "min_common_cCREs": int(pair_counts.min()),
        "max_common_cCREs": int(pair_counts.max()),
        "x_min": float(x.min()),
        "x_max": float(x.max()),
        "y_min": float(y.min()),
        "y_max": float(y.max()),
    }


def main() -> None:
    args = parse_args()
    suffix = f"t7_ge{threshold_suffix(args.t7_threshold)}"
    comparison, input_metadata = build_comparison(args)
    table_path = args.tables_dir / f"{args.stem}_{suffix}.csv"
    figure_pdf = args.figures_dir / f"{args.stem}_{suffix}.pdf"
    figure_png = args.figures_dir / f"{args.stem}_{suffix}.png"
    manifest_path = args.figures_dir / f"{args.stem}_{suffix}_manifest.json"

    args.tables_dir.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(table_path, index=False)
    statistics = plot_comparison(
        comparison,
        args.t7_threshold,
        figure_pdf,
        figure_png,
    )
    manifest = {
        **input_metadata,
        "method": METHOD,
        "t7_threshold": float(args.t7_threshold),
        "min_pairs": int(args.min_pairs),
        "activity_centering": args.activity_centering,
        "activity_calibration": args.activity_calibration,
        "atac_cpm": str(args.atac_cpm),
        "enhancer_rna": str(args.enhancer_rna),
        "subclass_annotation": str(args.subclass_annotation),
        "cre_info": str(args.cre_info),
        "h5ad": str(args.h5ad),
        "x_definition": (
            "Within-cell-type Spearman correlation on the common cCRE mask "
            "between Joint+dropout activity and ATAC CPM"
        ),
        "y_definition": (
            "Within-cell-type Spearman correlation on the same common cCRE "
            "mask between Joint+dropout activity and mean unstranded "
            "enhancer-RNA RPKM"
        ),
        "color_definition": "Number of cCREs in the shared per-cell-type mask",
        "color_map": "coolwarm",
        "figure_pdf": str(figure_pdf),
        "figure_png": str(figure_png),
        "comparison_table": str(table_path),
        **statistics,
    }
    write_json(manifest_path, manifest)
    print(
        f"Wrote {figure_pdf} with {statistics['n_cell_types']} cell types; "
        f"Pearson r={statistics['pearson_across_cell_types']:.3f}, "
        f"Spearman rho={statistics['spearman_across_cell_types']:.3f}"
    )


if __name__ == "__main__":
    main()
