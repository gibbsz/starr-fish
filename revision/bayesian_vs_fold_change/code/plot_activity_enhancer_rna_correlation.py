#!/usr/bin/env python3
"""Compare methods by correlation of activity with enhancer RNA signal.

For every activity-estimation method and matched cell subclass, this script
computes Spearman correlation across non-gene-body cCREs between the activity
estimate and the mean unstranded RNA RPKM signal. Subclass-cCRE pairs are first
restricted to total T7 count greater than or equal to the requested threshold.
The final scatter matrix compares these per-subclass correlations between
methods, matching the corresponding ATAC comparison figure.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis_utils import (
    ANALYSIS_DIR,
    DEFAULT_H5AD,
    REVISION_DATA,
    STARRFISH_DATA,
    log,
    write_json,
)
from plot_activity_atac_correlation import (
    METHODS,
    activity_atac_correlations,
    load_activity,
    scatter_matrix_figure,
    threshold_suffix,
)
from plot_method_activity_correlation import pair_count_totals


SPEARMAN_COLUMN = "spearman_enhancer_rna_rpkm"
PEARSON_COLUMN = "pearson_log1p_enhancer_rna_rpkm"
DEFAULT_STEM = "method_activity_enhancer_rna_correlation"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--enhancer-rna",
        type=Path,
        default=REVISION_DATA / "Enhancer_RNA.csv",
        help="cCRE-by-track matrix produced by revision/Data/Enhancer_RNA.py.",
    )
    parser.add_argument(
        "--subclass-annotation",
        type=Path,
        default=STARRFISH_DATA / "abc_atlas" / "cluster_annotation_term.csv",
        help=(
            "BICCN annotation used to map numeric bigWig track prefixes to "
            "the exact activity subclass names."
        ),
    )
    parser.add_argument(
        "--bootstrap-dir",
        type=Path,
        default=ANALYSIS_DIR / "results" / "bootstrap",
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
    parser.add_argument("--h5ad", type=Path, default=DEFAULT_H5AD)
    parser.add_argument("--t7-threshold", type=float, default=50.0)
    parser.add_argument("--min-pairs", type=int, default=10)
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=METHODS,
        default=list(METHODS),
    )
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
    parser.add_argument(
        "--no-png",
        action="store_true",
        help="Write only PDF, without the matching PNG preview.",
    )
    args = parser.parse_args()
    if args.t7_threshold < 0:
        parser.error("--t7-threshold must be non-negative")
    if args.min_pairs < 2:
        parser.error("--min-pairs must be at least 2")
    return args


def _subclass_number(column: str) -> int | None:
    match = re.match(r"^(\d+)_", str(column))
    return int(match.group(1)) if match is not None else None


def read_enhancer_rna(
    path: Path,
    subclass_annotation: Path,
    activity_groups: pd.Index,
) -> tuple[pd.DataFrame, dict]:
    """Return enhancer RNA as subclass-by-cCRE with exact subclass labels."""

    signal = pd.read_csv(path, index_col=0)
    signal.index = signal.index.astype(str)
    signal.columns = signal.columns.astype(str)
    if signal.index.has_duplicates:
        raise ValueError(f"{path} contains duplicate cCRE IDs")
    if signal.columns.has_duplicates:
        raise ValueError(f"{path} contains duplicate track names")

    annotation = pd.read_csv(subclass_annotation)
    required = {"subclass_number", "subclass"}
    missing = required.difference(annotation.columns)
    if missing:
        raise ValueError(
            f"{subclass_annotation} is missing columns: {sorted(missing)}"
        )
    subclass_map = annotation.loc[:, ["subclass_number", "subclass"]].dropna()
    subclass_map["subclass_number"] = subclass_map["subclass_number"].astype(int)
    subclass_map["subclass"] = (
        subclass_map["subclass"].astype(str).str.replace("/", "-", regex=False)
    )
    ambiguity = subclass_map.groupby("subclass_number")["subclass"].nunique()
    if (ambiguity > 1).any():
        bad = ambiguity[ambiguity > 1].index.astype(str).tolist()
        raise ValueError(
            f"Ambiguous subclass names for subclass numbers: {bad[:10]}"
        )
    number_to_subclass = (
        subclass_map.drop_duplicates("subclass_number")
        .set_index("subclass_number")["subclass"]
        .to_dict()
    )

    activity_group_set = set(activity_groups.astype(str))
    selected_columns: list[str] = []
    selected_groups: list[str] = []
    unmapped_tracks: list[str] = []
    annotation_only_tracks: list[str] = []
    for column in signal.columns:
        number = _subclass_number(column)
        subclass = number_to_subclass.get(number) if number is not None else None
        if subclass is None:
            unmapped_tracks.append(column)
            continue
        if subclass not in activity_group_set:
            annotation_only_tracks.append(column)
            continue
        selected_columns.append(column)
        selected_groups.append(subclass)

    if not selected_columns:
        raise ValueError(
            "No Enhancer_RNA tracks map to activity subclasses; check the "
            "BICCN subclass annotation."
        )
    duplicate_groups = pd.Index(selected_groups)[
        pd.Index(selected_groups).duplicated(keep=False)
    ].unique()
    if len(duplicate_groups):
        raise ValueError(
            "Multiple Enhancer_RNA tracks map to the same activity subclass: "
            f"{duplicate_groups[:10].tolist()}"
        )

    matched = signal.loc[:, selected_columns].apply(pd.to_numeric, errors="coerce")
    matched.columns = selected_groups
    matched = matched.T
    matched.index.name = "group"
    matched.columns.name = "cre"
    values = matched.to_numpy(float)
    if not np.isfinite(values).all():
        raise ValueError(f"{path} contains non-finite signal values")

    metadata = {
        "enhancer_rna_tracks_total": int(signal.shape[1]),
        "enhancer_rna_cres_total": int(signal.shape[0]),
        "subclass_tracks_matched_to_activity": int(len(selected_columns)),
        "tracks_without_subclass_number_mapping": unmapped_tracks,
        "subclass_tracks_absent_from_activity": annotation_only_tracks,
    }
    return matched, metadata


def main() -> None:
    args = parse_args()
    args.figures_dir.mkdir(parents=True, exist_ok=True)
    args.tables_dir.mkdir(parents=True, exist_ok=True)
    methods = tuple(dict.fromkeys(args.methods))

    log("[activity enhancer RNA] loading activity estimates")
    activity, activity_metadata = load_activity(args, methods)
    activity_groups = pd.Index(next(iter(activity.values())).index.astype(str))

    log("[activity enhancer RNA] loading and matching enhancer RNA tracks")
    enhancer_rna, signal_metadata = read_enhancer_rna(
        args.enhancer_rna,
        args.subclass_annotation,
        activity_groups,
    )
    all_cres = pd.Index(
        sorted(set().union(*(set(matrix.columns) for matrix in activity.values()))),
        dtype=str,
    )
    common_groups = activity_groups.intersection(enhancer_rna.index.astype(str))
    common_cres = all_cres.intersection(enhancer_rna.columns.astype(str))
    activity = {
        method: matrix.reindex(index=common_groups, columns=common_cres)
        for method, matrix in activity.items()
    }
    enhancer_rna = enhancer_rna.reindex(index=common_groups, columns=common_cres)

    log("[activity enhancer RNA] loading subclass-cCRE T7 totals")
    pair_t7, _ = pair_count_totals(args.h5ad, common_groups, common_cres)
    pair_t7 = pair_t7.reindex(
        index=common_groups, columns=common_cres
    ).fillna(0.0)

    threshold = float(args.t7_threshold)
    log(
        "[activity enhancer RNA] computing per-subclass correlations "
        f"for T7 >= {threshold:g}"
    )
    correlations = activity_atac_correlations(
        activity,
        enhancer_rna,
        args.min_pairs,
        methods,
        pair_mask=pair_t7.ge(threshold),
        t7_threshold=threshold,
    ).rename(
        columns={
            "spearman_atac_cpm": SPEARMAN_COLUMN,
            "pearson_log1p_atac_cpm": PEARSON_COLUMN,
        }
    )

    suffix = f"t7_ge{threshold_suffix(threshold)}"
    table_path = args.tables_dir / f"{args.stem}_by_subclass_{suffix}.csv"
    correlations.to_csv(table_path, index=False)

    finite = correlations[np.isfinite(correlations[SPEARMAN_COLUMN].to_numpy(float))]
    summary = (
        finite.groupby("method", sort=False)
        .agg(
            n_subclasses=("group", "nunique"),
            median_spearman=(SPEARMAN_COLUMN, "median"),
            mean_spearman=(SPEARMAN_COLUMN, "mean"),
            median_pearson_log1p=(PEARSON_COLUMN, "median"),
            median_pairs=("n_pairs", "median"),
            mean_pairs=("n_pairs", "mean"),
        )
        .reindex(methods)
        .reset_index()
    )
    summary_path = args.tables_dir / f"{args.stem}_summary_{suffix}.csv"
    summary.to_csv(summary_path, index=False)

    figure_path = (
        args.figures_dir / f"{args.stem}_spearman_scatter_{suffix}.pdf"
    )
    fig = scatter_matrix_figure(
        correlations,
        threshold,
        methods,
        correlation_column=SPEARMAN_COLUMN,
        signal_label="Enhancer RNA",
    )
    fig.savefig(figure_path, bbox_inches="tight")
    png_path = figure_path.with_suffix(".png")
    if not args.no_png:
        fig.savefig(png_path, bbox_inches="tight", dpi=200)
    plt.close(fig)

    manifest = {
        **activity_metadata,
        **signal_metadata,
        "enhancer_rna": str(args.enhancer_rna),
        "subclass_annotation": str(args.subclass_annotation),
        "h5ad": str(args.h5ad),
        "activity_calibration": args.activity_calibration,
        "activity_centering": args.activity_centering,
        "methods": list(methods),
        "min_pairs": int(args.min_pairs),
        "t7_threshold": threshold,
        "t7_filter": "subclass-cCRE total T7 count >= threshold",
        "matched_subclasses": int(len(common_groups)),
        "matched_non_gene_body_cres": int(len(common_cres)),
        "finite_subclasses_by_method": {
            str(row.method): int(row.n_subclasses)
            for row in summary.itertuples(index=False)
        },
        "correlation_definition": (
            "Spearman correlation across non-gene-body cCREs within subclass "
            "after the T7 filter, using activity estimate and raw mean "
            "unstranded RNA RPKM signal"
        ),
        "figure": str(figure_path),
        "png": None if args.no_png else str(png_path),
        "by_subclass_table": str(table_path),
        "summary_table": str(summary_path),
    }
    manifest_path = args.figures_dir / f"{args.stem}_manifest_{suffix}.json"
    write_json(manifest_path, manifest)
    log(
        f"[activity enhancer RNA] wrote {figure_path}; "
        f"{len(common_groups)} subclasses and {len(common_cres)} cCREs matched"
    )


if __name__ == "__main__":
    main()
