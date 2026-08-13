#!/usr/bin/env python3
"""Redraw the cell-type by motif heatmaps from persisted result tables.

This avoids rerunning the FIMO scan in ``run_tf_motif_analysis.py``: it reads
the exported matrices under ``results/tables`` and calls the same plotting
function, so the figures stay byte-for-byte consistent with the pipeline.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from run_tf_motif_analysis import plot_weighted_motif_activity_heatmap

CODE_DIR = Path(__file__).resolve().parent
DEFAULT_RESULTS_DIR = CODE_DIR.parent / "results"

HEATMAPS: dict[str, dict[str, object]] = {
    "activity_mean": {
        "matrix": "cell_type_by_motif_activity.csv.gz",
        "n_valid_source": "activity",
        "output_stem": "weighted_motif_activity_heatmap",
        "title": "Top variable motif activities across valid T7≥50 cCREs",
        "colorbar_label": (
            "Mean [activity − negative-control mean] × [−log10(best FIMO p)]"
        ),
        "selection": "variability",
        "fixed_color_limit": None,
    },
    "activity_zscore": {
        "matrix": "cell_type_by_motif_activity_zscore.csv.gz",
        "n_valid_source": "activity",
        "output_stem": "weighted_motif_activity_zscore_heatmap",
        "title": "Within-cell-type z-scored motif activities",
        "colorbar_label": "Motif activity z-score within cell type",
        "selection": "variability",
        "fixed_color_limit": None,
    },
    "pearson_r": {
        "matrix": "cell_type_by_motif_activity_pearson_r.csv.gz",
        "n_valid_source": "correlation",
        "output_stem": "motif_activity_pearson_correlation_heatmap",
        "title": "cCRE activity versus motif matching score: Pearson correlation",
        "colorbar_label": "Pearson r across all valid T7≥50 cCREs",
        "selection": "max_absolute",
        "fixed_color_limit": 1.0,
    },
    "spearman_rho": {
        "matrix": "cell_type_by_motif_activity_spearman_rho.csv.gz",
        "n_valid_source": "correlation",
        "output_stem": "motif_activity_spearman_correlation_heatmap",
        "title": "cCRE activity versus motif matching score: Spearman correlation",
        "colorbar_label": "Spearman ρ across all valid T7≥50 cCREs",
        "selection": "max_absolute",
        "fixed_color_limit": 1.0,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Directory holding the pipeline's tables/ and figures/.",
    )
    parser.add_argument(
        "--heatmaps",
        nargs="+",
        choices=sorted(HEATMAPS),
        default=["activity_zscore"],
        help="Which heatmaps to redraw.",
    )
    parser.add_argument("--activity-heatmap-top-motifs", type=int, default=100)
    parser.add_argument(
        "--activity-heatmap-min-valid-ccres",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--no-cluster-cell-types",
        dest="cluster_cell_types",
        action="store_false",
        help="Order rows by numbered prefix instead of clustering them.",
    )
    parser.set_defaults(cluster_cell_types=True)
    return parser.parse_args()


def load_n_valid_ccres(tables_dir: Path, source: str) -> pd.Series:
    """Recover per-cell-type valid cCRE counts from the exported long table."""
    long_path = tables_dir / (
        "motif_activity_by_cell_type_long.csv.gz"
        if source == "activity"
        else "motif_activity_correlation_by_cell_type_long.csv.gz"
    )
    long = pd.read_csv(long_path, usecols=["group", "n_valid_ccres"])
    counts = long.groupby("group", sort=True)["n_valid_ccres"].max()
    if counts.empty:
        raise ValueError(f"No n_valid_ccres recorded in {long_path}")
    return counts.astype(int)


def main() -> None:
    args = parse_args()
    tables_dir = args.results_dir / "tables"
    figures_dir = args.results_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    annotation = pd.read_csv(
        tables_dir / "hocomoco_motif_to_mouse_gene.csv.gz",
        usecols=["motif", "mouse_gene_symbol"],
    )
    prefixes = pd.read_csv(
        tables_dir / "cell_type_number_prefixes.csv",
        usecols=["group", "numbered_prefix"],
    ).set_index("group")["numbered_prefix"]

    for key in args.heatmaps:
        spec = HEATMAPS[key]
        matrix = pd.read_csv(tables_dir / str(spec["matrix"]), index_col=0)
        n_valid = load_n_valid_ccres(tables_dir, str(spec["n_valid_source"]))
        plot_weighted_motif_activity_heatmap(
            matrix,
            annotation,
            prefixes,
            n_valid.reindex(matrix.index).fillna(0).astype(int),
            figures_dir,
            args.activity_heatmap_top_motifs,
            args.activity_heatmap_min_valid_ccres,
            str(spec["output_stem"]),
            str(spec["title"]),
            str(spec["colorbar_label"]),
            selection=str(spec["selection"]),
            fixed_color_limit=spec["fixed_color_limit"],  # type: ignore[arg-type]
            cluster_cell_types=args.cluster_cell_types,
        )
        print(f"wrote {figures_dir / str(spec['output_stem'])}.{{png,pdf}}")


if __name__ == "__main__":
    main()
