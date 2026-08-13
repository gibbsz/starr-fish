#!/usr/bin/env python3
"""Motif activity next to the atlas expression of the corresponding TFs.

Rows and columns are taken verbatim from
``weighted_motif_activity_zscore_heatmap`` (same cell types, same clustered
motif order, same dendrograms).  Each motif column is joined to its mouse TF
gene in the whole-brain-atlas pseudobulk
(``revision/TF_enrichment/tables/atlas_pseudobulk.h5ad``: per-cell CPM over all
atlas genes, log1p, averaged within subclass, plus the per-subclass fraction of
cells with a non-zero count).  The joint figure stacks, in order, motif
activity, TF expression level, and TF detection fraction.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

from run_tf_motif_analysis import (
    HeatmapPanel,
    cell_type_axis_labels,
    draw_clustered_heatmap,
    draw_stacked_clustered_heatmaps,
    motif_axis_labels,
    prepare_motif_display,
)

CODE_DIR = Path(__file__).resolve().parent
ANALYSIS_DIR = CODE_DIR.parent
DEFAULT_RESULTS_DIR = ANALYSIS_DIR / "results"
DEFAULT_PSEUDOBULK = (
    ANALYSIS_DIR.parent / "TF_enrichment" / "tables" / "atlas_pseudobulk.h5ad"
)
EXPRESSION_LABEL = "Atlas TF expression (mean log1p CPM)"
FRACTION_LABEL = "Fraction of cells expressing the TF"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--pseudobulk", type=Path, default=DEFAULT_PSEUDOBULK)
    parser.add_argument("--activity-heatmap-top-motifs", type=int, default=100)
    parser.add_argument(
        "--activity-heatmap-min-valid-ccres",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--figures",
        choices=("all", "joint", "panels"),
        default="all",
        help=(
            "'joint' stacks motif activity, TF expression and TF detection "
            "fraction in one figure; 'panels' writes them separately."
        ),
    )
    parser.add_argument(
        "--expression-scaling",
        choices=("zscore", "raw"),
        default="zscore",
        help="Expression panel of the joint figure: z-scored or raw.",
    )
    parser.add_argument(
        "--no-cluster-cell-types",
        dest="cluster_cell_types",
        action="store_false",
    )
    parser.set_defaults(cluster_cell_types=True)
    return parser.parse_args()


def load_atlas_layers(
    pseudobulk_path: Path,
) -> dict[str, pd.DataFrame]:
    """Return subclass by gene-symbol matrices for expression and detection.

    Duplicated gene symbols are resolved to the identifier with the highest
    mean expression, i.e. the dominant locus for that symbol.
    """
    atlas = ad.read_h5ad(pseudobulk_path)
    groups = pd.Index(atlas.obs["subclass_normalized"], name="group")
    symbols = pd.Index(atlas.var["gene_symbol"], name="gene_symbol")
    keep = slice(None)
    if symbols.duplicated().any():
        mean_expression = np.asarray(atlas.X, dtype=np.float32).mean(axis=0)
        highest_mean_first = np.argsort(-mean_expression)
        keep = highest_mean_first[
            ~symbols[highest_mean_first].duplicated(keep="first")
        ]
    layers: dict[str, pd.DataFrame] = {}
    for name, matrix in (
        ("expression", atlas.X),
        ("fraction_detected", atlas.layers["fraction_detected"]),
    ):
        values = np.asarray(
            matrix.toarray() if hasattr(matrix, "toarray") else matrix,
            dtype=np.float32,
        )[:, keep]
        layers[name] = pd.DataFrame(values, index=groups, columns=symbols[keep])
    return layers


def resolve_gene_symbols(
    motifs: pd.Index,
    annotation: pd.DataFrame,
    atlas_symbols: pd.Index,
) -> tuple[pd.Series, pd.Series]:
    """Map motifs to atlas gene symbols, falling back to HOCOMOCO synonyms.

    HOCOMOCO reports human-style ``Znf*`` symbols where the atlas uses the MGI
    ``Zfp*`` names, so an unmatched primary symbol is retried against the
    semicolon-separated synonym list.  Returns the resolved symbol per motif
    (NaN when absent) and the mappings that needed a synonym.
    """
    lookup = {symbol.casefold(): symbol for symbol in atlas_symbols}
    annotation_index = annotation.drop_duplicates("motif").set_index("motif")
    resolved: dict[str, str] = {}
    via_synonym: dict[str, str] = {}
    for motif in motifs:
        primary = annotation_index.at[motif, "mouse_gene_symbol"]
        if not isinstance(primary, str):
            continue
        match = lookup.get(primary.casefold())
        if match is None:
            synonyms = annotation_index.at[motif, "mouse_gene_synonyms"]
            candidates = (
                [part.strip() for part in synonyms.split(";") if part.strip()]
                if isinstance(synonyms, str)
                else []
            )
            for candidate in candidates:
                match = lookup.get(candidate.casefold())
                if match is not None:
                    via_synonym[motif] = f"{primary}→{match}"
                    break
        if match is not None:
            resolved[motif] = match
    return (
        pd.Series(resolved, dtype=object).reindex(motifs),
        pd.Series(via_synonym, dtype=object),
    )


def project_atlas_onto_motifs(
    atlas_layer: pd.DataFrame,
    genes: pd.Series,
    display: pd.DataFrame,
) -> pd.DataFrame:
    """Build a cell type by motif matrix of the motif TF's atlas values."""
    present = genes.notna().to_numpy()
    projected = pd.DataFrame(
        np.nan,
        index=display.index,
        columns=display.columns,
        dtype=float,
    )
    projected.loc[:, present] = atlas_layer.loc[
        display.index, genes[present].to_numpy()
    ].to_numpy()
    return projected


def symmetric_color_limit(values: np.ndarray, quantile: float = 0.98) -> float:
    """Symmetric color limit from the given quantile of |finite values|."""
    finite_abs = np.abs(values[np.isfinite(values)])
    if not finite_abs.size:
        return 1.0
    return max(float(np.quantile(finite_abs, quantile)), 1e-6)


def upper_color_limit(values: np.ndarray, quantile: float = 0.99) -> float:
    """Upper color limit from the given quantile of the finite values."""
    finite = values[np.isfinite(values)]
    return float(np.quantile(finite, quantile)) if finite.size else 1.0


def zscore_across_cell_types(matrix: pd.DataFrame) -> pd.DataFrame:
    """Z-score every column over cell types, leaving constant columns NaN."""
    return matrix.sub(matrix.mean(axis=0), axis=1).div(
        matrix.std(axis=0, ddof=0).replace(0.0, np.nan), axis=1
    )


def main() -> None:
    args = parse_args()
    tables_dir = args.results_dir / "tables"
    figures_dir = args.results_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    activity = pd.read_csv(
        tables_dir / "cell_type_by_motif_activity_zscore.csv.gz",
        index_col=0,
    )
    prefixes = pd.read_csv(
        tables_dir / "cell_type_number_prefixes.csv",
        usecols=["group", "numbered_prefix"],
    ).set_index("group")["numbered_prefix"]
    annotation = pd.read_csv(
        tables_dir / "hocomoco_motif_to_mouse_gene.csv.gz",
        usecols=["motif", "mouse_gene_symbol", "mouse_gene_synonyms"],
    )
    n_valid = (
        pd.read_csv(
            tables_dir / "motif_activity_by_cell_type_long.csv.gz",
            usecols=["group", "n_valid_ccres"],
        )
        .groupby("group", sort=True)["n_valid_ccres"]
        .max()
        .astype(int)
    )

    prepared = prepare_motif_display(
        activity,
        prefixes,
        n_valid.reindex(activity.index).fillna(0).astype(int),
        args.activity_heatmap_top_motifs,
        args.activity_heatmap_min_valid_ccres,
        selection="variability",
        cluster_cell_types=args.cluster_cell_types,
    )
    if prepared is None:
        raise SystemExit("No motifs left after selection; nothing to plot.")
    display = prepared.values

    atlas_layers = load_atlas_layers(args.pseudobulk)
    missing_groups = sorted(
        set(display.index) - set(atlas_layers["expression"].index)
    )
    if missing_groups:
        raise SystemExit(
            "Cell types absent from the atlas pseudobulk: "
            f"{', '.join(missing_groups)}"
        )

    genes, via_synonym = resolve_gene_symbols(
        display.columns, annotation, atlas_layers["expression"].columns
    )
    if not via_synonym.empty:
        print(
            f"{via_synonym.size} motif TF genes matched through a synonym: "
            f"{', '.join(sorted(set(via_synonym.to_numpy())))}"
        )
    unresolved = sorted(
        annotation.drop_duplicates("motif")
        .set_index("motif")
        .loc[genes[genes.isna()].index, "mouse_gene_symbol"]
        .dropna()
    )
    if unresolved:
        print(
            f"{len(unresolved)} of {genes.size} motif TF genes absent from the "
            f"atlas (shown grey): {', '.join(unresolved)}"
        )

    expression = project_atlas_onto_motifs(
        atlas_layers["expression"], genes, display
    )
    fraction = project_atlas_onto_motifs(
        atlas_layers["fraction_detected"], genes, display
    )
    expression_zscore = zscore_across_cell_types(expression)
    for name, matrix in (
        ("expression", expression),
        ("expression_zscore", expression_zscore),
        ("fraction_detected", fraction),
    ):
        matrix.to_csv(tables_dir / f"motif_heatmap_tf_{name}_by_cell_type.csv.gz")

    column_labels = motif_axis_labels(display.columns, annotation)
    row_labels = cell_type_axis_labels(display.index, prefixes)
    row_ordering = (
        "hierarchically clustered on motif activity"
        if prepared.cell_type_linkage is not None
        else "ordered by numbered prefix"
    )
    y_label = f"Cell type, {row_ordering}"
    x_label = (
        "Mouse TF | HOCOMOCO motif, ordered as in the motif activity heatmap"
    )

    activity_values = display.to_numpy(dtype=float)
    activity_limit = symmetric_color_limit(activity_values)
    activity_panel = HeatmapPanel(
        activity_values,
        "Within-cell-type z-scored motif activities",
        y_label,
        "Motif activity z-score within cell type",
        -activity_limit,
        activity_limit,
    )
    expression_panel = HeatmapPanel(
        expression.to_numpy(dtype=float),
        "Atlas TF expression level",
        y_label,
        EXPRESSION_LABEL,
        0.0,
        upper_color_limit(expression.to_numpy(dtype=float)),
        "magma",
    )
    zscore_values = expression_zscore.to_numpy(dtype=float)
    zscore_limit = symmetric_color_limit(zscore_values)
    expression_zscore_panel = HeatmapPanel(
        zscore_values,
        "Atlas TF expression, z-scored across cell types",
        y_label,
        f"{EXPRESSION_LABEL} z-score across cell types",
        -zscore_limit,
        zscore_limit,
    )
    fraction_panel = HeatmapPanel(
        fraction.to_numpy(dtype=float),
        "Fraction of cells expressing the TF",
        y_label,
        FRACTION_LABEL,
        0.0,
        1.0,
        "viridis",
    )

    written: list[str] = []
    if args.figures in ("panels", "all"):
        for stem, panel in (
            ("motif_tf_expression_heatmap", expression_panel),
            ("motif_tf_expression_zscore_heatmap", expression_zscore_panel),
            ("motif_tf_fraction_detected_heatmap", fraction_panel),
        ):
            draw_clustered_heatmap(
                panel.values,
                row_labels,
                column_labels,
                figures_dir,
                stem,
                f"{panel.title} for the motifs in the activity heatmap",
                x_label,
                panel.y_label,
                panel.colorbar_label,
                panel.vmin,
                panel.vmax,
                column_linkage=prepared.motif_linkage,
                row_linkage=prepared.cell_type_linkage,
                color_map_name=panel.color_map_name,
            )
            written.append(stem)

    if args.figures in ("joint", "all"):
        stem = "motif_activity_with_tf_expression_heatmap"
        draw_stacked_clustered_heatmaps(
            [
                activity_panel,
                (
                    expression_zscore_panel
                    if args.expression_scaling == "zscore"
                    else expression_panel
                ),
                fraction_panel,
            ],
            row_labels,
            column_labels,
            figures_dir,
            stem,
            x_label,
            column_linkage=prepared.motif_linkage,
            row_linkage=prepared.cell_type_linkage,
        )
        written.append(stem)
    print("wrote " + ", ".join(written))


if __name__ == "__main__":
    main()
