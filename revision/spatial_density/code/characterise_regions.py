#!/usr/bin/env python3
"""Say what each delineated region *is*, and which cCREs share it.

``run_activity_density_hotspots.py`` returns regions as polygons and cell lists.
This turns them into something interpretable. The h5ad carries no CCF or atlas
annotation -- ``obs`` has only ``class``/``subclass``/``supertype``/``cluster_name``
-- so a region can only be named by what is inside it:

* **subclass enrichment** -- cells of each subclass inside the region against the
  rest of the same section, as a one-sided hypergeometric tail (the exact Fisher
  p, computed vectorised) with BH across subclasses.
* **marker genes** -- the 500 panel genes in ``obsm['X_raw']``, inside against
  outside, by Mann-Whitney U. What the region is anatomically usually falls out
  of the top few.
* **cross-cCRE sharing** -- Jaccard overlap between every pair of region masks in
  a section, then average-linkage clustering. This is the payoff: it collapses
  hundreds of separate maps into the handful of recurrent domains that actually
  exist, and names the cCREs sharing each.

Comparisons are always *within a section*: the two sections are separate
coordinate frames and separate pieces of tissue.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Sequence

import numpy as np
import pandas as pd

for _var in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_var, "4")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WORKFLOW_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
REPO_ROOT = os.path.abspath(os.path.join(WORKFLOW_DIR, "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from baystarrfish.data import read_obs_metadata  # noqa: E402
from baystarrfish.stats.fdr import bh_fdr  # noqa: E402

DEFAULT_H5AD = os.path.join(
    REPO_ROOT,
    "revision",
    "Data",
    "scdata_5_28_2025_BRBB500gn_final_CRE_T7CRE_NEWNEW.h5ad",
)
DEFAULT_RESULTS = os.path.join(WORKFLOW_DIR, "results")
DEFAULT_TOP_MARKERS = 15
DEFAULT_MIN_JACCARD = 0.1


def log(message: str) -> None:
    print(message, flush=True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n")[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--h5ad", default=DEFAULT_H5AD)
    parser.add_argument("--results", default=DEFAULT_RESULTS)
    parser.add_argument("--top-markers", type=int, default=DEFAULT_TOP_MARKERS)
    parser.add_argument("--min-jaccard", type=float, default=DEFAULT_MIN_JACCARD)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument(
        "--skip-markers",
        action="store_true",
        help="skip the gene pass, which is the only step that reads the 500-gene "
        "matrix out of the h5ad",
    )
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    if not os.path.exists(args.h5ad):
        raise FileNotFoundError(f"--h5ad not found: {args.h5ad}")
    for name in ("activity_density_regions.csv", "region_cell_membership.csv.gz"):
        path = os.path.join(args.results, name)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{path} missing; run run_activity_density_hotspots.py first"
            )
    if not 0 < args.alpha < 1:
        raise ValueError("--alpha must lie in (0, 1)")
    if not 0 <= args.min_jaccard <= 1:
        raise ValueError("--min-jaccard must lie in [0, 1]")
    if args.top_markers < 1:
        raise ValueError("--top-markers must be at least 1")


# --------------------------------------------------------------------------- #
# subclass composition
# --------------------------------------------------------------------------- #


def subclass_enrichment(
    membership: pd.DataFrame, metadata: pd.DataFrame, regions: pd.DataFrame
) -> pd.DataFrame:
    """One-sided hypergeometric enrichment of each subclass inside each region.

    The hypergeometric survival function is the exact one-sided Fisher p-value
    and is vectorised, so all (region x subclass) cells are tested in one call
    rather than in a Python loop over ``scipy.stats.fisher_exact``.
    """
    from scipy.stats import hypergeom

    section_of = dict(zip(regions["region_id"], regions["section"]))
    subclass_of = dict(zip(metadata["obs_name"], metadata["subclass"]))
    background = (
        metadata.groupby(["section", "subclass"]).size().rename("n_background")
    )
    section_totals = metadata.groupby("section").size()

    rows: list[dict[str, object]] = []
    for region_id, frame in membership.groupby("region_id"):
        section = section_of.get(region_id)
        if section is None:
            continue
        inside = frame["obs_name"].map(subclass_of).dropna()
        counts = inside.value_counts()
        n_inside = int(counts.sum())
        pool = int(section_totals[section])
        for subclass, drawn in counts.items():
            in_pool = int(background.loc[(section, subclass)])
            # sf(k-1) is P(X >= k): the probability of seeing at least this many
            # cells of the subclass when drawing n_inside cells without
            # replacement from the section.
            p_value = float(hypergeom.sf(drawn - 1, pool, in_pool, n_inside))
            expected = n_inside * in_pool / pool
            rows.append(
                {
                    "region_id": region_id,
                    "section": section,
                    "subclass": subclass,
                    "n_inside": int(drawn),
                    "n_section": in_pool,
                    "n_region_cells": n_inside,
                    "expected_inside": expected,
                    "fold_enrichment": float(drawn) / expected if expected else np.nan,
                    "p_hypergeometric": p_value,
                }
            )
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame["q_bh"] = bh_fdr(frame["p_hypergeometric"].to_numpy())
        frame = frame.sort_values(
            ["region_id", "p_hypergeometric"], kind="stable"
        ).reset_index(drop=True)
    return frame


# --------------------------------------------------------------------------- #
# marker genes
# --------------------------------------------------------------------------- #


def marker_genes(
    membership: pd.DataFrame,
    metadata: pd.DataFrame,
    regions: pd.DataFrame,
    h5ad_path: str,
    top: int,
) -> pd.DataFrame:
    """Panel genes separating each region from the rest of its section."""
    import h5py
    from scipy.stats import mannwhitneyu

    with h5py.File(h5ad_path, "r") as handle:
        genes = np.asarray(
            [
                name.decode() if isinstance(name, bytes) else str(name)
                for name in handle["var"]["_index"][:]
            ]
        )
        expression = np.asarray(handle["obsm"]["X_raw"][:], dtype=np.float32)

    position = pd.Index(metadata["obs_name"]).get_indexer
    section_of = dict(zip(regions["region_id"], regions["section"]))
    section_rows = {
        section: np.flatnonzero((metadata["section"] == section).to_numpy())
        for section in metadata["section"].unique()
    }

    rows: list[dict[str, object]] = []
    for region_id, frame in membership.groupby("region_id"):
        section = section_of.get(region_id)
        if section is None:
            continue
        inside = position(frame["obs_name"].to_numpy())
        inside = inside[inside >= 0]
        if inside.size < 2:
            continue
        pool = section_rows[section]
        outside = np.setdiff1d(pool, inside, assume_unique=False)
        if outside.size < 2:
            continue
        result = mannwhitneyu(
            expression[inside], expression[outside], axis=0, alternative="two-sided"
        )
        mean_in = expression[inside].mean(axis=0)
        mean_out = expression[outside].mean(axis=0)
        order = np.argsort(result.pvalue)[:top]
        for index in order:
            rows.append(
                {
                    "region_id": region_id,
                    "section": section,
                    "gene": genes[index],
                    "mean_inside": float(mean_in[index]),
                    "mean_outside": float(mean_out[index]),
                    "log2_fold_change": float(
                        np.log2((mean_in[index] + 1e-9) / (mean_out[index] + 1e-9))
                    ),
                    "p_mannwhitney": float(result.pvalue[index]),
                }
            )
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame["q_bh"] = bh_fdr(frame["p_mannwhitney"].to_numpy())
    return frame


# --------------------------------------------------------------------------- #
# cross-cCRE sharing
# --------------------------------------------------------------------------- #


def jaccard_matrix(
    membership: pd.DataFrame, regions: pd.DataFrame, min_jaccard: float
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pairwise cell-level Jaccard between cCREs' regions, plus a clustering.

    Overlap is measured on cell sets rather than on pixel masks: cells are the
    unit the regions were built from, and two regions found at different
    bandwidths can cover the same cells without covering the same pixels.

    The intersections come from one dense matrix product -- the pairwise loop
    over hundreds of cCREs would otherwise dominate the whole script.
    """
    section_of = dict(zip(regions["region_id"], regions["section"]))
    membership = membership.assign(
        section=membership["region_id"].map(section_of)
    ).dropna(subset=["section"])

    pair_rows: list[dict[str, object]] = []
    cluster_rows: list[dict[str, object]] = []
    for section, frame in membership.groupby("section"):
        cres = sorted(frame["cre"].unique())
        if len(cres) < 2:
            continue
        cells = pd.Index(sorted(frame["obs_name"].unique()))
        indicator = np.zeros((len(cres), len(cells)), dtype=np.float32)
        for row, cre in enumerate(cres):
            hit = cells.get_indexer(frame.loc[frame["cre"] == cre, "obs_name"].unique())
            indicator[row, hit[hit >= 0]] = 1.0
        sizes = indicator.sum(axis=1)
        intersection = indicator @ indicator.T
        union = sizes[:, None] + sizes[None, :] - intersection
        with np.errstate(divide="ignore", invalid="ignore"):
            jaccard = np.where(union > 0, intersection / union, 0.0)

        upper = np.triu_indices(len(cres), k=1)
        for i, j in zip(*upper):
            if jaccard[i, j] >= min_jaccard:
                pair_rows.append(
                    {
                        "section": section,
                        "cre_a": cres[i],
                        "cre_b": cres[j],
                        "jaccard": float(jaccard[i, j]),
                        "n_shared_cells": int(intersection[i, j]),
                    }
                )
        cluster_rows.extend(
            _cluster(section, cres, jaccard, min_jaccard)
        )
    return pd.DataFrame(pair_rows), pd.DataFrame(cluster_rows)


def _cluster(
    section: str, cres: list[str], jaccard: np.ndarray, min_jaccard: float
) -> list[dict[str, object]]:
    """Average-linkage clustering of cCREs by how much their regions overlap."""
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform

    distance = np.clip(1.0 - jaccard, 0.0, 1.0)
    np.fill_diagonal(distance, 0.0)
    distance = (distance + distance.T) / 2.0
    tree = linkage(squareform(distance, checks=False), method="average")
    labels = fcluster(tree, t=1.0 - min_jaccard, criterion="distance")
    return [
        {"section": section, "cre": cre, "region_cluster": int(label)}
        for cre, label in zip(cres, labels)
    ]


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    validate_args(args)

    regions = pd.read_csv(os.path.join(args.results, "activity_density_regions.csv"))
    membership = pd.read_csv(
        os.path.join(args.results, "region_cell_membership.csv.gz")
    )
    if regions.empty or membership.empty:
        log("[skip] no regions to characterise")
        return 0
    metadata = read_obs_metadata(args.h5ad)
    log(f"[input] {len(regions)} regions, {len(membership)} region-cell assignments")

    composition = subclass_enrichment(membership, metadata, regions)
    composition.to_csv(
        os.path.join(args.results, "region_subclass_enrichment.csv"), index=False
    )
    significant = (
        int((composition["q_bh"] <= args.alpha).sum()) if not composition.empty else 0
    )
    log(f"[subclass] {significant} enriched (region, subclass) pairs at q <= {args.alpha}")

    if args.skip_markers:
        log("[markers] skipped by --skip-markers")
    else:
        markers = marker_genes(
            membership, metadata, regions, args.h5ad, args.top_markers
        )
        markers.to_csv(
            os.path.join(args.results, "region_marker_genes.csv"), index=False
        )
        log(f"[markers] {len(markers)} region-gene rows")

    pairs, clusters = jaccard_matrix(membership, regions, args.min_jaccard)
    pairs.to_csv(os.path.join(args.results, "region_jaccard.csv"), index=False)
    clusters.to_csv(os.path.join(args.results, "region_clusters.csv"), index=False)
    n_clusters = clusters["region_cluster"].nunique() if not clusters.empty else 0
    log(
        f"[sharing] {len(pairs)} cCRE pairs above Jaccard {args.min_jaccard}; "
        f"{n_clusters} region clusters"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
