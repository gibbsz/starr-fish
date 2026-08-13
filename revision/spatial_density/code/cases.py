#!/usr/bin/env python3
"""Inputs and admissibility for the spatial density workflow.

Which cells a cCRE may be examined on is a substantive decision, not a plotting
detail: a (subclass, cCRE) pair carrying almost no T7 signal has no estimable
activity, so its cells contribute a near-constant, prior-determined value. Test
and figure must therefore agree exactly on the cell set -- otherwise a picture
shows cells the statistic never saw. That is why this lives in one module used
by both ``run_activity_density_hotspots.py`` and
``plot_activity_density_hotspots.py`` rather than being written twice.

``activity_density.py`` stays free of I/O; this module is the layer that reads
files and turns them into the geometry that module consumes.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, replace
from typing import Sequence

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WORKFLOW_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
REPO_ROOT = os.path.abspath(os.path.join(WORKFLOW_DIR, "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from activity_density import (  # noqa: E402
    BandwidthGeometry,
    SectionGrid,
    build_geometry,
    build_grid,
)

__all__ = [
    "CaseGeometry",
    "SectionContext",
    "build_admissible",
    "build_case",
    "build_section_contexts",
    "load_activity",
    "negative_control_cres",
    "selected_cres",
]

DEFAULT_H5AD = os.path.join(
    REPO_ROOT,
    "revision",
    "Data",
    "scdata_5_28_2025_BRBB500gn_final_CRE_T7CRE_NEWNEW.h5ad",
)
DEFAULT_ACTIVITY_NPZ = os.path.join(
    REPO_ROOT, "revision", "Bayes_OldData", "copy_number", "activity_normalized.npz"
)
DEFAULT_MANIFEST = os.path.join(
    REPO_ROOT, "revision", "Bayes_OldData", "bayesian", "run_manifest.json"
)
DEFAULT_T7_TOTALS = os.path.join(
    WORKFLOW_DIR, "results", "subclass_cre_t7_totals.csv.gz"
)


def log(message: str) -> None:
    print(message, flush=True)


# --------------------------------------------------------------------------- #
# inputs
# --------------------------------------------------------------------------- #


def negative_control_cres(manifest_path: str) -> set[str]:
    """The annotated controls that define the 1.0 baseline.

    They must not produce regions; a run where they do is mis-calibrated, not
    interesting.
    """
    with open(manifest_path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    controls = manifest.get("config", {}).get("annotated_negative_control_cre", [])
    return {str(name) for name in controls}


def selected_cres(
    available: Sequence[str], requested: Sequence[str] | None, limit: int | None
) -> list[str]:
    """The cCRE columns to work on, in a stable order."""
    present = list(dict.fromkeys(str(name) for name in available))
    if requested:
        unknown = sorted(set(requested) - set(present))
        if unknown:
            raise ValueError(f"cCRE(s) absent from the matrix: {unknown}")
        chosen = [name for name in present if name in set(requested)]
    else:
        chosen = present
    if limit is not None:
        chosen = chosen[:limit]
    if not chosen:
        raise ValueError("no cCREs selected")
    return chosen


def load_activity(
    npz_path: str, matrix_key: str, cres: Sequence[str] | None, limit: int | None
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """``(values, obs_names, cre_names)`` for the requested columns only."""
    with np.load(npz_path, allow_pickle=True) as store:
        if matrix_key not in store:
            raise KeyError(
                f"{npz_path} has no {matrix_key!r}; have {sorted(store.files)}"
            )
        npz_cres = [str(name) for name in store["cre_names"]]
        chosen = selected_cres(npz_cres, cres, limit)
        columns = pd.Index(npz_cres).get_indexer(chosen)
        if (columns < 0).any():
            raise ValueError("cCRE lookup failed after validation")
        values = np.asarray(store[matrix_key][:, columns], dtype=np.float32)
        obs_names = np.asarray([str(name) for name in store["obs_names"]], dtype=object)
    return values, obs_names, chosen


def build_admissible(
    path: str, cre_names: Sequence[str], vocabulary: np.ndarray, min_t7: float
) -> np.ndarray | None:
    """``(n_cre, n_subclass)`` mask of which subclasses a cCRE may be examined on.

    A (subclass, cCRE) pair below the T7 threshold is not a weak measurement, it
    is very nearly no measurement: the per-cell activity posterior falls back on
    the prior and returns a near-constant subclass-level value. Including those
    cells dilutes the surface and, under the within-subclass null, hands the
    permutation a block that cannot randomise.

    ``None`` when the filter is off, which keeps the unfiltered path free of a
    pointless all-true lookup.
    """
    if min_t7 <= 0:
        log("[t7] filter disabled (--min-t7 0): every subclass used for every cCRE")
        return None
    totals = pd.read_csv(path)
    missing = {"subclass", "cre", "t7_total"} - set(totals.columns)
    if missing:
        raise ValueError(f"{path} is missing columns {sorted(missing)}")
    passing = totals.loc[totals["t7_total"] >= min_t7, ["cre", "subclass"]]
    lookup = pd.Index(vocabulary)
    mask = np.zeros((len(cre_names), vocabulary.size), dtype=bool)
    positions = {name: index for index, name in enumerate(cre_names)}
    rows = passing["cre"].map(positions)
    columns = lookup.get_indexer(passing["subclass"].to_numpy())
    keep = rows.notna().to_numpy() & (columns >= 0)
    mask[rows[keep].to_numpy(dtype=int), columns[keep]] = True
    per_cre = mask.sum(axis=1)
    log(
        f"[t7] T7 >= {min_t7}: {int(mask.sum())} admissible (cCRE, subclass) pairs; "
        f"subclasses per cCRE median {int(np.median(per_cre))}, "
        f"{int((per_cre == 0).sum())} cCREs have none"
    )
    return mask


# --------------------------------------------------------------------------- #
# geometry
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class SectionContext:
    """One section's fixed frame and cell ordering.

    Cells are held sorted by subclass so the ``within_subclass`` null is a
    shuffle inside contiguous blocks rather than a scatter/gather.
    """

    section: str
    grid: SectionGrid  # the full-section frame: origin and shape every cCRE shares
    rows: np.ndarray  # positions into the activity matrix, subclass-sorted
    obs_names: np.ndarray
    subclass: np.ndarray
    subclass_codes: np.ndarray  # into the run-wide subclass vocabulary

    @property
    def n_cells(self) -> int:
        return int(self.rows.size)


@dataclass(frozen=True)
class CaseGeometry:
    """One cCRE's geometry within a section.

    The T7 admissibility filter is a per-(subclass, cCRE) decision, so the set of
    cells differs from cCRE to cCRE and the kernel geometry has to be rebuilt for
    each. The grid *frame* -- origin, shape, pixel size -- is inherited from the
    section unchanged, so surfaces stay pixel-comparable across cCREs and the
    figures overlay.
    """

    grid: SectionGrid
    geometries: tuple[BandwidthGeometry, ...]
    domain: np.ndarray
    block_starts: np.ndarray
    cells: np.ndarray  # positions within the section's cell ordering

    @property
    def n_cells(self) -> int:
        return int(self.cells.size)


def build_section_contexts(
    metadata: pd.DataFrame,
    keep: np.ndarray,
    sections: Sequence[str] | None,
    pixel_size: float,
    vocabulary: np.ndarray,
) -> dict[str, SectionContext]:
    """One fixed grid frame per section, shared by every cCRE."""
    contexts: dict[str, SectionContext] = {}
    wanted = set(sections) if sections else None
    for section, frame in metadata[keep].groupby("section", sort=True):
        if wanted is not None and section not in wanted:
            continue
        ordered = frame.sort_values(["subclass", "obs_name"], kind="stable")
        grid = build_grid(
            ordered["x"].to_numpy(),
            ordered["y"].to_numpy(),
            section=str(section),
            pixel_size=pixel_size,
        )
        subclass = ordered["subclass"].to_numpy()
        contexts[str(section)] = SectionContext(
            section=str(section),
            grid=grid,
            rows=ordered.index.to_numpy(),
            obs_names=ordered["obs_name"].to_numpy(),
            subclass=subclass,
            subclass_codes=pd.Index(vocabulary).get_indexer(subclass),
        )
        log(f"[grid] {section}: {grid.n_cells} cells, grid {grid.shape}")
    if not contexts:
        raise ValueError("no sections selected")
    return contexts


def build_case(
    context: SectionContext,
    cre_index: int,
    *,
    values: np.ndarray,
    admissible: np.ndarray | None,
    bandwidths: Sequence[float],
    min_effective_cells: float,
    max_cell_distance_factor: float,
    min_cells: int,
) -> CaseGeometry | None:
    """Geometry over the cells this cCRE may be examined on.

    A cell qualifies when its activity is finite *and* its subclass carries
    enough total T7 signal for this particular cCRE. Returns ``None`` when too
    little is left, which callers record rather than silently dropping.
    """
    keep = np.isfinite(values[context.rows, cre_index])
    if admissible is not None:
        keep &= admissible[cre_index][context.subclass_codes]
    if int(keep.sum()) < min_cells:
        return None

    cells = np.flatnonzero(keep)
    grid = replace(context.grid, pixel_index=context.grid.pixel_index[cells])
    try:
        geometries = tuple(
            build_geometry(
                grid,
                bandwidth=bandwidth,
                min_effective_cells=min_effective_cells,
                max_cell_distance=bandwidth * max_cell_distance_factor,
            )
            for bandwidth in bandwidths
        )
    except ValueError:
        # Too sparse for any pixel to survive the tissue and n_eff masks.
        return None
    domain = np.zeros(grid.shape, dtype=bool)
    for geometry in geometries:
        domain |= geometry.valid

    codes = context.subclass_codes[cells]
    changes = np.flatnonzero(codes[1:] != codes[:-1]) + 1
    return CaseGeometry(
        grid=grid,
        geometries=geometries,
        domain=domain,
        block_starts=np.concatenate(([0], changes, [codes.size])),
        cells=cells,
    )
