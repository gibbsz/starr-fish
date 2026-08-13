#!/usr/bin/env python3
"""Find, delineate and export the spatial regions where a cCRE is more active
than random reassignment of the same activities would produce.

The statistical unit is one (cCRE, section) pair. Per-cell Gamma-conjugate
activity posterior means from ``revision/Bayes_OldData`` are smoothed into a
local-mean-activity surface, standardised against the exact random-labelling
moments, and compared with the permutation distribution of the surface maximum.
See ``activity_density.py`` for the statistics; this module is I/O, argument
handling and parallelism only.

Two passes:

* **screen** -- every (cCRE, section): the joint-over-bandwidths maximum of the
  standardised surface against ``--permutations`` random relabellings, giving a
  family-wise ``p_fwer``. Benjamini-Hochberg across cCREs gives ``q_global_bh``.
* **regions** -- only the pairs that pass: the permuted surfaces are retained so
  the step-down can re-maximise over a shrinking domain, yielding disjoint
  simultaneously-significant regions, then a cell-level bootstrap puts a band on
  each boundary.

Outputs land under ``--outdir`` (default ``../results``):

    activity_density_summary.csv          one row per (cre, section)
    activity_density_scales.csv           one row per (cre, section, bandwidth)
    activity_density_regions.csv          one row per delineated region
    region_cell_membership.csv.gz         obs_name -> region, for downstream use
    regions/{cre}_{section}.geojson       region polygons in tissue coordinates
    surfaces/{cre}_{section}.npz          R, z, cell density, inclusion prob.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
from dataclasses import dataclass, replace
from multiprocessing import get_context
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy import ndimage

# BLAS threads fight the process pool for cores; every array here is small
# enough that threaded BLAS buys nothing.
for _var in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_var, "1")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WORKFLOW_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
REPO_ROOT = os.path.abspath(os.path.join(WORKFLOW_DIR, "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from activity_density import (  # noqa: E402
    MARK_TRANSFORMS,
    Region,
    SectionGrid,
    build_geometry,
    density_surface,
    extract_regions,
    mark_shuffler,
    null_maxima,
    null_z_stack,
    permutation_p_value,
    transform_marks,
    z_stack,
)
from cases import (  # noqa: E402
    CaseGeometry,
    SectionContext,
    build_admissible,
    build_case,
    build_section_contexts,
    load_activity,
    negative_control_cres,
)
from baystarrfish.data import read_obs_metadata  # noqa: E402
from baystarrfish.stats.fdr import bh_fdr  # noqa: E402

# ---- defaults ------------------------------------------------------------- #

DEFAULT_H5AD = os.path.join(
    REPO_ROOT,
    "revision",
    "Data",
    "scdata_5_28_2025_BRBB500gn_final_CRE_T7CRE_NEWNEW.h5ad",
)
DEFAULT_ACTIVITY_NPZ = os.path.join(
    REPO_ROOT,
    "revision",
    "Bayes_OldData",
    "copy_number",
    "activity_normalized.npz",
)
DEFAULT_MANIFEST = os.path.join(
    REPO_ROOT, "revision", "Bayes_OldData", "bayesian", "run_manifest.json"
)
DEFAULT_OUTDIR = os.path.join(WORKFLOW_DIR, "results")

MATRIX_KEYS = ("activity_normalized", "activity")
NULL_MODES = ("global", "within_subclass")
DEFAULT_BANDWIDTHS = (100.0, 200.0, 400.0)
DEFAULT_MIN_EFFECTIVE_CELLS = 25.0
DEFAULT_MIN_REGION_CELLS = 50
DEFAULT_MIN_T7 = 0.0
DEFAULT_MIN_CASE_CELLS = 500
DEFAULT_T7_TOTALS = os.path.join(WORKFLOW_DIR, "results", "subclass_cre_t7_totals.csv.gz")
# A pixel further than this many bandwidths from the nearest cell is outside the
# tissue; see build_geometry for why n_eff alone cannot express that.
TISSUE_MASK_BANDWIDTHS = 1.0
DEFAULT_PERMUTATIONS = 999
DEFAULT_BOOTSTRAP = 200
DEFAULT_ALPHA = 0.05
DEFAULT_MAX_REGIONS = 8
DEFAULT_SEED = 20260812
# Pixels only have to resolve the narrowest Gaussian; one grid then serves every
# bandwidth, so the cell -> pixel map is built once per section.
PIXELS_PER_BANDWIDTH = 2.0
INCLUSION_LEVELS = (0.05, 0.5, 0.95)


def log(message: str) -> None:
    print(message, flush=True)


# --------------------------------------------------------------------------- #
# arguments
# --------------------------------------------------------------------------- #


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n")[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--h5ad", default=DEFAULT_H5AD)
    parser.add_argument("--activity-npz", default=DEFAULT_ACTIVITY_NPZ)
    parser.add_argument(
        "--manifest",
        default=DEFAULT_MANIFEST,
        help="fit manifest, read only for the annotated negative controls",
    )
    parser.add_argument("--outdir", default=DEFAULT_OUTDIR)
    parser.add_argument(
        "--matrix",
        default=MATRIX_KEYS[0],
        choices=MATRIX_KEYS,
        help=(
            "activity_normalized is 1.0 at the negative-control level but is NaN "
            "for the 284 subclasses without a control reference; activity keeps "
            "every cell on an arbitrary scale"
        ),
    )
    parser.add_argument("--cres", nargs="*", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sections", nargs="*", default=None)
    parser.add_argument(
        "--bandwidths", nargs="+", type=float, default=list(DEFAULT_BANDWIDTHS)
    )
    parser.add_argument("--pixel-size", type=float, default=None)
    parser.add_argument(
        "--min-effective-cells", type=float, default=DEFAULT_MIN_EFFECTIVE_CELLS
    )
    parser.add_argument(
        "--min-region-cells",
        type=int,
        default=DEFAULT_MIN_REGION_CELLS,
        help="components holding fewer cells than this are dropped from the "
        "report; the count of drops is logged, never hidden",
    )
    parser.add_argument(
        "--mark-transform",
        default=MARK_TRANSFORMS[0],
        choices=MARK_TRANSFORMS,
        help="scale the local mean is taken on; rank (normal scores) makes the "
        "test insensitive to the activity's extreme right tail, which otherwise "
        "inflates the null maximum and costs power. Reported effect sizes stay "
        "on the raw activity scale regardless",
    )
    parser.add_argument(
        "--min-t7",
        type=float,
        default=DEFAULT_MIN_T7,
        help="drop every cell of a subclass from a cCRE's test when that "
        "(subclass, cCRE) pair carries less than this much total T7 signal. "
        "0 disables the filter. 50 is the threshold the upstream fit uses",
    )
    parser.add_argument(
        "--t7-totals",
        default=DEFAULT_T7_TOTALS,
        help="per-(subclass, cCRE) T7 totals from make_t7_totals.py; only read "
        "when --min-t7 is positive",
    )
    parser.add_argument(
        "--min-case-cells",
        type=int,
        default=DEFAULT_MIN_CASE_CELLS,
        help="a (cCRE, section) with fewer admissible cells than this is not "
        "tested; skips are counted and logged, never hidden",
    )
    parser.add_argument("--permutations", type=int, default=DEFAULT_PERMUTATIONS)
    parser.add_argument("--bootstrap", type=int, default=DEFAULT_BOOTSTRAP)
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    parser.add_argument("--max-regions", type=int, default=DEFAULT_MAX_REGIONS)
    parser.add_argument("--null", default=NULL_MODES[0], choices=NULL_MODES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument(
        "--region-jobs",
        type=int,
        default=None,
        help="workers for the region pass, which holds every permuted surface "
        "in memory; defaults to --jobs",
    )
    parser.add_argument(
        "--permute-observed",
        action="store_true",
        help="shuffle the marks once up front: a calibration run, where p_fwer "
        "must come out uniform and no region may be emitted",
    )
    parser.add_argument("--skip-regions", action="store_true")
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    for name in ("h5ad", "activity_npz", "manifest"):
        path = getattr(args, name)
        if not os.path.exists(path):
            raise FileNotFoundError(f"--{name.replace('_', '-')} not found: {path}")
    if args.permutations < 19:
        raise ValueError("--permutations below 19 cannot reach alpha = 0.05")
    if not 0 < args.alpha < 1:
        raise ValueError("--alpha must lie in (0, 1)")
    if any(bandwidth <= 0 for bandwidth in args.bandwidths):
        raise ValueError("--bandwidths must all be positive")
    if len(set(args.bandwidths)) != len(args.bandwidths):
        raise ValueError("--bandwidths contains duplicates")
    if args.pixel_size is not None and args.pixel_size <= 0:
        raise ValueError("--pixel-size must be positive")
    if args.jobs < 1 or (args.region_jobs is not None and args.region_jobs < 1):
        raise ValueError("job counts must be at least 1")
    if args.bootstrap < 0:
        raise ValueError("--bootstrap must be non-negative")
    if args.max_regions < 1:
        raise ValueError("--max-regions must be at least 1")
    if args.min_region_cells < 0:
        raise ValueError("--min-region-cells must be non-negative")
    if args.min_t7 < 0:
        raise ValueError("--min-t7 must be non-negative")
    if args.min_case_cells < 2:
        raise ValueError("--min-case-cells must be at least 2")
    if args.min_t7 > 0 and not os.path.exists(args.t7_totals):
        raise FileNotFoundError(
            f"--t7-totals not found: {args.t7_totals}; run make_t7_totals.py first"
        )
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be at least 1")


# --------------------------------------------------------------------------- #
# inputs
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# per-section setup
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# worker state
# --------------------------------------------------------------------------- #

_SHARED: dict[str, object] = {}


def _case(context: SectionContext, cre_index: int) -> CaseGeometry | None:
    """Worker-side adapter over :func:`cases.build_case`.

    The figures call the same builder with the same arguments, so a plot can
    never show cells the test did not use.
    """
    return build_case(
        context,
        cre_index,
        values=_SHARED["values"],  # type: ignore[arg-type]
        admissible=_SHARED["admissible"],  # type: ignore[arg-type]
        bandwidths=_SHARED["bandwidths"],  # type: ignore[arg-type]
        min_effective_cells=float(_SHARED["min_effective_cells"]),  # type: ignore[arg-type]
        max_cell_distance_factor=float(_SHARED["max_cell_distance_factor"]),  # type: ignore[arg-type]
        min_cells=int(_SHARED["min_case_cells"]),  # type: ignore[arg-type]
    )


def _raw_marks(
    context: SectionContext, case: CaseGeometry, cre_index: int
) -> np.ndarray:
    """The activity as estimated, on the scale the effect sizes are reported on."""
    values: np.ndarray = _SHARED["values"]  # type: ignore[assignment]
    return np.ascontiguousarray(
        values[context.rows[case.cells], cre_index], dtype=np.float32
    )


def _marks(context: SectionContext, case: CaseGeometry, cre_index: int) -> np.ndarray:
    """The activity on the scale the statistic is computed on."""
    return transform_marks(
        _raw_marks(context, case, cre_index),
        str(_SHARED["mark_transform"]),
    )


def _shuffler(case: CaseGeometry):
    null_mode: str = _SHARED["null"]  # type: ignore[assignment]
    starts = case.block_starts if null_mode == "within_subclass" else None
    return mark_shuffler(starts)


def _rng(cre_index: int, section_index: int, stream: int) -> np.random.Generator:
    seed: int = _SHARED["seed"]  # type: ignore[assignment]
    return np.random.default_rng(
        np.random.SeedSequence([seed, cre_index, section_index, stream])
    )


# --------------------------------------------------------------------------- #
# pass 1: screening
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ScreenRecord:
    cre: str
    section: str
    cre_index: int
    section_index: int
    n_cells: int
    mean_activity: float
    sd_activity: float
    max_z: float
    p_fwer: float
    selected_bandwidth: float
    max_z_per_bandwidth: tuple[float, ...]
    n_valid_per_bandwidth: tuple[int, ...]


def _screen(task: tuple[int, int]) -> ScreenRecord | None:
    cre_index, section_index = task
    contexts: list[SectionContext] = _SHARED["contexts"]  # type: ignore[assignment]
    cre_names: list[str] = _SHARED["cre_names"]  # type: ignore[assignment]
    context = contexts[section_index]
    case = _case(context, cre_index)
    if case is None:
        return None
    marks = _marks(context, case, cre_index)
    raw = _raw_marks(context, case, cre_index)
    geometries = list(case.geometries)

    observed = z_stack(
        case.grid,
        geometries,
        marks,
        mean=float(np.mean(marks)),
        variance=float(np.var(marks)),
    )
    maxima = null_maxima(
        case.grid,
        geometries,
        marks,
        n_permutations=int(_SHARED["permutations"]),  # type: ignore[arg-type]
        rng=_rng(cre_index, section_index, 0),
        shuffle=_shuffler(case),
    )
    per_bandwidth = tuple(
        float(observed[index][geometry.valid].max())
        for index, geometry in enumerate(geometries)
    )
    best = int(np.argmax(per_bandwidth))
    return ScreenRecord(
        cre=cre_names[cre_index],
        section=context.section,
        cre_index=cre_index,
        section_index=section_index,
        n_cells=case.n_cells,
        mean_activity=float(np.mean(raw)),
        sd_activity=float(np.std(raw)),
        max_z=float(per_bandwidth[best]),
        p_fwer=permutation_p_value(float(per_bandwidth[best]), maxima),
        selected_bandwidth=geometries[best].bandwidth,
        max_z_per_bandwidth=per_bandwidth,
        n_valid_per_bandwidth=tuple(g.n_valid for g in geometries),
    )


# --------------------------------------------------------------------------- #
# pass 2: region delineation
# --------------------------------------------------------------------------- #


def _bootstrap_inclusion(
    case: CaseGeometry,
    marks: np.ndarray,
    threshold: float,
    *,
    n_draws: int,
    rng: np.random.Generator,
    min_effective_cells: float,
    max_cell_distance_factor: float,
) -> np.ndarray:
    """Per-pixel probability of landing in the excursion set, over cell resampling.

    Resampling cells with replacement perturbs the kernel weights as well as the
    marks, so the geometry is rebuilt per draw -- that is the whole point: the
    boundary moves because the cells that define it are themselves a sample.
    """
    inclusion = np.zeros(case.grid.shape, dtype=np.float32)
    if n_draws <= 0:
        return inclusion
    n_cells = case.n_cells
    kept = 0
    for _ in range(n_draws):
        picks = rng.integers(0, n_cells, n_cells)
        grid = replace(case.grid, pixel_index=case.grid.pixel_index[picks])
        drawn = marks[picks]
        try:
            geometries = [
                build_geometry(
                    grid,
                    bandwidth=geometry.bandwidth,
                    min_effective_cells=min_effective_cells,
                    max_cell_distance=geometry.bandwidth * max_cell_distance_factor,
                )
                for geometry in case.geometries
            ]
        except ValueError:
            # A degenerate resample that empties the valid mask carries no
            # information about the boundary; drop it rather than bias the
            # denominator toward exclusion.
            continue
        stack = z_stack(
            grid,
            geometries,
            drawn,
            mean=float(np.mean(drawn)),
            variance=float(np.var(drawn)),
        )
        inclusion += (stack.max(axis=0) >= threshold).astype(np.float32)
        kept += 1
    return inclusion / kept if kept else inclusion


@dataclass(frozen=True)
class RegionRecord:
    cre: str
    section: str
    region_id: str
    rank: int
    bandwidth: float
    p_stepdown: float
    threshold: float
    area_units2: float
    perimeter_units: float
    centroid_x: float
    centroid_y: float
    peak_x: float
    peak_y: float
    peak_z: float
    n_cells: int
    mean_activity_in: float
    mean_activity_out: float
    rate_ratio: float
    boundary_band_units: float
    n_scales_detected: int


def _contours(
    grid: SectionGrid, mask: np.ndarray
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Outer rings and hole rings of a boolean mask, in tissue coordinates."""
    from contourpy import LineType, contour_generator

    xs, ys = grid.pixel_centres()
    filled = ndimage.binary_fill_holes(mask)
    holes = filled & ~mask

    def rings(binary: np.ndarray) -> list[np.ndarray]:
        if not binary.any():
            return []
        generator = contour_generator(
            x=xs, y=ys, z=binary.astype(np.float64), line_type=LineType.Separate
        )
        return [np.asarray(line, dtype=np.float64) for line in generator.lines(0.5)]

    return rings(filled), rings(holes)


def _ring_length(ring: np.ndarray) -> float:
    if ring.shape[0] < 2:
        return 0.0
    return float(np.hypot(*np.diff(ring, axis=0).T).sum())


def _delineate(task: tuple[int, int, float]) -> tuple[
    list[RegionRecord], list[tuple[str, str, str, float]], dict[str, object], int
]:
    cre_index, section_index, _ = task
    contexts: list[SectionContext] = _SHARED["contexts"]  # type: ignore[assignment]
    cre_names: list[str] = _SHARED["cre_names"]  # type: ignore[assignment]
    context = contexts[section_index]
    cre = cre_names[cre_index]
    case = _case(context, cre_index)
    if case is None:
        return [], [], {}, 0
    geometries = list(case.geometries)
    marks = _marks(context, case, cre_index)
    raw = _raw_marks(context, case, cre_index)
    alpha = float(_SHARED["alpha"])  # type: ignore[arg-type]
    min_region_cells = int(_SHARED["min_region_cells"])  # type: ignore[arg-type]

    observed = z_stack(
        case.grid,
        geometries,
        marks,
        mean=float(np.mean(marks)),
        variance=float(np.var(marks)),
    )
    stored = null_z_stack(
        case.grid,
        geometries,
        marks,
        n_permutations=int(_SHARED["permutations"]),  # type: ignore[arg-type]
        rng=_rng(cre_index, section_index, 0),
        shuffle=_shuffler(case),
        domain=case.domain,
    )
    regions = extract_regions(
        case.grid,
        geometries,
        observed,
        stored,
        domain=case.domain,
        alpha=alpha,
        max_regions=int(_SHARED["max_regions"]),  # type: ignore[arg-type]
    )
    del stored

    pixel_area = case.grid.pixel_size**2
    xs, ys = case.grid.pixel_centres()
    records: list[RegionRecord] = []
    membership: list[tuple[str, str, str, float]] = []
    polygons: list[dict[str, object]] = []
    inclusion_total = np.zeros(case.grid.shape, dtype=np.float32)
    kept = [region for region in regions if region.cells.size >= min_region_cells]
    dropped = len(regions) - len(kept)

    for region in kept:
        inclusion = _bootstrap_inclusion(
            case,
            marks,
            region.threshold,
            n_draws=int(_SHARED["bootstrap"]),  # type: ignore[arg-type]
            rng=_rng(cre_index, section_index, 10 + region.rank),
            min_effective_cells=float(_SHARED["min_effective_cells"]),  # type: ignore[arg-type]
            max_cell_distance_factor=float(_SHARED["max_cell_distance_factor"]),  # type: ignore[arg-type]
        )
        # Confine the band to this region's neighbourhood so a second hotspot
        # elsewhere does not inflate it.
        near = ndimage.binary_dilation(
            region.mask, iterations=max(geometries[0].kernel_radius_px, 1)
        )
        inclusion = np.where(near, inclusion, 0.0).astype(np.float32)
        inclusion_total = np.maximum(inclusion_total, inclusion)

        # Effect sizes stay on the raw activity scale whatever the statistic was
        # computed on, so a region's rate ratio is comparable with the plotted
        # activity rather than with normal scores.
        inside = region.cells
        outside_mask = np.ones(case.n_cells, dtype=bool)
        outside_mask[inside] = False
        mean_in = float(raw[inside].mean())
        mean_out = (
            float(raw[outside_mask].mean()) if outside_mask.any() else float("nan")
        )

        best_bw = next(
            index
            for index, geometry in enumerate(geometries)
            if geometry.bandwidth == region.bandwidth
        )
        surface = observed[best_bw]
        peak_flat = int(np.argmax(np.where(region.mask, surface, -np.inf)))
        peak_row, peak_col = np.unravel_index(peak_flat, case.grid.shape)
        detected = sum(
            1
            for index in range(len(geometries))
            if observed[index][region.mask].max() >= region.threshold
        )

        outer, holes = _contours(case.grid, region.mask)
        perimeter = sum(_ring_length(ring) for ring in outer)
        band_lo = float((inclusion >= INCLUSION_LEVELS[0]).sum()) * pixel_area
        band_hi = float((inclusion >= INCLUSION_LEVELS[2]).sum()) * pixel_area
        band = (band_lo - band_hi) / perimeter if perimeter > 0 else float("nan")

        region_id = f"{cre}_{context.section}_R{region.rank}"
        records.append(
            RegionRecord(
                cre=cre,
                section=context.section,
                region_id=region_id,
                rank=region.rank,
                bandwidth=region.bandwidth,
                p_stepdown=region.p_value,
                threshold=region.threshold,
                area_units2=float(region.mask.sum()) * pixel_area,
                perimeter_units=perimeter,
                centroid_x=float(xs[region.mask].mean()),
                centroid_y=float(ys[region.mask].mean()),
                peak_x=float(xs[peak_row, peak_col]),
                peak_y=float(ys[peak_row, peak_col]),
                peak_z=float(surface[peak_row, peak_col]),
                n_cells=int(inside.size),
                mean_activity_in=mean_in,
                mean_activity_out=mean_out,
                rate_ratio=mean_in / mean_out if mean_out else float("nan"),
                boundary_band_units=band,
                n_scales_detected=detected,
            )
        )
        cell_inclusion = inclusion.reshape(-1)[case.grid.pixel_index[inside]]
        membership.extend(
            zip(
                context.obs_names[case.cells[inside]].tolist(),
                [cre] * inside.size,
                [region_id] * inside.size,
                cell_inclusion.astype(float).tolist(),
            )
        )
        polygons.append(
            {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [ring.tolist() for ring in outer]
                    + [ring.tolist() for ring in holes],
                },
                "properties": {
                    "region_id": region_id,
                    "cre": cre,
                    "section": context.section,
                    "rank": region.rank,
                    "bandwidth": region.bandwidth,
                    "p_stepdown": region.p_value,
                    "threshold": region.threshold,
                },
            }
        )

    best = int(np.argmax([observed[i][g.valid].max() for i, g in enumerate(geometries)]))
    surfaces = {
        "cre": cre,
        "section": context.section,
        "bandwidths": np.array([g.bandwidth for g in geometries], dtype=np.float32),
        "selected_bandwidth": np.float32(geometries[best].bandwidth),
        "pixel_size": np.float32(case.grid.pixel_size),
        "origin": np.array(case.grid.origin, dtype=np.float32),
        "z": observed,
        "activity_density": np.stack(
            [density_surface(case.grid, g, raw) for g in geometries]
        ),
        "cell_density": np.stack([g.cell_density for g in geometries]),
        "valid": np.stack([g.valid for g in geometries]),
        "inclusion_probability": inclusion_total,
        "region_labels": _label_image(case.grid, kept),
        "geojson": {
            "type": "FeatureCollection",
            "name": f"{cre}_{context.section}",
            "features": polygons,
        },
    }
    return records, membership, surfaces, dropped


def _label_image(grid: SectionGrid, regions: Iterable[Region]) -> np.ndarray:
    labels = np.zeros(grid.shape, dtype=np.int16)
    for region in regions:
        labels[region.mask] = region.rank
    return labels


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #


def _run_pool(function, tasks, jobs: int) -> list:
    """Fan out over ``_SHARED``, which the workers inherit rather than receive.

    The activity matrix is hundreds of megabytes; passing it through
    ``initargs`` would pickle a private copy into every worker. Forking after
    ``_SHARED`` is populated shares it copy-on-write instead, and nothing here
    writes to it.
    """
    if jobs == 1:
        return [function(task) for task in tasks]
    with get_context("fork").Pool(jobs) as pool:
        return list(pool.imap(function, tasks, chunksize=1))


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    validate_args(args)

    outdir = os.path.abspath(args.outdir)
    for sub in ("", "regions", "surfaces", "logs"):
        os.makedirs(os.path.join(outdir, sub), exist_ok=True)

    log(f"[input] activity matrix {args.activity_npz} key={args.matrix}")
    values, npz_obs_names, cre_names = load_activity(
        args.activity_npz, args.matrix, args.cres, args.limit
    )
    metadata = read_obs_metadata(args.h5ad)
    order = pd.Index(npz_obs_names).get_indexer(metadata["obs_name"].to_numpy())
    if (order < 0).any():
        raise ValueError("the h5ad and the activity matrix disagree on cell names")
    values = values[order]

    finite = np.isfinite(values).all(axis=1)
    log(
        f"[input] {len(cre_names)} cCREs, {int(finite.sum())} of {finite.size} cells "
        f"usable ({int((~finite).sum())} lack a negative-control reference)"
    )

    vocabulary = np.unique(metadata["subclass"].to_numpy())
    admissible = build_admissible(
        args.t7_totals, cre_names, vocabulary, args.min_t7
    )

    pixel_size = args.pixel_size or min(args.bandwidths) / PIXELS_PER_BANDWIDTH
    contexts_by_name = build_section_contexts(
        metadata, finite, args.sections, pixel_size, vocabulary
    )
    contexts = [contexts_by_name[name] for name in sorted(contexts_by_name)]

    if args.permute_observed:
        log("[calibration] --permute-observed: shuffling the marks once up front")
        rng = np.random.default_rng(args.seed + 1)
        for context in contexts:
            codes = context.subclass_codes
            changes = np.flatnonzero(codes[1:] != codes[:-1]) + 1
            blocks = np.concatenate(([0], changes, [codes.size]))
            shuffle = mark_shuffler(
                blocks if args.null == "within_subclass" else None
            )
            for column in range(values.shape[1]):
                marks = np.ascontiguousarray(values[context.rows, column])
                shuffle(marks, rng)
                values[context.rows, column] = marks

    controls = negative_control_cres(args.manifest)
    _SHARED.update(
        {
            "values": values,
            "contexts": contexts,
            "cre_names": cre_names,
            "permutations": args.permutations,
            "bootstrap": args.bootstrap,
            "alpha": args.alpha,
            "max_regions": args.max_regions,
            "min_effective_cells": args.min_effective_cells,
            "max_cell_distance_factor": TISSUE_MASK_BANDWIDTHS,
            "bandwidths": tuple(sorted(args.bandwidths)),
            "admissible": admissible,
            "min_case_cells": args.min_case_cells,
            "min_region_cells": args.min_region_cells,
            "mark_transform": args.mark_transform,
            "null": args.null,
            "seed": args.seed,
        }
    )

    tasks = [
        (cre_index, section_index)
        for cre_index in range(len(cre_names))
        for section_index in range(len(contexts))
    ]
    log(f"[screen] {len(tasks)} (cCRE, section) pairs x {args.permutations} permutations")
    results: list[ScreenRecord | None] = _run_pool(_screen, tasks, args.jobs)
    screened = [record for record in results if record is not None]
    skipped = len(results) - len(screened)
    if skipped:
        log(
            f"[screen] {skipped} pairs not tested: fewer than "
            f"--min-case-cells {args.min_case_cells} cells survive T7 >= {args.min_t7}"
        )
    if not screened:
        raise ValueError("no (cCRE, section) pair had enough admissible cells to test")

    summary = pd.DataFrame(
        [
            {
                "cre": record.cre,
                "section": record.section,
                "null": args.null,
                "matrix": args.matrix,
                "mark_transform": args.mark_transform,
                "min_t7": args.min_t7,
                "n_cells": record.n_cells,
                "mean_activity": record.mean_activity,
                "sd_activity": record.sd_activity,
                "selected_bandwidth": record.selected_bandwidth,
                "max_z": record.max_z,
                "p_fwer": record.p_fwer,
                "is_negative_control": record.cre in controls,
            }
            for record in screened
        ]
    )
    summary["q_global_bh"] = bh_fdr(summary["p_fwer"].to_numpy())
    summary["significant"] = summary["q_global_bh"] <= args.alpha

    scales = pd.DataFrame(
        [
            {
                "cre": record.cre,
                "section": record.section,
                "null": args.null,
                "bandwidth": bandwidth,
                "max_z": max_z,
                "n_valid_pixels": n_valid,
            }
            for record in screened
            for bandwidth, max_z, n_valid in zip(
                sorted(args.bandwidths),
                record.max_z_per_bandwidth,
                record.n_valid_per_bandwidth,
            )
        ]
    )
    scales.to_csv(os.path.join(outdir, "activity_density_scales.csv"), index=False)
    log(
        f"[screen] {int(summary['significant'].sum())} of {len(summary)} pairs at "
        f"q <= {args.alpha}; controls significant: "
        f"{int((summary['significant'] & summary['is_negative_control']).sum())}"
    )

    region_frame = pd.DataFrame(columns=[field for field in RegionRecord.__annotations__])
    membership_rows: list[tuple[str, str, str, float]] = []
    if args.skip_regions:
        log("[regions] skipped by --skip-regions")
    else:
        chosen = [
            (record.cre_index, record.section_index, record.selected_bandwidth)
            for record, keep in zip(screened, summary["significant"].to_numpy())
            if keep
        ]
        log(f"[regions] delineating {len(chosen)} pairs")
        if chosen:
            results = _run_pool(_delineate, chosen, args.region_jobs or args.jobs)
            records: list[RegionRecord] = []
            dropped_total = 0
            for region_records, membership, surfaces, dropped in results:
                records.extend(region_records)
                membership_rows.extend(membership)
                _write_surfaces(outdir, surfaces)
                dropped_total += dropped
            if records:
                region_frame = pd.DataFrame([vars(record) for record in records])
            log(
                f"[regions] {len(records)} regions across {len(chosen)} pairs; "
                f"{dropped_total} components dropped under --min-region-cells "
                f"{args.min_region_cells}"
            )

    region_frame.to_csv(
        os.path.join(outdir, "activity_density_regions.csv"), index=False
    )
    summary.to_csv(os.path.join(outdir, "activity_density_summary.csv"), index=False)
    _write_membership(outdir, membership_rows)
    log(f"[done] wrote {outdir}")
    return 0


def _write_surfaces(outdir: str, surfaces: dict[str, object]) -> None:
    if not surfaces:  # the pair had too few admissible cells to delineate
        return
    cre = str(surfaces.pop("cre"))
    section = str(surfaces.pop("section"))
    geojson = surfaces.pop("geojson")
    with open(
        os.path.join(outdir, "regions", f"{cre}_{section}.geojson"),
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(geojson, handle)
    np.savez_compressed(
        os.path.join(outdir, "surfaces", f"{cre}_{section}.npz"), **surfaces
    )


def _write_membership(
    outdir: str, rows: Sequence[tuple[str, str, str, float]]
) -> None:
    path = os.path.join(outdir, "region_cell_membership.csv.gz")
    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        handle.write("obs_name,cre,region_id,inclusion_probability\n")
        for obs_name, cre, region_id, inclusion in rows:
            handle.write(f"{obs_name},{cre},{region_id},{inclusion:.4f}\n")


if __name__ == "__main__":
    raise SystemExit(main())
