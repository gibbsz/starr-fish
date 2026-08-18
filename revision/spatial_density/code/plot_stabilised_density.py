#!/usr/bin/env python3
"""Density-stabilised activity maps -- a visualisation, with no hypothesis test.

``R(u) = D(u)/N(u)`` is a kernel-weighted *mean*, so it is already unbiased for
local mean activity at any cell density. What density still buys is noise:
thinly-sampled areas throw extreme values by chance and dominate the eye. That
noise is removed in closed form by :func:`activity_density.stabilise` --
empirical-Bayes shrinkage using the exact per-pixel sampling variance the
geometry already carries. No permutations, no p-values.

Five panels per (cCRE, section):

1. the cells, coloured by activity, with T7-excluded cells dimmed;
2. the cell density ``N`` -- the artefact being corrected, shown so it can be
   audited rather than taken on trust;
3. ``R``, raw;
4. ``R`` stabilised, on the same colour limits as panel 3 so the size of the
   correction is visible;
5. the shrinkage weight ``w(u)`` -- where the map kept its value and where it
   was pulled to baseline.

Panels 3 and 4 show activity **relative to that cCRE's own section baseline**,
diverging about 1.0. That is the only self-consistent choice: empirical-Bayes
shrinks toward the section mean, so a pixel with no local evidence lands exactly
on the centre of the scale and reads as "nothing to say here" rather than as a
hotspot. The absolute baseline is printed in the panel title and recorded as
``baseline_mean`` in the summary, since it varies enormously between cCREs
(11.7x the negative-control level for one, 0.33x for another).

**This removes cell density only, not cell-type composition.** Activity differs
by subclass and subclasses are spatially organised, so bright areas may still be
cell-type anatomy rather than position. Nothing here is a significance claim.

Cells are selected through ``cases.build_case``, the same builder the permutation
workflow uses, so these figures show exactly the admissible cell set.
"""

from __future__ import annotations

import argparse
import os
import sys
from multiprocessing import get_context
from typing import Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LogNorm, TwoSlopeNorm  # noqa: E402

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
for _path in (REPO_ROOT, SCRIPT_DIR):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from activity_density import stabilise, transform_marks  # noqa: E402
from cases import (  # noqa: E402
    DEFAULT_ACTIVITY_NPZ,
    DEFAULT_H5AD,
    DEFAULT_MANIFEST,
    DEFAULT_T7_TOTALS,
    build_admissible,
    build_case,
    build_section_contexts,
    load_activity,
    negative_control_cres,
)
from baystarrfish.data import read_obs_metadata  # noqa: E402

DEFAULT_OUTDIR = os.path.join(WORKFLOW_DIR, "results")
DEFAULT_FIGDIR = os.path.join(WORKFLOW_DIR, "figures", "stabilised")
# 1.0 is the negative-control level; both the per-cell panel and the diverging
# surface panels are anchored there.
BASELINE = 1.0
FACECOLOR = "black"
EXCLUDED_CONTEXT_COLOR = "#141414"
TESTED_COLOR = "#454545"
# "rank" is deliberately absent: normal scores have no activity units, so a
# stabilised rank surface cannot be read on the 1.0 baseline scale.
DISPLAY_TRANSFORMS = ("log", "none")
DEFAULT_BANDWIDTH = 200.0
DEFAULT_MIN_T7 = 50.0
DEFAULT_MIN_CASE_CELLS = 500
DEFAULT_MIN_EFFECTIVE_CELLS = 25.0
TISSUE_MASK_BANDWIDTHS = 1.0
PIXELS_PER_BANDWIDTH = 2.0


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
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--t7-totals", default=DEFAULT_T7_TOTALS)
    parser.add_argument("--outdir", default=DEFAULT_OUTDIR)
    parser.add_argument("--figdir", default=DEFAULT_FIGDIR)
    parser.add_argument("--matrix", default="activity_normalized")
    parser.add_argument("--cres", nargs="*", default=None)
    parser.add_argument("--sections", nargs="*", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--bandwidth", type=float, default=DEFAULT_BANDWIDTH)
    parser.add_argument("--pixel-size", type=float, default=None)
    parser.add_argument("--min-t7", type=float, default=DEFAULT_MIN_T7)
    parser.add_argument("--min-case-cells", type=int, default=DEFAULT_MIN_CASE_CELLS)
    parser.add_argument(
        "--min-effective-cells", type=float, default=DEFAULT_MIN_EFFECTIVE_CELLS
    )
    parser.add_argument(
        "--mark-transform",
        default=DISPLAY_TRANSFORMS[0],
        choices=DISPLAY_TRANSFORMS,
        help="scale the local mean and the shrinkage are computed on. log "
        "shrinks on log1p and displays expm1 back, which keeps the 1.0 baseline "
        "exact while stopping single extreme cells from setting the map",
    )
    parser.add_argument("--formats", nargs="+", default=["png", "pdf"])
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--point-size", type=float, default=0.6)
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    for name in ("h5ad", "activity_npz", "manifest"):
        path = getattr(args, name)
        if not os.path.exists(path):
            raise FileNotFoundError(f"--{name.replace('_', '-')} not found: {path}")
    if args.min_t7 > 0 and not os.path.exists(args.t7_totals):
        raise FileNotFoundError(
            f"--t7-totals not found: {args.t7_totals}; run make_t7_totals.py first"
        )
    if args.bandwidth <= 0:
        raise ValueError("--bandwidth must be positive")
    if args.min_t7 < 0:
        raise ValueError("--min-t7 must be non-negative")
    if args.jobs < 1:
        raise ValueError("--jobs must be at least 1")
    if args.dpi < 30:
        raise ValueError("--dpi below 30 is unreadable")
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be at least 1")


# --------------------------------------------------------------------------- #
# rendering
# --------------------------------------------------------------------------- #


def _frames(grid, shape: tuple[int, int]):
    """``imshow`` extent (pixel edges) and pixel-centre coordinates."""
    rows, cols = shape
    x0, y0 = grid.origin
    pixel = grid.pixel_size
    extent = (x0, x0 + cols * pixel, y0, y0 + rows * pixel)
    return extent


def _to_activity(surface: np.ndarray, transform: str) -> np.ndarray:
    """Undo the working scale so every panel reads in activity units."""
    return np.expm1(surface) if transform == "log" else surface


def _relative(surface: np.ndarray, baseline: float) -> np.ndarray:
    """Activity as a multiple of the section baseline the shrinkage targets.

    Centring the display on the shrinkage target is what makes a fully-shrunk
    pixel land on 1.0: "no local evidence" then looks neutral instead of looking
    like whatever the section average happens to be.
    """
    if not np.isfinite(baseline) or baseline == 0:
        return surface
    return surface / baseline


def _diverging(*surfaces: np.ndarray) -> TwoSlopeNorm:
    """Colour limits centred on 1.0 -- parity with the baseline -- shared across panels.

    ``TwoSlopeNorm`` requires vmin < vcenter < vmax, which a map lying entirely
    above or below parity would violate; the guards keep it well-formed rather
    than letting the figure raise.
    """
    finite = np.concatenate([s[np.isfinite(s)].ravel() for s in surfaces])
    if finite.size == 0:
        return TwoSlopeNorm(vmin=0.0, vcenter=BASELINE, vmax=2.0)
    low = float(np.percentile(finite, 1.0))
    high = float(np.percentile(finite, 99.0))
    return TwoSlopeNorm(
        vmin=min(low, BASELINE - 1e-3),
        vcenter=BASELINE,
        vmax=max(high, BASELINE + 1e-3),
    )


def render(payload: dict[str, object], stem: str, args: argparse.Namespace) -> None:
    grid = payload["grid"]
    raw = np.asarray(payload["raw"])
    shrunk = np.asarray(payload["shrunk"])
    weight = np.asarray(payload["weight"])
    density = np.asarray(payload["cell_density"])
    valid = np.asarray(payload["valid"])
    cells: pd.DataFrame = payload["cells"]  # type: ignore[assignment]
    extent = _frames(grid, raw.shape)

    raw = np.where(valid, raw, np.nan)
    shrunk = np.where(valid, shrunk, np.nan)
    weight = np.where(valid, weight, np.nan)
    density = np.where(valid, density, np.nan)

    fig, axes = plt.subplots(1, 5, figsize=(27, 8), facecolor=FACECOLOR)
    for axis in axes:
        axis.set_facecolor(FACECOLOR)
        axis.set_xticks([])
        axis.set_yticks([])
        axis.set_aspect("equal")

    used = cells["admissible"].to_numpy()
    axes[0].scatter(
        cells.loc[~used, "x"], cells.loc[~used, "y"],
        c=EXCLUDED_CONTEXT_COLOR, s=args.point_size, linewidths=0,
    )
    activity = np.clip(cells.loc[used, "activity"].to_numpy(), BASELINE, None)
    vmax = float(np.percentile(activity, 99.5)) if activity.size else 2.0
    scatter = axes[0].scatter(
        cells.loc[used, "x"], cells.loc[used, "y"], c=activity, s=args.point_size,
        cmap="magma", norm=LogNorm(vmin=BASELINE, vmax=max(vmax, BASELINE * 2)),
        linewidths=0,
    )
    axes[0].set_title(
        f"{payload['cre']} {payload['section']}\n"
        f"{int(used.sum())} cells, {int((~used).sum())} excluded (T7 filter)",
        color="white",
    )
    fig.colorbar(scatter, ax=axes[0], fraction=0.04)

    image = axes[1].imshow(density, origin="lower", extent=extent, cmap="bone")
    axes[1].set_title("cell density N\n(the artefact being corrected)", color="white")
    fig.colorbar(image, ax=axes[1], fraction=0.04)

    baseline = float(payload["baseline"])  # type: ignore[arg-type]
    raw = _relative(raw, baseline)
    shrunk = _relative(shrunk, baseline)
    norm = _diverging(raw, shrunk)
    for axis, surface, title in (
        (axes[2], raw, f"R / baseline, raw (h={payload['bandwidth']:g})"),
        (axes[3], shrunk, f"R / baseline, stabilised\nbaseline = {baseline:.3g}x control"),
    ):
        image = axis.imshow(
            surface, origin="lower", extent=extent, cmap="RdBu_r", norm=norm
        )
        axis.set_title(title, color="white")
        fig.colorbar(image, ax=axis, fraction=0.04)

    image = axes[4].imshow(
        weight, origin="lower", extent=extent, cmap="viridis", vmin=0.0, vmax=1.0
    )
    tau = float(payload["tau_squared"])  # type: ignore[arg-type]
    axes[4].set_title(
        "shrinkage weight w\n"
        + (
            f"mean {np.nanmean(weight):.2f}, tau^2 = {tau:.3g}"
            if tau > 0
            else "tau^2 = 0: nothing exceeds sampling noise"
        ),
        color="white",
    )
    fig.colorbar(image, ax=axes[4], fraction=0.04)

    fig.text(
        0.5, 0.02,
        "Cell density normalised out; cell-type composition is NOT. "
        "Descriptive map, not a significance claim.",
        ha="center", color="#9e9e9e", fontsize=10,
    )
    for suffix in args.formats:
        fig.savefig(
            f"{stem}.{suffix}", dpi=args.dpi, bbox_inches="tight",
            facecolor=fig.get_facecolor(),
        )
    plt.close(fig)


# --------------------------------------------------------------------------- #
# per-pair work
# --------------------------------------------------------------------------- #

_SHARED: dict[str, object] = {}


def _one(task: tuple[int, int]) -> dict[str, object] | None:
    cre_index, section_index = task
    args: argparse.Namespace = _SHARED["args"]  # type: ignore[assignment]
    contexts: list = _SHARED["contexts"]  # type: ignore[assignment]
    cre_names: list[str] = _SHARED["cre_names"]  # type: ignore[assignment]
    values: np.ndarray = _SHARED["values"]  # type: ignore[assignment]
    metadata: pd.DataFrame = _SHARED["metadata"]  # type: ignore[assignment]
    controls: set[str] = _SHARED["controls"]  # type: ignore[assignment]
    context = contexts[section_index]
    cre = cre_names[cre_index]

    stem = os.path.join(args.figdir, f"{cre}_{context.section}_stabilised")
    case = build_case(
        context,
        cre_index,
        values=values,
        admissible=_SHARED["admissible"],  # type: ignore[arg-type]
        bandwidths=(args.bandwidth,),
        min_effective_cells=args.min_effective_cells,
        max_cell_distance_factor=TISSUE_MASK_BANDWIDTHS,
        min_cells=args.min_case_cells,
    )
    if case is None:
        return None

    geometry = case.geometries[0]
    raw_activity = np.ascontiguousarray(
        values[context.rows[case.cells], cre_index], dtype=np.float32
    )
    marks = transform_marks(raw_activity, args.mark_transform)
    surface = stabilise(case.grid, geometry, marks)
    baseline_activity = float(
        _to_activity(np.asarray(surface.prior_mean), args.mark_transform)
    )

    raw = _to_activity(surface.raw, args.mark_transform)
    shrunk = _to_activity(surface.shrunk, args.mark_transform)
    valid = geometry.valid

    if args.overwrite or not os.path.exists(f"{stem}.{args.formats[0]}"):
        cells = pd.DataFrame(
            {
                "x": metadata.loc[context.rows, "x"].to_numpy(),
                "y": metadata.loc[context.rows, "y"].to_numpy(),
                "activity": values[context.rows, cre_index],
                "admissible": np.isin(np.arange(context.n_cells), case.cells),
            }
        )
        render(
            {
                "cre": cre,
                "section": context.section,
                "grid": case.grid,
                "raw": raw,
                "shrunk": shrunk,
                "weight": surface.weight,
                "cell_density": geometry.cell_density,
                "valid": valid,
                "cells": cells,
                "bandwidth": args.bandwidth,
                "tau_squared": surface.tau_squared,
                "baseline": baseline_activity,
            },
            stem,
            args,
        )

    inside = shrunk[valid]
    relative = inside / baseline_activity if baseline_activity else inside
    peak = np.unravel_index(
        int(np.nanargmax(np.where(valid, shrunk, -np.inf))), shrunk.shape
    )
    xs, ys = case.grid.pixel_centres()
    pixel_area = case.grid.pixel_size**2
    return {
        "cre": cre,
        "section": context.section,
        "bandwidth": args.bandwidth,
        "mark_transform": args.mark_transform,
        "min_t7": args.min_t7,
        "n_cells": case.n_cells,
        "n_excluded": context.n_cells - case.n_cells,
        "baseline_mean": baseline_activity,
        "tau_squared": surface.tau_squared,
        "mean_weight": float(np.nanmean(surface.weight[valid])),
        "max_shrunk_activity": float(np.nanmax(inside)),
        "max_shrunk_relative": float(np.nanmax(relative)),
        "max_raw_activity": float(np.nanmax(raw[valid])),
        "peak_x": float(xs[peak]),
        "peak_y": float(ys[peak]),
        # Relative to this cCRE's own baseline, not to an absolute 1.0: an
        # absolute cut just measures how active the cCRE is overall.
        "area_above_2x_baseline": float(np.nansum(relative >= 2.0)) * pixel_area,
        "is_negative_control": cre in controls,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    validate_args(args)
    os.makedirs(args.figdir, exist_ok=True)
    os.makedirs(args.outdir, exist_ok=True)

    values, npz_obs_names, cre_names = load_activity(
        args.activity_npz, args.matrix, args.cres, args.limit
    )
    metadata = read_obs_metadata(args.h5ad)
    order = pd.Index(npz_obs_names).get_indexer(metadata["obs_name"].to_numpy())
    if (order < 0).any():
        raise ValueError("the h5ad and the activity matrix disagree on cell names")
    values = values[order]
    finite = np.isfinite(values).any(axis=1)

    vocabulary = np.unique(metadata["subclass"].to_numpy())
    admissible = build_admissible(args.t7_totals, cre_names, vocabulary, args.min_t7)
    pixel_size = args.pixel_size or args.bandwidth / PIXELS_PER_BANDWIDTH
    contexts_by_name = build_section_contexts(
        metadata, finite, args.sections, pixel_size, vocabulary
    )
    contexts = [contexts_by_name[name] for name in sorted(contexts_by_name)]

    _SHARED.update(
        {
            "args": args,
            "values": values,
            "metadata": metadata,
            "contexts": contexts,
            "cre_names": cre_names,
            "admissible": admissible,
            "controls": negative_control_cres(args.manifest),
        }
    )

    tasks = [
        (cre_index, section_index)
        for cre_index in range(len(cre_names))
        for section_index in range(len(contexts))
    ]
    log(f"[plot] {len(tasks)} (cCRE, section) pairs at h={args.bandwidth:g}")
    if args.jobs == 1:
        rows = [_one(task) for task in tasks]
    else:
        with get_context("fork").Pool(args.jobs) as pool:
            rows = list(pool.imap(_one, tasks, chunksize=1))

    kept = [row for row in rows if row is not None]
    skipped = len(rows) - len(kept)
    frame = pd.DataFrame(kept)
    path = os.path.join(args.outdir, "stabilised_density_summary.csv")
    if not frame.empty:
        frame = frame.sort_values("max_shrunk_relative", ascending=False)
    frame.to_csv(path, index=False)

    flat = int((frame["tau_squared"] == 0).sum()) if not frame.empty else 0
    log(
        f"[plot] {len(kept)} figures into {args.figdir}; {skipped} pairs had too "
        f"few admissible cells; {flat} had tau^2 = 0 (flat: nothing beyond noise)"
    )
    log(f"[plot] ranking written to {path} -- descriptive, not a significance test")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
