#!/usr/bin/env python3
"""Figures for the activity-density surfaces, significant or not.

A cCRE with no significant region still has a surface worth looking at, and the
standardised surface ``z`` needs no permutations -- its null moments are
closed-form -- so any (cCRE, section) pair can be drawn cheaply whether or not
it passed the test.

Four panels per pair, sharing one set of axes:

1. the cells, coloured by estimated activity. **Cells excluded by the T7
   admissibility filter are drawn as dark background, not coloured**: they were
   not in the test and must not appear to be in the picture.
2. the smoothed local-mean-activity surface ``R``;
3. the standardised surface ``z``, with the simultaneous threshold drawn when
   the pair produced regions;
4. the regions with their bootstrap band, or -- when there are none -- the peak
   location marked, annotated with the p-value it failed to reach.

Cells are selected through ``cases.build_case``, the same builder the test uses,
so a figure can never show cells the statistic did not see.

Plus an overview: the calibration QQ of ``p_fwer`` with negative controls picked
out, region sizes, and centroids.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LogNorm  # noqa: E402

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WORKFLOW_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
REPO_ROOT = os.path.abspath(os.path.join(WORKFLOW_DIR, "..", ".."))
for _path in (REPO_ROOT, SCRIPT_DIR):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from activity_density import density_surface, transform_marks, z_stack  # noqa: E402
from cases import (  # noqa: E402
    DEFAULT_ACTIVITY_NPZ,
    DEFAULT_H5AD,
    DEFAULT_T7_TOTALS,
    build_admissible,
    build_case,
    build_section_contexts,
    load_activity,
)
from baystarrfish.data import read_obs_metadata  # noqa: E402

DEFAULT_RESULTS = os.path.join(WORKFLOW_DIR, "results")
DEFAULT_FIGDIR = os.path.join(WORKFLOW_DIR, "figures")
# Matches _MODE_DEFAULT_VMIN["activity_posterior_normalized"] in
# baystarrfish/plotting/spatial.py: 1.0 is the negative-control level.
ACTIVITY_VMIN = 1.0
INCLUSION_LEVELS = (0.05, 0.5, 0.95)
FACECOLOR = "black"
# Excluded cells are shown for tissue context in panel 1 (dimmer than any
# real activity, so they cannot be mistaken for signal) and called out
# explicitly in panel 4, which is the "what was actually tested" panel.
EXCLUDED_CONTEXT_COLOR = "#141414"
EXCLUDED_CALLOUT_COLOR = "#C2185B"
TESTED_COLOR = "#454545"
DEFAULT_TOP = 24


def log(message: str) -> None:
    print(message, flush=True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n")[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--h5ad", default=DEFAULT_H5AD)
    parser.add_argument("--activity-npz", default=DEFAULT_ACTIVITY_NPZ)
    parser.add_argument("--results", default=DEFAULT_RESULTS)
    parser.add_argument("--figdir", default=DEFAULT_FIGDIR)
    parser.add_argument("--formats", nargs="+", default=["png", "pdf"])
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--point-size", type=float, default=0.6)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--only-overview", action="store_true")
    parser.add_argument("--cres", nargs="*", default=None)
    parser.add_argument(
        "--top",
        type=int,
        default=DEFAULT_TOP,
        help="when nothing is significant, draw this many pairs ranked by "
        "p_fwer then max_z, so a null result is still inspectable",
    )
    parser.add_argument(
        "--all-pairs",
        action="store_true",
        help="draw every tested pair rather than the significant ones plus --top",
    )
    # These must match the run that produced --results, or the figure would show
    # a different cell set from the test; they are echoed in the summary CSV.
    parser.add_argument("--min-t7", type=float, default=None)
    parser.add_argument("--t7-totals", default=DEFAULT_T7_TOTALS)
    parser.add_argument("--min-case-cells", type=int, default=500)
    parser.add_argument("--bandwidths", nargs="+", type=float, default=[100.0, 200.0, 400.0])
    parser.add_argument("--pixel-size", type=float, default=None)
    parser.add_argument("--min-effective-cells", type=float, default=25.0)
    parser.add_argument("--tissue-mask-bandwidths", type=float, default=1.0)
    parser.add_argument("--mark-transform", default=None)
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    summary = os.path.join(args.results, "activity_density_summary.csv")
    if not os.path.exists(summary):
        raise FileNotFoundError(
            f"{summary} missing; run run_activity_density_hotspots.py first"
        )
    if args.dpi < 30:
        raise ValueError("--dpi below 30 is unreadable")
    if args.top < 0:
        raise ValueError("--top must be non-negative")


def _save(fig: plt.Figure, stem: str, formats: Sequence[str], dpi: int) -> None:
    for suffix in formats:
        fig.savefig(
            f"{stem}.{suffix}",
            dpi=dpi,
            bbox_inches="tight",
            facecolor=fig.get_facecolor(),
        )
    plt.close(fig)


def _frames(grid, shape: tuple[int, int]) -> tuple[tuple[float, ...], np.ndarray, np.ndarray]:
    """``imshow`` extent (pixel edges) and ``contour`` coordinates (pixel centres).

    ``contour`` only honours ``extent`` in combination with ``origin``, so its
    coordinates are passed explicitly -- a silently mis-registered boundary would
    be worse than no figure.
    """
    rows, cols = shape
    x0, y0 = grid.origin
    pixel = grid.pixel_size
    extent = (x0, x0 + cols * pixel, y0, y0 + rows * pixel)
    xs = x0 + (np.arange(cols) + 0.5) * pixel
    ys = y0 + (np.arange(rows) + 0.5) * pixel
    grid_x, grid_y = np.meshgrid(xs, ys)
    return extent, grid_x, grid_y


def plot_pair(
    bundle: dict[str, object], regions: pd.DataFrame, stem: str, args: argparse.Namespace
) -> None:
    grid = bundle["grid"]
    z_all = np.asarray(bundle["z"])
    density_all = np.asarray(bundle["activity_density"])
    valid_all = np.asarray(bundle["valid"])
    bandwidths = np.asarray(bundle["bandwidths"])
    index = int(bundle["selected_index"])  # type: ignore[arg-type]
    selected = float(bandwidths[index])
    valid = valid_all[index]
    z = np.where(valid, z_all[index], np.nan)
    density = np.where(valid, density_all[index], np.nan)
    extent, grid_x, grid_y = _frames(grid, z.shape)

    cells = bundle["cells"]  # DataFrame with x, y, activity, admissible
    labels = np.asarray(bundle.get("region_labels", np.zeros(z.shape, dtype=np.int16)))
    inclusion = np.asarray(bundle.get("inclusion_probability", np.zeros(z.shape, np.float32)))
    threshold = float(regions["threshold"].iloc[0]) if len(regions) else np.nan

    fig, axes = plt.subplots(1, 4, figsize=(22, 8), facecolor=FACECOLOR)
    for axis in axes:
        axis.set_facecolor(FACECOLOR)
        axis.set_xticks([])
        axis.set_yticks([])
        axis.set_aspect("equal")

    # Panel 1 -- excluded cells are background, never coloured.
    used = cells["admissible"].to_numpy()
    axes[0].scatter(
        cells.loc[~used, "x"], cells.loc[~used, "y"],
        c=EXCLUDED_CONTEXT_COLOR, s=args.point_size, linewidths=0,
    )
    activity = np.clip(cells.loc[used, "activity"].to_numpy(), ACTIVITY_VMIN, None)
    vmax = float(np.percentile(activity, 99.5)) if activity.size else 2.0
    scatter = axes[0].scatter(
        cells.loc[used, "x"], cells.loc[used, "y"], c=activity, s=args.point_size,
        cmap="magma",
        norm=LogNorm(vmin=ACTIVITY_VMIN, vmax=max(vmax, ACTIVITY_VMIN * 2)),
        linewidths=0,
    )
    axes[0].set_title(
        f"{bundle['cre']} {bundle['section']}\n"
        f"{int(used.sum())} cells tested, {int((~used).sum())} excluded (T7 filter)",
        color="white",
    )
    fig.colorbar(scatter, ax=axes[0], fraction=0.04)

    image = axes[1].imshow(
        density, origin="lower", extent=extent, cmap="magma", vmin=ACTIVITY_VMIN,
        vmax=float(np.nanpercentile(density, 99.5)) if np.isfinite(density).any() else None,
    )
    axes[1].set_title(f"local mean activity R (h={selected:g})", color="white")
    fig.colorbar(image, ax=axes[1], fraction=0.04)

    image = axes[2].imshow(z, origin="lower", extent=extent, cmap="viridis")
    if np.isfinite(threshold):
        axes[2].contour(
            grid_x, grid_y, np.nan_to_num(z, nan=-np.inf),
            levels=[threshold], colors="white", linewidths=1.0,
        )
    axes[2].set_title(
        "standardised z"
        + (f", threshold {threshold:.2f}" if np.isfinite(threshold) else ""),
        color="white",
    )
    fig.colorbar(image, ax=axes[2], fraction=0.04)

    axes[3].scatter(
        cells.loc[used, "x"], cells.loc[used, "y"],
        c=TESTED_COLOR, s=args.point_size, linewidths=0, label="tested",
    )
    if (~used).any():
        axes[3].scatter(
            cells.loc[~used, "x"], cells.loc[~used, "y"],
            c=EXCLUDED_CALLOUT_COLOR, s=args.point_size, linewidths=0,
            label="excluded (T7 < threshold)",
        )
        legend = axes[3].legend(
            fontsize=7, loc="lower left", framealpha=0.3, markerscale=8
        )
        for text in legend.get_texts():
            text.set_color("white")
    if labels.any():
        axes[3].contour(
            grid_x, grid_y, (labels > 0).astype(float), levels=[0.5],
            colors="#FFD54F", linewidths=1.8,
        )
        for level, style in zip(INCLUSION_LEVELS[::2], ("dotted", "dashed")):
            if (inclusion >= level).any():
                axes[3].contour(
                    grid_x, grid_y, inclusion, levels=[level],
                    colors="#4FC3F7", linewidths=0.9, linestyles=style,
                )
        axes[3].set_title(
            f"{int(labels.max())} region(s); band = bootstrap 5%/95%", color="white"
        )
    else:
        # No region survived, so mark where the surface peaked and say plainly
        # that it did not clear the threshold.
        peak = np.unravel_index(int(np.nanargmax(np.where(valid, z, -np.inf))), z.shape)
        axes[3].plot(
            grid_x[peak], grid_y[peak], marker="+", ms=18, mew=2, color="#FF7043"
        )
        p_value = bundle.get("p_fwer")
        q_value = bundle.get("q_global_bh")
        axes[3].set_title(
            "no significant region; peak marked\n"
            f"max z = {np.nanmax(z):.1f}, p_fwer = {p_value:.3f}, q = {q_value:.3f}",
            color="white",
        )
    axes[3].set_xlim(axes[0].get_xlim())
    axes[3].set_ylim(axes[0].get_ylim())

    _save(fig, stem, args.formats, args.dpi)


def plot_overview(
    summary: pd.DataFrame, regions: pd.DataFrame, figdir: str, args: argparse.Namespace
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    for label, frame, colour in (
        ("cCREs", summary[~summary["is_negative_control"]], "#1f77b4"),
        ("negative controls", summary[summary["is_negative_control"]], "#d62728"),
    ):
        if frame.empty:
            continue
        observed = np.sort(frame["p_fwer"].to_numpy())
        expected = (np.arange(observed.size) + 0.5) / observed.size
        axes[0].plot(expected, observed, "o", ms=3, color=colour, label=label)
    axes[0].plot([0, 1], [0, 1], "k--", lw=1)
    axes[0].set(xlabel="expected quantile", ylabel="p_fwer", title="calibration")
    axes[0].legend(fontsize=8)

    axes[1].hist(summary["max_z"].to_numpy(), bins=40, color="#1f77b4")
    axes[1].set(xlabel="max z", ylabel="pairs", title=f"{len(summary)} pairs tested")

    if not regions.empty:
        for section, frame in regions.groupby("section"):
            axes[2].scatter(
                frame["centroid_x"], frame["centroid_y"],
                s=np.sqrt(frame["area_units2"]) / 40.0, alpha=0.5, label=str(section),
            )
        axes[2].set_aspect("equal")
        axes[2].legend(fontsize=8)
        axes[2].set(title=f"{len(regions)} region centroids", xlabel="x", ylabel="y")
    else:
        axes[2].text(0.5, 0.5, "no significant regions", ha="center", va="center")
        axes[2].set(title="regions", xticks=[], yticks=[])

    fig.tight_layout()
    _save(fig, os.path.join(figdir, "overview"), args.formats, args.dpi)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    validate_args(args)
    os.makedirs(args.figdir, exist_ok=True)

    summary = pd.read_csv(os.path.join(args.results, "activity_density_summary.csv"))
    regions_path = os.path.join(args.results, "activity_density_regions.csv")
    regions = (
        pd.read_csv(regions_path) if os.path.exists(regions_path) else pd.DataFrame()
    )
    plot_overview(summary, regions, args.figdir, args)
    log(f"[overview] {len(summary)} pairs, {len(regions)} regions")
    if args.only_overview:
        return 0

    # Reproduce the run's own settings unless overridden, so the figures show the
    # cell set the test used.
    min_t7 = args.min_t7
    if min_t7 is None:
        min_t7 = float(summary["min_t7"].iloc[0]) if "min_t7" in summary else 0.0
    transform = args.mark_transform or (
        str(summary["mark_transform"].iloc[0]) if "mark_transform" in summary else "rank"
    )
    matrix_key = (
        str(summary["matrix"].iloc[0]) if "matrix" in summary else "activity_normalized"
    )

    chosen = summary.copy()
    if args.cres:
        chosen = chosen[chosen["cre"].isin(args.cres)]
    elif not args.all_pairs:
        significant = chosen[chosen.get("significant", False)]
        ranked = chosen.sort_values(["p_fwer", "max_z"], ascending=[True, False])
        chosen = pd.concat([significant, ranked.head(args.top)]).drop_duplicates(
            subset=["cre", "section"]
        )
    if chosen.empty:
        log("[pairs] nothing to draw")
        return 0
    log(f"[pairs] drawing {len(chosen)} pairs (min_t7={min_t7}, transform={transform})")

    values, npz_obs_names, cre_names = load_activity(
        args.activity_npz, matrix_key, sorted(chosen["cre"].unique()), None
    )
    metadata = read_obs_metadata(args.h5ad)
    order = pd.Index(npz_obs_names).get_indexer(metadata["obs_name"].to_numpy())
    if (order < 0).any():
        raise ValueError("the h5ad and the activity matrix disagree on cell names")
    values = values[order]
    positions = {name: index for index, name in enumerate(cre_names)}

    vocabulary = np.unique(metadata["subclass"].to_numpy())
    admissible = build_admissible(args.t7_totals, cre_names, vocabulary, min_t7)
    finite = np.isfinite(values).any(axis=1)
    pixel_size = args.pixel_size or min(args.bandwidths) / 2.0
    contexts = build_section_contexts(
        metadata,
        finite,
        sorted(chosen["section"].unique()),
        pixel_size,
        vocabulary,
    )
    bandwidths = tuple(sorted(args.bandwidths))

    drawn = skipped = 0
    for _, row in chosen.iterrows():
        cre, section = str(row["cre"]), str(row["section"])
        stem = os.path.join(args.figdir, f"{cre}_{section}_activity_density")
        if not args.overwrite and os.path.exists(f"{stem}.{args.formats[0]}"):
            continue
        context = contexts[section]
        cre_index = positions[cre]
        case = build_case(
            context,
            cre_index,
            values=values,
            admissible=admissible,
            bandwidths=bandwidths,
            min_effective_cells=args.min_effective_cells,
            max_cell_distance_factor=args.tissue_mask_bandwidths,
            min_cells=args.min_case_cells,
        )
        if case is None:
            skipped += 1
            continue

        raw = np.ascontiguousarray(
            values[context.rows[case.cells], cre_index], dtype=np.float32
        )
        marks = transform_marks(raw, transform)
        geometries = list(case.geometries)
        z = z_stack(
            case.grid, geometries, marks,
            mean=float(np.mean(marks)), variance=float(np.var(marks)),
        )
        best = int(
            np.argmax([z[i][g.valid].max() for i, g in enumerate(geometries)])
        )
        cells = pd.DataFrame(
            {
                "x": metadata.loc[context.rows, "x"].to_numpy(),
                "y": metadata.loc[context.rows, "y"].to_numpy(),
                "activity": values[context.rows, cre_index],
                "admissible": np.isin(np.arange(context.n_cells), case.cells),
            }
        )
        bundle: dict[str, object] = {
            "cre": cre,
            "section": section,
            "grid": case.grid,
            "bandwidths": np.array([g.bandwidth for g in geometries]),
            "selected_index": best,
            "z": z,
            "activity_density": np.stack(
                [density_surface(case.grid, g, raw) for g in geometries]
            ),
            "valid": np.stack([g.valid for g in geometries]),
            "cells": cells,
            "p_fwer": float(row["p_fwer"]),
            "q_global_bh": float(row["q_global_bh"]),
        }
        # A delineated pair already has its regions and bootstrap band on disk;
        # reuse them rather than recomputing what permutations produced.
        surface_path = os.path.join(args.results, "surfaces", f"{cre}_{section}.npz")
        if os.path.exists(surface_path):
            with np.load(surface_path, allow_pickle=True) as stored:
                bundle["region_labels"] = stored["region_labels"]
                bundle["inclusion_probability"] = stored["inclusion_probability"]

        plot_pair(
            bundle,
            regions[(regions.get("cre") == cre) & (regions.get("section") == section)]
            if not regions.empty
            else pd.DataFrame(),
            stem,
            args,
        )
        drawn += 1
    log(f"[pairs] drew {drawn} figures into {args.figdir}; {skipped} had no usable case")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
