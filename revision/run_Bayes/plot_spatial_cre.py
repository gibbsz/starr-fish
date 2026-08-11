"""Render one spatial map for one cCRE, from a fit already on disk.

    python revision/run_Bayes/plot_spatial_cre.py \
        --fit-dir revision/Bayes_OldData/bayesian \
        --cre CRE007 --mode activity_posterior_normalized \
        --outdir revision/Bayes_OldData/visualization

Nothing here refits or resamples: the posterior draws are read from ``--fit-dir``
and the per-cell quantities are reconstructed deterministically from them.

Why this is minutes rather than half an hour
--------------------------------------------
The full copy-number matrix is 408,621 x 389, but a map of one cCRE needs only
that column -- plus the negative controls, whose T7 totals decide which cell types
have a background reference at all. Every (cell, cCRE) pair is conditionally
independent given the parameters, and the collapse is keyed on
``(cell type, cCRE, t7, cre)``, so restricting the columns is *exact*: the numbers
are identical to slicing them out of the full matrix.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from baystarrfish._log import log
from baystarrfish.data import CountData
from baystarrfish.inference.copy_number import (
    DEFAULT_CONTROL_T7_THRESHOLD,
    infer_copy_number_from_fit,
)
from baystarrfish.plotting import SPATIAL_MODES, plot_spatial

#: What each mode needs computed. Modes absent from here read raw counts only.
_MODE_REQUIREMENTS: dict[str, dict[str, bool]] = {
    "copy_number": {},
    "activity": {},
    "activity_posterior": {"return_activity": True},
    "activity_posterior_normalized": {"return_activity_normalized": True},
}


def _subset_columns(data: CountData, keep: Sequence[str]) -> CountData:
    """A CountData restricted to ``keep``, in that order."""
    names = [str(name) for name in data.cre_names]
    missing = [name for name in keep if name not in names]
    if missing:
        raise ValueError(f"cCRE(s) {missing} are not in the data ({len(names)} columns)")
    index = np.array([names.index(str(name)) for name in keep], dtype=np.int64)
    return dataclasses.replace(
        data,
        t7=np.asarray(data.t7)[:, index],
        cre=np.asarray(data.cre)[:, index],
        lib_size_log=np.asarray(data.lib_size_log)[index],
        cre_names=[str(name) for name in keep],
        negative_control_mask=(
            None if data.negative_control_mask is None
            else np.asarray(data.negative_control_mask)[index]
        ),
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--fit-dir", type=Path, required=True)
    parser.add_argument("--cre", required=True, help="e.g. CRE007")
    parser.add_argument("--mode", default="activity_posterior_normalized",
                        choices=[m for m in SPATIAL_MODES if m != "celltype"])
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--h5ad", type=Path, default=None,
                        help="defaults to the path recorded in the fit manifest")
    parser.add_argument("--max-draws", type=int, default=200,
                        help="thin the posterior; 200 is within Monte Carlo error "
                             "of the full 1,000 for a posterior mean")
    parser.add_argument("--negative-control-t7-threshold", type=float,
                        default=DEFAULT_CONTROL_T7_THRESHOLD)
    parser.add_argument("--celltypes", nargs="+", default=None,
                        help="give these cell types their own colours on top of "
                             "the value map; every other cell keeps the default "
                             "ramp colour and size still tracks the value")
    parser.add_argument("--exclude-celltypes", nargs="+", default=None,
                        help="withhold the value for these cell types; their "
                             "cells stay as grey background and do not set the "
                             "colour scale")
    parser.add_argument("--level", choices=["subclass", "class"], default="subclass",
                        help="granularity --celltypes names are matched at")
    parser.add_argument("--vmin", type=float, default=None)
    parser.add_argument("--vmax", type=float, default=None)
    parser.add_argument("--log", action="store_true", help="log1p the values first")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--suffix", default="", help="appended to the output stem")
    args = parser.parse_args(argv)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    manifest = json.loads((args.fit_dir / "run_manifest.json").read_text())
    config = manifest.get("config", {})
    controls = [str(c) for c in config.get("annotated_negative_control_cre", [])]

    data = CountData.from_h5ad(
        args.h5ad or Path(config["input"]["path"]),
        section=manifest.get("section", "all"),
        negative_control_mode=manifest.get("negative_control_mode", "ordinary"),
    )

    requirements = _MODE_REQUIREMENTS.get(args.mode, {})
    matrix = None
    if args.mode in _MODE_REQUIREMENTS:
        # Only the columns the map and its reference depend on.
        needed = [args.cre] + [c for c in controls if c != args.cre]
        subset = _subset_columns(data, needed)
        log(f"[plot] reconstructing {len(needed)} of {data.n_cre} cCRE columns "
            f"({args.cre} + {len(needed) - 1} controls)")
        matrix = infer_copy_number_from_fit(
            subset, args.fit_dir, max_draws=args.max_draws,
            negative_control_t7_threshold=args.negative_control_t7_threshold,
            **requirements,
        )
        data = subset

    # Highlighting a name that matches nothing is visible in the figure (a colour
    # simply never appears); a typo in --celltypes is not worth aborting a run
    # over. It is still worth reporting.
    labels = np.asarray(
        data.subclass if args.level == "subclass" else data.class_
    ).astype(str)
    if args.celltypes:
        unknown = sorted(set(args.celltypes) - set(np.unique(labels)))
        if unknown:
            log(f"[plot] WARNING highlighted cell type(s) {unknown} are absent at "
                f"level={args.level!r}; nothing will be drawn for them")
    if args.celltypes and args.exclude_celltypes:
        both = sorted(set(args.celltypes) & set(args.exclude_celltypes))
        if both:
            raise SystemExit(
                f"cell type(s) {both} are both highlighted and excluded; "
                "exclusion wins, so this is almost certainly a mistake"
            )

    fig = plot_spatial(
        data, args.mode, cre=args.cre, copies=matrix, activity=matrix,
        celltypes=args.celltypes, exclude_celltypes=args.exclude_celltypes,
        level=args.level, vmin=args.vmin, vmax=args.vmax, log=args.log,
    )
    if args.exclude_celltypes:
        dropped = int(np.isin(labels, [str(c) for c in args.exclude_celltypes]).sum())
        log(f"[plot] excluded {len(args.exclude_celltypes)} cell type(s), "
            f"{dropped:,} of {len(labels):,} cells shown as grey without a value")

    values = None
    if matrix is not None and args.mode == "activity_posterior_normalized":
        values = matrix.matrix("activity_normalized")[:, 0]
        drawn = np.isfinite(values) & (
            (np.asarray(data.t7)[:, 0] > 0) | (np.asarray(data.cre)[:, 0] > 0)
        )
        log(f"[plot] {args.cre}: {int(drawn.sum()):,} cells with evidence and a "
            f"reference; {int((~np.isfinite(values)).sum()):,} cells have no "
            "eligible background (drawn grey)")
        if drawn.any():
            shown = values[drawn]
            log(f"[plot]   fold change over control mean: median {np.median(shown):.3f}, "
                f"90th pct {np.percentile(shown, 90):.3f}, max {shown.max():.3f}; "
                f"{100 * (shown > 1).mean():.1f}% above background")

    args.outdir.mkdir(parents=True, exist_ok=True)
    by_type = "_bycelltype" if args.celltypes else ""
    trimmed = "_subset" if args.exclude_celltypes else ""
    stem = f"{args.cre}_{args.mode}_spatial{by_type}{trimmed}{args.suffix}"
    for extension in ("png", "pdf"):
        path = args.outdir / f"{stem}.{extension}"
        fig.savefig(path, dpi=args.dpi, facecolor="black", bbox_inches="tight")
        log(f"[plot] wrote {path}")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
