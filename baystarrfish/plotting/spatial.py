"""Spatial maps of the tissue section, on black.

One visual grammar, four things to look at:

``celltype``
    which cells are which type -- a categorical colour per highlighted type
``cre``
    raw enhancer-channel counts for one cCRE (or aggregated across cCREs)
``t7``
    raw constitutive-channel counts, the infection readout
``copy_number``
    the inferred latent copies ``E[k | obs]`` from
    :mod:`baystarrfish.inference.copy_number`
``activity``
    per-cell inferred activity, ``cCRE counts / E[k | obs]`` -- the enhancer
    output normalised by how much virus the model thinks the cell received

Every mode draws *all* cells first as small grey dots, so the tissue outline is
always visible and a sparse signal is never mistaken for a sparse section. The
three value modes then overlay the cells carrying signal, encoding magnitude
three ways at once -- dot size, opacity and colour -- because on a black field a
single channel is hard to read at the dot sizes a 400,000-cell section forces.

Conventions (black facecolor, grey background dots, size 5-30, alpha ramp,
equal aspect, no ticks, the 7-dot scale bar) follow
``STARRFISH.utils.STARRFISH.plot_gene`` so figures sit alongside the existing
ones. Two deliberate departures, both noted where they occur: value cells are
drawn in ascending order so the strongest are never buried, and the scale bar
uses the same linear size ramp as the plot rather than a cubic one.

Each mode gets its own hue so the three value maps are distinguishable at a
glance even when cropped out of their titles.
"""

from __future__ import annotations

from typing import Literal, Mapping, Sequence

import numpy as np

__all__ = [
    "MODE_COLORS",
    "SPATIAL_MODES",
    "SpatialMode",
    "evidence_mask",
    "plot_spatial",
    "spatial_values",
]

SpatialMode = Literal["celltype", "cre", "t7", "copy_number", "activity"]

SPATIAL_MODES: tuple[str, ...] = (
    "celltype", "cre", "t7", "copy_number", "activity",
)

#: Ramp endpoint per value mode. Distinct hues, all reading as "hot" against
#: black, all reaching white at zero so the low end fades into the grey field.
MODE_COLORS: dict[str, str] = {
    "cre": "#FF6B6B",        # enhancer channel -- the plot_gene red
    "t7": "#4DD0E1",         # constitutive channel -- cyan, clearly not the cCRE
    "copy_number": "#FFC857",  # inferred latent -- amber, clearly not a measurement
    "activity": "#9CCC65",     # derived quantity -- green, neither channel
}

_MODE_LABELS = {
    "cre": "cCRE counts",
    "t7": "T7 counts",
    "copy_number": "inferred AAV copies  E[k | obs]",
    "activity": "inferred activity  cCRE / E[k | obs]",
}


def _ramp(color: str):
    """White-to-``color`` colormap, matching plot_gene's custom_cmap."""
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list("baystarrfish", ["#FFFFFF", color])


def _select(matrix, names, cre, aggregate, what):
    """One column by name, or an aggregate across all of them."""
    matrix = np.asarray(matrix)
    if cre is not None:
        if aggregate is not None:
            raise ValueError("pass either cre= or aggregate=, not both")
        if cre not in names:
            raise ValueError(f"unknown cCRE {cre!r}")
        return np.asarray(matrix[:, names.index(cre)], dtype=np.float64)
    if aggregate is None:
        raise ValueError(
            f"mode={what!r} needs either cre='CRE123' or aggregate='sum'|'mean'"
        )
    if aggregate == "sum":
        return np.asarray(matrix.sum(axis=1), dtype=np.float64)
    if aggregate == "mean":
        return np.asarray(matrix.mean(axis=1), dtype=np.float64)
    raise ValueError(f"unknown aggregate {aggregate!r}; expected 'sum' or 'mean'")


def _copies_matrix(data, copies, mode):
    if copies is None:
        raise ValueError(
            f"mode={mode!r} needs copies=, a CopyNumberMatrix or an "
            "(n_cells, n_cre) array from baystarrfish.infer_copy_number"
        )
    matrix = np.asarray(getattr(copies, "copies", copies))
    names = [str(n) for n in getattr(copies, "cre_names", data.cre_names)]
    if matrix.shape[0] != data.n_cells:
        raise ValueError(f"values have {matrix.shape[0]} rows for {data.n_cells} cells")
    return matrix, names


def spatial_values(
    data,
    mode: SpatialMode,
    *,
    cre: str | None = None,
    aggregate: Literal["sum", "mean"] | None = None,
    copies=None,
) -> np.ndarray:
    """The per-cell quantity a value mode draws, before any scaling.

    Separated from the drawing so the numbers can be checked, reused, or plotted
    by something else entirely.

    ``activity`` is ``cCRE counts / E[k | obs]``. The model says
    ``E[cre | k] = k * gamma``, so this is the per-cell moment estimator of the
    activity ``gamma`` -- the enhancer output per virus copy, which is what makes
    it comparable between a cell that received one copy and one that received
    thirty. It cannot blow up: ``k = 0`` is a point mass forcing both channels to
    zero, so any cell with ``cre > 0`` has ``P(k = 0 | obs) = 0`` and therefore
    ``E[k] >= 1``. The ratio is bounded above by the raw count.
    """
    if mode not in SPATIAL_MODES:
        raise ValueError(f"unknown mode {mode!r}; expected one of {list(SPATIAL_MODES)}")
    if mode == "celltype":
        raise ValueError("mode='celltype' is categorical and has no per-cell value")

    if mode == "activity":
        numerator = _select(data.cre, list(data.cre_names), cre, aggregate, mode)
        matrix, names = _copies_matrix(data, copies, mode)
        denominator = _select(matrix, names, cre, aggregate, mode)
        # Guaranteed >= 1 wherever the numerator is positive (see the docstring),
        # so this floor never binds on a matrix from this model. It is here so a
        # hand-supplied denominator cannot silently produce an infinity.
        return numerator / np.maximum(denominator, 1e-12)

    if mode == "copy_number":
        matrix, names = _copies_matrix(data, copies, mode)
    else:
        matrix = data.cre if mode == "cre" else data.t7
        names = [str(n) for n in data.cre_names]
        if np.asarray(matrix).shape[0] != data.n_cells:
            raise ValueError(
                f"values have {np.asarray(matrix).shape[0]} rows for {data.n_cells} cells"
            )
    return _select(matrix, names, cre, aggregate, mode)


def evidence_mask(data, *, cre: str | None, aggregate: str | None) -> np.ndarray:
    """Cells whose counts actually informed the estimate for this cCRE.

    ``E[k | obs]`` is defined for every cell, but where both channels read zero
    it collapses to the cell-type baseline ``E[k | 0, 0]`` -- identical for every
    cell of a subclass, and carrying no cell-specific information. On the real
    section that is 99.3% of the matrix, so drawing it paints the whole tissue
    one colour and hides the 0.7% that is actually measured.
    """
    names = list(data.cre_names)
    if cre is not None:
        column = names.index(cre)
        return (np.asarray(data.t7)[:, column] > 0) | (np.asarray(data.cre)[:, column] > 0)
    return (np.asarray(data.t7).sum(axis=1) > 0) | (np.asarray(data.cre).sum(axis=1) > 0)


def _coordinates(data, transpose: int, flipx: int, flipy: int) -> np.ndarray:
    if data.spatial is None:
        raise ValueError(
            "this CountData carries no spatial coordinates; the input needs "
            "obsm['X_spatial']"
        )
    return np.asarray(data.spatial)[:, ::transpose] * [flipx, flipy]


def _region_mask(coords, x_region, y_region) -> np.ndarray:
    keep = np.ones(len(coords), dtype=bool)
    if x_region is not None:
        keep &= (coords[:, 0] > x_region[0]) & (coords[:, 0] < x_region[1])
    if y_region is not None:
        keep &= (coords[:, 1] > y_region[0]) & (coords[:, 1] < y_region[1])
    if not keep.any():
        raise ValueError("the requested region contains no cells")
    return keep


def _normalise(values, vmin, vmax, log, subset=None):
    """Scale to [0, 1], with a robust upper end unless one is given.

    ``subset`` restricts which values set the upper end; everything is still
    scaled, so cells outside the subset simply clip.
    """
    values = np.asarray(values, dtype=np.float64)
    if log:
        values = np.log1p(np.maximum(values, 0.0))
    reference = values if subset is None else values[np.asarray(subset, dtype=bool)]
    # Zero, not values.min(): every mode here is a count or an expected count,
    # so zero is the meaningful floor. Anchoring at the observed minimum would
    # make the whole ramp depend on whichever cell happened to be dimmest.
    low = 0.0 if vmin is None else float(vmin)
    if vmax is None:
        positive = reference[reference > low]
        # Counts are heavy-tailed: one 300-count cell would otherwise flatten
        # every other cell to invisibility. The 99th percentile keeps the
        # dynamic range usable; pass vmax= to override.
        high = float(np.percentile(positive, 99.0)) if positive.size >= 100 else (
            float(np.nanmax(reference)) if reference.size else low + 1.0
        )
    else:
        high = float(vmax)
    if not np.isfinite(high) or high <= low:
        high = low + 1.0
    return np.clip((values - low) / (high - low), 0.0, 1.0), low, high


def _scale_bar(ax, low, high, color, size_min, size_max, label, n=7):
    """Seven dots showing the size/alpha/colour ramp, min and max labelled."""
    ramp = _ramp(color)
    fractions = np.linspace(0.0, 1.0, n)
    # Linear, exactly as the main plot scales size. plot_gene cubes this, which
    # makes the legend disagree with the figure it documents.
    sizes = size_min + fractions * (size_max - size_min)
    spacing = 0.08
    ax.set_facecolor("black")
    ax.axis("off")
    for i, (fraction, size) in enumerate(zip(fractions, sizes)):
        ax.scatter(i * spacing, 0.25, s=size, alpha=max(fraction, 0.15),
                   color=ramp(fraction), edgecolors="none")
    span = (n - 1) * spacing
    ax.text(-0.3, 0.25, f"{low:.3g}", va="center", ha="center", color="white", fontsize=8)
    ax.text(span + 0.3, 0.25, f"{high:.3g}", va="center", ha="center",
            color="white", fontsize=8)
    ax.text(span / 2, 0.45, label, ha="center", va="top", color="white", fontsize=9)
    ax.set_xlim(-0.35, span + 0.35)
    ax.set_ylim(0, 1.0)


def plot_spatial(
    data,
    mode: SpatialMode = "cre",
    *,
    cre: str | None = None,
    aggregate: Literal["sum", "mean"] | None = None,
    copies=None,
    celltypes: Sequence[str] | None = None,
    level: Literal["subclass", "class"] = "subclass",
    palette: Mapping[str, str] | Sequence[str] | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    min_value: float | None = None,
    log: bool = False,
    size_background: float = 3.0,
    size_min: float = 5.0,
    size_max: float = 30.0,
    background_alpha: float = 0.7,
    transpose: int = 1,
    flipx: int = 1,
    flipy: int = 1,
    x_region: tuple[float, float] | None = None,
    y_region: tuple[float, float] | None = None,
    title: str | None = None,
    show_scale_bar: bool = True,
    show_legend: bool = True,
    figsize: tuple[float, float] = (12.0, 10.0),
    ax=None,
    rasterized: bool = True,
):
    """Draw one spatial map and return its ``Figure``.

    Parameters
    ----------
    data : CountData
        Must carry ``spatial``; ``CountData.from_h5ad`` fills it from
        ``obsm['X_spatial']``.
    mode
        ``'celltype'``, ``'cre'``, ``'t7'`` or ``'copy_number'``.
    cre, aggregate
        Value modes need exactly one: a single cCRE by name, or ``'sum'`` /
        ``'mean'`` across all of them.
    copies
        For ``mode='copy_number'`` and ``mode='activity'``: a
        :class:`~baystarrfish.inference.copy_number.CopyNumberMatrix` or a plain
        ``(n_cells, n_cre)`` array.
    celltypes
        Which types to colour in ``'celltype'`` mode. Default: every type, which
        for 328 subclasses is a smear -- name the handful you care about.
    vmin, vmax
        Value-scale ends. ``vmax=None`` uses the 99th percentile of the values
        above ``vmin``, because counts are heavy-tailed enough that the raw
        maximum flattens everything else.
    min_value
        Overlay only cells above this value; the rest stay grey. Default: any
        positive value for ``'cre'`` / ``'t7'``, and for ``'copy_number'`` the
        cells with a nonzero read in either channel -- and the same for
        ``'activity'`` -- since everywhere else the estimate is the cell-type
        baseline rather than a measurement (see :func:`evidence_mask`). An
        evidence-bearing cell with no cCRE transcript draws at zero activity,
        which is a measurement (infected but silent), not missing data.
    log
        Scale by ``log1p`` before normalising. Useful for copy number, which
        spans several orders of magnitude.
    ax
        Draw into an existing axes instead of building a figure. The scale bar
        and legend are then the caller's problem.
    """
    import matplotlib.pyplot as plt

    if mode not in SPATIAL_MODES:
        raise ValueError(f"unknown mode {mode!r}; expected one of {list(SPATIAL_MODES)}")
    if level not in {"subclass", "class"}:
        raise ValueError(f"unknown level {level!r}; expected 'subclass' or 'class'")

    coords = _coordinates(data, transpose, flipx, flipy)
    keep = _region_mask(coords, x_region, y_region)
    coords = coords[keep]

    external_ax = ax is not None
    if external_ax:
        fig = ax.figure
        ax_bar = None
    else:
        fig = plt.figure(figsize=figsize, facecolor="black")
        if show_scale_bar and mode != "celltype":
            # The bar sits in a narrow centred column. Spanning the full width
            # strands the dots in the middle with the labels at the far edges;
            # putting it in a side column pushes the map off-centre.
            grid = fig.add_gridspec(
                2, 3, height_ratios=[0.93, 0.07],
                width_ratios=[0.3, 0.4, 0.3], hspace=0.03,
            )
            ax, ax_bar = fig.add_subplot(grid[0, :]), fig.add_subplot(grid[1, 1])
        else:
            ax, ax_bar = fig.add_subplot(111), None
    ax.set_facecolor("black")

    # Every mode: all cells, grey, underneath. The tissue outline must be
    # visible even where nothing is detected.
    ax.scatter(coords[:, 0], coords[:, 1], c="grey", s=size_background, marker=".",
               alpha=background_alpha, rasterized=rasterized, edgecolors="none")

    if mode == "celltype":
        labels = np.asarray(data.subclass if level == "subclass" else data.class_)[keep]
        wanted = list(dict.fromkeys(celltypes)) if celltypes is not None else sorted(
            np.unique(labels)
        )
        colors = _celltype_colors(wanted, palette)
        for name in wanted:
            hit = labels == name
            if not hit.any():
                continue
            # color=, not c=: matplotlib warns that a single RGB sequence given
            # to c= may be read as per-point values when the lengths collide.
            ax.scatter(coords[hit, 0], coords[hit, 1], color=colors[name],
                       s=size_max, marker=".", label=str(name),
                       rasterized=rasterized, edgecolors="none")
        if show_legend and wanted:
            legend = ax.legend(
                fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1.0),
                frameon=False, markerscale=1.5,
            ) if len(wanted) > 5 else ax.legend(
                fontsize=10, loc="lower right", frameon=False, markerscale=1.5
            )
            for text in legend.get_texts():
                text.set_color("white")
        label = f"cell type ({level})"
    else:
        values = spatial_values(data, mode, cre=cre, aggregate=aggregate,
                                copies=copies)[keep]
        if min_value is not None:
            visible = values > float(min_value)
        elif mode in {"copy_number", "activity"}:
            visible = evidence_mask(data, cre=cre, aggregate=aggregate)[keep]
        else:
            visible = values > (0.0 if vmin is None else float(vmin))
        # Scale over the cells actually drawn. Including the ones left grey would
        # set the ramp from a population that never appears -- for copy number
        # that is 92% of the section sitting at the cell-type baseline, which
        # pushes every drawn cell past the top of the ramp and flattens the map.
        fraction, low, high = _normalise(values, vmin, vmax, log, subset=visible)
        color = MODE_COLORS[mode]
        sizes = size_min + fraction * (size_max - size_min)
        # Ascending, so the strongest cells land on top instead of being buried
        # under whichever neighbour happened to come later in the array.
        order = np.argsort(fraction, kind="stable")
        drawn = order[visible[order]]
        ax.scatter(
            coords[drawn, 0], coords[drawn, 1],
            c=_ramp(color)(fraction[drawn]), s=sizes[drawn],
            alpha=np.clip(fraction[drawn], 0.15, 1.0),
            rasterized=rasterized, edgecolors="none",
        )
        label = _MODE_LABELS[mode] + (" (log1p)" if log else "")
        if ax_bar is not None:
            _scale_bar(ax_bar, low, high, color, size_min, size_max, label)

    ax.set_aspect("equal")
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    if title is None:
        detail = cre if cre else (f"{aggregate} over {data.n_cre} cCREs"
                                  if aggregate else "")
        title = f"{label}{' - ' + detail if detail else ''}"
    if title:
        ax.set_title(title, color="white", fontsize=16)
    if not external_ax:
        fig.patch.set_facecolor("black")
    return fig


def _celltype_colors(names, palette) -> dict[str, str]:
    """Resolve a colour per cell type, cycling whatever palette is given."""
    import matplotlib.pyplot as plt

    if isinstance(palette, Mapping):
        fallback = list(palette.values()) or ["#fabed4"]
        return {
            name: palette.get(name, fallback[i % len(fallback)])
            for i, name in enumerate(names)
        }
    if palette is None:
        palette = list(plt.get_cmap("tab20").colors)
    palette = list(palette)
    return {name: palette[i % len(palette)] for i, name in enumerate(names)}
