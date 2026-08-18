#!/usr/bin/env python3
"""Kernel activity-density surfaces, their random-labelling null, and the
simultaneous excursion regions that fall out of it.

The statistical unit is one (cCRE, section) pair. Cells sit at fixed positions
and carry a mark ``a_i`` -- the per-cell Gamma-conjugate activity posterior mean
from ``revision/Bayes_OldData/copy_number/activity_normalized.npz``. The surface
tested is the kernel-weighted *local mean* of the mark

    R(u) = sum_i w_i(u) a_i,     w_i(u) = K_h(u - x_i) / sum_j K_h(u - x_j)

rather than the raw activity density ``sum_i K_h(u - x_i) a_i``. The denominator
is invariant under the null, so the two give identical p-values, but ``R`` is on
the same 1.0-is-baseline scale as the plotted activity and does not simply track
where cells are dense.

The null is random labelling: the observed marks are reassigned to cells,
holding the positions fixed. Under it the finite-population moments of ``R(u)``
are exact,

    E0[R(u)]   = mean(a)
    Var0[R(u)] = var(a) * n/(n-1) * (sum_i w_i(u)^2 - 1/n)

so the standardised surface ``z = (R - E0) / sqrt(Var0)`` -- Getis-Ord Gi* with
Gaussian weights -- costs no permutations. Permutations are still needed for
p-values: the activity distribution is heavy-tailed (median 1.45, q99 23.6) and
the normal approximation fails in exactly the upper tail that matters.

``sum_i w_i(u)^2`` is one extra convolution rather than a second pass over
cells: the square of a normalised Gaussian of width sigma is a normalised
Gaussian of width sigma/sqrt(2), scaled by 1/(4 pi sigma^2).

Regions come from the same permutations. With ``c_alpha`` the (1-alpha) quantile
of the permutation maxima taken over every valid pixel *and* every bandwidth,

    E(alpha) = { u : z(u) >= c_alpha }

contains no null pixel with probability at least 1-alpha, so the whole set is
significant simultaneously -- no per-pixel correction and no correction for
having searched. ``E(alpha)`` is empty exactly when the global max-test fails,
which is why the test and the region are one object.

This module holds no I/O and no argument parsing; see
``run_activity_density_hotspots.py``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import ndimage

__all__ = [
    "MARK_TRANSFORMS",
    "BandwidthGeometry",
    "Region",
    "SectionGrid",
    "StabilisedSurface",
    "build_geometry",
    "build_grid",
    "density_surface",
    "extract_regions",
    "mark_shuffler",
    "null_maxima",
    "null_z_stack",
    "permutation_p_value",
    "stabilise",
    "transform_marks",
    "z_stack",
]

MARK_TRANSFORMS = ("rank", "log", "none")

# The Gaussian is truncated at this many standard deviations by every
# convolution here; keep it in one place so the kernel footprint used to grow
# the exclusion zone during step-down matches the smoothing that produced it.
TRUNCATE = 4.0


# --------------------------------------------------------------------------- #
# geometry
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class SectionGrid:
    """Cell -> pixel binning for one section, shared by every cCRE.

    Built once and reused across cCREs, bandwidths and permutations: the
    positions never change, only the marks attached to them.
    """

    section: str
    pixel_index: np.ndarray  # (n_cells,) int64, row-major into ``shape``
    shape: tuple[int, int]  # (n_rows, n_cols) == (y, x)
    origin: tuple[float, float]  # (x_min, y_min) of the pixel-(0, 0) corner
    pixel_size: float

    @property
    def n_cells(self) -> int:
        return int(self.pixel_index.size)

    @property
    def n_pixels(self) -> int:
        return int(self.shape[0] * self.shape[1])

    def pixel_centres(self) -> tuple[np.ndarray, np.ndarray]:
        """``(x, y)`` centre coordinate of every pixel, each of shape ``shape``."""
        rows, cols = self.shape
        xs = self.origin[0] + (np.arange(cols) + 0.5) * self.pixel_size
        ys = self.origin[1] + (np.arange(rows) + 0.5) * self.pixel_size
        return np.meshgrid(xs, ys)

    def bin_marks(self, marks: np.ndarray) -> np.ndarray:
        """Sum ``marks`` into pixels, returned already reshaped to the grid."""
        if marks.shape != (self.n_cells,):
            raise ValueError(
                f"marks has shape {marks.shape}, expected ({self.n_cells},)"
            )
        binned = np.bincount(
            self.pixel_index, weights=marks, minlength=self.n_pixels
        )
        return binned.reshape(self.shape).astype(np.float32, copy=False)

    def cells_in_mask(self, mask: np.ndarray) -> np.ndarray:
        """Indices of the cells whose pixel is set in a grid-shaped ``mask``."""
        if mask.shape != self.shape:
            raise ValueError(f"mask has shape {mask.shape}, expected {self.shape}")
        return np.flatnonzero(mask.reshape(-1)[self.pixel_index])


def build_grid(
    x: np.ndarray, y: np.ndarray, *, section: str, pixel_size: float
) -> SectionGrid:
    """Lay a regular lattice over one section's bounding box.

    One pixel size serves every bandwidth -- only the smoothing sigma changes --
    so the cell->pixel map is computed once per section.
    """
    if pixel_size <= 0:
        raise ValueError(f"pixel_size must be positive, got {pixel_size}")
    if x.shape != y.shape or x.ndim != 1:
        raise ValueError("x and y must be 1-D arrays of the same length")
    if x.size == 0:
        raise ValueError(f"section {section!r} has no cells")

    x_min = float(np.min(x))
    y_min = float(np.min(y))
    cols = int(np.floor((float(np.max(x)) - x_min) / pixel_size)) + 1
    rows = int(np.floor((float(np.max(y)) - y_min) / pixel_size)) + 1

    col = np.clip(((x - x_min) / pixel_size).astype(np.int64), 0, cols - 1)
    row = np.clip(((y - y_min) / pixel_size).astype(np.int64), 0, rows - 1)
    return SectionGrid(
        section=section,
        pixel_index=row * cols + col,
        shape=(rows, cols),
        origin=(x_min, y_min),
        pixel_size=float(pixel_size),
    )


@dataclass(frozen=True)
class BandwidthGeometry:
    """Everything about a (section, bandwidth) that does not depend on the mark.

    ``variance_factor`` is the mark-free half of ``Var0``; multiplying it by the
    population variance of the marks and taking the square root gives the null
    standard deviation of ``R(u)``. It is zero outside ``valid``.
    """

    bandwidth: float
    sigma_px: float
    cell_density: np.ndarray  # sum_i K(u - x_i), grid-shaped float32
    n_effective: np.ndarray  # (sum_i k_i)^2 / sum_i k_i^2, grid-shaped float32
    valid: np.ndarray  # bool, grid-shaped
    variance_factor: np.ndarray  # (1/n_eff - 1/n) * n/(n-1), grid-shaped float32

    @property
    def n_valid(self) -> int:
        return int(np.count_nonzero(self.valid))

    @property
    def kernel_radius_px(self) -> int:
        """How far one cell's influence reaches, in pixels."""
        return int(math.ceil(TRUNCATE * self.sigma_px))


def build_geometry(
    grid: SectionGrid,
    *,
    bandwidth: float,
    min_effective_cells: float,
    max_cell_distance: float,
) -> BandwidthGeometry:
    """Precompute the null's mark-free terms for one bandwidth.

    The valid mask needs *both* of its conditions and neither is redundant.

    ``min_effective_cells`` bounds the effective sample size
    ``n_eff = (sum k)^2 / sum k^2``, the quantity the null variance actually
    depends on -- it keeps pixels whose local mean rests on too few cells out of
    the search domain.

    ``max_cell_distance`` is the tissue mask, and it cannot be replaced by an
    ``n_eff`` threshold: far outside the tissue every cell is roughly equidistant,
    so the weights are near-uniform and ``n_eff`` grows toward the whole section
    rather than shrinking. Requiring a real cell within ``max_cell_distance``
    of the pixel is the condition that actually means "inside the tissue".
    """
    if bandwidth <= 0:
        raise ValueError(f"bandwidth must be positive, got {bandwidth}")
    sigma_px = bandwidth / grid.pixel_size
    if sigma_px < 1.0:
        raise ValueError(
            f"bandwidth {bandwidth} is under one pixel ({grid.pixel_size}); "
            "the Gaussian would be aliased -- lower --pixel-size"
        )
    n_cells = grid.n_cells
    if n_cells < 2:
        raise ValueError("need at least two cells to standardise the surface")

    counts = grid.bin_marks(np.ones(n_cells, dtype=np.float32))
    density = ndimage.gaussian_filter(
        counts, sigma_px, mode="constant", truncate=TRUNCATE
    )
    # (K_sigma)^2 == K_{sigma/sqrt(2)} / (4 pi sigma^2) for a normalised Gaussian.
    sum_sq = ndimage.gaussian_filter(
        counts, sigma_px / math.sqrt(2.0), mode="constant", truncate=TRUNCATE
    ) / (4.0 * math.pi * sigma_px * sigma_px)

    with np.errstate(divide="ignore", invalid="ignore"):
        n_effective = np.where(sum_sq > 0, density * density / sum_sq, 0.0)
    occupied = counts > 0
    distance = ndimage.distance_transform_edt(~occupied) * grid.pixel_size
    valid = (
        (n_effective >= min_effective_cells)
        & (distance <= max_cell_distance)
        & (density > 0)
    )
    if not valid.any():
        raise ValueError(
            f"no pixel is both within {max_cell_distance} of a cell and backed by "
            f"{min_effective_cells} effective cells at bandwidth {bandwidth}; "
            "widen the bandwidth or lower the thresholds"
        )

    variance_factor = np.zeros(grid.shape, dtype=np.float32)
    scale = n_cells / (n_cells - 1.0)
    variance_factor[valid] = (
        (1.0 / n_effective[valid] - 1.0 / n_cells) * scale
    ).astype(np.float32)
    # A pixel whose weights are so diffuse that the factor underflows carries no
    # information; drop it rather than divide by ~0 when standardising.
    valid &= variance_factor > 0

    return BandwidthGeometry(
        bandwidth=float(bandwidth),
        sigma_px=float(sigma_px),
        cell_density=density.astype(np.float32, copy=False),
        n_effective=n_effective.astype(np.float32, copy=False),
        valid=valid,
        variance_factor=variance_factor,
    )


# --------------------------------------------------------------------------- #
# marks
# --------------------------------------------------------------------------- #


def transform_marks(marks: np.ndarray, transform: str) -> np.ndarray:
    """Put the marks on a scale where a local mean is not one cell's opinion.

    The per-cell activity is violently right-skewed -- median 1.45 against a 99th
    percentile of 23.6, with no upper bound -- so a raw local mean is dominated by
    whichever extreme cell happens to be nearby. That costs power rather than
    creating false positives: the permutation null inherits the same tail, so its
    maximum is inflated wherever the outlier lands, and a genuinely broad but
    moderate elevation cannot beat it.

    ``rank`` (the default) replaces each activity by its van der Waerden normal
    score within the section, turning the statistic into a spatial rank test:
    a region is enriched when the cells in it are *consistently* high, not when
    one of them is enormous. ``log`` keeps a ratio interpretation while damping
    the tail. ``none`` tests the raw activity.

    The transform is order-preserving and applied once, before permutation --
    permuting then transforming would give the same marks, so the null is
    unchanged and the test stays exact.
    """
    if transform not in MARK_TRANSFORMS:
        raise ValueError(f"unknown transform {transform!r}; pick from {MARK_TRANSFORMS}")
    if transform == "none":
        return np.ascontiguousarray(marks, dtype=np.float32)
    if transform == "log":
        if np.any(marks < 0):
            raise ValueError("log transform needs non-negative activities")
        return np.log1p(marks).astype(np.float32)

    from scipy.special import ndtri
    from scipy.stats import rankdata

    ranks = rankdata(marks, method="average")
    return ndtri((ranks - 0.5) / ranks.size).astype(np.float32)


# --------------------------------------------------------------------------- #
# surfaces
# --------------------------------------------------------------------------- #


def density_surface(
    grid: SectionGrid, geometry: BandwidthGeometry, marks: np.ndarray
) -> np.ndarray:
    """The local mean activity ``R(u)``, zero outside the valid mask."""
    weighted = ndimage.gaussian_filter(
        grid.bin_marks(marks), geometry.sigma_px, mode="constant", truncate=TRUNCATE
    )
    out = np.zeros(grid.shape, dtype=np.float32)
    np.divide(weighted, geometry.cell_density, out=out, where=geometry.valid)
    return out


@dataclass(frozen=True)
class StabilisedSurface:
    """A density-noise-stabilised activity map and the terms behind it."""

    raw: np.ndarray  # R(u), already density-normalised
    shrunk: np.ndarray  # the map to show; NaN outside the valid mask
    weight: np.ndarray  # w(u) in [0, 1]: how far each pixel kept its value
    sigma: np.ndarray  # sqrt of the per-pixel sampling variance
    tau_squared: float
    prior_mean: float


def stabilise(
    grid: SectionGrid,
    geometry: BandwidthGeometry,
    marks: np.ndarray,
    *,
    prior_mean: float | None = None,
) -> StabilisedSurface:
    """Remove the cell-density artefact from ``R`` in closed form.

    ``R = D/N`` is a weighted *mean*, so it is already unbiased for local mean
    activity at any cell density -- dividing by ``N`` is the density
    normalisation. What density still buys is **noise**: the sampling variance of
    ``R(u)`` falls with the local effective sample size, so thinly-sampled areas
    throw extreme values by chance and dominate the eye. That variance is exactly
    ``var(marks) * geometry.variance_factor``, already computed by
    :func:`build_geometry`, so no estimation machinery is needed here.

    The correction is empirical-Bayes shrinkage. Treating the true surface as
    ``theta(u) ~ (mu, tau^2)`` observed through noise ``sigma^2(u)``, method of
    moments gives ``tau^2`` from the spatial variance of ``R`` minus the average
    sampling variance, and each pixel is pulled toward ``mu`` by the share of its
    variance that is noise:

        w(u) = tau^2 / (tau^2 + sigma^2(u))
        shrunk(u) = mu + w(u) * (R(u) - mu)

    Well-sampled pixels keep their value (``w -> 1``); thin ones collapse to the
    baseline (``w -> 0``). The output stays on the scale of ``marks``.

    Two caveats worth knowing rather than discovering:

    * the smoothing makes neighbouring pixels highly correlated, so ``tau^2`` is
      a moment estimate whose *precision* is overstated -- its expectation is
      sound, but do not read it as a well-determined variance component;
    * ``tau_squared == 0`` returns a flat map at ``mu``. That is the honest
      reading of "nothing here exceeds sampling noise", not a failure, and
      callers should surface it rather than quietly plotting a blank panel.

    This removes cell density only. Activity differs by cell type and cell types
    are spatially organised, so structure that survives may still be cell-type
    anatomy rather than position.
    """
    if marks.shape != (grid.n_cells,):
        raise ValueError(f"marks has shape {marks.shape}, expected ({grid.n_cells},)")
    valid = geometry.valid
    if not valid.any():
        raise ValueError("geometry has no valid pixels to stabilise")

    raw = density_surface(grid, geometry, marks)
    variance = float(np.var(marks))
    sigma_squared = (variance * geometry.variance_factor).astype(np.float32)
    mu = float(np.mean(marks)) if prior_mean is None else float(prior_mean)

    spatial_variance = float(np.var(raw[valid]))
    mean_noise = float(np.mean(sigma_squared[valid]))
    tau_squared = max(0.0, spatial_variance - mean_noise)

    weight = np.zeros(grid.shape, dtype=np.float32)
    shrunk = np.full(grid.shape, np.nan, dtype=np.float32)
    if tau_squared > 0:
        weight[valid] = tau_squared / (tau_squared + sigma_squared[valid])
    shrunk[valid] = mu + weight[valid] * (raw[valid] - mu)

    return StabilisedSurface(
        raw=raw,
        shrunk=shrunk,
        weight=weight,
        sigma=np.sqrt(sigma_squared, dtype=np.float32),
        tau_squared=tau_squared,
        prior_mean=mu,
    )


def z_stack(
    grid: SectionGrid,
    geometries: list[BandwidthGeometry],
    marks: np.ndarray,
    *,
    mean: float,
    variance: float,
) -> np.ndarray:
    """Standardised surfaces for every bandwidth, stacked as ``(n_bw, rows, cols)``.

    ``mean``/``variance`` are the population moments of the *observed* marks and
    are permutation-invariant, so they are computed once by the caller and
    passed in rather than recomputed for every draw.
    """
    stack = np.zeros((len(geometries), *grid.shape), dtype=np.float32)
    for index, geometry in enumerate(geometries):
        surface = density_surface(grid, geometry, marks)
        sd = np.sqrt(variance * geometry.variance_factor, dtype=np.float32)
        np.divide(
            surface - np.float32(mean),
            sd,
            out=stack[index],
            where=geometry.valid,
        )
    return stack


# --------------------------------------------------------------------------- #
# the null
# --------------------------------------------------------------------------- #


def mark_shuffler(
    block_starts: np.ndarray | None,
) -> "callable[[np.ndarray, np.random.Generator], None]":
    """In-place shuffler implementing one of the two nulls.

    ``block_starts is None`` gives the global null (activity reassigned to any
    cell). Otherwise the marks are permuted only inside contiguous blocks, which
    the caller arranges to be cells of one subclass -- what survives that null is
    enrichment beyond what the local cell-type composition already explains.
    """
    if block_starts is None:

        def shuffle_global(values: np.ndarray, rng: np.random.Generator) -> None:
            rng.shuffle(values)

        return shuffle_global

    bounds = np.asarray(block_starts, dtype=np.int64)

    def shuffle_blocks(values: np.ndarray, rng: np.random.Generator) -> None:
        for start, stop in zip(bounds[:-1], bounds[1:]):
            if stop - start > 1:
                rng.shuffle(values[start:stop])

    return shuffle_blocks


def null_maxima(
    grid: SectionGrid,
    geometries: list[BandwidthGeometry],
    marks: np.ndarray,
    *,
    n_permutations: int,
    rng: np.random.Generator,
    shuffle: "callable[[np.ndarray, np.random.Generator], None]",
) -> np.ndarray:
    """Screening pass: the joint maximum of each permuted surface.

    Deliberately O(n_pixels + n_permutations) -- the permuted surfaces are
    discarded as they are produced. Only the pairs that survive screening pay
    for :func:`null_z_stack`, which has to keep them.
    """
    mean = float(np.mean(marks))
    variance = float(np.var(marks))
    any_valid = np.zeros(grid.shape, dtype=bool)
    for geometry in geometries:
        any_valid |= geometry.valid

    maxima = np.empty(n_permutations, dtype=np.float32)
    work = marks.copy()
    for draw in range(n_permutations):
        shuffle(work, rng)
        stack = z_stack(grid, geometries, work, mean=mean, variance=variance)
        maxima[draw] = np.max(stack[:, any_valid], initial=-np.inf)
    return maxima


def null_z_stack(
    grid: SectionGrid,
    geometries: list[BandwidthGeometry],
    marks: np.ndarray,
    *,
    n_permutations: int,
    rng: np.random.Generator,
    shuffle: "callable[[np.ndarray, np.random.Generator], None]",
    domain: np.ndarray,
) -> np.ndarray:
    """Retain every permuted surface, restricted to ``domain``.

    Shape ``(n_permutations, n_bw, n_domain_pixels)``. Only run for cCREs that
    already passed screening -- the step-down needs to re-maximise each draw over
    a shrinking domain, which the summary alone cannot support.
    """
    mean = float(np.mean(marks))
    variance = float(np.var(marks))
    pixels = np.flatnonzero(domain.reshape(-1))
    stored = np.empty(
        (n_permutations, len(geometries), pixels.size), dtype=np.float32
    )
    work = marks.copy()
    for draw in range(n_permutations):
        shuffle(work, rng)
        stack = z_stack(grid, geometries, work, mean=mean, variance=variance)
        stored[draw] = stack.reshape(len(geometries), -1)[:, pixels]
    return stored


def permutation_p_value(observed: float, null: np.ndarray) -> float:
    """The add-one estimator, which never returns an impossible zero."""
    if null.size == 0:
        return float("nan")
    return float((1 + int(np.count_nonzero(null >= observed))) / (null.size + 1))


# --------------------------------------------------------------------------- #
# regions
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Region:
    """One simultaneously-significant spatial domain.

    ``cells`` are the indices of the cells inside it. A component can be
    genuinely tiny -- a single edge pixel holding no cell at all -- so the caller
    filters on ``cells.size`` and reports how many it dropped, rather than this
    function silently hiding them.
    """

    rank: int
    p_value: float
    threshold: float
    bandwidth: float
    mask: np.ndarray  # bool, grid-shaped
    cells: np.ndarray  # int, indices into the section's cell ordering


def _disk(radius: int) -> np.ndarray:
    span = np.arange(-radius, radius + 1)
    yy, xx = np.meshgrid(span, span, indexing="ij")
    return (yy * yy + xx * xx) <= radius * radius


def extract_regions(
    grid: SectionGrid,
    geometries: list[BandwidthGeometry],
    observed: np.ndarray,
    null_stack: np.ndarray,
    *,
    domain: np.ndarray,
    alpha: float,
    max_regions: int,
) -> list[Region]:
    """Step-down extraction of disjoint simultaneously-significant regions.

    Each round takes the joint maximum of every permuted surface over the
    *current* domain, tests the observed maximum against it, thresholds at the
    resulting ``c_alpha``, and peels off the connected component holding the
    peak. The component is then removed together with one kernel footprint
    around it -- the neighbouring pixels are driven by the same cells, so
    leaving them in would let one hotspot be rediscovered as its own secondary.

    Because every round re-derives its threshold from the null restricted to
    what is left, the p-values need no further multiplicity correction.
    """
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    n_bw = len(geometries)
    if observed.shape[0] != n_bw or null_stack.shape[1] != n_bw:
        raise ValueError("observed and null stacks disagree on the bandwidth axis")

    pixels = np.flatnonzero(domain.reshape(-1))
    if pixels.size != null_stack.shape[2]:
        raise ValueError("null_stack was built for a different domain")

    flat_observed = observed.reshape(n_bw, -1)[:, pixels]
    alive = np.ones(pixels.size, dtype=bool)
    regions: list[Region] = []

    for rank in range(1, max_regions + 1):
        if not alive.any():
            break
        null_max = null_stack[:, :, alive].max(axis=(1, 2))
        observed_max = float(flat_observed[:, alive].max())
        p_value = permutation_p_value(observed_max, null_max)
        if p_value > alpha:
            break

        threshold = float(np.quantile(null_max, 1.0 - alpha))
        best_bw = int(np.argmax(flat_observed[:, alive].max(axis=1)))
        above = np.zeros(pixels.size, dtype=bool)
        above[alive] = flat_observed[best_bw, alive] >= threshold
        if not above.any():
            break

        grid_mask = np.zeros(grid.n_pixels, dtype=bool)
        grid_mask[pixels] = above
        labels, n_labels = ndimage.label(grid_mask.reshape(grid.shape))
        if n_labels == 0:
            break
        peak_flat = pixels[np.argmax(np.where(above, flat_observed[best_bw], -np.inf))]
        peak_label = labels.reshape(-1)[peak_flat]
        component = labels == peak_label

        regions.append(
            Region(
                rank=rank,
                p_value=p_value,
                threshold=threshold,
                bandwidth=geometries[best_bw].bandwidth,
                mask=component,
                cells=grid.cells_in_mask(component),
            )
        )

        excluded = ndimage.binary_dilation(
            component, structure=_disk(geometries[best_bw].kernel_radius_px)
        )
        alive &= ~excluded.reshape(-1)[pixels]

    return regions
