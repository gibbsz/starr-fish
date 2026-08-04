"""Collapsed sufficient statistics for the (cells x cCREs) count arrays.

Scalability comes from collapsing the observation array to weighted unique
``(group, cre, *counts)`` rows: ~99.85% of (cell, cCRE) pairs are all-zero and
share an identical marginal within a cell type, so they collapse to a single
weighted row. A 408,621 x 400 array reduces to a few thousand rows.

Also holds the latent copy-number truncation helpers: ``kmax`` selection and the
audit of how much probability mass the truncation discards.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd

# jax_enable_x64 must be set before the first jnp array; see baystarrfish._jax_setup.
from .. import _jax_setup as _jax_setup  # noqa: F401

import jax.numpy as jnp


@dataclass
class CollapsedStats:
    """Weighted unique-row representation of the observation array.

    Each row is one distinct ``(group, cre, *channel_counts)`` pattern; ``weight``
    is the number of cells contributing it. The likelihood is the ``weight``-
    weighted sum of per-row marginal log-likelihoods.

    Attributes
    ----------
    group : (M,) int
        Group index per row (class or subclass, depending on granularity).
    cre : (M,) int
        CRE column index per row, in ``[0, n_cre)``.
    counts : dict[str, (M,) int]
        Observed count per channel (``"t7"`` and optionally ``"cre"``).
    weight : (M,) int
        Number of cells with this exact pattern.
    n_per_group : (n_group,) int
        Cell count per group.
    class_of_group : (n_group,) int or None
        For subclass-level stats, the parent class index of each subclass; None
        when ``group`` already is the class.
    n_group, n_cre, n_class : int
        Dimensions.
    channels : tuple[str, ...]
        Channel names present, in order.
    """

    group: np.ndarray
    cre: np.ndarray
    counts: Mapping[str, np.ndarray]
    weight: np.ndarray
    n_per_group: np.ndarray
    n_group: int
    n_cre: int
    channels: tuple
    class_of_group: np.ndarray | None = None
    n_class: int | None = None

    def to_jax(self) -> "CollapsedStats":
        """Return a copy with array fields moved to JAX device arrays."""
        return CollapsedStats(
            group=jnp.asarray(self.group),
            cre=jnp.asarray(self.cre),
            counts={k: jnp.asarray(v) for k, v in self.counts.items()},
            weight=jnp.asarray(self.weight, dtype=jnp.float64),
            n_per_group=jnp.asarray(self.n_per_group),
            n_group=self.n_group,
            n_cre=self.n_cre,
            channels=self.channels,
            class_of_group=None if self.class_of_group is None else jnp.asarray(self.class_of_group),
            n_class=self.n_class,
        )


def _nonzero_coords(mat) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(rows, cols, values)`` of the nonzero entries of ``mat``.

    Accepts dense ``np.ndarray``, ``pandas.DataFrame`` or any object exposing a
    SciPy-sparse ``.tocoo()``; values are cast to ``int64``.
    """
    if hasattr(mat, "tocoo"):  # scipy sparse
        coo = mat.tocoo()
        return coo.row.astype(np.int64), coo.col.astype(np.int64), coo.data.astype(np.int64)
    arr = mat.values if isinstance(mat, pd.DataFrame) else np.asarray(mat)
    rows, cols = np.nonzero(arr)
    return rows.astype(np.int64), cols.astype(np.int64), arr[rows, cols].astype(np.int64)


def build_sufficient_stats(
    channel_mats: Mapping[str, "np.ndarray | pd.DataFrame"],
    group_idx: np.ndarray,
    n_group: int,
    n_cre: int,
    class_of_group: np.ndarray | None = None,
    n_class: int | None = None,
) -> CollapsedStats:
    """Collapse cell-level count matrices to weighted unique rows.

    Parameters
    ----------
    channel_mats : ordered mapping name -> (n_cells, n_cre) integer matrix
        e.g. ``{"t7": t7, "cre": cre}`` (joint) or ``{"t7": t7}`` (T7-only).
        Dense, DataFrame or SciPy-sparse are all accepted.
    group_idx : (n_cells,) int
        Group (class or subclass) index per cell, in ``[0, n_group)``.
    n_group, n_cre : int
        Dimensions.
    class_of_group, n_class : optional
        Parent-class map for subclass-level stats (enables the nested hierarchy).

    Returns
    -------
    CollapsedStats
        Rows include the single all-zero ``(group, cre, 0, ...)`` pattern per
        ``(group, cre)`` carrying weight ``N_group - n_nonzero_cells``.
    """
    channels = tuple(channel_mats.keys())
    group_idx = np.asarray(group_idx).astype(np.int64)
    n_cells = group_idx.shape[0]

    # 1. Union of nonzero (cell, cre) positions across channels. The CRE-index
    # column is named "cre_idx" to avoid colliding with a "cre" channel name.
    frames = []
    for name, mat in channel_mats.items():
        rows, cols, vals = _nonzero_coords(mat)
        frames.append(pd.DataFrame({"cell": rows, "cre_idx": cols, name: vals}))
    # Outer-join channels on (cell, cre_idx); missing entries are true zeros.
    merged = frames[0]
    for f in frames[1:]:
        merged = merged.merge(f, on=["cell", "cre_idx"], how="outer")
    for name in channels:
        merged[name] = merged[name].fillna(0).astype(np.int64)
    merged["group"] = group_idx[merged["cell"].to_numpy()]

    # 2. Group nonzero patterns -> weights.
    key_cols = ["group", "cre_idx", *channels]
    nz = (
        merged.groupby(key_cols, sort=False).size().reset_index(name="weight")
        if len(merged)
        else pd.DataFrame(columns=[*key_cols, "weight"])
    )

    # 3. Per-(group, cre) count of cells with any nonzero -> all-zero remainder.
    if len(merged):
        nonzero_cells = merged.groupby(["group", "cre_idx"], sort=False).size().reset_index(name="n_nz")
    else:
        nonzero_cells = pd.DataFrame(columns=["group", "cre_idx", "n_nz"])
    n_per_group = np.bincount(group_idx, minlength=n_group).astype(np.int64)

    # Full (group, cre) grid; all-zero weight = N_group - n_nonzero_cells.
    grid = pd.MultiIndex.from_product(
        [np.arange(n_group), np.arange(n_cre)], names=["group", "cre_idx"]
    ).to_frame(index=False)
    grid = grid.merge(nonzero_cells, on=["group", "cre_idx"], how="left")
    grid["n_nz"] = grid["n_nz"].fillna(0).astype(np.int64)
    grid["weight"] = n_per_group[grid["group"].to_numpy()] - grid["n_nz"].to_numpy()
    allzero = grid.loc[grid["weight"] > 0, ["group", "cre_idx", "weight"]].copy()
    for name in channels:
        allzero[name] = np.int64(0)

    rows = pd.concat([nz[[*key_cols, "weight"]], allzero[[*key_cols, "weight"]]], ignore_index=True)
    rows = rows.astype(np.int64)

    return CollapsedStats(
        group=rows["group"].to_numpy(),
        cre=rows["cre_idx"].to_numpy(),
        counts={name: rows[name].to_numpy() for name in channels},
        weight=rows["weight"].to_numpy(),
        n_per_group=n_per_group,
        n_group=n_group,
        n_cre=n_cre,
        channels=channels,
        class_of_group=None if class_of_group is None else np.asarray(class_of_group).astype(np.int64),
        n_class=n_class,
    )


def summarize_evidence(stats: CollapsedStats) -> dict:
    """Compute per-(group, cre) and global evidence counts.

    Returns a dict with a per-pair :class:`pandas.DataFrame` (``"per_pair"``)
    holding ``n_t7_pos``, ``n_cre_pos`` (if present), ``n_double_pos`` and
    ``n_total``, plus scalar totals and observed zero fractions matching the
    pre-fit audit described in the plan.
    """
    # "cre_idx" holds the CRE column index; channel counts keep their own names
    # ("t7", "cre") so a "cre" channel never collides with the index column.
    df = pd.DataFrame({"group": np.asarray(stats.group), "cre_idx": np.asarray(stats.cre),
                       "weight": np.asarray(stats.weight)})
    for name in stats.channels:
        df[name] = np.asarray(stats.counts[name])

    pos = {f"n_{name}_pos": (df[name] > 0) for name in stats.channels}
    agg_cols = {}
    for col, mask in pos.items():
        df[col] = np.where(mask, df["weight"], 0)
        agg_cols[col] = "sum"
    if {"t7", "cre"} <= set(stats.channels):
        df["n_double_pos"] = np.where((df["t7"] > 0) & (df["cre"] > 0), df["weight"], 0)
        agg_cols["n_double_pos"] = "sum"
    df["n_total"] = df["weight"]
    agg_cols["n_total"] = "sum"

    per_pair = df.groupby(["group", "cre_idx"], sort=True).agg(agg_cols).reset_index()
    per_pair = per_pair.rename(columns={"cre_idx": "cre"})

    total_cells = int(stats.n_per_group.sum())
    totals = {
        "n_cells": total_cells,
        "n_cre": int(stats.n_cre),
        "n_group": int(stats.n_group),
        "n_pairs": int(stats.n_group * stats.n_cre),
    }
    denom = total_cells * stats.n_cre
    for name in stats.channels:
        n_pos = int(per_pair[f"n_{name}_pos"].sum())
        totals[f"n_{name}_pos"] = n_pos
        totals[f"{name}_zero_fraction"] = float(1.0 - n_pos / denom)
    if "n_double_pos" in per_pair:
        totals["n_double_pos"] = int(per_pair["n_double_pos"].sum())
        totals["n_union_nonzero"] = int(
            np.where((df[[*stats.channels]].to_numpy() > 0).any(axis=1), df["weight"], 0).sum()
        )
    return {"per_pair": per_pair, "totals": totals}


def choose_kmax(lam_max: float, max_count: int, beta_t7: float,
                poisson_tol: float = 1e-8, floor: int = 5, cap: int = 60) -> int:
    """Pick a copy-number truncation ``Kmax``.

    Large enough that (a) the Poisson(``lam_max``) tail beyond ``Kmax`` is below
    ``poisson_tol`` and (b) ``Kmax * beta_t7`` covers the bulk of observed counts,
    but capped at ``cap`` so the (n_rows x Kmax) likelihood tensors fit in GPU
    memory. A few extreme-count outliers may be truncated; the caller should log
    how many (see :func:`count_kmax_truncated`). Validate with :func:`kmax_tail_mass`.
    """
    from scipy.stats import poisson

    k_poisson = int(poisson.ppf(1.0 - poisson_tol, max(lam_max, 1e-12)))
    k_counts = int(np.ceil(2.0 * max_count / max(beta_t7, 1e-6)))
    return int(min(cap, max(floor, k_poisson, k_counts)))


def count_kmax_truncated(stats: CollapsedStats, kmax: int, beta_t7: float) -> int:
    """Number of cells whose observed counts imply a copy number above ``kmax``
    (i.e. ``max(t7, cre)/beta_t7 > kmax``) and are therefore modelled imperfectly."""
    counts = np.maximum.reduce([np.asarray(stats.counts[c]) for c in stats.channels])
    mask = counts > kmax * max(beta_t7, 1e-6)
    return int(np.asarray(stats.weight)[mask].sum())


def kmax_tail_mass(lam: np.ndarray, kmax: int) -> float:
    """Max over groups/CREs of Poisson(``lam``) tail mass above ``kmax``."""
    from scipy.stats import poisson

    return float(np.max(poisson.sf(kmax, np.asarray(lam))))


__all__ = [
    "CollapsedStats",
    "build_sufficient_stats",
    "summarize_evidence",
    "choose_kmax",
    "count_kmax_truncated",
    "kmax_tail_mass",
]
