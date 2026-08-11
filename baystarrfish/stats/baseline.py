"""The negative-control reference that turns an arbitrary-scale activity into a
fold change.

``log_gamma`` means nothing on its own: its scale is set jointly by the infection
rate and the library abundance, so every published claim is a *contrast* against
the negative controls of the same cell type. This module owns that reference, so
the cell-type test (:func:`baystarrfish.stats.negative_control.negative_control_test`)
and the per-cell normalised activity
(:func:`baystarrfish.inference.posterior_k.posterior_k_moments`) cannot drift
apart::

    log b[d, s] = mean_{j' in controls(s)} log_gamma[d, s, j']

The mean of *logs* -- the geometric mean of the activities. That is the centre a
log contrast is symmetric about, and it is what the published tables used. The
arithmetic mean would be dominated by whichever control is most active and would
shrink every fold change. A direct consequence, and the cheapest check that this
module is correct: the geometric mean of ``gamma / b`` over the controls of a
cell type is exactly 1, so no individual control's target is 1.

The reference is **per draw**. Dividing inside the draw carries its uncertainty
through instead of treating it as a fixed offset; ``gamma`` and ``b`` share the
scale factors that make either one arbitrary, so their ratio is far better
determined than either term. On the production fit the two forms differ by 0.8%
at the median and up to 17% in the tail -- small, but free to get right.

Eligibility is a T7 filter, not a cCRE filter: a cell type whose controls carry
too little constitutive signal has no reference to contrast against and is
dropped rather than handed a baseline the data cannot support. Those columns come
back ``NaN``, which propagates through every downstream per-cell value and keeps
them out of the figures without a second masking step.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["NegativeControlBaseline", "negative_control_log_baseline"]


@dataclass(frozen=True)
class NegativeControlBaseline:
    """The per-draw, per-cell-type control reference, and who is eligible for one.

    Attributes
    ----------
    log_mean : (n_draws, n_groups)
        ``mean_{j'} log_gamma[d, s, j']`` over the selected controls -- the log of
        their geometric mean activity. ``NaN`` for ineligible cell types.
    log_sd : (n_draws, n_groups)
        Spread across the selected controls within a draw (``ddof=1``); zero when
        a cell type has a single surviving control, ``NaN`` when ineligible.
    log_reference : (n_draws, n_groups)
        ``log_mean + control_sd_multiplier * log_sd`` -- the value to divide by.
        Identical to ``log_mean`` at the default multiplier of zero.
    eligible : (n_groups,) bool
        Whether the cell type has a usable reference at all.
    control_indices : tuple of int arrays
        Columns actually used per cell type; empty where ineligible.
    n_controls : (n_groups,) int
        Length of each entry above, for reporting.
    control_t7_total : (n_groups,) float
        Pooled constitutive-channel total over the controls, the quantity the
        eligibility filter tests. ``NaN`` when no ``t7_totals`` was supplied.
    """

    log_mean: np.ndarray
    log_sd: np.ndarray
    log_reference: np.ndarray
    eligible: np.ndarray
    control_indices: tuple[np.ndarray, ...]
    n_controls: np.ndarray
    control_t7_total: np.ndarray

    @property
    def n_draws(self) -> int:
        return int(self.log_reference.shape[0])

    @property
    def n_groups(self) -> int:
        return int(self.log_reference.shape[1])

    def reference(self) -> np.ndarray:
        """``b[d, s]`` on the natural scale -- the value an activity divides by."""
        return np.exp(self.log_reference)


def negative_control_log_baseline(
    log_gamma: np.ndarray,
    control_indices: np.ndarray,
    *,
    t7_totals: np.ndarray | None = None,
    t7_threshold: float = 0.0,
    individual_control_t7_threshold: float | None = None,
    control_sd_multiplier: float = 0.0,
) -> NegativeControlBaseline:
    """Build the control reference, one column per cell type.

    Parameters
    ----------
    log_gamma : (n_draws, n_groups, n_cre)
        Posterior draws of log activity.
    control_indices
        Columns of ``log_gamma`` holding the negative controls.
    t7_totals : (n_groups, n_cre), optional
        Per-pair constitutive totals driving eligibility. Omit to accept every
        cell type and use every control -- correct for a synthetic check, wrong
        for anything that must match a published table.
    t7_threshold
        Minimum *pooled* control T7 for a cell type to get a reference, used only
        when ``individual_control_t7_threshold`` is None.
    individual_control_t7_threshold
        If given, controls are filtered one by one at this threshold and the
        reference is the mean over the survivors; a cell type is eligible as long
        as at least one survives.
    control_sd_multiplier
        Raise the reference by this many control standard deviations (per draw)
        for a stricter null. Requires at least two surviving controls.
    """
    log_gamma = np.asarray(log_gamma)
    if log_gamma.ndim != 3:
        raise ValueError(
            f"log_gamma has shape {log_gamma.shape}; expected "
            "(n_draws, n_groups, n_cre)"
        )
    if control_sd_multiplier < 0:
        raise ValueError("control_sd_multiplier must be non-negative")
    n_draws, n_groups, n_cre = log_gamma.shape

    control_indices = np.asarray(control_indices, dtype=np.int64).ravel()
    if control_indices.size == 0:
        raise ValueError("no negative-control columns given")
    if control_indices.min() < 0 or control_indices.max() >= n_cre:
        raise ValueError(
            f"control_indices out of range for {n_cre} cCRE columns: "
            f"{control_indices.min()}..{control_indices.max()}"
        )

    if t7_totals is not None:
        t7_totals = np.asarray(t7_totals, dtype=np.float64)
        if t7_totals.shape != (n_groups, n_cre):
            raise ValueError(
                f"t7_totals has shape {t7_totals.shape}; expected "
                f"{(n_groups, n_cre)} to match log_gamma"
            )

    log_mean = np.full((n_draws, n_groups), np.nan, dtype=np.float64)
    log_sd = np.full((n_draws, n_groups), np.nan, dtype=np.float64)
    log_reference = np.full((n_draws, n_groups), np.nan, dtype=np.float64)
    eligible = np.zeros(n_groups, dtype=bool)
    n_controls = np.zeros(n_groups, dtype=np.int64)
    control_t7_total = np.full(n_groups, np.nan, dtype=np.float64)
    selected: list[np.ndarray] = []

    for group in range(n_groups):
        if t7_totals is None:
            chosen = control_indices
            passes = True
        else:
            control_t7 = t7_totals[group, control_indices]
            control_t7_total[group] = float(control_t7.sum())
            if individual_control_t7_threshold is None:
                chosen = control_indices
                passes = control_t7_total[group] >= t7_threshold
            else:
                chosen = control_indices[control_t7 >= individual_control_t7_threshold]
                passes = len(chosen) > 0
        if not passes:
            selected.append(np.empty(0, dtype=np.int64))
            continue

        draws = log_gamma[:, group, chosen].astype(np.float64, copy=False)
        if control_sd_multiplier > 0 and draws.shape[1] < 2:
            raise ValueError(
                "At least two selected negative controls are required for an SD reference"
            )
        mean = draws.mean(axis=1)
        sd = (
            draws.std(axis=1, ddof=1)
            if draws.shape[1] >= 2
            else np.zeros(draws.shape[0], dtype=np.float64)
        )
        log_mean[:, group] = mean
        log_sd[:, group] = sd
        log_reference[:, group] = mean + control_sd_multiplier * sd
        eligible[group] = True
        n_controls[group] = len(chosen)
        selected.append(np.asarray(chosen, dtype=np.int64))

    return NegativeControlBaseline(
        log_mean=log_mean,
        log_sd=log_sd,
        log_reference=log_reference,
        eligible=eligible,
        control_indices=tuple(selected),
        n_controls=n_controls,
        control_t7_total=control_t7_total,
    )
