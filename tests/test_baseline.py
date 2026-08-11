"""The negative-control reference shared by the cell-type test and the per-cell map.

The reference is the one number that decides what "background" means, so both the
published table and any normalised activity divide by it. These tests pin the
three properties that make it trustworthy: it is the geometric mean (so no
individual control's target is 1), it is per draw, and a cell type without enough
control signal gets ``NaN`` rather than a fabricated baseline.
"""

from __future__ import annotations

import numpy as np
import pytest

from baystarrfish.stats import negative_control_log_baseline, negative_control_test

N_DRAW, N_GROUP, N_CRE = 32, 4, 6
CONTROLS = np.array([3, 4, 5])
TARGETS = np.array([0, 1, 2])


@pytest.fixture
def log_gamma(rng):
    return rng.normal(0.3, 0.6, size=(N_DRAW, N_GROUP, N_CRE))


def test_reference_is_the_geometric_mean_of_the_controls(log_gamma):
    baseline = negative_control_log_baseline(log_gamma, CONTROLS)
    np.testing.assert_allclose(
        baseline.log_reference, log_gamma[:, :, CONTROLS].mean(axis=2), rtol=0, atol=0
    )
    # The defining property: gamma / b has geometric mean exactly 1 across the
    # controls of a cell type. This is why each control's target is its own
    # number and not 1, and why a check against 1 per control would be wrong.
    ratio = np.exp(log_gamma[:, :, CONTROLS] - baseline.log_reference[:, :, None])
    np.testing.assert_allclose(np.exp(np.log(ratio).mean(axis=2)), 1.0, atol=1e-12)


def test_reference_is_not_the_arithmetic_mean(log_gamma):
    """AM-GM: the arithmetic mean is strictly larger unless the controls agree."""
    baseline = negative_control_log_baseline(log_gamma, CONTROLS)
    arithmetic = np.exp(log_gamma[:, :, CONTROLS]).mean(axis=2)
    assert np.all(arithmetic > baseline.reference() - 1e-12)
    assert np.any(arithmetic > baseline.reference() * 1.001)


def test_the_reference_varies_across_draws(log_gamma):
    """It is a posterior, not a constant -- dividing per draw is meaningful."""
    baseline = negative_control_log_baseline(log_gamma, CONTROLS)
    assert np.all(baseline.log_reference.std(axis=0) > 1e-6)


def test_pooled_t7_filter_marks_a_cell_type_ineligible(log_gamma):
    t7_totals = np.full((N_GROUP, N_CRE), 100.0)
    t7_totals[2, CONTROLS] = 1.0  # pooled control T7 of 3, far below the threshold
    baseline = negative_control_log_baseline(
        log_gamma, CONTROLS, t7_totals=t7_totals, t7_threshold=50.0
    )
    assert list(baseline.eligible) == [True, True, False, True]
    assert np.all(np.isnan(baseline.log_reference[:, 2]))
    assert np.all(np.isfinite(baseline.log_reference[:, [0, 1, 3]]))
    assert baseline.n_controls[2] == 0
    assert baseline.control_indices[2].size == 0
    np.testing.assert_allclose(baseline.control_t7_total[2], 3.0)


def test_individual_filter_keeps_only_the_surviving_controls(log_gamma):
    t7_totals = np.full((N_GROUP, N_CRE), 100.0)
    t7_totals[1, CONTROLS[0]] = 2.0
    baseline = negative_control_log_baseline(
        log_gamma, CONTROLS, t7_totals=t7_totals,
        individual_control_t7_threshold=50.0,
    )
    assert baseline.n_controls[1] == 2
    np.testing.assert_array_equal(baseline.control_indices[1], CONTROLS[1:])
    np.testing.assert_allclose(
        baseline.log_reference[:, 1], log_gamma[:, 1, CONTROLS[1:]].mean(axis=1)
    )


def test_sd_multiplier_raises_the_reference_and_needs_two_controls(log_gamma):
    plain = negative_control_log_baseline(log_gamma, CONTROLS)
    strict = negative_control_log_baseline(log_gamma, CONTROLS, control_sd_multiplier=1.0)
    assert np.all(strict.log_reference > plain.log_reference)
    np.testing.assert_allclose(
        strict.log_reference,
        plain.log_mean + log_gamma[:, :, CONTROLS].std(axis=2, ddof=1),
    )
    with pytest.raises(ValueError, match="at least two|At least two"):
        negative_control_log_baseline(
            log_gamma, CONTROLS[:1], control_sd_multiplier=1.0
        )


def test_malformed_inputs_are_rejected(log_gamma):
    with pytest.raises(ValueError, match="expected \\(n_draws"):
        negative_control_log_baseline(log_gamma[0], CONTROLS)
    with pytest.raises(ValueError, match="no negative-control columns"):
        negative_control_log_baseline(log_gamma, np.array([], dtype=int))
    with pytest.raises(ValueError, match="out of range"):
        negative_control_log_baseline(log_gamma, np.array([N_CRE]))
    with pytest.raises(ValueError, match="must be non-negative"):
        negative_control_log_baseline(log_gamma, CONTROLS, control_sd_multiplier=-1.0)
    with pytest.raises(ValueError, match="t7_totals has shape"):
        negative_control_log_baseline(
            log_gamma, CONTROLS, t7_totals=np.zeros((N_GROUP, N_CRE + 1))
        )


# ---- the test that consumes it must be unchanged by the refactor ----------- #


def test_negative_control_test_contrast_matches_an_inline_computation(log_gamma):
    """The published contrast, recomputed here from scratch, must still match.

    ``negative_control_test`` now delegates its reference to this module. That
    refactor is only safe if the numbers are untouched, so this rebuilds the
    contrast inline rather than trusting the shared code path.
    """
    t7_totals = np.full((N_GROUP, N_CRE), 400.0)
    table = negative_control_test(
        log_gamma=log_gamma,
        groups=np.array([f"g{i}" for i in range(N_GROUP)]),
        cre_names=np.array([f"CRE{i:03d}" for i in range(N_CRE)]),
        target_indices=TARGETS,
        control_indices=CONTROLS,
        t7_totals=t7_totals,
        group_classes=np.array(["cls"] * N_GROUP),
        group_cell_counts=np.full(N_GROUP, 10),
        t7_threshold=50.0,
        effect_threshold=0.1,
        individual_control_t7_threshold=None,
        method="test",
    )
    assert len(table) == N_GROUP * len(TARGETS)
    for group_idx in range(N_GROUP):
        reference = log_gamma[:, group_idx, CONTROLS].mean(axis=1)
        for target in TARGETS:
            contrasts = log_gamma[:, group_idx, target] - reference - 0.1
            row = table[
                (table["group"] == f"g{group_idx}")
                & (table["cre"] == f"CRE{target:03d}")
            ]
            assert len(row) == 1
            np.testing.assert_allclose(
                row["effect_vs_control_reference_mean"].iloc[0], contrasts.mean()
            )
            np.testing.assert_allclose(
                row["p_right"].iloc[0], (contrasts <= 0.0).mean()
            )
    assert set(table["n_negative_controls"]) == {len(CONTROLS)}


def test_ineligible_cell_types_are_dropped_from_the_table(log_gamma):
    t7_totals = np.full((N_GROUP, N_CRE), 400.0)
    t7_totals[0, CONTROLS] = 0.0
    table = negative_control_test(
        log_gamma=log_gamma,
        groups=np.array([f"g{i}" for i in range(N_GROUP)]),
        cre_names=np.array([f"CRE{i:03d}" for i in range(N_CRE)]),
        target_indices=TARGETS,
        control_indices=CONTROLS,
        t7_totals=t7_totals,
        group_classes=np.array(["cls"] * N_GROUP),
        group_cell_counts=np.full(N_GROUP, 10),
        t7_threshold=50.0,
        effect_threshold=0.0,
        individual_control_t7_threshold=None,
        method="test",
    )
    assert "g0" not in set(table["group"])
    assert len(table) == (N_GROUP - 1) * len(TARGETS)
