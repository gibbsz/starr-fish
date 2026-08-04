"""The negative-control contrast, on posteriors with a known answer.

Each test constructs draws where the correct call is unambiguous -- a target far
above the controls, a target identical to them, a target below -- and checks the
reported tail probability, rather than re-deriving the arithmetic.
"""

from __future__ import annotations

import numpy as np
import pytest

from baystarrfish.stats import negative_control_test

N_DRAWS, N_GROUPS, N_CRE = 500, 2, 5
TARGETS = np.array([0, 1, 2])
CONTROLS = np.array([3, 4])
CRE_NAMES = np.array([f"CRE{i:03d}" for i in range(N_CRE)])
GROUPS = np.array(["subA", "subB"])
CLASSES = np.array(["clsA", "clsB"])
CELL_COUNTS = np.array([120, 90])


def _draws(rng, offsets):
    """Posterior draws where cCRE j sits `offsets[j]` above a shared baseline."""
    base = rng.normal(0.0, 0.05, size=(N_DRAWS, N_GROUPS, 1))
    noise = rng.normal(0.0, 0.05, size=(N_DRAWS, N_GROUPS, N_CRE))
    return base + noise + np.asarray(offsets)[None, None, :]


def _run(draws, t7_totals=None, **kwargs):
    if t7_totals is None:
        t7_totals = np.full((N_GROUPS, N_CRE), 500.0)
    params = dict(
        log_gamma=draws, groups=GROUPS, cre_names=CRE_NAMES,
        target_indices=TARGETS, control_indices=CONTROLS, t7_totals=t7_totals,
        group_classes=CLASSES, group_cell_counts=CELL_COUNTS,
        t7_threshold=50.0, effect_threshold=0.0,
        individual_control_t7_threshold=None, method="unit-test",
    )
    params.update(kwargs)
    return negative_control_test(**params)


def test_clear_signal_and_clear_null_get_the_right_tail_probabilities(rng):
    # cCRE 0 far above controls, cCRE 1 equal to them, cCRE 2 far below.
    frame = _run(_draws(rng, [3.0, 0.0, -3.0, 0.0, 0.0]))
    assert len(frame) == N_GROUPS * len(TARGETS)
    by_cre = frame.groupby("cre")["p_right"].max()
    assert by_cre["CRE000"] < 1e-6          # unmistakable activity
    assert 0.2 < by_cre["CRE001"] < 0.8     # indistinguishable from control
    assert by_cre["CRE002"] > 1 - 1e-6      # below control


def test_the_control_mean_is_subtracted_draw_by_draw(rng):
    """A shared per-draw offset must cancel, not survive as spurious signal."""
    base = rng.normal(0.0, 2.0, size=(N_DRAWS, 1, 1))  # huge shared wobble
    draws = base + rng.normal(0.0, 0.05, size=(N_DRAWS, N_GROUPS, N_CRE))
    frame = _run(draws)
    # Everything is null; nothing should look significant despite the wobble.
    assert frame["q_right"].min() > 0.01


def test_effect_threshold_shifts_the_null(rng):
    draws = _draws(rng, [1.0, 0.0, 0.0, 0.0, 0.0])
    lenient = _run(draws, effect_threshold=0.0)
    strict = _run(draws, effect_threshold=2.0)
    lenient_p = lenient.loc[lenient["cre"] == "CRE000", "p_right"].max()
    strict_p = strict.loc[strict["cre"] == "CRE000", "p_right"].max()
    assert lenient_p < 1e-6 < strict_p


def test_pairs_below_the_t7_threshold_are_dropped(rng):
    t7 = np.full((N_GROUPS, N_CRE), 500.0)
    t7[0, 0] = 1.0  # one target has no evidence
    frame = _run(_draws(rng, np.zeros(N_CRE)), t7_totals=t7)
    assert not ((frame["group"] == "subA") & (frame["cre"] == "CRE000")).any()
    assert ((frame["group"] == "subB") & (frame["cre"] == "CRE000")).any()


def test_a_group_whose_controls_fail_the_filter_is_dropped_entirely(rng):
    t7 = np.full((N_GROUPS, N_CRE), 500.0)
    t7[0, CONTROLS] = 1.0  # no usable reference for subA
    frame = _run(_draws(rng, np.zeros(N_CRE)), t7_totals=t7)
    assert set(frame["group"]) == {"subB"}


def test_no_eligible_pair_anywhere_raises_rather_than_returning_empty(rng):
    with pytest.raises(ValueError, match="No cCRE-cell-type pairs"):
        _run(_draws(rng, np.zeros(N_CRE)), t7_totals=np.ones((N_GROUPS, N_CRE)))


def test_sd_multiplier_raises_the_reference(rng):
    draws = _draws(rng, [1.0, 0.0, 0.0, 0.0, 0.0])
    plain = _run(draws, control_sd_multiplier=0.0)
    strict = _run(draws, control_sd_multiplier=3.0)
    assert (
        strict["effect_vs_control_reference_mean"].mean()
        < plain["effect_vs_control_reference_mean"].mean()
    )


def test_sd_multiplier_needs_at_least_two_controls(rng):
    with pytest.raises(ValueError, match="At least two"):
        _run(_draws(rng, np.zeros(N_CRE)),
             control_indices=np.array([3]), control_sd_multiplier=1.0)


def test_negative_sd_multiplier_is_rejected(rng):
    with pytest.raises(ValueError, match="non-negative"):
        _run(_draws(rng, np.zeros(N_CRE)), control_sd_multiplier=-1.0)


def test_individual_control_filter_selects_the_surviving_controls(rng):
    t7 = np.full((N_GROUPS, N_CRE), 500.0)
    t7[:, CONTROLS[0]] = 5.0  # one control is too sparse to trust
    frame = _run(_draws(rng, np.zeros(N_CRE)), t7_totals=t7,
                 individual_control_t7_threshold=50.0)
    assert (frame["n_negative_controls"] == 1).all()
    assert (frame["negative_controls_used"] == CRE_NAMES[CONTROLS[1]]).all()


def test_backward_compatible_column_aliases_are_present(rng):
    frame = _run(_draws(rng, np.zeros(N_CRE)))
    for new, old in (
        ("effect_vs_control_reference_mean", "effect_vs_mean_control_mean"),
        ("effect_vs_control_reference_lo90", "effect_vs_mean_control_lo90"),
        ("effect_vs_control_reference_hi90", "effect_vs_mean_control_hi90"),
        ("posterior_probability_above_control_reference",
         "posterior_probability_above_mean_control"),
    ):
        np.testing.assert_array_equal(frame[new].to_numpy(), frame[old].to_numpy())


def test_q_values_are_the_bh_adjustment_of_p_right(rng):
    from baystarrfish.stats import bh_fdr

    frame = _run(_draws(rng, [3.0, 0.0, -3.0, 0.0, 0.0]))
    np.testing.assert_allclose(
        frame["q_right"].to_numpy(), bh_fdr(frame["p_right"].to_numpy())
    )
