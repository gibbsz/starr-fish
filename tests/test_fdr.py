"""BH-FDR, against statsmodels.

This is the test that licensed collapsing seven separate bh_fdr definitions into
one. It is parameterised over the input regimes those definitions actually saw,
including the point mass at zero that posterior-tail p-values produce.
"""

from __future__ import annotations

import numpy as np
import pytest

from baystarrfish.stats import bh_fdr

multipletests = pytest.importorskip(
    "statsmodels.stats.multitest", reason="statsmodels is a dev-extra dependency"
).multipletests


def _reference(pvalues):
    values = np.asarray(pvalues, dtype=float)
    out = np.full(values.shape, np.nan, dtype=float)
    valid = np.isfinite(values)
    if valid.any():
        out[valid] = multipletests(values[valid], method="fdr_bh")[1]
    return out


def _cases(rng):
    return {
        "uniform": rng.uniform(size=10_000),
        "heavy ties": rng.choice([0.0, 0.01, 0.5, 1.0], size=5_000),
        "all zero": np.zeros(50),
        "all one": np.ones(50),
        "with nan": np.where(rng.uniform(size=1000) < 0.1, np.nan, rng.uniform(size=1000)),
        "2d pooled": rng.uniform(size=(300, 40)),
        "singleton": np.array([0.3]),
        "tiny": rng.uniform(size=500) ** 12,
        "mass at zero": np.where(rng.uniform(size=5000) < 0.3, 0.0, rng.uniform(size=5000)),
    }


def test_matches_statsmodels_across_every_input_regime(rng):
    for name, pvalues in _cases(rng).items():
        got, want = bh_fdr(pvalues), _reference(pvalues)
        np.testing.assert_array_equal(np.isnan(got), np.isnan(want), err_msg=name)
        finite = ~np.isnan(got)
        np.testing.assert_allclose(got[finite], want[finite], atol=1e-12, err_msg=name)


def test_shape_is_preserved(rng):
    for pvalues in _cases(rng).values():
        assert bh_fdr(pvalues).shape == np.asarray(pvalues).shape


def test_non_finite_entries_are_excluded_from_the_denominator():
    """A pair that could not be tested must not inflate the correction."""
    with_nan = bh_fdr(np.array([0.01, 0.02, np.nan, np.inf, -np.inf]))
    without = bh_fdr(np.array([0.01, 0.02]))
    np.testing.assert_allclose(with_nan[:2], without)
    assert np.isnan(with_nan[2:]).all()


def test_q_values_are_monotone_in_p(rng):
    pvalues = rng.uniform(size=2000)
    order = np.argsort(pvalues)
    q = bh_fdr(pvalues)[order]
    assert np.all(np.diff(q) >= -1e-12)


def test_q_values_stay_in_the_unit_interval(rng):
    q = bh_fdr(rng.uniform(size=5000) ** 0.05)
    assert q.min() >= 0.0 and q.max() <= 1.0


def test_all_nan_input_returns_all_nan():
    assert np.isnan(bh_fdr(np.full(7, np.nan))).all()


def test_accepts_a_pandas_series():
    pd = pytest.importorskip("pandas")
    values = [0.001, 0.4, 0.9]
    np.testing.assert_allclose(bh_fdr(pd.Series(values)), bh_fdr(np.array(values)))
