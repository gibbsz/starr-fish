"""The likelihood, against independent references.

These are the tests that would catch a silent change to the model itself, so
each one checks against something outside the package: scipy's negative
binomial, an explicit sum over the latent grid, or a limiting case with a known
closed form.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import nbinom, poisson

from baystarrfish._jax_setup import assert_x64_enabled
from baystarrfish.inference.posterior_k import channel_logprob as np_channel_logprob
from baystarrfish.model.likelihood import _channel_logprob, _nb2_logprob


def test_x64_is_enabled():
    """Guards the import-order bug: float32 makes the marginal produce nan."""
    assert_x64_enabled()


def test_nb2_matches_scipy(rng):
    counts = rng.integers(0, 40, size=200).astype(float)
    mean = rng.lognormal(1.0, 0.6, size=200)
    conc = 3.7
    got = np.asarray(_nb2_logprob(counts, mean, conc))
    want = nbinom.logpmf(counts, conc, conc / (conc + mean))
    # 1e-5: JAX's lgamma is a lower-precision approximation than scipy's.
    # Ample to catch a wrong parameterisation, which is off by orders of magnitude.
    np.testing.assert_allclose(got, want, rtol=1e-5)


def test_nb2_variance_parameterisation(rng):
    """NB2 means var = mu + mu^2/phi; check by sampling, not by re-deriving."""
    from baystarrfish.model.forward import nb2_sample

    mu, phi, n = 6.0, 2.5, 400_000
    draws = nb2_sample(rng, np.full(n, mu), phi)
    assert abs(draws.mean() - mu) < 0.05
    assert abs(draws.var() - (mu + mu**2 / phi)) < 0.6


def test_zero_copy_is_a_point_mass_not_a_zero_mean_nb():
    """k=0 must force an exact zero: P(obs>0 | k=0) = 0, P(obs=0 | k=0) = 1."""
    obs = np.array([[0.0], [1.0], [7.0]])
    k = np.array([[0, 1, 2]], dtype=float)
    out = np.asarray(_channel_logprob(obs, k, np.full((3, 1), 3.0), 2.0))
    assert out[0, 0] == 0.0
    assert np.isneginf(out[1:, 0]).all()
    assert np.isfinite(out[:, 1:]).all()


def test_dropout_at_zero_reduces_to_plain_nb():
    obs = np.array([[0.0], [3.0]])
    k = np.array([[0, 1, 4]], dtype=float)
    per_copy = np.full((2, 1), 2.0)
    plain = np.asarray(_channel_logprob(obs, k, per_copy, 3.0, None))
    with_zero_dropout = np.asarray(_channel_logprob(obs, k, per_copy, 3.0, 1e-300))
    np.testing.assert_allclose(plain, with_zero_dropout, atol=1e-12)


def test_dropout_moves_mass_only_onto_the_zero_observation():
    """Dropout may only inflate P(obs=0 | k>0); it must not touch obs>0 upward."""
    obs = np.array([[0.0], [5.0]])
    k = np.array([[1, 3]], dtype=float)
    per_copy = np.full((2, 1), 2.0)
    plain = np.asarray(_channel_logprob(obs, k, per_copy, 3.0))
    dropped = np.asarray(_channel_logprob(obs, k, per_copy, 3.0, 0.3))
    assert (dropped[0] > plain[0]).all()
    assert (dropped[1] < plain[1]).all()


def test_channel_is_a_normalised_distribution_over_counts():
    """Sum over all achievable counts must be 1 for each k, dropout or not.

    Truncated at 20,000 counts; NB2 with phi=3 has a heavy enough tail that the
    residual mass, not floating point, sets the tolerance.
    """
    counts = np.arange(0, 20_000, dtype=float)[:, None]
    for p_drop in (None, 0.25):
        for k in (1.0, 3.0):
            logp = np.asarray(
                _channel_logprob(counts, np.array([[k]]), np.full((len(counts), 1), 2.0),
                                 3.0, p_drop)
            )
            assert abs(np.exp(logp).sum() - 1.0) < 1e-5, (k, p_drop)


@pytest.mark.parametrize("p_drop", [None, 0.02, 0.4, 0.95])
def test_numpy_and_jax_channel_logprob_agree(p_drop, rng):
    """posterior_k re-implements the channel in NumPy; pin the two together."""
    obs = rng.poisson(0.5, size=(150, 1)).astype(float)
    k = np.arange(0, 25, dtype=float)[None, :]
    per_copy = rng.lognormal(0.5, 0.4, size=(150, 1))
    a = np.asarray(np_channel_logprob(obs, k, per_copy, 3.7, p_drop))
    b = np.asarray(_channel_logprob(obs, k, per_copy, 3.7, p_drop))
    np.testing.assert_array_equal(np.isneginf(a), np.isneginf(b))
    finite = np.isfinite(a)
    np.testing.assert_allclose(a[finite], b[finite], atol=1e-9)


def test_marginal_equals_an_explicit_sum_over_k(rng):
    """The whole point of the model: logsumexp over k reproduces the mixture."""
    from baystarrfish.model.collapse import build_sufficient_stats
    from baystarrfish.model.likelihood import marginal_loglik

    n_cells, n_cre, kmax = 60, 3, 40
    t7 = rng.poisson(0.4, size=(n_cells, n_cre))
    cre = rng.poisson(0.2, size=(n_cells, n_cre))
    group = np.zeros(n_cells, dtype=np.int64)
    stats = build_sufficient_stats({"t7": t7, "cre": cre}, group, 1, n_cre,
                                   class_of_group=None, n_class=None)
    lam = np.full(len(stats.weight), 0.35)
    gamma = np.full(len(stats.weight), 1.8)
    got = np.asarray(marginal_loglik(stats, lam, 4.0, 3.0, gamma, 2.5, kmax=kmax))

    obs_t7 = np.asarray(stats.counts["t7"])
    obs_cre = np.asarray(stats.counts["cre"])
    want = np.empty(len(obs_t7))
    for i in range(len(obs_t7)):
        total = 0.0
        for k in range(kmax + 1):
            pk = poisson.pmf(k, lam[i])
            if k == 0:
                channel = float(obs_t7[i] == 0) * float(obs_cre[i] == 0)
            else:
                channel = (
                    nbinom.pmf(obs_t7[i], 3.0, 3.0 / (3.0 + 4.0 * k))
                    * nbinom.pmf(obs_cre[i], 2.5, 2.5 / (2.5 + gamma[i] * k))
                )
            total += pk * channel
        want[i] = np.log(total)
    np.testing.assert_allclose(got, want, rtol=1e-8)
