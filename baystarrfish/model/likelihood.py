"""Marginal likelihood over the latent copy number ``k``.

::

    latent copies   k_{ij}   ~ Poisson(lambda_{s,j}),  lambda_{s,j} = rho_s * a_j
    T7 channel      t7_{ij}  | k ~ NB2(mean = k * beta_t7,      disp = phi_t7)
    cCRE channel    cre_{ij} | k ~ NB2(mean = k * gamma_{s,j},  disp = phi_cre)

``k = 0`` forces both channels to exactly zero (a point mass, not an NB with
mean 0), and optional zero-inflated measurement dropout applies only at ``k > 0``.
The discrete ``k`` is marginalised analytically by ``logsumexp`` over the
truncated grid ``0..kmax``, so the target is a continuous-parameter model that
NumPyro can fit by SVI or NUTS.

``binary_infection_loglik`` is the alternative infection model: a shared
Bernoulli gate with probability ``1 - exp(-rho_s * a_j)`` (cloglog) replaces the
Poisson copy count, i.e. a shared-gate zero-inflated negative binomial.
"""

from __future__ import annotations

import numpy as np

# jax_enable_x64 must be set before the first jnp array; see baystarrfish._jax_setup.
from .. import _jax_setup as _jax_setup  # noqa: F401

import jax.numpy as jnp
from jax.scipy.special import logsumexp

import numpyro.distributions as dist

from .collapse import CollapsedStats


def _nb2_logprob(count, mean, conc):
    """NB2 log-pmf with mean ``mean`` (>0) and dispersion ``conc`` (var = mu + mu^2/conc)."""
    rate = conc / mean
    return dist.GammaPoisson(concentration=conc, rate=rate).log_prob(count)


def _channel_logprob(obs, k, per_copy, phi, p_drop=None):
    """log P(obs | k) for one NB channel over a k-grid.

    Parameters
    ----------
    obs : (M, 1) array
    k : (1, K) array of integer copy numbers including 0
    per_copy : (M, 1) array, the per-copy mean (beta_t7 or gamma)
    phi : scalar dispersion
    p_drop : optional scalar measurement-dropout probability for k > 0
    """
    safe_k = jnp.where(k == 0, 1.0, k)          # avoid mean=0 at k=0
    mean = per_copy * safe_k                     # (M, K)
    nb = _nb2_logprob(obs, mean, phi)            # finite everywhere (mean>0)
    if p_drop is not None:
        log_keep = jnp.log1p(-p_drop)
        log_drop = jnp.log(p_drop)
        nb = jnp.where(
            obs == 0,
            jnp.logaddexp(log_drop, log_keep + nb),
            log_keep + nb,
        )
    point_mass = jnp.where(obs == 0, 0.0, -jnp.inf)  # (M, 1) broadcast over K
    return jnp.where(k == 0, point_mass, nb)


def marginal_loglik(stats: CollapsedStats, lam, beta_t7, phi_t7,
                    gamma=None, phi_cre=None, kmax: int = 30,
                    p_drop_t7=None, p_drop_cre=None):
    """Per-row marginal log-likelihood, ``logsumexp_k`` over ``0..kmax``.

    Parameters
    ----------
    stats : CollapsedStats (JAX arrays)
    lam : (M,) expected copies for each row, ``rho[group] * a[cre]``.
    beta_t7, phi_t7 : scalars
    gamma : (M,) per-copy CRE mean for each row, or None for a T7-only model.
    phi_cre : scalar or None
    kmax : int

    Returns
    -------
    (M,) array of marginal log-likelihoods (un-weighted).
    """
    k = jnp.arange(0, kmax + 1)                  # (K,)
    k_row = k[None, :]
    log_pk = dist.Poisson(lam[:, None]).log_prob(k_row)            # (M, K)

    t7 = stats.counts["t7"][:, None]
    ll = log_pk + _channel_logprob(t7, k_row, beta_t7, phi_t7, p_drop_t7)

    if gamma is not None:
        cre = stats.counts["cre"][:, None]
        ll = ll + _channel_logprob(cre, k_row, gamma[:, None], phi_cre, p_drop_cre)

    return logsumexp(ll, axis=1)                  # (M,)


def gauss_hermite_rule(n_points: int = 7) -> tuple[np.ndarray, np.ndarray]:
    """Return nodes and log-weights for E[f(Z)], Z ~ Normal(0, 1)."""
    if n_points <= 0:
        raise ValueError("n_points must be positive")
    nodes, weights = np.polynomial.hermite.hermgauss(n_points)
    nodes = np.sqrt(2.0) * nodes
    log_weights = np.log(weights) - 0.5 * np.log(np.pi)
    return nodes.astype(np.float64), log_weights.astype(np.float64)


def cre_marginal_loglik(
    stats: CollapsedStats,
    log_lambda_mean,
    log_lambda_sd,
    gh_nodes,
    gh_log_weights,
    gamma,
    phi_cre,
    p_drop_cre,
    kmax: int = 30,
):
    """CRE-only marginal likelihood integrating over T7-derived log infection.

    ``log_lambda_mean`` and ``log_lambda_sd`` are row-level summaries of the T7
    posterior for log(rho_s * a_j). The expectation over that posterior is
    approximated by Gauss-Hermite quadrature.
    """
    k = jnp.arange(0, kmax + 1)
    log_lambda = log_lambda_mean[:, None] + log_lambda_sd[:, None] * gh_nodes[None, :]
    lam = jnp.exp(log_lambda)
    log_pk = dist.Poisson(lam[:, :, None]).log_prob(k[None, None, :])

    cre = stats.counts["cre"][:, None, None]
    channel_ll = _channel_logprob(
        cre, k[None, None, :], gamma[:, None, None], phi_cre, p_drop_cre
    )
    ll_given_lambda = logsumexp(log_pk + channel_ll, axis=2)
    return logsumexp(gh_log_weights[None, :] + ll_given_lambda, axis=1)


def binary_infection_loglik(stats: CollapsedStats, infection_rate, beta_t7, phi_t7,
                            gamma=None, phi_cre=None):
    """Per-row likelihood after marginalizing a shared binary infection event.

    ``infection_rate`` is the positive infection hazard ``rho[group] * a[cre]``.
    The corresponding infection probability uses the complementary-log-log link:
    ``p_infected = 1 - exp(-infection_rate)``.

    If uninfected, every observed channel is exactly zero. If infected, T7 and
    CRE counts are conditionally independent NB2 variables and may themselves be
    zero, making this a shared-gate zero-inflated negative-binomial model.
    """
    rate = jnp.maximum(infection_rate, jnp.finfo(jnp.float64).tiny)
    log_p_uninfected = -rate
    log_p_infected = jnp.log(-jnp.expm1(-rate))

    all_zero = stats.counts["t7"] == 0
    infected_ll = _nb2_logprob(stats.counts["t7"], beta_t7, phi_t7)
    if gamma is not None:
        all_zero = all_zero & (stats.counts["cre"] == 0)
        infected_ll = infected_ll + _nb2_logprob(stats.counts["cre"], gamma, phi_cre)

    uninfected_ll = jnp.where(all_zero, log_p_uninfected, -jnp.inf)
    return jnp.logaddexp(uninfected_ll, log_p_infected + infected_ll)


__all__ = [
    "marginal_loglik",
    "gauss_hermite_rule",
    "cre_marginal_loglik",
    "binary_infection_loglik",
]
