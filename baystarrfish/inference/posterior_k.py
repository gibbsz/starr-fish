"""Recover the posterior of the latent copy number ``k`` after fitting.

``k`` is marginalised analytically during inference, so it is never sampled and
never appears in the posterior draws. It is still the quantity people want to
see -- "how many virus copies does the model think this cell received?" -- and it
is recoverable exactly, because the same grid the fit summed over can be
renormalised into a distribution::

    P(k | t7, cre, theta) proportional to
        Poisson(k; rho_s a_j) NB2(t7 | k beta) NB2(cre | k gamma)   [x dropout]

    E[k | obs] = mean over posterior draws of sum_k k P(k | obs, theta_d)

Deliberately NumPy rather than JAX: this streams over ~160 million (cell, cCRE)
pairs in chunks on a CPU node with no accelerator, which is the opposite of the
fitting workload. :func:`channel_logprob` mirrors
:func:`baystarrfish.model.likelihood._channel_logprob` exactly -- ``tests`` pins
the two implementations to each other, including the dropout branch, so the
duplication cannot drift.
"""

from __future__ import annotations

from typing import Mapping, NamedTuple

import numpy as np
from scipy.special import gammaln

__all__ = [
    "PosteriorKMoments",
    "channel_logprob",
    "nb2_logpmf",
    "posterior_k_expectation",
    "posterior_k_moments",
]


class PosteriorKMoments(NamedTuple):
    """Posterior summaries of the latent copy number, per observation.

    ``mean`` and ``sd`` describe how many copies; ``p_infected`` is
    ``P(k >= 1 | obs)``, the probability the cell received *any* copy of that
    construct. They answer different questions and can disagree: a pair can be
    almost certainly infected (``p_infected`` near 1) with a low expected count,
    or carry a high ``mean`` driven by a heavy tail while ``p_infected`` is
    modest.

    ``activity`` is the posterior mean per-cell enhancer activity -- see
    :func:`posterior_k_moments` for the derivation. It is a posterior of a
    quantity the model contains, not a ratio formed after the fact.
    """

    mean: np.ndarray
    sd: np.ndarray
    p_infected: np.ndarray
    activity: np.ndarray


def nb2_logpmf(count, mean, conc):
    """NB2 log-pmf with ``var = mean + mean**2 / conc`` (NumPy mirror of the model)."""
    p = conc / (conc + mean)
    return (
        gammaln(count + conc)
        - gammaln(conc)
        - gammaln(count + 1)
        + conc * np.log(p)
        + count * np.log1p(-p)
    )


def channel_logprob(obs, k, per_copy, phi, p_drop=None):
    """``log P(obs | k)`` for one channel over a ``k`` grid.

    ``k = 0`` is a point mass at zero, not an NB with mean zero. Dropout applies
    only where ``k > 0``: an uninfected observation is already an exact zero, so
    attributing it to dropout would double-count.
    """
    safe_k = np.where(k == 0, 1.0, k)
    nb = nb2_logpmf(obs, per_copy * safe_k, phi)
    if p_drop is not None:
        log_keep = np.log1p(-p_drop)
        log_drop = np.log(p_drop)
        nb = np.where(
            obs == 0,
            np.logaddexp(log_drop, log_keep + nb),
            log_keep + nb,
        )
    point_mass = np.where(obs == 0, 0.0, -np.inf)
    return np.where(k == 0, point_mass, nb)


def posterior_k_moments(
    t7,
    cre,
    group_idx,
    cre_idx,
    draws: Mapping[str, np.ndarray],
    kmax: int,
    *,
    chunk: int = 400,
) -> PosteriorKMoments:
    """Posterior summaries of ``k`` and of the per-cell activity, over draws.

    The sd combines both sources of uncertainty by the law of total variance::

        Var[k | obs] = E_theta[ Var(k | obs, theta) ] + Var_theta[ E(k | obs, theta) ]

    i.e. the spread of the copy number given a fixed parameter draw, plus the
    spread induced by not knowing the parameters. Reporting only the first would
    understate the uncertainty of a rarely-observed cCRE.

    ``activity`` is the posterior mean of the per-cell enhancer activity. The
    cCRE channel is ``NB2(k * gamma, phi_cre)``, which *is* ``Poisson(k * gamma *
    G)`` with ``G ~ Gamma(phi_cre, phi_cre)``: the per-cell deviation from the
    cell type's activity is a latent variable the model already has. Its
    posterior is conjugate, ``G | cre, k ~ Gamma(phi + cre, phi + k * gamma)``,
    so the activity is ``gamma * E[G]`` marginalised over ``P(k | obs)``.

    That shrinks toward the cell-type activity when the evidence is thin and
    relaxes to the moment estimator ``cre / k`` when counts are large, and unlike
    the ratio it stays finite at ``cre = 0`` (returning a small positive number
    rather than a hard zero, which is the difference between "silent" and
    "no information").

    Parameters
    ----------
    t7, cre : (n_pairs,) observed counts.
    group_idx, cre_idx : (n_pairs,) indices into the cell-type and cCRE axes.
    draws
        Posterior arrays ``rho`` (D, S), ``a`` (D, J), ``log_gamma`` (D, S, J),
        ``beta_t7`` (D,), ``phi_t7`` (D,), ``phi_cre`` (D,). If the fit included
        measurement dropout, ``p_drop_t7`` / ``p_drop_cre`` (D,) must be present
        too -- omitting them silently evaluates a different model.
    kmax
        The truncation used *at fit time*. Using a different grid here changes
        the normalisation and therefore the answer.
    chunk
        Pairs per block. Peak memory is ``n_draws x chunk x (kmax + 1)``.
    """
    rho, a, log_gamma = draws["rho"], draws["a"], draws["log_gamma"]
    beta, phi_t7, phi_cre = draws["beta_t7"], draws["phi_t7"], draws["phi_cre"]

    group_idx = np.asarray(group_idx)
    cre_idx = np.asarray(cre_idx)
    n_pairs = len(group_idx)
    k = np.arange(0, kmax + 1, dtype=np.float64)
    log_k_factorial = gammaln(k + 1)
    mean = np.empty(n_pairs, dtype=np.float64)
    sd = np.empty(n_pairs, dtype=np.float64)
    p_infected = np.empty(n_pairs, dtype=np.float64)
    activity = np.empty(n_pairs, dtype=np.float64)

    def _per_draw(value):
        return None if value is None else np.asarray(value)[:, None, None]

    drop_t7 = _per_draw(draws.get("p_drop_t7"))
    drop_cre = _per_draw(draws.get("p_drop_cre"))

    for start in range(0, n_pairs, chunk):
        block = slice(start, start + chunk)
        s, j = group_idx[block], cre_idx[block]
        t7_block = np.asarray(t7[block], dtype=np.float64)[None, :, None]
        cre_block = np.asarray(cre[block], dtype=np.float64)[None, :, None]
        lam = (rho[:, s] * a[:, j])[:, :, None]
        gamma = np.exp(log_gamma[:, s, j])[:, :, None]
        log_posterior = (
            k * np.log(lam)
            - lam
            - log_k_factorial
            + channel_logprob(t7_block, k, beta[:, None, None], phi_t7[:, None, None], drop_t7)
            + channel_logprob(cre_block, k, gamma, phi_cre[:, None, None], drop_cre)
        )
        weights = np.exp(log_posterior - log_posterior.max(axis=-1, keepdims=True))
        weights /= weights.sum(axis=-1, keepdims=True)

        first = (weights * k).sum(axis=-1)                      # (D, C)
        second = (weights * k**2).sum(axis=-1)                  # (D, C)
        mean[block] = first.mean(axis=0)
        within = (second - first**2).mean(axis=0)
        between = first.var(axis=0)
        sd[block] = np.sqrt(np.maximum(within + between, 0.0))
        # P(k >= 1 | obs) = E_theta[ 1 - P(k = 0 | obs, theta) ], by the law of
        # total expectation -- the same average over draws as the mean, so the
        # parameter uncertainty is carried through here too.
        p_infected[block] = (1.0 - weights[..., 0]).mean(axis=0)

        # Per-cell activity, as a posterior rather than a ratio. NB2(k*gamma,
        # phi) IS Poisson(k*gamma*G) with G ~ Gamma(phi, phi), so the cell's
        # multiplicative deviation from its cell type's activity is a latent
        # variable the model already contains, and it is conjugate:
        #
        #     G | cre, k  ~  Gamma(phi + cre,  phi + k*gamma)
        #     activity    =  gamma * E[G | cre, k],  averaged over P(k | obs)
        #
        # phi > 0 keeps the denominator positive even at k = 0, where this
        # correctly returns gamma itself -- no counts, so the prior stands.
        phi = phi_cre[:, None, None]
        posterior_g = (phi + cre_block) / (phi + k * gamma)
        activity[block] = ((weights * posterior_g).sum(axis=-1) * gamma[:, :, 0]).mean(axis=0)
    return PosteriorKMoments(
        mean=mean, sd=sd, p_infected=p_infected, activity=activity
    )


def posterior_k_expectation(
    t7,
    cre,
    group_idx,
    cre_idx,
    draws: Mapping[str, np.ndarray],
    kmax: int,
    *,
    chunk: int = 400,
) -> np.ndarray:
    """``E[k | observed counts]`` per (cell, cCRE) pair, averaged over draws."""
    return posterior_k_moments(
        t7, cre, group_idx, cre_idx, draws, kmax, chunk=chunk
    ).mean
