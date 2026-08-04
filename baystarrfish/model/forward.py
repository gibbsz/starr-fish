"""Canonical forward sampler for the BAYSTARRFISH generative model (NumPy).

The single encoding of "draw the latent infection state, then NB2 channel counts,
with exact zeros where there is no infection and optional measurement dropout".

Before this module existed the same three lines were re-derived in four places --
the posterior predictive check, the simulation-recovery test, the joint-dropout
simulation study, and a figure script -- and they had already drifted: one copy
omitted the dropout term entirely while being applied to dropout fits.

Sampling order is load-bearing. ``sample_channel`` draws the NB2 counts *before*
the dropout Bernoulli, matching the order the posterior predictive check has
always used, so a shared ``rng`` produces an identical stream.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

__all__ = ["nb2_sample", "sample_latent_multiplier", "sample_channel"]


def nb2_sample(rng: np.random.Generator, mean, conc):
    """Sample NB2(mean, conc) via Gamma-Poisson; ``mean`` array, ``conc`` scalar.

    Parameterised so that ``var = mean + mean**2 / conc``, matching
    :func:`baystarrfish.model.likelihood._nb2_logprob`. Non-positive means are
    clamped rather than rejected: callers pass ``mean = 0`` wherever the latent
    multiplier is zero and then overwrite those entries with an exact zero.
    """
    mean = np.where(mean <= 0, 1e-9, mean)
    rate = conc / mean
    lam = rng.gamma(shape=conc, scale=1.0 / rate)
    return rng.poisson(lam)


def sample_latent_multiplier(
    rng: np.random.Generator,
    infection_rate,
    infection_model: Literal["copy_number", "copy_number_dropout", "binary"] = "copy_number",
):
    """Draw the latent infection state that scales both channel means.

    ``copy_number`` / ``copy_number_dropout`` draw a Poisson copy count ``k``;
    ``binary`` draws a single shared infection gate with probability
    ``1 - exp(-infection_rate)`` (cloglog).
    """
    if infection_model in {"copy_number", "copy_number_dropout"}:
        return rng.poisson(infection_rate)
    if infection_model == "binary":
        return rng.binomial(1, -np.expm1(-infection_rate))
    raise ValueError(
        f"unsupported infection_model={infection_model}; available "
        "['binary', 'copy_number', 'copy_number_dropout']"
    )


def sample_channel(
    rng: np.random.Generator,
    latent_multiplier,
    per_copy,
    phi,
    p_drop: float | None = None,
):
    """Draw one observed channel given the latent infection state.

    Parameters
    ----------
    latent_multiplier : array
        Latent copy count ``k`` (or the 0/1 infection gate).
    per_copy : array or scalar
        Expected counts per infected copy -- ``beta_t7`` for the T7 channel,
        ``exp(log_gamma[group, cre])`` for the cCRE channel.
    phi : float
        NB2 dispersion (concentration) for this channel.
    p_drop : float, optional
        Zero-inflated measurement dropout probability. Applied only where
        ``latent_multiplier > 0``: an uninfected cell is already an exact zero,
        so dropout there is unidentifiable and must not consume probability.
    """
    counts = nb2_sample(rng, per_copy * latent_multiplier, phi)
    if p_drop is not None:
        dropped = rng.binomial(1, p_drop, size=counts.shape).astype(bool)
        counts = np.where((latent_multiplier > 0) & dropped, 0, counts)
    return np.where(latent_multiplier == 0, 0, counts)
