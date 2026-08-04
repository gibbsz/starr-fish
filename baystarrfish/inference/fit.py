"""Inference drivers: SVI (production) and NUTS (calibration only).

Production fits use ``SVI`` with an ``AutoNormal`` guide, Adam at ``lr=5e-3``,
30,000 steps and 1,000 posterior draws. ``fit_nuts`` is exact but only tractable
at class level; use it to sanity-check the variational approximation.

``_add_deterministics`` replays the model over the posterior draws with
``observe=False`` under ``jax.vmap``, which recovers ``log_gamma``, ``log_rho``
and the other deterministic sites without paying for the (M x Kmax) likelihood
factor 1,000 times.
"""

from __future__ import annotations

import numpy as np

# jax_enable_x64 must be set before the first jnp array; see baystarrfish._jax_setup.
from .. import _jax_setup as _jax_setup  # noqa: F401

import jax

import numpyro
from numpyro.infer import MCMC, NUTS, SVI, Predictive, Trace_ELBO
from numpyro.infer.autoguide import AutoLowRankMultivariateNormal, AutoNormal
from numpyro.infer.initialization import init_to_value

from ..model.collapse import CollapsedStats
from ..model.priors import ModelPriors


def fit_svi(model, stats: CollapsedStats, lib_size_centered, kmax: int,
            priors: ModelPriors = ModelPriors(), *, negative_control_mask=None,
            init_values: dict | None = None,
            num_steps: int = 20000, lr: float = 5e-3, guide: str = "AutoNormal",
            num_posterior: int = 1000, seed: int = 0):
    """Fit by stochastic variational inference.

    Returns ``(samples, info)`` where ``samples`` is a dict of posterior draws
    (drawn from the fitted guide) and ``info`` holds ``losses``, ``params`` and
    the ``guide`` object.
    """
    init_loc = init_to_value(values=init_values) if init_values else None
    guide_cls = {"AutoNormal": AutoNormal,
                 "AutoLowRankMultivariateNormal": AutoLowRankMultivariateNormal}[guide]
    guide_obj = guide_cls(model, init_loc_fn=init_loc) if init_loc else guide_cls(model)

    svi = SVI(model, guide_obj, numpyro.optim.Adam(lr), loss=Trace_ELBO())
    result = svi.run(jax.random.PRNGKey(seed), num_steps, stats, lib_size_centered,
                     kmax, negative_control_mask, priors)

    pred = Predictive(guide_obj, params=result.params, num_samples=num_posterior)
    samples = pred(jax.random.PRNGKey(seed + 1))
    # Recover deterministic sites by replaying the model on guide draws.
    samples = _add_deterministics(model, samples, stats, lib_size_centered, kmax,
                                  negative_control_mask, priors, seed + 2)
    return samples, {"losses": np.asarray(result.losses), "params": result.params, "guide": guide_obj}


def fit_nuts(model, stats: CollapsedStats, lib_size_centered, kmax: int,
             priors: ModelPriors = ModelPriors(), *, negative_control_mask=None,
             init_values: dict | None = None,
             num_warmup: int = 1000, num_samples: int = 1000, num_chains: int = 2,
             seed: int = 0):
    """Fit by NUTS. Feasible at class granularity; avoid on the full subclass model."""
    init_strategy = init_to_value(values=init_values) if init_values else None
    kernel = NUTS(model, init_strategy=init_strategy) if init_strategy else NUTS(model)
    mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples,
                num_chains=num_chains, progress_bar=True)
    mcmc.run(jax.random.PRNGKey(seed), stats, lib_size_centered, kmax,
             negative_control_mask, priors)
    # NUTS get_samples() omits deterministic sites; materialise them by replay.
    samples = _add_deterministics(model, mcmc.get_samples(), stats, lib_size_centered,
                                  kmax, negative_control_mask, priors, seed + 1)
    return samples, {"mcmc": mcmc}


def _add_deterministics(model, samples, stats, lib_size_centered, kmax,
                        negative_control_mask, priors, seed):
    """Replay the model under posterior draws to materialise deterministic sites."""
    from functools import partial
    from numpyro.handlers import substitute, trace, seed as seed_handler

    # observe=False skips the (M x Kmax) likelihood factor, so the vmap over all
    # posterior draws only materialises the cheap deterministic transforms
    # (log_rho/log_a/log_gamma) instead of a (n_draws x M x Kmax) tensor.
    det_model = partial(model, observe=False)

    def one(draw):
        tr = trace(seed_handler(substitute(det_model, draw), jax.random.PRNGKey(seed))).get_trace(
            stats, lib_size_centered, kmax, negative_control_mask, priors)
        return {k: site["value"] for k, site in tr.items() if site["type"] == "deterministic"}

    det = jax.vmap(one)(dict(samples))
    out = dict(samples)
    out.update(det)
    return out


__all__ = [
    "fit_svi",
    "fit_nuts",
]
