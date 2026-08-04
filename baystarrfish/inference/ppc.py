"""Posterior predictive checks and the forward sampler used to generate them.

Replicates are drawn from the fitted posterior through the same generative path
as the likelihood (latent Poisson ``k``, NB2 channels, exact zeros at ``k = 0``,
optional dropout) and compared to the observed weighted count summaries.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np

from ..model.collapse import CollapsedStats
from ..model.forward import sample_channel, sample_latent_multiplier
from .summarize import _ci


def posterior_predictive_check(samples: Mapping[str, np.ndarray], stats: CollapsedStats,
                               lib_size_centered, kmax: int, level: str,
                               n_draws: int = 100, seed: int = 0,
                               infection_model: str = "copy_number") -> dict:
    """Compare observed vs posterior-predictive zero-fraction & mean-nonzero per channel.

    Draws the selected latent infection state, then channel counts, and aggregates
    the same statistics computed on the observed collapsed table.
    """
    rng = np.random.default_rng(seed)
    n_total = float(np.asarray(stats.weight).sum())

    obs = {}
    for name in stats.channels:
        c = np.asarray(stats.counts[name]); w = np.asarray(stats.weight)
        obs[name] = {"zero_fraction": float(w[c == 0].sum() / n_total),
                     "mean_nonzero": float(np.average(c[c > 0], weights=w[c > 0])) if (c > 0).any() else 0.0}

    log_rho = np.asarray(samples["log_rho"]); log_a = np.asarray(samples["log_a"])
    beta = np.asarray(samples["beta_t7"]); phi_t7 = np.asarray(samples["phi_t7"])
    p_drop_t7 = np.asarray(samples["p_drop_t7"]) if "p_drop_t7" in samples else None
    p_drop_cre = np.asarray(samples["p_drop_cre"]) if "p_drop_cre" in samples else None
    has_cre = "log_gamma" in samples
    if has_cre:
        log_gamma = np.asarray(samples["log_gamma"]); phi_cre = np.asarray(samples["phi_cre"])
    grp = np.asarray(stats.group); cre_idx = np.asarray(stats.cre); w = np.asarray(stats.weight)

    draw_ids = rng.integers(0, log_rho.shape[0], size=n_draws)
    rep = {name: {"zero_fraction": [], "mean_nonzero": []} for name in stats.channels}
    for d in draw_ids:
        infection_rate = np.exp(log_rho[d][grp] + log_a[d][cre_idx])
        latent_multiplier = sample_latent_multiplier(rng, infection_rate, infection_model)
        for name in stats.channels:
            if name == "t7":
                per_copy = beta[d]; phi = phi_t7[d]
                p_drop = None if p_drop_t7 is None else p_drop_t7[d]
            else:
                per_copy = np.exp(log_gamma[d][grp, cre_idx]); phi = phi_cre[d]
                p_drop = None if p_drop_cre is None else p_drop_cre[d]
            sim = sample_channel(rng, latent_multiplier, per_copy, phi, p_drop)
            rep[name]["zero_fraction"].append(w[sim == 0].sum() / n_total)
            nz = sim > 0
            rep[name]["mean_nonzero"].append(np.average(sim[nz], weights=w[nz]) if nz.any() else 0.0)

    summary = {}
    for name in stats.channels:
        summary[name] = {
            "obs": obs[name],
            "rep_zero_fraction": (float(np.mean(rep[name]["zero_fraction"])), *(_ci(np.array(rep[name]["zero_fraction"])))),
            "rep_mean_nonzero": (float(np.mean(rep[name]["mean_nonzero"])), *(_ci(np.array(rep[name]["mean_nonzero"])))),
        }
    return summary


def _weighted_channel_summary(counts, weight):
    return {
        "zero_fraction": float(weight[counts == 0].sum() / weight.sum()),
        "mean_nonzero": (
            float(np.average(counts[counts > 0], weights=weight[counts > 0]))
            if (counts > 0).any()
            else 0.0
        ),
    }


def _weighted_joint_summary(t7, cre, weight):
    total = float(weight.sum())
    return {
        "all_zero_fraction": float(weight[(t7 == 0) & (cre == 0)].sum() / total),
        "t7_only_fraction": float(weight[(t7 > 0) & (cre == 0)].sum() / total),
        "cre_only_fraction": float(weight[(t7 == 0) & (cre > 0)].sum() / total),
        "double_positive_fraction": float(weight[(t7 > 0) & (cre > 0)].sum() / total),
    }


def _rep_interval(values):
    values = np.asarray(values, dtype=np.float64)
    return (float(values.mean()), *(_ci(values)))


def posterior_predictive_check_decoupled(
    t7_samples: Mapping[str, np.ndarray],
    cre_samples: Mapping[str, np.ndarray],
    stats: CollapsedStats,
    n_draws: int = 100,
    seed: int = 0,
) -> dict:
    """Posterior predictive checks for the decoupled T7 and CRE stages."""
    rng = np.random.default_rng(seed)
    weight = np.asarray(stats.weight, dtype=np.float64)
    t7_obs = np.asarray(stats.counts["t7"])
    cre_obs = np.asarray(stats.counts["cre"])
    grp = np.asarray(stats.group)
    cre_idx = np.asarray(stats.cre)

    obs = {
        "t7": _weighted_channel_summary(t7_obs, weight),
        "cre": _weighted_channel_summary(cre_obs, weight),
        "joint": _weighted_joint_summary(t7_obs, cre_obs, weight),
    }
    rep = {
        "t7": {"zero_fraction": [], "mean_nonzero": []},
        "cre": {"zero_fraction": [], "mean_nonzero": []},
        "joint": {key: [] for key in obs["joint"]},
    }

    log_rho = np.asarray(t7_samples["log_rho"])
    log_a = np.asarray(t7_samples["log_a"])
    beta = np.asarray(t7_samples["beta_t7"])
    phi_t7 = np.asarray(t7_samples["phi_t7"])
    p_drop_t7 = np.asarray(t7_samples["p_drop_t7"]) if "p_drop_t7" in t7_samples else None
    log_gamma = np.asarray(cre_samples["log_gamma"])
    phi_cre = np.asarray(cre_samples["phi_cre"])
    p_drop_cre = np.asarray(cre_samples["p_drop_cre"]) if "p_drop_cre" in cre_samples else None

    t7_draws = rng.integers(0, log_rho.shape[0], size=n_draws)
    cre_draws = rng.integers(0, log_gamma.shape[0], size=n_draws)
    for d_t7, d_cre in zip(t7_draws, cre_draws):
        lam = np.exp(log_rho[d_t7][grp] + log_a[d_t7][cre_idx])
        k_t7 = rng.poisson(lam)
        k_cre = rng.poisson(lam)

        sim_t7 = sample_channel(
            rng, k_t7, beta[d_t7], phi_t7[d_t7],
            None if p_drop_t7 is None else p_drop_t7[d_t7],
        )
        gamma = np.exp(log_gamma[d_cre][grp, cre_idx])
        sim_cre = sample_channel(
            rng, k_cre, gamma, phi_cre[d_cre],
            None if p_drop_cre is None else p_drop_cre[d_cre],
        )

        for name, sim in (("t7", sim_t7), ("cre", sim_cre)):
            summary = _weighted_channel_summary(sim, weight)
            for key, value in summary.items():
                rep[name][key].append(value)
        joint = _weighted_joint_summary(sim_t7, sim_cre, weight)
        for key, value in joint.items():
            rep["joint"][key].append(value)

    return {
        name: {
            "obs": obs[name],
            **{f"rep_{key}": _rep_interval(values) for key, values in rep[name].items()},
        }
        for name in ("t7", "cre", "joint")
    }


__all__ = [
    "posterior_predictive_check",
    "posterior_predictive_check_decoupled",
]
