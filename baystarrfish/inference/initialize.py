"""Method-of-moments initialisation of the variational / MCMC starting point.

Avoids needing to run the torch EM (``T7CRE_*_DistributionEM`` in
``STARRFISH.utils``) to get a sane init: the moments of the collapsed statistics
pin down ``beta_t7``, the infection rates and the activity scale well enough that
SVI converges without a warm start.
"""

from __future__ import annotations

import numpy as np

from ..model.collapse import CollapsedStats
from ..model.priors import ModelPriors


def init_from_moments(stats: CollapsedStats, lib_size_centered, priors: ModelPriors,
                      level: str, channel: str, negative_control_mask=None,
                      activity_model: str = "direct") -> dict:
    """Crude method-of-moments init for the raw (non-centered) sites.

    Anchors ``beta_t7`` at the mean T7 among T7-positive rows (k~=1 under rare
    infection), ``rho`` from per-group T7-positive fractions, and CRE baseline
    ``alpha`` from mean CRE among CRE-positive rows. Returned dict is suitable
    for :func:`numpyro.infer.initialization.init_to_value`.
    """
    w = np.asarray(stats.weight)
    t7 = np.asarray(stats.counts["t7"])
    grp = np.asarray(stats.group)
    cre_idx = np.asarray(stats.cre)
    lib = np.asarray(lib_size_centered)

    pos = t7 > 0
    beta_t7 = float(np.average(t7[pos], weights=w[pos])) if pos.any() else 1.0
    beta_t7 = max(beta_t7, 1.0)

    # per-group T7-positive fraction -> rho_g ~ mean_j[ -log(1-p_{g,j}) / a_j ]
    a_j = np.exp(lib)  # mean(log a)=0 => a_j ~ relative abundance
    n_per_group = np.asarray(stats.n_per_group)
    pos_weight = np.where(pos, w, 0.0)
    n_pos_gj = np.zeros((stats.n_group, stats.n_cre))
    np.add.at(n_pos_gj, (grp, cre_idx), pos_weight)
    p_gj = n_pos_gj / np.maximum(n_per_group[:, None], 1)
    lam_gj = -np.log1p(-np.clip(p_gj, 0, 1 - 1e-9))
    with np.errstate(divide="ignore", invalid="ignore"):
        rho_g = np.nanmean(np.where(a_j[None, :] > 0, lam_gj / a_j[None, :], np.nan), axis=1)
    rho_g = np.clip(np.nan_to_num(rho_g, nan=np.exp(priors.mu_rho_loc)), 1e-6, None)
    log_rho_g = np.log(rho_g)
    mu_rho = float(np.mean(log_rho_g))

    init = {
        "tau_a": np.float64(priors.tau_a_scale * 0.5),
        "eps_a_raw": np.zeros(stats.n_cre),
        "beta_t7": np.float64(beta_t7),
        "phi_t7": np.float64(2.0),
        "mu_rho": np.float64(mu_rho),
        "sigma_u": np.float64(0.5),
        "u_raw": np.zeros(stats.n_group if level == "class" else int(stats.n_class)),
    }
    if level == "subclass":
        init["sigma_w"] = np.float64(0.5)
        init["w_raw"] = np.zeros(stats.n_group)

    if channel == "joint":
        cre = np.asarray(stats.counts["cre"])
        cpos = cre > 0
        gamma0 = float(np.average(cre[cpos], weights=w[cpos])) if cpos.any() else beta_t7
        if activity_model == "direct":
            if negative_control_mask is not None:
                raise ValueError(
                    "direct activity requires ordinary negative controls"
                )
            init.update({
                "mu_gamma": np.float64(np.log(max(gamma0, 1.0))),
                "sigma_gamma": np.float64(1.0),
                "gamma_raw": np.zeros((stats.n_group, stats.n_cre)),
                "phi_cre": np.float64(2.0),
            })
        else:
            if activity_model != "hierarchical":
                raise ValueError(f"unsupported activity_model={activity_model}")
            init.update({
                "mu_alpha": np.float64(np.log(max(gamma0, 1.0))),
                "sigma_alpha": np.float64(1.0),
                "alpha_raw": np.zeros(stats.n_cre),
                "sigma_eta": np.float64(0.5),
                "phi_cre": np.float64(2.0),
            })
            if level == "class":
                init["eta_raw"] = np.zeros((stats.n_group, stats.n_cre))
            else:
                init["eta_raw"] = np.zeros((int(stats.n_class), stats.n_cre))
                init["sigma_delta"] = np.float64(0.5)
                init["delta_raw"] = np.zeros((stats.n_group, stats.n_cre))
            if negative_control_mask is not None:
                init["alpha_neg"] = np.float64(np.log(max(gamma0, 1.0)))
                if level == "class":
                    init["eta_neg_raw"] = np.zeros(stats.n_group)
                else:
                    init["eta_neg_raw"] = np.zeros(int(stats.n_class))
                    init["delta_neg_raw"] = np.zeros(stats.n_group)
    return init


def init_cre_from_moments(
    stats: CollapsedStats,
    priors: ModelPriors,
    negative_control_mask=None,
    activity_model: str = "direct",
) -> dict:
    """Method-of-moments initial values for the CRE-only conditional model.

    Mirrors the activity branch of :func:`init_from_moments`; the two must agree on
    the site names each parameterisation samples, or SVI silently starts from the
    prior for the sites the init does not mention.
    """
    w = np.asarray(stats.weight)
    cre = np.asarray(stats.counts["cre"])
    cpos = cre > 0
    gamma0 = float(np.average(cre[cpos], weights=w[cpos])) if cpos.any() else 1.0
    gamma0 = max(gamma0, 1.0)
    init = {
        "phi_cre": np.float64(2.0),
        "p_drop_cre": np.float64(
            priors.p_drop_cre_alpha / (priors.p_drop_cre_alpha + priors.p_drop_cre_beta)
        ),
    }
    if activity_model == "direct":
        if negative_control_mask is not None:
            raise ValueError(
                "direct activity requires ordinary negative controls"
            )
        init.update({
            "mu_gamma": np.float64(np.log(gamma0)),
            "sigma_gamma": np.float64(1.0),
            "gamma_raw": np.zeros((stats.n_group, stats.n_cre)),
        })
        return init
    if activity_model != "hierarchical":
        raise ValueError(f"unsupported activity_model={activity_model}")
    init.update({
        "mu_alpha": np.float64(np.log(gamma0)),
        "sigma_alpha": np.float64(1.0),
        "alpha_raw": np.zeros(stats.n_cre),
        "sigma_eta": np.float64(0.5),
        "eta_raw": np.zeros((int(stats.n_class), stats.n_cre)),
        "sigma_delta": np.float64(0.5),
        "delta_raw": np.zeros((stats.n_group, stats.n_cre)),
    })
    if negative_control_mask is not None:
        init["alpha_neg"] = np.float64(np.log(gamma0))
        init["eta_neg_raw"] = np.zeros(int(stats.n_class))
        init["delta_neg_raw"] = np.zeros(stats.n_group)
    return init


__all__ = [
    "init_from_moments",
    "init_cre_from_moments",
]
