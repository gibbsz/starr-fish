"""Reusable NumPyro sampling blocks shared by the model functions.

Each ``_sample_*`` block draws one group of latent variables in a non-centered
parameterisation (``*_raw ~ Normal(0, 1)`` scaled by a ``sigma_*``), and each
``_*obs_factor`` adds the collapsed log-likelihood as a single
``numpyro.factor``.

Site order matters: ``AutoNormal`` initialisation and the SVI RNG stream depend
on the order in which ``numpyro.sample`` is called, so these blocks must be
composed in the same sequence by every model function.
"""

from __future__ import annotations

# jax_enable_x64 must be set before the first jnp array; see baystarrfish._jax_setup.
from .. import _jax_setup as _jax_setup  # noqa: F401

import jax.numpy as jnp

import numpyro
import numpyro.distributions as dist

from .collapse import CollapsedStats
from .likelihood import (
    binary_infection_loglik,
    cre_marginal_loglik,
    marginal_loglik,
)
from .priors import ModelPriors


def _sample_abundance(lib_size_centered, priors: ModelPriors):
    """Latent log-abundance with informative ``lib_size`` prior, mean-zero constrained."""
    n_cre = lib_size_centered.shape[0]
    tau_a = numpyro.sample("tau_a", dist.HalfNormal(priors.tau_a_scale))
    eps_raw = numpyro.sample("eps_a_raw", dist.Normal(0.0, 1.0).expand([n_cre]).to_event(1))
    log_a = lib_size_centered + tau_a * eps_raw
    log_a = log_a - jnp.mean(log_a)              # fix shared scale: mean(log a) = 0
    return numpyro.deterministic("log_a", log_a)


def _sample_t7_params(priors: ModelPriors):
    beta_t7 = numpyro.sample("beta_t7", dist.LogNormal(priors.beta_t7_loc, priors.beta_t7_scale))
    phi_t7 = numpyro.sample("phi_t7", dist.HalfNormal(priors.phi_t7_scale))
    return beta_t7, phi_t7


def _sample_t7_dropout(priors: ModelPriors):
    return numpyro.sample(
        "p_drop_t7", dist.Beta(priors.p_drop_t7_alpha, priors.p_drop_t7_beta)
    )


def _sample_cre_dropout(priors: ModelPriors):
    return numpyro.sample(
        "p_drop_cre", dist.Beta(priors.p_drop_cre_alpha, priors.p_drop_cre_beta)
    )


def _sample_infection_classlevel(n_group, priors: ModelPriors):
    """log_rho_g = mu_rho + sigma_u * u_raw_g  (group == class)."""
    mu_rho = numpyro.sample("mu_rho", dist.Normal(priors.mu_rho_loc, priors.mu_rho_scale))
    sigma_u = numpyro.sample("sigma_u", dist.HalfNormal(priors.sigma_u_scale))
    u_raw = numpyro.sample("u_raw", dist.Normal(0.0, 1.0).expand([n_group]).to_event(1))
    log_rho = mu_rho + sigma_u * u_raw
    return numpyro.deterministic("log_rho", log_rho)


def _sample_infection_subclasslevel(n_subclass, n_class, class_of_subclass, priors: ModelPriors):
    """log_rho_s = mu_rho + sigma_u * u_class[class(s)] + sigma_w * w_raw_s.

    ``n_class`` must be a concrete Python int (static shape); deriving it from a
    traced ``class_of_subclass`` array would make ``.expand([n_class])`` dynamic.
    """
    mu_rho = numpyro.sample("mu_rho", dist.Normal(priors.mu_rho_loc, priors.mu_rho_scale))
    sigma_u = numpyro.sample("sigma_u", dist.HalfNormal(priors.sigma_u_scale))
    sigma_w = numpyro.sample("sigma_w", dist.HalfNormal(priors.sigma_w_scale))
    u_raw = numpyro.sample("u_raw", dist.Normal(0.0, 1.0).expand([n_class]).to_event(1))
    w_raw = numpyro.sample("w_raw", dist.Normal(0.0, 1.0).expand([n_subclass]).to_event(1))
    log_rho = mu_rho + sigma_u * u_raw[class_of_subclass] + sigma_w * w_raw
    return numpyro.deterministic("log_rho", log_rho)


def _sample_activity_direct(n_group, n_cre, priors: ModelPriors, negative_control_mask=None):
    """Exchangeable raw activity with no cCRE/class/subclass decomposition."""
    if negative_control_mask is not None:
        raise ValueError(
            "direct activity requires ordinary negative controls; pooled/shared "
            "negative-control parameters are not supported"
        )
    mu_gamma = numpyro.sample(
        "mu_gamma", dist.Normal(0.0, priors.mu_gamma_scale)
    )
    sigma_gamma = numpyro.sample(
        "sigma_gamma", dist.HalfNormal(priors.sigma_gamma_scale)
    )
    gamma_raw = numpyro.sample(
        "gamma_raw",
        dist.Normal(0.0, 1.0).expand([n_group, n_cre]).to_event(2),
    )
    return numpyro.deterministic(
        "log_gamma", mu_gamma + sigma_gamma * gamma_raw
    )


def _sample_activity_classlevel(
    n_group,
    n_cre,
    priors: ModelPriors,
    negative_control_mask=None,
    activity_model="direct",
):
    """log_gamma_{g,j} = alpha_j + eta_{g,j}; controls share alpha/eta by group."""
    if activity_model == "direct":
        return _sample_activity_direct(
            n_group, n_cre, priors, negative_control_mask
        )
    if activity_model != "hierarchical":
        raise ValueError(f"unsupported activity_model={activity_model}")
    mu_alpha = numpyro.sample("mu_alpha", dist.Normal(0.0, priors.mu_alpha_scale))
    sigma_alpha = numpyro.sample("sigma_alpha", dist.HalfNormal(priors.sigma_alpha_scale))
    alpha_raw = numpyro.sample("alpha_raw", dist.Normal(0.0, 1.0).expand([n_cre]).to_event(1))
    alpha = mu_alpha + sigma_alpha * alpha_raw
    sigma_eta = numpyro.sample("sigma_eta", dist.HalfNormal(priors.sigma_eta_scale))
    eta_raw = numpyro.sample("eta_raw", dist.Normal(0.0, 1.0).expand([n_group, n_cre]).to_event(2))
    eta = sigma_eta * eta_raw
    if negative_control_mask is not None:
        mask = jnp.asarray(negative_control_mask, dtype=bool)
        alpha_neg = numpyro.sample("alpha_neg", dist.Normal(mu_alpha, sigma_alpha))
        eta_neg_raw = numpyro.sample(
            "eta_neg_raw", dist.Normal(0.0, 1.0).expand([n_group]).to_event(1)
        )
        eta_neg = numpyro.deterministic("eta_neg", sigma_eta * eta_neg_raw)
        alpha = jnp.where(mask, alpha_neg, alpha)
        eta = jnp.where(mask[None, :], eta_neg[:, None], eta)
        numpyro.deterministic("log_gamma_neg", alpha_neg + eta_neg)
    alpha = numpyro.deterministic("alpha", alpha)
    eta = numpyro.deterministic("eta", eta)
    log_gamma = alpha[None, :] + eta
    return numpyro.deterministic("log_gamma", log_gamma)


def _sample_activity_subclasslevel(
    n_subclass,
    n_class,
    n_cre,
    class_of_subclass,
    priors: ModelPriors,
    negative_control_mask=None,
    activity_model="direct",
):
    """log_gamma_{s,j} = alpha_j + eta_{class(s),j} + delta_{s,j}.

    Negative-control cCREs share one alpha, one class-level pattern, and one
    subclass-level pattern, so they vary by class/subclass but not by
    negative-control cCRE identity.
    """
    if activity_model == "direct":
        return _sample_activity_direct(
            n_subclass, n_cre, priors, negative_control_mask
        )
    if activity_model != "hierarchical":
        raise ValueError(f"unsupported activity_model={activity_model}")
    mu_alpha = numpyro.sample("mu_alpha", dist.Normal(0.0, priors.mu_alpha_scale))
    sigma_alpha = numpyro.sample("sigma_alpha", dist.HalfNormal(priors.sigma_alpha_scale))
    alpha_raw = numpyro.sample("alpha_raw", dist.Normal(0.0, 1.0).expand([n_cre]).to_event(1))
    alpha = mu_alpha + sigma_alpha * alpha_raw
    sigma_eta = numpyro.sample("sigma_eta", dist.HalfNormal(priors.sigma_eta_scale))
    eta_raw = numpyro.sample("eta_raw", dist.Normal(0.0, 1.0).expand([n_class, n_cre]).to_event(2))
    eta = sigma_eta * eta_raw
    sigma_delta = numpyro.sample("sigma_delta", dist.HalfNormal(priors.sigma_delta_scale))
    delta_raw = numpyro.sample("delta_raw", dist.Normal(0.0, 1.0).expand([n_subclass, n_cre]).to_event(2))
    delta = sigma_delta * delta_raw
    if negative_control_mask is not None:
        mask = jnp.asarray(negative_control_mask, dtype=bool)
        alpha_neg = numpyro.sample("alpha_neg", dist.Normal(mu_alpha, sigma_alpha))
        eta_neg_raw = numpyro.sample(
            "eta_neg_raw", dist.Normal(0.0, 1.0).expand([n_class]).to_event(1)
        )
        eta_neg = numpyro.deterministic("eta_neg", sigma_eta * eta_neg_raw)
        delta_neg_raw = numpyro.sample(
            "delta_neg_raw", dist.Normal(0.0, 1.0).expand([n_subclass]).to_event(1)
        )
        delta_neg = numpyro.deterministic("delta_neg", sigma_delta * delta_neg_raw)
        alpha = jnp.where(mask, alpha_neg, alpha)
        eta = jnp.where(mask[None, :], eta_neg[:, None], eta)
        delta = jnp.where(mask[None, :], delta_neg[:, None], delta)
        numpyro.deterministic(
            "log_gamma_neg",
            alpha_neg + eta_neg[class_of_subclass] + delta_neg,
        )
    alpha = numpyro.deterministic("alpha", alpha)
    eta = numpyro.deterministic("eta", eta)
    delta = numpyro.deterministic("delta", delta)
    log_gamma = alpha[None, :] + eta[class_of_subclass, :] + delta
    return numpyro.deterministic("log_gamma", log_gamma)


def _obs_factor(
    stats: CollapsedStats,
    lam_row,
    beta_t7,
    phi_t7,
    gamma_row,
    phi_cre,
    kmax,
    p_drop_t7=None,
    p_drop_cre=None,
):
    """Add the weighted marginal log-likelihood to the joint density."""
    ll = marginal_loglik(
        stats,
        lam_row,
        beta_t7,
        phi_t7,
        gamma_row,
        phi_cre,
        kmax,
        p_drop_t7=p_drop_t7,
        p_drop_cre=p_drop_cre,
    )
    numpyro.factor("obs", jnp.sum(stats.weight * ll))


def _cre_conditional_obs_factor(
    stats: CollapsedStats,
    log_lambda_mean_row,
    log_lambda_sd_row,
    gh_nodes,
    gh_log_weights,
    gamma_row,
    phi_cre,
    p_drop_cre,
    kmax,
):
    """Add the CRE-only conditional likelihood to the joint density."""
    ll = cre_marginal_loglik(
        stats,
        log_lambda_mean_row,
        log_lambda_sd_row,
        gh_nodes,
        gh_log_weights,
        gamma_row,
        phi_cre,
        p_drop_cre,
        kmax,
    )
    numpyro.factor("obs", jnp.sum(stats.weight * ll))


def _binary_obs_factor(stats: CollapsedStats, infection_rate_row, beta_t7, phi_t7,
                       gamma_row, phi_cre):
    """Add the weighted shared-gate zero-inflated NB likelihood."""
    ll = binary_infection_loglik(
        stats, infection_rate_row, beta_t7, phi_t7, gamma_row, phi_cre)
    numpyro.factor("obs", jnp.sum(stats.weight * ll))

