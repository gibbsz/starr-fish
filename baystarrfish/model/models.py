"""The NumPyro model functions, composed from :mod:`baystarrfish.model.blocks`.

All models share the signature ``(stats, lib_size_centered, kmax,
negative_control_mask, priors, [activity_model], observe=True)``. Passing
``observe=False`` skips the (M x Kmax) observation factor, which is what lets
:func:`baystarrfish.inference.fit._add_deterministics` ``jax.vmap`` a cheap
replay over all posterior draws to recover the deterministic sites.

Three families, selected by ``infection_model``:

``copy_number``
    latent Poisson copy count, marginalised.
``copy_number_dropout``
    as above plus zero-inflated per-channel measurement dropout.
``binary``
    a shared Bernoulli infection gate instead of a copy count.

Two activity parameterisations, selected by ``activity_model``: ``hierarchical``
(``log_gamma = alpha_j + eta_{class,j} + delta_{subclass,j}``) and ``direct``
(exchangeable over the full subclass x cCRE matrix; used for the production fit).
"""

from __future__ import annotations

# jax_enable_x64 must be set before the first jnp array; see baystarrfish._jax_setup.
from .. import _jax_setup as _jax_setup  # noqa: F401

import jax.numpy as jnp

import numpyro
import numpyro.distributions as dist

from .blocks import (
    _binary_obs_factor,
    _cre_conditional_obs_factor,
    _obs_factor,
    _sample_abundance,
    _sample_activity_classlevel,
    _sample_activity_subclasslevel,
    _sample_cre_dropout,
    _sample_infection_classlevel,
    _sample_infection_subclasslevel,
    _sample_t7_dropout,
    _sample_t7_params,
)
from .collapse import CollapsedStats
from .priors import ModelPriors


def model_t7_classlevel(stats: CollapsedStats, lib_size_centered, kmax: int,
                        negative_control_mask=None,
                        priors: ModelPriors = ModelPriors(), observe: bool = True):
    """Stage-1 T7-only infection calibration (group == class). Fits rho, a, beta_t7, phi_t7.

    ``observe=False`` skips the (M x Kmax) likelihood factor — used to cheaply replay
    the deterministic sites under posterior draws without re-materialising the tensor.
    """
    log_a = _sample_abundance(lib_size_centered, priors)
    log_rho = _sample_infection_classlevel(stats.n_group, priors)
    beta_t7, phi_t7 = _sample_t7_params(priors)
    if observe:
        lam_row = jnp.exp(log_rho[stats.group] + log_a[stats.cre])
        _obs_factor(stats, lam_row, beta_t7, phi_t7, None, None, kmax)


def model_t7_full(stats: CollapsedStats, lib_size_centered, kmax: int,
                  negative_control_mask=None,
                  priors: ModelPriors = ModelPriors(), observe: bool = True):
    """T7-only infection calibration at subclass granularity."""
    assert stats.class_of_group is not None, "model_t7_full needs class_of_group on stats"
    n_class = int(stats.n_class)
    log_a = _sample_abundance(lib_size_centered, priors)
    log_rho = _sample_infection_subclasslevel(
        stats.n_group, n_class, stats.class_of_group, priors
    )
    beta_t7, phi_t7 = _sample_t7_params(priors)
    if observe:
        lam_row = jnp.exp(log_rho[stats.group] + log_a[stats.cre])
        _obs_factor(stats, lam_row, beta_t7, phi_t7, None, None, kmax)


def model_t7_full_dropout(stats: CollapsedStats, lib_size_centered, kmax: int,
                          negative_control_mask=None,
                          priors: ModelPriors = ModelPriors(), observe: bool = True):
    """T7-only subclass infection calibration with one global T7 dropout rate."""
    assert stats.class_of_group is not None, "model_t7_full_dropout needs class_of_group on stats"
    n_class = int(stats.n_class)
    log_a = _sample_abundance(lib_size_centered, priors)
    log_rho = _sample_infection_subclasslevel(
        stats.n_group, n_class, stats.class_of_group, priors
    )
    beta_t7, phi_t7 = _sample_t7_params(priors)
    p_drop_t7 = _sample_t7_dropout(priors)
    if observe:
        lam_row = jnp.exp(log_rho[stats.group] + log_a[stats.cre])
        _obs_factor(
            stats, lam_row, beta_t7, phi_t7, None, None, kmax,
            p_drop_t7=p_drop_t7,
        )


def model_classlevel(stats: CollapsedStats, lib_size_centered, kmax: int,
                     negative_control_mask=None,
                     priors: ModelPriors = ModelPriors(), activity_model="hierarchical",
                     observe: bool = True):
    """Stage-2 joint CRE+T7 model at class granularity (group == class)."""
    log_a = _sample_abundance(lib_size_centered, priors)
    log_rho = _sample_infection_classlevel(stats.n_group, priors)
    beta_t7, phi_t7 = _sample_t7_params(priors)
    log_gamma = _sample_activity_classlevel(
        stats.n_group, stats.n_cre, priors, negative_control_mask, activity_model
    )
    phi_cre = numpyro.sample("phi_cre", dist.HalfNormal(priors.phi_cre_scale))
    if observe:
        lam_row = jnp.exp(log_rho[stats.group] + log_a[stats.cre])
        gamma_row = jnp.exp(log_gamma[stats.group, stats.cre])
        _obs_factor(stats, lam_row, beta_t7, phi_t7, gamma_row, phi_cre, kmax)


def model_full(stats: CollapsedStats, lib_size_centered, kmax: int,
               negative_control_mask=None,
               priors: ModelPriors = ModelPriors(), activity_model="hierarchical",
               observe: bool = True):
    """Stage-3 full model: subclass nested in class (group == subclass)."""
    assert stats.class_of_group is not None, "model_full needs class_of_group on stats"
    n_class = int(stats.n_class)
    log_a = _sample_abundance(lib_size_centered, priors)
    log_rho = _sample_infection_subclasslevel(stats.n_group, n_class, stats.class_of_group, priors)
    beta_t7, phi_t7 = _sample_t7_params(priors)
    log_gamma = _sample_activity_subclasslevel(
        stats.n_group, n_class, stats.n_cre, stats.class_of_group, priors,
        negative_control_mask, activity_model)
    phi_cre = numpyro.sample("phi_cre", dist.HalfNormal(priors.phi_cre_scale))
    if observe:
        lam_row = jnp.exp(log_rho[stats.group] + log_a[stats.cre])
        gamma_row = jnp.exp(log_gamma[stats.group, stats.cre])
        _obs_factor(stats, lam_row, beta_t7, phi_t7, gamma_row, phi_cre, kmax)


def model_classlevel_dropout(stats: CollapsedStats, lib_size_centered, kmax: int,
                             negative_control_mask=None,
                             priors: ModelPriors = ModelPriors(), activity_model="hierarchical",
                             observe: bool = True):
    """Joint class model with global T7 and CRE zero-inflated dropout rates."""
    log_a = _sample_abundance(lib_size_centered, priors)
    log_rho = _sample_infection_classlevel(stats.n_group, priors)
    beta_t7, phi_t7 = _sample_t7_params(priors)
    p_drop_t7 = _sample_t7_dropout(priors)
    log_gamma = _sample_activity_classlevel(
        stats.n_group, stats.n_cre, priors, negative_control_mask, activity_model
    )
    phi_cre = numpyro.sample("phi_cre", dist.HalfNormal(priors.phi_cre_scale))
    p_drop_cre = _sample_cre_dropout(priors)
    if observe:
        lam_row = jnp.exp(log_rho[stats.group] + log_a[stats.cre])
        gamma_row = jnp.exp(log_gamma[stats.group, stats.cre])
        _obs_factor(
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


def model_full_dropout(stats: CollapsedStats, lib_size_centered, kmax: int,
                       negative_control_mask=None,
                       priors: ModelPriors = ModelPriors(), activity_model="hierarchical",
                       observe: bool = True):
    """Joint subclass model with global T7 and CRE zero-inflated dropout rates."""
    assert stats.class_of_group is not None, "model_full_dropout needs class_of_group on stats"
    n_class = int(stats.n_class)
    log_a = _sample_abundance(lib_size_centered, priors)
    log_rho = _sample_infection_subclasslevel(stats.n_group, n_class, stats.class_of_group, priors)
    beta_t7, phi_t7 = _sample_t7_params(priors)
    p_drop_t7 = _sample_t7_dropout(priors)
    log_gamma = _sample_activity_subclasslevel(
        stats.n_group, n_class, stats.n_cre, stats.class_of_group, priors,
        negative_control_mask, activity_model)
    phi_cre = numpyro.sample("phi_cre", dist.HalfNormal(priors.phi_cre_scale))
    p_drop_cre = _sample_cre_dropout(priors)
    if observe:
        lam_row = jnp.exp(log_rho[stats.group] + log_a[stats.cre])
        gamma_row = jnp.exp(log_gamma[stats.group, stats.cre])
        _obs_factor(
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


def model_cre_conditional_subclass(
    stats: CollapsedStats,
    log_lambda_mean,
    log_lambda_sd,
    gh_nodes,
    gh_log_weights,
    kmax: int,
    negative_control_mask=None,
    priors: ModelPriors = ModelPriors(),
    observe: bool = True,
):
    """CRE-only subclass activity model conditioned on T7 infection posterior."""
    assert stats.class_of_group is not None, "model_cre_conditional_subclass needs class_of_group"
    n_class = int(stats.n_class)
    log_gamma = _sample_activity_subclasslevel(
        stats.n_group, n_class, stats.n_cre, stats.class_of_group, priors, negative_control_mask
    )
    phi_cre = numpyro.sample("phi_cre", dist.HalfNormal(priors.phi_cre_scale))
    p_drop_cre = _sample_cre_dropout(priors)
    if observe:
        gamma_row = jnp.exp(log_gamma[stats.group, stats.cre])
        _cre_conditional_obs_factor(
            stats,
            log_lambda_mean[stats.group, stats.cre],
            log_lambda_sd[stats.group, stats.cre],
            gh_nodes,
            gh_log_weights,
            gamma_row,
            phi_cre,
            p_drop_cre,
            kmax,
        )


def model_cre_conditional_subclass_no_dropout(
    stats: CollapsedStats,
    log_lambda_mean,
    log_lambda_sd,
    gh_nodes,
    gh_log_weights,
    kmax: int,
    negative_control_mask=None,
    priors: ModelPriors = ModelPriors(),
    observe: bool = True,
):
    """CRE-only subclass activity model conditioned on T7 posterior, without dropout."""
    assert stats.class_of_group is not None, "model_cre_conditional_subclass_no_dropout needs class_of_group"
    n_class = int(stats.n_class)
    log_gamma = _sample_activity_subclasslevel(
        stats.n_group, n_class, stats.n_cre, stats.class_of_group, priors, negative_control_mask
    )
    phi_cre = numpyro.sample("phi_cre", dist.HalfNormal(priors.phi_cre_scale))
    if observe:
        gamma_row = jnp.exp(log_gamma[stats.group, stats.cre])
        _cre_conditional_obs_factor(
            stats,
            log_lambda_mean[stats.group, stats.cre],
            log_lambda_sd[stats.group, stats.cre],
            gh_nodes,
            gh_log_weights,
            gamma_row,
            phi_cre,
            None,
            kmax,
        )


def model_binary_t7_classlevel(stats: CollapsedStats, lib_size_centered, kmax: int,
                               negative_control_mask=None,
                               priors: ModelPriors = ModelPriors(), observe: bool = True):
    """T7-only class model with a shared binary infection event."""
    log_a = _sample_abundance(lib_size_centered, priors)
    log_rho = _sample_infection_classlevel(stats.n_group, priors)
    beta_t7, phi_t7 = _sample_t7_params(priors)
    if observe:
        infection_rate_row = jnp.exp(log_rho[stats.group] + log_a[stats.cre])
        _binary_obs_factor(stats, infection_rate_row, beta_t7, phi_t7, None, None)


def model_binary_classlevel(stats: CollapsedStats, lib_size_centered, kmax: int,
                            negative_control_mask=None,
                            priors: ModelPriors = ModelPriors(), activity_model="hierarchical",
                            observe: bool = True):
    """Joint class model with a shared binary infection event."""
    log_a = _sample_abundance(lib_size_centered, priors)
    log_rho = _sample_infection_classlevel(stats.n_group, priors)
    beta_t7, phi_t7 = _sample_t7_params(priors)
    log_gamma = _sample_activity_classlevel(
        stats.n_group, stats.n_cre, priors, negative_control_mask, activity_model
    )
    phi_cre = numpyro.sample("phi_cre", dist.HalfNormal(priors.phi_cre_scale))
    if observe:
        infection_rate_row = jnp.exp(log_rho[stats.group] + log_a[stats.cre])
        gamma_row = jnp.exp(log_gamma[stats.group, stats.cre])
        _binary_obs_factor(stats, infection_rate_row, beta_t7, phi_t7, gamma_row, phi_cre)


def model_binary_full(stats: CollapsedStats, lib_size_centered, kmax: int,
                      negative_control_mask=None,
                      priors: ModelPriors = ModelPriors(), activity_model="hierarchical",
                      observe: bool = True):
    """Joint subclass model with a shared binary infection event."""
    assert stats.class_of_group is not None, "model_binary_full needs class_of_group on stats"
    n_class = int(stats.n_class)
    log_a = _sample_abundance(lib_size_centered, priors)
    log_rho = _sample_infection_subclasslevel(stats.n_group, n_class, stats.class_of_group, priors)
    beta_t7, phi_t7 = _sample_t7_params(priors)
    log_gamma = _sample_activity_subclasslevel(
        stats.n_group, n_class, stats.n_cre, stats.class_of_group, priors,
        negative_control_mask, activity_model)
    phi_cre = numpyro.sample("phi_cre", dist.HalfNormal(priors.phi_cre_scale))
    if observe:
        infection_rate_row = jnp.exp(log_rho[stats.group] + log_a[stats.cre])
        gamma_row = jnp.exp(log_gamma[stats.group, stats.cre])
        _binary_obs_factor(stats, infection_rate_row, beta_t7, phi_t7, gamma_row, phi_cre)


__all__ = [
    "model_t7_classlevel",
    "model_t7_full",
    "model_t7_full_dropout",
    "model_classlevel",
    "model_full",
    "model_classlevel_dropout",
    "model_full_dropout",
    "model_cre_conditional_subclass",
    "model_cre_conditional_subclass_no_dropout",
    "model_binary_t7_classlevel",
    "model_binary_classlevel",
    "model_binary_full",
]
