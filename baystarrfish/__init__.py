"""BAYSTARRFISH -- a STARR-FISH in a bay.

Bayesian hierarchical inference of cis-regulatory element (cCRE) activity from
single-cell STARR-FISH data. ``BAY`` carries the Bayes.

Generative model
----------------
::

    latent copies   k_{ij}   ~ Poisson(lambda_{s,j}),  lambda_{s,j} = rho_s * a_j
    T7 channel      t7_{ij}  | k ~ NB2(mean = k * beta_t7,      disp = phi_t7)
    cCRE channel    cre_{ij} | k ~ NB2(mean = k * gamma_{s,j},  disp = phi_cre)

with ``k = 0`` forcing both channels to exactly zero, optional zero-inflated
measurement dropout on each channel, a two-level (class -> subclass) hierarchy
on the infection rate ``rho`` and the activity ``gamma``, and an informative
nanopore-library prior on the per-cCRE abundance ``a``. The discrete ``k`` is
marginalised analytically over a truncated grid, so the target is a
continuous-parameter model that NumPyro fits by SVI or NUTS.

Quickstart
----------
::

    import baystarrfish as bsf

    data = bsf.CountData.from_anndata("scdata_..._CRE_T7CRE_NEWNEW.h5ad")
    fit = bsf.fit(data, infection_model="copy_number_dropout",
                  activity_model="direct", level="subclass")
    bsf.write_fit(fit, "results/bayesian", tag="subclass_joint_dropout")
    calls = bsf.negative_control_test(...)

Attribute access is lazy (PEP 562): ``import baystarrfish.data`` does not import
JAX or NumPyro, so the plotting/analysis environment never needs them.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

# Eager, and it must stay eager: this sets jax_enable_x64 process-wide. Deferring
# it behind the lazy __getattr__ below would leave float32 as the default dtype
# for any array created between `import baystarrfish` and the first model call.
# It imports `jax` only -- never `numpyro` -- so the data/stats layers stay light.
from . import _jax_setup as _jax_setup  # noqa: F401
from ._version import __version__

# name -> submodule providing it. Lazily resolved by __getattr__ below so that
# importing a leaf subpackage (e.g. baystarrfish.data) does not pull in JAX.
_EXPORTS: dict[str, str] = {
    # --- model: collapsed sufficient statistics -------------------------- #
    "CollapsedStats": "._bayesian_hierarchical",
    "build_sufficient_stats": "._bayesian_hierarchical",
    "summarize_evidence": "._bayesian_hierarchical",
    "choose_kmax": "._bayesian_hierarchical",
    "count_kmax_truncated": "._bayesian_hierarchical",
    "kmax_tail_mass": "._bayesian_hierarchical",
    # --- model: likelihood ----------------------------------------------- #
    "marginal_loglik": "._bayesian_hierarchical",
    "gauss_hermite_rule": "._bayesian_hierarchical",
    "cre_marginal_loglik": "._bayesian_hierarchical",
    "binary_infection_loglik": "._bayesian_hierarchical",
    # --- model: priors and model functions ------------------------------- #
    "ModelPriors": "._bayesian_hierarchical",
    "model_t7_classlevel": "._bayesian_hierarchical",
    "model_t7_full": "._bayesian_hierarchical",
    "model_t7_full_dropout": "._bayesian_hierarchical",
    "model_classlevel": "._bayesian_hierarchical",
    "model_full": "._bayesian_hierarchical",
    "model_classlevel_dropout": "._bayesian_hierarchical",
    "model_full_dropout": "._bayesian_hierarchical",
    "model_cre_conditional_subclass": "._bayesian_hierarchical",
    "model_cre_conditional_subclass_no_dropout": "._bayesian_hierarchical",
    "model_binary_t7_classlevel": "._bayesian_hierarchical",
    "model_binary_classlevel": "._bayesian_hierarchical",
    "model_binary_full": "._bayesian_hierarchical",
    "MODELS": "._bayesian_hierarchical",
    "COPY_NUMBER_DROPOUT_MODELS": "._bayesian_hierarchical",
    "BINARY_INFECTION_MODELS": "._bayesian_hierarchical",
    "MODEL_FAMILIES": "._bayesian_hierarchical",
    # --- inference -------------------------------------------------------- #
    "init_from_moments": "._bayesian_hierarchical",
    "init_cre_from_moments": "._bayesian_hierarchical",
    "fit_svi": "._bayesian_hierarchical",
    "fit_nuts": "._bayesian_hierarchical",
    "summarize_posterior": "._bayesian_hierarchical",
    "summarize_binary_infection": "._bayesian_hierarchical",
    "summarize_log_lambda_posterior": "._bayesian_hierarchical",
    "summarize_lognormal_infection": "._bayesian_hierarchical",
    "posterior_predictive_check": "._bayesian_hierarchical",
    "posterior_predictive_check_decoupled": "._bayesian_hierarchical",
    "run_model": "._bayesian_hierarchical",
    "run_decoupled_model": "._bayesian_hierarchical",
}

# Documented public aliases for the two entry points.
_ALIASES: dict[str, str] = {
    "fit": "run_model",
    "fit_decoupled": "run_decoupled_model",
}

__all__ = ["__version__", *sorted(_EXPORTS), *sorted(_ALIASES)]

if TYPE_CHECKING:  # pragma: no cover - import-time cost avoided at runtime
    from ._bayesian_hierarchical import (  # noqa: F401
        BINARY_INFECTION_MODELS,
        COPY_NUMBER_DROPOUT_MODELS,
        MODEL_FAMILIES,
        MODELS,
        CollapsedStats,
        ModelPriors,
        binary_infection_loglik,
        build_sufficient_stats,
        choose_kmax,
        count_kmax_truncated,
        cre_marginal_loglik,
        fit_nuts,
        fit_svi,
        gauss_hermite_rule,
        init_cre_from_moments,
        init_from_moments,
        kmax_tail_mass,
        marginal_loglik,
        posterior_predictive_check,
        posterior_predictive_check_decoupled,
        run_decoupled_model,
        run_model,
        summarize_binary_infection,
        summarize_evidence,
        summarize_log_lambda_posterior,
        summarize_lognormal_infection,
        summarize_posterior,
    )


def __getattr__(name: str) -> object:
    """Resolve public names on first access (PEP 562)."""
    target = _ALIASES.get(name, name)
    module_path = _EXPORTS.get(target)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    value = getattr(import_module(module_path, __name__), target)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(__all__)
