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

# Eager, and it must stay eager: it requests jax_enable_x64 process-wide before
# anything can create a float32 array. It imports no JAX itself (see the module
# docstring), so the data / stats / io layers stay free of the inference stack.
from . import _jax_setup as _jax_setup  # noqa: F401
from ._version import __version__

# name -> submodule providing it. Lazily resolved by __getattr__ below so that
# importing a leaf subpackage (e.g. baystarrfish.data) does not pull in JAX.
_EXPORTS: dict[str, str] = {
    # --- model: collapsed sufficient statistics -------------------------- #
    "CollapsedStats": ".model.collapse",
    "build_sufficient_stats": ".model.collapse",
    "summarize_evidence": ".model.collapse",
    "choose_kmax": ".model.collapse",
    "count_kmax_truncated": ".model.collapse",
    "kmax_tail_mass": ".model.collapse",
    # --- model: likelihood ----------------------------------------------- #
    "marginal_loglik": ".model.likelihood",
    "gauss_hermite_rule": ".model.likelihood",
    "cre_marginal_loglik": ".model.likelihood",
    "binary_infection_loglik": ".model.likelihood",
    # --- model: priors and model functions ------------------------------- #
    "ModelPriors": ".model.priors",
    "model_t7_classlevel": ".model.models",
    "model_t7_full": ".model.models",
    "model_t7_full_dropout": ".model.models",
    "model_classlevel": ".model.models",
    "model_full": ".model.models",
    "model_classlevel_dropout": ".model.models",
    "model_full_dropout": ".model.models",
    "model_cre_conditional_subclass": ".model.models",
    "model_cre_conditional_subclass_no_dropout": ".model.models",
    "model_binary_t7_classlevel": ".model.models",
    "model_binary_classlevel": ".model.models",
    "model_binary_full": ".model.models",
    "MODELS": ".model.registry",
    "COPY_NUMBER_DROPOUT_MODELS": ".model.registry",
    "BINARY_INFECTION_MODELS": ".model.registry",
    "MODEL_FAMILIES": ".model.registry",
    # --- inference -------------------------------------------------------- #
    "init_from_moments": ".inference.initialize",
    "init_cre_from_moments": ".inference.initialize",
    "fit_svi": ".inference.fit",
    "fit_nuts": ".inference.fit",
    "summarize_posterior": ".inference.summarize",
    "summarize_binary_infection": ".inference.summarize",
    "summarize_log_lambda_posterior": ".inference.summarize",
    "summarize_lognormal_infection": ".inference.summarize",
    "posterior_predictive_check": ".inference.ppc",
    "posterior_predictive_check_decoupled": ".inference.ppc",
    "run_model": ".inference.run",
    "run_decoupled_model": ".inference.run",
    # --- data, statistics and serialisation (no JAX, no NumPyro) ----------- #
    "CountData": ".data.counts",
    "read_and_prepare_adata": ".data.anndata",
    "bh_fdr": ".stats.fdr",
    "negative_control_test": ".stats.negative_control",
    "write_fit": ".io.results",
    "read_fit": ".io.results",
    "load_gamma": ".io.results",
    "load_posterior_samples": ".io.results",
    "log": "._log",
}

# Subpackages, resolvable as attributes without being imported up front:
# `import baystarrfish as bsf` then `bsf.data.CountData` works, and `bsf.data`
# alone still costs no JAX import.
_SUBMODULES = ("data", "inference", "io", "model", "simulate", "stats")

# Documented public aliases for the two entry points.
_ALIASES: dict[str, str] = {
    "fit": "run_model",
    "fit_decoupled": "run_decoupled_model",
}

__all__ = ["__version__", *sorted(_EXPORTS), *sorted(_ALIASES), *_SUBMODULES]

if TYPE_CHECKING:  # pragma: no cover - import-time cost avoided at runtime
    from .inference import (  # noqa: F401
        fit_nuts,
        fit_svi,
        init_cre_from_moments,
        init_from_moments,
        posterior_predictive_check,
        posterior_predictive_check_decoupled,
        run_decoupled_model,
        run_model,
        summarize_binary_infection,
        summarize_log_lambda_posterior,
        summarize_lognormal_infection,
        summarize_posterior,
    )
    from .model import (  # noqa: F401
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
        gauss_hermite_rule,
        kmax_tail_mass,
        marginal_loglik,
        model_binary_classlevel,
        model_binary_full,
        model_binary_t7_classlevel,
        model_classlevel,
        model_classlevel_dropout,
        model_cre_conditional_subclass,
        model_cre_conditional_subclass_no_dropout,
        model_full,
        model_full_dropout,
        model_t7_classlevel,
        model_t7_full,
        model_t7_full_dropout,
        summarize_evidence,
    )


def __getattr__(name: str) -> object:
    """Resolve public names and subpackages on first access (PEP 562)."""
    from importlib import import_module

    if name in _SUBMODULES:
        value = import_module(f".{name}", __name__)
    else:
        target = _ALIASES.get(name, name)
        module_path = _EXPORTS.get(target)
        if module_path is None:
            raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
        value = getattr(import_module(module_path, __name__), target)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(__all__)
