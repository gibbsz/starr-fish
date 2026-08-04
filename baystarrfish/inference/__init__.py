"""Fitting, summarisation and posterior predictive checks.

:func:`run_model` and :func:`run_decoupled_model` are the entry points; the other
modules exist so each stage is testable in isolation.
"""

from __future__ import annotations

from .fit import fit_nuts, fit_svi
from .initialize import init_cre_from_moments, init_from_moments
from .ppc import posterior_predictive_check, posterior_predictive_check_decoupled
from .run import run_decoupled_model, run_model
from .summarize import (
    summarize_binary_infection,
    summarize_log_lambda_posterior,
    summarize_lognormal_infection,
    summarize_posterior,
)

__all__ = [
    "fit_nuts",
    "fit_svi",
    "init_cre_from_moments",
    "init_from_moments",
    "posterior_predictive_check",
    "posterior_predictive_check_decoupled",
    "run_decoupled_model",
    "run_model",
    "summarize_binary_infection",
    "summarize_log_lambda_posterior",
    "summarize_lognormal_infection",
    "summarize_posterior",
]
