"""The BAYSTARRFISH generative model: priors, likelihood and NumPyro models.

Layered strictly one way, no cycles::

    priors -> collapse -> likelihood -> blocks -> models -> registry

Importing this subpackage pulls in NumPyro. If you only need to load or reshape
data, import :mod:`baystarrfish.data` instead.
"""

from __future__ import annotations

from .collapse import (
    CollapsedStats,
    build_sufficient_stats,
    choose_kmax,
    count_kmax_truncated,
    kmax_tail_mass,
    summarize_evidence,
)
from .likelihood import (
    binary_infection_loglik,
    cre_marginal_loglik,
    gauss_hermite_rule,
    marginal_loglik,
)
from .models import (
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
)
from .priors import ModelPriors
from .registry import (
    BINARY_INFECTION_MODELS,
    COPY_NUMBER_DROPOUT_MODELS,
    MODEL_FAMILIES,
    MODELS,
)

__all__ = [
    "BINARY_INFECTION_MODELS",
    "COPY_NUMBER_DROPOUT_MODELS",
    "CollapsedStats",
    "MODELS",
    "MODEL_FAMILIES",
    "ModelPriors",
    "binary_infection_loglik",
    "build_sufficient_stats",
    "choose_kmax",
    "count_kmax_truncated",
    "cre_marginal_loglik",
    "gauss_hermite_rule",
    "kmax_tail_mass",
    "marginal_loglik",
    "model_binary_classlevel",
    "model_binary_full",
    "model_binary_t7_classlevel",
    "model_classlevel",
    "model_classlevel_dropout",
    "model_cre_conditional_subclass",
    "model_cre_conditional_subclass_no_dropout",
    "model_full",
    "model_full_dropout",
    "model_t7_classlevel",
    "model_t7_full",
    "model_t7_full_dropout",
    "summarize_evidence",
]
