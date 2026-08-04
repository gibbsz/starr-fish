"""Every registered model must trace, and the observe=False replay must work.

``_add_deterministics`` vmaps each model over 1,000 posterior draws with the
observation factor switched off; if any model function loses its ``observe``
keyword during a refactor the production fit dies only at the very end, after
hours of SVI. These tests cost a second and catch it at import time.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest
from numpyro import handlers

from baystarrfish.model.collapse import build_sufficient_stats
from baystarrfish.model.priors import ModelPriors
from baystarrfish.model.registry import (
    BINARY_INFECTION_MODELS,
    COPY_NUMBER_DROPOUT_MODELS,
    MODEL_FAMILIES,
    MODELS,
)

ALL_MODELS = [
    pytest.param(fn, id=f"{family}-{level}-{channel}")
    for family, registry in MODEL_FAMILIES.items()
    for (level, channel), fn in registry.items()
]


def _observation_factor(trace):
    """numpyro.factor appears as an observed sample site with a Unit fn."""
    for site in trace.values():
        if site["type"] == "sample" and type(site["fn"]).__name__ == "Unit":
            return site
    return None


def _toy_stats(n_group=3, n_cre=4, n_cells=90, seed=0):
    rng = np.random.default_rng(seed)
    group = np.repeat(np.arange(n_group), n_cells // n_group).astype(np.int64)
    t7 = rng.poisson(0.4, size=(n_cells, n_cre))
    cre = rng.poisson(0.2, size=(n_cells, n_cre))
    stats = build_sufficient_stats(
        {"t7": t7, "cre": cre}, group, n_group, n_cre,
        class_of_group=np.array([0, 0, 1], dtype=np.int64), n_class=2,
    ).to_jax()
    return stats, np.linspace(-0.2, 0.2, n_cre)


def test_the_registries_partition_by_infection_family():
    assert set(MODEL_FAMILIES) == {"copy_number", "copy_number_dropout", "binary"}
    assert MODEL_FAMILIES["copy_number"] is MODELS
    assert MODEL_FAMILIES["copy_number_dropout"] is COPY_NUMBER_DROPOUT_MODELS
    assert MODEL_FAMILIES["binary"] is BINARY_INFECTION_MODELS
    for registry in MODEL_FAMILIES.values():
        assert {"t7", "joint"} == {channel for _, channel in registry}


@pytest.mark.parametrize("model", ALL_MODELS)
def test_every_model_accepts_the_uniform_signature(model):
    params = inspect.signature(model).parameters
    for required in ("stats", "lib_size_centered", "kmax",
                     "negative_control_mask", "priors", "observe"):
        assert required in params, f"{model.__name__} lost '{required}'"
    assert params["observe"].default is True


@pytest.mark.parametrize("model", ALL_MODELS)
def test_every_model_traces_and_yields_finite_log_density(model):
    stats, lib = _toy_stats()
    with handlers.seed(rng_seed=0):
        trace = handlers.trace(model).get_trace(stats, lib, 8, None, ModelPriors())
    assert trace, f"{model.__name__} sampled nothing"
    for name, site in trace.items():
        if site["type"] == "sample" and name != "obs":
            assert np.isfinite(np.asarray(site["value"])).all(), f"{model.__name__}:{name}"
    # numpyro.factor records an observed sample site with a Unit distribution.
    assert _observation_factor(trace) is not None, (
        f"{model.__name__} never added an observation factor"
    )
    log_density = np.asarray(_observation_factor(trace)["fn"].log_factor)
    assert np.isfinite(log_density).all(), f"{model.__name__} log-density is not finite"


@pytest.mark.parametrize("model", ALL_MODELS)
def test_observe_false_drops_the_observation_factor(model):
    """This is what makes the deterministic-site replay affordable."""
    stats, lib = _toy_stats()
    with handlers.seed(rng_seed=0):
        trace = handlers.trace(model).get_trace(
            stats, lib, 8, None, ModelPriors(), observe=False
        )
    assert _observation_factor(trace) is None
    assert any(site["type"] == "deterministic" for site in trace.values()), (
        f"{model.__name__} exposes no deterministic sites to replay"
    )


def test_direct_activity_refuses_a_negative_control_mask():
    """Pooling controls is meaningless when activity is already exchangeable."""
    from baystarrfish.model.models import model_full

    stats, lib = _toy_stats()
    mask = np.array([True, False, False, False])
    with pytest.raises(ValueError, match="ordinary negative controls"):
        with handlers.seed(rng_seed=0):
            handlers.trace(model_full).get_trace(
                stats, lib, 8, mask, ModelPriors(), activity_model="direct"
            )
