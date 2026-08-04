"""Shared fixtures. Importing baystarrfish first is what pins float64."""

from __future__ import annotations

import numpy as np
import pytest

import baystarrfish  # noqa: F401  -- sets JAX_ENABLE_X64 before any jnp array


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(20260804)


@pytest.fixture
def toy_counts(rng):
    """A small (cells x cCREs) dataset with the real data's sparsity regime."""
    n_cells, n_cre, n_group = 240, 6, 3
    group = np.repeat(np.arange(n_group), n_cells // n_group)
    lam = np.exp(rng.normal(-1.5, 0.3, size=(n_group, n_cre)))
    k = rng.poisson(lam[group])
    from baystarrfish.model.forward import sample_channel

    t7 = sample_channel(rng, k, 4.0, 3.0)
    cre = sample_channel(rng, k, 2.0, 3.0)
    return {
        "t7": t7.astype(np.int64),
        "cre": cre.astype(np.int64),
        "group": group.astype(np.int64),
        "n_group": n_group,
        "n_cre": n_cre,
    }
