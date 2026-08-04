"""Process-wide JAX configuration, isolated so it cannot be lost in a refactor.

Double precision is not optional here. The latent copy-number marginal
(:func:`baystarrfish.model.likelihood.marginal_loglik`) mixes infection rates on
the order of ``1e-6`` with ``-inf`` point masses inside a ``logsumexp``; in
float32 that combination silently produces ``nan`` gradients and the SVI fit
diverges.

It is requested through ``JAX_ENABLE_X64`` rather than ``jax.config.update`` so
that importing ``baystarrfish`` does **not** import JAX. Two reasons:

* the data / stats / io layers then work in an environment without JAX at all;
* a caller that sets ``JAX_PLATFORMS=cpu`` at runtime -- as the ``--cpu`` flag of
  the fitting runners does -- must be able to do so before the backend is
  chosen, which importing JAX at package-import time would pre-empt.

``setdefault`` so an explicit ``JAX_ENABLE_X64=0`` from the caller still wins;
the assertion below is what catches the consequences.

Every module that touches ``jnp`` imports this one first.
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("JAX_ENABLE_X64", "1")

if "jax" in sys.modules:  # already imported: the env var is too late, set it directly
    sys.modules["jax"].config.update("jax_enable_x64", True)

__all__ = ["assert_x64_enabled"]


def assert_x64_enabled() -> None:
    """Raise if double precision is not active (guards against import-order bugs)."""
    import jax.numpy as jnp

    dtype = jnp.zeros(1).dtype
    if dtype != jnp.float64:
        raise RuntimeError(
            f"baystarrfish requires jax_enable_x64; default float dtype is {dtype}. "
            "Set JAX_ENABLE_X64=1, or import baystarrfish before creating any JAX array."
        )
