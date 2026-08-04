"""Process-wide JAX configuration, isolated so it cannot be lost in a refactor.

Import this module **before** any ``jax.numpy`` array is created. Every
``baystarrfish`` submodule that touches ``jnp`` imports it first.

Double precision is not optional here. The latent copy-number marginal
(:func:`baystarrfish.model.likelihood.marginal_loglik`) mixes infection rates on
the order of ``1e-6`` with ``-inf`` point masses inside a ``logsumexp``; in
float32 that combination silently produces ``nan`` gradients and the SVI fit
diverges.
"""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)


def assert_x64_enabled() -> None:
    """Raise if double precision is not active (guards against import-order bugs)."""
    import jax.numpy as jnp

    dtype = jnp.zeros(1).dtype
    if dtype != jnp.float64:
        raise RuntimeError(
            f"baystarrfish requires jax_enable_x64; default float dtype is {dtype}. "
            "Something created a JAX array before baystarrfish was imported."
        )


__all__ = ["assert_x64_enabled"]
