"""Deprecated shim. The Bayesian model now lives in the ``baystarrfish`` package.

This module moved to the repository-root package ``baystarrfish`` so that it is
installable with its real dependencies (jax, numpyro) and importable without
``STARRFISH/__init__.py``'s torch / scvi-tools / cmdstanpy imports.

Migrate ``from STARRFISH import bayesian_hierarchical as bh`` to
``import baystarrfish as bh``; the public API is unchanged.
"""

from __future__ import annotations

import warnings

warnings.warn(
    "STARRFISH.bayesian_hierarchical has moved to the 'baystarrfish' package; "
    "import baystarrfish instead. This shim will be removed in a future release.",
    DeprecationWarning,
    stacklevel=2,
)

from baystarrfish import *  # noqa: F401,F403,E402
from baystarrfish import __all__ as _bsf_all  # noqa: E402

__all__ = list(_bsf_all)
