"""Turning posteriors into calls: negative-control contrasts and FDR control.

Free of JAX and NumPyro -- the statistics operate on posterior draws already
written to disk, so this runs wherever the figures are made.
"""

from __future__ import annotations

from .baseline import NegativeControlBaseline, negative_control_log_baseline
from .fdr import bh_fdr
from .negative_control import negative_control_test

__all__ = [
    "NegativeControlBaseline",
    "bh_fdr",
    "negative_control_log_baseline",
    "negative_control_test",
]
