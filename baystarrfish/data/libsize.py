"""Nanopore per-cCRE library abundance -- the informative prior on ``log a``.

The AAV library is not equimolar: some cCREs are represented an order of
magnitude more often than others, which confounds infection rate with abundance.
Long-read counts of the plasmid pool pin the abundance down, so the model can
attribute the remaining variation to infection and activity.

Counts enter the model as ``log1p(counts)`` and are mean-centered inside
:func:`baystarrfish.inference.run.run_model`, which fixes the otherwise
unidentifiable shared scale between ``rho`` and ``a``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from . import paths

__all__ = ["library_counts", "library_size_log"]


def library_counts(
    cre_names: Sequence[str], path: Path | str | None = None
) -> np.ndarray:
    """Nanopore counts aligned to ``cre_names``; unlisted cCREs get zero.

    A missing cCRE is a genuine zero (absent from the sequenced pool), not a
    missing value, so ``fill_value=0`` is correct rather than lenient.
    """
    frame = pd.read_csv(path or paths.libsize_csv(), index_col=0)
    frame.index = frame.index.astype(str)
    aligned = frame.reindex(pd.Index(cre_names, dtype=str), fill_value=0)
    return aligned["counts"].to_numpy(dtype=np.float64)


def library_size_log(
    cre_names: Sequence[str], path: Path | str | None = None
) -> np.ndarray:
    """``log1p`` of :func:`library_counts`, the form the model consumes."""
    return np.log1p(library_counts(cre_names, path))
