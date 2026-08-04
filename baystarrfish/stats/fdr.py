"""Benjamini-Hochberg false discovery rate control.

Hand-rolled rather than delegated to ``statsmodels.stats.multitest`` so the
package's core has no statsmodels dependency. It is verified equal to
``multipletests(method="fdr_bh")`` to machine precision -- including exact ties,
all-zero p-values and the point mass at zero that posterior-tail p-values
produce -- by ``tests/test_fdr.py``.

This replaces seven separate definitions of ``bh_fdr`` that had accumulated
across the analysis scripts. They differed only in spelling (variable names,
ndarray vs Series vs DataFrame inputs), not in behaviour.
"""

from __future__ import annotations

import numpy as np

__all__ = ["bh_fdr"]


def bh_fdr(pvalues) -> np.ndarray:
    """Return BH-adjusted q-values, preserving shape and non-finite entries.

    Non-finite inputs (``NaN``, ``inf``) are excluded from the multiple-testing
    correction *and* from the denominator, and come back as ``NaN``: a pair that
    could not be tested must not inflate the correction for the pairs that could.

    Works on any shape. A 2-D input is corrected jointly over all finite entries,
    which is what pooling a (cell type x cCRE) p-value matrix means.
    """
    values = np.asarray(pvalues, dtype=float)
    output = np.full(values.shape, np.nan, dtype=float)
    valid = np.isfinite(values)
    if valid.any():
        valid_values = values[valid]
        order = np.argsort(valid_values)
        ranked = valid_values[order]
        # Step up from the largest p-value, taking a running minimum so the
        # q-values stay monotone in p.
        adjusted = np.minimum.accumulate(
            (ranked * len(ranked) / np.arange(1, len(ranked) + 1))[::-1]
        )[::-1]
        restored = np.empty_like(adjusted)
        restored[order] = np.clip(adjusted, 0.0, 1.0)
        output[valid] = restored
    return output
