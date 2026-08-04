"""cCRE blacklisting and negative-control handling.

Three mutually exclusive ways to treat the annotated negative controls, selected
by ``negative_control_mode``:

``ordinary``
    Controls are modelled exactly like any other cCRE. Required by the ``direct``
    activity parameterisation, and what the production fit uses -- the downstream
    test then contrasts each target against the mean of the control posteriors.
``pooled``
    Controls share a single set of activity parameters inside the model
    (``alpha_neg`` / ``eta_neg`` / ``delta_neg``), tightening the reference at the
    cost of assuming the controls are exchangeable.
``ordinary_and_pooled``
    Controls stay ordinary *and* a synthetic cCRE is appended whose counts are
    the per-cell sums across controls, giving a single high-evidence reference
    column without removing the individual ones.
"""

from __future__ import annotations

from typing import Iterable, Literal

import numpy as np
import pandas as pd

from . import paths

#: Name of the synthetic cCRE appended in ``ordinary_and_pooled`` mode.
POOLED_NEGATIVE_CONTROL_NAME = "NEGATIVE_CONTROL_POOL"

NegativeControlMode = Literal["ordinary", "pooled", "ordinary_and_pooled"]

NEGATIVE_CONTROL_MODES: tuple[str, ...] = (
    "ordinary",
    "pooled",
    "ordinary_and_pooled",
)

__all__ = [
    "NEGATIVE_CONTROL_MODES",
    "POOLED_NEGATIVE_CONTROL_NAME",
    "NegativeControlMode",
    "build_pooled_negative_control",
    "cre_blacklist",
    "negative_control_names",
]


def cre_blacklist(columns: Iterable[str]) -> list[str]:
    """Return the manuscript base plus >20% barcode-mismatch blacklist."""
    mismatch = pd.read_csv(paths.mismatch_csv(), index_col=0)
    failed = mismatch.index[
        mismatch["MismatchPercent"].astype(float).gt(20)
    ].astype(str)
    available = set(map(str, columns))
    return sorted((set(paths.BASE_BLACKLIST) | set(failed)) & available)


def negative_control_names(
    cre_info: pd.DataFrame, blacklist: Iterable[str]
) -> list[str]:
    """Annotated negative-control cCREs that survived blacklisting."""
    blocked = set(map(str, blacklist))
    mask = cre_info["labeling_type"].astype(str).str.lower().eq("negative control")
    return [name for name in cre_info.index[mask].astype(str) if name not in blocked]


def build_pooled_negative_control(
    t7: np.ndarray,
    cre: np.ndarray,
    library_counts: np.ndarray,
    cre_names: list[str],
    negative_control_mask: np.ndarray,
    negative_controls: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], np.ndarray, dict]:
    """Append a synthetic cCRE holding the per-cell sums across the controls.

    Returns ``(t7, cre, library_counts, cre_names, model_mask, provenance)`` where
    ``model_mask`` selects only the appended column, so the pooled column -- and
    not the individual controls -- is what the model treats as a shared control.
    """
    if not negative_control_mask.any():
        raise ValueError("cannot append a pooled cCRE without negative controls")
    if POOLED_NEGATIVE_CONTROL_NAME in cre_names:
        raise ValueError(
            f"reserved pooled cCRE name already exists: {POOLED_NEGATIVE_CONTROL_NAME}"
        )
    pooled_t7 = t7[:, negative_control_mask].sum(axis=1, keepdims=True)
    pooled_cre = cre[:, negative_control_mask].sum(axis=1, keepdims=True)
    pooled_library_count = float(library_counts[negative_control_mask].sum())
    t7 = np.concatenate([t7, pooled_t7], axis=1)
    cre = np.concatenate([cre, pooled_cre], axis=1)
    library_counts = np.concatenate(
        [library_counts, np.asarray([pooled_library_count], dtype=np.float64)]
    )
    cre_names = [*cre_names, POOLED_NEGATIVE_CONTROL_NAME]
    model_mask = np.zeros(len(cre_names), dtype=bool)
    model_mask[-1] = True
    provenance = {
        "name": POOLED_NEGATIVE_CONTROL_NAME,
        "constituent_cre": negative_controls,
        "nanopore_count": pooled_library_count,
        "construction": "per-cell sums of T7 and CRE counts across constituents",
    }
    return t7, cre, library_counts, cre_names, model_mask, provenance
