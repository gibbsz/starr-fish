"""Stream per-cell ``obsm`` count matrices into subclass-by-cCRE totals.

Both transcript species live in the H5AD as an ``obsm`` *group* holding one
1-D per-cell dataset per cCRE (``obsm/CRE/CRE001``, ``obsm/T7CRE/CRE001``, ...).
Summing them over the cells of each subclass is the raw-count aggregation every
export and every count-concordance figure needs, and doing it column by column
through ``h5py`` keeps peak memory at one cCRE instead of the whole matrix.

This module is the single definition of that aggregation. It also returns the
two per-group facts that always travel with the totals -- the ``class_name`` a
subclass belongs to and its cell count -- because they come from the same
``obs`` codes and would otherwise cost a second open of a multi-GB file.

    from baystarrfish.data.grouped_counts import read_grouped_counts
    counts = read_grouped_counts(h5ad, posterior_groups, cre_names)
    counts.totals["CRE"]    # (n_groups, n_cres) float64
    counts.totals["T7CRE"]
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import h5py
import numpy as np
import pandas as pd

from ..io.results import decode_strings
from .anndata import normalize_celltype_labels

__all__ = [
    "DEFAULT_COUNT_KEYS",
    "GroupedCounts",
    "grouped_obsm_totals",
    "read_grouped_counts",
]

#: The two transcript species, in the order the exports report them.
DEFAULT_COUNT_KEYS: tuple[str, ...] = ("CRE", "T7CRE")

#: Datasets renamed across H5AD vintages. Results are keyed by the *requested*
#: name regardless of which alias the file actually carries.
_OBSM_ALIASES: Mapping[str, tuple[str, ...]] = {
    "CRE": ("CRE",),
    "T7CRE": ("T7CRE", "T7"),
}

_SUBCLASS_COLUMN = "subclass_name"
_CLASS_COLUMN = "class_name"


@dataclass(frozen=True)
class GroupedCounts:
    """Subclass-by-cCRE count totals plus the per-group facts that go with them.

    ``groups`` and ``cres`` echo the requested axes verbatim -- no sorting -- so
    every array in ``totals`` is aligned to the caller's order.
    """

    groups: np.ndarray
    cres: np.ndarray
    totals: dict[str, np.ndarray]
    group_classes: np.ndarray
    group_cell_counts: np.ndarray

    def frame(self, key: str) -> pd.DataFrame:
        """One species' totals as a labelled subclass x cCRE frame."""
        return pd.DataFrame(
            self.totals[key],
            index=pd.Index(self.groups, name="subclass"),
            columns=self.cres,
        )


def _normalized_categories(column: h5py.Group) -> tuple[np.ndarray, np.ndarray]:
    """Decode a categorical ``obs`` column into normalized categories and codes."""
    categories = normalize_celltype_labels(
        pd.Series(decode_strings(column["categories"][...]))
    ).to_numpy()
    return categories, column["codes"][...].astype(np.int64)


def _class_of_subclass(
    subclass_codes: np.ndarray, class_codes: np.ndarray, n_subclasses: int
) -> np.ndarray:
    """Map every subclass code to its unique class code, or -1 if unobserved."""
    mapping = np.full(n_subclasses, -1, dtype=np.int64)
    for subclass_idx, class_idx in np.unique(
        np.column_stack([subclass_codes, class_codes]), axis=0
    ):
        if mapping[subclass_idx] not in {-1, class_idx}:
            raise ValueError(
                f"subclass code {subclass_idx} maps to more than one class_name"
            )
        mapping[subclass_idx] = class_idx
    return mapping


def _resolve_obsm(obsm: h5py.Group, key: str) -> h5py.Group:
    for alias in _OBSM_ALIASES.get(key, (key,)):
        if alias in obsm:
            candidate = obsm[alias]
            if not isinstance(candidate, h5py.Group):
                raise TypeError(
                    f"obsm/{alias} is a dataset, not a per-cCRE group; "
                    "this reader only handles the column-per-cCRE layout"
                )
            return candidate
    raise KeyError(
        f"H5AD has no obsm entry for {key!r} (tried {_OBSM_ALIASES.get(key, (key,))})"
    )


def read_grouped_counts(
    h5ad: Path | str,
    groups: Sequence[str] | np.ndarray | pd.Index,
    cres: Sequence[str] | np.ndarray | pd.Index,
    keys: Sequence[str] = DEFAULT_COUNT_KEYS,
) -> GroupedCounts:
    """Sum each requested ``obsm`` count matrix over the cells of every subclass.

    Args:
        h5ad: Dataset to stream. Opened once.
        groups: Subclass labels to report, in the caller's order. Must all exist
            in the normalized ``obs/subclass_name`` categories.
        cres: cCRE column names to report, in the caller's order. Must all exist
            in every requested ``obsm`` group.
        keys: Which count matrices to sum. Defaults to both species.

    Returns:
        A :class:`GroupedCounts` whose ``totals[key]`` is ``(len(groups), len(cres))``
        float64, and whose ``group_classes`` / ``group_cell_counts`` are aligned to
        ``groups``.

    Cells whose ``subclass_name`` or ``class_name`` code is negative (unassigned)
    are excluded from both the totals and the cell counts.
    """
    requested_keys = tuple(dict.fromkeys(keys))
    if not requested_keys:
        raise ValueError("keys must name at least one obsm count matrix")
    group_names = np.asarray(groups, dtype=str)
    cre_names = np.asarray(cres, dtype=str)

    with h5py.File(h5ad, "r") as handle:
        obs = handle["obs"]
        subclass_categories, subclass_codes = _normalized_categories(
            obs[_SUBCLASS_COLUMN]
        )
        class_categories, class_codes = _normalized_categories(obs[_CLASS_COLUMN])

        lookup = {name: idx for idx, name in enumerate(subclass_categories)}
        missing_groups = sorted(set(group_names.tolist()) - set(lookup))
        if missing_groups:
            raise ValueError(f"H5AD is missing requested subclasses: {missing_groups}")
        to_h5 = np.asarray([lookup[name] for name in group_names], dtype=np.int64)

        valid = (subclass_codes >= 0) & (class_codes >= 0)
        all_valid = bool(valid.all())
        codes = subclass_codes if all_valid else subclass_codes[valid]
        n_subclasses = len(subclass_categories)

        cell_counts = np.bincount(codes, minlength=n_subclasses)
        class_of_subclass = _class_of_subclass(
            codes, class_codes if all_valid else class_codes[valid], n_subclasses
        )
        group_classes = class_categories[class_of_subclass[to_h5]]

        totals: dict[str, np.ndarray] = {}
        for key in requested_keys:
            matrix = _resolve_obsm(handle["obsm"], key)
            missing_cres = sorted(set(cre_names.tolist()) - set(matrix.keys()))
            if missing_cres:
                raise ValueError(
                    f"obsm/{key} is missing requested cCREs: {missing_cres}"
                )
            values = np.empty((len(group_names), len(cre_names)), dtype=np.float64)
            for column, cre in enumerate(cre_names):
                per_cell = matrix[cre][...].astype(np.float64, copy=False)
                if not all_valid:
                    per_cell = per_cell[valid]
                grouped = np.bincount(
                    codes, weights=per_cell, minlength=n_subclasses
                )
                values[:, column] = grouped[to_h5]
            totals[key] = values

    return GroupedCounts(
        groups=group_names,
        cres=cre_names,
        totals=totals,
        group_classes=group_classes,
        group_cell_counts=cell_counts[to_h5],
    )


def grouped_obsm_totals(
    h5ad: Path | str,
    groups: Sequence[str] | np.ndarray | pd.Index,
    cres: Sequence[str] | np.ndarray | pd.Index,
    keys: Sequence[str] = DEFAULT_COUNT_KEYS,
) -> dict[str, pd.DataFrame]:
    """:func:`read_grouped_counts` when only the labelled count frames are wanted."""
    counts = read_grouped_counts(h5ad, groups, cres, keys)
    return {key: counts.frame(key) for key in counts.totals}
