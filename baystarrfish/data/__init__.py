"""AnnData-to-arrays adapter for STARR-FISH inputs.

Deliberately free of JAX and NumPyro: the plotting and analysis environment
imports this subpackage to reload the same inputs a fit used, without paying for
(or installing) the inference stack.

    from baystarrfish.data import CountData
    data = CountData.from_h5ad(section="sec1", negative_control_mode="ordinary")
"""

from __future__ import annotations

from . import paths
from .anndata import (
    aligned_obsm_frame,
    canonical_cre_info,
    normalize_celltype_labels,
    read_and_prepare_adata,
    read_obs_metadata,
    section_labels,
    select_cre_info,
    standardize_obs,
)
from .controls import (
    NEGATIVE_CONTROL_MODES,
    POOLED_NEGATIVE_CONTROL_NAME,
    build_pooled_negative_control,
    cre_blacklist,
    negative_control_names,
)
from .counts import CountData
from .grouped_counts import (
    DEFAULT_COUNT_KEYS,
    GroupedCounts,
    grouped_obsm_totals,
    read_grouped_counts,
)
from .libsize import library_counts, library_size_log

__all__ = [
    "DEFAULT_COUNT_KEYS",
    "NEGATIVE_CONTROL_MODES",
    "POOLED_NEGATIVE_CONTROL_NAME",
    "CountData",
    "GroupedCounts",
    "aligned_obsm_frame",
    "build_pooled_negative_control",
    "canonical_cre_info",
    "cre_blacklist",
    "grouped_obsm_totals",
    "library_counts",
    "library_size_log",
    "negative_control_names",
    "read_grouped_counts",
    "normalize_celltype_labels",
    "paths",
    "read_and_prepare_adata",
    "read_obs_metadata",
    "section_labels",
    "select_cre_info",
    "standardize_obs",
]
