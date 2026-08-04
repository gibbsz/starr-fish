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
    read_and_prepare_adata,
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
from .libsize import library_counts, library_size_log

__all__ = [
    "NEGATIVE_CONTROL_MODES",
    "POOLED_NEGATIVE_CONTROL_NAME",
    "CountData",
    "aligned_obsm_frame",
    "build_pooled_negative_control",
    "canonical_cre_info",
    "cre_blacklist",
    "library_counts",
    "library_size_log",
    "negative_control_names",
    "paths",
    "read_and_prepare_adata",
    "section_labels",
    "select_cre_info",
    "standardize_obs",
]
