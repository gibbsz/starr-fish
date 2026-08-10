#!/usr/bin/env python3
"""Compatibility shim -- the input preparation now lives in ``baystarrfish``.

Every function and path constant this module used to define moved into the
installable package:

===============================  ==========================================
was ``analysis_utils.X``          now
===============================  ==========================================
path constants                    ``baystarrfish.data.paths`` (functions,
                                  overridable via ``BAYSTARRFISH_*`` env vars)
``read_and_prepare_adata`` etc.   ``baystarrfish.data.anndata``
``cre_blacklist``,
``negative_control_names``        ``baystarrfish.data.controls``
``write_json``, ``jsonable``,
``atomic_save_array``,
``input_fingerprint``             ``baystarrfish.io``
``log``                           ``baystarrfish``
===============================  ==========================================

New code should import from ``baystarrfish`` directly. This module stays so the
~30 plotting and statistics scripts in this directory keep running unchanged,
and so the sibling-module import style they rely on (slurm ``cd``s to the repo
root, putting this directory on ``sys.path``) does not have to change.

``ANALYSIS_DIR`` and ``CODE_DIR`` are genuinely local to this analysis round and
are still defined here.
"""

from __future__ import annotations

from pathlib import Path

from baystarrfish._log import log
from baystarrfish.data.anndata import (
    aligned_obsm_frame,
    canonical_cre_info,
    read_and_prepare_adata,
    section_labels,
    select_cre_info,
    standardize_obs,
)
from baystarrfish.data.controls import cre_blacklist, negative_control_names
from baystarrfish.data.paths import (
    BASE_BLACKLIST,
    cre_info_fallback_csv,
    data_root,
    default_h5ad,
    libsize_csv,
    mismatch_csv,
    repo_root,
    revision_data_root,
    starrfish_root,
)
from baystarrfish.io import atomic_save_array, input_fingerprint, jsonable, write_json

# Local to this analysis directory, not to the package.
CODE_DIR = Path(__file__).resolve().parent
ANALYSIS_DIR = CODE_DIR.parent

# The production joint+dropout fit on the original (5/28 NEWNEW) dataset. It
# lives outside this analysis directory because the same fit backs the
# origin-vs-new comparison. Activity is identical to the superseded
# results/bayesian to ~1e-13 relative; this one also stores log_rho and log_a,
# so copy number is recoverable from it.
OLD_DATA_BAYES = ANALYSIS_DIR.parent / "Bayes_OldData" / "bayesian"

# The whole-dataset bootstrap on the same original data, likewise hoisted out of
# this analysis directory. Per-section bootstraps stay under results/sections/.
OLD_DATA_BOOTSTRAP = ANALYSIS_DIR.parent / "Bootstrap_OldData"

# Figures are split by role. Every producer writes into FIGURES_WORK; the curated
# manuscript set is assembled into FIGURES_FINAL by
# figure_final/collect_final_figures.py, which owns the list of what counts as
# final. Nothing should write to FIGURES_FINAL directly.
FIGURES_WORK = ANALYSIS_DIR / "results" / "figures" / "work"
FIGURES_FINAL = ANALYSIS_DIR / "results" / "figures" / "final"

# Resolved once here for backwards compatibility. Prefer the package functions
# in new code -- they honour the BAYSTARRFISH_* environment overrides at call
# time, whereas these snapshots are frozen at import.
REPO_ROOT = repo_root()
REVISION_DATA = revision_data_root()
STARRFISH_ROOT = starrfish_root()
STARRFISH_DATA = data_root()
CRE_INFO_FALLBACK_CSV = cre_info_fallback_csv()
DEFAULT_H5AD = default_h5ad()
LIBSIZE_CSV = libsize_csv()
MISMATCH_CSV = mismatch_csv()

__all__ = [
    "ANALYSIS_DIR",
    "BASE_BLACKLIST",
    "CODE_DIR",
    "CRE_INFO_FALLBACK_CSV",
    "DEFAULT_H5AD",
    "LIBSIZE_CSV",
    "MISMATCH_CSV",
    "FIGURES_FINAL",
    "FIGURES_WORK",
    "OLD_DATA_BAYES",
    "OLD_DATA_BOOTSTRAP",
    "REPO_ROOT",
    "REVISION_DATA",
    "STARRFISH_DATA",
    "STARRFISH_ROOT",
    "aligned_obsm_frame",
    "atomic_save_array",
    "canonical_cre_info",
    "cre_blacklist",
    "input_fingerprint",
    "jsonable",
    "log",
    "negative_control_names",
    "read_and_prepare_adata",
    "section_labels",
    "select_cre_info",
    "standardize_obs",
    "write_json",
]
