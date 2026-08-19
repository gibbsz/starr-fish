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

import sys
from pathlib import Path

import numpy as np

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

# revision/run_Bayes owns the exported-matrix naming and readers (activity_matrix_io).
# Importing analysis_utils makes it importable, so no script has to repeat this insert.
RUN_BAYES_CODE = ANALYSIS_DIR.parent / "run_Bayes"
if str(RUN_BAYES_CODE) not in sys.path:
    sys.path.insert(0, str(RUN_BAYES_CODE))

# The production joint+dropout fit on the original (5/28 NEWNEW) dataset. It
# lives outside this analysis directory because the same fit backs the
# origin-vs-new comparison. Activity is identical to the superseded
# results/bayesian to ~1e-13 relative; this one also stores log_rho and log_a,
# so copy number is recoverable from it.
OLD_DATA_BAYES = ANALYSIS_DIR.parent / "Bayes_OldData" / "bayesian"

# The whole-dataset bootstrap on the same original data, likewise hoisted out of
# this analysis directory. Per-section bootstraps stay under results/sections/.
OLD_DATA_BOOTSTRAP = ANALYSIS_DIR.parent / "Bootstrap_OldData"

# The four ablation arms that back the published method-comparison figures live in
# their own top-level revision/Bayesian_ablation/, each carrying its posterior under
# bayesian/ and the exported matrices under tables/ -- the same layout as
# Bayes_OldData / Bayes_NewData. Every arm there uses the production `direct` activity
# parameterisation with ordinary negative controls, so the four differ only in channel
# (joint vs decoupled) and dropout. The superseded hierarchical fits are under
# Bayesian_ablation/archive/ and are deliberately NOT resolvable here.
#
# Other arms are exploratory and stay where they were fitted, so resolve arm names
# through ``ablation_root`` rather than hardcoding either location.
ABLATION_DIR = ANALYSIS_DIR / "results" / "ablation"
RELOCATED_ABLATION_DIR = ANALYSIS_DIR.parent / "Bayesian_ablation"

#: Arms living under revision/Bayesian_ablation/. The suffix names what the arm HAS:
#: ``bayesian_decoupled`` is the plain two-stage fit, ``bayesian_decoupled_dropout``
#: adds zero-inflated dropout. ``bayesian_joint_dropout`` is a symlink to Bayes_OldData
#: -- it IS the production fit, not a copy of it.
RELOCATED_ABLATION_ARMS = frozenset(
    {
        "bayesian_joint",
        "bayesian_joint_dropout",
        "bayesian_decoupled",
        "bayesian_decoupled_dropout",
    }
)


def ablation_root(name: str) -> Path:
    """The posterior directory of one ablation arm, relocated or not.

    Raises for a name that is neither relocated nor present under results/ablation/,
    rather than returning a path that does not exist. A silent miss here surfaces much
    later as a confusing FileNotFoundError on run_manifest.json, which is exactly how a
    renamed arm goes unnoticed.
    """
    if name in RELOCATED_ABLATION_ARMS:
        return RELOCATED_ABLATION_DIR / name / "bayesian"
    candidate = ABLATION_DIR / name
    if not candidate.is_dir():
        raise KeyError(
            f"unknown ablation arm {name!r}: not in RELOCATED_ABLATION_ARMS "
            f"{sorted(RELOCATED_ABLATION_ARMS)} and no directory at {candidate}"
        )
    return candidate


def ablation_tables(name: str) -> Path:
    """The exported ``tables/`` directory of one relocated ablation arm."""
    if name not in RELOCATED_ABLATION_ARMS:
        raise KeyError(
            f"{name!r} has no exported tables/; only {sorted(RELOCATED_ABLATION_ARMS)} "
            "were relocated to revision/Bayesian_ablation/"
        )
    return RELOCATED_ABLATION_DIR / name / "tables"


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

#: Side of one panel, in inches. Every multi-panel figure in this analysis sizes its
#: canvas as this times the grid shape (plus a fixed allowance for a colorbar or an
#: outside legend), so a panel reads the same whether it sits in a 2 x 2 or a 6 x 6.
PANEL_SIDE_INCHES = 4.0

#: The panel side the marker sizes were originally tuned against. :func:`marker_scale`
#: in the correlation script rescales from it so point density is preserved.
REFERENCE_PANEL_SIDE_INCHES = 2.7

def fit_panel_size(fig, axes, side: float = PANEL_SIDE_INCHES, passes: int = 4) -> None:
    """Resize ``fig`` until each panel in ``axes`` is ``side`` inches square.

    ``constrained_layout`` hands the axes whatever is left after tick labels, axis
    labels, the caption and any colorbar, so creating the canvas at ``side * n`` leaves
    the panels short by an amount that depends on the figure -- 3.67 x 3.45 in for a
    2 x 2 correlation matrix, 3.83 x 3.76 for a 6 x 6. Measuring that shortfall and
    adding it back converges in two or three passes, which pins the panel geometry
    without anyone having to hand-tune a figsize per figure.

    Pass the array ``plt.subplots`` returned, not ``fig.axes``: colorbar and legend
    axes must not be averaged in. Uneven ``width_ratios`` are honoured -- it is the
    MEAN panel that lands on ``side``.
    """
    grid = np.atleast_2d(np.asarray(axes, dtype=object))
    n_rows, n_cols = grid.shape
    panels = [ax for ax in grid.ravel() if ax is not None]
    if not panels:
        return
    for _ in range(passes):
        fig.canvas.draw()
        width, height = fig.get_size_inches()
        mean_w = float(np.mean([ax.get_position().width for ax in panels])) * width
        mean_h = float(np.mean([ax.get_position().height for ax in panels])) * height
        if abs(mean_w - side) < 0.02 and abs(mean_h - side) < 0.02:
            return
        fig.set_size_inches(
            n_cols * side + (width - n_cols * mean_w),
            n_rows * side + (height - n_rows * mean_h),
        )
    fig.canvas.draw()


#: Internal method key -> the name the manuscript figures print for it. The keys stay
#: as they are because they identify an arm in the ablation registry -- ``Joint+dropout``
#: names the joint channel plus zero-inflated dropout, which is what distinguishes that
#: fit from its three siblings -- and because they are values in the shipped ``method``
#: columns. Only what a reader sees on an axis or in a legend is rewritten.
FIGURE_METHOD_LABELS = {
    "Joint+dropout": "Bayesian",
}


def display_label(method: str) -> str:
    """The printable name for a method key, including composed variants.

    The precision-recall script builds keys like ``"Joint+dropout mean controls"`` by
    suffixing the arm name, so the substitution is applied to a leading key as well as
    to an exact match. Unknown keys pass through unchanged.
    """
    for key, label in FIGURE_METHOD_LABELS.items():
        if method == key:
            return label
        if method.startswith(f"{key} "):
            return f"{label}{method[len(key):]}"
    return method


__all__ = [
    "FIGURE_METHOD_LABELS",
    "PANEL_SIDE_INCHES",
    "REFERENCE_PANEL_SIDE_INCHES",
    "ABLATION_DIR",
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
    "RELOCATED_ABLATION_ARMS",
    "RELOCATED_ABLATION_DIR",
    "REPO_ROOT",
    "RUN_BAYES_CODE",
    "REVISION_DATA",
    "STARRFISH_DATA",
    "STARRFISH_ROOT",
    "ablation_root",
    "ablation_tables",
    "aligned_obsm_frame",
    "atomic_save_array",
    "canonical_cre_info",
    "cre_blacklist",
    "display_label",
    "fit_panel_size",
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
