"""Filesystem locations of the STARR-FISH reference inputs.

Every path is resolved through a function, not frozen at import time, so an
environment variable set by a test or by another install can take effect. A
published package must not bake ``/gpfs/commons/groups/ren_lab/...`` into
importable constants.

Environment overrides
---------------------
``BAYSTARRFISH_REPO_ROOT``
    Root of the ``starr-fish`` checkout. Defaults to the parent of the installed
    package, which is correct for the editable install used in this repo.
``BAYSTARRFISH_DATA_ROOT``
    Directory holding the nanopore-count and barcode-mismatch reference CSVs.
    Defaults to ``<repo>/STARRFISH_in_vivo/Data``.
``BAYSTARRFISH_H5AD``
    The default single-cell input. Defaults to the 5/28 NEWNEW dataset under
    ``<repo>/revision/Data``.

The legacy upper-case constant names (``DEFAULT_H5AD``, ``LIBSIZE_CSV``,
``MISMATCH_CSV``, ``CRE_INFO_FALLBACK_CSV``, ``REPO_ROOT``, ``STARRFISH_ROOT``,
``STARRFISH_DATA``, ``REVISION_DATA``) remain readable as module attributes and
are evaluated on access.
"""

from __future__ import annotations

import os
from pathlib import Path

ENV_REPO_ROOT = "BAYSTARRFISH_REPO_ROOT"
ENV_DATA_ROOT = "BAYSTARRFISH_DATA_ROOT"
ENV_H5AD = "BAYSTARRFISH_H5AD"

#: cCREs excluded in the manuscript irrespective of barcode-mismatch rate.
BASE_BLACKLIST: tuple[str, ...] = ("CRE001", "CRE061", "CRE143")

#: Filename of the 5/28 dataset used for every published fit.
DEFAULT_H5AD_NAME = "scdata_5_28_2025_BRBB500gn_final_CRE_T7CRE_NEWNEW.h5ad"

__all__ = [
    "BASE_BLACKLIST",
    "DEFAULT_H5AD_NAME",
    "ENV_DATA_ROOT",
    "ENV_H5AD",
    "ENV_REPO_ROOT",
    "cre_info_fallback_csv",
    "data_root",
    "default_h5ad",
    "libsize_csv",
    "mismatch_csv",
    "repo_root",
    "revision_data_root",
    "starrfish_root",
]


def _env_path(name: str) -> Path | None:
    value = os.environ.get(name)
    return Path(value).expanduser() if value else None


def repo_root() -> Path:
    """Root of the ``starr-fish`` checkout."""
    return _env_path(ENV_REPO_ROOT) or Path(__file__).resolve().parents[2]


def starrfish_root() -> Path:
    """``STARRFISH_in_vivo/`` -- the in-vivo analysis tree."""
    return repo_root() / "STARRFISH_in_vivo"


def data_root() -> Path:
    """Directory holding the nanopore and barcode-mismatch reference CSVs."""
    return _env_path(ENV_DATA_ROOT) or starrfish_root() / "Data"


def revision_data_root() -> Path:
    """Directory holding the revision-round single-cell inputs."""
    return repo_root() / "revision" / "Data"


def default_h5ad() -> Path:
    """The single-cell input used for every published fit."""
    return _env_path(ENV_H5AD) or revision_data_root() / DEFAULT_H5AD_NAME


def libsize_csv() -> Path:
    """Nanopore per-cCRE library counts (the informative abundance prior)."""
    return data_root() / "SFv8_400CRE_nanopore_counts.csv"


def mismatch_csv() -> Path:
    """Per-barcode ONT mismatch rates; >20% defines the blacklist."""
    return data_root() / "AAV_ONT_Barcode_Counts_vs_Mismatch_Percentage.csv"


def cre_info_fallback_csv() -> Path:
    """Used only when the input lacks ``uns['CRE_info']``."""
    return starrfish_root() / "results" / "cre_info.csv"


_LEGACY_CONSTANTS = {
    "REPO_ROOT": repo_root,
    "STARRFISH_ROOT": starrfish_root,
    "STARRFISH_DATA": data_root,
    "REVISION_DATA": revision_data_root,
    "DEFAULT_H5AD": default_h5ad,
    "LIBSIZE_CSV": libsize_csv,
    "MISMATCH_CSV": mismatch_csv,
    "CRE_INFO_FALLBACK_CSV": cre_info_fallback_csv,
}


def __getattr__(name: str) -> Path:
    """Expose the legacy upper-case names, evaluated on access."""
    resolver = _LEGACY_CONSTANTS.get(name)
    if resolver is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return resolver()


def __dir__() -> list[str]:
    return sorted([*__all__, *_LEGACY_CONSTANTS])
