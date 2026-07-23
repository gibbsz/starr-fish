#!/usr/bin/env python3
"""Shared input preparation for the 5/28 bootstrap and Bayesian analyses."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

import anndata as ad
import numpy as np
import pandas as pd


CODE_DIR = Path(__file__).resolve().parent
ANALYSIS_DIR = CODE_DIR.parent
REPO_ROOT = ANALYSIS_DIR.parents[1]
REVISION_DATA = REPO_ROOT / "revision" / "Data"
STARRFISH_ROOT = REPO_ROOT / "STARRFISH_in_vivo"
STARRFISH_DATA = STARRFISH_ROOT / "Data"
CRE_INFO_FALLBACK_CSV = STARRFISH_ROOT / "results" / "cre_info.csv"
DEFAULT_H5AD = (
    REVISION_DATA / "scdata_5_28_2025_BRBB500gn_final_CRE_T7CRE_NEWNEW.h5ad"
)
LIBSIZE_CSV = STARRFISH_DATA / "SFv8_400CRE_nanopore_counts.csv"
MISMATCH_CSV = STARRFISH_DATA / "AAV_ONT_Barcode_Counts_vs_Mismatch_Percentage.csv"
BASE_BLACKLIST = ("CRE001", "CRE061", "CRE143")


def log(message: str) -> None:
    print(message, flush=True)


def standardize_obs(adata: ad.AnnData) -> None:
    """Create stable class/subclass labels without Allen numeric prefixes."""
    required = {"subclass_name", "class_name"}
    missing = required - set(adata.obs.columns)
    if missing:
        raise KeyError(f"missing required obs columns: {sorted(missing)}")
    for source, target in (("subclass_name", "subclass"), ("class_name", "class")):
        adata.obs[target] = (
            adata.obs[source]
            .astype(str)
            .str.replace(r"^\d+\s+", "", regex=True)
            .str.replace("/", "-", regex=False)
        )


def canonical_cre_info(adata: ad.AnnData) -> pd.DataFrame:
    """Map CRE_info rows to the CRE001... column convention used by obsm."""
    if "CRE_info" in adata.uns:
        cre_info = adata.uns["CRE_info"].copy()
        cre_info.index = pd.Index(
            [f"CRE{i + 1:03d}" for i in range(len(cre_info))], name="cre"
        )
    else:
        if not CRE_INFO_FALLBACK_CSV.exists():
            raise KeyError(
                "input does not contain uns['CRE_info'] and fallback "
                f"{CRE_INFO_FALLBACK_CSV} does not exist"
            )
        log(
            "[input] uns['CRE_info'] missing; using fallback "
            f"{CRE_INFO_FALLBACK_CSV}"
        )
        cre_info = pd.read_csv(CRE_INFO_FALLBACK_CSV, index_col=0)
        cre_info.index = pd.Index(cre_info.index.astype(str), name="cre")
    if {"best_subclass", "label"}.issubset(cre_info.columns):
        empty = cre_info["best_subclass"].fillna("").eq("")
        cre_info.loc[empty, "best_subclass"] = cre_info.loc[empty, "label"]
        cre_info["best_subclass"] = (
            cre_info["best_subclass"]
            .astype(str)
            .str.replace("_", " ", regex=False)
            .str.replace("/", "-", regex=False)
        )
    return cre_info


def aligned_obsm_frame(
    adata: ad.AnnData, key: str, columns: Iterable[str]
) -> pd.DataFrame:
    """Return one named count matrix aligned to canonical cCRE columns."""
    if key not in adata.obsm:
        raise KeyError(f"missing obsm[{key!r}]")
    columns = pd.Index(columns, dtype=str)
    matrix = adata.obsm[key]
    if isinstance(matrix, pd.DataFrame):
        frame = matrix.copy()
        frame.columns = frame.columns.astype(str)
        missing = columns.difference(frame.columns)
        if len(missing):
            raise ValueError(
                f"obsm[{key!r}] is missing {len(missing)} canonical columns: "
                f"{missing[:5].tolist()}"
            )
        return frame.loc[:, columns]
    if matrix.shape[1] != len(columns):
        raise ValueError(
            f"unnamed obsm[{key!r}] has {matrix.shape[1]} columns; "
            f"expected {len(columns)}"
        )
    return pd.DataFrame(matrix, index=adata.obs_names, columns=columns)


def cre_blacklist(columns: Iterable[str]) -> list[str]:
    """Return the manuscript base plus >20% barcode-mismatch blacklist."""
    mismatch = pd.read_csv(MISMATCH_CSV, index_col=0)
    failed = mismatch.index[
        mismatch["MismatchPercent"].astype(float).gt(20)
    ].astype(str)
    available = set(map(str, columns))
    return sorted((set(BASE_BLACKLIST) | set(failed)) & available)


def select_cre_info(cre_info: pd.DataFrame, max_cres: int | None) -> pd.DataFrame:
    """Optionally shrink smoke tests while always retaining negative controls."""
    if max_cres is None or max_cres >= len(cre_info):
        return cre_info
    if max_cres < 1:
        raise ValueError("--max-cres must be positive")
    negative = cre_info.index[
        cre_info["labeling_type"].astype(str).str.lower().eq("negative control")
    ]
    primary = cre_info.index[~cre_info.index.isin(negative)][:max_cres]
    selected = primary.append(negative).drop_duplicates()
    return cre_info.loc[selected]


def section_labels(obs_names: Iterable[str]) -> pd.Series:
    """Map the 5/28 z-scan identifiers to the two physical sections."""
    names = pd.Index(obs_names).astype(str)
    zscan = names.to_series(index=names).str.extract(
        r"^Conv_zscan([12])_", expand=False
    )
    labels = zscan.map({"2": "sec1", "1": "sec2"})
    if labels.isna().any():
        examples = names[labels.isna().to_numpy()][:5].tolist()
        raise ValueError(f"cannot assign section for obs names: {examples}")
    return labels


def read_and_prepare_adata(
    path: Path,
    *,
    section: str = "all",
    max_cells: int | None = None,
    max_cres: int | None = None,
    seed: int = 0,
) -> ad.AnnData:
    """Load the input and retain only named, CRE_info-backed count columns."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    if section not in {"all", "sec1", "sec2"}:
        raise ValueError(f"unsupported section: {section}")
    log(f"[input] loading {path}")
    if section == "all" and max_cells is None:
        adata = ad.read_h5ad(path)
    else:
        backed = ad.read_h5ad(path, backed="r")
        try:
            positions = np.arange(backed.n_obs)
            if section != "all":
                labels = section_labels(backed.obs_names)
                positions = positions[labels.to_numpy() == section]
            if max_cells is not None and max_cells < len(positions):
                rng = np.random.default_rng(seed)
                positions = np.sort(
                    rng.choice(positions, size=max_cells, replace=False)
                )
            adata = backed[positions, :].to_memory()
        finally:
            backed.file.close()

    standardize_obs(adata)
    adata.obs["section"] = section_labels(adata.obs_names).to_numpy()
    if section != "all" and not adata.obs["section"].eq(section).all():
        raise AssertionError(f"loaded observations do not all belong to {section}")
    cre_info = select_cre_info(canonical_cre_info(adata), max_cres)
    columns = cre_info.index.astype(str)
    adata.obsm["CRE"] = aligned_obsm_frame(adata, "CRE", columns)
    t7_key = "T7CRE" if "T7CRE" in adata.obsm else "T7"
    adata.obsm["T7CRE"] = aligned_obsm_frame(adata, t7_key, columns)
    adata.uns["CRE_info"] = cre_info

    # These are the only matrices used by the two analyses. X_raw is retained
    # because STARRFISH.average_bootstrap_test accesses it during preprocessing.
    keep_obsm = {"CRE", "T7CRE", "X_raw", "X_spatial"}
    for key in list(adata.obsm.keys()):
        if key not in keep_obsm:
            del adata.obsm[key]
    log(
        f"[input] prepared section={section}, {adata.n_obs:,} cells, "
        f"{adata.n_vars:,} genes, "
        f"{len(columns)} mapped cCREs"
    )
    return adata


def negative_control_names(cre_info: pd.DataFrame, blacklist: Iterable[str]) -> list[str]:
    blocked = set(map(str, blacklist))
    mask = cre_info["labeling_type"].astype(str).str.lower().eq("negative control")
    return [name for name in cre_info.index[mask].astype(str) if name not in blocked]


def input_fingerprint(path: Path) -> dict:
    """Cheap, reproducible identity record without hashing a multi-GB file."""
    path = Path(path).resolve()
    stat = path.stat()
    payload = f"{path}|{stat.st_size}|{stat.st_mtime_ns}".encode()
    return {
        "path": str(path),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "metadata_sha256": hashlib.sha256(payload).hexdigest(),
    }


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def atomic_save_array(path: Path, array: np.ndarray) -> None:
    """Avoid leaving a complete-looking partial .npy after interruption."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as handle:
        np.save(handle, array)
    tmp.replace(path)
