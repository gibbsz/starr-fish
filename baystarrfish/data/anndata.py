"""Load and normalise the STARR-FISH single-cell input.

The on-disk AnnData needs three kinds of repair before it can drive a fit:

* ``obs`` cell-type labels carry Allen numeric prefixes and ``/`` separators;
* ``uns['CRE_info']`` is positional while ``obsm`` is keyed ``CRE001..CRE400``;
* the two count matrices live under ``obsm['CRE']`` and ``obsm['T7CRE']`` (older
  inputs call the latter ``'T7'``) and are not guaranteed to be column-aligned.

Both ``obsm['CRE']`` and ``obsm['T7CRE']`` are transcript species -- the cCRE
reporter readout and the constitutive T7 readout of the same construct. Neither
is a DNA copy-number measurement; the copy number ``k`` is the model's latent.

This module imports only ``anndata``/``pandas``/``numpy`` -- never JAX or
NumPyro -- so the plotting environment can use it without the inference stack.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import anndata as ad
import numpy as np
import pandas as pd

from .._log import log
from . import paths

__all__ = [
    "aligned_obsm_frame",
    "canonical_cre_info",
    "normalize_celltype_labels",
    "read_and_prepare_adata",
    "read_obs_metadata",
    "section_labels",
    "select_cre_info",
    "standardize_obs",
]


def normalize_celltype_labels(labels: pd.Series) -> pd.Series:
    """Strip the Allen numeric prefix and make ``/`` filesystem-safe.

    The single definition of the label convention every fit and every
    downstream analysis indexes groups by.
    """
    return (
        labels.astype(str)
        .str.replace(r"^\d+\s+", "", regex=True)
        .str.replace("/", "-", regex=False)
    )


def standardize_obs(adata: ad.AnnData) -> None:
    """Create stable class/subclass labels without Allen numeric prefixes."""
    required = {"subclass_name", "class_name"}
    missing = required - set(adata.obs.columns)
    if missing:
        raise KeyError(f"missing required obs columns: {sorted(missing)}")
    for source, target in (("subclass_name", "subclass"), ("class_name", "class")):
        adata.obs[target] = normalize_celltype_labels(adata.obs[source])


def canonical_cre_info(adata: ad.AnnData) -> pd.DataFrame:
    """Map CRE_info rows to the CRE001... column convention used by obsm."""
    fallback_csv = paths.cre_info_fallback_csv()
    if "CRE_info" in adata.uns:
        cre_info = adata.uns["CRE_info"].copy()
        cre_info.index = pd.Index(
            [f"CRE{i + 1:03d}" for i in range(len(cre_info))], name="cre"
        )
    else:
        if not fallback_csv.exists():
            raise KeyError(
                "input does not contain uns['CRE_info'] and fallback "
                f"{fallback_csv} does not exist"
            )
        log(f"[input] uns['CRE_info'] missing; using fallback {fallback_csv}")
        cre_info = pd.read_csv(fallback_csv, index_col=0)
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


def select_cre_info(cre_info: pd.DataFrame, max_cres: int | None) -> pd.DataFrame:
    """Optionally shrink smoke tests while always retaining negative controls."""
    if max_cres is None or max_cres >= len(cre_info):
        return cre_info
    if max_cres < 1:
        raise ValueError("max_cres must be positive")
    negative = cre_info.index[
        cre_info["labeling_type"].astype(str).str.lower().eq("negative control")
    ]
    primary = cre_info.index[~cre_info.index.isin(negative)][:max_cres]
    selected = primary.append(negative).drop_duplicates()
    return cre_info.loc[selected]


def section_labels(obs_names: Iterable[str]) -> pd.Series:
    """Map the 5/28 z-scan identifiers to the two physical sections.

    The 2 -> sec1 / 1 -> sec2 inversion is deliberate: the z-scan acquisition
    order is the reverse of the physical section order.
    """
    names = pd.Index(obs_names).astype(str)
    zscan = names.to_series(index=names).str.extract(
        r"^Conv_zscan([12])_", expand=False
    )
    labels = zscan.map({"2": "sec1", "1": "sec2"})
    if labels.isna().any():
        examples = names[labels.isna().to_numpy()][:5].tolist()
        raise ValueError(f"cannot assign section for obs names: {examples}")
    return labels


def _decode(values: Iterable[bytes | str]) -> np.ndarray:
    """HDF5 string arrays come back as ``bytes``; normalise to ``str``."""
    return np.asarray(
        [value.decode() if isinstance(value, bytes) else str(value) for value in values],
        dtype=object,
    )


def read_obs_metadata(
    path: Path | str | None = None,
    *,
    spatial_key: str = "X_spatial",
    celltype_keys: Iterable[str] = ("subclass_name", "class_name"),
) -> pd.DataFrame:
    """``(obs_name, section, x, y)`` plus cell-type labels, without the counts.

    Uses ``h5py`` rather than ``anndata``: even a backed read materialises every
    ``obsm`` entry, and ``CRE``/``T7CRE``/``X_raw`` are hundreds of megabytes
    that a spatial analysis never touches. Labels get the same normalisation
    :func:`standardize_obs` applies, so they index exactly the groups a fit was
    built for. ``subclass_name`` yields a ``subclass`` column, ``class_name`` a
    ``class`` column -- the ``_name`` suffix is dropped.
    """
    import h5py

    resolved = Path(path) if path is not None else paths.default_h5ad()
    wanted = list(celltype_keys)
    columns: dict[str, np.ndarray] = {}
    with h5py.File(resolved, "r") as handle:
        obs = handle["obs"]
        obs_names = _decode(obs["_index"][:])
        for key in wanted:
            if key not in obs:
                raise KeyError(f"{resolved} has no obs[{key!r}]; have {list(obs)}")
            node = obs[key]
            if not isinstance(node, h5py.Group):
                raise TypeError(f"obs[{key!r}] is not a categorical encoding")
            codes = np.asarray(node["codes"][:], dtype=np.int64)
            if codes.size and codes.min() < 0:
                raise ValueError(f"obs[{key!r}] has unassigned (-1) codes")
            columns[key] = _decode(node["categories"][:])[codes]
        obsm = handle["obsm"]
        if spatial_key not in obsm:
            raise KeyError(f"obsm[{spatial_key!r}] absent; have {list(obsm)}")
        coords = np.asarray(obsm[spatial_key][:, :2], dtype=np.float64)

    if coords.shape[0] != obs_names.size:
        raise ValueError(
            f"obs has {obs_names.size} rows but obsm[{spatial_key!r}] has "
            f"{coords.shape[0]}"
        )
    frame = pd.DataFrame({"obs_name": obs_names})
    for key, values in columns.items():
        target = key[: -len("_name")] if key.endswith("_name") else key
        frame[target] = normalize_celltype_labels(pd.Series(values)).to_numpy()
    frame["section"] = section_labels(obs_names).to_numpy()
    frame["x"] = coords[:, 0]
    frame["y"] = coords[:, 1]
    return frame


def read_and_prepare_adata(
    path: Path | str | None = None,
    *,
    section: str = "all",
    max_cells: int | None = None,
    max_cres: int | None = None,
    seed: int = 0,
) -> ad.AnnData:
    """Load the input and retain only named, CRE_info-backed count columns.

    A section or cell subsample is taken through a backed read so a smoke run
    never materialises the full 3.3 GB object.
    """
    path = Path(path) if path is not None else paths.default_h5ad()
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
