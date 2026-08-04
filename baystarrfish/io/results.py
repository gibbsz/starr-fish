"""Read and write a fit directory.

The on-disk layout is unchanged from the analysis scripts that produced the
published results, so existing ``results/`` trees remain readable::

    <outdir>/
      run_manifest.json                    tag, input fingerprint, shapes, config
      cre_info.csv  cre_blacklist.csv  negative_controls.csv
      subclass_cell_counts.csv
      <tag>_gamma.csv                      tidy per-(cell type, cCRE) activity
      <tag>_rho.csv                        per-cell-type infection
      <tag>_delta_mean.csv                 subclass deviations (hierarchical only)
      <tag>_evidence_per_pair.csv          double-positive support per pair
      <tag>_evidence_totals.json  <tag>_ppc.json  <tag>_diagnostics.json
      <tag>_losses.npy                     ELBO trace (or _losses_t7/_losses_cre)
      <tag>_scalar_samples.npz             global parameter draws
      <tag>_posterior_samples.npz          log_gamma draws, float32, + name arrays
      <tag>_result.pkl                     the result dict minus the posterior

``write_fit`` handles both the single-stage and the two-stage decoupled result
shapes; which files appear is driven by which keys the result actually holds.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd

from .._log import log
from .serialize import input_fingerprint, jsonable, write_json

__all__ = [
    "decode_strings",
    "fit_tag",
    "load_gamma",
    "load_posterior_samples",
    "read_fit",
    "write_fit",
]

#: Summary tables written without their index (they carry explicit columns).
_TIDY_SUMMARY_KEYS = ("rho", "infection", "gamma")


def decode_strings(values) -> np.ndarray:
    """Normalise an object/bytes name array from ``.npz`` to plain ``str``.

    ``np.savez`` round-trips ``dtype=object`` string arrays as bytes on some
    NumPy/HDF5 combinations, so cCRE and cell-type names come back mixed.
    """
    values = np.asarray(values)
    return np.asarray(
        [
            value.decode("utf-8") if isinstance(value, (bytes, np.bytes_)) else str(value)
            for value in values.ravel()
        ]
    ).reshape(values.shape)


def fit_tag(root: Path | str, requested: str | None = None) -> str:
    """The file-stem tag of the fit in ``root``, from its manifest."""
    if requested:
        return requested
    manifest = json.loads((Path(root) / "run_manifest.json").read_text())
    return str(manifest["tag"])


def _float32_posterior(posterior: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Halve the on-disk size of the draws; 24 significant bits is ample here."""
    return {
        key: (
            np.asarray(value, dtype=np.float32)
            if np.issubdtype(np.asarray(value).dtype, np.floating)
            else np.asarray(value)
        )
        for key, value in posterior.items()
    }


def _write_side_tables(outdir: Path, tables: Mapping[str, pd.DataFrame | pd.Series]) -> None:
    for name, table in tables.items():
        # Series of per-cCRE names are column-like: no index. Everything else
        # (cre_info, cell counts) is keyed by its index and must retain it.
        index = not (isinstance(table, pd.Series) and table.name == "cre")
        table.to_csv(outdir / f"{name}.csv", index=index)


def _write_losses(prefix: Path, diagnostics: dict) -> dict:
    """Persist ELBO traces as .npy and replace them with scalar summaries.

    A 30,000-step trace does not belong in a JSON file, but "did it start high,
    end low, and stay finite" is exactly what you check first.
    """
    trimmed = {}
    for key, value in diagnostics.items():
        if not key.startswith("losses"):
            trimmed[key] = value
            continue
        losses = np.asarray(value)
        suffix = key[len("losses"):]  # "" | "_t7" | "_cre"
        np.save(f"{prefix}_{key}.npy", losses)
        trimmed[f"loss{suffix}_start"] = float(losses[0])
        trimmed[f"loss{suffix}_end"] = float(losses[-1])
        trimmed[f"loss{suffix}_all_finite"] = bool(np.isfinite(losses).all())
    return trimmed


def write_fit(
    result: dict,
    outdir: Path | str,
    tag: str,
    *,
    data=None,
    input_path: Path | str | None = None,
    manifest_extra: Mapping | None = None,
    save_pickle: bool = True,
) -> Path:
    """Serialise a fit and its provenance into ``outdir``.

    Parameters
    ----------
    result
        The dict returned by :func:`baystarrfish.fit` or
        :func:`baystarrfish.fit_decoupled`. Posterior draws are *popped* from it
        before pickling, so the pickle stays small; pass a copy if the caller
        still needs them.
    data : CountData, optional
        Provides the side tables (``cre_info``, blacklist, controls, cell counts)
        and the shape fields of the manifest.
    manifest_extra
        Merged into ``run_manifest.json``. Use it for run-identifying fields the
        package cannot know, such as the CLI method variant.

    Returns the manifest path.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    prefix = outdir / tag

    if data is not None:
        _write_side_tables(outdir, data.side_tables())

    summary = result.get("summary", {})
    for key in _TIDY_SUMMARY_KEYS:
        if key in summary:
            summary[key].to_csv(f"{prefix}_{key}.csv", index=False)
    if "delta_mean" in summary:
        summary["delta_mean"].to_csv(f"{prefix}_delta_mean.csv")

    # evidence, t7_evidence, cre_evidence -- whichever the result carries.
    for key in [k for k in result if k == "evidence" or k.endswith("_evidence")]:
        block = result[key]
        stem = "evidence" if key == "evidence" else key
        block["per_pair"].to_csv(f"{prefix}_{stem}_per_pair.csv", index=False)
        write_json(Path(f"{prefix}_{stem}_totals.json"), jsonable(block["totals"]))

    if "ppc" in result:
        write_json(Path(f"{prefix}_ppc.json"), jsonable(result["ppc"]))
    if "diagnostics" in result:
        write_json(
            Path(f"{prefix}_diagnostics.json"),
            jsonable(_write_losses(prefix, dict(result["diagnostics"]))),
        )
    if "scalar_samples" in result:
        np.savez(f"{prefix}_scalar_samples.npz", **result["scalar_samples"])

    names = {
        "group_names": np.asarray(result["group_names"], dtype=object),
        "cre_names": np.asarray(result["cre_names"], dtype=object),
    }
    if "log_lambda_mean" in result:
        np.savez_compressed(
            f"{prefix}_log_lambda_summary.npz",
            log_lambda_mean=np.asarray(result["log_lambda_mean"], dtype=np.float32),
            log_lambda_sd=np.asarray(result["log_lambda_sd"], dtype=np.float32),
            **names,
        )
    for key in ("posterior_samples", "infection_posterior_samples"):
        if key in result:
            np.savez_compressed(
                f"{prefix}_{key}.npz", **_float32_posterior(result.pop(key)), **names
            )

    if save_pickle:
        with Path(f"{prefix}_result.pkl").open("wb") as handle:
            pickle.dump(result, handle)

    manifest = {"tag": tag, "config": result.get("config", {})}
    if input_path is not None:
        manifest["input"] = input_fingerprint(input_path)
    if data is not None:
        manifest.update(
            {
                "n_cells": data.n_cells,
                "n_cres_mapped": int(len(data.cre_info)) if data.cre_info is not None else None,
                "n_cres_fitted": data.n_cre,
                "n_subclasses": data.n_subclasses,
                "n_classes": data.n_classes,
                "section": data.section,
                "blacklist": data.blacklist,
                "negative_controls": data.negative_controls,
                "negative_control_mode": data.negative_control_mode,
                "pooled_negative_control": data.pooled_negative_control,
            }
        )
    if manifest_extra:
        manifest.update(dict(manifest_extra))
    manifest_path = outdir / "run_manifest.json"
    write_json(manifest_path, jsonable(manifest))
    log(f"[io] wrote fit '{tag}' to {outdir}")
    return manifest_path


def load_posterior_samples(
    root: Path | str,
    tag: str | None = None,
    *,
    sites: Iterable[str] | None = None,
    kind: str = "posterior_samples",
) -> dict[str, np.ndarray]:
    """Load ``<tag>_<kind>.npz``, decoding the name arrays to ``str``.

    ``sites`` restricts which draw arrays are materialised -- a full
    ``log_gamma`` block is 444 MB for the production fit, so read only what you
    need. The ``group_names`` / ``cre_names`` arrays are always returned.
    """
    root = Path(root)
    tag = fit_tag(root, tag)
    path = root / f"{tag}_{kind}.npz"
    with np.load(path, allow_pickle=True) as handle:
        keys = set(handle.files)
        wanted = keys if sites is None else ({*sites} & keys) | {"group_names", "cre_names"}
        missing = set() if sites is None else {*sites} - keys
        if missing:
            raise KeyError(f"{path} has no site(s) {sorted(missing)}; available {sorted(keys)}")
        out = {key: handle[key] for key in wanted}
    for key in ("group_names", "cre_names"):
        if key in out:
            out[key] = decode_strings(out[key])
    return out


def load_gamma(root: Path | str, tag: str | None = None) -> pd.DataFrame:
    """The tidy per-(cell type, cCRE) activity table of a fit."""
    root = Path(root)
    return pd.read_csv(root / f"{fit_tag(root, tag)}_gamma.csv")


def read_fit(root: Path | str, tag: str | None = None) -> dict:
    """Unpickle ``<tag>_result.pkl`` (the result dict minus posterior draws)."""
    root = Path(root)
    with (root / f"{fit_tag(root, tag)}_result.pkl").open("rb") as handle:
        return pickle.load(handle)
