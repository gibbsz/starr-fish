#!/usr/bin/env python3
"""Fit the Bayesian model to within-subclass bootstrapped meta-cells."""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from analysis_utils import (
    ANALYSIS_DIR,
    DEFAULT_H5AD,
    LIBSIZE_CSV,
    STARRFISH_ROOT,
    cre_blacklist,
    input_fingerprint,
    log,
    negative_control_names,
    read_and_prepare_adata,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5ad", type=Path, default=DEFAULT_H5AD)
    parser.add_argument("--outdir", type=Path, default=None)
    parser.add_argument(
        "--section", choices=["all", "sec1", "sec2"], default="all"
    )
    parser.add_argument(
        "--bootstrap-size",
        type=int,
        default=100,
        help="Cells sampled with replacement per meta-cell within each subclass.",
    )
    parser.add_argument(
        "--bootstrap-number",
        type=int,
        default=100,
        help="Bootstrapped meta-cell replicates to generate per subclass.",
    )
    parser.add_argument("--channel", choices=["t7", "joint"], default="joint")
    parser.add_argument("--method", choices=["svi", "nuts"], default="svi")
    parser.add_argument(
        "--infection-model",
        choices=["copy_number", "binary"],
        default="copy_number",
    )
    parser.add_argument("--kmax", type=int, default=500)
    parser.add_argument("--steps", type=int, default=30_000)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument(
        "--guide",
        choices=["AutoNormal", "AutoLowRankMultivariateNormal"],
        default="AutoNormal",
    )
    parser.add_argument("--num-warmup", type=int, default=1_000)
    parser.add_argument("--num-samples", type=int, default=1_000)
    parser.add_argument("--num-chains", type=int, default=2)
    parser.add_argument("--num-posterior", type=int, default=1_000)
    parser.add_argument(
        "--posterior-sites",
        nargs="+",
        default=["log_gamma"],
        help=(
            "Posterior sites to save in *_posterior_samples.npz. "
            "Use 'all' to save every sampled and deterministic posterior site."
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-cells", type=int, default=None, help="Smoke testing only.")
    parser.add_argument("--max-cres", type=int, default=None, help="Smoke testing only.")
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def jsonable(obj):
    if isinstance(obj, dict):
        return {key: jsonable(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(value) for value in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def default_outdir(args: argparse.Namespace) -> Path:
    dirname = (
        "bayesian_bootstrap_metacells_"
        f"size{args.bootstrap_size}_number{args.bootstrap_number}"
    )
    if args.section == "all":
        return ANALYSIS_DIR / "results" / "ablation" / dirname
    return (
        ANALYSIS_DIR
        / "results"
        / "ablation"
        / "sections"
        / args.section
        / dirname
    )


def validate_bootstrap_args(bootstrap_size: int, bootstrap_number: int) -> None:
    if bootstrap_size <= 0:
        raise ValueError("--bootstrap-size must be positive")
    if bootstrap_number <= 0:
        raise ValueError("--bootstrap-number must be positive")


def build_bootstrap_metacells(
    adata,
    cre_names: list[str],
    *,
    bootstrap_size: int,
    bootstrap_number: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Sum fixed-size with-replacement cell samples within each subclass."""

    validate_bootstrap_args(bootstrap_size, bootstrap_number)
    obs = adata.obs[["subclass", "class"]].astype(str).copy()
    mapping = obs.drop_duplicates()
    if mapping["subclass"].duplicated().any():
        duplicated = mapping.loc[
            mapping["subclass"].duplicated(keep=False), "subclass"
        ].unique()
        raise ValueError(
            "subclass does not nest cleanly within class: "
            f"{sorted(map(str, duplicated))[:5]}"
        )

    subclass_values = obs["subclass"].to_numpy()
    subclass_order = pd.Index(sorted(obs["subclass"].unique()), dtype=str)
    class_by_subclass = mapping.set_index("subclass")["class"].to_dict()
    t7 = np.asarray(adata.obsm["T7CRE"].loc[:, cre_names].to_numpy(), dtype=np.int64)
    cre = np.asarray(adata.obsm["CRE"].loc[:, cre_names].to_numpy(), dtype=np.int64)

    n_meta = len(subclass_order) * bootstrap_number
    n_cre = len(cre_names)
    meta_t7 = np.empty((n_meta, n_cre), dtype=np.int64)
    meta_cre = np.empty((n_meta, n_cre), dtype=np.int64)
    records = []
    rng = np.random.default_rng(seed)

    row = 0
    for subclass in subclass_order:
        positions = np.flatnonzero(subclass_values == subclass)
        if len(positions) == 0:
            raise AssertionError(f"no cells found for subclass {subclass!r}")
        for replicate in range(bootstrap_number):
            sampled = rng.choice(positions, size=bootstrap_size, replace=True)
            meta_t7[row] = t7[sampled].sum(axis=0, dtype=np.int64)
            meta_cre[row] = cre[sampled].sum(axis=0, dtype=np.int64)
            records.append(
                {
                    "metacell": f"metacell_{row:06d}",
                    "subclass": subclass,
                    "class": class_by_subclass[subclass],
                    "bootstrap_replicate": replicate,
                    "bootstrap_size": bootstrap_size,
                    "source_subclass_cells": int(len(positions)),
                }
            )
            row += 1

    meta_obs = pd.DataFrame.from_records(records).set_index("metacell")
    return meta_t7, meta_cre, meta_obs


def main() -> None:
    args = parse_args()
    if args.outdir is None:
        args.outdir = default_outdir(args)
    if args.cpu:
        os.environ["JAX_PLATFORMS"] = "cpu"

    # Import the standalone module directly, avoiding STARRFISH/__init__.py and
    # its unrelated torch/scvi/cmdstanpy imports.
    sys.path.insert(0, str(STARRFISH_ROOT / "STARRFISH"))
    import bayesian_hierarchical as bh

    adata = read_and_prepare_adata(
        args.h5ad,
        section=args.section,
        max_cells=args.max_cells,
        max_cres=args.max_cres,
        seed=args.seed,
    )
    cre_info = adata.uns["CRE_info"].copy()
    blacklist = cre_blacklist(cre_info.index)
    cre_names = [name for name in cre_info.index.astype(str) if name not in blacklist]
    negative_controls = negative_control_names(cre_info, blacklist)
    negative_control_mask = np.isin(cre_names, negative_controls)

    raw_libsize = pd.read_csv(LIBSIZE_CSV, index_col=0)
    raw_libsize.index = raw_libsize.index.astype(str)
    raw_libsize = raw_libsize.reindex(cre_names, fill_value=0)
    lib_size_log = np.log1p(raw_libsize["counts"].to_numpy(dtype=np.float64))

    args.outdir.mkdir(parents=True, exist_ok=True)
    cre_info.to_csv(args.outdir / "cre_info.csv")
    pd.Series(blacklist, name="cre").to_csv(
        args.outdir / "cre_blacklist.csv", index=False
    )
    pd.Series(negative_controls, name="cre").to_csv(
        args.outdir / "negative_controls.csv", index=False
    )
    source_cell_counts = adata.obs["subclass"].astype(str).value_counts().sort_index()
    source_cell_counts.rename("n_cells").to_csv(
        args.outdir / "subclass_cell_counts.csv"
    )
    source_cell_counts.rename("n_source_cells").to_csv(
        args.outdir / "source_subclass_cell_counts.csv"
    )

    t7, cre, meta_obs = build_bootstrap_metacells(
        adata,
        cre_names,
        bootstrap_size=args.bootstrap_size,
        bootstrap_number=args.bootstrap_number,
        seed=args.seed,
    )
    meta_obs.to_csv(args.outdir / "metacell_obs.csv")
    meta_obs["subclass"].value_counts().sort_index().rename("n_metacells").to_csv(
        args.outdir / "metacell_subclass_counts.csv"
    )
    subclasses = meta_obs["subclass"].to_numpy()
    classes = meta_obs["class"].to_numpy()
    n_source_cells = int(adata.n_obs)

    tag = f"subclass_{args.channel}_{args.infection_model}_{args.method}"
    log(
        f"[bayesian-metacell] fitting {len(cre_names)} cCREs, "
        f"{meta_obs['subclass'].nunique()} subclasses, "
        f"{len(meta_obs):,} meta-cells, "
        f"bootstrap_size={args.bootstrap_size}, tag={tag}"
    )
    posterior_sites = (
        ["all"] if any(site.lower() == "all" for site in args.posterior_sites)
        else args.posterior_sites
    )
    result = bh.run_model(
        t7,
        cre,
        subclasses,
        classes,
        lib_size_log,
        cre_names,
        level="subclass",
        channel=args.channel,
        method=args.method,
        kmax=args.kmax,
        num_steps=args.steps,
        lr=args.lr,
        guide=args.guide,
        num_warmup=args.num_warmup,
        num_samples=args.num_samples,
        num_chains=args.num_chains,
        num_posterior=args.num_posterior,
        seed=args.seed,
        negative_control_mask=negative_control_mask,
        infection_model=args.infection_model,
        posterior_sites_to_return=posterior_sites,
    )
    result["config"].update(
        {
            "input": input_fingerprint(args.h5ad),
            "input_transform": "bootstrap_metacells",
            "blacklist_cre": blacklist,
            "max_cells": args.max_cells,
            "max_cres": args.max_cres,
            "section": args.section,
            "bootstrap_size": args.bootstrap_size,
            "bootstrap_number": args.bootstrap_number,
            "n_source_cells": n_source_cells,
            "n_metacells": int(len(meta_obs)),
            "posterior_sites_to_return": posterior_sites,
        }
    )

    prefix = args.outdir / tag
    summary = result["summary"]
    for key in ("rho", "infection", "gamma"):
        if key in summary:
            summary[key].to_csv(f"{prefix}_{key}.csv", index=False)
    if "delta_mean" in summary:
        summary["delta_mean"].to_csv(f"{prefix}_delta_mean.csv")
    result["evidence"]["per_pair"].to_csv(
        f"{prefix}_evidence_per_pair.csv", index=False
    )
    write_json(
        Path(f"{prefix}_evidence_totals.json"),
        jsonable(result["evidence"]["totals"]),
    )
    write_json(Path(f"{prefix}_ppc.json"), jsonable(result["ppc"]))

    diagnostics = {
        key: value for key, value in result["diagnostics"].items() if key != "losses"
    }
    if "losses" in result["diagnostics"]:
        losses = np.asarray(result["diagnostics"]["losses"])
        diagnostics.update(
            {
                "loss_start": float(losses[0]),
                "loss_end": float(losses[-1]),
                "loss_all_finite": bool(np.isfinite(losses).all()),
            }
        )
        np.save(f"{prefix}_losses.npy", losses)
    write_json(Path(f"{prefix}_diagnostics.json"), jsonable(diagnostics))
    np.savez(f"{prefix}_scalar_samples.npz", **result["scalar_samples"])

    posterior = result.pop("posterior_samples")
    posterior = {
        key: np.asarray(value, dtype=np.float32)
        if np.issubdtype(np.asarray(value).dtype, np.floating)
        else np.asarray(value)
        for key, value in posterior.items()
    }
    posterior["group_names"] = np.asarray(result["group_names"], dtype=object)
    posterior["cre_names"] = np.asarray(result["cre_names"], dtype=object)
    np.savez_compressed(f"{prefix}_posterior_samples.npz", **posterior)

    with Path(f"{prefix}_result.pkl").open("wb") as handle:
        pickle.dump(result, handle)
    write_json(
        args.outdir / "run_manifest.json",
        {
            "tag": tag,
            "method_variant": "bayesian_bootstrap_metacells",
            "input": input_fingerprint(args.h5ad),
            "n_source_cells": n_source_cells,
            "n_metacells": int(len(meta_obs)),
            "n_cres_mapped": int(len(cre_info)),
            "n_cres_fitted": int(len(cre_names)),
            "n_subclasses": int(meta_obs["subclass"].nunique()),
            "n_classes": int(meta_obs["class"].nunique()),
            "section": args.section,
            "bootstrap_size": args.bootstrap_size,
            "bootstrap_number": args.bootstrap_number,
            "blacklist": blacklist,
            "negative_controls": negative_controls,
            "config": result["config"],
        },
    )
    log(f"[bayesian-metacell] wrote intermediates to {args.outdir}")


if __name__ == "__main__":
    main()
