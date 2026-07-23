#!/usr/bin/env python3
"""Fit decoupled T7 infection and cCRE activity Bayesian models."""

from __future__ import annotations

import argparse
import dataclasses
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


TAG_DROPOUT = "subclass_cre_decoupled_copy_number_dropout_svi"
TAG_NO_DROPOUT = "subclass_cre_decoupled_copy_number_no_dropout_svi"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5ad", type=Path, default=DEFAULT_H5AD)
    parser.add_argument("--outdir", type=Path, default=None)
    parser.add_argument(
        "--section", choices=["all", "sec1", "sec2"], default="all"
    )
    parser.add_argument("--kmax", type=int, default=None)
    parser.add_argument("--steps-t7", type=int, default=30_000)
    parser.add_argument("--steps-cre", type=int, default=30_000)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument(
        "--guide",
        choices=["AutoNormal", "AutoLowRankMultivariateNormal"],
        default="AutoNormal",
    )
    parser.add_argument("--num-posterior", type=int, default=1_000)
    parser.add_argument("--infection-quadrature-points", type=int, default=7)
    parser.add_argument(
        "--dropout-model",
        choices=["zero_inflated", "none"],
        default="zero_inflated",
    )
    parser.add_argument("--dropout-prior-label", default="default_beta_1_9")
    parser.add_argument("--p-drop-t7-alpha", type=float, default=1.0)
    parser.add_argument("--p-drop-t7-beta", type=float, default=9.0)
    parser.add_argument("--p-drop-cre-alpha", type=float, default=1.0)
    parser.add_argument("--p-drop-cre-beta", type=float, default=9.0)
    parser.add_argument(
        "--posterior-sites",
        nargs="+",
        default=["log_gamma"],
        help=(
            "CRE posterior sites to save in *_posterior_samples.npz. "
            "Use 'all' to save every sampled and deterministic CRE posterior site."
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-cells", type=int, default=None, help="Smoke testing only.")
    parser.add_argument("--max-cres", type=int, default=None, help="Smoke testing only.")
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def default_outdir(args: argparse.Namespace) -> Path:
    name = (
        "bayesian_decoupled"
        if args.dropout_model == "zero_inflated"
        else "bayesian_decoupled_no_dropout"
    )
    if args.section == "all":
        return ANALYSIS_DIR / "results" / "ablation" / name
    return ANALYSIS_DIR / "results" / "ablation" / "sections" / args.section / name


def run_tag(args: argparse.Namespace) -> str:
    return TAG_DROPOUT if args.dropout_model == "zero_inflated" else TAG_NO_DROPOUT


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


def _float32_posterior(samples: dict) -> dict:
    return {
        key: np.asarray(value, dtype=np.float32)
        if np.issubdtype(np.asarray(value).dtype, np.floating)
        else np.asarray(value)
        for key, value in samples.items()
    }


def dropout_prior_config(args: argparse.Namespace) -> dict:
    return {
        "label": args.dropout_prior_label,
        "p_drop_t7_alpha": args.p_drop_t7_alpha,
        "p_drop_t7_beta": args.p_drop_t7_beta,
        "p_drop_t7_mean": args.p_drop_t7_alpha
        / (args.p_drop_t7_alpha + args.p_drop_t7_beta),
        "p_drop_cre_alpha": args.p_drop_cre_alpha,
        "p_drop_cre_beta": args.p_drop_cre_beta,
        "p_drop_cre_mean": args.p_drop_cre_alpha
        / (args.p_drop_cre_alpha + args.p_drop_cre_beta),
    }


def main() -> None:
    args = parse_args()
    if args.outdir is None:
        args.outdir = default_outdir(args)
    if args.cpu:
        os.environ["JAX_PLATFORMS"] = "cpu"

    sys.path.insert(0, str(STARRFISH_ROOT / "STARRFISH"))
    import bayesian_hierarchical as bh

    priors = dataclasses.replace(
        bh.ModelPriors(),
        p_drop_t7_alpha=args.p_drop_t7_alpha,
        p_drop_t7_beta=args.p_drop_t7_beta,
        p_drop_cre_alpha=args.p_drop_cre_alpha,
        p_drop_cre_beta=args.p_drop_cre_beta,
    )

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

    t7 = adata.obsm["T7CRE"].loc[:, cre_names].to_numpy()
    cre = adata.obsm["CRE"].loc[:, cre_names].to_numpy()
    subclasses = adata.obs["subclass"].astype(str).to_numpy()
    classes = adata.obs["class"].astype(str).to_numpy()

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
    pd.Series(adata.obs["subclass"].value_counts(), name="n_cells").to_csv(
        args.outdir / "subclass_cell_counts.csv"
    )

    posterior_sites = (
        ["all"] if any(site.lower() == "all" for site in args.posterior_sites)
        else args.posterior_sites
    )
    tag = run_tag(args)
    log(
        f"[bayesian-decoupled] fitting {len(cre_names)} cCREs, "
        f"{adata.obs['subclass'].nunique()} subclasses, tag={tag}"
    )
    result = bh.run_decoupled_model(
        t7,
        cre,
        subclasses,
        classes,
        lib_size_log,
        cre_names,
        kmax=args.kmax,
        steps_t7=args.steps_t7,
        steps_cre=args.steps_cre,
        lr=args.lr,
        guide=args.guide,
        num_posterior=args.num_posterior,
        seed=args.seed,
        priors=priors,
        negative_control_mask=negative_control_mask,
        infection_quadrature_points=args.infection_quadrature_points,
        posterior_sites_to_return=posterior_sites,
        dropout_model=args.dropout_model,
    )
    result["config"].update(
        {
            "input": input_fingerprint(args.h5ad),
            "blacklist_cre": blacklist,
            "max_cells": args.max_cells,
            "max_cres": args.max_cres,
            "section": args.section,
            "posterior_sites_to_return": posterior_sites,
            "dropout_prior": dropout_prior_config(args),
            "dropout_model": args.dropout_model,
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
    result["t7_evidence"]["per_pair"].to_csv(
        f"{prefix}_t7_evidence_per_pair.csv", index=False
    )
    result["cre_evidence"]["per_pair"].to_csv(
        f"{prefix}_cre_evidence_per_pair.csv", index=False
    )
    write_json(
        Path(f"{prefix}_evidence_totals.json"),
        jsonable(result["evidence"]["totals"]),
    )
    write_json(
        Path(f"{prefix}_t7_evidence_totals.json"),
        jsonable(result["t7_evidence"]["totals"]),
    )
    write_json(
        Path(f"{prefix}_cre_evidence_totals.json"),
        jsonable(result["cre_evidence"]["totals"]),
    )
    write_json(Path(f"{prefix}_ppc.json"), jsonable(result["ppc"]))

    diagnostics = {
        key: value
        for key, value in result["diagnostics"].items()
        if key not in {"losses_t7", "losses_cre"}
    }
    losses_t7 = np.asarray(result["diagnostics"]["losses_t7"])
    losses_cre = np.asarray(result["diagnostics"]["losses_cre"])
    diagnostics.update(
        {
            "loss_t7_start": float(losses_t7[0]),
            "loss_t7_end": float(losses_t7[-1]),
            "loss_cre_start": float(losses_cre[0]),
            "loss_cre_end": float(losses_cre[-1]),
        }
    )
    np.save(f"{prefix}_losses_t7.npy", losses_t7)
    np.save(f"{prefix}_losses_cre.npy", losses_cre)
    write_json(Path(f"{prefix}_diagnostics.json"), jsonable(diagnostics))

    np.savez(f"{prefix}_scalar_samples.npz", **result["scalar_samples"])
    np.savez_compressed(
        f"{prefix}_log_lambda_summary.npz",
        log_lambda_mean=np.asarray(result["log_lambda_mean"], dtype=np.float32),
        log_lambda_sd=np.asarray(result["log_lambda_sd"], dtype=np.float32),
        group_names=np.asarray(result["group_names"], dtype=object),
        cre_names=np.asarray(result["cre_names"], dtype=object),
    )

    posterior = _float32_posterior(result.pop("posterior_samples"))
    posterior["group_names"] = np.asarray(result["group_names"], dtype=object)
    posterior["cre_names"] = np.asarray(result["cre_names"], dtype=object)
    np.savez_compressed(f"{prefix}_posterior_samples.npz", **posterior)

    infection_posterior = _float32_posterior(result.pop("infection_posterior_samples"))
    infection_posterior["group_names"] = np.asarray(result["group_names"], dtype=object)
    infection_posterior["cre_names"] = np.asarray(result["cre_names"], dtype=object)
    np.savez_compressed(
        f"{prefix}_infection_posterior_samples.npz", **infection_posterior
    )

    with Path(f"{prefix}_result.pkl").open("wb") as handle:
        pickle.dump(result, handle)
    write_json(
        args.outdir / "run_manifest.json",
        {
            "tag": tag,
            "method_variant": "bayesian_decoupled_t7_cre",
            "input": input_fingerprint(args.h5ad),
            "n_cells": int(adata.n_obs),
            "n_cres_mapped": int(len(cre_info)),
            "n_cres_fitted": int(len(cre_names)),
            "n_subclasses": int(adata.obs["subclass"].nunique()),
            "n_classes": int(adata.obs["class"].nunique()),
            "section": args.section,
            "blacklist": blacklist,
            "negative_controls": negative_controls,
            "config": result["config"],
            "dropout_prior": dropout_prior_config(args),
            "dropout_model": args.dropout_model,
        },
    )
    log(f"[bayesian-decoupled] wrote intermediates to {args.outdir}")


if __name__ == "__main__":
    main()
