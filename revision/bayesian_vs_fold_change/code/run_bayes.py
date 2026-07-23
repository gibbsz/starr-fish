#!/usr/bin/env python3
"""Fit the hierarchical joint CRE/T7 model to the 5/28 BRBB500gn dataset."""

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


POOLED_NEGATIVE_CONTROL_NAME = "NEGATIVE_CONTROL_POOL"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5ad", type=Path, default=DEFAULT_H5AD)
    parser.add_argument("--outdir", type=Path, default=None)
    parser.add_argument(
        "--section", choices=["all", "sec1", "sec2"], default="all"
    )
    parser.add_argument("--level", choices=["class", "subclass"], default="subclass")
    parser.add_argument("--channel", choices=["t7", "joint"], default="joint")
    parser.add_argument("--method", choices=["svi", "nuts"], default="svi")
    parser.add_argument(
        "--infection-model",
        choices=["copy_number", "copy_number_dropout", "binary"],
        default="copy_number",
    )
    parser.add_argument("--dropout-prior-label", default="default_beta_1_9")
    parser.add_argument("--p-drop-t7-alpha", type=float, default=1.0)
    parser.add_argument("--p-drop-t7-beta", type=float, default=9.0)
    parser.add_argument("--p-drop-cre-alpha", type=float, default=1.0)
    parser.add_argument("--p-drop-cre-beta", type=float, default=9.0)
    parser.add_argument(
        "--negative-control-mode",
        choices=["pooled", "ordinary", "ordinary-and-pooled"],
        default="pooled",
        help=(
            "Pool annotated negative controls through shared activity parameters, "
            "fit them as ordinary cCREs, or fit the seven ordinary columns plus "
            "one appended all-seven pooled pseudo-cCRE in the same model."
        ),
    )
    parser.add_argument(
        "--activity-model",
        choices=["hierarchical", "direct"],
        default="hierarchical",
        help=(
            "Use the alpha/eta/delta hierarchy or directly estimate an exchangeable "
            "raw log_gamma matrix. Direct activity requires ordinary negative controls."
        ),
    )
    parser.add_argument("--kmax", type=int, default=None)
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


def default_outdir(args: argparse.Namespace) -> Path:
    variant = method_variant(args)
    if args.section == "all":
        if variant == "bayesian_joint_dropout_direct_activity_ordinary_negative_controls":
            return ANALYSIS_DIR / "results" / "bayesian"
        return ANALYSIS_DIR / "results" / "ablation" / variant
    if variant == "bayesian_joint_dropout_ordinary_and_pooled_negative_controls":
        return ANALYSIS_DIR / "results" / "sections" / args.section / "bayesian"
    return (
        ANALYSIS_DIR
        / "results"
        / "ablation"
        / "sections"
        / args.section
        / variant
    )


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


def method_variant(args: argparse.Namespace) -> str:
    base = (
        "bayesian_joint_dropout"
        if args.infection_model == "copy_number_dropout"
        else "bayesian_joint"
    )
    if args.activity_model == "direct":
        base += "_direct_activity"
    control_suffix = {
        "pooled": "",
        "ordinary": "_ordinary_negative_controls",
        "ordinary-and-pooled": "_ordinary_and_pooled_negative_controls",
    }
    return base + control_suffix[args.negative_control_mode]


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


def main() -> None:
    args = parse_args()
    if args.activity_model == "direct" and args.negative_control_mode != "ordinary":
        raise ValueError(
            "--activity-model direct requires --negative-control-mode ordinary"
        )
    if args.outdir is None:
        args.outdir = default_outdir(args)
    if args.cpu:
        os.environ["JAX_PLATFORMS"] = "cpu"

    # Import the standalone module directly, avoiding STARRFISH/__init__.py and
    # its unrelated torch/scvi/cmdstanpy imports.
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
    library_counts = raw_libsize["counts"].to_numpy(dtype=np.float64)

    pooled_negative_control = None
    if args.negative_control_mode == "pooled":
        model_negative_control_mask = negative_control_mask
    elif args.negative_control_mode == "ordinary":
        model_negative_control_mask = None
    else:
        if not negative_control_mask.any():
            raise ValueError("cannot append a pooled cCRE without negative controls")
        if POOLED_NEGATIVE_CONTROL_NAME in cre_names:
            raise ValueError(
                f"reserved pooled cCRE name already exists: {POOLED_NEGATIVE_CONTROL_NAME}"
            )
        pooled_t7 = t7[:, negative_control_mask].sum(axis=1, keepdims=True)
        pooled_cre = cre[:, negative_control_mask].sum(axis=1, keepdims=True)
        pooled_library_count = float(library_counts[negative_control_mask].sum())
        t7 = np.concatenate([t7, pooled_t7], axis=1)
        cre = np.concatenate([cre, pooled_cre], axis=1)
        library_counts = np.concatenate(
            [library_counts, np.asarray([pooled_library_count], dtype=np.float64)]
        )
        cre_names = [*cre_names, POOLED_NEGATIVE_CONTROL_NAME]
        model_negative_control_mask = np.zeros(len(cre_names), dtype=bool)
        model_negative_control_mask[-1] = True
        pooled_negative_control = {
            "name": POOLED_NEGATIVE_CONTROL_NAME,
            "constituent_cre": negative_controls,
            "nanopore_count": pooled_library_count,
            "construction": "per-cell sums of T7 and CRE counts across constituents",
        }

    lib_size_log = np.log1p(library_counts)

    args.outdir.mkdir(parents=True, exist_ok=True)
    cre_info.to_csv(args.outdir / "cre_info.csv")
    pd.Series(blacklist, name="cre").to_csv(
        args.outdir / "cre_blacklist.csv", index=False
    )
    pd.Series(negative_controls, name="cre").to_csv(
        args.outdir / "negative_controls.csv", index=False
    )
    if pooled_negative_control is not None:
        pd.DataFrame(
            {
                "pooled_cre": POOLED_NEGATIVE_CONTROL_NAME,
                "constituent_cre": negative_controls,
            }
        ).to_csv(args.outdir / "pooled_negative_control_definition.csv", index=False)
    pd.Series(adata.obs["subclass"].value_counts(), name="n_cells").to_csv(
        args.outdir / "subclass_cell_counts.csv"
    )

    tag = f"{args.level}_{args.channel}_{args.infection_model}_{args.method}"
    log(
        f"[bayesian] fitting {len(cre_names)} cCREs, "
        f"{len(np.unique(subclasses))} subclasses, tag={tag}"
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
        level=args.level,
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
        negative_control_mask=model_negative_control_mask,
        infection_model=args.infection_model,
        activity_model=args.activity_model,
        priors=priors,
        posterior_sites_to_return=posterior_sites,
    )
    result["config"].update(
        {
            "input": input_fingerprint(args.h5ad),
            "blacklist_cre": blacklist,
            "max_cells": args.max_cells,
            "max_cres": args.max_cres,
            "section": args.section,
            "posterior_sites_to_return": posterior_sites,
            "dropout_model": (
                "zero_inflated"
                if args.infection_model == "copy_number_dropout"
                else "none"
            ),
            "dropout_prior": dropout_prior_config(args),
            "negative_control_mode": args.negative_control_mode,
            "activity_model": args.activity_model,
            "annotated_negative_control_cre": negative_controls,
            "pooled_negative_control": pooled_negative_control,
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
            "input": input_fingerprint(args.h5ad),
            "n_cells": int(adata.n_obs),
            "n_cres_mapped": int(len(cre_info)),
            "n_cres_fitted": int(len(cre_names)),
            "n_subclasses": int(adata.obs["subclass"].nunique()),
            "n_classes": int(adata.obs["class"].nunique()),
            "section": args.section,
            "blacklist": blacklist,
            "negative_controls": negative_controls,
            "negative_control_mode": args.negative_control_mode,
            "activity_model": args.activity_model,
            "pooled_negative_control": pooled_negative_control,
            "method_variant": method_variant(args),
            "dropout_prior": dropout_prior_config(args),
            "config": result["config"],
        },
    )
    log(f"[bayesian] wrote intermediates to {args.outdir}")


if __name__ == "__main__":
    main()
