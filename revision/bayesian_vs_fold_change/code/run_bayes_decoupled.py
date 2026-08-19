#!/usr/bin/env python3
"""Fit decoupled T7 infection and cCRE activity Bayesian models."""

from __future__ import annotations

import argparse
import dataclasses
import os
from pathlib import Path

# Lazy facade: no JAX or NumPyro import until a model symbol is touched, which is
# what lets --cpu set JAX_PLATFORMS before the backend is chosen.
import baystarrfish as bsf
from baystarrfish.data import CountData
from baystarrfish.io import input_fingerprint, write_fit

from analysis_utils import ANALYSIS_DIR, DEFAULT_H5AD, log


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
    parser.add_argument(
        "--negative-control-mode",
        choices=["pooled", "ordinary", "ordinary-and-pooled"],
        default="ordinary",
        help=(
            "Fit the annotated negative controls as ordinary cCREs (the default, and "
            "the only mode --activity-model direct accepts), pool them through shared "
            "activity parameters, or add an appended all-seven pooled pseudo-cCRE."
        ),
    )
    parser.add_argument(
        "--activity-model",
        choices=["hierarchical", "direct"],
        default="direct",
        help=(
            "Directly estimate an exchangeable raw log_gamma matrix (the default; "
            "requires ordinary negative controls) or use the alpha/eta/delta "
            "hierarchy. Must match the fit this run will be compared against."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace a completed fit instead of refusing to start.",
    )
    return parser.parse_args()


def default_outdir(args: argparse.Namespace) -> Path:
    # The suffix names what the arm HAS, not what it lacks: bayesian_decoupled is the
    # plain two-stage fit and bayesian_decoupled_dropout adds zero-inflated dropout.
    name = (
        "bayesian_decoupled_dropout"
        if args.dropout_model == "zero_inflated"
        else "bayesian_decoupled"
    )
    if args.section == "all":
        return ANALYSIS_DIR / "results" / "ablation" / name
    return ANALYSIS_DIR / "results" / "ablation" / "sections" / args.section / name


def run_tag(args: argparse.Namespace) -> str:
    return TAG_DROPOUT if args.dropout_model == "zero_inflated" else TAG_NO_DROPOUT




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
    if args.activity_model == "direct" and args.negative_control_mode != "ordinary":
        raise ValueError(
            "--activity-model direct requires --negative-control-mode ordinary"
        )
    if args.outdir is None:
        args.outdir = default_outdir(args)
    existing_manifest = args.outdir / "run_manifest.json"
    if existing_manifest.exists() and not args.overwrite:
        raise SystemExit(
            f"refusing to overwrite the completed fit in {args.outdir}\n"
            f"  {existing_manifest} already exists\n"
            "pass --overwrite, or point --outdir somewhere else"
        )
    if args.cpu:
        # Must precede the first touch of a model symbol.
        os.environ["JAX_PLATFORMS"] = "cpu"

    priors = dataclasses.replace(
        bsf.ModelPriors(),
        p_drop_t7_alpha=args.p_drop_t7_alpha,
        p_drop_t7_beta=args.p_drop_t7_beta,
        p_drop_cre_alpha=args.p_drop_cre_alpha,
        p_drop_cre_beta=args.p_drop_cre_beta,
    )

    data = CountData.from_h5ad(
        args.h5ad,
        section=args.section,
        max_cells=args.max_cells,
        max_cres=args.max_cres,
        seed=args.seed,
        negative_control_mode=args.negative_control_mode.replace("-", "_"),
    )

    posterior_sites = (
        ["all"] if any(site.lower() == "all" for site in args.posterior_sites)
        else args.posterior_sites
    )
    tag = run_tag(args)
    log(
        f"[bayesian-decoupled] fitting {data.n_cre} cCREs, "
        f"{data.n_subclasses} subclasses, tag={tag}"
    )
    result = bsf.fit_decoupled(
        **data.to_run_kwargs(),
        kmax=args.kmax,
        steps_t7=args.steps_t7,
        steps_cre=args.steps_cre,
        lr=args.lr,
        guide=args.guide,
        num_posterior=args.num_posterior,
        seed=args.seed,
        priors=priors,
        infection_quadrature_points=args.infection_quadrature_points,
        posterior_sites_to_return=posterior_sites,
        dropout_model=args.dropout_model,
        activity_model=args.activity_model,
    )
    result["config"].update(
        {
            "input": input_fingerprint(args.h5ad),
            "blacklist_cre": data.blacklist,
            "max_cells": args.max_cells,
            "max_cres": args.max_cres,
            "section": args.section,
            "posterior_sites_to_return": posterior_sites,
            "dropout_prior": dropout_prior_config(args),
            "dropout_model": args.dropout_model,
            "activity_model": args.activity_model,
            "negative_control_mode": args.negative_control_mode.replace("-", "_"),
        }
    )

    write_fit(
        result,
        args.outdir,
        tag,
        data=data,
        input_path=args.h5ad,
        manifest_extra={
            "method_variant": "bayesian_decoupled_t7_cre",
            "dropout_prior": dropout_prior_config(args),
            "dropout_model": args.dropout_model,
        },
    )
    log(f"[bayesian-decoupled] wrote intermediates to {args.outdir}")


if __name__ == "__main__":
    main()
