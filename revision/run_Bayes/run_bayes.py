#!/usr/bin/env python3
"""Fit the hierarchical joint CRE/T7 model to the 5/28 BRBB500gn dataset."""

from __future__ import annotations

import argparse
import dataclasses
import os
import sys
from pathlib import Path

# Importing baystarrfish costs nothing: the facade is lazy and pulls in neither
# JAX nor NumPyro until a model symbol is touched, which is what lets --cpu set
# JAX_PLATFORMS below before the backend is chosen.
import baystarrfish as bsf
from baystarrfish.data import CountData
from baystarrfish.data.controls import POOLED_NEGATIVE_CONTROL_NAME  # noqa: F401
from baystarrfish.io import input_fingerprint, write_fit

# analysis_utils lives with the analysis it configures, not with this driver. It
# resolves ANALYSIS_DIR from its own location, so the ablation and per-section
# outputs below keep landing under bayesian_vs_fold_change/results/ regardless of
# where this script is invoked from.
ANALYSIS_CODE = Path(__file__).resolve().parent.parent / "bayesian_vs_fold_change" / "code"
if str(ANALYSIS_CODE) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_CODE))

from analysis_utils import (  # noqa: E402  (path injected above)
    ANALYSIS_DIR,
    DEFAULT_H5AD,
    OLD_DATA_BAYES,
    log,
)


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
        default="ordinary",
        help=(
            "Fit the annotated negative controls as ordinary cCREs (the default, and "
            "the only mode --activity-model direct accepts), pool them through shared "
            "activity parameters, or fit the seven ordinary columns plus one appended "
            "all-seven pooled pseudo-cCRE in the same model."
        ),
    )
    parser.add_argument(
        "--activity-model",
        choices=["hierarchical", "direct"],
        default="direct",
        help=(
            "Directly estimate an exchangeable raw log_gamma matrix (the default; "
            "requires ordinary negative controls) or use the alpha/eta/delta "
            "hierarchy. The two shrink thinly-measured pairs toward different "
            "targets, so fits made under different settings are not comparable "
            "pair-for-pair."
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
        default=["log_gamma", "log_rho", "log_a"],
        help=(
            "Posterior sites to save in *_posterior_samples.npz. "
            "Use 'all' to save every sampled and deterministic posterior site. "
            "log_rho and log_a are in the default because the latent copy number "
            "cannot be recovered without them and they cannot be reconstructed "
            "after the fact; together they add ~3 MB to a 444 MB file."
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-cells", type=int, default=None, help="Smoke testing only.")
    parser.add_argument("--max-cres", type=int, default=None, help="Smoke testing only.")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Replace a completed fit. Without it, an outdir that already holds a "
            "run_manifest.json is refused. The default outdir for the production "
            "settings is Bayes_OldData/bayesian, so a bare invocation would otherwise "
            "overwrite the shipped fit."
        ),
    )
    return parser.parse_args()


def default_outdir(args: argparse.Namespace) -> Path:
    variant = method_variant(args)
    if args.section == "all":
        if variant == "bayesian_joint_dropout_direct_activity_ordinary_negative_controls":
            return OLD_DATA_BAYES
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



def main() -> None:
    args = parse_args()
    if args.activity_model == "direct" and args.negative_control_mode != "ordinary":
        raise ValueError(
            "--activity-model direct requires --negative-control-mode ordinary"
        )
    if args.outdir is None:
        args.outdir = default_outdir(args)
    # Fits are cheap to recompute (~13 min) but their outputs are cited downstream,
    # so replacing one must be deliberate. This matters most for the default settings,
    # whose default outdir is the shipped production fit.
    existing_manifest = Path(args.outdir) / "run_manifest.json"
    if existing_manifest.exists() and not args.overwrite:
        raise SystemExit(
            f"refusing to overwrite the completed fit in {args.outdir}\n"
            f"  {existing_manifest} already exists\n"
            "pass --overwrite, or point --outdir somewhere else"
        )
    if args.cpu:
        # Must precede the first touch of a model symbol, which is what triggers
        # the JAX import and therefore the backend choice.
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

    tag = f"{args.level}_{args.channel}_{args.infection_model}_{args.method}"
    log(
        f"[bayesian] fitting {data.n_cre} cCREs, "
        f"{data.n_subclasses} subclasses, tag={tag}"
    )
    posterior_sites = (
        ["all"] if any(site.lower() == "all" for site in args.posterior_sites)
        else args.posterior_sites
    )
    result = bsf.fit(
        **data.to_run_kwargs(),
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
        infection_model=args.infection_model,
        activity_model=args.activity_model,
        priors=priors,
        posterior_sites_to_return=posterior_sites,
    )
    result["config"].update(
        {
            "input": input_fingerprint(args.h5ad),
            "blacklist_cre": data.blacklist,
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
            "annotated_negative_control_cre": data.negative_controls,
            "pooled_negative_control": data.pooled_negative_control,
        }
    )

    write_fit(
        result,
        args.outdir,
        tag,
        data=data,
        input_path=args.h5ad,
        manifest_extra={
            "method_variant": method_variant(args),
            "activity_model": args.activity_model,
            "dropout_prior": dropout_prior_config(args),
        },
    )
    log(f"[bayesian] wrote intermediates to {args.outdir}")


if __name__ == "__main__":
    main()
