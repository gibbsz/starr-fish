#!/usr/bin/env python3
"""Fit the Bayesian model to within-subclass bootstrapped meta-cells."""

from __future__ import annotations

import argparse
import dataclasses
import os
from pathlib import Path

import numpy as np
import pandas as pd

# Lazy facade: no JAX or NumPyro import until a model symbol is touched, which is
# what lets --cpu set JAX_PLATFORMS before the backend is chosen.
import baystarrfish as bsf
from baystarrfish.data import CountData
from baystarrfish.io import input_fingerprint, write_fit

from analysis_utils import ANALYSIS_DIR, DEFAULT_H5AD, log


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
    data: CountData,
    *,
    bootstrap_size: int,
    bootstrap_number: int,
    seed: int,
) -> tuple[CountData, pd.DataFrame]:
    """Sum fixed-size with-replacement cell samples within each subclass.

    Returns a ``CountData`` whose rows are meta-cells rather than cells; the
    cCRE axis, library prior and control mask are inherited unchanged.
    """

    validate_bootstrap_args(bootstrap_size, bootstrap_number)
    obs = pd.DataFrame({"subclass": data.subclass, "class": data.class_}).astype(str)
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
    t7 = np.asarray(data.t7, dtype=np.int64)
    cre = np.asarray(data.cre, dtype=np.int64)

    n_meta = len(subclass_order) * bootstrap_number
    meta_t7 = np.empty((n_meta, data.n_cre), dtype=np.int64)
    meta_cre = np.empty((n_meta, data.n_cre), dtype=np.int64)
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
    metacells = dataclasses.replace(
        data,
        t7=meta_t7,
        cre=meta_cre,
        subclass=meta_obs["subclass"].to_numpy(),
        class_=meta_obs["class"].to_numpy(),
        # Cell counts of the source data, which is what the fit is a summary of.
        subclass_cell_counts=obs["subclass"].value_counts().sort_index().rename("n_cells"),
    )
    return metacells, meta_obs


def main() -> None:
    args = parse_args()
    if args.outdir is None:
        args.outdir = default_outdir(args)
    if args.cpu:
        # Must precede the first touch of a model symbol.
        os.environ["JAX_PLATFORMS"] = "cpu"

    # The meta-cell ablation pools the annotated controls in-model.
    cells = CountData.from_h5ad(
        args.h5ad,
        section=args.section,
        max_cells=args.max_cells,
        max_cres=args.max_cres,
        seed=args.seed,
        negative_control_mode="pooled",
    )
    n_source_cells = cells.n_cells

    data, meta_obs = build_bootstrap_metacells(
        cells,
        bootstrap_size=args.bootstrap_size,
        bootstrap_number=args.bootstrap_number,
        seed=args.seed,
    )

    args.outdir.mkdir(parents=True, exist_ok=True)
    source_cell_counts = pd.Series(cells.subclass).value_counts().sort_index()
    source_cell_counts.rename("n_source_cells").to_csv(
        args.outdir / "source_subclass_cell_counts.csv"
    )
    meta_obs.to_csv(args.outdir / "metacell_obs.csv")
    meta_obs["subclass"].value_counts().sort_index().rename("n_metacells").to_csv(
        args.outdir / "metacell_subclass_counts.csv"
    )

    tag = f"subclass_{args.channel}_{args.infection_model}_{args.method}"
    log(
        f"[bayesian-metacell] fitting {data.n_cre} cCREs, "
        f"{data.n_subclasses} subclasses, "
        f"{data.n_cells:,} meta-cells, "
        f"bootstrap_size={args.bootstrap_size}, tag={tag}"
    )
    posterior_sites = (
        ["all"] if any(site.lower() == "all" for site in args.posterior_sites)
        else args.posterior_sites
    )
    result = bsf.fit(
        **data.to_run_kwargs(),
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
        infection_model=args.infection_model,
        posterior_sites_to_return=posterior_sites,
    )
    result["config"].update(
        {
            "input": input_fingerprint(args.h5ad),
            "input_transform": "bootstrap_metacells",
            "blacklist_cre": data.blacklist,
            "max_cells": args.max_cells,
            "max_cres": args.max_cres,
            "section": args.section,
            "bootstrap_size": args.bootstrap_size,
            "bootstrap_number": args.bootstrap_number,
            "n_source_cells": n_source_cells,
            "n_metacells": data.n_cells,
            "posterior_sites_to_return": posterior_sites,
        }
    )

    write_fit(
        result,
        args.outdir,
        tag,
        data=data,
        input_path=args.h5ad,
        manifest_extra={
            "method_variant": "bayesian_bootstrap_metacells",
            "n_source_cells": n_source_cells,
            "n_metacells": data.n_cells,
            "bootstrap_size": args.bootstrap_size,
            "bootstrap_number": args.bootstrap_number,
        },
    )
    log(f"[bayesian-metacell] wrote intermediates to {args.outdir}")


if __name__ == "__main__":
    main()
