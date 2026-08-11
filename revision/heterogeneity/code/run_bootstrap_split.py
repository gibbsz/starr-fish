#!/usr/bin/env python3
"""Run the manuscript bootstrap on within-subclass heterogeneity labels.

Top-T7 subclasses are either partitioned into ``N_GROUPS`` random subsets or
relabelled with their annotated ``supertype_name`` values; all other subclasses
stay intact. Identical to
``revision/run_Bootstrap/run_bootstrap.py`` except for the
relabelling step, so subgroup activities are comparable to the intact fit.
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

CODE_DIR = Path(__file__).resolve().parent
ANALYSIS_DIR = CODE_DIR.parent
REF_CODE = ANALYSIS_DIR.parent / "bayesian_vs_fold_change" / "code"
sys.path.insert(0, str(REF_CODE))
sys.path.insert(0, str(CODE_DIR))

from analysis_utils import (  # noqa: E402
    DEFAULT_H5AD,
    LIBSIZE_CSV,
    STARRFISH_ROOT,
    atomic_save_array,
    cre_blacklist,
    input_fingerprint,
    log,
    negative_control_names,
    read_and_prepare_adata,
    write_json,
)

from relabel import (  # noqa: E402
    N_GROUPS,
    SPLIT_SEED,
    TOP_N,
    relabel_subclasses,
    relabel_subclasses_from_obs,
    top_subclasses,
)

DEFAULT_OUTDIR = ANALYSIS_DIR / "results" / "split" / "bootstrap"
DEFAULT_SUPERTYPE_OUTDIR = ANALYSIS_DIR / "results" / "supertype" / "bootstrap"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5ad", type=Path, default=DEFAULT_H5AD)
    parser.add_argument("--outdir", type=Path, default=None)
    parser.add_argument(
        "--grouping",
        choices=["random", "supertype"],
        default="random",
        help=(
            "Within-target grouping strategy. 'random' preserves the existing "
            "five-way split; 'supertype' uses h5ad obs['supertype_name']."
        ),
    )
    parser.add_argument("--bootstrap-number", type=int, default=10_000)
    parser.add_argument("--n-jobs", type=int, default=62)
    parser.add_argument("--min-detected-cells", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--top-n", type=int, default=TOP_N)
    parser.add_argument("--n-groups", type=int, default=N_GROUPS)
    parser.add_argument("--split-seed", type=int, default=SPLIT_SEED)
    parser.add_argument("--max-cells", type=int, default=None, help="Smoke testing only.")
    parser.add_argument("--max-cres", type=int, default=None, help="Smoke testing only.")
    return parser.parse_args()


def import_starrfish():
    os.chdir(STARRFISH_ROOT)
    sys.path.insert(0, str(STARRFISH_ROOT))
    from STARRFISH import STARRFISH

    return STARRFISH


def main() -> None:
    args = parse_args()
    if args.outdir is None:
        args.outdir = (
            DEFAULT_OUTDIR
            if args.grouping == "random"
            else DEFAULT_SUPERTYPE_OUTDIR
        )
    args.outdir = args.outdir.resolve()
    args.outdir.mkdir(parents=True, exist_ok=True)
    adata = read_and_prepare_adata(
        args.h5ad, section="all", max_cells=args.max_cells,
        max_cres=args.max_cres, seed=args.seed,
    )

    targets = top_subclasses(n=args.top_n)
    present = set(adata.obs["subclass"].astype(str).unique())
    missing = [t for t in targets if t not in present]
    if missing:
        raise ValueError(f"top subclasses absent from adata: {missing}")
    original = adata.obs["subclass"].astype(str).to_numpy()
    if args.grouping == "random":
        relabelled, assignment = relabel_subclasses(
            original, targets, n_groups=args.n_groups, seed=args.split_seed
        )
        subgroups_by_subclass = {
            target: [
                f"{target}_group_{group}"
                for group in range(1, args.n_groups + 1)
            ]
            for target in targets
        }
        subgroup_obs_column = None
    else:
        subgroup_obs_column = "supertype_name"
        if subgroup_obs_column not in adata.obs.columns:
            raise KeyError(
                f"h5ad obs is missing required column {subgroup_obs_column!r}"
            )
        relabelled, assignment, subgroups_by_subclass = (
            relabel_subclasses_from_obs(
                original,
                adata.obs[subgroup_obs_column].to_numpy(),
                targets,
            )
        )
    adata.obs["subclass"] = pd.Categorical(relabelled)
    n_subgroups_by_subclass = {
        target: len(members) for target, members in subgroups_by_subclass.items()
    }
    ordered_members = [
        member
        for target in targets
        for member in subgroups_by_subclass[target]
    ]
    subgroup_cell_counts = (
        assignment["new_subclass"]
        .value_counts()
        .reindex(ordered_members)
        .astype(int)
        .to_dict()
    )
    log(
        f"[bootstrap-split] grouping={args.grouping}; relabelled {len(targets)} "
        f"subclasses into {sum(n_subgroups_by_subclass.values())} groups; "
        f"{adata.obs['subclass'].nunique()} total subclasses"
    )

    cre_info = adata.uns["CRE_info"].copy()
    blacklist = cre_blacklist(cre_info.index)
    negative_controls = negative_control_names(cre_info, blacklist)
    cre_info.to_csv(args.outdir / "cre_info.csv")
    pd.Series(blacklist, name="cre").to_csv(args.outdir / "cre_blacklist.csv", index=False)
    pd.Series(negative_controls, name="cre").to_csv(
        args.outdir / "negative_controls.csv", index=False
    )
    assignment.to_csv(args.outdir / "cell_group_assignment.csv", index=False)
    pd.DataFrame(
        [
            {
                "original_subclass": target,
                "new_subclass": member,
                "n_cells": subgroup_cell_counts[member],
            }
            for target in targets
            for member in subgroups_by_subclass[target]
        ]
    ).to_csv(args.outdir / "subgroup_cell_counts.csv", index=False)
    pd.Series(targets, name="subclass").to_csv(
        args.outdir / "split_subclasses.csv", index=False
    )

    STARRFISH = import_starrfish()
    starrfish = STARRFISH(
        adata,
        cre_tag="obsm:CRE",
        t7_tag="obsm:T7CRE",
        celltype_tag="obs:subclass",
        atac_cpm=None,
        atac_counts=None,
        lib_size=str(LIBSIZE_CSV),
        blacklist_cre=blacklist,
    )
    del adata
    gc.collect()

    config = {
        "cell_types_to_use": None,
        "normalize_by_cell_rna": False,
        "normalize_by_cell_volume": False,
        "normalize_by_cell_t7": False,
        "normalize_by_celltype_rna": False,
        "normalize_by_celltype_volume": False,
        "normalize_by_celltype_t7": True,
        "filter_by_cell_t7": None,
        "normalize_by_negative_control": False,
        "normalize_by_libsize": False,
        "log_transform": False,
        "bootstrap_number": args.bootstrap_number,
        "bootstrap_to_fixed_pct": 1,
        "bootstrap_to_fixed_sample_size": None,
        "load_stored": False,
        "n_jobs": args.n_jobs,
    }
    log(
        f"[bootstrap-split] starting {args.bootstrap_number:,} iterations with "
        f"{args.n_jobs} workers"
    )
    result = starrfish.average_bootstrap_test(**config)
    activity = result["celltype_activity"]
    atomic_save_array(
        args.outdir / "celltype_activity_array.npy", result["celltype_activity_array"]
    )
    atomic_save_array(args.outdir / "celltype_CRE_raw.npy", result["celltype_CRE_raw"])
    atomic_save_array(args.outdir / "celltype_T7_raw.npy", result["celltype_T7_raw"])
    activity.to_csv(args.outdir / "celltype_activity.csv")
    write_json(
        args.outdir / "bootstrap_axes.json",
        {
            "axis_order": ["bootstrap", "subclass", "cCRE"],
            "subclasses": activity.index.astype(str).tolist(),
            "cres": activity.columns.astype(str).tolist(),
        },
    )

    detected = (
        (starrfish.get_cre_expression() > 0)
        .groupby(starrfish.get_celltypes(), observed=True)
        .sum()
    )
    filter_mask = detected < args.min_detected_cells
    blacklist_present = filter_mask.columns.intersection(blacklist)
    filter_mask.loc[:, blacklist_present] = True
    q_two, q_right, q_left, calibrated, threshold_centered = (
        starrfish.average_bootstrap_test_q(
            result,
            threshold="neg_control_mean",
            norm="T7",
            tail="all",
            to_filter=filter_mask,
            calibrate="self-CRE",
        )
    )
    detected.to_csv(args.outdir / "detected_cells_per_subclass_ccre.csv")
    filter_mask.to_csv(args.outdir / "qvalue_filter_mask.csv")
    q_two.to_csv(args.outdir / "qvalues_two_sided.csv")
    q_right.to_csv(args.outdir / "qvalues_right.csv")
    q_left.to_csv(args.outdir / "qvalues_left.csv")
    calibrated.to_csv(args.outdir / "log_activity_self_cre_calibrated.csv")
    threshold_centered.to_csv(args.outdir / "log_activity_vs_negative_control.csv")
    cell_counts = starrfish.get_celltypes().value_counts().sort_index()
    cell_counts.rename("n_cells").to_csv(args.outdir / "subclass_cell_counts.csv")
    grouping_config = {
        "grouping": args.grouping,
        "subgroup_obs_column": subgroup_obs_column,
        "subgroups_by_subclass": subgroups_by_subclass,
        "n_subgroups_by_subclass": n_subgroups_by_subclass,
        "subgroup_cell_counts": subgroup_cell_counts,
    }
    if args.grouping == "random":
        grouping_config.update(
            {
                "n_groups": args.n_groups,
                "split_seed": args.split_seed,
            }
        )
    write_json(
        args.outdir / "run_manifest.json",
        {
            "input": input_fingerprint(args.h5ad),
            "n_cells": int(cell_counts.sum()),
            "n_subclasses": int(len(cell_counts)),
            "n_cres": int(activity.shape[1]),
            "split_subclasses": targets,
            "blacklist": blacklist,
            "negative_controls": negative_controls,
            "min_detected_cells": args.min_detected_cells,
            "seed": args.seed,
            "config": {k: v for k, v in config.items() if k != "cell_types_to_use"},
            **grouping_config,
        },
    )
    log(f"[bootstrap-split] wrote intermediates and q-values to {args.outdir}")


if __name__ == "__main__":
    main()
