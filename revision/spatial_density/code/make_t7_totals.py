#!/usr/bin/env python3
"""Total T7 counts per (subclass, cCRE), cached for the spatial density filter.

A cCRE's activity in a subclass is only estimable where that subclass carries
enough T7 signal for the construct. Below the threshold the per-cell posterior is
driven by the prior rather than by data, so those cells contribute a nearly
constant, subclass-determined value -- which both dilutes the spatial statistic
and, under the within-subclass null, forms permutation blocks that barely
randomise. Excluding them is a per-(subclass, cCRE) decision, not a per-cell one.

Computed straight from ``obsm['T7CRE']`` rather than taken from
``joint_dropout_direct_activity_mean_negative_control_tests_t7_ge50.csv.gz``:
that table is already filtered to the pairs that pass and covers only the 205
cCREs of the heatmap subset, whereas the activity matrix has 389. This script
covers all of them and is cross-checked against that table by --verify.

Writes a long CSV: ``subclass, cre, t7_total, cre_total, n_cells``.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Sequence

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WORKFLOW_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
REPO_ROOT = os.path.abspath(os.path.join(WORKFLOW_DIR, "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from baystarrfish.data import read_obs_metadata  # noqa: E402

DEFAULT_H5AD = os.path.join(
    REPO_ROOT,
    "revision",
    "Data",
    "scdata_5_28_2025_BRBB500gn_final_CRE_T7CRE_NEWNEW.h5ad",
)
DEFAULT_ACTIVITY_NPZ = os.path.join(
    REPO_ROOT, "revision", "Bayes_OldData", "copy_number", "activity_normalized.npz"
)
DEFAULT_OUT = os.path.join(WORKFLOW_DIR, "results", "subclass_cre_t7_totals.csv.gz")
DEFAULT_VERIFY = os.path.join(
    REPO_ROOT,
    "revision",
    "bayesian_vs_fold_change",
    "results",
    "tables",
    "joint_dropout_direct_activity_mean_negative_control_tests_t7_ge50.csv.gz",
)
# Preference order matches revision/t7_subclass_correlation; older inputs name it "T7".
T7_KEYS = ("T7CRE", "T7")


def log(message: str) -> None:
    print(message, flush=True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n")[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--h5ad", default=DEFAULT_H5AD)
    parser.add_argument("--activity-npz", default=DEFAULT_ACTIVITY_NPZ)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--verify-against", default=DEFAULT_VERIFY)
    parser.add_argument("--min-t7", type=float, default=50.0)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    import h5py

    if not os.path.exists(args.h5ad):
        raise FileNotFoundError(args.h5ad)

    metadata = read_obs_metadata(args.h5ad)
    subclass = metadata["subclass"].to_numpy()
    vocabulary, codes = np.unique(subclass, return_inverse=True)
    cells_per_subclass = np.bincount(codes, minlength=vocabulary.size)

    with np.load(args.activity_npz, allow_pickle=True) as store:
        cre_names = [str(name) for name in store["cre_names"]]
        npz_obs = pd.Index([str(name) for name in store["obs_names"]])
    # The T7 datasets are in h5ad row order; align the subclass codes to it.
    if not npz_obs.equals(pd.Index(metadata["obs_name"].to_numpy())):
        order = npz_obs.get_indexer(metadata["obs_name"].to_numpy())
        if (order < 0).any():
            raise ValueError("h5ad and activity matrix disagree on cell names")

    rows: list[pd.DataFrame] = []
    with h5py.File(args.h5ad, "r") as handle:
        obsm = handle["obsm"]
        key = next((name for name in T7_KEYS if name in obsm), None)
        if key is None:
            raise KeyError(f"none of {T7_KEYS} in obsm; have {list(obsm)}")
        group = obsm[key]
        log(f"[input] obsm[{key!r}], {len(cre_names)} cCREs x {codes.size} cells")
        for index, cre in enumerate(cre_names):
            if cre not in group:
                raise KeyError(f"obsm[{key!r}] has no {cre!r}")
            counts = np.asarray(group[cre][:], dtype=np.float64)
            totals = np.bincount(codes, weights=counts, minlength=vocabulary.size)
            rows.append(
                pd.DataFrame(
                    {
                        "subclass": vocabulary,
                        "cre": cre,
                        "t7_total": totals,
                        "cre_total": float(counts.sum()),
                        "n_cells": cells_per_subclass,
                    }
                )
            )
            if (index + 1) % 50 == 0:
                log(f"[progress] {index + 1}/{len(cre_names)}")

    frame = pd.concat(rows, ignore_index=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    frame.to_csv(args.out, index=False)
    passing = frame["t7_total"] >= args.min_t7
    log(
        f"[done] {len(frame)} (subclass, cCRE) pairs -> {args.out}; "
        f"{int(passing.sum())} ({100 * passing.mean():.1f}%) reach T7 >= {args.min_t7}"
    )

    if args.verify_against and os.path.exists(args.verify_against):
        reference = pd.read_csv(args.verify_against)[
            ["group", "cre", "target_t7_total"]
        ].rename(columns={"group": "subclass"})
        merged = reference.merge(frame, on=["subclass", "cre"], how="left")
        delta = (merged["t7_total"] - merged["target_t7_total"]).abs()
        log(
            f"[verify] {len(merged)} pairs vs the t7_ge50 tests table: "
            f"max abs difference {np.nanmax(delta.to_numpy()):.3f}, "
            f"{int(merged['t7_total'].isna().sum())} unmatched"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
