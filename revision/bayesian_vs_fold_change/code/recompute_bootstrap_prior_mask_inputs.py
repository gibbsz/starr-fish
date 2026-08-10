#!/usr/bin/env python3
"""Recompute Bootstrap q-value inputs without the <5-detected-cells mask.

The original bootstrap jobs stored the full bootstrap activity array, then wrote
q-values with a count-based pair filter. For the prior-dominated sensitivity
plots we need the same bootstrap statistic, but without that count filter. The
plots apply the shared cCRE blacklist + Bayesian prior_dominated mask later.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analysis_utils import ANALYSIS_DIR, OLD_DATA_BOOTSTRAP, log, write_json
from baystarrfish.stats import bh_fdr as _bh_fdr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "bootstrap_dirs",
        type=Path,
        nargs="*",
        default=[
            OLD_DATA_BOOTSTRAP,
            ANALYSIS_DIR / "results" / "sections" / "sec1" / "bootstrap",
            ANALYSIS_DIR / "results" / "sections" / "sec2" / "bootstrap",
        ],
    )
    parser.add_argument("--chunk-size", type=int, default=250)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def bh_fdr(frame: pd.DataFrame) -> pd.DataFrame:
    """BH over every finite entry of the frame, jointly."""
    return pd.DataFrame(
        _bh_fdr(frame.to_numpy(dtype=float)), index=frame.index, columns=frame.columns
    )


def log_chunk(array: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.log(array.astype(np.float64, copy=False))
    out[~np.isfinite(out)] = np.nan
    return out


def recompute_one(root: Path, chunk_size: int, force: bool) -> None:
    root = root.resolve()
    required = [root / "bootstrap_axes.json", root / "celltype_activity_array.npy"]
    missing = [path.name for path in required if not path.exists()]
    if missing:
        log(
            "[bootstrap prior-mask inputs] missing "
            + ", ".join(missing)
            + f"; skipping {root}"
        )
        return

    out_path = root / "qvalues_prior_mask_right.csv"
    if out_path.exists() and not force:
        log(f"[bootstrap prior-mask inputs] exists, skipping {root}")
        return

    axes = json.loads((root / "bootstrap_axes.json").read_text())
    groups = pd.Index(axes["subclasses"], dtype=str)
    cres = pd.Index(axes["cres"], dtype=str)
    negative_controls = pd.Index(
        pd.read_csv(root / "negative_controls.csv").iloc[:, 0].astype(str)
    )
    negative_mask = cres.isin(negative_controls)
    if not negative_mask.any():
        raise ValueError(f"{root} has no negative-control cCREs in bootstrap axes")

    activity_array = np.load(root / "celltype_activity_array.npy", mmap_mode="r")
    n_boot, n_groups, n_cres = activity_array.shape
    if n_groups != len(groups) or n_cres != len(cres):
        raise ValueError(
            f"{root} axis mismatch: array={activity_array.shape}, "
            f"groups={len(groups)}, cres={len(cres)}"
        )

    log(f"[bootstrap prior-mask inputs] recomputing {root}")
    cre_sum = np.zeros(n_cres, dtype=np.float64)
    cre_count = np.zeros(n_cres, dtype=np.float64)
    for start in range(0, n_boot, chunk_size):
        chunk = log_chunk(activity_array[start : start + chunk_size])
        finite = np.isfinite(chunk)
        cre_sum += np.nansum(chunk, axis=(0, 1))
        cre_count += finite.sum(axis=(0, 1))
    cre_mean = np.divide(
        cre_sum,
        cre_count,
        out=np.full(n_cres, np.nan, dtype=np.float64),
        where=cre_count > 0,
    )

    mean_sum = np.zeros((n_groups, n_cres), dtype=np.float64)
    mean_count = np.zeros((n_groups, n_cres), dtype=np.float64)
    neg_sum = np.zeros(n_groups, dtype=np.float64)
    neg_count = np.zeros(n_groups, dtype=np.float64)
    for start in range(0, n_boot, chunk_size):
        chunk = log_chunk(activity_array[start : start + chunk_size])
        chunk = chunk - cre_mean.reshape(1, 1, -1)
        finite = np.isfinite(chunk)
        mean_sum += np.nansum(chunk, axis=0)
        mean_count += finite.sum(axis=0)

        neg = np.nanmean(chunk[:, :, negative_mask], axis=2)
        neg_sum += np.nansum(neg, axis=0)
        neg_count += np.isfinite(neg).sum(axis=0)

    calibrated = np.divide(
        mean_sum,
        mean_count,
        out=np.full((n_groups, n_cres), np.nan, dtype=np.float64),
        where=mean_count > 0,
    )
    threshold = np.divide(
        neg_sum,
        neg_count,
        out=np.full(n_groups, np.nan, dtype=np.float64),
        where=neg_count > 0,
    )
    threshold_centered = calibrated - threshold.reshape(-1, 1)

    p_right = np.full((n_groups, n_cres), np.nan, dtype=np.float64)
    p_left = np.full((n_groups, n_cres), np.nan, dtype=np.float64)
    valid_group = np.isfinite(threshold)
    if valid_group.any():
        less = np.zeros((n_groups, n_cres), dtype=np.float64)
        greater = np.zeros((n_groups, n_cres), dtype=np.float64)
        threshold_3d = threshold.reshape(1, -1, 1)
        for start in range(0, n_boot, chunk_size):
            chunk = log_chunk(activity_array[start : start + chunk_size])
            chunk = chunk - cre_mean.reshape(1, 1, -1)
            less += (chunk < threshold_3d).sum(axis=0)
            greater += (chunk > threshold_3d).sum(axis=0)
        p_right[valid_group] = less[valid_group] / n_boot
        p_left[valid_group] = greater[valid_group] / n_boot

    calibrated_df = pd.DataFrame(calibrated, index=groups, columns=cres)
    threshold_centered_df = pd.DataFrame(
        threshold_centered, index=groups, columns=cres
    )
    p_right_df = pd.DataFrame(p_right, index=groups, columns=cres)
    p_left_df = pd.DataFrame(p_left, index=groups, columns=cres)
    q_right = bh_fdr(p_right_df)
    q_left = bh_fdr(p_left_df)
    q_two = pd.DataFrame(
        np.fmin(q_right.to_numpy(float), q_left.to_numpy(float)),
        index=groups,
        columns=cres,
    )

    calibrated_df.to_csv(root / "log_activity_prior_mask_self_cre_calibrated.csv")
    threshold_centered_df.to_csv(
        root / "log_activity_prior_mask_vs_negative_control.csv"
    )
    q_two.to_csv(root / "qvalues_prior_mask_two_sided.csv")
    q_right.to_csv(root / "qvalues_prior_mask_right.csv")
    q_left.to_csv(root / "qvalues_prior_mask_left.csv")
    write_json(
        root / "prior_mask_qvalue_manifest.json",
        {
            "source_array": str(root / "celltype_activity_array.npy"),
            "calibrate": "self-CRE",
            "threshold": "neg_control_mean",
            "removed_filters": [
                "min_detected_cells",
                "bootstrap_min_1pct_finite_samples",
            ],
            "remaining_plot_filters": ["cre_blacklist", "bayesian_prior_dominated"],
            "n_bootstrap": int(n_boot),
            "n_groups": int(n_groups),
            "n_cres": int(n_cres),
            "chunk_size": int(chunk_size),
        },
    )
    log(f"[bootstrap prior-mask inputs] wrote {root}")


def main() -> None:
    args = parse_args()
    for root in args.bootstrap_dirs:
        recompute_one(root, args.chunk_size, args.force)


if __name__ == "__main__":
    main()
