#!/usr/bin/env python3
"""T7-filtered cCRE-vs-negative-control activity tests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analysis_utils import ANALYSIS_DIR, DEFAULT_H5AD, log, write_json
from baystarrfish.stats import bh_fdr
from plot_method_activity_correlation import (
    pair_count_totals,
    read_cre_blacklist,
)


METHODS = (
    "Bootstrap",
    "Joint",
    "Decoupled",
    "Joint+dropout",
    "Decoupled+dropout",
    "Metacell Bayesian",
)
METHOD_ROOTS = {
    "Bootstrap": ANALYSIS_DIR / "results" / "bootstrap",
    "Joint": ANALYSIS_DIR / "results" / "ablation" / "bayesian_joint",
    "Decoupled": (
        ANALYSIS_DIR / "results" / "ablation" / "bayesian_decoupled_no_dropout"
    ),
    "Joint+dropout": (
        ANALYSIS_DIR / "results" / "ablation" / "bayesian_joint_dropout"
    ),
    "Decoupled+dropout": (
        ANALYSIS_DIR / "results" / "ablation" / "bayesian_decoupled"
    ),
    "Metacell Bayesian": ANALYSIS_DIR
    / "results"
    / "ablation"
    / "bayesian_bootstrap_metacells_size100_number100",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5ad", type=Path, default=DEFAULT_H5AD)
    parser.add_argument(
        "--tables-dir", type=Path, default=ANALYSIS_DIR / "results" / "tables"
    )
    parser.add_argument(
        "--figures-dir", type=Path, default=ANALYSIS_DIR / "results" / "figures"
    )
    parser.add_argument(
        "--t7-thresholds",
        type=float,
        nargs="+",
        default=[5.0, 10.0, 20.0, 50.0, 100.0],
        help="Subclass-cCRE total T7 count filters. Uses >= threshold.",
    )
    parser.add_argument("--bootstrap-log-chunk-size", type=int, default=250)
    parser.add_argument("--q-cutoff", type=float, default=0.05)
    parser.add_argument(
        "--activity-centering",
        choices=["none", "self-cre", "posterior-alpha"],
        default="none",
        help=(
            "Use raw activity, empirical self-cCRE centering, or exact posterior "
            "alpha subtraction for Bayesian methods (bootstrap remains raw)."
        ),
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=METHODS,
        default=list(METHODS),
        help="Methods to include in the output.",
    )
    parser.add_argument(
        "--joint-root",
        type=Path,
        default=METHOD_ROOTS["Joint"],
        help="Bayesian run used for the Joint method.",
    )
    parser.add_argument(
        "--decoupled-root",
        type=Path,
        default=METHOD_ROOTS["Decoupled"],
        help="Bayesian run used for the Decoupled method.",
    )
    parser.add_argument(
        "--joint-dropout-root",
        type=Path,
        default=METHOD_ROOTS["Joint+dropout"],
        help="Bayesian run used for the Joint+dropout method.",
    )
    parser.add_argument(
        "--decoupled-dropout-root",
        type=Path,
        default=METHOD_ROOTS["Decoupled+dropout"],
        help="Bayesian run used for the Decoupled+dropout method.",
    )
    parser.add_argument("--stem", default="method_activity_t7_filter_negative_control")
    return parser.parse_args()


def discover_tag(root: Path) -> str:
    return json.loads((root / "run_manifest.json").read_text())["tag"]


def read_negative_controls(root: Path) -> list[str]:
    return pd.read_csv(root / "negative_controls.csv").iloc[:, 0].astype(str).tolist()




def threshold_suffix(threshold: float) -> str:
    threshold = float(threshold)
    if threshold.is_integer():
        return str(int(threshold))
    return str(threshold).replace(".", "p")


def aligned_t7_totals(
    h5ad: Path,
    groups: pd.Index,
    cres: pd.Index,
) -> pd.DataFrame:
    pair_t7, _ = pair_count_totals(h5ad, groups, cres)
    groups = groups.astype(str)
    cres = cres.astype(str)
    return pair_t7.reindex(index=groups, columns=cres).fillna(0.0)


def finite_mean(values: np.ndarray, counts: np.ndarray) -> np.ndarray:
    return np.divide(
        values,
        counts,
        out=np.full(values.shape, np.nan, dtype=np.float64),
        where=counts > 0,
    )


def bootstrap_base_statistics(
    root: Path,
    chunk_size: int,
    activity_centering: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, list[str]]:
    axes = json.loads((root / "bootstrap_axes.json").read_text())
    groups = pd.Index(axes["subclasses"], dtype=str)
    cres = pd.Index(axes["cres"], dtype=str)
    negative_controls = read_negative_controls(root)
    negative_mask = cres.isin(negative_controls)
    if not negative_mask.any():
        raise ValueError(f"{root} has no negative controls in bootstrap axes")

    activity_array = np.load(root / "celltype_activity_array.npy", mmap_mode="r")
    cre_raw = np.load(root / "celltype_CRE_raw.npy", mmap_mode="r")
    t7_raw = np.load(root / "celltype_T7_raw.npy", mmap_mode="r")
    n_boot, n_groups, n_cres = activity_array.shape
    if n_groups != len(groups) or n_cres != len(cres):
        raise ValueError(f"{root} bootstrap axes do not match array shape")
    if cre_raw.shape != activity_array.shape or t7_raw.shape != activity_array.shape:
        raise ValueError(f"{root} bootstrap raw count arrays do not match activity shape")

    log_sum = np.zeros((n_groups, n_cres), dtype=np.float64)
    log_count = np.zeros((n_groups, n_cres), dtype=np.float64)
    negative_sum = np.zeros(n_groups, dtype=np.float64)
    negative_count = np.zeros(n_groups, dtype=np.float64)
    n_less_equal = np.zeros((n_groups, n_cres), dtype=np.float64)
    n_testable = np.zeros((n_groups, n_cres), dtype=np.float64)

    if activity_centering == "self-cre":
        center_sum = np.zeros(n_cres, dtype=np.float64)
        center_count = np.zeros(n_cres, dtype=np.float64)
        for start in range(0, n_boot, chunk_size):
            stop = min(start + chunk_size, n_boot)
            with np.errstate(divide="ignore", invalid="ignore"):
                logged = np.log(
                    activity_array[start:stop].astype(np.float64, copy=False)
                )
            finite = np.isfinite(logged)
            center_sum += np.where(finite, logged, 0.0).sum(axis=(0, 1))
            center_count += finite.sum(axis=(0, 1))
        cre_center = finite_mean(center_sum, center_count)

        for start in range(0, n_boot, chunk_size):
            stop = min(start + chunk_size, n_boot)
            with np.errstate(divide="ignore", invalid="ignore"):
                logged = np.log(
                    activity_array[start:stop].astype(np.float64, copy=False)
                )
            logged = logged - cre_center[None, None, :]
            finite = np.isfinite(logged)
            log_sum += np.where(finite, logged, 0.0).sum(axis=0)
            log_count += finite.sum(axis=0)
            negative_finite = finite[:, :, negative_mask]
            negative_draw_sum = np.where(
                negative_finite, logged[:, :, negative_mask], 0.0
            ).sum(axis=2)
            negative_draw_count = negative_finite.sum(axis=2)
            negative_logged = np.divide(
                negative_draw_sum,
                negative_draw_count,
                out=np.full(negative_draw_sum.shape, np.nan, dtype=np.float64),
                where=negative_draw_count > 0,
            )
            negative_draw_finite = np.isfinite(negative_logged)
            negative_sum += np.where(
                negative_draw_finite, negative_logged, 0.0
            ).sum(axis=0)
            negative_count += negative_draw_finite.sum(axis=0)

        negative_threshold = finite_mean(negative_sum, negative_count)
        for start in range(0, n_boot, chunk_size):
            stop = min(start + chunk_size, n_boot)
            with np.errstate(divide="ignore", invalid="ignore"):
                logged = np.log(
                    activity_array[start:stop].astype(np.float64, copy=False)
                )
            logged = logged - cre_center[None, None, :]
            finite = np.isfinite(logged)
            testable = finite & np.isfinite(negative_threshold)[None, :, None]
            n_less_equal += (
                (logged < negative_threshold[None, :, None]) & testable
            ).sum(axis=0)
            n_testable += testable.sum(axis=0)
    else:
        for start in range(0, n_boot, chunk_size):
            stop = min(start + chunk_size, n_boot)
            chunk = activity_array[start:stop]
            with np.errstate(divide="ignore", invalid="ignore"):
                logged = np.log(chunk.astype(np.float64, copy=False))
                negative_cre = cre_raw[start:stop, :, negative_mask].sum(axis=2)
                negative_t7 = t7_raw[start:stop, :, negative_mask].sum(axis=2)
                negative_logged = np.log(negative_cre / negative_t7)
            finite = np.isfinite(logged)
            log_sum += np.where(finite, logged, 0.0).sum(axis=0)
            log_count += finite.sum(axis=0)
            negative_finite = np.isfinite(negative_logged)
            negative_sum += np.where(
                negative_finite, negative_logged, 0.0
            ).sum(axis=0)
            negative_count += negative_finite.sum(axis=0)
            testable = finite & negative_finite[:, :, None]
            n_less_equal += (
                (logged <= negative_logged[:, :, None]) & testable
            ).sum(axis=0)
            n_testable += testable.sum(axis=0)

    activity = pd.DataFrame(
        finite_mean(log_sum, log_count),
        index=groups,
        columns=cres,
    )
    negative_mean = pd.Series(
        finite_mean(negative_sum, negative_count),
        index=groups,
        name="negative_control_log",
    )
    pvalues = pd.DataFrame(
        np.divide(
            n_less_equal,
            n_testable,
            out=np.full((n_groups, n_cres), np.nan, dtype=np.float64),
            where=n_testable > 0,
        ),
        index=groups,
        columns=cres,
    )
    return activity, pvalues, negative_mean, negative_controls


def bayesian_base_statistics(
    root: Path,
    activity_centering: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, list[str], int]:
    tag = discover_tag(root)
    negative_controls = read_negative_controls(root)
    with np.load(root / f"{tag}_posterior_samples.npz", allow_pickle=True) as posterior:
        log_gamma = posterior["log_gamma"].astype(np.float32)
        groups = pd.Index(posterior["group_names"].astype(str), dtype=str)
        cres = pd.Index(posterior["cre_names"].astype(str), dtype=str)
        log_gamma_neg = (
            posterior["log_gamma_neg"].astype(np.float32)
            if "log_gamma_neg" in posterior.files
            else None
        )
        if activity_centering == "posterior-alpha":
            required = {"alpha", "alpha_neg", "log_gamma_neg"}
            missing = required.difference(posterior.files)
            if missing:
                raise ValueError(
                    f"{root} is missing posterior sites required for alpha subtraction: "
                    f"{sorted(missing)}"
                )
            alpha = posterior["alpha"].astype(np.float32)
            alpha_neg = posterior["alpha_neg"].astype(np.float32)
        else:
            alpha = None
            alpha_neg = None

    negative_mask = cres.isin(negative_controls)
    if not negative_mask.any():
        raise ValueError(f"{root} has no negative controls in posterior")

    if activity_centering == "posterior-alpha":
        expected_alpha_shape = (log_gamma.shape[0], log_gamma.shape[2])
        if alpha.shape != expected_alpha_shape:
            raise ValueError(
                f"{root} alpha shape {alpha.shape} does not match {expected_alpha_shape}"
            )
        if alpha_neg.shape != (log_gamma.shape[0],):
            raise ValueError(
                f"{root} alpha_neg shape {alpha_neg.shape} does not match "
                f"({log_gamma.shape[0]},)"
            )
        if log_gamma_neg.shape != log_gamma.shape[:2]:
            raise ValueError(
                f"{root} log_gamma_neg shape {log_gamma_neg.shape} does not match "
                f"posterior draw/group shape {log_gamma.shape[:2]}"
            )
        activity_draws = log_gamma - alpha[:, None, :]
        negative_draws = log_gamma_neg - alpha_neg[:, None]
        activity_values = activity_draws.mean(axis=0, dtype=np.float64)
        negative_mean = negative_draws.mean(axis=0, dtype=np.float64)
        pvalues = (activity_draws <= negative_draws[:, :, None]).mean(
            axis=0, dtype=np.float64
        )
    elif activity_centering == "self-cre":
        cre_center = log_gamma.mean(axis=(0, 1), dtype=np.float64).astype(np.float32)
        centered = log_gamma - cre_center[None, None, :]
        negative_draws = centered[:, :, negative_mask].mean(axis=2)
        negative_mean = negative_draws.mean(axis=0, dtype=np.float64)
        activity_values = centered.mean(axis=0, dtype=np.float64)
        pvalues = (centered < negative_mean[None, :, None]).mean(
            axis=0, dtype=np.float64
        )
    elif log_gamma_neg is not None:
        if log_gamma_neg.shape != log_gamma.shape[:2]:
            raise ValueError(
                f"{root} log_gamma_neg shape {log_gamma_neg.shape} does not match "
                f"posterior draw/group shape {log_gamma.shape[:2]}"
            )
        negative_draws = log_gamma_neg
        activity_values = log_gamma.mean(axis=0, dtype=np.float64)
        negative_mean = negative_draws.mean(axis=0, dtype=np.float64)
        pvalues = (log_gamma <= negative_draws[:, :, None]).mean(
            axis=0, dtype=np.float64
        )
    else:
        negative_draws = log_gamma[:, :, negative_mask].mean(axis=2)
        activity_values = log_gamma.mean(axis=0, dtype=np.float64)
        negative_mean = negative_draws.mean(axis=0, dtype=np.float64)
        pvalues = (log_gamma <= negative_draws[:, :, None]).mean(
            axis=0, dtype=np.float64
        )
    return (
        pd.DataFrame(activity_values, index=groups, columns=cres),
        pd.DataFrame(pvalues, index=groups, columns=cres),
        pd.Series(negative_mean, index=groups, name="negative_control_log"),
        negative_controls,
        int(log_gamma.shape[0]),
    )


def long_test_table(
    method: str,
    activity: pd.DataFrame,
    pvalues: pd.DataFrame,
    negative_mean: pd.Series,
    negative_controls: list[str],
    pair_t7: pd.DataFrame,
    thresholds: list[float],
    q_cutoff: float,
    *,
    n_draws: int,
    blacklist: set[str],
) -> pd.DataFrame:
    activity = activity.copy()
    activity.index = activity.index.astype(str)
    activity.columns = activity.columns.astype(str)
    pvalues = pvalues.reindex(index=activity.index, columns=activity.columns)
    negative_mean = negative_mean.reindex(activity.index.astype(str))
    pair_t7 = pair_t7.reindex(index=activity.index, columns=activity.columns).fillna(0.0)

    negative_controls = [cre for cre in negative_controls if cre in pair_t7.columns]
    negative_t7_total = pair_t7.loc[:, negative_controls].sum(axis=1)
    negative_set = set(negative_controls)
    records = []
    group_values = activity.index.to_numpy(str)
    cre_values = activity.columns.to_numpy(str)
    group_grid, cre_grid = np.meshgrid(group_values, cre_values, indexing="ij")
    is_negative = np.isin(cre_grid.ravel(), list(negative_set))
    is_blacklisted = np.isin(cre_grid.ravel(), list(blacklist))
    activity_flat = activity.to_numpy(float).ravel()
    p_flat = pvalues.to_numpy(float).ravel()
    pair_t7_flat = pair_t7.to_numpy(float).ravel()
    negative_log_flat = np.repeat(negative_mean.to_numpy(float), len(cre_values))
    negative_t7_flat = np.repeat(negative_t7_total.to_numpy(float), len(cre_values))

    for threshold in thresholds:
        valid = (
            (pair_t7_flat >= threshold)
            & (negative_t7_flat >= threshold)
            & ~is_negative
            & ~is_blacklisted
            & np.isfinite(activity_flat)
            & np.isfinite(p_flat)
            & np.isfinite(negative_log_flat)
        )
        threshold_p = np.where(valid, p_flat, np.nan)
        threshold_q = bh_fdr(threshold_p)
        frame = pd.DataFrame(
            {
                "t7_threshold": threshold,
                "method": method,
                "group": group_grid.ravel(),
                "cre": cre_grid.ravel(),
                "activity_log": activity_flat,
                "negative_control_log": negative_log_flat,
                "effect_vs_negative_log": activity_flat - negative_log_flat,
                "p_right": threshold_p,
                "q_right": threshold_q,
                "significant_q": threshold_q <= q_cutoff,
                "target_t7_total": pair_t7_flat,
                "negative_control_t7_total": negative_t7_flat,
                "passes_target_t7": pair_t7_flat >= threshold,
                "passes_negative_control_t7": negative_t7_flat >= threshold,
                "is_negative_control": is_negative,
                "is_blacklisted": is_blacklisted,
                "n_negative_controls": len(negative_controls),
                "n_draws": n_draws,
            }
        )
        records.append(frame.loc[valid].copy())
    return pd.concat(records, ignore_index=True)


def summary_table(results: pd.DataFrame, q_cutoff: float) -> pd.DataFrame:
    if results.empty:
        return pd.DataFrame()
    return (
        results.groupby(["t7_threshold", "method"], sort=False)
        .agg(
            tested_pairs=("q_right", "size"),
            significant_pairs=("significant_q", "sum"),
            tested_cell_types=("group", "nunique"),
            tested_cres=("cre", "nunique"),
            median_effect=("effect_vs_negative_log", "median"),
            median_target_t7=("target_t7_total", "median"),
            median_negative_control_t7=("negative_control_t7_total", "median"),
        )
        .assign(q_cutoff=q_cutoff)
        .reset_index()
    )


def main() -> None:
    args = parse_args()
    args.tables_dir.mkdir(parents=True, exist_ok=True)
    args.figures_dir.mkdir(parents=True, exist_ok=True)
    thresholds = sorted({float(threshold) for threshold in args.t7_thresholds})
    method_roots = dict(METHOD_ROOTS)
    method_roots["Joint"] = args.joint_root
    method_roots["Decoupled"] = args.decoupled_root
    method_roots["Joint+dropout"] = args.joint_dropout_root
    method_roots["Decoupled+dropout"] = args.decoupled_dropout_root
    methods = tuple(dict.fromkeys(args.methods))
    blacklist = set().union(
        *(read_cre_blacklist(method_roots[method]) for method in methods)
    )

    log("[t7 negative-control stats] loading pair-level T7 totals")
    # Use the broad bootstrap axes so pair-level T7 is available for all mapped cCREs.
    bootstrap_axes = json.loads(
        (METHOD_ROOTS["Bootstrap"] / "bootstrap_axes.json").read_text()
    )
    all_groups = pd.Index(bootstrap_axes["subclasses"], dtype=str)
    all_cres = pd.Index(bootstrap_axes["cres"], dtype=str)
    pair_t7_all = aligned_t7_totals(args.h5ad, all_groups, all_cres)

    output_frames = []
    manifest_methods = {}
    for method in methods:
        root = method_roots[method]
        log(f"[t7 negative-control stats] computing {method}")
        if method == "Bootstrap":
            axes = json.loads((root / "bootstrap_axes.json").read_text())
            bootstrap_groups = pd.Index(axes["subclasses"], dtype=str)
            bootstrap_cres = pd.Index(axes["cres"], dtype=str)
            method_t7 = pair_t7_all.reindex(
                index=bootstrap_groups.astype(str), columns=bootstrap_cres.astype(str)
            ).fillna(0.0)
            activity, pvalues, negative_mean, negative_controls = bootstrap_base_statistics(
                root, args.bootstrap_log_chunk_size, args.activity_centering
            )
            n_draws = int(np.load(root / "celltype_activity_array.npy", mmap_mode="r").shape[0])
        else:
            activity, pvalues, negative_mean, negative_controls, n_draws = (
                bayesian_base_statistics(root, args.activity_centering)
            )
            method_t7 = pair_t7_all.reindex(
                index=activity.index.astype(str), columns=activity.columns.astype(str)
            ).fillna(0.0)
        table = long_test_table(
            method,
            activity,
            pvalues,
            negative_mean,
            negative_controls,
            method_t7,
            thresholds,
            args.q_cutoff,
            n_draws=n_draws,
            blacklist=blacklist,
        )
        output_frames.append(table)
        manifest_methods[method] = {
            "root": str(root),
            "n_draws": n_draws,
            "negative_controls": negative_controls,
        }

    results = pd.concat(output_frames, ignore_index=True)
    result_path = args.tables_dir / f"{args.stem}_tests.csv.gz"
    summary_path = args.tables_dir / f"{args.stem}_summary.csv"
    results.to_csv(result_path, index=False)
    summary = summary_table(results, args.q_cutoff)
    summary.to_csv(summary_path, index=False)

    for threshold, frame in results.groupby("t7_threshold", sort=True):
        suffix = threshold_suffix(float(threshold))
        frame.to_csv(args.tables_dir / f"{args.stem}_t7_ge{suffix}_tests.csv.gz", index=False)

    if args.activity_centering == "posterior-alpha":
        pvalue_definition = (
            "Bayesian p_right = Pr((log_gamma - alpha) <= "
            "(log_gamma_neg - alpha_neg)); bootstrap uses its raw draw-matched contrast"
        )
        bootstrap_negative_control_activity = (
            "unchanged raw bootstrap test: per draw, log(sum bootstrapped "
            "negative-control cCRE counts / sum bootstrapped negative-control T7 counts)"
        )
        bayesian_negative_control_activity = (
            "exact posterior alpha subtraction: target log_gamma - alpha compared "
            "draw-by-draw with log_gamma_neg - alpha_neg"
        )
    elif args.activity_centering == "self-cre":
        pvalue_definition = (
            "p_right = Pr(centered target activity draw < subclass mean of "
            "centered negative-control activity draws)"
        )
        bootstrap_negative_control_activity = (
            "self-CRE-centered log(cCRE/T7): subtract each cCRE's mean over all "
            "bootstrap draws and subclasses, average centered negative-control "
            "cCREs per draw/subclass, then average over draws for the subclass threshold"
        )
        bayesian_negative_control_activity = (
            "self-CRE-centered log_gamma: subtract each cCRE's mean over all posterior "
            "draws and subclasses, average centered negative-control cCREs per "
            "draw/subclass, then average over draws for the subclass threshold"
        )
    else:
        pvalue_definition = (
            "p_right = Pr(target_activity_draw <= negative_control_activity_draw)"
        )
        bootstrap_negative_control_activity = (
            "per bootstrap draw, log(sum bootstrapped negative-control cCRE counts / "
            "sum bootstrapped negative-control T7 counts) per cell type"
        )
        bayesian_negative_control_activity = (
            "raw, uncalibrated posterior draws: explicit log_gamma_neg when saved "
            "(including alpha_neg + eta_neg + delta_neg), otherwise mean log_gamma "
            "across negative-control cCREs per cell type; target log_gamma includes "
            "alpha + eta + delta"
        )

    manifest = {
        "test": "one-sided cCRE activity higher than the method-specific negative-control baseline",
        "activity_centering": args.activity_centering,
        "pvalue_definition": pvalue_definition,
        "bootstrap_negative_control_activity": bootstrap_negative_control_activity,
        "bayesian_negative_control_activity": bayesian_negative_control_activity,
        "fdr": "Benjamini-Hochberg per method and T7 threshold over valid non-negative-control, non-blacklisted cCRE-cell-type pairs",
        "t7_filter": "target cCRE-cell-type total T7 >= threshold; negative-control baseline eligible when sum T7 over all negative-control cCREs in the cell type >= threshold",
        "thresholds": thresholds,
        "q_cutoff": args.q_cutoff,
        "methods": manifest_methods,
        "blacklisted_cres_removed": sorted(blacklist),
        "outputs": {
            "tests": str(result_path),
            "summary": str(summary_path),
        },
    }
    write_json(args.figures_dir / f"{args.stem}_manifest.json", manifest)
    log(
        "[t7 negative-control stats] wrote "
        f"{len(results):,} tests to {result_path} and summary to {summary_path}"
    )


if __name__ == "__main__":
    main()
