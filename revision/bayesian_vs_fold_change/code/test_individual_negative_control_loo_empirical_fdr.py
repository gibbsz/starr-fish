#!/usr/bin/env python3
"""Test raw activity using a leave-one-out individual-control empirical null."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import spearmanr

from analysis_utils import ANALYSIS_DIR, DEFAULT_H5AD, write_json
from plot_method_activity_correlation import read_cre_blacklist


POOLED_NAME = "NEGATIVE_CONTROL_POOL"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bayes-dir",
        type=Path,
        default=ANALYSIS_DIR / "results" / "bayesian",
    )
    parser.add_argument("--h5ad", type=Path, default=DEFAULT_H5AD)
    parser.add_argument("--t7-threshold", type=float, default=50.0)
    parser.add_argument("--effect-threshold", type=float, default=0.0)
    parser.add_argument("--fdr", type=float, default=0.05)
    parser.add_argument(
        "--figures-dir", type=Path, default=ANALYSIS_DIR / "results" / "figures"
    )
    parser.add_argument(
        "--tables-dir", type=Path, default=ANALYSIS_DIR / "results" / "tables"
    )
    parser.add_argument(
        "--stem",
        default=(
            "joint_dropout_direct_activity_individual_negative_control_"
            "loo_empirical_fdr"
        ),
    )
    return parser.parse_args()


def decode_strings(values: np.ndarray) -> np.ndarray:
    return np.asarray(
        [value.decode() if isinstance(value, bytes) else str(value) for value in values]
    )


def normalize_labels(values: np.ndarray) -> np.ndarray:
    return np.asarray(
        [re.sub(r"^\d+\s+", "", value).replace("/", "-") for value in values]
    )


def load_grouped_t7(
    h5ad: Path, posterior_groups: np.ndarray, cre_names: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return group-by-cCRE T7 totals, group classes, and group cell counts."""
    with h5py.File(h5ad, "r") as handle:
        subclass = handle["obs"]["subclass_name"]
        subclass_categories = normalize_labels(
            decode_strings(subclass["categories"][...])
        )
        subclass_codes = subclass["codes"][...].astype(np.int64)
        class_data = handle["obs"]["class_name"]
        class_categories = normalize_labels(decode_strings(class_data["categories"][...]))
        class_codes = class_data["codes"][...].astype(np.int64)
        valid = (subclass_codes >= 0) & (class_codes >= 0)

        subclass_lookup = {name: idx for idx, name in enumerate(subclass_categories)}
        missing_groups = sorted(set(posterior_groups) - set(subclass_lookup))
        if missing_groups:
            raise ValueError(f"H5AD is missing posterior subclasses: {missing_groups}")
        posterior_to_h5 = np.asarray(
            [subclass_lookup[name] for name in posterior_groups], dtype=np.int64
        )
        cell_counts_h5 = np.bincount(
            subclass_codes[valid], minlength=len(subclass_categories)
        )

        class_of_h5 = np.full(len(subclass_categories), -1, dtype=np.int64)
        pairs = np.unique(
            np.column_stack([subclass_codes[valid], class_codes[valid]]), axis=0
        )
        for subclass_idx, class_idx in pairs:
            if class_of_h5[subclass_idx] not in {-1, class_idx}:
                raise ValueError("subclass does not map uniquely to class")
            class_of_h5[subclass_idx] = class_idx
        group_classes = class_categories[class_of_h5[posterior_to_h5]]

        t7_group = handle["obsm"]["T7CRE"]
        missing_cres = sorted(set(cre_names) - set(t7_group.keys()))
        if missing_cres:
            raise ValueError(f"T7 matrix is missing fitted cCREs: {missing_cres}")
        totals = np.empty((len(posterior_groups), len(cre_names)), dtype=np.float64)
        for cre_idx, cre in enumerate(cre_names):
            values = t7_group[cre][...].astype(np.float64, copy=False)
            grouped = np.bincount(
                subclass_codes[valid],
                weights=values[valid],
                minlength=len(subclass_categories),
            )
            totals[:, cre_idx] = grouped[posterior_to_h5]

    return totals, group_classes, cell_counts_h5[posterior_to_h5]


def balanced_target_score(
    target_draws: np.ndarray,
    control_draws: np.ndarray,
    effect_threshold: float,
) -> np.ndarray:
    """Compare each target with six controls per draw, omitting controls evenly."""
    n_draws = target_draws.shape[0]
    include = np.ones((n_draws, control_draws.shape[1]), dtype=bool)
    include[np.arange(n_draws), np.arange(n_draws) % control_draws.shape[1]] = False
    comparisons = (
        target_draws[:, :, None]
        > control_draws[:, None, :] + effect_threshold
    )
    return np.sum(comparisons & include[:, None, :], axis=(0, 2)) / float(
        include.sum()
    )


def loo_control_scores(
    control_draws: np.ndarray, effect_threshold: float
) -> np.ndarray:
    n_draws, n_controls = control_draws.shape
    scores = np.empty(n_controls, dtype=np.float64)
    for heldout in range(n_controls):
        keep = np.arange(n_controls) != heldout
        scores[heldout] = np.mean(
            control_draws[:, heldout, None]
            > control_draws[:, keep] + effect_threshold
        )
    return scores


def empirical_tail(scores: np.ndarray, null_scores: np.ndarray) -> np.ndarray:
    sorted_null = np.sort(np.asarray(null_scores, dtype=np.float64))
    counts = len(sorted_null) - np.searchsorted(sorted_null, scores, side="left")
    return (1.0 + counts) / (1.0 + len(sorted_null))


def assign_empirical_fdr(
    target: pd.DataFrame, null_scores: np.ndarray
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if len(target) == 0 or len(null_scores) == 0:
        raise ValueError("empirical FDR requires nonempty target and null scores")
    target = target.copy()
    target["empirical_p"] = empirical_tail(
        target["test_statistic"].to_numpy(float), null_scores
    )

    unique_scores = np.sort(target["test_statistic"].unique())[::-1]
    curve_rows = []
    n_target = len(target)
    n_null = len(null_scores)
    for threshold in unique_scores:
        discoveries = int(target["test_statistic"].ge(threshold).sum())
        null_exceedances = int(np.sum(null_scores >= threshold))
        null_tail = (1.0 + null_exceedances) / (1.0 + n_null)
        expected_false = n_target * null_tail
        raw_fdr = min(1.0, expected_false / max(discoveries, 1))
        curve_rows.append(
            {
                "test_statistic_threshold": float(threshold),
                "target_discoveries": discoveries,
                "null_exceedances": null_exceedances,
                "empirical_null_tail": float(null_tail),
                "expected_false_discoveries": float(expected_false),
                "raw_empirical_fdr": float(raw_fdr),
            }
        )
    curve = pd.DataFrame(curve_rows)
    curve["empirical_q"] = np.minimum.accumulate(
        curve["raw_empirical_fdr"].to_numpy(float)[::-1]
    )[::-1]
    score_to_q = curve.set_index("test_statistic_threshold")["empirical_q"]
    target["empirical_q"] = target["test_statistic"].map(score_to_q)
    return target, curve


def loo_calibration(
    loo: pd.DataFrame, alpha_levels: tuple[float, ...] = (0.01, 0.05, 0.10)
) -> tuple[pd.DataFrame, pd.DataFrame]:
    scores = loo["test_statistic"].to_numpy(float)
    leave_one_out_p = np.empty(len(loo), dtype=np.float64)
    for idx, score in enumerate(scores):
        calibration = np.delete(scores, idx)
        leave_one_out_p[idx] = empirical_tail(np.asarray([score]), calibration)[0]
    calibrated = loo.copy()
    calibrated["self_excluded_empirical_p"] = leave_one_out_p

    class_rows = []
    for heldout_class, test in calibrated.groupby("class", sort=True):
        train = calibrated.loc[calibrated["class"] != heldout_class]
        if len(train) == 0:
            continue
        p_values = empirical_tail(
            test["test_statistic"].to_numpy(float),
            train["test_statistic"].to_numpy(float),
        )
        row = {
            "heldout_class": heldout_class,
            "n_test": int(len(test)),
            "n_calibration": int(len(train)),
        }
        for alpha in alpha_levels:
            row[f"fpr_at_{alpha:g}"] = float(np.mean(p_values <= alpha))
            row[f"n_false_at_{alpha:g}"] = int(np.sum(p_values <= alpha))
        class_rows.append(row)
    return calibrated, pd.DataFrame(class_rows)


def compute_scores(
    log_gamma: np.ndarray,
    groups: np.ndarray,
    cre_names: np.ndarray,
    target_indices: np.ndarray,
    control_indices: np.ndarray,
    t7_totals: np.ndarray,
    group_classes: np.ndarray,
    group_cell_counts: np.ndarray,
    threshold: float,
    effect_threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    target_rows = []
    loo_rows = []
    control_names = cre_names[control_indices]
    for group_idx, group in enumerate(groups):
        group_draws = log_gamma[:, group_idx, :]
        control_draws = group_draws[:, control_indices]
        control_t7 = t7_totals[group_idx, control_indices]
        control_total = float(control_t7.sum())
        min_six_control_total = float(control_total - control_t7.max())

        target_scores = balanced_target_score(
            group_draws[:, target_indices], control_draws, effect_threshold
        )
        target_delta = (
            group_draws[:, target_indices] - control_draws.mean(axis=1)[:, None]
        )
        eligible_target = (
            (t7_totals[group_idx, target_indices] >= threshold)
            & (min_six_control_total >= threshold)
        )
        for local_idx in np.flatnonzero(eligible_target):
            draws = target_delta[:, local_idx]
            cre_idx = target_indices[local_idx]
            target_rows.append(
                {
                    "group": group,
                    "class": group_classes[group_idx],
                    "cre": cre_names[cre_idx],
                    "n_cells": int(group_cell_counts[group_idx]),
                    "target_t7_total": float(t7_totals[group_idx, cre_idx]),
                    "negative_control_t7_total": control_total,
                    "minimum_six_control_t7_total": min_six_control_total,
                    "test_statistic": float(target_scores[local_idx]),
                    "effect_vs_control_mean": float(draws.mean()),
                    "effect_vs_control_mean_lo90": float(np.quantile(draws, 0.05)),
                    "effect_vs_control_mean_hi90": float(np.quantile(draws, 0.95)),
                }
            )

        control_scores = loo_control_scores(control_draws, effect_threshold)
        for heldout in range(len(control_indices)):
            remaining_t7 = float(control_total - control_t7[heldout])
            eligible = control_t7[heldout] >= threshold and remaining_t7 >= threshold
            if not eligible:
                continue
            other = np.arange(len(control_indices)) != heldout
            draws = (
                control_draws[:, heldout]
                - control_draws[:, other].mean(axis=1)
            )
            loo_rows.append(
                {
                    "group": group,
                    "class": group_classes[group_idx],
                    "heldout_control": control_names[heldout],
                    "n_cells": int(group_cell_counts[group_idx]),
                    "heldout_t7_total": float(control_t7[heldout]),
                    "remaining_six_t7_total": remaining_t7,
                    "test_statistic": float(control_scores[heldout]),
                    "effect_vs_other_control_mean": float(draws.mean()),
                    "effect_vs_other_control_mean_lo90": float(np.quantile(draws, 0.05)),
                    "effect_vs_other_control_mean_hi90": float(np.quantile(draws, 0.95)),
                }
            )
    return pd.DataFrame(target_rows), pd.DataFrame(loo_rows)


def plot_diagnostics(
    target: pd.DataFrame,
    loo: pd.DataFrame,
    curve: pd.DataFrame,
    class_calibration: pd.DataFrame,
    fdr_level: float,
    output: Path,
) -> None:
    sns.set_theme(context="paper", style="whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.2), constrained_layout=True)

    ax = axes[0, 0]
    bins = np.linspace(0, 1, 31)
    ax.hist(
        target["test_statistic"], bins=bins, density=True, alpha=0.42,
        color="#4477AA", label=f"Targets (n={len(target):,})",
    )
    ax.hist(
        loo["test_statistic"], bins=bins, density=True, alpha=0.62,
        color="#CC6677", label=f"LOO controls (n={len(loo):,})",
    )
    significant = target["empirical_q"].le(fdr_level)
    if significant.any():
        cutoff = float(target.loc[significant, "test_statistic"].min())
        ax.axvline(cutoff, color="black", linestyle="--", linewidth=1.0,
                   label=f"q <= {fdr_level:g} cutoff")
    ax.set_xlabel("Posterior superiority statistic")
    ax.set_ylabel("Density")
    ax.set_title("Target and leave-one-out null scores")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[0, 1]
    ax.plot(
        curve["test_statistic_threshold"], curve["raw_empirical_fdr"],
        color="0.65", linewidth=1.0, label="Raw empirical FDR",
    )
    ax.plot(
        curve["test_statistic_threshold"], curve["empirical_q"],
        color="#2F6F8F", linewidth=1.4, label="Monotonic q-value",
    )
    ax.axhline(fdr_level, color="#AA3344", linestyle="--", linewidth=1.0)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Posterior superiority threshold")
    ax.set_ylabel("Estimated FDR")
    ax.set_title("Empirical FDR curve")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1, 0]
    rho = spearmanr(loo["heldout_t7_total"], loo["test_statistic"])
    ax.scatter(
        np.log10(loo["heldout_t7_total"] + 1.0), loo["test_statistic"],
        s=26, color="#CC6677", alpha=0.7, linewidths=0,
    )
    ax.set_xlabel("Held-out control T7, log10(count + 1)")
    ax.set_ylabel("LOO null statistic")
    ax.set_title(
        f"LOO statistic versus T7\nSpearman rho={rho.statistic:.3f}, p={rho.pvalue:.3g}"
    )

    ax = axes[1, 1]
    if len(class_calibration):
        shown = class_calibration.sort_values("fpr_at_0.05", ascending=False)
        x = np.arange(len(shown))
        ax.bar(x, shown["fpr_at_0.05"], color="#777777", width=0.75)
        ax.axhline(0.05, color="#AA3344", linestyle="--", linewidth=1.0)
        ax.set_xticks(x)
        ax.set_xticklabels(shown["heldout_class"], rotation=90, fontsize=6)
        ax.set_ylabel("Held-out LOO FPR at p <= 0.05")
        ax.set_title("Cell-class-held-out null calibration")
    else:
        ax.text(0.5, 0.5, "No class-held-out calibration available",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()

    fig.suptitle(
        "Individual negative-control LOO empirical-FDR test",
        fontsize=12,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), bbox_inches="tight", dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.figures_dir.mkdir(parents=True, exist_ok=True)
    args.tables_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((args.bayes_dir / "run_manifest.json").read_text())
    posterior_path = args.bayes_dir / f"{manifest['tag']}_posterior_samples.npz"
    negative_controls = pd.read_csv(args.bayes_dir / "negative_controls.csv").iloc[
        :, 0
    ].astype(str).tolist()
    blacklist = read_cre_blacklist(args.bayes_dir)

    with np.load(posterior_path, allow_pickle=True) as posterior:
        groups = posterior["group_names"].astype(str)
        all_cre_names = posterior["cre_names"].astype(str)
        ordinary_mask = all_cre_names != POOLED_NAME
        cre_names = all_cre_names[ordinary_mask]
        log_gamma = posterior["log_gamma"][:, :, ordinary_mask].astype(np.float32)

    control_indices = np.flatnonzero(np.isin(cre_names, negative_controls))
    if len(control_indices) != 7:
        raise ValueError(f"Expected seven ordinary negative controls; found {len(control_indices)}")
    target_indices = np.flatnonzero(
        ~np.isin(cre_names, negative_controls) & ~np.isin(cre_names, list(blacklist))
    )
    t7_totals, group_classes, group_cell_counts = load_grouped_t7(
        args.h5ad, groups, cre_names
    )
    target, loo = compute_scores(
        log_gamma,
        groups,
        cre_names,
        target_indices,
        control_indices,
        t7_totals,
        group_classes,
        group_cell_counts,
        args.t7_threshold,
        args.effect_threshold,
    )
    if len(loo) == 0:
        raise ValueError("No eligible leave-one-out negative-control tests")
    target, curve = assign_empirical_fdr(
        target, loo["test_statistic"].to_numpy(float)
    )
    target["significant_empirical_fdr"] = target["empirical_q"].le(args.fdr)
    loo, class_calibration = loo_calibration(loo)

    prefix = args.tables_dir / args.stem
    target_path = Path(f"{prefix}_target_tests.csv")
    loo_path = Path(f"{prefix}_loo_null.csv")
    curve_path = Path(f"{prefix}_fdr_curve.csv")
    class_path = Path(f"{prefix}_class_heldout_calibration.csv")
    target.to_csv(target_path, index=False)
    loo.to_csv(loo_path, index=False)
    curve.to_csv(curve_path, index=False)
    class_calibration.to_csv(class_path, index=False)

    figure_path = args.figures_dir / f"{args.stem}.pdf"
    plot_diagnostics(target, loo, curve, class_calibration, args.fdr, figure_path)
    significant = target.loc[target["significant_empirical_fdr"]]
    score_t7 = spearmanr(loo["heldout_t7_total"], loo["test_statistic"])
    calibration_summary = {}
    for alpha in (0.01, 0.05, 0.10):
        column = f"fpr_at_{alpha:g}"
        weighted = (
            np.average(class_calibration[column], weights=class_calibration["n_test"])
            if len(class_calibration)
            else np.nan
        )
        calibration_summary[f"class_heldout_weighted_fpr_at_{alpha:g}"] = float(weighted)

    result_manifest = {
        "model": "Joint+dropout ordinary negative-control activities",
        "bayes_dir": str(args.bayes_dir),
        "posterior": str(posterior_path),
        "pooled_negative_control_used": False,
        "negative_controls": negative_controls,
        "t7_threshold": args.t7_threshold,
        "effect_threshold_log_activity": args.effect_threshold,
        "target_reference": (
            "six of seven controls per posterior draw; omitted control cycles evenly"
        ),
        "loo_reference": "held-out control compared with the other six",
        "empirical_null": "all eligible LOO control-cell-type test statistics",
        "empirical_fdr": {
            "target_null_fraction": 1.0,
            "plus_one_tail_correction": True,
            "target_level": args.fdr,
        },
        "counts": {
            "eligible_target_tests": int(len(target)),
            "eligible_loo_null_tests": int(len(loo)),
            "significant_target_tests": int(len(significant)),
            "significant_ccres": int(significant["cre"].nunique()),
            "significant_cell_types": int(significant["group"].nunique()),
        },
        "loo_statistic_vs_t7": {
            "spearman_rho": float(score_t7.statistic),
            "p": float(score_t7.pvalue),
        },
        "calibration": calibration_summary,
        "outputs": {
            "target_tests": str(target_path),
            "loo_null": str(loo_path),
            "fdr_curve": str(curve_path),
            "class_heldout_calibration": str(class_path),
            "figure_pdf": str(figure_path),
            "figure_png": str(figure_path.with_suffix('.png')),
        },
    }
    manifest_path = args.figures_dir / f"{args.stem}_manifest.json"
    result_manifest["outputs"]["manifest"] = str(manifest_path)
    write_json(manifest_path, result_manifest)
    print(json.dumps(result_manifest, indent=2))


if __name__ == "__main__":
    main()
