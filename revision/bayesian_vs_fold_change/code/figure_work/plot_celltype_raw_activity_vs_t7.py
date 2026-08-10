#!/usr/bin/env python3
"""Plot raw posterior activity against cell-type-specific total T7 counts."""

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
from scipy.stats import pearsonr, spearmanr

# The shared analysis layer (analysis_utils and the plot_* modules that other
# scripts import) stays in the parent code/ directory.
import sys as _sys
from pathlib import Path as _Path
_CODE_DIR = _Path(__file__).resolve().parent.parent
if str(_CODE_DIR) not in _sys.path:
    _sys.path.insert(0, str(_CODE_DIR))

from analysis_utils import (
    ANALYSIS_DIR,
    DEFAULT_H5AD,
    FIGURES_WORK,
    OLD_DATA_BAYES,
    write_json,
)
from plot_method_activity_correlation import read_cre_blacklist
from test_individual_negative_control_loo_empirical_fdr import assign_empirical_fdr


POOLED_NAME = "NEGATIVE_CONTROL_POOL"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bayes-dir",
        type=Path,
        default=OLD_DATA_BAYES,
    )
    parser.add_argument("--group", default="OB Eomes Ms4a15 Glut")
    parser.add_argument("--h5ad", type=Path, default=DEFAULT_H5AD)
    parser.add_argument("--t7-threshold", type=float, default=50.0)
    parser.add_argument("--q-cutoff", type=float, default=0.05)
    parser.add_argument(
        "--loo-evaluated-tests",
        type=Path,
        default=ANALYSIS_DIR
        / "results"
        / "tables"
        / "method_activity_t7_filter_evaluated_tests.csv.gz",
    )
    parser.add_argument(
        "--loo-null",
        type=Path,
        default=ANALYSIS_DIR
        / "results"
        / "tables"
        / (
            "joint_dropout_direct_activity_individual_negative_control_"
            "loo_empirical_fdr_loo_null.csv"
        ),
    )
    parser.add_argument(
        "--figures-dir", type=Path, default=FIGURES_WORK
    )
    parser.add_argument(
        "--tables-dir", type=Path, default=ANALYSIS_DIR / "results" / "tables"
    )
    parser.add_argument("--stem", default=None)
    return parser.parse_args()


def safe_token(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_").lower()


def posterior_interval(draws: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        draws.mean(axis=0, dtype=np.float64),
        np.quantile(draws, 0.05, axis=0),
        np.quantile(draws, 0.95, axis=0),
    )


def decode_strings(values: np.ndarray) -> np.ndarray:
    return np.asarray(
        [value.decode() if isinstance(value, bytes) else str(value) for value in values]
    )


def celltype_t7_totals(
    h5ad: Path, group: str, cre_names: np.ndarray
) -> tuple[np.ndarray, int]:
    with h5py.File(h5ad, "r") as handle:
        subclass = handle["obs"]["subclass_name"]
        categories = decode_strings(subclass["categories"][...])
        normalized = np.asarray(
            [re.sub(r"^\d+\s+", "", value).replace("/", "-") for value in categories]
        )
        matches = np.flatnonzero(normalized == group)
        if len(matches) != 1:
            raise ValueError(f"Expected one exact subclass {group!r}; found {len(matches)}")
        cell_mask = subclass["codes"][...] == int(matches[0])
        t7_group = handle["obsm"]["T7CRE"]
        missing = sorted(set(cre_names) - set(t7_group.keys()))
        if missing:
            raise ValueError(f"T7 matrix is missing fitted cCREs: {missing}")
        totals = np.asarray(
            [float(t7_group[name][...][cell_mask].sum()) for name in cre_names],
            dtype=float,
        )
    return totals, int(cell_mask.sum())


def correlation_summary(frame: pd.DataFrame) -> dict:
    if len(frame) < 3:
        return {"n": int(len(frame)), "pearson_r": None, "pearson_p": None,
                "spearman_rho": None, "spearman_p": None}
    x = frame["log10_t7_plus1"].to_numpy(float)
    y = frame["raw_activity_mean"].to_numpy(float)
    pearson = pearsonr(x, y)
    spearman = spearmanr(x, y)
    return {
        "n": int(len(frame)),
        "pearson_r": float(pearson.statistic),
        "pearson_p": float(pearson.pvalue),
        "spearman_rho": float(spearman.statistic),
        "spearman_p": float(spearman.pvalue),
    }


def load_data(
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, dict, pd.DataFrame, pd.DataFrame]:
    manifest = json.loads((args.bayes_dir / "run_manifest.json").read_text())
    posterior_path = args.bayes_dir / f"{manifest['tag']}_posterior_samples.npz"
    negative_controls = pd.read_csv(args.bayes_dir / "negative_controls.csv").iloc[
        :, 0
    ].astype(str).tolist()
    blacklist = read_cre_blacklist(args.bayes_dir)

    with np.load(posterior_path, allow_pickle=True) as posterior:
        required = {"log_gamma", "log_gamma_neg", "group_names", "cre_names"}
        missing = required.difference(posterior.files)
        if missing:
            raise ValueError(f"Posterior is missing required sites: {sorted(missing)}")
        groups = posterior["group_names"].astype(str)
        cre_names_all = posterior["cre_names"].astype(str)
        matches = np.flatnonzero(groups == args.group)
        if len(matches) != 1:
            raise ValueError(f"Expected one posterior group {args.group!r}; found {len(matches)}")
        group_idx = int(matches[0])
        ordinary_mask = cre_names_all != POOLED_NAME
        cre_names = cre_names_all[ordinary_mask]
        activity_draws = posterior["log_gamma"][:, group_idx, ordinary_mask].astype(
            np.float64
        )
        pooled_draws = posterior["log_gamma_neg"][:, group_idx].astype(np.float64)

    t7_totals, n_cells = celltype_t7_totals(args.h5ad, args.group, cre_names)
    activity_mean, activity_lo, activity_hi = posterior_interval(activity_draws)
    pooled_mean, pooled_lo, pooled_hi = posterior_interval(pooled_draws[:, None])
    frame = pd.DataFrame(
        {
            "group": args.group,
            "cre": cre_names,
            "is_negative_control": np.isin(cre_names, negative_controls),
            "is_blacklisted": np.isin(cre_names, list(blacklist)),
            "t7_total": t7_totals,
            "log10_t7_plus1": np.log10(t7_totals + 1.0),
            "raw_activity_mean": activity_mean,
            "raw_activity_lo90": activity_lo,
            "raw_activity_hi90": activity_hi,
        }
    )
    frame["passes_t7_filter"] = (
        frame["t7_total"].ge(args.t7_threshold)
        & ~frame["is_negative_control"]
        & ~frame["is_blacklisted"]
    )
    frame["is_target"] = ~frame["is_negative_control"] & ~frame["is_blacklisted"]

    loo_tests = pd.read_csv(args.loo_evaluated_tests)
    loo_tests = loo_tests.loc[
        loo_tests["method"].eq("Joint+dropout LOO")
        & np.isclose(
            loo_tests["t7_threshold"].to_numpy(float), args.t7_threshold
        )
    ].copy()
    if loo_tests.empty:
        raise ValueError(
            f"No Joint+dropout LOO tests found at T7 >= {args.t7_threshold:g}"
        )
    if loo_tests.duplicated(["group", "cre"]).any():
        raise ValueError("Joint+dropout LOO tests contain duplicate group-cCRE pairs")
    loo_null = pd.read_csv(args.loo_null)
    recomputed, loo_curve = assign_empirical_fdr(
        loo_tests, loo_null["test_statistic"].to_numpy(float)
    )
    q_recalculation_error = float(
        np.max(
            np.abs(
                recomputed["empirical_q"].to_numpy(float)
                - loo_tests["q_right"].to_numpy(float)
            )
        )
    )
    group_tests = loo_tests.loc[loo_tests["group"].eq(args.group), [
        "group", "cre", "test_statistic", "p_right", "q_right"
    ]].rename(
        columns={
            "test_statistic": "loo_test_statistic",
            "p_right": "loo_empirical_p",
            "q_right": "loo_empirical_q",
        }
    )
    frame = frame.merge(group_tests, on=["group", "cre"], how="left", validate="one_to_one")
    frame["loo_tested"] = frame["loo_test_statistic"].notna()
    frame["loo_significant"] = frame["loo_empirical_q"].le(args.q_cutoff)

    control_t7 = float(
        frame.loc[frame["is_negative_control"], "t7_total"].sum()
    )
    target = frame.loc[frame["is_target"]]
    eligible = frame.loc[frame["passes_t7_filter"]]
    best = loo_curve.loc[loo_curve["raw_empirical_fdr"].idxmin()]
    max_null_score = float(loo_null["test_statistic"].max())
    metadata = {
        "model": "Joint+dropout ordinary-and-pooled negative controls",
        "bayes_dir": str(args.bayes_dir),
        "posterior": str(posterior_path),
        "h5ad": str(args.h5ad),
        "group": args.group,
        "n_cells": n_cells,
        "t7_threshold": args.t7_threshold,
        "activity_definition": "raw posterior log_gamma; alpha is not subtracted",
        "x_definition": "total T7CRE count in the selected subclass",
        "negative_controls": negative_controls,
        "pooled_negative_control": {
            "t7_total": control_t7,
            "raw_activity_mean": float(pooled_mean[0]),
            "raw_activity_lo90": float(pooled_lo[0]),
            "raw_activity_hi90": float(pooled_hi[0]),
            "included_in_correlations": False,
        },
        "correlations": {
            "all_nonblacklisted_targets": correlation_summary(target),
            "targets_t7_ge_threshold": correlation_summary(eligible),
            "ordinary_negative_controls": correlation_summary(
                frame.loc[frame["is_negative_control"]]
            ),
        },
        "loo_test": {
            "evaluated_tests": str(args.loo_evaluated_tests),
            "null_scores": str(args.loo_null),
            "q_cutoff": args.q_cutoff,
            "n_global_tests": int(len(loo_tests)),
            "n_group_tests": int(len(group_tests)),
            "n_null_scores": int(len(loo_null)),
            "n_global_significant": int(loo_tests["q_right"].le(args.q_cutoff).sum()),
            "n_group_significant": int(group_tests["loo_empirical_q"].le(args.q_cutoff).sum()),
            "minimum_empirical_tail": float(1.0 / (len(loo_null) + 1.0)),
            "maximum_null_score": max_null_score,
            "targets_above_maximum_null": int(
                loo_tests["test_statistic"].gt(max_null_score).sum()
            ),
            "best_test_statistic_threshold": float(best["test_statistic_threshold"]),
            "discoveries_at_best_threshold": int(best["target_discoveries"]),
            "expected_false_at_best_threshold": float(best["expected_false_discoveries"]),
            "minimum_empirical_q": float(best["empirical_q"]),
            "q_recalculation_max_absolute_error": q_recalculation_error,
        },
    }
    return frame, metadata, loo_curve, loo_null


def count_ticks(maximum: float) -> list[int]:
    candidates = [0, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000]
    ticks = [value for value in candidates if value <= maximum]
    if not ticks or ticks[-1] < maximum * 0.55:
        ticks.append(int(np.ceil(maximum)))
    return ticks


def plot(
    frame: pd.DataFrame,
    metadata: dict,
    loo_curve: pd.DataFrame,
    loo_null: pd.DataFrame,
    output: Path,
) -> None:
    sns.set_theme(context="paper", style="whitegrid")
    target = frame.loc[frame["is_target"]].copy()
    low = target.loc[~target["passes_t7_filter"]]
    eligible = target.loc[target["passes_t7_filter"]]
    controls = frame.loc[frame["is_negative_control"]].copy()
    pooled = metadata["pooled_negative_control"]
    eligible_stats = metadata["correlations"]["targets_t7_ge_threshold"]
    loo_stats = metadata["loo_test"]
    tested = eligible.loc[eligible["loo_tested"]].copy()
    untested = eligible.loc[~eligible["loo_tested"]].copy()
    significant = tested.loc[tested["loo_significant"]]

    fig, (ax, fdr_ax) = plt.subplots(
        1,
        2,
        figsize=(13.4, 6.3),
        gridspec_kw={"width_ratios": [1.55, 1.0]},
        constrained_layout=True,
    )
    threshold_x = np.log10(metadata["t7_threshold"] + 1.0)
    ax.axvspan(
        -0.03,
        threshold_x,
        color="0.92",
        alpha=0.8,
        linewidth=0,
        label=f"T7 < {metadata['t7_threshold']:g}",
    )
    ax.axvline(threshold_x, color="0.35", linestyle="--", linewidth=0.9)
    ax.scatter(
        low["log10_t7_plus1"],
        low["raw_activity_mean"],
        s=18,
        color="0.60",
        alpha=0.38,
        linewidths=0,
        rasterized=True,
    )
    if len(untested):
        ax.scatter(
            untested["log10_t7_plus1"],
            untested["raw_activity_mean"],
            s=22,
            color="#4477AA",
            alpha=0.38,
            linewidths=0,
            rasterized=True,
            label=f"T7-passing targets outside common test set (n={len(untested)})",
        )
    score_points = ax.scatter(
        tested["log10_t7_plus1"],
        tested["raw_activity_mean"],
        c=tested["loo_test_statistic"],
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
        s=27,
        alpha=0.74,
        linewidths=0,
        rasterized=True,
        label=f"LOO-tested targets (n={len(tested)})",
    )
    cbar = fig.colorbar(score_points, ax=ax, pad=0.015, fraction=0.045)
    cbar.set_label("LOO posterior comparison score")
    high_score = tested.loc[tested["loo_test_statistic"].ge(0.99)]
    ax.scatter(
        high_score["log10_t7_plus1"],
        high_score["raw_activity_mean"],
        s=54,
        facecolors="none",
        edgecolors="#222222",
        linewidths=0.75,
        zorder=3,
        label=f"LOO score >= 0.99 (n={len(high_score)})",
    )
    ax.scatter(
        significant["log10_t7_plus1"],
        significant["raw_activity_mean"],
        marker="X",
        s=75,
        color="#D62728",
        edgecolors="white",
        linewidths=0.6,
        zorder=6,
        label=f"Empirical q <= {loo_stats['q_cutoff']:g} (n={len(significant)})",
    )

    palette = dict(
        zip(sorted(controls["cre"]), sns.color_palette("tab10", len(controls)))
    )
    label_offsets = {
        "CRE328": (7, 17),
        "CRE330": (7, 12),
        "CRE331": (7, 8),
        "CRE332": (8, -18),
        "CRE333": (5, -20),
        "CRE336": (6, 14),
        "CRE337": (7, -18),
    }
    for row in controls.itertuples(index=False):
        ax.errorbar(
            row.log10_t7_plus1,
            row.raw_activity_mean,
            yerr=np.asarray(
                [[
                    row.raw_activity_mean - row.raw_activity_lo90
                ], [
                    row.raw_activity_hi90 - row.raw_activity_mean
                ]]
            ),
            fmt="^",
            color=palette[row.cre],
            capsize=2,
            markersize=6,
            zorder=4,
        )
        dx, dy = label_offsets.get(row.cre, (5, 3))
        ax.annotate(
            row.cre,
            (row.log10_t7_plus1, row.raw_activity_mean),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=7,
        )

    pooled_x = np.log10(float(pooled["t7_total"]) + 1.0)
    ax.errorbar(
        pooled_x,
        float(pooled["raw_activity_mean"]),
        yerr=np.asarray(
            [[
                float(pooled["raw_activity_mean"] - pooled["raw_activity_lo90"])
            ], [
                float(pooled["raw_activity_hi90"] - pooled["raw_activity_mean"])
            ]]
        ),
        fmt="*",
        color="black",
        capsize=2.5,
        markersize=12,
        zorder=5,
        label="Pooled negative-control activity",
    )
    ax.annotate(
        "Pooled controls",
        (pooled_x, float(pooled["raw_activity_mean"])),
        xytext=(6, 5),
        textcoords="offset points",
        fontsize=7,
        fontweight="bold",
    )
    ax.scatter([], [], marker="^", s=34, color="black", label="Individual controls")

    max_count = max(float(frame["t7_total"].max()), float(pooled["t7_total"]))
    ticks = count_ticks(max_count)
    ax.set_xticks(np.log10(np.asarray(ticks, dtype=float) + 1.0))
    ax.set_xticklabels([str(value) for value in ticks])
    ax.set_xlim(-0.03, np.log10(max_count + 1.0) + 0.08)
    ax.set_xlabel(f"Total T7 count in {metadata['group']}")
    ax.set_ylabel(r"Raw posterior activity, $\log\gamma_{s,j}$")
    ax.set_title(
        "Raw activity and individual-control LOO score\n"
        f"T7 >= {metadata['t7_threshold']:g} targets: "
        f"Pearson r={eligible_stats['pearson_r']:.3f}, "
        f"Spearman rho={eligible_stats['spearman_rho']:.3f}"
    )
    ax.legend(frameon=False, fontsize=7.5, loc="lower left")
    sns.despine(ax=ax)

    curve = loo_curve.sort_values("test_statistic_threshold")
    fdr_ax.step(
        curve["test_statistic_threshold"],
        curve["empirical_q"],
        where="post",
        color="#3B5B92",
        linewidth=1.7,
        label="Global empirical q-value curve",
    )
    fdr_ax.scatter(
        tested["loo_test_statistic"],
        tested["loo_empirical_q"],
        c=tested["loo_test_statistic"],
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
        s=13,
        alpha=0.42,
        linewidths=0,
        rasterized=True,
        label=f"{metadata['group']} targets",
    )
    fdr_ax.scatter(
        loo_null["test_statistic"],
        np.full(len(loo_null), 0.985),
        marker="|",
        s=55,
        color="#D55E00",
        linewidths=0.8,
        label=f"LOO null scores (n={len(loo_null)})",
    )
    fdr_ax.axhline(
        loo_stats["q_cutoff"],
        color="#D62728",
        linestyle="--",
        linewidth=1.0,
        label=f"q = {loo_stats['q_cutoff']:g}",
    )
    fdr_ax.axhline(
        loo_stats["minimum_empirical_q"],
        color="0.35",
        linestyle=":",
        linewidth=0.9,
    )
    fdr_ax.text(
        0.035,
        0.11,
        f"Smallest null tail = 1/{loo_stats['n_null_scores'] + 1} "
        f"= {loo_stats['minimum_empirical_tail']:.3f}\n"
        f"Best threshold = {loo_stats['best_test_statistic_threshold']:.3f}\n"
        f"Discoveries = {loo_stats['discoveries_at_best_threshold']:,}\n"
        f"Expected false = {loo_stats['expected_false_at_best_threshold']:.1f}\n"
        f"Minimum q = {loo_stats['minimum_empirical_q']:.3f}",
        transform=fdr_ax.transAxes,
        fontsize=8.5,
        va="bottom",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.9,
              "edgecolor": "0.75"},
    )
    fdr_ax.set_xlim(-0.02, 1.02)
    fdr_ax.set_ylim(0.0, 1.02)
    fdr_ax.set_xlabel("LOO posterior comparison score threshold")
    fdr_ax.set_ylabel("Empirical q-value")
    fdr_ax.set_title(
        "Why no calls pass empirical FDR\n"
        f"{loo_stats['n_global_tests']:,} common-pair tests, "
        f"{loo_stats['n_null_scores']} eligible null scores"
    )
    fdr_ax.legend(frameon=False, fontsize=7.5, loc="upper left")
    sns.despine(ax=fdr_ax)
    fig.suptitle(
        f"{metadata['group']}: raw joint+dropout activity and the new LOO test\n"
        "Activity is raw posterior log-gamma; alpha is not subtracted",
        fontsize=12,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), bbox_inches="tight", dpi=220)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.figures_dir.mkdir(parents=True, exist_ok=True)
    args.tables_dir.mkdir(parents=True, exist_ok=True)
    stem = args.stem or f"joint_dropout_raw_activity_vs_t7_{safe_token(args.group)}"
    frame, metadata, loo_curve, loo_null = load_data(args)
    table_path = args.tables_dir / f"{stem}.csv"
    curve_path = args.tables_dir / f"{stem}_loo_fdr_curve.csv"
    figure_path = args.figures_dir / f"{stem}.pdf"
    manifest_path = args.figures_dir / f"{stem}_manifest.json"
    frame.to_csv(table_path, index=False)
    loo_curve.to_csv(curve_path, index=False)
    plot(frame, metadata, loo_curve, loo_null, figure_path)
    write_json(
        manifest_path,
        {
            **metadata,
            "outputs": {
                "pdf": str(figure_path),
                "png": str(figure_path.with_suffix('.png')),
                "table": str(table_path),
                "loo_fdr_curve": str(curve_path),
                "manifest": str(manifest_path),
            },
        },
    )
    print(json.dumps(metadata["correlations"], indent=2))


if __name__ == "__main__":
    main()
