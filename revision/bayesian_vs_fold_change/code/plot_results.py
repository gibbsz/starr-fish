#!/usr/bin/env python3
"""Compare bootstrap and Bayesian activity estimates and generate final plots."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import spearmanr

from analysis_utils import (
    ANALYSIS_DIR,
    FIGURES_WORK,
    OLD_DATA_BAYES,
    OLD_DATA_BOOTSTRAP,
    log,
    write_json,
)
from baystarrfish.stats import bh_fdr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bootstrap-dir", type=Path, default=OLD_DATA_BOOTSTRAP
    )
    parser.add_argument(
        "--bayes-dir", type=Path, default=OLD_DATA_BAYES
    )
    parser.add_argument(
        "--figures-dir", type=Path, default=FIGURES_WORK
    )
    parser.add_argument(
        "--tables-dir", type=Path, default=ANALYSIS_DIR / "results" / "tables"
    )
    parser.add_argument("--bayes-tag", default=None)
    parser.add_argument("--q-cutoff", type=float, default=0.05)
    parser.add_argument("--min-cells-heatmap", type=int, default=1_000)
    parser.add_argument("--max-cres-heatmap", type=int, default=80)
    return parser.parse_args()


def save_figure(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), bbox_inches="tight", dpi=200)
    plt.close(fig)


def discover_bayes_tag(bayes_dir: Path, requested: str | None) -> str:
    if requested:
        return requested
    manifest = json.loads((bayes_dir / "run_manifest.json").read_text())
    return str(manifest["tag"])




def bayesian_significance(
    gamma: pd.DataFrame,
    posterior_path: Path,
    negative_controls: set[str],
    min_detected_cells: int,
    *,
    filter_negative_controls: bool = True,
    filter_prior_dominated: bool = True,
) -> pd.DataFrame:
    """Match the bootstrap self-cCRE/negative-control right-tail test."""
    with np.load(posterior_path, allow_pickle=True) as posterior:
        log_gamma = posterior["log_gamma"].astype(np.float32)
        groups = posterior["group_names"].astype(str)
        cres = posterior["cre_names"].astype(str)

    n_draws, n_groups, n_cres = log_gamma.shape
    self_cre_mean = log_gamma.mean(axis=(0, 1), dtype=np.float64).astype(np.float32)
    calibrated = log_gamma - self_cre_mean[None, None, :]
    negative_mask = np.isin(cres, list(negative_controls))
    if not negative_mask.any():
        raise ValueError("Bayesian posterior contains no negative-control cCREs")
    negative_threshold = calibrated[:, :, negative_mask].mean(axis=(0, 2))
    p_right = (
        calibrated <= negative_threshold[None, :, None]
    ).mean(axis=0, dtype=np.float64)
    effect = calibrated.mean(axis=0, dtype=np.float64) - negative_threshold[:, None]

    group_grid, cre_grid = np.meshgrid(groups, cres, indexing="ij")
    output = pd.DataFrame(
        {
            "group": group_grid.ravel(),
            "cre": cre_grid.ravel(),
            "bayesian_effect_log": effect.ravel(),
            "bayesian_p_right": p_right.ravel(),
        }
    )
    evidence_columns = [
        name
        for name in ("group", "cre", "n_cre_pos", "n_double_pos", "prior_dominated")
        if name in gamma.columns
    ]
    output = output.merge(
        gamma[evidence_columns], on=["group", "cre"], how="left"
    )
    output["is_negative_control"] = output["cre"].isin(negative_controls)
    output["n_cre_pos"] = output["n_cre_pos"].fillna(0).astype(int)
    output["prior_dominated"] = (
        output["prior_dominated"].fillna(True).astype(bool)
    )
    invalid = (
        output["n_cre_pos"].lt(min_detected_cells)
        | ~np.isfinite(output["bayesian_effect_log"])
        | ~np.isfinite(output["bayesian_p_right"])
    )
    if filter_negative_controls:
        invalid = invalid | output["is_negative_control"]
    if filter_prior_dominated:
        invalid = invalid | output["prior_dominated"]
    output.loc[invalid, "bayesian_p_right"] = np.nan
    output["bayesian_q_right"] = bh_fdr(output["bayesian_p_right"].to_numpy())
    output["n_posterior_draws"] = n_draws
    return output


def wide_to_long(frame: pd.DataFrame, value_name: str) -> pd.DataFrame:
    frame.index = frame.index.astype(str)
    frame.columns = frame.columns.astype(str)
    return (
        frame.rename_axis(index="group", columns="cre")
        .stack(future_stack=True)
        .rename(value_name)
        .reset_index()
    )


def read_blacklist(run_dir: Path) -> set[str]:
    path = run_dir / "cre_blacklist.csv"
    if not path.exists():
        return set()
    return set(pd.read_csv(path).iloc[:, 0].astype(str))


def candidate_pair_mask(gamma: pd.DataFrame, blacklist: set[str]) -> pd.DataFrame:
    """Use only cCRE blacklist and Bayesian prior-dominated filters."""
    candidate = (
        ~gamma.assign(
            group=gamma["group"].astype(str),
            cre=gamma["cre"].astype(str),
            prior_dominated=gamma["prior_dominated"].astype(bool),
        )
        .pivot(index="group", columns="cre", values="prior_dominated")
        .astype(bool)
    )
    candidate.index = candidate.index.astype(str)
    candidate.columns = candidate.columns.astype(str)
    candidate.loc[:, candidate.columns.intersection(blacklist)] = False
    return candidate


def require_finite_mask(
    candidate: pd.DataFrame, frame: pd.DataFrame
) -> pd.DataFrame:
    finite = frame.copy()
    finite.index = finite.index.astype(str)
    finite.columns = finite.columns.astype(str)
    finite = finite.reindex(index=candidate.index, columns=candidate.columns)
    return candidate & finite.notna()


def apply_candidate_mask(
    frame: pd.DataFrame, candidate: pd.DataFrame
) -> pd.DataFrame:
    frame = frame.copy()
    frame.index = frame.index.astype(str)
    frame.columns = frame.columns.astype(str)
    frame = frame.reindex(index=candidate.index, columns=candidate.columns)
    return frame.where(candidate)


def correlation(x: pd.Series, y: pd.Series) -> tuple[float, int]:
    valid = np.isfinite(x.to_numpy(dtype=float)) & np.isfinite(
        y.to_numpy(dtype=float)
    )
    if valid.sum() < 3:
        return np.nan, int(valid.sum())
    return float(spearmanr(x.to_numpy()[valid], y.to_numpy()[valid]).statistic), int(
        valid.sum()
    )


def build_comparison(args: argparse.Namespace) -> tuple[pd.DataFrame, dict]:
    tag = discover_bayes_tag(args.bayes_dir, args.bayes_tag)
    gamma_path = args.bayes_dir / f"{tag}_gamma.csv"
    posterior_path = args.bayes_dir / f"{tag}_posterior_samples.npz"
    gamma = pd.read_csv(gamma_path)
    candidate = candidate_pair_mask(gamma, read_blacklist(args.bayes_dir))
    boot_effect_path = args.bootstrap_dir / "log_activity_prior_mask_vs_negative_control.csv"
    if not boot_effect_path.exists():
        boot_effect_path = args.bootstrap_dir / "log_activity_vs_negative_control.csv"
    boot_q_path = args.bootstrap_dir / "qvalues_prior_mask_right.csv"
    if not boot_q_path.exists():
        boot_q_path = args.bootstrap_dir / "qvalues_right.csv"
    boot_effect = pd.read_csv(
        boot_effect_path, index_col=0
    )
    boot_q = pd.read_csv(boot_q_path, index_col=0)
    candidate = require_finite_mask(candidate, boot_q)
    boot_effect = apply_candidate_mask(boot_effect, candidate)
    boot_q = apply_candidate_mask(boot_q, candidate)
    boot = wide_to_long(boot_effect, "bootstrap_effect_log").merge(
        wide_to_long(boot_q, "bootstrap_q_right"),
        on=["group", "cre"],
        how="outer",
    )

    negative_controls = set(
        pd.read_csv(args.bayes_dir / "negative_controls.csv")["cre"].astype(str)
    )
    bayes = bayesian_significance(
        gamma,
        posterior_path,
        negative_controls,
        0,
        filter_negative_controls=False,
        filter_prior_dominated=False,
    )
    comparison = boot.merge(bayes, on=["group", "cre"], how="inner")
    candidate_long = wide_to_long(candidate.astype(int), "candidate_pair")
    comparison = comparison.merge(
        candidate_long, on=["group", "cre"], how="left"
    )
    comparison["candidate_pair"] = (
        comparison["candidate_pair"].fillna(0).astype(bool)
    )
    comparison.loc[
        ~comparison["candidate_pair"],
        [
            "bootstrap_effect_log",
            "bootstrap_q_right",
            "bayesian_effect_log",
            "bayesian_p_right",
            "bayesian_q_right",
        ],
    ] = np.nan

    cell_counts = pd.read_csv(
        args.bootstrap_dir / "subclass_cell_counts.csv", index_col=0
    ).iloc[:, 0]
    cell_counts.index = cell_counts.index.astype(str)
    comparison["n_cells"] = comparison["group"].map(cell_counts)
    comparison["bootstrap_significant"] = (
        comparison["bootstrap_q_right"] <= args.q_cutoff
    )
    comparison["bayesian_significant"] = (
        comparison["bayesian_q_right"] <= args.q_cutoff
    )
    activity_rho, n_activity = correlation(
        comparison["bootstrap_effect_log"], comparison["bayesian_effect_log"]
    )

    tested = comparison[
        comparison["bootstrap_q_right"].notna()
        & comparison["bayesian_q_right"].notna()
    ]
    boot_sig = tested["bootstrap_significant"]
    bayes_sig = tested["bayesian_significant"]
    n_both = int((boot_sig & bayes_sig).sum())
    n_union = int((boot_sig | bayes_sig).sum())
    summary = {
        "bayes_tag": tag,
        "q_cutoff": args.q_cutoff,
        "n_common_pairs": int(len(comparison)),
        "n_activity_pairs": n_activity,
        "activity_spearman": activity_rho,
        "n_pairs_tested_by_both": int(len(tested)),
        "n_bootstrap_significant": int(boot_sig.sum()),
        "n_bayesian_significant": int(bayes_sig.sum()),
        "n_significant_both": n_both,
        "significant_jaccard": float(n_both / n_union) if n_union else np.nan,
    }
    return comparison, summary


def plot_activity_heatmaps(comparison: pd.DataFrame, args: argparse.Namespace) -> None:
    usable = comparison[
        comparison["n_cells"].ge(args.min_cells_heatmap)
        & ~comparison["is_negative_control"]
    ].copy()
    score = (
        usable.assign(
            score=usable[
                ["bootstrap_effect_log", "bayesian_effect_log"]
            ].abs().mean(axis=1)
        )
        .groupby("cre")["score"]
        .mean()
        .sort_values(ascending=False)
    )
    selected_cres = score.head(args.max_cres_heatmap).index
    usable = usable[usable["cre"].isin(selected_cres)]
    if usable.empty:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.text(
            0.5,
            0.5,
            "No finite activity pairs pass the heatmap filters",
            ha="center",
            va="center",
        )
        ax.set_axis_off()
        save_figure(fig, args.figures_dir / "activity_heatmaps")
        return
    groups = (
        usable[["group", "n_cells"]]
        .drop_duplicates()
        .sort_values(["n_cells", "group"], ascending=[False, True])["group"]
    )
    cres = score.index.intersection(selected_cres)
    matrices = [
        usable.pivot(index="cre", columns="group", values=value)
        .reindex(index=cres, columns=groups)
        for value in ("bootstrap_effect_log", "bayesian_effect_log")
    ]
    finite = np.concatenate(
        [matrix.to_numpy().ravel() for matrix in matrices]
    )
    finite = finite[np.isfinite(finite)]
    vmax = float(np.percentile(np.abs(finite), 98)) if finite.size else 1.0
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(max(14, len(groups) * 0.28), max(12, len(cres) * 0.24)),
        sharex=True,
        constrained_layout=True,
    )
    titles = ["Bootstrap", "Bayesian hierarchical model"]
    for ax, matrix, title in zip(axes, matrices, titles):
        sns.heatmap(
            matrix,
            ax=ax,
            cmap="RdBu_r",
            center=0,
            vmin=-vmax,
            vmax=vmax,
            xticklabels=True,
            yticklabels=True,
            cbar_kws={"label": "log activity above negative-control mean"},
        )
        ax.set_title(title)
        ax.set_xlabel("")
        ax.set_ylabel("cCRE")
        ax.tick_params(axis="y", labelsize=6)
    axes[-1].set_xlabel("Subclass")
    axes[-1].tick_params(axis="x", rotation=90, labelsize=6)
    save_figure(fig, args.figures_dir / "activity_heatmaps")


def plot_activity_scatter(
    comparison: pd.DataFrame, summary: dict, args: argparse.Namespace
) -> None:
    valid = comparison[
        np.isfinite(comparison["bootstrap_effect_log"])
        & np.isfinite(comparison["bayesian_effect_log"])
        & ~comparison["is_negative_control"]
    ]
    fig, ax = plt.subplots(figsize=(7, 6))
    if valid.empty:
        ax.text(
            0.5,
            0.5,
            "No finite activity pairs available",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_axis_off()
        save_figure(fig, args.figures_dir / "activity_concordance")
        return
    image = ax.hexbin(
        valid["bootstrap_effect_log"],
        valid["bayesian_effect_log"],
        gridsize=70,
        mincnt=1,
        bins="log",
        cmap="viridis",
    )
    lo = float(
        np.nanmin(
            valid[["bootstrap_effect_log", "bayesian_effect_log"]].to_numpy()
        )
    )
    hi = float(
        np.nanmax(
            valid[["bootstrap_effect_log", "bayesian_effect_log"]].to_numpy()
        )
    )
    ax.plot([lo, hi], [lo, hi], color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Bootstrap effect (natural log)")
    ax.set_ylabel("Bayesian effect (natural log)")
    ax.set_title(
        f"Activity concordance: Spearman ρ={summary['activity_spearman']:.3f} "
        f"(n={summary['n_activity_pairs']:,})"
    )
    fig.colorbar(image, ax=ax, label="log10(pair count)")
    fig.tight_layout()
    save_figure(fig, args.figures_dir / "activity_concordance")


def plot_significance_overlap(comparison: pd.DataFrame, args: argparse.Namespace) -> None:
    tested = comparison[
        comparison["bootstrap_q_right"].notna()
        & comparison["bayesian_q_right"].notna()
    ]
    boot = tested["bootstrap_significant"]
    bayes = tested["bayesian_significant"]
    counts = pd.Series(
        {
            "Bootstrap only": int((boot & ~bayes).sum()),
            "Both": int((boot & bayes).sum()),
            "Bayesian only": int((~boot & bayes).sum()),
        }
    )
    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(
        counts.index,
        counts.values,
        color=["#f58518", "#7a5195", "#4c78a8"],
    )
    ax.bar_label(bars, fmt="{:,.0f}", padding=3)
    ax.set_ylabel("Significant subclass–cCRE pairs")
    ax.set_title(f"Right-tail calls at q ≤ {args.q_cutoff:g}")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    save_figure(fig, args.figures_dir / "significance_overlap")


def per_subclass_table(comparison: pd.DataFrame, q_cutoff: float) -> pd.DataFrame:
    rows = []
    for group, frame in comparison.groupby("group", sort=False):
        rho, n_activity = correlation(
            frame["bootstrap_effect_log"], frame["bayesian_effect_log"]
        )
        tested = frame[
            frame["bootstrap_q_right"].notna()
            & frame["bayesian_q_right"].notna()
        ]
        boot = tested["bootstrap_q_right"].le(q_cutoff)
        bayes = tested["bayesian_q_right"].le(q_cutoff)
        both = int((boot & bayes).sum())
        union = int((boot | bayes).sum())
        rows.append(
            {
                "group": group,
                "n_cells": int(frame["n_cells"].dropna().iloc[0])
                if frame["n_cells"].notna().any()
                else 0,
                "activity_spearman": rho,
                "n_activity_pairs": n_activity,
                "n_tested_both": int(len(tested)),
                "n_bootstrap_significant": int(boot.sum()),
                "n_bayesian_significant": int(bayes.sum()),
                "n_significant_both": both,
                "significant_jaccard": both / union if union else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["n_cells", "group"], ascending=[False, True]
    )


def plot_subclass_concordance(per_group: pd.DataFrame, args: argparse.Namespace) -> None:
    valid = per_group[
        per_group["activity_spearman"].notna() & per_group["n_cells"].gt(0)
    ]
    fig, ax = plt.subplots(figsize=(7, 5.5))
    if valid.empty:
        ax.text(
            0.5,
            0.5,
            "No subclasses have a finite activity correlation",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_axis_off()
        save_figure(fig, args.figures_dir / "subclass_concordance")
        return
    points = ax.scatter(
        valid["n_cells"],
        valid["activity_spearman"],
        c=valid["n_significant_both"],
        cmap="magma",
        s=30,
        alpha=0.8,
    )
    ax.set_xscale("log")
    ax.axhline(0, color="grey", linewidth=0.8)
    ax.set_xlabel("Cells per subclass (log scale)")
    ax.set_ylabel("Bootstrap–Bayesian Spearman ρ")
    ax.set_title("Per-subclass activity concordance")
    fig.colorbar(points, ax=ax, label="Significant calls shared")
    fig.tight_layout()
    save_figure(fig, args.figures_dir / "subclass_concordance")


def plot_top_cre_call_counts(
    comparison: pd.DataFrame, args: argparse.Namespace
) -> pd.DataFrame:
    noncontrol = comparison[~comparison["is_negative_control"]]
    counts = (
        noncontrol.groupby("cre")
        .agg(
            bootstrap_significant=("bootstrap_significant", "sum"),
            bayesian_significant=("bayesian_significant", "sum"),
        )
        .astype(int)
    )
    counts["either"] = counts.max(axis=1)
    counts = counts.sort_values(
        ["either", "bootstrap_significant", "bayesian_significant"],
        ascending=False,
    )
    top = counts.head(30).sort_values("either")
    fig, ax = plt.subplots(figsize=(8, 8))
    y = np.arange(len(top))
    ax.barh(
        y - 0.18,
        top["bootstrap_significant"],
        height=0.36,
        label="Bootstrap",
        color="#f58518",
    )
    ax.barh(
        y + 0.18,
        top["bayesian_significant"],
        height=0.36,
        label="Bayesian",
        color="#4c78a8",
    )
    ax.set_yticks(y)
    ax.set_yticklabels(top.index, fontsize=8)
    ax.set_xlabel("Subclasses with a significant right-tail call")
    ax.set_title("Top cCREs by significant subclass count")
    ax.legend(frameon=False)
    fig.tight_layout()
    save_figure(fig, args.figures_dir / "top_cre_significant_calls")
    return counts.drop(columns="either")


def main() -> None:
    args = parse_args()
    args.figures_dir.mkdir(parents=True, exist_ok=True)
    args.tables_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="talk")

    comparison, summary = build_comparison(args)
    comparison.to_csv(args.tables_dir / "method_comparison_long.csv", index=False)
    per_group = per_subclass_table(comparison, args.q_cutoff)
    per_group.to_csv(args.tables_dir / "per_subclass_comparison.csv", index=False)
    summary_frame = pd.DataFrame([summary])
    summary_frame.to_csv(args.tables_dir / "comparison_summary.csv", index=False)
    write_json(args.tables_dir / "comparison_summary.json", summary)

    plot_activity_heatmaps(comparison, args)
    plot_activity_scatter(comparison, summary, args)
    plot_significance_overlap(comparison, args)
    plot_subclass_concordance(per_group, args)
    cre_counts = plot_top_cre_call_counts(comparison, args)
    cre_counts.to_csv(args.tables_dir / "significant_calls_per_cre.csv")
    log(
        f"[plots] wrote {len(list(args.figures_dir.glob('*.pdf')))} PDF/PNG figure "
        f"pairs to {args.figures_dir} and tables to {args.tables_dir}"
    )


if __name__ == "__main__":
    main()
