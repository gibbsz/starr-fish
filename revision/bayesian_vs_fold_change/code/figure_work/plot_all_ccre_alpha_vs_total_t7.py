#!/usr/bin/env python3
"""Plot posterior alpha against total T7 counts for every fitted cCRE."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr, rankdata, spearmanr

# The shared analysis layer (analysis_utils and the plot_* modules that other
# scripts import) stays in the parent code/ directory.
import sys as _sys
from pathlib import Path as _Path
_CODE_DIR = _Path(__file__).resolve().parent.parent
if str(_CODE_DIR) not in _sys.path:
    _sys.path.insert(0, str(_CODE_DIR))

from analysis_utils import ANALYSIS_DIR, DEFAULT_H5AD, FIGURES_WORK, write_json


POOLED_NAME = "NEGATIVE_CONTROL_POOL"
T7_THRESHOLDS = (0, 25, 50, 100, 250, 500, 1000, 2000, 3000)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=ANALYSIS_DIR
        / "results"
        / "ablation"
        / "bayesian_joint_dropout_ordinary_and_pooled_negative_controls",
    )
    parser.add_argument("--h5ad", type=Path, default=DEFAULT_H5AD)
    parser.add_argument(
        "--tables-dir", type=Path, default=ANALYSIS_DIR / "results" / "tables"
    )
    parser.add_argument(
        "--figures-dir", type=Path, default=FIGURES_WORK
    )
    parser.add_argument(
        "--stem", default="joint_dropout_all_ccre_alpha_vs_total_t7"
    )
    return parser.parse_args()


def correlations(x: np.ndarray, alpha: np.ndarray) -> dict:
    alpha_mean = alpha.mean(axis=0)
    pearson = pearsonr(x, alpha_mean)
    spearman = spearmanr(x, alpha_mean)

    x_centered = x - x.mean()
    x_scale = np.sqrt(np.square(x_centered).sum())
    alpha_centered = alpha - alpha.mean(axis=1, keepdims=True)
    posterior_pearson = (
        alpha_centered @ x_centered
        / (np.sqrt(np.square(alpha_centered).sum(axis=1)) * x_scale)
    )
    x_rank = rankdata(x)
    x_rank_centered = x_rank - x_rank.mean()
    x_rank_scale = np.sqrt(np.square(x_rank_centered).sum())
    alpha_rank = np.apply_along_axis(rankdata, 1, alpha)
    alpha_rank_centered = alpha_rank - alpha_rank.mean(axis=1, keepdims=True)
    posterior_spearman = (
        alpha_rank_centered @ x_rank_centered
        / (np.sqrt(np.square(alpha_rank_centered).sum(axis=1)) * x_rank_scale)
    )
    return {
        "n_ccres": int(alpha.shape[1]),
        "pearson_posterior_mean": {
            "r": float(pearson.statistic),
            "p": float(pearson.pvalue),
        },
        "spearman_posterior_mean": {
            "rho": float(spearman.statistic),
            "p": float(spearman.pvalue),
        },
        "posterior_draw_correlations": {
            "pearson_median": float(np.median(posterior_pearson)),
            "pearson_q025": float(np.quantile(posterior_pearson, 0.025)),
            "pearson_q975": float(np.quantile(posterior_pearson, 0.975)),
            "pearson_probability_positive": float((posterior_pearson > 0).mean()),
            "spearman_median": float(np.median(posterior_spearman)),
            "spearman_q025": float(np.quantile(posterior_spearman, 0.025)),
            "spearman_q975": float(np.quantile(posterior_spearman, 0.975)),
            "spearman_probability_positive": float((posterior_spearman > 0).mean()),
        },
    }


def load_data(args: argparse.Namespace) -> tuple[pd.DataFrame, np.ndarray, dict]:
    manifest = json.loads((args.root / "run_manifest.json").read_text())
    posterior_path = args.root / f"{manifest['tag']}_posterior_samples.npz"
    negative_controls = pd.read_csv(args.root / "negative_controls.csv").iloc[
        :, 0
    ].astype(str).tolist()
    with np.load(posterior_path, allow_pickle=True) as posterior:
        alpha_all = posterior["alpha"].astype(np.float64)
        cre_names_all = posterior["cre_names"].astype(str)

    ordinary_mask = cre_names_all != POOLED_NAME
    cre_names = cre_names_all[ordinary_mask]
    alpha = alpha_all[:, ordinary_mask]
    with h5py.File(args.h5ad, "r") as handle:
        t7_group = handle["obsm"]["T7CRE"]
        missing = sorted(set(cre_names) - set(t7_group.keys()))
        if missing:
            raise ValueError(f"T7 matrix is missing fitted cCREs: {missing}")
        totals = np.asarray(
            [float(t7_group[name][...].sum()) for name in cre_names], dtype=float
        )

    frame = pd.DataFrame(
        {
            "cre": cre_names,
            "is_negative_control": np.isin(cre_names, negative_controls),
            "total_t7": totals,
            "log10_total_t7_plus1": np.log10(totals + 1.0),
            "alpha_mean": alpha.mean(axis=0),
            "alpha_sd": alpha.std(axis=0, ddof=1),
            "alpha_q025": np.quantile(alpha, 0.025, axis=0),
            "alpha_q975": np.quantile(alpha, 0.975, axis=0),
        }
    )
    all_stats = correlations(frame["log10_total_t7_plus1"].to_numpy(), alpha)
    target_mask = ~frame["is_negative_control"].to_numpy()
    target_stats = correlations(
        frame.loc[target_mask, "log10_total_t7_plus1"].to_numpy(),
        alpha[:, target_mask],
    )
    control_mask = frame["is_negative_control"].to_numpy()
    control_stats = correlations(
        frame.loc[control_mask, "log10_total_t7_plus1"].to_numpy(),
        alpha[:, control_mask],
    )
    threshold_sensitivity = []
    for threshold in T7_THRESHOLDS:
        keep = frame["total_t7"].to_numpy() >= threshold
        x = frame.loc[keep, "log10_total_t7_plus1"].to_numpy(float)
        y = frame.loc[keep, "alpha_mean"].to_numpy(float)
        threshold_sensitivity.append(
            {
                "minimum_total_t7": threshold,
                "n_ccres": int(keep.sum()),
                "pearson_r": float(pearsonr(x, y).statistic),
                "pearson_p": float(pearsonr(x, y).pvalue),
                "spearman_rho": float(spearmanr(x, y).statistic),
                "spearman_p": float(spearmanr(x, y).pvalue),
            }
        )
    metadata = {
        "root": str(args.root),
        "posterior": str(posterior_path),
        "t7_source": str(args.h5ad),
        "t7_definition": "sum of T7CRE counts across every cell for each cCRE",
        "x_transform": "log10(total_t7 + 1)",
        "pooled_pseudo_ccre_excluded": True,
        "negative_controls_included_in_all_correlation": True,
        "correlations": {
            "all_ordinary_ccres": all_stats,
            "targets_only": target_stats,
            "negative_controls_only": control_stats,
        },
        "t7_threshold_sensitivity": threshold_sensitivity,
    }
    return frame, alpha, metadata


def plot(frame: pd.DataFrame, metadata: dict, output: Path) -> None:
    sns.set_theme(context="paper", style="whitegrid")
    targets = frame.loc[~frame["is_negative_control"]].copy()
    controls = frame.loc[frame["is_negative_control"]].copy()
    all_stats = metadata["correlations"]["all_ordinary_ccres"]
    target_stats = metadata["correlations"]["targets_only"]

    quantiles = min(10, int(frame["total_t7"].nunique()))
    frame = frame.copy()
    frame["t7_bin"] = pd.qcut(
        frame["total_t7"], q=quantiles, duplicates="drop"
    )
    binned = (
        frame.groupby("t7_bin", observed=True)
        .agg(
            x=("log10_total_t7_plus1", "median"),
            alpha_median=("alpha_mean", "median"),
            alpha_q25=("alpha_mean", lambda values: values.quantile(0.25)),
            alpha_q75=("alpha_mean", lambda values: values.quantile(0.75)),
            n=("cre", "size"),
        )
        .reset_index(drop=True)
    )

    fig, axes = plt.subplots(
        1, 3, figsize=(16.8, 5.2), constrained_layout=True,
        gridspec_kw={"width_ratios": [1.55, 1.0, 1.0]},
    )
    ax = axes[0]
    sns.regplot(
        data=targets,
        x="log10_total_t7_plus1",
        y="alpha_mean",
        scatter=False,
        ci=95,
        color="#4C78A8",
        line_kws={"linewidth": 1.1},
        ax=ax,
    )
    ax.scatter(
        targets["log10_total_t7_plus1"],
        targets["alpha_mean"],
        s=14,
        alpha=0.45,
        color="#4C78A8",
        linewidths=0,
        label=f"Target cCREs (n={len(targets)})",
    )
    palette = dict(
        zip(sorted(controls["cre"]), sns.color_palette("tab10", len(controls)))
    )
    for row in controls.itertuples(index=False):
        ax.errorbar(
            row.log10_total_t7_plus1,
            row.alpha_mean,
            yerr=np.asarray(
                [[row.alpha_mean - row.alpha_q025], [row.alpha_q975 - row.alpha_mean]]
            ),
            fmt="o",
            color=palette[row.cre],
            capsize=2,
            markersize=5,
            zorder=3,
        )
        ax.annotate(
            row.cre,
            (row.log10_total_t7_plus1, row.alpha_mean),
            xytext=(4, 3),
            textcoords="offset points",
            fontsize=6,
        )
    ax.scatter([], [], s=22, color="black", label="Negative controls (n=7)")
    ax.set_xlabel("Total T7 across all cells, log10(count + 1)")
    ax.set_ylabel("Posterior mean alpha")
    ax.set_title(
        "All fitted ordinary cCREs\n"
        f"All: Pearson r={all_stats['pearson_posterior_mean']['r']:.3f}, "
        f"Spearman rho={all_stats['spearman_posterior_mean']['rho']:.3f}\n"
        f"Targets only: Pearson r={target_stats['pearson_posterior_mean']['r']:.3f}, "
        f"Spearman rho={target_stats['spearman_posterior_mean']['rho']:.3f}"
    )
    ax.legend(frameon=False, fontsize=7, loc="best")

    ax = axes[1]
    x = binned["x"].to_numpy(float)
    y = binned["alpha_median"].to_numpy(float)
    lower = y - binned["alpha_q25"].to_numpy(float)
    upper = binned["alpha_q75"].to_numpy(float) - y
    ax.errorbar(
        x,
        y,
        yerr=np.vstack([lower, upper]),
        fmt="o-",
        color="#2F6F8F",
        capsize=3,
        linewidth=1.2,
        markersize=5,
    )
    for row in binned.itertuples(index=False):
        ax.annotate(
            f"n={row.n}",
            (row.x, row.alpha_median),
            xytext=(0, 7),
            textcoords="offset points",
            ha="center",
            fontsize=6,
        )
    ax.set_xlabel("Median log10(total T7 + 1) in T7 quantile bin")
    ax.set_ylabel("Median alpha (IQR)")
    ax.set_title("T7-depth trend across cCREs")

    ax = axes[2]
    sensitivity = pd.DataFrame(metadata["t7_threshold_sensitivity"])
    sensitivity["x"] = np.log10(sensitivity["minimum_total_t7"] + 1.0)
    ax.plot(
        sensitivity["x"],
        sensitivity["pearson_r"],
        "o-",
        color="#D95F02",
        linewidth=1.2,
        markersize=4,
        label="Pearson r",
    )
    ax.plot(
        sensitivity["x"],
        sensitivity["spearman_rho"],
        "s--",
        color="#5E3C99",
        linewidth=1.0,
        markersize=4,
        label="Spearman rho",
    )
    ax.axhline(0, color="black", linewidth=0.7)
    ax.set_xticks(sensitivity["x"])
    ax.set_xticklabels(
        [
            f"{int(row.minimum_total_t7)}\n(n={int(row.n_ccres)})"
            for row in sensitivity.itertuples(index=False)
        ],
        rotation=45,
        ha="right",
    )
    ax.set_xlabel("Minimum total T7")
    ax.set_ylabel("Correlation with alpha")
    ax.set_title("Sensitivity to low-depth cCREs")
    ax.legend(frameon=False, fontsize=7)

    fig.suptitle(
        "Joint+dropout posterior alpha versus total T7",
        fontsize=12,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.tables_dir.mkdir(parents=True, exist_ok=True)
    args.figures_dir.mkdir(parents=True, exist_ok=True)
    frame, _alpha, metadata = load_data(args)
    table_path = args.tables_dir / f"{args.stem}.csv"
    figure_path = args.figures_dir / f"{args.stem}.pdf"
    manifest_path = args.figures_dir / f"{args.stem}_manifest.json"
    frame.to_csv(table_path, index=False)
    plot(frame, metadata, figure_path)
    write_json(
        manifest_path,
        {
            **metadata,
            "outputs": {
                "table": str(table_path),
                "figure": str(figure_path),
                "manifest": str(manifest_path),
            },
        },
    )
    print(json.dumps(metadata["correlations"], indent=2))


if __name__ == "__main__":
    main()
