#!/usr/bin/env python3
"""Plot ordinary negative-control alpha posteriors against the pooled alpha."""

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
from scipy.stats import pearsonr, spearmanr

from analysis_utils import (
    ANALYSIS_DIR,
    DEFAULT_H5AD,
    LIBSIZE_CSV,
    read_and_prepare_adata,
    write_json,
)


POOLED_NAME = "NEGATIVE_CONTROL_POOL"


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
    parser.add_argument("--library-size-csv", type=Path, default=LIBSIZE_CSV)
    parser.add_argument(
        "--tables-dir", type=Path, default=ANALYSIS_DIR / "results" / "tables"
    )
    parser.add_argument(
        "--figures-dir", type=Path, default=ANALYSIS_DIR / "results" / "figures"
    )
    parser.add_argument(
        "--stem", default="joint_dropout_individual_negative_control_alpha"
    )
    return parser.parse_args()


def load_alpha(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    manifest = json.loads((args.root / "run_manifest.json").read_text())
    posterior_path = args.root / f"{manifest['tag']}_posterior_samples.npz"
    negative_controls = pd.read_csv(args.root / "negative_controls.csv").iloc[
        :, 0
    ].astype(str).tolist()
    with np.load(posterior_path, allow_pickle=True) as posterior:
        alpha = posterior["alpha"].astype(np.float64)
        alpha_neg = posterior["alpha_neg"].astype(np.float64)
        cre_names = posterior["cre_names"].astype(str)

    lookup = {name: idx for idx, name in enumerate(cre_names)}
    missing = sorted(set(negative_controls + [POOLED_NAME]) - set(lookup))
    if missing:
        raise ValueError(f"posterior is missing expected cCREs: {missing}")
    pooled_error = np.max(np.abs(alpha[:, lookup[POOLED_NAME]] - alpha_neg))
    if pooled_error > 1e-6:
        raise ValueError("pooled alpha column does not match alpha_neg")

    library = pd.read_csv(args.library_size_csv, index_col=0)["counts"]
    library.index = library.index.astype(str)
    adata = read_and_prepare_adata(args.h5ad)
    t7_total = adata.obsm["T7CRE"].loc[:, negative_controls].sum(axis=0)
    t7_total.index = t7_total.index.astype(str)
    target_mask = ~np.isin(cre_names, negative_controls + [POOLED_NAME])
    target_means = alpha[:, target_mask].mean(axis=0)

    draw_frames = []
    summary_rows = []
    for cre in negative_controls:
        draws = alpha[:, lookup[cre]]
        difference = draws - alpha_neg
        draw_frames.append(
            pd.DataFrame(
                {"draw": np.arange(len(draws)), "cre": cre, "alpha": draws}
            )
        )
        summary_rows.append(
            {
                "cre": cre,
                "alpha_mean": float(draws.mean()),
                "alpha_sd": float(draws.std(ddof=1)),
                "alpha_q025": float(np.quantile(draws, 0.025)),
                "alpha_q975": float(np.quantile(draws, 0.975)),
                "difference_vs_pool_mean": float(difference.mean()),
                "difference_vs_pool_q025": float(np.quantile(difference, 0.025)),
                "difference_vs_pool_q975": float(np.quantile(difference, 0.975)),
                "probability_greater_than_pool": float((difference > 0).mean()),
                "fold_vs_pool": float(np.exp(difference.mean())),
                "nanopore_count": float(library.get(cre, np.nan)),
                "t7_total": float(t7_total.get(cre, np.nan)),
                "target_alpha_percentile": float(
                    (target_means < draws.mean()).mean()
                ),
            }
        )

    draw_frames.append(
        pd.DataFrame(
            {
                "draw": np.arange(len(alpha_neg)),
                "cre": "Pooled all seven",
                "alpha": alpha_neg,
            }
        )
    )
    summary_rows.append(
        {
            "cre": "Pooled all seven",
            "alpha_mean": float(alpha_neg.mean()),
            "alpha_sd": float(alpha_neg.std(ddof=1)),
            "alpha_q025": float(np.quantile(alpha_neg, 0.025)),
            "alpha_q975": float(np.quantile(alpha_neg, 0.975)),
            "difference_vs_pool_mean": 0.0,
            "difference_vs_pool_q025": 0.0,
            "difference_vs_pool_q975": 0.0,
            "probability_greater_than_pool": np.nan,
            "fold_vs_pool": 1.0,
            "nanopore_count": float(library.reindex(negative_controls).fillna(0).sum()),
            "t7_total": float(t7_total.reindex(negative_controls).fillna(0).sum()),
            "target_alpha_percentile": float(
                (target_means < alpha_neg.mean()).mean()
            ),
        }
    )
    metadata = {
        "root": str(args.root),
        "posterior": str(posterior_path),
        "t7_source": str(args.h5ad),
        "t7_definition": "sum of T7CRE counts across all cells for each cCRE",
        "negative_controls": negative_controls,
        "pooled_alpha_site": "alpha_neg",
        "ordinary_alpha_site": "alpha at each annotated negative-control cCRE",
    }
    return pd.concat(draw_frames, ignore_index=True), pd.DataFrame(summary_rows), metadata


def alpha_t7_correlations(draws: pd.DataFrame, summary: pd.DataFrame) -> dict:
    controls = summary[summary["cre"] != "Pooled all seven"].copy()
    order = controls["cre"].astype(str).tolist()
    x = np.log10(1.0 + controls.set_index("cre").loc[order, "t7_total"].to_numpy(float))
    y = controls.set_index("cre").loc[order, "alpha_mean"].to_numpy(float)
    pearson = pearsonr(x, y)
    spearman = spearmanr(x, y)

    posterior = draws[draws["cre"].isin(order)].pivot(
        index="draw", columns="cre", values="alpha"
    ).reindex(columns=order)
    posterior_pearson = np.asarray(
        [pearsonr(x, row).statistic for row in posterior.to_numpy(float)]
    )
    posterior_spearman = np.asarray(
        [spearmanr(x, row).statistic for row in posterior.to_numpy(float)]
    )
    leave_one_out_pearson = []
    leave_one_out_spearman = []
    for omitted in range(len(order)):
        keep = np.arange(len(order)) != omitted
        leave_one_out_pearson.append(pearsonr(x[keep], y[keep]).statistic)
        leave_one_out_spearman.append(spearmanr(x[keep], y[keep]).statistic)

    return {
        "n_controls": len(order),
        "x": "log10(1 + total T7 count across all cells)",
        "pooled_point_excluded": True,
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
        "leave_one_control_out": {
            "pearson_min": float(np.min(leave_one_out_pearson)),
            "pearson_max": float(np.max(leave_one_out_pearson)),
            "spearman_min": float(np.min(leave_one_out_spearman)),
            "spearman_max": float(np.max(leave_one_out_spearman)),
        },
    }


def plot_alpha(draws: pd.DataFrame, summary: pd.DataFrame, output: Path) -> None:
    sns.set_theme(context="paper", style="whitegrid")
    controls = summary[summary["cre"] != "Pooled all seven"].copy()
    order = controls.sort_values("alpha_mean", ascending=False)["cre"].tolist()
    labels = order + ["Pooled all seven"]
    palette = dict(
        zip(labels, sns.color_palette("tab10", n_colors=len(labels)))
    )
    palette["Pooled all seven"] = "black"

    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.8), constrained_layout=True)

    ax = axes[0]
    for label in labels:
        sns.kdeplot(
            data=draws[draws["cre"] == label],
            x="alpha",
            color=palette[label],
            linewidth=1.3,
            label=label,
            ax=ax,
        )
    ax.set_xlabel("alpha posterior draw")
    ax.set_ylabel("Density")
    ax.set_title("Posterior distributions")
    ax.legend(frameon=False, fontsize=6)

    ax = axes[1]
    forest = controls.set_index("cre").loc[order].reset_index()
    y = np.arange(len(forest))
    x = forest["difference_vs_pool_mean"].to_numpy(float)
    lower = x - forest["difference_vs_pool_q025"].to_numpy(float)
    upper = forest["difference_vs_pool_q975"].to_numpy(float) - x
    excludes_zero = (
        forest["difference_vs_pool_q025"].gt(0)
        | forest["difference_vs_pool_q975"].lt(0)
    )
    colors = np.where(excludes_zero, "#D95F02", "#666666")
    for yi, xi, lo, hi, color in zip(y, x, lower, upper, colors):
        ax.errorbar(
            xi,
            yi,
            xerr=np.asarray([[lo], [hi]]),
            fmt="o",
            color=color,
            capsize=2.5,
            markersize=4,
        )
    ax.axvline(0.0, color="black", linestyle="--", linewidth=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels(forest["cre"])
    ax.invert_yaxis()
    ax.set_xlabel("alpha(cCRE) - alpha(pooled)")
    ax.set_title("Matched posterior contrasts (95% intervals)")

    ax = axes[2]
    plot_data = summary.copy()
    for row in plot_data.itertuples(index=False):
        is_pool = row.cre == "Pooled all seven"
        ax.errorbar(
            np.log10(1.0 + row.nanopore_count),
            row.alpha_mean,
            yerr=np.asarray(
                [[row.alpha_mean - row.alpha_q025], [row.alpha_q975 - row.alpha_mean]]
            ),
            fmt="*" if is_pool else "o",
            markersize=9 if is_pool else 5,
            capsize=2,
            color="black" if is_pool else palette[row.cre],
        )
        ax.annotate(
            row.cre.replace("Pooled all seven", "Pooled"),
            (np.log10(1.0 + row.nanopore_count), row.alpha_mean),
            xytext=(4, 3),
            textcoords="offset points",
            fontsize=6,
        )
    ax.set_xlabel("log10(1 + nanopore count)")
    ax.set_ylabel("Posterior mean alpha")
    ax.set_title("Alpha versus library abundance")

    fig.suptitle(
        "Joint+dropout negative-control cCRE baselines from one ordinary-and-pooled fit",
        fontsize=12,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def plot_alpha_vs_t7(
    summary: pd.DataFrame, correlations: dict, output: Path
) -> None:
    sns.set_theme(context="paper", style="whitegrid")
    controls = summary[summary["cre"] != "Pooled all seven"].copy()
    controls["log10_t7"] = np.log10(1.0 + controls["t7_total"])
    pooled = summary[summary["cre"] == "Pooled all seven"].iloc[0]

    fig, ax = plt.subplots(figsize=(6.4, 5.2), constrained_layout=True)
    sns.regplot(
        data=controls,
        x="log10_t7",
        y="alpha_mean",
        scatter=False,
        ci=None,
        color="#555555",
        line_kws={"linewidth": 1.0, "linestyle": "--"},
        ax=ax,
    )
    palette = dict(
        zip(
            sorted(controls["cre"].astype(str)),
            sns.color_palette("tab10", n_colors=len(controls)),
        )
    )
    label_positions = {
        "CRE328": (6, 5, "left"),
        "CRE330": (7, -12, "left"),
        "CRE331": (7, 5, "left"),
        "CRE332": (7, 5, "left"),
        "CRE333": (8, 9, "left"),
        "CRE336": (-8, -12, "right"),
        "CRE337": (-8, 9, "right"),
    }
    for row in controls.itertuples(index=False):
        ax.errorbar(
            row.log10_t7,
            row.alpha_mean,
            yerr=np.asarray(
                [[row.alpha_mean - row.alpha_q025], [row.alpha_q975 - row.alpha_mean]]
            ),
            fmt="o",
            color=palette[row.cre],
            capsize=2.5,
            markersize=6,
        )
        dx, dy, alignment = label_positions[row.cre]
        ax.annotate(
            row.cre,
            (row.log10_t7, row.alpha_mean),
            xytext=(dx, dy),
            textcoords="offset points",
            ha=alignment,
            fontsize=7,
        )
    pooled_x = np.log10(1.0 + float(pooled["t7_total"]))
    ax.errorbar(
        pooled_x,
        float(pooled["alpha_mean"]),
        yerr=np.asarray(
            [[
                float(pooled["alpha_mean"] - pooled["alpha_q025"]),
            ], [
                float(pooled["alpha_q975"] - pooled["alpha_mean"]),
            ]]
        ),
        fmt="*",
        color="black",
        capsize=2.5,
        markersize=11,
    )
    ax.annotate(
        "Pooled (excluded from correlation)",
        (pooled_x, float(pooled["alpha_mean"])),
        xytext=(-5, 8),
        textcoords="offset points",
        ha="right",
        fontsize=7,
    )
    pearson = correlations["pearson_posterior_mean"]
    spearman = correlations["spearman_posterior_mean"]
    ax.set_xlabel("Total T7 count across all cells, log10(count + 1)")
    ax.set_ylabel("Posterior mean alpha")
    ax.set_title(
        "Ordinary negative-control alpha versus total T7\n"
        f"Pearson r={pearson['r']:.3f}, p={pearson['p']:.3g}; "
        f"Spearman rho={spearman['rho']:.3f}, p={spearman['p']:.3g}"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.tables_dir.mkdir(parents=True, exist_ok=True)
    args.figures_dir.mkdir(parents=True, exist_ok=True)
    draws, summary, metadata = load_alpha(args)
    correlations = alpha_t7_correlations(draws, summary)
    table_path = args.tables_dir / f"{args.stem}_summary.csv"
    figure_path = args.figures_dir / f"{args.stem}.pdf"
    t7_figure_path = args.figures_dir / f"{args.stem}_vs_total_t7.pdf"
    summary.to_csv(table_path, index=False)
    plot_alpha(draws, summary, figure_path)
    plot_alpha_vs_t7(summary, correlations, t7_figure_path)
    write_json(
        args.figures_dir / f"{args.stem}_manifest.json",
        {
            **metadata,
            "alpha_t7_correlations": correlations,
            "outputs": {
                "summary": str(table_path),
                "figure": str(figure_path),
                "total_t7_figure": str(t7_figure_path),
            },
            "interpretation": (
                "alpha is the cCRE-wide log-activity baseline; comparisons use "
                "matched posterior draws from the ordinary-and-pooled joint fit"
            ),
        },
    )


if __name__ == "__main__":
    main()
