#!/usr/bin/env python3
"""Diagnose why ordinary negative-control alpha estimates differ from the pool."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import h5py
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr, spearmanr

from analysis_utils import (
    ANALYSIS_DIR,
    DEFAULT_H5AD,
    LIBSIZE_CSV,
    write_json,
)


POOLED_NAME = "NEGATIVE_CONTROL_POOL"
POOLED_LABEL = "Pooled all seven"


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
        "--stem", default="joint_dropout_negative_control_alpha_count_diagnostic"
    )
    return parser.parse_args()


def exact_permutation_pvalue(x: np.ndarray, y: np.ndarray, statistic: str) -> float:
    """Two-sided exact permutation p-value for seven paired observations."""
    observed = (
        pearsonr(x, y).statistic
        if statistic == "pearson"
        else spearmanr(x, y).statistic
    )
    permuted = []
    for order in itertools.permutations(range(len(y))):
        candidate = y[np.asarray(order)]
        value = (
            pearsonr(x, candidate).statistic
            if statistic == "pearson"
            else spearmanr(x, candidate).statistic
        )
        permuted.append(value)
    permuted = np.asarray(permuted)
    return float((np.abs(permuted) >= abs(observed) - 1e-12).mean())


def correlation_summary(x: np.ndarray, y: np.ndarray) -> dict:
    pearson = pearsonr(x, y)
    spearman = spearmanr(x, y)
    leave_one_out = []
    for omitted in range(len(x)):
        keep = np.arange(len(x)) != omitted
        leave_one_out.append(float(pearsonr(x[keep], y[keep]).statistic))
    return {
        "pearson_r": float(pearson.statistic),
        "pearson_parametric_p": float(pearson.pvalue),
        "pearson_exact_permutation_p": exact_permutation_pvalue(x, y, "pearson"),
        "spearman_rho": float(spearman.statistic),
        "spearman_parametric_p": float(spearman.pvalue),
        "spearman_exact_permutation_p": exact_permutation_pvalue(x, y, "spearman"),
        "leave_one_out_pearson_min": float(np.min(leave_one_out)),
        "leave_one_out_pearson_max": float(np.max(leave_one_out)),
    }


def load_data(args: argparse.Namespace) -> tuple[pd.DataFrame, dict]:
    manifest = json.loads((args.root / "run_manifest.json").read_text())
    posterior_path = args.root / f"{manifest['tag']}_posterior_samples.npz"
    controls = pd.read_csv(args.root / "negative_controls.csv").iloc[:, 0].astype(str)
    controls = controls.tolist()

    with np.load(posterior_path, allow_pickle=True) as posterior:
        alpha = posterior["alpha"].astype(np.float64)
        alpha_neg = posterior["alpha_neg"].astype(np.float64)
        log_gamma = posterior["log_gamma"].astype(np.float64)
        log_gamma_neg = posterior["log_gamma_neg"].astype(np.float64)
        cre_names = posterior["cre_names"].astype(str)
    lookup = {name: idx for idx, name in enumerate(cre_names)}

    with h5py.File(args.h5ad, "r") as handle:
        t7 = pd.Series(
            {name: float(handle["obsm"]["T7CRE"][name][...].sum()) for name in controls}
        )
        cre = pd.Series(
            {name: float(handle["obsm"]["CRE"][name][...].sum()) for name in controls}
        )
    library = pd.read_csv(args.library_size_csv, index_col=0)["counts"].astype(float)
    library.index = library.index.astype(str)

    rows = []
    for name in controls:
        draws = alpha[:, lookup[name]]
        mean_log_gamma_draws = log_gamma[:, :, lookup[name]].mean(axis=1)
        mean_effect_draws = mean_log_gamma_draws - draws
        rows.append(
            {
                "cre": name,
                "is_pooled": False,
                "alpha_mean": float(draws.mean()),
                "alpha_sd": float(draws.std(ddof=1)),
                "alpha_q025": float(np.quantile(draws, 0.025)),
                "alpha_q975": float(np.quantile(draws, 0.975)),
                "mean_log_gamma": float(mean_log_gamma_draws.mean()),
                "mean_log_gamma_q025": float(
                    np.quantile(mean_log_gamma_draws, 0.025)
                ),
                "mean_log_gamma_q975": float(
                    np.quantile(mean_log_gamma_draws, 0.975)
                ),
                "mean_eta_plus_delta": float(mean_effect_draws.mean()),
                "t7_total": float(t7[name]),
                "cre_total": float(cre[name]),
                "nanopore_count": float(library[name]),
            }
        )
    rows.append(
        {
            "cre": POOLED_LABEL,
            "is_pooled": True,
            "alpha_mean": float(alpha_neg.mean()),
            "alpha_sd": float(alpha_neg.std(ddof=1)),
            "alpha_q025": float(np.quantile(alpha_neg, 0.025)),
            "alpha_q975": float(np.quantile(alpha_neg, 0.975)),
            "mean_log_gamma": float(log_gamma_neg.mean(axis=1).mean()),
            "mean_log_gamma_q025": float(
                np.quantile(log_gamma_neg.mean(axis=1), 0.025)
            ),
            "mean_log_gamma_q975": float(
                np.quantile(log_gamma_neg.mean(axis=1), 0.975)
            ),
            "mean_eta_plus_delta": float(
                (log_gamma_neg.mean(axis=1) - alpha_neg).mean()
            ),
            "t7_total": float(t7.sum()),
            "cre_total": float(cre.sum()),
            "nanopore_count": float(library.reindex(controls).sum()),
        }
    )
    frame = pd.DataFrame(rows)
    frame["log_t7"] = np.log(frame["t7_total"] + 0.5)
    frame["log_cre"] = np.log(frame["cre_total"] + 0.5)
    frame["log_cre_t7_ratio"] = np.log(
        (frame["cre_total"] + 0.5) / (frame["t7_total"] + 0.5)
    )
    frame["log_nanopore"] = np.log1p(frame["nanopore_count"])
    frame["log_t7_per_nanopore"] = frame["log_t7"] - frame["log_nanopore"]

    ordinary = frame.loc[~frame["is_pooled"]].copy()
    x_ratio = ordinary["log_cre_t7_ratio"].to_numpy(float)
    y_alpha = ordinary["alpha_mean"].to_numpy(float)
    design = np.column_stack([np.ones(len(ordinary)), x_ratio])
    ratio_fit = np.linalg.lstsq(design, y_alpha, rcond=None)[0]
    frame["alpha_residual_after_cre_t7_ratio"] = np.nan
    frame.loc[~frame["is_pooled"], "alpha_residual_after_cre_t7_ratio"] = (
        y_alpha - design @ ratio_fit
    )

    metrics = [
        "log_t7",
        "log_cre",
        "log_cre_t7_ratio",
        "log_nanopore",
        "log_t7_per_nanopore",
    ]
    correlations = {
        metric: correlation_summary(
            ordinary[metric].to_numpy(float), ordinary["alpha_mean"].to_numpy(float)
        )
        for metric in metrics
    }
    residual = y_alpha - design @ ratio_fit
    correlations["log_t7_vs_alpha_residual_after_cre_t7_ratio"] = correlation_summary(
        ordinary["log_t7"].to_numpy(float), residual
    )
    correlations["alpha_vs_mean_eta_plus_delta"] = correlation_summary(
        ordinary["alpha_mean"].to_numpy(float),
        ordinary["mean_eta_plus_delta"].to_numpy(float),
    )

    two_predictor = np.column_stack(
        [
            np.ones(len(ordinary)),
            ordinary["log_cre_t7_ratio"].to_numpy(float),
            ordinary["log_t7"].to_numpy(float),
        ]
    )
    coefficients = np.linalg.lstsq(
        two_predictor, ordinary["alpha_mean"].to_numpy(float), rcond=None
    )[0]
    fitted = two_predictor @ coefficients
    total_ss = np.square(y_alpha - y_alpha.mean()).sum()
    residual_ss = np.square(y_alpha - fitted).sum()
    metadata = {
        "n_ordinary_controls": len(ordinary),
        "pooled_point_excluded_from_correlations": True,
        "pseudocount_for_log_count_ratio": 0.5,
        "correlations": correlations,
        "alpha_model_log_cre_t7_ratio_plus_log_t7": {
            "intercept": float(coefficients[0]),
            "log_cre_t7_ratio_coefficient": float(coefficients[1]),
            "log_t7_coefficient": float(coefficients[2]),
            "r_squared": float(1.0 - residual_ss / total_ss),
            "warning": "descriptive only; seven observations and two predictors",
        },
        "posterior_decomposition": {
            "alpha_range": float(
                ordinary["alpha_mean"].max() - ordinary["alpha_mean"].min()
            ),
            "mean_log_gamma_range": float(
                ordinary["mean_log_gamma"].max()
                - ordinary["mean_log_gamma"].min()
            ),
            "mean_eta_plus_delta_range": float(
                ordinary["mean_eta_plus_delta"].max()
                - ordinary["mean_eta_plus_delta"].min()
            ),
            "mean_log_gamma_definition": (
                "unweighted arithmetic mean of posterior log_gamma over 328 subclasses"
            ),
        },
        "sources": {
            "posterior": str(posterior_path),
            "counts": str(args.h5ad),
            "nanopore": str(args.library_size_csv),
        },
    }
    return frame, metadata


def add_scatter(
    ax: plt.Axes,
    frame: pd.DataFrame,
    x: str,
    xlabel: str,
    title: str,
    correlation: dict | None,
    residual_y: bool = False,
) -> None:
    ordinary = frame.loc[~frame["is_pooled"]].copy()
    y = "alpha_residual_after_cre_t7_ratio" if residual_y else "alpha_mean"
    sns.regplot(
        data=ordinary,
        x=x,
        y=y,
        scatter=False,
        ci=None,
        color="0.35",
        line_kws={"linestyle": "--", "linewidth": 0.9},
        ax=ax,
    )
    palette = dict(
        zip(sorted(ordinary["cre"]), sns.color_palette("tab10", len(ordinary)))
    )
    for row in ordinary.itertuples(index=False):
        y_value = getattr(row, y)
        if residual_y:
            ax.plot(getattr(row, x), y_value, "o", color=palette[row.cre], ms=5)
        else:
            ax.errorbar(
                getattr(row, x),
                y_value,
                yerr=np.asarray(
                    [[row.alpha_mean - row.alpha_q025], [row.alpha_q975 - row.alpha_mean]]
                ),
                fmt="o",
                color=palette[row.cre],
                capsize=2,
                ms=5,
            )
        ax.annotate(
            row.cre,
            (getattr(row, x), y_value),
            xytext=(4, 3),
            textcoords="offset points",
            fontsize=6,
        )
    if not residual_y:
        pooled = frame.loc[frame["is_pooled"]].iloc[0]
        ax.errorbar(
            pooled[x],
            pooled[y],
            yerr=np.asarray(
                [[pooled.alpha_mean - pooled.alpha_q025], [pooled.alpha_q975 - pooled.alpha_mean]]
            ),
            fmt="*",
            color="black",
            capsize=2,
            ms=9,
        )
        ax.annotate(
            "Pooled",
            (pooled[x], pooled[y]),
            xytext=(4, 3),
            textcoords="offset points",
            fontsize=6,
        )
    else:
        ax.axhline(0, color="black", lw=0.7)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Alpha residual" if residual_y else "Posterior mean alpha")
    if correlation is not None:
        title += (
            f"\nr={correlation['pearson_r']:.3f}, "
            f"exact p={correlation['pearson_exact_permutation_p']:.3g}"
        )
    ax.set_title(title, fontsize=9)


def plot_diagnostic(frame: pd.DataFrame, metadata: dict, output: Path) -> None:
    sns.set_theme(context="paper", style="whitegrid")
    correlations = metadata["correlations"]
    fig, axes = plt.subplots(2, 3, figsize=(15.2, 9.0), constrained_layout=True)
    panels = [
        (
            "log_t7",
            "ln(total T7 + 0.5)",
            "Alpha versus T7 abundance",
            correlations["log_t7"],
            False,
        ),
        (
            "log_cre",
            "ln(total CRE + 0.5)",
            "Alpha versus CRE output",
            correlations["log_cre"],
            False,
        ),
        (
            "log_cre_t7_ratio",
            "ln[(total CRE + 0.5)/(total T7 + 0.5)]",
            "Alpha versus raw activity ratio",
            correlations["log_cre_t7_ratio"],
            False,
        ),
        (
            "log_nanopore",
            "ln(1 + Nanopore count)",
            "Alpha versus abundance prior input",
            correlations["log_nanopore"],
            False,
        ),
        (
            "log_t7_per_nanopore",
            "ln(total T7 + 0.5) - ln(1 + Nanopore)",
            "Alpha versus T7 excess over Nanopore",
            correlations["log_t7_per_nanopore"],
            False,
        ),
        (
            "log_t7",
            "ln(total T7 + 0.5)",
            "T7 after removing CRE/T7 effect from alpha",
            correlations["log_t7_vs_alpha_residual_after_cre_t7_ratio"],
            True,
        ),
    ]
    for ax, panel in zip(axes.flat, panels):
        add_scatter(ax, frame, *panel)
    fig.suptitle(
        "Negative-control alpha count diagnostics\n"
        "Correlations use seven ordinary controls; pooled pseudo-cCRE is reference only",
        fontsize=12,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def plot_decomposition(frame: pd.DataFrame, metadata: dict, output: Path) -> None:
    sns.set_theme(context="paper", style="whitegrid")
    ordinary = frame.loc[~frame["is_pooled"]].copy()
    order = ordinary.sort_values("alpha_mean", ascending=False)["cre"].tolist()
    display = frame.set_index("cre").loc[order + [POOLED_LABEL]].reset_index()
    y = np.arange(len(display))
    colors = ["#4C78A8"] * len(order) + ["black"]

    fig, axes = plt.subplots(1, 3, figsize=(14.6, 4.8), constrained_layout=True)
    for row, yi, color in zip(display.itertuples(index=False), y, colors):
        axes[0].errorbar(
            row.alpha_mean,
            yi,
            xerr=np.asarray(
                [[row.alpha_mean - row.alpha_q025], [row.alpha_q975 - row.alpha_mean]]
            ),
            fmt="*" if row.is_pooled else "o",
            color=color,
            capsize=2,
        )
        axes[1].errorbar(
            row.mean_log_gamma,
            yi,
            xerr=np.asarray(
                [[
                    row.mean_log_gamma - row.mean_log_gamma_q025
                ], [
                    row.mean_log_gamma_q975 - row.mean_log_gamma
                ]]
            ),
            fmt="*" if row.is_pooled else "o",
            color=color,
            capsize=2,
        )
    for ax in axes[:2]:
        ax.set_yticks(y)
        ax.set_yticklabels(display["cre"].replace({POOLED_LABEL: "Pooled"}))
        ax.invert_yaxis()
    axes[0].set_xlabel("Posterior alpha")
    axes[0].set_title("Nominal baseline alpha")
    axes[1].set_xlabel("Mean posterior log_gamma")
    axes[1].set_title("Actual mean across subclasses")

    corr = metadata["correlations"]["alpha_vs_mean_eta_plus_delta"]
    sns.regplot(
        data=ordinary,
        x="alpha_mean",
        y="mean_eta_plus_delta",
        ci=None,
        scatter_kws={"s": 30},
        line_kws={"linestyle": "--", "linewidth": 0.9},
        color="#D95F02",
        ax=axes[2],
    )
    for row in ordinary.itertuples(index=False):
        axes[2].annotate(
            row.cre,
            (row.alpha_mean, row.mean_eta_plus_delta),
            xytext=(4, 3),
            textcoords="offset points",
            fontsize=6,
        )
    axes[2].axhline(0, color="black", lw=0.7)
    axes[2].set_xlabel("Posterior mean alpha")
    axes[2].set_ylabel("Mean eta + delta across subclasses")
    axes[2].set_title(
        "Mean subclass effect is near zero\n"
        f"r={corr['pearson_r']:.3f}, "
        f"exact p={corr['pearson_exact_permutation_p']:.3g}"
    )
    fig.suptitle(
        "Negative-control activity decomposition: alpha matches mean log activity",
        fontsize=12,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.tables_dir.mkdir(parents=True, exist_ok=True)
    args.figures_dir.mkdir(parents=True, exist_ok=True)
    frame, metadata = load_data(args)
    table_path = args.tables_dir / f"{args.stem}.csv"
    figure_path = args.figures_dir / f"{args.stem}.pdf"
    decomposition_path = args.figures_dir / f"{args.stem}_decomposition.pdf"
    manifest_path = args.figures_dir / f"{args.stem}_manifest.json"
    frame.to_csv(table_path, index=False)
    plot_diagnostic(frame, metadata, figure_path)
    plot_decomposition(frame, metadata, decomposition_path)
    write_json(
        manifest_path,
        {
            **metadata,
            "outputs": {
                "table": str(table_path),
                "figure": str(figure_path),
                "decomposition_figure": str(decomposition_path),
                "manifest": str(manifest_path),
            },
        },
    )
    print(json.dumps(metadata["correlations"], indent=2))


if __name__ == "__main__":
    main()
