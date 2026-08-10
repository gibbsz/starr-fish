#!/usr/bin/env python3
"""Plot posterior mean cCRE activity against alpha for one cell type."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# The shared analysis layer (analysis_utils and the plot_* modules that other
# scripts import) stays in the parent code/ directory.
import sys as _sys
from pathlib import Path as _Path
_CODE_DIR = _Path(__file__).resolve().parent.parent
if str(_CODE_DIR) not in _sys.path:
    _sys.path.insert(0, str(_CODE_DIR))

from analysis_utils import ANALYSIS_DIR, DEFAULT_H5AD, FIGURES_WORK, write_json
from plot_method_activity_correlation import pair_count_totals, read_cre_blacklist


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bayes-dir",
        type=Path,
        default=ANALYSIS_DIR
        / "results"
        / "ablation"
        / "bayesian_joint_components",
    )
    parser.add_argument("--group", default="OB Eomes Ms4a15 Glut")
    parser.add_argument(
        "--figures-dir", type=Path, default=FIGURES_WORK
    )
    parser.add_argument(
        "--tables-dir", type=Path, default=ANALYSIS_DIR / "results" / "tables"
    )
    parser.add_argument("--h5ad", type=Path, default=DEFAULT_H5AD)
    parser.add_argument("--t7-threshold", type=float, default=50.0)
    parser.add_argument("--stem", default=None)
    return parser.parse_args()


def safe_token(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_").lower()


def posterior_interval(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        values.mean(axis=0, dtype=np.float64),
        np.percentile(values, 5, axis=0),
        np.percentile(values, 95, axis=0),
    )


def main() -> None:
    args = parse_args()
    args.figures_dir.mkdir(parents=True, exist_ok=True)
    args.tables_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((args.bayes_dir / "run_manifest.json").read_text())
    tag = str(manifest["tag"])
    posterior_path = args.bayes_dir / f"{tag}_posterior_samples.npz"

    with np.load(posterior_path, allow_pickle=True) as posterior:
        required = {"log_gamma", "alpha", "log_gamma_neg", "alpha_neg"}
        missing = required.difference(posterior.files)
        if missing:
            raise ValueError(f"Posterior is missing required sites: {sorted(missing)}")
        groups = posterior["group_names"].astype(str)
        cres = posterior["cre_names"].astype(str)
        matches = np.flatnonzero(groups == args.group)
        if len(matches) != 1:
            raise ValueError(
                f"Expected one exact match for {args.group!r}; found {len(matches)}"
            )
        group_idx = int(matches[0])
        activity_draws = posterior["log_gamma"][:, group_idx, :].astype(np.float32)
        alpha_draws = posterior["alpha"].astype(np.float32)
        negative_activity_draws = posterior["log_gamma_neg"][:, group_idx].astype(
            np.float32
        )
        negative_alpha_draws = posterior["alpha_neg"].astype(np.float32)

    negative_controls = set(
        pd.read_csv(args.bayes_dir / "negative_controls.csv").iloc[:, 0].astype(str)
    )
    is_negative = np.isin(cres, list(negative_controls))
    blacklist = read_cre_blacklist(args.bayes_dir)
    is_blacklisted = np.isin(cres, list(blacklist))
    pair_t7, _ = pair_count_totals(
        args.h5ad,
        pd.Index([args.group], dtype=str),
        pd.Index(cres, dtype=str),
    )
    target_t7 = pair_t7.loc[args.group].to_numpy(float)
    negative_t7_total = float(
        pair_t7.loc[args.group, pair_t7.columns.isin(negative_controls)].sum()
    )
    passes_t7_filter = (
        (target_t7 >= args.t7_threshold)
        & (negative_t7_total >= args.t7_threshold)
        & ~is_negative
        & ~is_blacklisted
    )
    if not passes_t7_filter.any():
        raise ValueError(
            f"No cCREs pass the exact T7 >= {args.t7_threshold:g} eligibility "
            f"mask for {args.group!r}"
        )

    activity_mean, activity_lo, activity_hi = posterior_interval(activity_draws)
    alpha_mean, alpha_lo, alpha_hi = posterior_interval(alpha_draws)
    residual_draws = activity_draws - alpha_draws
    residual_mean, residual_lo, residual_hi = posterior_interval(residual_draws)
    neg_activity_mean, neg_activity_lo, neg_activity_hi = posterior_interval(
        negative_activity_draws
    )
    neg_alpha_mean, neg_alpha_lo, neg_alpha_hi = posterior_interval(
        negative_alpha_draws
    )
    negative_residual_draws = negative_activity_draws - negative_alpha_draws
    neg_residual_mean, neg_residual_lo, neg_residual_hi = posterior_interval(
        negative_residual_draws
    )

    table = pd.DataFrame(
        {
            "group": args.group,
            "cre": cres,
            "is_negative_control": is_negative,
            "is_blacklisted": is_blacklisted,
            "target_t7_total": target_t7,
            "negative_control_t7_total": negative_t7_total,
            "passes_t7_filter": passes_t7_filter,
            "activity_mean": activity_mean,
            "activity_lo90": activity_lo,
            "activity_hi90": activity_hi,
            "alpha_mean": alpha_mean,
            "alpha_lo90": alpha_lo,
            "alpha_hi90": alpha_hi,
            "activity_minus_alpha_mean": residual_mean,
            "activity_minus_alpha_lo90": residual_lo,
            "activity_minus_alpha_hi90": residual_hi,
            "posterior_p_le_negative": (
                residual_draws <= negative_residual_draws[:, None]
            ).mean(axis=0, dtype=np.float64),
        }
    )

    threshold_token = f"{args.t7_threshold:g}".replace(".", "p")
    stem = args.stem or (
        f"joint_alpha_activity_{safe_token(args.group)}_t7_ge{threshold_token}"
    )
    table_path = args.tables_dir / f"{stem}.csv"
    table.to_csv(table_path, index=False)

    sns.set_theme(context="paper", style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 6.2), constrained_layout=True)
    ax, residual_ax = axes
    eligible = table.loc[table["passes_t7_filter"]].copy()
    ax.scatter(
        eligible["alpha_mean"],
        eligible["activity_mean"],
        s=24,
        color="#4477AA",
        alpha=0.62,
        linewidths=0,
        rasterized=True,
        label=f"T7 >= {args.t7_threshold:g} cCREs (n={len(eligible)})",
    )

    limits = np.concatenate(
        [
            eligible["alpha_mean"].to_numpy(float),
            eligible["activity_mean"].to_numpy(float),
            np.asarray([neg_alpha_mean, neg_activity_mean], dtype=float),
        ]
    )
    lo = float(np.nanmin(limits)) - 0.25
    hi = float(np.nanmax(limits)) + 0.25
    ax.plot([lo, hi], [lo, hi], color="0.35", linestyle="--", linewidth=1.0)
    ax.axvline(neg_alpha_mean, color="#CC6677", linestyle=":", linewidth=0.9)
    ax.axhline(neg_activity_mean, color="#CC6677", linestyle=":", linewidth=0.9)
    ax.errorbar(
        neg_alpha_mean,
        neg_activity_mean,
        xerr=np.asarray(
            [[neg_alpha_mean - neg_alpha_lo], [neg_alpha_hi - neg_alpha_mean]]
        ),
        yerr=np.asarray(
            [
                [neg_activity_mean - neg_activity_lo],
                [neg_activity_hi - neg_activity_mean],
            ]
        ),
        fmt="*",
        markersize=15,
        color="#AA3344",
        ecolor="#AA3344",
        elinewidth=1.2,
        capsize=3,
        markeredgecolor="white",
        markeredgewidth=0.7,
        zorder=5,
        label=f"shared negative controls (n={int(is_negative.sum())})",
    )
    ax.annotate(
        "negative controls",
        (neg_alpha_mean, neg_activity_mean),
        xytext=(8, -15),
        textcoords="offset points",
        fontsize=8,
        color="#882233",
    )

    rho = float(
        eligible[["alpha_mean", "activity_mean"]]
        .corr(method="spearman")
        .iloc[0, 1]
    )
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"Posterior mean $\alpha_j$ (cCRE baseline)")
    ax.set_ylabel(r"Posterior mean $\log\gamma_{s,j}$ (cell-type activity)")
    ax.set_title(
        f"Activity and alpha after T7 >= {args.t7_threshold:g} filter\n"
        rf"Spearman $\rho$={rho:.3f}"
    )
    ax.legend(frameon=False, loc="upper left")
    sns.despine(ax=ax)

    ranked = eligible.sort_values("activity_minus_alpha_mean").reset_index(drop=True)
    x = np.arange(len(ranked))
    residual_ax.vlines(
        x,
        ranked["activity_minus_alpha_lo90"],
        ranked["activity_minus_alpha_hi90"],
        color="#4477AA",
        alpha=0.35,
        linewidth=0.8,
    )
    residual_ax.scatter(
        x,
        ranked["activity_minus_alpha_mean"],
        s=23,
        color="#4477AA",
        alpha=0.8,
        linewidths=0,
        zorder=3,
    )
    residual_ax.axhspan(
        neg_residual_lo,
        neg_residual_hi,
        color="#CC6677",
        alpha=0.16,
        linewidth=0,
        label="negative-control 90% posterior interval",
    )
    residual_ax.axhline(
        neg_residual_mean,
        color="#AA3344",
        linewidth=1.4,
        label="negative-control posterior mean",
    )
    residual_ax.axhline(0.0, color="0.35", linestyle="--", linewidth=0.9)
    residual_ax.set_xlim(-1, len(ranked))
    residual_ax.set_xlabel("cCREs ranked by posterior mean residual")
    residual_ax.set_ylabel(r"Posterior $\log\gamma_{s,j} - \alpha_j$")
    residual_ax.set_title(
        r"Activity minus alpha: $\eta_{class,j} + \delta_{s,j}$" "\n"
        "points are posterior means; bars are 90% intervals"
    )
    if len(ranked) <= 35:
        residual_ax.set_xticks(x, ranked["cre"], rotation=90)
    else:
        step = max(1, int(np.ceil(len(ranked) / 20)))
        shown = x[::step]
        residual_ax.set_xticks(shown, ranked.loc[shown, "cre"], rotation=90)
    residual_ax.legend(frameon=False, loc="upper left", fontsize=8)
    sns.despine(ax=residual_ax)

    fig.suptitle(f"Joint model: {args.group}", fontsize=13)

    pdf_path = args.figures_dir / f"{stem}.pdf"
    png_path = args.figures_dir / f"{stem}.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight", dpi=220)
    plt.close(fig)

    write_json(
        args.figures_dir / f"{stem}_manifest.json",
        {
            "model": "Joint",
            "bayes_dir": str(args.bayes_dir),
            "posterior": str(posterior_path),
            "group": args.group,
            "t7_threshold": args.t7_threshold,
            "n_eligible_cres": int(passes_t7_filter.sum()),
            "negative_control_t7_total": negative_t7_total,
            "blacklist": sorted(blacklist),
            "negative_controls": sorted(negative_controls),
            "negative_control_posterior_mean": {
                "alpha": float(neg_alpha_mean),
                "activity": float(neg_activity_mean),
                "activity_minus_alpha": float(neg_residual_mean),
            },
            "outputs": {
                "pdf": str(pdf_path),
                "png": str(png_path),
                "table": str(table_path),
            },
        },
    )


if __name__ == "__main__":
    main()
