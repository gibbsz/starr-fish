#!/usr/bin/env python3
"""Plot activity correlations for bootstrap and joint dropout-prior runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

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

from analysis_utils import (
    ANALYSIS_DIR,
    DEFAULT_H5AD,
    FIGURES_WORK,
    OLD_DATA_BOOTSTRAP,
    ablation_root,
    log,
    write_json,
)
from plot_method_activity_heatmap import (
    combined_axes,
    read_cre_blacklist,
    t7_pair_totals,
    trim_empty_axes,
)
from plot_section_reproducibility import bayesian_base, bootstrap_base


METHOD_DIRS = {
    "Joint Bayes no dropout": (
        ablation_root("bayesian_joint")
    ),
    "Joint Bayes dropout default Beta(1,9)": ablation_root("bayesian_joint_dropout"),
    "Joint Bayes dropout moderate Beta(2,5)": ANALYSIS_DIR
    / "results"
    / "ablation"
    / "bayesian_joint_dropout_moderate",
    "Joint Bayes dropout high Beta(5,5)": ANALYSIS_DIR
    / "results"
    / "ablation"
    / "bayesian_joint_dropout_high",
    "Joint Bayes dropout strong high Beta(8,2)": ANALYSIS_DIR
    / "results"
    / "ablation"
    / "bayesian_joint_dropout_strongly_high",
}

PLOT_LABELS = {
    "Bootstrap": "Bootstrap",
    "Joint Bayes no dropout": "Joint\nno dropout",
    "Joint Bayes dropout default Beta(1,9)": "Joint dropout\nBeta(1,9)",
    "Joint Bayes dropout moderate Beta(2,5)": "Joint dropout\nBeta(2,5)",
    "Joint Bayes dropout high Beta(5,5)": "Joint dropout\nBeta(5,5)",
    "Joint Bayes dropout strong high Beta(8,2)": "Joint dropout\nBeta(8,2)",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bootstrap-dir", type=Path, default=OLD_DATA_BOOTSTRAP
    )
    parser.add_argument(
        "--figures-dir", type=Path, default=FIGURES_WORK
    )
    parser.add_argument("--h5ad", type=Path, default=DEFAULT_H5AD)
    parser.add_argument("--bootstrap-log-chunk-size", type=int, default=250)
    parser.add_argument("--t7-threshold", type=float, default=100.0)
    parser.add_argument("--stem", default="method_activity_correlation_dropout_prior_5x5")
    return parser.parse_args()


def discover_tag(root: Path) -> str:
    return json.loads((root / "run_manifest.json").read_text())["tag"]


def mean_log_beta_t7(root: Path) -> float:
    tag = discover_tag(root)
    with np.load(root / f"{tag}_scalar_samples.npz", allow_pickle=True) as samples:
        beta_t7 = np.asarray(samples["beta_t7"], dtype=float).reshape(-1)
    return float(np.log(beta_t7).mean())


def finite_series(values: pd.Series) -> pd.Series:
    values = pd.to_numeric(values, errors="coerce")
    return values[np.isfinite(values.to_numpy(float))]


def axis_limit(values: pd.Series) -> tuple[float, float]:
    finite = finite_series(values).to_numpy(float)
    if finite.size == 0:
        return 0.0, 1.0
    lo, hi = np.percentile(finite, [0.5, 99.5])
    if not np.isfinite(lo) or not np.isfinite(hi):
        return 0.0, 1.0
    if lo == hi:
        width = max(abs(lo) * 0.05, 0.5)
        return float(lo - width), float(hi + width)
    pad = (hi - lo) * 0.03
    return float(lo - pad), float(hi + pad)


def stack_methods(matrices: dict[str, pd.DataFrame], methods: list[str]) -> pd.DataFrame:
    series = []
    for method in methods:
        matrix = matrices[method]
        index = pd.MultiIndex.from_product(
            [matrix.index.astype(str), matrix.columns.astype(str)],
            names=["group", "cre"],
        )
        series.append(
            pd.Series(
                matrix.to_numpy(float).ravel(),
                index=index,
                name=method,
            )
        )
    return pd.concat(series, axis=1)


def load_activity(args: argparse.Namespace) -> dict[str, pd.DataFrame]:
    holder = SimpleNamespace(
        activity_calibration="none",
        bootstrap_log_chunk_size=args.bootstrap_log_chunk_size,
    )
    activity = {"Bootstrap": bootstrap_base(args.bootstrap_dir, holder)[0]}
    for label, root in METHOD_DIRS.items():
        if not (root / "run_manifest.json").exists():
            raise FileNotFoundError(f"missing completed run: {root}")
        activity[label] = bayesian_base(root, holder)[0] - mean_log_beta_t7(root)
    return activity


def load_prior_summary() -> dict[str, dict]:
    summary = {}
    for label, root in METHOD_DIRS.items():
        manifest = json.loads((root / "run_manifest.json").read_text())
        config = manifest.get("config", {})
        prior = dict(manifest.get("dropout_prior") or config.get("dropout_prior", {}))
        prior["dropout_model"] = config.get("dropout_model")
        tag = discover_tag(root)
        with np.load(root / f"{tag}_scalar_samples.npz", allow_pickle=True) as samples:
            if "p_drop_t7" in samples.files:
                prior["p_drop_t7_posterior_mean"] = float(
                    np.asarray(samples["p_drop_t7"], dtype=float).mean()
                )
            if "p_drop_cre" in samples.files:
                prior["p_drop_cre_posterior_mean"] = float(
                    np.asarray(samples["p_drop_cre"], dtype=float).mean()
                )
            prior["mean_log_beta_t7"] = float(
                np.log(np.asarray(samples["beta_t7"], dtype=float).reshape(-1)).mean()
            )
        summary[label] = prior
    return summary


def blacklist_union(args: argparse.Namespace) -> set[str]:
    blacklist = set(read_cre_blacklist(args.bootstrap_dir))
    for root in METHOD_DIRS.values():
        blacklist.update(read_cre_blacklist(root))
    return blacklist


def prepare_matrices(
    args: argparse.Namespace,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, list[str], int]:
    raw = load_activity(args)
    rows, columns = combined_axes(raw)
    blacklist = blacklist_union(args)
    columns = pd.Index(
        [cre for cre in columns.astype(str) if cre not in blacklist],
        dtype=str,
    )
    matrices = {
        method: matrix.reindex(index=rows, columns=columns)
        for method, matrix in raw.items()
    }
    pair_t7 = t7_pair_totals(args.h5ad, rows, columns)
    return matrices, pair_t7, list(matrices), len(blacklist)


def correlation_stats(
    variant: str,
    x_method: str,
    y_method: str,
    wide: pd.DataFrame,
) -> dict:
    pair = wide[[x_method, y_method]].replace([np.inf, -np.inf], np.nan).dropna()
    return {
        "variant": variant,
        "x_method": x_method,
        "y_method": y_method,
        "n_pairs": int(len(pair)),
        "pearson": float(pair[x_method].corr(pair[y_method], method="pearson"))
        if len(pair) > 1
        else np.nan,
        "spearman": float(pair[x_method].corr(pair[y_method], method="spearman"))
        if len(pair) > 1
        else np.nan,
    }


def plot_matrix(
    wide: pd.DataFrame,
    methods: list[str],
    variant: str,
    filter_label: str,
    output: Path,
) -> list[dict]:
    sns.set_theme(context="paper", style="white")
    limits = {method: axis_limit(wide[method]) for method in methods}
    cell_size = 2.9
    fig, axes = plt.subplots(
        len(methods),
        len(methods),
        figsize=(cell_size * len(methods), cell_size * len(methods)),
        constrained_layout=True,
        squeeze=False,
    )
    correlations = []
    for row, y_method in enumerate(methods):
        for col, x_method in enumerate(methods):
            ax = axes[row, col]
            ax.set_xlim(limits[x_method])
            if row == col:
                values = finite_series(wide[x_method])
                sns.histplot(
                    values,
                    bins=80,
                    color="#4c78a8",
                    edgecolor=None,
                    ax=ax,
                )
                ax.text(
                    0.03,
                    0.95,
                    f"n={len(values):,}",
                    transform=ax.transAxes,
                    ha="left",
                    va="top",
                    fontsize=7,
                )
            else:
                stats = correlation_stats(variant, x_method, y_method, wide)
                correlations.append(stats)
                pair = (
                    wide[[x_method, y_method]]
                    .replace([np.inf, -np.inf], np.nan)
                    .dropna()
                )
                ax.scatter(
                    pair[x_method],
                    pair[y_method],
                    s=1.2,
                    alpha=0.06,
                    linewidths=0,
                    color="#2f6f8f",
                    rasterized=True,
                )
                ax.set_ylim(limits[y_method])
                lo = max(limits[x_method][0], limits[y_method][0])
                hi = min(limits[x_method][1], limits[y_method][1])
                if lo < hi:
                    ax.plot([lo, hi], [lo, hi], color="black", linewidth=0.5, alpha=0.5)
                ax.text(
                    0.03,
                    0.97,
                    "r={pearson:.3f}\nrho={spearman:.3f}\nn={n_pairs:,}".format(
                        **stats
                    ),
                    transform=ax.transAxes,
                    ha="left",
                    va="top",
                    fontsize=6.5,
                    bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
                )
            if row == len(methods) - 1:
                ax.set_xlabel(PLOT_LABELS[x_method], fontsize=8)
            else:
                ax.set_xlabel("")
                ax.set_xticklabels([])
            if col == 0:
                ax.set_ylabel(PLOT_LABELS[y_method], fontsize=8)
            else:
                ax.set_ylabel("")
                ax.set_yticklabels([])
    fig.suptitle(
        "Bootstrap, joint no-dropout, and joint Bayesian dropout-prior sensitivity\n"
        f"{filter_label}; Bayesian values use log_gamma - E[log(beta_t7)]; "
        "black line is y = x",
        fontsize=13,
    )
    sns.despine(fig=fig)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    return correlations


def plot_correlations(args: argparse.Namespace) -> dict:
    matrices, pair_t7, methods, n_blacklisted = prepare_matrices(args)
    variants = {
        "complete": {
            "filter_label": "complete, blacklist cCREs removed",
            "mask": pd.DataFrame(True, index=pair_t7.index, columns=pair_t7.columns),
        },
        "t7_gt100": {
            "filter_label": f"subclass-cCRE total T7 > {args.t7_threshold:g}",
            "mask": pair_t7.gt(args.t7_threshold),
        },
    }
    outputs = {}
    summaries = {}
    correlations = []
    for variant, spec in variants.items():
        masked = {
            method: matrix.where(spec["mask"])
            for method, matrix in matrices.items()
        }
        trimmed, rows, columns, present = trim_empty_axes(masked)
        wide = stack_methods(trimmed, methods)
        output = args.figures_dir / f"{args.stem}_{variant}.pdf"
        correlations.extend(
            plot_matrix(wide, methods, variant, spec["filter_label"], output)
        )
        outputs[variant] = str(output)
        summaries[variant] = {
            "rows": int(len(rows)),
            "columns": int(len(columns)),
            "finite_pairs_any_method": int(present.to_numpy(bool).sum()),
            "passing_filter_pairs": int(spec["mask"].to_numpy(bool).sum()),
        }
    summary = {
        "methods": methods,
        "plot_labels": PLOT_LABELS,
        "outputs": outputs,
        "variants": summaries,
        "correlations": correlations,
        "blacklisted_cres_removed": int(n_blacklisted),
        "t7_threshold": args.t7_threshold,
        "bayesian_method": "joint",
        "matrix_size": [len(methods), len(methods)],
        "bayesian_activity_scale": "log_gamma - mean_log_beta_t7",
        "black_line": "identity line y = x, not a regression line",
        "dropout_prior_summary": load_prior_summary(),
    }
    write_json(args.figures_dir / f"{args.stem}_manifest.json", summary)
    return summary


def main() -> None:
    args = parse_args()
    args.figures_dir.mkdir(parents=True, exist_ok=True)
    summary = plot_correlations(args)
    log(
        "[dropout prior activity correlation] wrote "
        f"{len(summary['outputs'])} {len(summary['methods'])}x{len(summary['methods'])} "
        "scatter-matrix PDFs"
    )


if __name__ == "__main__":
    main()
