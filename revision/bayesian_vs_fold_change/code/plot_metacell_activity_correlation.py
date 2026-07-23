#!/usr/bin/env python3
"""Compare Bootstrap, joint Bayesian, and metacell Bayesian activity estimates."""

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

from analysis_utils import ANALYSIS_DIR, log, write_json
from plot_method_activity_correlation import axis_limit, finite_series, mean_log_beta_t7
from plot_section_reproducibility import discover_bayes_tag


METHODS = ("Bootstrap", "Joint Bayesian", "Metacell Bayesian")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bootstrap-dir", type=Path, default=ANALYSIS_DIR / "results" / "bootstrap"
    )
    parser.add_argument(
        "--joint-bayesian-dir",
        type=Path,
        default=ANALYSIS_DIR / "results" / "ablation" / "bayesian_joint",
    )
    parser.add_argument(
        "--metacell-bayesian-dir",
        type=Path,
        default=ANALYSIS_DIR
        / "results"
        / "ablation"
        / "bayesian_bootstrap_metacells_size100_number100",
    )
    parser.add_argument(
        "--figures-dir", type=Path, default=ANALYSIS_DIR / "results" / "figures"
    )
    parser.add_argument("--bootstrap-log-chunk-size", type=int, default=250)
    parser.add_argument(
        "--stem", default="method_activity_correlation_metacell_joint_bootstrap"
    )
    return parser.parse_args()


def read_blacklist(root: Path) -> set[str]:
    path = root / "cre_blacklist.csv"
    if not path.exists():
        return set()
    return set(pd.read_csv(path).iloc[:, 0].astype(str))


def load_bootstrap_mean_log_activity(root: Path, chunk_size: int) -> pd.DataFrame:
    axes = json.loads((root / "bootstrap_axes.json").read_text())
    groups = pd.Index(axes["subclasses"], dtype=str)
    cres = pd.Index(axes["cres"], dtype=str)
    activity_array = np.load(root / "celltype_activity_array.npy", mmap_mode="r")
    n_boot, n_groups, n_cres = activity_array.shape
    if n_groups != len(groups) or n_cres != len(cres):
        raise ValueError(
            f"{root} axis mismatch: array={activity_array.shape}, "
            f"groups={len(groups)}, cres={len(cres)}"
        )

    log_sum = np.zeros((n_groups, n_cres), dtype=np.float64)
    log_count = np.zeros((n_groups, n_cres), dtype=np.float64)
    for start in range(0, n_boot, chunk_size):
        chunk = activity_array[start : start + chunk_size]
        with np.errstate(divide="ignore", invalid="ignore"):
            logged = np.log(chunk.astype(np.float64, copy=False))
        finite = np.isfinite(logged)
        log_sum += np.where(finite, logged, 0.0).sum(axis=0)
        log_count += finite.sum(axis=0)
    mean_log = np.divide(
        log_sum,
        log_count,
        out=np.full((n_groups, n_cres), np.nan, dtype=np.float64),
        where=log_count > 0,
    )
    return pd.DataFrame(mean_log, index=groups, columns=cres)


def load_corrected_bayesian_activity(root: Path) -> tuple[pd.DataFrame, float, str]:
    tag = discover_bayes_tag(root)
    with np.load(root / f"{tag}_posterior_samples.npz", allow_pickle=True) as posterior:
        log_gamma = posterior["log_gamma"].astype(np.float32)
        groups = pd.Index(posterior["group_names"].astype(str), dtype=str)
        cres = pd.Index(posterior["cre_names"].astype(str), dtype=str)
    correction = mean_log_beta_t7(root)
    activity = pd.DataFrame(
        log_gamma.mean(axis=0, dtype=np.float64) - correction,
        index=groups,
        columns=cres,
    )
    return activity, correction, tag


def combined_axes(matrices: dict[str, pd.DataFrame]) -> tuple[pd.Index, pd.Index]:
    rows: pd.Index | None = None
    cols: pd.Index | None = None
    for matrix in matrices.values():
        index = pd.Index(matrix.index.astype(str))
        columns = pd.Index(matrix.columns.astype(str))
        rows = index if rows is None else rows.union(index)
        cols = columns if cols is None else cols.union(columns)
    if rows is None or cols is None:
        raise ValueError("no activity matrices were loaded")
    return rows, cols


def stack_methods(matrices: dict[str, pd.DataFrame]) -> pd.DataFrame:
    series = []
    for method in METHODS:
        matrix = matrices[method]
        index = pd.MultiIndex.from_product(
            [matrix.index.astype(str), matrix.columns.astype(str)],
            names=["group", "cre"],
        )
        series.append(
            pd.Series(matrix.to_numpy(float).ravel(), index=index, name=method)
        )
    return pd.concat(series, axis=1)


def correlation_row(x_method: str, y_method: str, wide: pd.DataFrame) -> dict:
    pair = wide[[x_method, y_method]].replace([np.inf, -np.inf], np.nan).dropna()
    return {
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


def plot_scatter_matrix(wide: pd.DataFrame, output: Path) -> list[dict]:
    sns.set_theme(context="paper", style="white")
    limits = {method: axis_limit(wide[method]) for method in METHODS}
    fig, axes = plt.subplots(
        3,
        3,
        figsize=(10.8, 10.8),
        constrained_layout=True,
        squeeze=False,
    )
    correlations = []
    for row, y_method in enumerate(METHODS):
        for col, x_method in enumerate(METHODS):
            ax = axes[row, col]
            ax.set_xlim(limits[x_method])
            if row == col:
                values = finite_series(wide[x_method])
                sns.histplot(values, bins=80, color="#4c78a8", edgecolor=None, ax=ax)
                ax.text(
                    0.03,
                    0.95,
                    f"n={len(values):,}",
                    transform=ax.transAxes,
                    ha="left",
                    va="top",
                    fontsize=8,
                )
            else:
                stats = correlation_row(x_method, y_method, wide)
                correlations.append(stats)
                pair = wide[[x_method, y_method]].replace(
                    [np.inf, -np.inf], np.nan
                ).dropna()
                ax.scatter(
                    pair[x_method],
                    pair[y_method],
                    s=1.6,
                    alpha=0.08,
                    linewidths=0,
                    color="#2f6f8f",
                )
                ax.set_ylim(limits[y_method])
                lo = max(limits[x_method][0], limits[y_method][0])
                hi = min(limits[x_method][1], limits[y_method][1])
                if lo < hi:
                    ax.plot([lo, hi], [lo, hi], color="black", linewidth=0.7, alpha=0.6)
                ax.text(
                    0.03,
                    0.97,
                    "r={pearson:.3f}\nrho={spearman:.3f}\nn={n_pairs:,}".format(
                        **stats
                    ),
                    transform=ax.transAxes,
                    ha="left",
                    va="top",
                    fontsize=8,
                    bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
                )
            if row == len(METHODS) - 1:
                ax.set_xlabel(x_method)
            else:
                ax.set_xlabel("")
                ax.set_xticklabels([])
            if col == 0:
                ax.set_ylabel(y_method)
            else:
                ax.set_ylabel("")
                ax.set_yticklabels([])
    fig.suptitle(
        "Activity correlations: Bootstrap, joint Bayesian, metacell Bayesian\n"
        "Bayesian values use log_gamma - E[log(beta_t7)]; black line is y = x.",
        fontsize=12,
    )
    sns.despine(fig=fig)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    return correlations


def main() -> None:
    args = parse_args()
    args.figures_dir.mkdir(parents=True, exist_ok=True)

    log("[metacell comparison] loading Bootstrap mean log activity")
    bootstrap = load_bootstrap_mean_log_activity(
        args.bootstrap_dir, args.bootstrap_log_chunk_size
    )
    log("[metacell comparison] loading joint Bayesian activity")
    joint, joint_correction, joint_tag = load_corrected_bayesian_activity(
        args.joint_bayesian_dir
    )
    log("[metacell comparison] loading metacell Bayesian activity")
    metacell, metacell_correction, metacell_tag = load_corrected_bayesian_activity(
        args.metacell_bayesian_dir
    )

    matrices = {
        "Bootstrap": bootstrap,
        "Joint Bayesian": joint,
        "Metacell Bayesian": metacell,
    }
    rows, columns = combined_axes(matrices)
    blacklist_sources = {
        "Bootstrap": sorted(read_blacklist(args.bootstrap_dir)),
        "Joint Bayesian": sorted(read_blacklist(args.joint_bayesian_dir)),
        "Metacell Bayesian": sorted(read_blacklist(args.metacell_bayesian_dir)),
    }
    blacklist = set().union(*(set(values) for values in blacklist_sources.values()))
    columns = pd.Index([cre for cre in columns.astype(str) if cre not in blacklist])
    aligned = {
        method: matrix.reindex(index=rows, columns=columns)
        for method, matrix in matrices.items()
    }
    wide = stack_methods(aligned)
    output = args.figures_dir / f"{args.stem}.pdf"
    correlations = plot_scatter_matrix(wide, output)
    corr_path = args.figures_dir / f"{args.stem}_correlations.csv"
    pd.DataFrame(correlations).to_csv(corr_path, index=False)
    manifest = {
        "output": str(output),
        "correlations": str(corr_path),
        "methods": list(METHODS),
        "rows": int(len(rows)),
        "columns_after_blacklist": int(len(columns)),
        "blacklisted_cres_removed": int(len(blacklist)),
        "blacklist_sources": blacklist_sources,
        "activity_scale": {
            "Bootstrap": "mean over bootstrap samples of log(cCRE/T7)",
            "Joint Bayesian": "posterior mean log_gamma - mean_log_beta_t7",
            "Metacell Bayesian": "posterior mean log_gamma - mean_log_beta_t7",
        },
        "mean_log_beta_t7_corrections": {
            "Joint Bayesian": joint_correction,
            "Metacell Bayesian": metacell_correction,
        },
        "tags": {
            "Joint Bayesian": joint_tag,
            "Metacell Bayesian": metacell_tag,
        },
        "black_line": "identity line y = x, not a regression line",
        "bootstrap_dir": str(args.bootstrap_dir),
        "joint_bayesian_dir": str(args.joint_bayesian_dir),
        "metacell_bayesian_dir": str(args.metacell_bayesian_dir),
    }
    write_json(args.figures_dir / f"{args.stem}_manifest.json", manifest)
    log(f"[metacell comparison] wrote {output}")


if __name__ == "__main__":
    main()
