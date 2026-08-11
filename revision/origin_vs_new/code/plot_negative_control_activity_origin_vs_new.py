#!/usr/bin/env python3
"""Compare mean ordinary-negative-control activity between fitted experiments."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve()
ANALYSIS_DIR = HERE.parent.parent
REPO_ROOT = ANALYSIS_DIR.parents[1]
DEFAULT_ORIGIN_BAYES = (
    REPO_ROOT / "revision" / "bayesian_vs_fold_change" / "results" / "bayesian"
)
DEFAULT_NEW_BAYES = REPO_ROOT / "revision" / "Bayes_NewData" / "bayesian"
DEFAULT_COMPARISON_DIR = ANALYSIS_DIR / "results" / "comparison"
DEFAULT_SHARED_TESTS = (
    DEFAULT_COMPARISON_DIR
    / "tables"
    / "shared_pair_mean_plus_1sd_comparison_t7_ge50.csv.gz"
)
DEFAULT_STEM = "negative_control_mean_activity_origin_vs_new_scatter"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--origin-bayes", type=Path, default=DEFAULT_ORIGIN_BAYES)
    parser.add_argument("--new-bayes", type=Path, default=DEFAULT_NEW_BAYES)
    parser.add_argument("--shared-tests", type=Path, default=DEFAULT_SHARED_TESTS)
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=DEFAULT_COMPARISON_DIR / "figures",
    )
    parser.add_argument(
        "--tables-dir",
        type=Path,
        default=DEFAULT_COMPARISON_DIR / "tables",
    )
    parser.add_argument("--stem", default=DEFAULT_STEM)
    parser.add_argument(
        "--label-outliers",
        type=int,
        default=8,
        help="Number of cell types with the largest absolute changes to label.",
    )
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def read_one_column(path: Path) -> list[str]:
    values = pd.read_csv(path).iloc[:, 0].astype(str).tolist()
    return list(dict.fromkeys(values))


def posterior_path(bayes_dir: Path) -> Path:
    manifest = json.loads((bayes_dir / "run_manifest.json").read_text())
    return bayes_dir / f"{manifest['tag']}_posterior_samples.npz"


def mean_negative_control_activity(
    bayes_dir: Path,
) -> tuple[pd.Series, list[str], Path]:
    """Posterior mean log_gamma, averaged over the seven ordinary controls."""
    controls = read_one_column(bayes_dir / "negative_controls.csv")
    if len(controls) != 7:
        raise ValueError(
            f"Expected seven ordinary negative controls in {bayes_dir}; "
            f"found {len(controls)}"
        )
    path = posterior_path(bayes_dir)
    with np.load(path, allow_pickle=True) as posterior:
        required = {"log_gamma", "group_names", "cre_names"}
        missing = required.difference(posterior.files)
        if missing:
            raise ValueError(f"{path} is missing arrays: {sorted(missing)}")
        groups = pd.Index(posterior["group_names"].astype(str), dtype=str)
        cres = pd.Index(posterior["cre_names"].astype(str), dtype=str)
        missing_controls = [control for control in controls if control not in cres]
        if missing_controls:
            raise ValueError(f"{path} is missing controls: {missing_controls}")
        control_indices = cres.get_indexer(controls)
        log_gamma = posterior["log_gamma"]
        values = log_gamma[:, :, control_indices].mean(
            axis=(0, 2), dtype=np.float64
        )
        del log_gamma
    gc.collect()
    if groups.has_duplicates:
        raise ValueError(f"{path} has duplicate group names")
    return pd.Series(values, index=groups, name="mean_control_activity"), controls, path


def concordance_correlation(x: np.ndarray, y: np.ndarray) -> float:
    covariance = float(np.mean((x - x.mean()) * (y - y.mean())))
    denominator = float(x.var() + y.var() + (x.mean() - y.mean()) ** 2)
    return 2.0 * covariance / denominator if denominator > 0 else float("nan")


def comparison_table(
    origin: pd.Series,
    new: pd.Series,
    shared_t7_groups: set[str],
) -> pd.DataFrame:
    comparison = pd.concat(
        [
            origin.rename("origin_mean_negative_control_activity"),
            new.rename("new_mean_negative_control_activity"),
        ],
        axis=1,
        join="inner",
    ).rename_axis("group").reset_index()
    comparison["new_minus_origin"] = (
        comparison["new_mean_negative_control_activity"]
        - comparison["origin_mean_negative_control_activity"]
    )
    comparison["in_shared_t7_ge50_test_universe"] = comparison["group"].isin(
        shared_t7_groups
    )
    return comparison.sort_values("new_minus_origin", ascending=False).reset_index(
        drop=True
    )


def plot_scatter(
    comparison: pd.DataFrame,
    output_pdf: Path,
    output_png: Path,
    label_outliers: int,
    dpi: int,
) -> dict:
    x = comparison["origin_mean_negative_control_activity"].to_numpy(float)
    y = comparison["new_mean_negative_control_activity"].to_numpy(float)
    delta = comparison["new_minus_origin"].to_numpy(float)
    shared_mask = comparison["in_shared_t7_ge50_test_universe"].to_numpy(bool)
    limit_low = float(min(x.min(), y.min()))
    limit_high = float(max(x.max(), y.max()))
    padding = max((limit_high - limit_low) * 0.06, 0.05)
    limits = (limit_low - padding, limit_high + padding)
    delta_limit = max(float(np.max(np.abs(delta))), 1e-6)
    norm = TwoSlopeNorm(vmin=-delta_limit, vcenter=0.0, vmax=delta_limit)

    fig, ax = plt.subplots(figsize=(7.1, 6.3), constrained_layout=True)
    points = ax.scatter(
        x,
        y,
        c=delta,
        cmap="coolwarm",
        norm=norm,
        s=25,
        alpha=0.82,
        edgecolors="none",
        zorder=2,
    )
    if shared_mask.any():
        ax.scatter(
            x[shared_mask],
            y[shared_mask],
            facecolors="none",
            edgecolors="black",
            s=48,
            linewidths=0.65,
            label="Cell type in shared T7≥50 test universe",
            zorder=3,
        )
    ax.plot(limits, limits, linestyle="--", color="0.3", linewidth=1.0, zorder=1)
    ax.set_xlim(limits)
    ax.set_ylim(limits)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Original run: mean negative-control activity")
    ax.set_ylabel("New low-dose run: mean negative-control activity")

    pearson = float(pd.Series(x).corr(pd.Series(y), method="pearson"))
    spearman = float(pd.Series(x).corr(pd.Series(y), method="spearman"))
    ccc = float(concordance_correlation(x, y))
    mean_delta = float(delta.mean())
    median_delta = float(np.median(delta))
    fraction_higher = float(np.mean(delta > 0))
    shared_t7_delta = delta[shared_mask]
    ax.set_title(
        "Mean activity of the same seven negative controls by cell type\n"
        f"n={len(comparison)}; Pearson r={pearson:.3f}; "
        f"mean Δ(new−origin)={mean_delta:+.3f}"
    )
    ax.text(
        0.02,
        0.98,
        f"Spearman ρ={spearman:.3f}\nCCC={ccc:.3f}\n"
        f"{fraction_higher:.1%} higher in new",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        bbox={"facecolor": "white", "edgecolor": "0.75", "alpha": 0.85},
    )
    if shared_mask.any():
        ax.legend(loc="lower right", fontsize=7, frameon=True)
    colorbar = fig.colorbar(points, ax=ax, pad=0.02)
    colorbar.set_label("New − original mean negative-control activity")

    n_labels = min(max(label_outliers, 0), len(comparison))
    label_indices = np.argsort(np.abs(delta))[-n_labels:][::-1]
    labels = [
        ax.text(
            x[index],
            y[index],
            comparison.iloc[index]["group"],
            fontsize=5.8,
            zorder=4,
        )
        for index in label_indices
    ]
    if labels:
        try:
            from adjustText import adjust_text

            adjust_text(
                labels,
                x=x,
                y=y,
                ax=ax,
                expand=(1.08, 1.18),
                force_text=(0.35, 0.55),
                arrowprops={"arrowstyle": "-", "color": "0.45", "lw": 0.4},
            )
        except ImportError:  # pragma: no cover - fallback for minimal environments
            for label_number, (label, index) in enumerate(zip(labels, label_indices)):
                label.set_position(
                    (
                        x[index] + (0.03 if label_number % 2 == 0 else -0.03),
                        y[index] + 0.03 * (1 + label_number % 3),
                    )
                )
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(color="0.9", linewidth=0.5)
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_pdf, bbox_inches="tight")
    fig.savefig(output_png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return {
        "n_shared_cell_types": int(len(comparison)),
        "n_shared_t7_ge50_cell_types": int(shared_mask.sum()),
        "pearson_r": pearson,
        "spearman_rho": spearman,
        "concordance_correlation_coefficient": ccc,
        "origin_mean": float(x.mean()),
        "new_mean": float(y.mean()),
        "mean_new_minus_origin": mean_delta,
        "median_new_minus_origin": median_delta,
        "fraction_higher_in_new": fraction_higher,
        "delta_min": float(delta.min()),
        "delta_max": float(delta.max()),
        "shared_t7_ge50_subset": {
            "n_cell_types": int(shared_mask.sum()),
            "mean_new_minus_origin": float(shared_t7_delta.mean())
            if shared_t7_delta.size
            else None,
            "median_new_minus_origin": float(np.median(shared_t7_delta))
            if shared_t7_delta.size
            else None,
            "fraction_higher_in_new": float(np.mean(shared_t7_delta > 0))
            if shared_t7_delta.size
            else None,
        },
    }


def main() -> None:
    args = parse_args()
    origin, origin_controls, origin_posterior = mean_negative_control_activity(
        args.origin_bayes
    )
    new, new_controls, new_posterior = mean_negative_control_activity(args.new_bayes)
    if origin_controls != new_controls:
        raise ValueError(
            "Ordinary negative-control lists differ between experiments: "
            f"origin={origin_controls}, new={new_controls}"
        )

    shared_t7_groups: set[str] = set()
    if args.shared_tests.exists():
        shared_t7_groups = set(
            pd.read_csv(args.shared_tests, usecols=["group"])["group"].astype(str)
        )
    comparison = comparison_table(origin, new, shared_t7_groups)
    origin_only = sorted(origin.index.difference(new.index).astype(str))
    new_only = sorted(new.index.difference(origin.index).astype(str))

    args.figures_dir.mkdir(parents=True, exist_ok=True)
    args.tables_dir.mkdir(parents=True, exist_ok=True)
    output_pdf = args.figures_dir / f"{args.stem}.pdf"
    output_png = args.figures_dir / f"{args.stem}.png"
    output_table = args.tables_dir / f"{args.stem}.csv"
    output_manifest = args.tables_dir / f"{args.stem}_manifest.json"
    summary = plot_scatter(
        comparison,
        output_pdf,
        output_png,
        args.label_outliers,
        args.dpi,
    )
    comparison.to_csv(output_table, index=False)
    manifest = {
        "definition": (
            "for each fitted cell type, posterior mean log_gamma averaged across "
            "posterior draws and the same seven ordinary negative-control cCREs"
        ),
        "negative_controls": origin_controls,
        "inputs": {
            "origin_posterior": str(origin_posterior.resolve()),
            "new_posterior": str(new_posterior.resolve()),
            "shared_t7_test_table": str(args.shared_tests.resolve())
            if args.shared_tests.exists()
            else None,
        },
        "cell_type_matching": {
            "shared": int(len(comparison)),
            "origin_only": origin_only,
            "new_only": new_only,
        },
        "summary": summary,
        "outputs": {
            "pdf": str(output_pdf.resolve()),
            "png": str(output_png.resolve()),
            "values": str(output_table.resolve()),
        },
    }
    output_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
