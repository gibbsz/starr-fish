#!/usr/bin/env python3
"""Compare old and new manuscript-style bootstrap activity and calls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

from compare_origin_vs_new import (
    bh_fdr,
    call_metrics,
    read_one_column,
    safe_correlations,
    write_json,
)


HERE = Path(__file__).resolve()
ANALYSIS_DIR = HERE.parent.parent
REPO_ROOT = ANALYSIS_DIR.parents[1]
ORIGIN_ANALYSIS = REPO_ROOT / "revision" / "bayesian_vs_fold_change"
NEW_RESULTS = REPO_ROOT / "revision" / "Bayes_NewData"

DEFAULT_ORIGIN_BOOTSTRAP = REPO_ROOT / "revision" / "Bootstrap_OldData"
DEFAULT_NEW_BOOTSTRAP = REPO_ROOT / "revision" / "Bootstrap_NewData"
DEFAULT_ORIGIN_TESTS = (
    ORIGIN_ANALYSIS
    / "results"
    / "tables"
    / "bootstrap_mean_negative_control_tests.csv.gz"
)
DEFAULT_NEW_TESTS = (
    NEW_RESULTS
    / "tables"
    / "new_bootstrap_mean_negative_control_tests_t7_ge50.csv.gz"
)
DEFAULT_OUTPUT_DIR = ANALYSIS_DIR / "results" / "comparison" / "bootstrap"

KEY = ["group", "cre"]
TEST_COLUMNS = [
    "target_t7_total",
    "negative_control_t7_total",
    "activity_mean",
    "effect_vs_mean_control_mean",
    "posterior_probability_above_mean_control",
    "p_right",
    "n_testable_bootstraps",
    "mean_controls_used_per_reference",
    "n_bootstraps_with_control_reference",
    "q_right",
    "significant_q",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--origin-bootstrap", type=Path, default=DEFAULT_ORIGIN_BOOTSTRAP
    )
    parser.add_argument("--new-bootstrap", type=Path, default=DEFAULT_NEW_BOOTSTRAP)
    parser.add_argument("--origin-tests", type=Path, default=DEFAULT_ORIGIN_TESTS)
    parser.add_argument("--new-tests", type=Path, default=DEFAULT_NEW_TESTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--t7-threshold", type=float, default=50.0)
    parser.add_argument("--q-cutoff", type=float, default=0.05)
    parser.add_argument(
        "--bootstrap-chunk-size",
        type=int,
        default=128,
        help="Bootstrap replicates per chunk when summarizing individual controls.",
    )
    return parser.parse_args()


def read_manifest(directory: Path) -> dict:
    return json.loads((directory / "run_manifest.json").read_text())


def validate_runs(
    origin_dir: Path, new_dir: Path
) -> tuple[dict, dict, list[str], list[str]]:
    origin = read_manifest(origin_dir)
    new = read_manifest(new_dir)
    required_config = {
        "bootstrap_number": 10_000,
        "bootstrap_to_fixed_pct": 1,
        "bootstrap_to_fixed_sample_size": None,
        "filter_by_cell_t7": None,
        "load_stored": False,
        "log_transform": False,
        "normalize_by_cell_rna": False,
        "normalize_by_cell_t7": False,
        "normalize_by_cell_volume": False,
        "normalize_by_celltype_rna": False,
        "normalize_by_celltype_t7": True,
        "normalize_by_celltype_volume": False,
        "normalize_by_libsize": False,
        "normalize_by_negative_control": False,
    }
    for label, manifest in (("origin", origin), ("new", new)):
        mismatches = {
            key: {"expected": value, "observed": manifest["config"].get(key)}
            for key, value in required_config.items()
            if manifest["config"].get(key) != value
        }
        if manifest.get("min_detected_cells") != 5:
            mismatches["min_detected_cells"] = {
                "expected": 5,
                "observed": manifest.get("min_detected_cells"),
            }
        if manifest.get("seed") != 0:
            mismatches["seed"] = {"expected": 0, "observed": manifest.get("seed")}
        if mismatches:
            raise ValueError(f"{label} bootstrap configuration mismatch: {mismatches}")

    origin_blacklist = read_one_column(origin_dir / "cre_blacklist.csv")
    new_blacklist = read_one_column(new_dir / "cre_blacklist.csv")
    if origin_blacklist != new_blacklist:
        raise ValueError("origin and new bootstrap blacklists differ")
    origin_controls = read_one_column(origin_dir / "negative_controls.csv")
    new_controls = read_one_column(new_dir / "negative_controls.csv")
    if origin_controls != new_controls:
        raise ValueError("origin and new bootstrap negative controls differ")
    return origin, new, origin_blacklist, origin_controls


def centered_negative_control_activity(
    bootstrap_dir: Path,
    controls: list[str],
    chunk_size: int,
) -> pd.DataFrame:
    """Mean bootstrap log activity relative to the replicate-wise control mean."""
    if chunk_size <= 0:
        raise ValueError("--bootstrap-chunk-size must be positive")
    axes = json.loads((bootstrap_dir / "bootstrap_axes.json").read_text())
    groups = pd.Index(axes["subclasses"], dtype=str)
    cres = pd.Index(axes["cres"], dtype=str)
    missing_controls = [control for control in controls if control not in cres]
    if missing_controls:
        raise ValueError(
            f"{bootstrap_dir} bootstrap axes are missing controls: {missing_controls}"
        )
    control_indices = cres.get_indexer(controls)
    activity = np.load(
        bootstrap_dir / "celltype_activity_array.npy", mmap_mode="r"
    )
    if activity.shape[1:] != (len(groups), len(cres)):
        raise ValueError(
            f"{bootstrap_dir} activity array shape {activity.shape} does not match axes"
        )

    effect_sum = np.zeros((len(groups), len(controls)), dtype=np.float64)
    effect_count = np.zeros((len(groups), len(controls)), dtype=np.int64)
    for start in range(0, activity.shape[0], chunk_size):
        stop = min(start + chunk_size, activity.shape[0])
        with np.errstate(divide="ignore", invalid="ignore"):
            logged = np.log(
                activity[start:stop, :, control_indices].astype(
                    np.float64, copy=False
                )
            )
        finite = np.isfinite(logged)
        reference_count = finite.sum(axis=2)
        reference_sum = np.where(finite, logged, 0.0).sum(axis=2)
        reference = np.divide(
            reference_sum,
            reference_count,
            out=np.full(reference_sum.shape, np.nan),
            where=reference_count > 0,
        )
        valid = finite & np.isfinite(reference)[:, :, None]
        effect = logged - reference[:, :, None]
        effect_sum += np.where(valid, effect, 0.0).sum(axis=0)
        effect_count += valid.sum(axis=0)

    effect_mean = np.divide(
        effect_sum,
        effect_count,
        out=np.full(effect_sum.shape, np.nan),
        where=effect_count > 0,
    )
    group_grid, cre_grid = np.meshgrid(
        groups.to_numpy(str), np.asarray(controls, dtype=str), indexing="ij"
    )
    return pd.DataFrame(
        {
            "group": group_grid.ravel(),
            "cre": cre_grid.ravel(),
            "centered_log_activity_mean": effect_mean.ravel(),
            "n_testable_bootstraps": effect_count.ravel(),
        }
    )


def read_tests(
    path: Path,
    *,
    threshold: float,
    q_cutoff: float,
    blacklist: list[str],
    controls: list[str],
) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = set(KEY + TEST_COLUMNS + ["t7_threshold"])
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    if frame.duplicated(KEY).any():
        raise ValueError(f"{path} contains duplicate subclass-cCRE pairs")
    thresholds = frame["t7_threshold"].dropna().unique()
    if len(thresholds) != 1 or not np.isclose(thresholds[0], threshold):
        raise ValueError(f"{path} does not contain only T7 >= {threshold:g}")
    if (frame["target_t7_total"] < threshold).any():
        raise ValueError(f"{path} contains target T7 below {threshold:g}")
    if (frame["negative_control_t7_total"] < threshold).any():
        raise ValueError(f"{path} contains control T7 below {threshold:g}")
    forbidden = set(blacklist) | set(controls)
    if set(frame["cre"].astype(str)) & forbidden:
        raise ValueError(f"{path} contains blacklisted or negative-control targets")
    if not frame["significant_q"].astype(bool).equals(frame["q_right"].le(q_cutoff)):
        raise ValueError(f"{path} has inconsistent significant_q values")
    frame["group"] = frame["group"].astype(str)
    frame["cre"] = frame["cre"].astype(str)
    return frame


def prefixed(frame: pd.DataFrame, prefix: str) -> pd.DataFrame:
    return frame[KEY + TEST_COLUMNS].rename(
        columns={column: f"{prefix}_{column}" for column in TEST_COLUMNS}
    )


def plot_activity(
    overlap: pd.DataFrame,
    negative_control_activity: pd.DataFrame,
    figures_dir: Path,
    t7_threshold: float = 50,
) -> None:
    threshold_token = f"{t7_threshold:g}".replace(".", "p")
    x = overlap["origin_effect_vs_mean_control_mean"].to_numpy(float)
    y = overlap["new_effect_vs_mean_control_mean"].to_numpy(float)
    control_x = negative_control_activity[
        "origin_centered_log_activity_mean"
    ].to_numpy(float)
    control_y = negative_control_activity[
        "new_centered_log_activity_mean"
    ].to_numpy(float)
    pearson, spearman = safe_correlations(pd.Series(x), pd.Series(y))
    limits = np.nanquantile(np.concatenate([x, y]), [0.005, 0.995])
    if not np.isfinite(limits).all() or limits[0] == limits[1]:
        limits = np.asarray([-1.0, 1.0])
    finite_controls = np.isfinite(control_x) & np.isfinite(control_y)
    if finite_controls.any():
        limits[0] = min(
            float(limits[0]),
            float(control_x[finite_controls].min()),
            float(control_y[finite_controls].min()),
        )
        limits[1] = max(
            float(limits[1]),
            float(control_x[finite_controls].max()),
            float(control_y[finite_controls].max()),
        )
    padding = max(float(limits[1] - limits[0]) * 0.025, 0.05)
    limits = np.asarray([limits[0] - padding, limits[1] + padding])

    fig, ax = plt.subplots(figsize=(7.0, 6.3))
    ax.plot(
        limits,
        limits,
        linestyle="--",
        linewidth=1.2,
        color="black",
        zorder=1,
    )
    ax.scatter(
        x,
        y,
        s=15,
        alpha=0.20,
        color="#2F6F8F",
        edgecolors="none",
        rasterized=True,
        zorder=2,
        label=f"T7≥{t7_threshold:g} target pairs (n={len(overlap):,})",
    )

    finite_visible = (
        np.isfinite(x)
        & np.isfinite(y)
        & (x >= limits[0])
        & (x <= limits[1])
        & (y >= limits[0])
        & (y <= limits[1])
    )
    density_points = np.vstack([x[finite_visible], y[finite_visible]])
    if density_points.shape[1] >= 3:
        try:
            kde = gaussian_kde(density_points)
            grid_values = np.linspace(limits[0], limits[1], 180)
            grid_x, grid_y = np.meshgrid(grid_values, grid_values)
            grid_density = kde(
                np.vstack([grid_x.ravel(), grid_y.ravel()])
            ).reshape(grid_x.shape)
            point_density = kde(density_points)
            contour_levels = np.unique(
                np.quantile(point_density, [0.10, 0.25, 0.45, 0.65, 0.82, 0.93])
            )
            contour_levels = contour_levels[contour_levels > 0]
            if contour_levels.size:
                ax.contour(
                    grid_x,
                    grid_y,
                    grid_density,
                    levels=contour_levels,
                    colors="#0B4F6C",
                    linewidths=np.linspace(0.7, 1.4, contour_levels.size),
                    alpha=0.95,
                    zorder=3,
                )
        except np.linalg.LinAlgError:
            pass

    if finite_controls.any():
        ax.scatter(
            control_x[finite_controls],
            control_y[finite_controls],
            s=31,
            alpha=0.82,
            color="#D55E00",
            edgecolors="white",
            linewidths=0.35,
            rasterized=True,
            zorder=4,
            label=(
                f"Negative controls ({finite_controls.sum():,} finite of 7 cCREs × "
                f"{negative_control_activity['group'].nunique()} cell types)"
            ),
        )

    ax.set_xlim(limits)
    ax.set_ylim(limits)
    ax.set_xlabel("Original bootstrap centered log activity")
    ax.set_ylabel("New low-dose bootstrap centered log activity")
    ax.set_title(
        f"Bootstrap overlap filter: T7 ≥ {t7_threshold:g} in both datasets\n"
        f"n={len(overlap):,}; Pearson r={pearson:.3f}; Spearman ρ={spearman:.3f}"
    )
    ax.legend(loc="lower right", frameon=True, fontsize=8)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(
            figures_dir
            / f"overlap_t7_ge{threshold_token}_bootstrap_activity.{suffix}",
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(fig)


def plot_calls(overlap: pd.DataFrame, figures_dir: Path) -> None:
    metrics = call_metrics(
        overlap["origin_significant_overlap_q"],
        overlap["new_significant_overlap_q"],
    )
    values = [
        metrics["origin_only_significant"],
        metrics["both_significant"],
        metrics["new_only_significant"],
    ]
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.4))
    axes[0].bar(
        ["Original only", "Both", "New only"],
        values,
        color=["#4477AA", "#6E6E6E", "#CC6677"],
    )
    axes[0].set_ylabel("Significant subclass–cCRE pairs")
    axes[0].set_title("Bootstrap overlap universe, BH q ≤ 0.05")
    for index, value in enumerate(values):
        axes[0].text(index, value, f"{value:,}", ha="center", va="bottom")

    matrix = np.asarray(
        [
            [metrics["neither_significant"], metrics["new_only_significant"]],
            [metrics["origin_only_significant"], metrics["both_significant"]],
        ]
    )
    image = axes[1].imshow(matrix, cmap="Blues")
    axes[1].set_xticks([0, 1], ["New not sig.", "New sig."])
    axes[1].set_yticks([0, 1], ["Origin not sig.", "Origin sig."])
    axes[1].set_title(
        f"Call concordance={metrics['call_concordance']:.3f}\n"
        f"Significant-pair Jaccard={metrics['significant_jaccard']:.3f}"
    )
    for row in range(2):
        for column in range(2):
            axes[1].text(
                column,
                row,
                f"{matrix[row, column]:,}",
                ha="center",
                va="center",
                color="white" if matrix[row, column] > matrix.max() / 2 else "black",
            )
    fig.colorbar(image, ax=axes[1], fraction=0.046, pad=0.04)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(
            figures_dir / f"overlap_t7_ge50_bootstrap_calls.{suffix}", dpi=300
        )
    plt.close(fig)


def write_report(path: Path, summary: dict) -> None:
    stats = summary["statistics"]
    activity = summary["activity"]
    text = f"""# Original versus new low-dose bootstrap comparison

The primary universe contains {stats['overlap_t7_ge50_testable_pairs']:,}
subclass-cCRE pairs with target T7 >= 50, combined seven-control T7 >= 50,
and a finite 10,000-replicate bootstrap test in both datasets. BH was
recomputed within this identical overlap for each dataset.

Centered bootstrap log activity has Pearson
`r={activity['overlap_centered_activity_pearson']:.3f}` and Spearman
`rho={activity['overlap_centered_activity_spearman']:.3f}`. The mean
new-minus-original change is
{activity['overlap_mean_change_new_minus_origin']:.3f} log units.

| Overlap-filtered call status | Pairs |
|---|---:|
| Significant in both | {stats['both_significant']:,} |
| Original only | {stats['origin_only_significant']:,} |
| New low-dose only | {stats['new_only_significant']:,} |
| Significant in neither | {stats['neither_significant']:,} |

Call concordance is {stats['call_concordance']:.1%}; significant-pair
Jaccard is {stats['significant_jaccard']:.3f}. The original bootstrap calls
{stats['origin_significant_pairs']:,} overlap pairs significant and the new
bootstrap calls {stats['new_significant_pairs']:,}.

Complete outputs:

- `tables/overlap_t7_ge50_bootstrap_pair_comparison.csv.gz`
- `tables/overlap_t7_ge50_bootstrap_tests_long.csv.gz`
- `tables/per_subclass_bootstrap_comparison.csv`
- `tables/bootstrap_comparison_summary.json`
- `figures/overlap_t7_ge50_bootstrap_activity.pdf`
- `figures/overlap_t7_ge50_bootstrap_calls.pdf`
"""
    path.write_text(text)


def main() -> None:
    args = parse_args()
    tables_dir = args.output_dir / "tables"
    figures_dir = args.output_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    origin_manifest, new_manifest, blacklist, controls = validate_runs(
        args.origin_bootstrap, args.new_bootstrap
    )
    origin = read_tests(
        args.origin_tests,
        threshold=args.t7_threshold,
        q_cutoff=args.q_cutoff,
        blacklist=blacklist,
        controls=controls,
    )
    new = read_tests(
        args.new_tests,
        threshold=args.t7_threshold,
        q_cutoff=args.q_cutoff,
        blacklist=blacklist,
        controls=controls,
    )
    all_pairs = prefixed(origin, "origin").merge(
        prefixed(new, "new"),
        on=KEY,
        how="outer",
        validate="one_to_one",
        indicator="test_pair_scope",
    )
    all_pairs.to_csv(
        tables_dir / "all_bootstrap_eligible_pair_comparison.csv.gz", index=False
    )

    overlap = all_pairs.loc[all_pairs["test_pair_scope"].eq("both")].copy()
    overlap["origin_q_overlap"] = bh_fdr(overlap["origin_p_right"])
    overlap["new_q_overlap"] = bh_fdr(overlap["new_p_right"])
    overlap["origin_significant_overlap_q"] = overlap["origin_q_overlap"].le(
        args.q_cutoff
    )
    overlap["new_significant_overlap_q"] = overlap["new_q_overlap"].le(args.q_cutoff)
    overlap["activity_change_new_minus_origin"] = (
        overlap["new_effect_vs_mean_control_mean"]
        - overlap["origin_effect_vs_mean_control_mean"]
    )
    overlap["absolute_activity_change"] = overlap[
        "activity_change_new_minus_origin"
    ].abs()
    overlap["overlap_t7_ge50_testable_filter"] = True
    overlap["overlap_call_status"] = np.select(
        [
            overlap["origin_significant_overlap_q"]
            & overlap["new_significant_overlap_q"],
            overlap["origin_significant_overlap_q"]
            & ~overlap["new_significant_overlap_q"],
            ~overlap["origin_significant_overlap_q"]
            & overlap["new_significant_overlap_q"],
        ],
        ["both_significant", "origin_only_significant", "new_only_significant"],
        default="neither_significant",
    )
    overlap.to_csv(
        tables_dir / "overlap_t7_ge50_bootstrap_pair_comparison.csv.gz",
        index=False,
    )
    overlap.nlargest(100, "absolute_activity_change").to_csv(
        tables_dir / "top_100_changed_overlap_bootstrap_pairs.csv", index=False
    )
    origin_control_activity = centered_negative_control_activity(
        args.origin_bootstrap,
        controls,
        args.bootstrap_chunk_size,
    )
    new_control_activity = centered_negative_control_activity(
        args.new_bootstrap,
        controls,
        args.bootstrap_chunk_size,
    )
    control_activity = origin_control_activity.rename(
        columns={
            "centered_log_activity_mean": "origin_centered_log_activity_mean",
            "n_testable_bootstraps": "origin_n_testable_bootstraps",
        }
    ).merge(
        new_control_activity.rename(
            columns={
                "centered_log_activity_mean": "new_centered_log_activity_mean",
                "n_testable_bootstraps": "new_n_testable_bootstraps",
            }
        ),
        on=KEY,
        how="inner",
        validate="one_to_one",
    )
    shared_groups = set(overlap["group"].astype(str))
    overlap_control_activity = control_activity.loc[
        control_activity["group"].astype(str).isin(shared_groups)
    ].copy()
    overlap_control_activity["centered_activity_change_new_minus_origin"] = (
        overlap_control_activity["new_centered_log_activity_mean"]
        - overlap_control_activity["origin_centered_log_activity_mean"]
    )
    expected_control_points = len(shared_groups) * len(controls)
    if len(overlap_control_activity) != expected_control_points:
        raise ValueError(
            "Expected one bootstrap point per shared cell type and control; "
            f"expected {expected_control_points}, found {len(overlap_control_activity)}"
        )
    overlap_control_activity.to_csv(
        tables_dir / "overlap_t7_ge50_bootstrap_negative_control_activity.csv",
        index=False,
    )

    long_frames = []
    for dataset in ("origin", "new"):
        frame = overlap[KEY].copy()
        frame["dataset"] = dataset
        for column in (
            "target_t7_total",
            "negative_control_t7_total",
            "activity_mean",
            "effect_vs_mean_control_mean",
            "p_right",
            "n_testable_bootstraps",
        ):
            frame[column] = overlap[f"{dataset}_{column}"]
        frame["q_right_overlap"] = overlap[f"{dataset}_q_overlap"]
        frame["significant_overlap_q"] = overlap[
            f"{dataset}_significant_overlap_q"
        ]
        long_frames.append(frame)
    pd.concat(long_frames, ignore_index=True).to_csv(
        tables_dir / "overlap_t7_ge50_bootstrap_tests_long.csv.gz", index=False
    )

    subclass_rows = []
    for group, frame in overlap.groupby("group", sort=True):
        pearson, spearman = safe_correlations(
            frame["origin_effect_vs_mean_control_mean"],
            frame["new_effect_vs_mean_control_mean"],
        )
        metrics = call_metrics(
            frame["origin_significant_overlap_q"],
            frame["new_significant_overlap_q"],
        )
        subclass_rows.append(
            {
                "group": group,
                "overlap_t7_ge50_testable_pairs": len(frame),
                "activity_pearson": pearson,
                "activity_spearman": spearman,
                "mean_change_new_minus_origin": frame[
                    "activity_change_new_minus_origin"
                ].mean(),
                **metrics,
            }
        )
    pd.DataFrame(subclass_rows).to_csv(
        tables_dir / "per_subclass_bootstrap_comparison.csv", index=False
    )

    pearson, spearman = safe_correlations(
        overlap["origin_effect_vs_mean_control_mean"],
        overlap["new_effect_vs_mean_control_mean"],
    )
    control_pearson, control_spearman = safe_correlations(
        overlap_control_activity["origin_centered_log_activity_mean"],
        overlap_control_activity["new_centered_log_activity_mean"],
    )
    finite_control_points = int(
        (
            np.isfinite(
                overlap_control_activity[
                    "origin_centered_log_activity_mean"
                ].to_numpy(float)
            )
            & np.isfinite(
                overlap_control_activity[
                    "new_centered_log_activity_mean"
                ].to_numpy(float)
            )
        ).sum()
    )
    metrics = call_metrics(
        overlap["origin_significant_overlap_q"],
        overlap["new_significant_overlap_q"],
    )
    delta = overlap["activity_change_new_minus_origin"]
    summary = {
        "inputs": {
            "origin_h5ad": origin_manifest["input"],
            "new_h5ad": new_manifest["input"],
            "origin_tests": str(args.origin_tests.resolve()),
            "new_tests": str(args.new_tests.resolve()),
        },
        "method": {
            "bootstrap_number": 10_000,
            "seed": 0,
            "min_detected_cells": 5,
            "origin_parallel_workers": origin_manifest["config"]["n_jobs"],
            "new_parallel_workers": new_manifest["config"]["n_jobs"],
            "parallel_workers_note": (
                "execution-only; replicate i uses random_state=i, so worker count "
                "does not change bootstrap samples"
            ),
            "blacklist": blacklist,
            "negative_controls": controls,
        },
        "statistics": {
            "primary_filter": (
                "target T7 >= 50 and combined seven-control T7 >= 50 in both "
                "datasets, with a finite bootstrap test in both; BH recomputed "
                "within this identical overlap"
            ),
            "overlap_t7_ge50_testable_pairs": len(overlap),
            "origin_significant_pairs": int(
                overlap["origin_significant_overlap_q"].sum()
            ),
            "new_significant_pairs": int(
                overlap["new_significant_overlap_q"].sum()
            ),
            **metrics,
            "secondary_origin_eligible_pairs": len(origin),
            "secondary_new_eligible_pairs": len(new),
        },
        "activity": {
            "overlap_centered_activity_pearson": pearson,
            "overlap_centered_activity_spearman": spearman,
            "overlap_mean_change_new_minus_origin": float(delta.mean()),
            "overlap_median_change_new_minus_origin": float(delta.median()),
            "overlap_mean_absolute_change": float(delta.abs().mean()),
            "overlap_root_mean_square_change": float(
                np.sqrt(np.mean(delta.to_numpy(float) ** 2))
            ),
            "overlap_negative_control_points": len(overlap_control_activity),
            "overlap_negative_control_finite_points": finite_control_points,
            "overlap_negative_control_cell_types": len(shared_groups),
            "overlap_negative_control_centered_activity_pearson": control_pearson,
            "overlap_negative_control_centered_activity_spearman": control_spearman,
        },
    }
    write_json(tables_dir / "bootstrap_comparison_summary.json", summary)
    pd.json_normalize(summary, sep=".").T.rename(columns={0: "value"}).to_csv(
        tables_dir / "bootstrap_comparison_summary.csv"
    )
    write_report(args.output_dir / "REPORT.md", summary)
    plot_activity(
        overlap,
        overlap_control_activity,
        figures_dir,
        t7_threshold=args.t7_threshold,
    )
    plot_calls(overlap, figures_dir)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
