#!/usr/bin/env python3
"""Compare original and new Bayesian activity estimates and significant pairs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde, pearsonr, spearmanr
from statsmodels.stats.multitest import multipletests


HERE = Path(__file__).resolve()
ANALYSIS_DIR = HERE.parent.parent
REPO_ROOT = ANALYSIS_DIR.parents[1]
ORIGIN_ANALYSIS = REPO_ROOT / "revision" / "bayesian_vs_fold_change"
NEW_RESULTS = REPO_ROOT / "revision" / "Bayes_NewData"

DEFAULT_ORIGIN_BAYES = ORIGIN_ANALYSIS / "results" / "bayesian"
DEFAULT_NEW_BAYES = NEW_RESULTS / "bayesian"
DEFAULT_ORIGIN_TESTS = (
    ORIGIN_ANALYSIS
    / "results"
    / "tables"
    / "joint_dropout_direct_activity_mean_negative_control_tests_t7_ge50.csv.gz"
)
DEFAULT_NEW_TESTS = (
    NEW_RESULTS / "tables" / "new_mean_negative_control_tests_t7_ge50.csv.gz"
)
DEFAULT_OUTPUT_DIR = ANALYSIS_DIR / "results" / "comparison"

KEY = ["group", "cre"]
ACTIVITY_COLUMNS = [
    "group",
    "cre",
    "raw_activity_mean",
    "centered_activity_mean",
]
TEST_VALUE_COLUMNS = [
    "class",
    "n_cells",
    "target_t7_total",
    "negative_control_t7_total",
    "activity_mean",
    "mean_negative_control_activity_mean",
    "effect_vs_mean_control_mean",
    "effect_vs_mean_control_lo90",
    "effect_vs_mean_control_hi90",
    "posterior_probability_above_mean_control",
    "p_right",
    "q_right",
    "significant_q",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--origin-bayes", type=Path, default=DEFAULT_ORIGIN_BAYES)
    parser.add_argument("--new-bayes", type=Path, default=DEFAULT_NEW_BAYES)
    parser.add_argument("--origin-tests", type=Path, default=DEFAULT_ORIGIN_TESTS)
    parser.add_argument("--new-tests", type=Path, default=DEFAULT_NEW_TESTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--t7-threshold", type=float, default=50.0)
    parser.add_argument("--q-cutoff", type=float, default=0.05)
    return parser.parse_args()


def read_manifest(directory: Path) -> dict:
    return json.loads((directory / "run_manifest.json").read_text())


def read_one_column(path: Path) -> list[str]:
    return pd.read_csv(path).iloc[:, 0].astype(str).tolist()


def validate_model_runs(
    origin_dir: Path, new_dir: Path
) -> tuple[dict, dict, list[str], list[str]]:
    origin = read_manifest(origin_dir)
    new = read_manifest(new_dir)
    required_config = {
        "level": "subclass",
        "channel": "joint",
        "infection_model": "copy_number_dropout",
        "activity_model": "direct",
        "negative_control_mode": "ordinary",
        "method": "svi",
        "guide": "AutoNormal",
        "kmax": 60,
        "num_steps": 30_000,
        "seed": 0,
    }
    for label, manifest in (("origin", origin), ("new", new)):
        config = manifest["config"]
        mismatches = {
            key: {"expected": expected, "observed": config.get(key)}
            for key, expected in required_config.items()
            if config.get(key) != expected
        }
        if mismatches:
            raise ValueError(f"{label} model configuration mismatch: {mismatches}")

    origin_blacklist = read_one_column(origin_dir / "cre_blacklist.csv")
    new_blacklist = read_one_column(new_dir / "cre_blacklist.csv")
    if origin_blacklist != new_blacklist:
        raise ValueError(
            f"blacklists differ: origin={origin_blacklist}, new={new_blacklist}"
        )
    origin_controls = read_one_column(origin_dir / "negative_controls.csv")
    new_controls = read_one_column(new_dir / "negative_controls.csv")
    if origin_controls != new_controls:
        raise ValueError(
            f"negative controls differ: origin={origin_controls}, new={new_controls}"
        )
    return origin, new, origin_blacklist, origin_controls


def posterior_path(directory: Path, manifest: dict) -> Path:
    return directory / f"{manifest['tag']}_posterior_samples.npz"


def posterior_activity(
    directory: Path,
    manifest: dict,
    negative_controls: list[str],
    blacklist: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    path = posterior_path(directory, manifest)
    with np.load(path, allow_pickle=True) as posterior:
        groups = posterior["group_names"].astype(str)
        cres = posterior["cre_names"].astype(str)
        log_gamma = posterior["log_gamma"]
    if log_gamma.shape[0] != 1_000:
        raise ValueError(
            f"{path} has {log_gamma.shape[0]} posterior draws; expected 1000"
        )

    control_idx = np.flatnonzero(np.isin(cres, negative_controls))
    if len(control_idx) != len(negative_controls):
        found = cres[control_idx].tolist()
        raise ValueError(
            f"{path} has {len(control_idx)} of {len(negative_controls)} controls: {found}"
        )
    target_idx = np.flatnonzero(
        ~np.isin(cres, negative_controls) & ~np.isin(cres, blacklist)
    )
    raw_mean = log_gamma[:, :, target_idx].mean(axis=0, dtype=np.float64)
    control_raw_mean = log_gamma[:, :, control_idx].mean(axis=0, dtype=np.float64)
    control_mean = control_raw_mean.mean(axis=1, dtype=np.float64)
    centered_mean = raw_mean - control_mean[:, None]
    control_centered_mean = control_raw_mean - control_mean[:, None]

    target_index = pd.MultiIndex.from_product(
        [groups, cres[target_idx]], names=KEY
    )
    control_index = pd.MultiIndex.from_product(
        [groups, cres[control_idx]], names=KEY
    )
    target_activity = pd.DataFrame(
        {
            "raw_activity_mean": raw_mean.reshape(-1),
            "centered_activity_mean": centered_mean.reshape(-1),
        },
        index=target_index,
    ).reset_index()
    control_activity = pd.DataFrame(
        {
            "raw_activity_mean": control_raw_mean.reshape(-1),
            "centered_activity_mean": control_centered_mean.reshape(-1),
        },
        index=control_index,
    ).reset_index()
    return target_activity, control_activity


def read_tests(
    path: Path,
    *,
    expected_threshold: float,
    q_cutoff: float,
    blacklist: list[str],
    negative_controls: list[str],
) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = set(KEY + TEST_VALUE_COLUMNS + ["t7_threshold"]) - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    if frame.duplicated(KEY).any():
        examples = frame.loc[frame.duplicated(KEY, keep=False), KEY].head().to_dict(
            "records"
        )
        raise ValueError(f"{path} has duplicate pair keys: {examples}")
    thresholds = frame["t7_threshold"].dropna().unique()
    if len(thresholds) != 1 or not np.isclose(thresholds[0], expected_threshold):
        raise ValueError(
            f"{path} threshold is {thresholds.tolist()}, expected {expected_threshold}"
        )
    if (frame["target_t7_total"] < expected_threshold).any():
        raise ValueError(f"{path} contains target T7 below {expected_threshold}")
    if (frame["negative_control_t7_total"] < expected_threshold).any():
        raise ValueError(f"{path} contains control T7 below {expected_threshold}")
    forbidden = set(blacklist) | set(negative_controls)
    observed_forbidden = sorted(set(frame["cre"].astype(str)) & forbidden)
    if observed_forbidden:
        raise ValueError(f"{path} includes forbidden target cCREs: {observed_forbidden}")
    derived_significant = frame["q_right"].le(q_cutoff)
    if not derived_significant.equals(frame["significant_q"].astype(bool)):
        raise ValueError(f"{path} significant_q is inconsistent with q <= {q_cutoff}")
    frame["group"] = frame["group"].astype(str)
    frame["cre"] = frame["cre"].astype(str)
    return frame


def prefixed(frame: pd.DataFrame, prefix: str, columns: list[str]) -> pd.DataFrame:
    selected = frame[KEY + columns].copy()
    return selected.rename(columns={column: f"{prefix}_{column}" for column in columns})


def bh_fdr(values: pd.Series) -> np.ndarray:
    array = values.to_numpy(float)
    output = np.full(len(array), np.nan)
    valid = np.isfinite(array)
    if valid.any():
        output[valid] = multipletests(array[valid], method="fdr_bh")[1]
    return output


def safe_correlations(x: pd.Series, y: pd.Series) -> tuple[float, float]:
    valid = np.isfinite(x.to_numpy(float)) & np.isfinite(y.to_numpy(float))
    if valid.sum() < 2:
        return np.nan, np.nan
    xv = x.to_numpy(float)[valid]
    yv = y.to_numpy(float)[valid]
    if np.ptp(xv) == 0 or np.ptp(yv) == 0:
        return np.nan, np.nan
    return float(pearsonr(xv, yv).statistic), float(spearmanr(xv, yv).statistic)


def call_metrics(
    origin_calls: pd.Series, new_calls: pd.Series
) -> dict[str, int | float]:
    origin = origin_calls.astype(bool).to_numpy()
    new = new_calls.astype(bool).to_numpy()
    both = int(np.sum(origin & new))
    origin_only = int(np.sum(origin & ~new))
    new_only = int(np.sum(~origin & new))
    neither = int(np.sum(~origin & ~new))
    union = both + origin_only + new_only
    return {
        "both_significant": both,
        "origin_only_significant": origin_only,
        "new_only_significant": new_only,
        "neither_significant": neither,
        "significant_union": union,
        "significant_jaccard": both / union if union else np.nan,
        "call_concordance": (both + neither) / len(origin) if len(origin) else np.nan,
    }


def make_per_subclass(shared: pd.DataFrame) -> pd.DataFrame:
    records = []
    for group, frame in shared.groupby("group", sort=True):
        pearson, spearman = safe_correlations(
            frame["origin_effect_vs_mean_control_mean"],
            frame["new_effect_vs_mean_control_mean"],
        )
        metrics = call_metrics(
            frame["origin_significant_common_q"],
            frame["new_significant_common_q"],
        )
        records.append(
            {
                "group": group,
                "class": frame["origin_class"].iloc[0],
                "overlap_t7_ge50_pairs": len(frame),
                "centered_activity_pearson": pearson,
                "centered_activity_spearman": spearman,
                "centered_activity_mean_change_new_minus_origin": frame[
                    "centered_activity_change_new_minus_origin"
                ].mean(),
                "centered_activity_median_change_new_minus_origin": frame[
                    "centered_activity_change_new_minus_origin"
                ].median(),
                **metrics,
            }
        )
    return pd.DataFrame(records)


def plot_activity(
    shared: pd.DataFrame,
    negative_control_activity: pd.DataFrame,
    output_dir: Path,
    t7_threshold: float = 50,
) -> None:
    threshold_token = f"{t7_threshold:g}".replace(".", "p")
    x = shared["origin_effect_vs_mean_control_mean"].to_numpy(float)
    y = shared["new_effect_vs_mean_control_mean"].to_numpy(float)
    control_x = negative_control_activity[
        "origin_centered_activity_mean"
    ].to_numpy(float)
    control_y = negative_control_activity[
        "new_centered_activity_mean"
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
        label=f"T7≥{t7_threshold:g} target pairs (n={len(shared):,})",
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
                "Negative controls "
                f"(7 cCREs × {negative_control_activity['group'].nunique()} cell types; "
                f"n={finite_controls.sum():,})"
            ),
        )

    ax.set_xlim(limits)
    ax.set_ylim(limits)
    ax.set_xlabel("Original centered posterior activity")
    ax.set_ylabel("New low-dose centered posterior activity")
    ax.set_title(
        f"Overlap filter: T7 ≥ {t7_threshold:g} in both datasets\n"
        f"n={len(shared):,}; Pearson r={pearson:.3f}; Spearman ρ={spearman:.3f}"
    )
    ax.legend(loc="lower right", frameon=True, fontsize=8)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(
            output_dir
            / f"overlap_t7_ge{threshold_token}_activity_concordance.{suffix}",
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(fig)


def plot_calls(shared: pd.DataFrame, output_dir: Path) -> None:
    metrics = call_metrics(
        shared["origin_significant_common_q"],
        shared["new_significant_common_q"],
    )
    labels = ["Original only", "Both", "New only"]
    values = [
        metrics["origin_only_significant"],
        metrics["both_significant"],
        metrics["new_only_significant"],
    ]
    colors = ["#4477AA", "#6E6E6E", "#CC6677"]

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.4))
    axes[0].bar(labels, values, color=colors)
    axes[0].set_ylabel("Significant subclass–cCRE pairs")
    axes[0].set_title("Overlap-filtered universe, BH q ≤ 0.05")
    for index, value in enumerate(values):
        axes[0].text(index, value, f"{value:,}", ha="center", va="bottom")

    matrix = np.asarray(
        [
            [
                metrics["neither_significant"],
                metrics["new_only_significant"],
            ],
            [
                metrics["origin_only_significant"],
                metrics["both_significant"],
            ],
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
            output_dir
            / f"overlap_t7_ge50_significant_call_concordance.{suffix}",
            dpi=300,
        )
    plt.close(fig)


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def main() -> None:
    args = parse_args()
    tables_dir = args.output_dir / "tables"
    figures_dir = args.output_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    origin_manifest, new_manifest, blacklist, controls = validate_model_runs(
        args.origin_bayes, args.new_bayes
    )
    origin_tests = read_tests(
        args.origin_tests,
        expected_threshold=args.t7_threshold,
        q_cutoff=args.q_cutoff,
        blacklist=blacklist,
        negative_controls=controls,
    )
    new_tests = read_tests(
        args.new_tests,
        expected_threshold=args.t7_threshold,
        q_cutoff=args.q_cutoff,
        blacklist=blacklist,
        negative_controls=controls,
    )

    origin_activity, origin_control_activity = posterior_activity(
        args.origin_bayes, origin_manifest, controls, blacklist
    )
    new_activity, new_control_activity = posterior_activity(
        args.new_bayes, new_manifest, controls, blacklist
    )
    all_activity = prefixed(
        origin_activity, "origin", ACTIVITY_COLUMNS[2:]
    ).merge(
        prefixed(new_activity, "new", ACTIVITY_COLUMNS[2:]),
        on=KEY,
        how="outer",
        validate="one_to_one",
        indicator="activity_pair_scope",
    )
    all_activity.to_csv(
        tables_dir / "all_common_and_dataset_specific_activity.csv.gz",
        index=False,
    )
    common_activity = all_activity.loc[
        all_activity["activity_pair_scope"].eq("both")
    ].copy()
    common_activity["raw_activity_change_new_minus_origin"] = (
        common_activity["new_raw_activity_mean"]
        - common_activity["origin_raw_activity_mean"]
    )
    common_activity["centered_activity_change_new_minus_origin"] = (
        common_activity["new_centered_activity_mean"]
        - common_activity["origin_centered_activity_mean"]
    )
    common_activity["absolute_centered_activity_change"] = common_activity[
        "centered_activity_change_new_minus_origin"
    ].abs()
    common_activity.to_csv(tables_dir / "all_common_activity.csv.gz", index=False)

    control_activity = prefixed(
        origin_control_activity, "origin", ACTIVITY_COLUMNS[2:]
    ).merge(
        prefixed(new_control_activity, "new", ACTIVITY_COLUMNS[2:]),
        on=KEY,
        how="outer",
        validate="one_to_one",
        indicator="control_pair_scope",
    )

    pair_comparison = prefixed(
        origin_tests, "origin", TEST_VALUE_COLUMNS
    ).merge(
        prefixed(new_tests, "new", TEST_VALUE_COLUMNS),
        on=KEY,
        how="outer",
        validate="one_to_one",
        indicator="test_pair_scope",
    )
    pair_comparison.to_csv(
        tables_dir / "all_eligible_pair_comparison.csv.gz", index=False
    )

    shared = pair_comparison.loc[pair_comparison["test_pair_scope"].eq("both")].copy()
    shared["origin_q_common_universe"] = bh_fdr(shared["origin_p_right"])
    shared["new_q_common_universe"] = bh_fdr(shared["new_p_right"])
    shared["origin_significant_common_q"] = shared[
        "origin_q_common_universe"
    ].le(args.q_cutoff)
    shared["new_significant_common_q"] = shared["new_q_common_universe"].le(
        args.q_cutoff
    )
    shared["raw_activity_change_new_minus_origin"] = (
        shared["new_activity_mean"] - shared["origin_activity_mean"]
    )
    shared["centered_activity_change_new_minus_origin"] = (
        shared["new_effect_vs_mean_control_mean"]
        - shared["origin_effect_vs_mean_control_mean"]
    )
    shared["absolute_centered_activity_change"] = shared[
        "centered_activity_change_new_minus_origin"
    ].abs()
    shared["common_q_call_status"] = np.select(
        [
            shared["origin_significant_common_q"]
            & shared["new_significant_common_q"],
            shared["origin_significant_common_q"]
            & ~shared["new_significant_common_q"],
            ~shared["origin_significant_common_q"]
            & shared["new_significant_common_q"],
        ],
        ["both_significant", "origin_only_significant", "new_only_significant"],
        default="neither_significant",
    )
    shared["overlap_t7_ge50_filter"] = True
    shared.to_csv(tables_dir / "shared_test_pair_comparison.csv.gz", index=False)
    shared.to_csv(
        tables_dir / "overlap_t7_ge50_pair_comparison.csv.gz", index=False
    )
    shared.nlargest(100, "absolute_centered_activity_change").to_csv(
        tables_dir / "top_100_changed_shared_test_pairs.csv", index=False
    )
    shared.nlargest(100, "absolute_centered_activity_change").to_csv(
        tables_dir / "top_100_changed_overlap_t7_ge50_pairs.csv", index=False
    )
    shared_groups = set(shared["group"].astype(str))
    overlap_control_activity = control_activity.loc[
        control_activity["control_pair_scope"].eq("both")
        & control_activity["group"].astype(str).isin(shared_groups)
    ].copy()
    overlap_control_activity["centered_activity_change_new_minus_origin"] = (
        overlap_control_activity["new_centered_activity_mean"]
        - overlap_control_activity["origin_centered_activity_mean"]
    )
    expected_control_points = len(shared_groups) * len(controls)
    if len(overlap_control_activity) != expected_control_points:
        raise ValueError(
            "Expected one point per shared cell type and ordinary negative control; "
            f"expected {expected_control_points}, found {len(overlap_control_activity)}"
        )
    overlap_control_activity.to_csv(
        tables_dir / "overlap_t7_ge50_negative_control_activity.csv",
        index=False,
    )

    overlap_tests = []
    for dataset in ("origin", "new"):
        selected = shared[KEY].copy()
        selected["dataset"] = dataset
        selected["class"] = shared[f"{dataset}_class"]
        selected["target_t7_total"] = shared[f"{dataset}_target_t7_total"]
        selected["negative_control_t7_total"] = shared[
            f"{dataset}_negative_control_t7_total"
        ]
        selected["activity_mean"] = shared[f"{dataset}_activity_mean"]
        selected["effect_vs_mean_control_mean"] = shared[
            f"{dataset}_effect_vs_mean_control_mean"
        ]
        selected["p_right"] = shared[f"{dataset}_p_right"]
        selected["q_right_overlap"] = shared[f"{dataset}_q_common_universe"]
        selected["significant_overlap_q"] = shared[
            f"{dataset}_significant_common_q"
        ]
        overlap_tests.append(selected)
    pd.concat(overlap_tests, ignore_index=True).to_csv(
        tables_dir / "overlap_t7_ge50_tests_long.csv.gz", index=False
    )

    native_origin_calls = set(
        map(
            tuple,
            origin_tests.loc[origin_tests["significant_q"].astype(bool), KEY].to_numpy(),
        )
    )
    native_new_calls = set(
        map(tuple, new_tests.loc[new_tests["significant_q"].astype(bool), KEY].to_numpy())
    )
    all_native_keys = pd.DataFrame(
        sorted(native_origin_calls | native_new_calls), columns=KEY
    )
    all_native_keys["origin_significant_native_q"] = [
        tuple(row) in native_origin_calls for row in all_native_keys[KEY].to_numpy()
    ]
    all_native_keys["new_significant_native_q"] = [
        tuple(row) in native_new_calls for row in all_native_keys[KEY].to_numpy()
    ]
    all_native_keys["native_q_call_status"] = np.select(
        [
            all_native_keys["origin_significant_native_q"]
            & all_native_keys["new_significant_native_q"],
            all_native_keys["origin_significant_native_q"]
            & ~all_native_keys["new_significant_native_q"],
        ],
        ["both_significant", "origin_only_significant"],
        default="new_only_significant",
    )
    all_native_keys.to_csv(
        tables_dir / "native_significant_pair_union.csv.gz", index=False
    )
    for status in (
        "both_significant",
        "origin_only_significant",
        "new_only_significant",
    ):
        all_native_keys.loc[all_native_keys["native_q_call_status"].eq(status)].to_csv(
            tables_dir / f"native_{status}.csv", index=False
        )

    per_subclass = make_per_subclass(shared)
    per_subclass.to_csv(tables_dir / "per_subclass_comparison.csv", index=False)

    all_activity_pearson, all_activity_spearman = safe_correlations(
        common_activity["origin_centered_activity_mean"],
        common_activity["new_centered_activity_mean"],
    )
    shared_pearson, shared_spearman = safe_correlations(
        shared["origin_effect_vs_mean_control_mean"],
        shared["new_effect_vs_mean_control_mean"],
    )
    control_pearson, control_spearman = safe_correlations(
        overlap_control_activity["origin_centered_activity_mean"],
        overlap_control_activity["new_centered_activity_mean"],
    )
    shared_call_metrics = call_metrics(
        shared["origin_significant_common_q"],
        shared["new_significant_common_q"],
    )
    native_both = len(native_origin_calls & native_new_calls)
    native_union = len(native_origin_calls | native_new_calls)
    shared_delta = shared["centered_activity_change_new_minus_origin"]

    summary = {
        "inputs": {
            "origin_h5ad": origin_manifest["input"],
            "new_h5ad": new_manifest["input"],
            "origin_tests": str(args.origin_tests.resolve()),
            "new_tests": str(args.new_tests.resolve()),
        },
        "model": {
            "method_variant": origin_manifest["method_variant"],
            "origin_cells": origin_manifest["n_cells"],
            "new_cells": new_manifest["n_cells"],
            "origin_subclasses": origin_manifest["n_subclasses"],
            "new_subclasses": new_manifest["n_subclasses"],
            "blacklist": blacklist,
            "negative_controls": controls,
        },
        "statistics": {
            "t7_threshold": args.t7_threshold,
            "q_cutoff": args.q_cutoff,
            "primary_filter": (
                "identical overlap of pairs with target T7 >= 50 and combined "
                "seven-control T7 >= 50 in both datasets; BH recomputed within "
                "this pair universe for each dataset"
            ),
            "overlap_t7_ge50_pairs": len(shared),
            "overlap_origin_significant_pairs": int(
                shared["origin_significant_common_q"].sum()
            ),
            "overlap_new_significant_pairs": int(
                shared["new_significant_common_q"].sum()
            ),
            "overlap_significant_both": shared_call_metrics["both_significant"],
            "overlap_significant_origin_only": shared_call_metrics[
                "origin_only_significant"
            ],
            "overlap_significant_new_only": shared_call_metrics[
                "new_only_significant"
            ],
            "overlap_significant_neither": shared_call_metrics[
                "neither_significant"
            ],
            "overlap_call_concordance": shared_call_metrics["call_concordance"],
            "overlap_significant_jaccard": shared_call_metrics[
                "significant_jaccard"
            ],
            "secondary_dataset_specific_results": {
            "origin_eligible_pairs": len(origin_tests),
            "new_eligible_pairs": len(new_tests),
            "origin_native_significant_pairs": len(native_origin_calls),
            "new_native_significant_pairs": len(native_new_calls),
            "origin_native_significant_fraction": len(native_origin_calls)
            / len(origin_tests),
            "new_native_significant_fraction": len(native_new_calls) / len(new_tests),
            "native_significant_both": native_both,
            "native_significant_jaccard": (
                native_both / native_union if native_union else np.nan
            ),
            },
        },
        "activity": {
            "all_common_pairs": len(common_activity),
            "all_common_centered_activity_pearson": all_activity_pearson,
            "all_common_centered_activity_spearman": all_activity_spearman,
            "overlap_t7_ge50_centered_activity_pearson": shared_pearson,
            "overlap_t7_ge50_centered_activity_spearman": shared_spearman,
            "overlap_t7_ge50_origin_mean": float(
                shared["origin_effect_vs_mean_control_mean"].mean()
            ),
            "overlap_t7_ge50_new_mean": float(
                shared["new_effect_vs_mean_control_mean"].mean()
            ),
            "overlap_t7_ge50_mean_change_new_minus_origin": float(
                shared_delta.mean()
            ),
            "overlap_t7_ge50_median_change_new_minus_origin": float(
                shared_delta.median()
            ),
            "overlap_t7_ge50_mean_absolute_change": float(shared_delta.abs().mean()),
            "overlap_t7_ge50_root_mean_square_change": float(
                np.sqrt(np.mean(shared_delta.to_numpy(float) ** 2))
            ),
            "overlap_t7_ge50_negative_control_points": len(
                overlap_control_activity
            ),
            "overlap_t7_ge50_negative_control_cell_types": len(shared_groups),
            "overlap_t7_ge50_negative_control_centered_activity_pearson": (
                control_pearson
            ),
            "overlap_t7_ge50_negative_control_centered_activity_spearman": (
                control_spearman
            ),
        },
    }
    write_json(tables_dir / "comparison_summary.json", summary)
    pd.json_normalize(summary, sep=".").T.rename(columns={0: "value"}).to_csv(
        tables_dir / "comparison_summary.csv"
    )

    plot_activity(
        shared,
        overlap_control_activity,
        figures_dir,
        t7_threshold=args.t7_threshold,
    )
    plot_calls(shared, figures_dir)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
