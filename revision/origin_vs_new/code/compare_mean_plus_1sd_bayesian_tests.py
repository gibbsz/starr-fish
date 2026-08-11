#!/usr/bin/env python3
"""Compare old/new Bayesian calls using a draw-wise mean-plus-one-SD control."""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve()
ANALYSIS_DIR = HERE.parent.parent
REPO_ROOT = ANALYSIS_DIR.parents[1]
ORIGIN_ANALYSIS = REPO_ROOT / "revision" / "bayesian_vs_fold_change"
ORIGIN_CODE = ORIGIN_ANALYSIS / "code"
if str(ORIGIN_CODE) not in sys.path:
    sys.path.insert(0, str(ORIGIN_CODE))

from test_individual_negative_control_loo_empirical_fdr import (  # noqa: E402
    POOLED_NAME,
    load_grouped_t7,
)
from test_mean_negative_control_activity import compute_tests  # noqa: E402


NEW_RESULTS = REPO_ROOT / "revision" / "Bayes_NewData"

DEFAULT_ORIGIN_BAYES = ORIGIN_ANALYSIS / "results" / "bayesian"
DEFAULT_NEW_BAYES = NEW_RESULTS / "bayesian"
DEFAULT_ORIGIN_H5AD = (
    REPO_ROOT
    / "revision"
    / "Data"
    / "scdata_5_28_2025_BRBB500gn_final_CRE_T7CRE_NEWNEW.h5ad"
)
DEFAULT_NEW_H5AD = (
    REPO_ROOT
    / "revision"
    / "Data"
    / "scdata_07_29_2026_SFv8_low_dose_final_CRE_T7.h5ad"
)
DEFAULT_OUTPUT_DIR = ANALYSIS_DIR / "results" / "comparison"
KEY = ["group", "cre"]
SD_MULTIPLIER = 1.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--origin-bayes", type=Path, default=DEFAULT_ORIGIN_BAYES)
    parser.add_argument("--new-bayes", type=Path, default=DEFAULT_NEW_BAYES)
    parser.add_argument("--origin-h5ad", type=Path, default=DEFAULT_ORIGIN_H5AD)
    parser.add_argument("--new-h5ad", type=Path, default=DEFAULT_NEW_H5AD)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--t7-threshold", type=float, default=50.0)
    parser.add_argument("--q-cutoff", type=float, default=0.05)
    return parser.parse_args()


def read_manifest(bayes_dir: Path) -> dict:
    return json.loads((bayes_dir / "run_manifest.json").read_text())


def read_one_column(path: Path) -> list[str]:
    return pd.read_csv(path).iloc[:, 0].astype(str).tolist()


def validate_inputs(origin_bayes: Path, new_bayes: Path) -> tuple[list[str], list[str]]:
    origin_controls = read_one_column(origin_bayes / "negative_controls.csv")
    new_controls = read_one_column(new_bayes / "negative_controls.csv")
    if origin_controls != new_controls:
        raise ValueError("Original and new negative-control lists differ")
    if len(origin_controls) != 7:
        raise ValueError(f"Expected seven negative controls; found {len(origin_controls)}")

    origin_blacklist = read_one_column(origin_bayes / "cre_blacklist.csv")
    new_blacklist = read_one_column(new_bayes / "cre_blacklist.csv")
    if origin_blacklist != new_blacklist:
        raise ValueError("Original and new cCRE blacklists differ")
    return origin_controls, origin_blacklist


def run_dataset_test(
    *,
    bayes_dir: Path,
    h5ad: Path,
    controls: list[str],
    blacklist: list[str],
    t7_threshold: float,
    q_cutoff: float,
) -> tuple[pd.DataFrame, Path]:
    manifest = read_manifest(bayes_dir)
    posterior_path = bayes_dir / f"{manifest['tag']}_posterior_samples.npz"
    with np.load(posterior_path, allow_pickle=True) as posterior:
        groups = posterior["group_names"].astype(str)
        all_cre_names = posterior["cre_names"].astype(str)
        ordinary_mask = all_cre_names != POOLED_NAME
        cre_names = all_cre_names[ordinary_mask]
        log_gamma = posterior["log_gamma"][:, :, ordinary_mask].astype(np.float32)

    control_indices = np.flatnonzero(np.isin(cre_names, controls))
    if len(control_indices) != 7:
        found = cre_names[control_indices].tolist()
        raise ValueError(f"{posterior_path} has {len(control_indices)} controls: {found}")
    target_indices = np.flatnonzero(
        ~np.isin(cre_names, controls) & ~np.isin(cre_names, blacklist)
    )
    t7_totals, group_classes, group_cell_counts = load_grouped_t7(
        h5ad, groups, cre_names
    )
    tests = compute_tests(
        log_gamma,
        groups,
        cre_names,
        target_indices,
        control_indices,
        t7_totals,
        group_classes,
        group_cell_counts,
        t7_threshold,
        0.0,
        None,
        "Joint+dropout mean+1 SD controls",
        SD_MULTIPLIER,
    )
    tests["significant_q"] = tests["q_right"].le(q_cutoff)
    del log_gamma, t7_totals
    gc.collect()
    return tests, posterior_path


def bh_fdr(values: pd.Series) -> np.ndarray:
    array = values.to_numpy(float)
    output = np.full(len(array), np.nan, dtype=float)
    valid = np.isfinite(array)
    if valid.any():
        valid_values = array[valid]
        order = np.argsort(valid_values)
        ranked = valid_values[order]
        adjusted = np.minimum.accumulate(
            (ranked * len(ranked) / np.arange(1, len(ranked) + 1))[::-1]
        )[::-1]
        restored = np.empty_like(adjusted)
        restored[order] = np.clip(adjusted, 0.0, 1.0)
        output[valid] = restored
    return output


def call_metrics(origin_calls: pd.Series, new_calls: pd.Series) -> dict[str, float | int]:
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


def compare_shared(
    origin: pd.DataFrame, new: pd.DataFrame, q_cutoff: float
) -> tuple[pd.DataFrame, dict[str, float | int]]:
    columns = [
        "class",
        "n_cells",
        "target_t7_total",
        "negative_control_t7_total",
        "n_negative_controls",
        "negative_controls_used",
        "control_sd_multiplier",
        "activity_mean",
        "mean_negative_control_activity_mean",
        "negative_control_activity_sd_mean",
        "control_reference_activity_mean",
        "effect_vs_control_reference_mean",
        "effect_vs_control_reference_lo90",
        "effect_vs_control_reference_hi90",
        "posterior_probability_above_control_reference",
        "p_right",
        "q_right",
        "significant_q",
    ]

    def prefixed(frame: pd.DataFrame, prefix: str) -> pd.DataFrame:
        selected = frame[KEY + columns].copy()
        return selected.rename(
            columns={column: f"{prefix}_{column}" for column in columns}
        )

    shared = prefixed(origin, "origin").merge(
        prefixed(new, "new"), on=KEY, how="inner", validate="one_to_one"
    )
    shared["origin_q_common_universe"] = bh_fdr(shared["origin_p_right"])
    shared["new_q_common_universe"] = bh_fdr(shared["new_p_right"])
    shared["origin_significant_common_q"] = shared[
        "origin_q_common_universe"
    ].le(q_cutoff)
    shared["new_significant_common_q"] = shared["new_q_common_universe"].le(
        q_cutoff
    )
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
    return shared, call_metrics(
        shared["origin_significant_common_q"], shared["new_significant_common_q"]
    )


def plot_calls(
    shared: pd.DataFrame,
    metrics: dict[str, float | int],
    figures_dir: Path,
    t7_threshold: float,
    q_cutoff: float,
) -> None:
    labels = ["Original only", "Both", "New only"]
    values = [
        metrics["origin_only_significant"],
        metrics["both_significant"],
        metrics["new_only_significant"],
    ]
    colors = ["#4477AA", "#6E6E6E", "#CC6677"]

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.7))
    axes[0].bar(labels, values, color=colors)
    axes[0].set_ylabel("Significant subclass–cCRE pairs")
    axes[0].set_title(
        f"Mean + 1 SD control reference\nShared T7 ≥ {t7_threshold:g}; BH q ≤ {q_cutoff:g}"
    )
    maximum_value = max(max(values), 1)
    axes[0].set_ylim(0, maximum_value * 1.15)
    y_offset = maximum_value * 0.02
    for index, value in enumerate(values):
        axes[0].text(index, value + y_offset, f"{value:,}", ha="center", va="bottom")

    matrix = np.asarray(
        [
            [metrics["neither_significant"], metrics["new_only_significant"]],
            [metrics["origin_only_significant"], metrics["both_significant"]],
        ]
    )
    image = axes[1].imshow(matrix, cmap="Blues")
    axes[1].set_xticks([0, 1], ["New not sig.", "New sig."])
    axes[1].set_yticks([0, 1], ["Original not sig.", "Original sig."])
    axes[1].set_title(
        f"n={len(shared):,}; call concordance={metrics['call_concordance']:.3f}\n"
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
            figures_dir / f"shared_pair_significant_call_concordance.{suffix}",
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

    controls, blacklist = validate_inputs(args.origin_bayes, args.new_bayes)
    origin, origin_posterior = run_dataset_test(
        bayes_dir=args.origin_bayes,
        h5ad=args.origin_h5ad,
        controls=controls,
        blacklist=blacklist,
        t7_threshold=args.t7_threshold,
        q_cutoff=args.q_cutoff,
    )
    origin_path = tables_dir / "origin_mean_plus_1sd_tests_t7_ge50.csv.gz"
    origin.to_csv(origin_path, index=False)

    new, new_posterior = run_dataset_test(
        bayes_dir=args.new_bayes,
        h5ad=args.new_h5ad,
        controls=controls,
        blacklist=blacklist,
        t7_threshold=args.t7_threshold,
        q_cutoff=args.q_cutoff,
    )
    new_path = tables_dir / "new_mean_plus_1sd_tests_t7_ge50.csv.gz"
    new.to_csv(new_path, index=False)

    shared, metrics = compare_shared(origin, new, args.q_cutoff)
    shared_path = tables_dir / "shared_pair_mean_plus_1sd_comparison_t7_ge50.csv.gz"
    shared.to_csv(shared_path, index=False)
    plot_calls(shared, metrics, figures_dir, args.t7_threshold, args.q_cutoff)

    summary = {
        "test": {
            "reference": (
                "within each posterior draw and subclass: mean(log_gamma of all seven "
                "negative controls) + sample SD(log_gamma of all seven negative controls)"
            ),
            "contrast": "target log_gamma minus the draw-wise control reference",
            "p_right": "posterior fraction of contrasts <= 0",
            "multiple_testing": (
                "BH recomputed separately for original and new within the exact shared "
                "eligible pair universe"
            ),
            "t7_filter": (
                f"target T7 >= {args.t7_threshold:g} and combined seven-control T7 >= "
                f"{args.t7_threshold:g} in both datasets"
            ),
            "q_cutoff": args.q_cutoff,
            "control_sd_multiplier": SD_MULTIPLIER,
            "negative_controls": controls,
            "blacklist": blacklist,
        },
        "inputs": {
            "origin_h5ad": str(args.origin_h5ad.resolve()),
            "new_h5ad": str(args.new_h5ad.resolve()),
            "origin_posterior": str(origin_posterior.resolve()),
            "new_posterior": str(new_posterior.resolve()),
        },
        "counts": {
            "origin_eligible_pairs": int(len(origin)),
            "new_eligible_pairs": int(len(new)),
            "origin_native_significant_pairs": int(origin["significant_q"].sum()),
            "new_native_significant_pairs": int(new["significant_q"].sum()),
            "shared_eligible_pairs": int(len(shared)),
            **metrics,
        },
        "outputs": {
            "origin_tests": str(origin_path.resolve()),
            "new_tests": str(new_path.resolve()),
            "shared_comparison": str(shared_path.resolve()),
            "figure_pdf": str(
                (figures_dir / "shared_pair_significant_call_concordance.pdf").resolve()
            ),
            "figure_png": str(
                (figures_dir / "shared_pair_significant_call_concordance.png").resolve()
            ),
        },
    }
    summary_path = tables_dir / "mean_plus_1sd_significant_call_summary.json"
    write_json(summary_path, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
