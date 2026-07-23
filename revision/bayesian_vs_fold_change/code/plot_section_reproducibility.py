#!/usr/bin/env python3
"""Compare independent section fits of the selected Bayesian model.

Only two figure pairs are produced:

1. sec1/sec2 correlation of posterior-mean raw log activity;
2. concordance of one-sided tests against the draw-wise mean activity of the
   seven ordinary negative controls.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
from statsmodels.stats.multitest import multipletests

from analysis_utils import ANALYSIS_DIR, DEFAULT_H5AD, log, write_json
from plot_results import bayesian_significance
from plot_section_reproducibility_research_filters import (
    bayesian_uncalibrated,
    bootstrap_uncalibrated,
)


SECTIONS = ("sec1", "sec2")
EXPECTED_MODEL = "bayesian_joint_dropout_ordinary_and_pooled_negative_controls"
MODEL_DIRNAME = "bayesian"
POOLED_NAME = "NEGATIVE_CONTROL_POOL"


def bootstrap_base(
    root: Path, args: argparse.Namespace
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load a bootstrap run for the cross-method ablation plotting scripts."""
    if args.activity_calibration == "none":
        activity, qvalues = bootstrap_uncalibrated(
            root, args.bootstrap_log_chunk_size
        )
    else:
        activity_path = root / "log_activity_prior_mask_vs_negative_control.csv"
        if not activity_path.exists():
            activity_path = root / "log_activity_vs_negative_control.csv"
        qvalue_path = root / "qvalues_prior_mask_right.csv"
        if not qvalue_path.exists():
            qvalue_path = root / "qvalues_right.csv"
        activity = pd.read_csv(activity_path, index_col=0)
        qvalues = pd.read_csv(qvalue_path, index_col=0)
    activity.index = activity.index.astype(str)
    activity.columns = activity.columns.astype(str)
    qvalues.index = qvalues.index.astype(str)
    qvalues.columns = qvalues.columns.astype(str)
    return activity, qvalues


def bayesian_base(
    root: Path, args: argparse.Namespace
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load a Bayesian run for the cross-method ablation plotting scripts."""
    manifest = json.loads((root / "run_manifest.json").read_text())
    tag = str(manifest["tag"])
    gamma = pd.read_csv(root / f"{tag}_gamma.csv")
    negative_controls = set(
        pd.read_csv(root / "negative_controls.csv").iloc[:, 0].astype(str)
    )
    significance = bayesian_significance(
        gamma,
        root / f"{tag}_posterior_samples.npz",
        negative_controls,
        0,
        filter_negative_controls=False,
        filter_prior_dominated=False,
    )
    if args.activity_calibration == "none":
        activity, qvalues = bayesian_uncalibrated(root)
    else:
        activity = significance.pivot(
            index="group", columns="cre", values="bayesian_effect_log"
        )
        qvalues = significance.pivot(
            index="group", columns="cre", values="bayesian_q_right"
        )
    activity.index = activity.index.astype(str)
    activity.columns = activity.columns.astype(str)
    qvalues.index = qvalues.index.astype(str)
    qvalues.columns = qvalues.columns.astype(str)
    return activity, qvalues


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sections-dir",
        type=Path,
        default=ANALYSIS_DIR / "results" / "sections",
        help="Parent directory containing sec1/ and sec2/ model outputs.",
    )
    parser.add_argument(
        "--model-dirname",
        default=MODEL_DIRNAME,
        help="Bayesian result-directory name below each section.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ANALYSIS_DIR / "results" / "section_reproducibility",
    )
    parser.add_argument("--h5ad", type=Path, default=DEFAULT_H5AD)
    parser.add_argument(
        "--t7-threshold",
        type=float,
        default=50.0,
        help=(
            "Minimum section-specific T7 total for each target; the summed T7 "
            "of all seven controls must also meet this threshold."
        ),
    )
    parser.add_argument(
        "--effect-threshold",
        type=float,
        default=0.0,
        help="Minimum target-minus-mean-control log-activity effect.",
    )
    parser.add_argument("--q-cutoff", type=float, default=0.05)
    parser.add_argument("--min-points", type=int, default=15)
    return parser.parse_args()


def save_figure(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), bbox_inches="tight", dpi=200)
    plt.close(fig)


def decode_strings(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    if values.dtype.kind in {"S", "O"}:
        return np.asarray(
            [
                value.decode("utf-8")
                if isinstance(value, (bytes, np.bytes_))
                else str(value)
                for value in values
            ],
            dtype=str,
        )
    return values.astype(str)


def normalize_labels(values: np.ndarray) -> np.ndarray:
    return np.asarray(
        [re.sub(r"^\d+\s+", "", value).replace("/", "-") for value in values],
        dtype=str,
    )


def bh_fdr(pvalues: np.ndarray) -> np.ndarray:
    values = np.asarray(pvalues, dtype=float)
    output = np.full(values.shape, np.nan, dtype=float)
    valid = np.isfinite(values)
    if valid.any():
        output[valid] = multipletests(values[valid], method="fdr_bh")[1]
    return output


def posterior_mean_control_centered_activity(
    log_gamma: np.ndarray,
    control_indices: np.ndarray,
    target_indices: np.ndarray,
) -> np.ndarray:
    """Return posterior-mean target activity minus the seven-control mean."""
    target_mean = log_gamma[:, :, target_indices].mean(axis=0, dtype=np.float64)
    control_mean = log_gamma[:, :, control_indices].mean(axis=(0, 2), dtype=np.float64)
    return target_mean - control_mean[:, None]


def load_section_t7_totals(
    h5ad: Path,
    section: str,
    groups: np.ndarray,
    cre_names: np.ndarray,
) -> np.ndarray:
    """Return section-specific subclass-by-cCRE T7 totals."""
    prefix = {"sec1": "Conv_zscan2_", "sec2": "Conv_zscan1_"}[section]
    with h5py.File(h5ad, "r") as handle:
        obs_names = decode_strings(handle["obs"]["_index"][...])
        section_mask = np.fromiter(
            (name.startswith(prefix) for name in obs_names),
            dtype=bool,
            count=len(obs_names),
        )
        subclass = handle["obs"]["subclass_name"]
        categories = normalize_labels(decode_strings(subclass["categories"][...]))
        codes = subclass["codes"][...].astype(np.int64)
        valid = section_mask & (codes >= 0)

        category_lookup = {name: index for index, name in enumerate(categories)}
        missing_groups = sorted(set(groups) - set(category_lookup))
        if missing_groups:
            raise ValueError(
                f"{section}: H5AD is missing posterior subclasses: {missing_groups}"
            )
        posterior_to_h5 = np.asarray(
            [category_lookup[name] for name in groups], dtype=np.int64
        )

        t7_group = handle["obsm"]["T7CRE"]
        missing_cres = sorted(set(cre_names) - set(t7_group.keys()))
        if missing_cres:
            raise ValueError(f"H5AD T7 matrix is missing fitted cCREs: {missing_cres}")

        totals = np.empty((len(groups), len(cre_names)), dtype=np.float64)
        for cre_index, cre in enumerate(cre_names):
            values = t7_group[cre][...].astype(np.float64, copy=False)
            grouped = np.bincount(
                codes[valid],
                weights=values[valid],
                minlength=len(categories),
            )
            totals[:, cre_index] = grouped[posterior_to_h5]
    return totals


def compute_mean_control_tests(
    *,
    log_gamma: np.ndarray,
    groups: np.ndarray,
    cre_names: np.ndarray,
    negative_controls: list[str],
    t7_totals: np.ndarray,
    t7_threshold: float,
    effect_threshold: float,
    section: str,
) -> pd.DataFrame:
    """Test targets against the draw-wise mean of all seven ordinary controls."""
    control_indices = np.flatnonzero(np.isin(cre_names, negative_controls))
    if len(control_indices) != 7:
        raise ValueError(
            f"{section}: expected seven ordinary negative controls; "
            f"found {len(control_indices)}"
        )
    target_indices = np.flatnonzero(~np.isin(cre_names, negative_controls))
    rows: list[pd.DataFrame] = []
    for group_index, group in enumerate(groups):
        control_t7_total = float(t7_totals[group_index, control_indices].sum())
        if control_t7_total < t7_threshold:
            continue

        mean_control_draws = (
            log_gamma[:, group_index, control_indices]
            .astype(np.float64, copy=False)
            .mean(axis=1)
        )
        eligible = t7_totals[group_index, target_indices] >= t7_threshold
        selected = target_indices[eligible]
        if not len(selected):
            continue

        target_draws = log_gamma[:, group_index, selected].astype(
            np.float64, copy=False
        )
        activity_contrasts = target_draws - mean_control_draws[:, None]
        contrasts = activity_contrasts - effect_threshold
        contrast_lo, contrast_hi = np.quantile(contrasts, [0.05, 0.95], axis=0)
        rows.append(
            pd.DataFrame(
                {
                    "section": section,
                    "group": group,
                    "cre": cre_names[selected],
                    "target_t7_total": t7_totals[group_index, selected],
                    "negative_control_t7_total": control_t7_total,
                    "n_negative_controls": len(control_indices),
                    "negative_controls_used": ",".join(
                        cre_names[control_indices].tolist()
                    ),
                    "activity_mean": target_draws.mean(axis=0),
                    "mean_negative_control_activity_mean": float(
                        mean_control_draws.mean()
                    ),
                    "activity_vs_mean_control_mean": activity_contrasts.mean(axis=0),
                    "effect_vs_mean_control_mean": contrasts.mean(axis=0),
                    "effect_vs_mean_control_lo90": contrast_lo,
                    "effect_vs_mean_control_hi90": contrast_hi,
                    "posterior_probability_above_mean_control": (contrasts > 0.0).mean(
                        axis=0
                    ),
                    "p_right": (contrasts <= 0.0).mean(axis=0),
                }
            )
        )
    if not rows:
        raise ValueError(f"{section}: no pairs passed the T7 filters")
    return pd.concat(rows, ignore_index=True)


def load_section(
    *,
    root: Path,
    section: str,
    h5ad: Path,
    t7_threshold: float,
    effect_threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    manifest_path = root / "run_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"missing {manifest_path}; run submit_sections.sh to fit {section}"
        )
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("section") != section:
        raise ValueError(
            f"{root}: expected section={section}, found {manifest.get('section')}"
        )
    if manifest.get("method_variant") != EXPECTED_MODEL:
        raise ValueError(
            f"{root}: expected {EXPECTED_MODEL}, found {manifest.get('method_variant')}"
        )

    posterior_path = root / f"{manifest['tag']}_posterior_samples.npz"
    negative_controls = (
        pd.read_csv(root / "negative_controls.csv").iloc[:, 0].astype(str).tolist()
    )
    if len(negative_controls) != 7:
        raise ValueError(
            f"{root}: expected seven annotated controls; found {len(negative_controls)}"
        )
    blacklist_path = root / "cre_blacklist.csv"
    blacklist = (
        set(pd.read_csv(blacklist_path).iloc[:, 0].astype(str))
        if blacklist_path.exists()
        else set()
    )

    with np.load(posterior_path, allow_pickle=True) as posterior:
        groups = posterior["group_names"].astype(str)
        all_cre_names = posterior["cre_names"].astype(str)
        if POOLED_NAME not in all_cre_names:
            raise ValueError(f"{root}: pooled negative-control pseudo-cCRE is missing")
        ordinary = all_cre_names != POOLED_NAME
        cre_names = all_cre_names[ordinary]
        log_gamma = posterior["log_gamma"][:, :, ordinary].astype(np.float32)

    control_indices = np.flatnonzero(np.isin(cre_names, negative_controls))
    if len(control_indices) != 7:
        raise ValueError(
            f"{root}: expected seven posterior control columns; "
            f"found {len(control_indices)}"
        )
    target_mask = ~np.isin(cre_names, [*negative_controls, *blacklist])
    target_indices = np.flatnonzero(target_mask)
    activity = pd.DataFrame(
        posterior_mean_control_centered_activity(
            log_gamma,
            control_indices,
            target_indices,
        ),
        index=pd.Index(groups, name="group"),
        columns=pd.Index(cre_names[target_indices], name="cre"),
    )
    t7_totals = load_section_t7_totals(h5ad, section, groups, cre_names)
    tests = compute_mean_control_tests(
        log_gamma=log_gamma,
        groups=groups,
        cre_names=cre_names,
        negative_controls=negative_controls,
        t7_totals=t7_totals,
        t7_threshold=t7_threshold,
        effect_threshold=effect_threshold,
        section=section,
    )
    tests = tests.loc[~tests["cre"].isin(blacklist)].reset_index(drop=True)
    metadata = {
        "root": str(root),
        "manifest": str(manifest_path),
        "posterior": str(posterior_path),
        "method_variant": manifest["method_variant"],
        "negative_controls": negative_controls,
        "n_groups": int(len(groups)),
        "n_target_ccres": int(target_mask.sum()),
        "n_eligible_tests_before_shared_filter": int(len(tests)),
    }
    return activity, tests, metadata


def restrict_to_shared_eligibility(
    tests: dict[str, pd.DataFrame],
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Use one pair universe that passes the T7 filters in both sections.

    P-values are computed from each section's posterior, but BH correction is
    applied only after this shared pair universe has been established.
    """
    eligibility_columns = [
        "group",
        "cre",
        "target_t7_total",
        "negative_control_t7_total",
    ]
    shared = tests["sec1"][eligibility_columns].merge(
        tests["sec2"][eligibility_columns],
        on=["group", "cre"],
        how="inner",
        suffixes=("_sec1", "_sec2"),
        validate="one_to_one",
    )
    if shared.empty:
        raise ValueError("no subclass-cCRE pairs pass the T7 filters in both sections")
    shared["shared_t7_eligible"] = True
    shared_keys = shared[["group", "cre"]]

    filtered: dict[str, pd.DataFrame] = {}
    for section in SECTIONS:
        frame = shared_keys.merge(
            tests[section],
            on=["group", "cre"],
            how="left",
            validate="one_to_one",
        )
        if frame["p_right"].isna().any():
            raise AssertionError(
                f"{section}: shared eligibility produced missing p-values"
            )
        frame["q_right"] = bh_fdr(frame["p_right"].to_numpy(float))
        filtered[section] = frame
    return filtered, shared


def wide_to_long(frame: pd.DataFrame, value_name: str) -> pd.DataFrame:
    return (
        frame.rename_axis(index="group", columns="cre")
        .reset_index()
        .melt(id_vars="group", var_name="cre", value_name=value_name)
    )


def correlation_values(
    x: pd.Series | np.ndarray,
    y: pd.Series | np.ndarray,
    min_points: int,
) -> tuple[int, float, float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    if len(x) < min_points or np.unique(x).size < 2 or np.unique(y).size < 2:
        return int(len(x)), np.nan, np.nan
    return (
        int(len(x)),
        float(spearmanr(x, y).statistic),
        float(pearsonr(x, y).statistic),
    )


def activity_correlation_tables(
    activity: dict[str, pd.DataFrame],
    shared_eligibility: pd.DataFrame,
    min_points: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    common_groups = activity["sec1"].index.intersection(activity["sec2"].index)
    common_cres = activity["sec1"].columns.intersection(activity["sec2"].columns)
    sec1 = activity["sec1"].reindex(index=common_groups, columns=common_cres)
    sec2 = activity["sec2"].reindex(index=common_groups, columns=common_cres)
    pair = wide_to_long(sec1, "activity_sec1").merge(
        wide_to_long(sec2, "activity_sec2"),
        on=["group", "cre"],
        how="inner",
        validate="one_to_one",
    )
    pair = pair.merge(
        shared_eligibility,
        on=["group", "cre"],
        how="inner",
        validate="one_to_one",
    )
    pair["measured"] = np.isfinite(pair["activity_sec1"]) & np.isfinite(
        pair["activity_sec2"]
    )
    measured = pair.loc[pair["measured"]]

    rows = []
    n, rho, pearson = correlation_values(
        measured["activity_sec1"], measured["activity_sec2"], min_points
    )
    rows.append(
        {
            "axis": "all_pairs",
            "unit": "all",
            "n_points": n,
            "spearman": rho,
            "pearson": pearson,
        }
    )
    for axis, unit_column in (
        ("within_subclass_across_ccres", "group"),
        ("across_subclasses_per_ccre", "cre"),
    ):
        for unit, frame in measured.groupby(unit_column, sort=False):
            n, rho, pearson = correlation_values(
                frame["activity_sec1"], frame["activity_sec2"], min_points
            )
            rows.append(
                {
                    "axis": axis,
                    "unit": unit,
                    "n_points": n,
                    "spearman": rho,
                    "pearson": pearson,
                }
            )
    return pair, pd.DataFrame(rows)


def plot_activity_correlation(
    pair: pd.DataFrame,
    correlations: pd.DataFrame,
    figures: Path,
    t7_threshold: float,
) -> None:
    frame = pair.loc[pair["measured"]]
    if frame.empty:
        raise ValueError("no common activity pairs to plot")
    overall = correlations.loc[correlations["axis"].eq("all_pairs")].iloc[0]
    fig, ax = plt.subplots(figsize=(6.6, 6.0), constrained_layout=True)
    image = ax.hexbin(
        frame["activity_sec1"],
        frame["activity_sec2"],
        gridsize=65,
        mincnt=1,
        bins="log",
        cmap="viridis",
    )
    limits = np.nanpercentile(
        frame[["activity_sec1", "activity_sec2"]].to_numpy(float),
        [0.25, 99.75],
    )
    if np.isfinite(limits).all() and limits[0] < limits[1]:
        ax.plot(limits, limits, color="black", linestyle="--", linewidth=1)
        ax.set_xlim(*limits)
        ax.set_ylim(*limits)
    ax.set_xlabel("Section 1 log activity − mean of 7 controls")
    ax.set_ylabel("Section 2 log activity − mean of 7 controls")
    ax.set_title(
        "Negative-control-centered Bayesian activity reproducibility\n"
        f"Shared target T7≥{t7_threshold:g} in both sections"
    )
    ax.text(
        0.03,
        0.97,
        (
            f"Spearman ρ={overall.spearman:.3f}\n"
            f"Pearson r={overall.pearson:.3f}\n"
            f"n={int(overall.n_points):,} subclass–cCRE pairs"
        ),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
    )
    fig.colorbar(image, ax=ax, label="Pair count (log scale)")
    save_figure(fig, figures / "section_bayesian_activity_correlation")


def plot_activity_correlation_violins(
    correlations: pd.DataFrame,
    figures: Path,
    t7_threshold: float,
    min_points: int = 15,
) -> None:
    definitions = (
        (
            "within_subclass_across_ccres",
            "Within cell type\n(across cCREs)",
            "#4C78A8",
        ),
        (
            "across_subclasses_per_ccre",
            "Across cell types\n(per cCRE)",
            "#F58518",
        ),
    )
    fig, ax = plt.subplots(figsize=(7.2, 5.8), constrained_layout=True)
    rng = np.random.default_rng(0)
    any_values = False
    for position, (axis_name, label, color) in enumerate(definitions):
        values = (
            correlations.loc[correlations["axis"].eq(axis_name), "spearman"]
            .dropna()
            .to_numpy(float)
        )
        if not len(values):
            continue
        any_values = True
        if len(values) >= 2 and np.unique(values).size >= 2:
            violin = ax.violinplot(
                [values],
                positions=[position],
                widths=0.7,
                showmedians=True,
            )
            violin["bodies"][0].set_facecolor(color)
            violin["bodies"][0].set_alpha(0.35)
        ax.scatter(
            position + rng.uniform(-0.10, 0.10, len(values)),
            values,
            s=14,
            alpha=0.5,
            color=color,
        )
        ax.scatter(
            [position],
            [np.median(values)],
            marker="D",
            s=45,
            color="black",
            zorder=5,
        )
        ax.text(
            position,
            1.04,
            f"n={len(values):,}\nmedian ρ={np.median(values):.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    if not any_values:
        raise ValueError("no unit-level activity correlations to plot")
    ax.axhline(0, color="grey", linestyle="--", linewidth=0.8)
    ax.set_xticks(np.arange(len(definitions)))
    ax.set_xticklabels([item[1] for item in definitions])
    ax.set_xlim(-0.6, len(definitions) - 0.4)
    ax.set_ylim(-1.05, 1.18)
    ax.set_ylabel("Section 1/Section 2 Spearman ρ")
    ax.set_title(
        "Activity-correlation distributions\n"
        "Activity minus the seven-control mean; "
        f"shared target T7≥{t7_threshold:g}; "
        f"≥{min_points} valid points per correlation"
    )
    save_figure(fig, figures / "section_bayesian_activity_correlation_violins")


def call_metrics(frame: pd.DataFrame) -> dict:
    sig1 = frame["significant_sec1"].astype(bool).to_numpy()
    sig2 = frame["significant_sec2"].astype(bool).to_numpy()
    n = len(frame)
    both = int((sig1 & sig2).sum())
    sec1_only = int((sig1 & ~sig2).sum())
    sec2_only = int((~sig1 & sig2).sum())
    neither = int((~sig1 & ~sig2).sum())
    n_sig1 = both + sec1_only
    n_sig2 = both + sec2_only
    concordance = (both + neither) / n if n else np.nan
    expected = (
        (n_sig1 / n) * (n_sig2 / n) + (1.0 - n_sig1 / n) * (1.0 - n_sig2 / n)
        if n
        else np.nan
    )
    kappa = (
        (concordance - expected) / (1.0 - expected) if n and expected < 1.0 else np.nan
    )
    return {
        "n_tested_common": n,
        "n_valid_ccres": n,
        "n_significant_sec1": n_sig1,
        "n_significant_sec2": n_sig2,
        "n_both_significant": both,
        "n_reproducible_significant_ccres": both,
        "n_sec1_only": sec1_only,
        "n_sec2_only": sec2_only,
        "n_both_nonsignificant": neither,
        "n_reproducible_nonsignificant_ccres": neither,
        "n_concordant": both + neither,
        "n_consistent_calls": both + neither,
        "n_reproducible_ccres": both + neither,
        "concordance": concordance,
        "call_concordance": concordance,
        "reproducibility": concordance,
        "significant_in_both_fraction": both / n if n else np.nan,
        "significant_jaccard": both / (both + sec1_only + sec2_only)
        if both + sec1_only + sec2_only
        else np.nan,
        "positive_agreement": 2 * both / (n_sig1 + n_sig2)
        if n_sig1 + n_sig2
        else np.nan,
        "cohen_kappa": kappa,
    }


def test_concordance_tables(
    tests: dict[str, pd.DataFrame],
    q_cutoff: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    keep = [
        "group",
        "cre",
        "activity_mean",
        "activity_vs_mean_control_mean",
        "effect_vs_mean_control_mean",
        "posterior_probability_above_mean_control",
        "p_right",
        "q_right",
    ]
    pair = tests["sec1"][keep].merge(
        tests["sec2"][keep],
        on=["group", "cre"],
        how="inner",
        suffixes=("_sec1", "_sec2"),
        validate="one_to_one",
    )
    pair["significant_sec1"] = pair["q_right_sec1"].le(q_cutoff)
    pair["significant_sec2"] = pair["q_right_sec2"].le(q_cutoff)
    pair["concordant"] = pair["significant_sec1"].eq(pair["significant_sec2"])
    pair["reproducible"] = pair["concordant"]
    pair["call_category"] = np.select(
        [
            pair["significant_sec1"] & pair["significant_sec2"],
            pair["significant_sec1"] & ~pair["significant_sec2"],
            ~pair["significant_sec1"] & pair["significant_sec2"],
        ],
        ["both_significant", "sec1_only", "sec2_only"],
        default="both_nonsignificant",
    )
    overall = pd.DataFrame([{"unit": "all", **call_metrics(pair)}])
    by_group = pd.DataFrame(
        [
            {"group": group, **call_metrics(frame)}
            for group, frame in pair.groupby("group", sort=False)
        ]
    )
    return pair, overall, by_group


def plot_reproducibility_by_celltype(
    overall: pd.DataFrame,
    by_group: pd.DataFrame,
    figures: Path,
    q_cutoff: float,
    t7_threshold: float,
) -> None:
    """Plot consistent sec1/sec2 calls over shared T7-valid cCREs."""
    ordered = by_group.sort_values(
        ["reproducibility", "n_valid_ccres", "group"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    positions = np.arange(len(ordered), dtype=float)
    fig, ax = plt.subplots(
        figsize=(max(10, 0.62 * max(len(ordered), 1) + 2), 6.5),
        constrained_layout=True,
    )
    bars = ax.bar(
        positions,
        ordered["reproducibility"],
        color="#4C78A8",
        width=0.78,
    )
    for bar, row in zip(bars, ordered.itertuples(index=False)):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            float(row.reproducibility) + 0.015,
            f"{int(row.n_reproducible_ccres)}/{int(row.n_valid_ccres)}",
            ha="center",
            va="bottom",
            rotation=90,
            fontsize=7,
        )

    overall_value = float(overall.iloc[0]["reproducibility"])
    ax.axhline(
        overall_value,
        color="#E45756",
        linestyle="--",
        linewidth=1.2,
        label=f"Overall: {overall_value:.3f}",
    )
    ax.set_xticks(positions)
    ax.set_xticklabels(ordered["group"], rotation=60, ha="right", fontsize=8)
    ax.set_xlim(-0.6, max(len(ordered) - 0.4, 0.4))
    maximum = float(
        np.nanmax(np.append(ordered["reproducibility"].to_numpy(float), overall_value))
    )
    y_top = 1.12 if maximum >= 0.9 else max(0.1, maximum * 1.35)
    ax.set_ylim(0, y_top)
    ax.set_ylabel("Reproducible cCREs / valid cCREs")
    ax.set_xlabel("Cell type")
    ax.set_title(
        "Reproducibility of cCRE calls by cell type\n"
        f"Same call in both sections at BH q≤{q_cutoff:g} "
        "(both significant or both non-significant) / "
        f"target T7≥{t7_threshold:g} in both sections"
    )
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    save_figure(fig, figures / "section_bayesian_reproducibility_by_celltype")


def plot_test_concordance(
    overall: pd.DataFrame,
    by_group: pd.DataFrame,
    figures: Path,
    q_cutoff: float,
    t7_threshold: float,
) -> None:
    metrics = overall.iloc[0]
    counts = np.asarray(
        [
            [metrics.n_both_nonsignificant, metrics.n_sec1_only],
            [metrics.n_sec2_only, metrics.n_both_significant],
        ],
        dtype=int,
    )
    total = int(counts.sum())
    annotations = np.asarray(
        [
            [
                f"{counts[row, column]:,}\n({counts[row, column] / total:.1%})"
                if total
                else "0"
                for column in range(2)
            ]
            for row in range(2)
        ]
    )

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(11.5, 5.2),
        constrained_layout=True,
        gridspec_kw={"width_ratios": [1.0, 1.35]},
    )
    sns.heatmap(
        counts,
        annot=annotations,
        fmt="",
        cmap="Blues",
        cbar=False,
        linewidths=1,
        linecolor="white",
        xticklabels=["Not significant", "Significant"],
        yticklabels=["Not significant", "Significant"],
        ax=axes[0],
    )
    axes[0].set_xlabel("Section 1 call")
    axes[0].set_ylabel("Section 2 call")
    axes[0].set_title(
        "Common-test call table\n"
        f"Concordance={metrics.concordance:.3f}, "
        f"κ={metrics.cohen_kappa:.3f}"
    )

    rng = np.random.default_rng(0)
    series = [
        ("All-call concordance", by_group["concordance"].dropna(), "#4C78A8"),
        (
            "Significant-call Jaccard",
            by_group["significant_jaccard"].dropna(),
            "#F58518",
        ),
    ]
    for position, (label, values, color) in enumerate(series):
        array = values.to_numpy(float)
        if len(array):
            violin = axes[1].violinplot(
                [array],
                positions=[position],
                widths=0.65,
                showmedians=True,
            )
            violin["bodies"][0].set_facecolor(color)
            violin["bodies"][0].set_alpha(0.3)
            axes[1].scatter(
                position + rng.uniform(-0.10, 0.10, len(array)),
                array,
                s=10,
                alpha=0.45,
                color=color,
            )
        overall_value = (
            metrics.concordance if position == 0 else metrics.significant_jaccard
        )
        if np.isfinite(overall_value):
            axes[1].scatter(
                [position],
                [overall_value],
                marker="D",
                s=55,
                color="black",
                zorder=5,
                label="Overall" if position == 0 else None,
            )
    axes[1].set_xticks([0, 1])
    axes[1].set_xticklabels([item[0] for item in series], rotation=12, ha="right")
    axes[1].set_xlim(-0.6, 1.6)
    axes[1].set_ylim(0, 1.03)
    axes[1].set_ylabel("Concordance")
    axes[1].set_title(f"Across {len(by_group):,} common subclasses")
    axes[1].legend(frameon=False, fontsize=8, loc="lower left")

    fig.suptitle(
        "Bayesian mean-negative-control test concordance\n"
        f"BH q≤{q_cutoff:g}; shared pair universe with target and "
        f"combined-control T7≥{t7_threshold:g} in both sections",
        fontsize=12,
    )
    save_figure(fig, figures / "section_bayesian_test_concordance")


def main() -> None:
    args = parse_args()
    if args.t7_threshold < 0:
        raise ValueError("--t7-threshold must be nonnegative")
    if not 0 < args.q_cutoff <= 1:
        raise ValueError("--q-cutoff must be in (0, 1]")

    figures = args.output_dir / "figures"
    tables = args.output_dir / "tables"
    figures.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="talk")

    activity: dict[str, pd.DataFrame] = {}
    tests: dict[str, pd.DataFrame] = {}
    section_metadata = {}
    for section in SECTIONS:
        root = args.sections_dir / section / args.model_dirname
        activity[section], tests[section], section_metadata[section] = load_section(
            root=root,
            section=section,
            h5ad=args.h5ad,
            t7_threshold=args.t7_threshold,
            effect_threshold=args.effect_threshold,
        )
        log(
            f"[section reproducibility] {section}: "
            f"{section_metadata[section]['n_eligible_tests_before_shared_filter']:,} "
            "section-eligible tests before the shared filter"
        )

    tests, shared_eligibility = restrict_to_shared_eligibility(tests)
    for section in SECTIONS:
        section_metadata[section]["n_shared_eligible_tests"] = int(len(tests[section]))
    log(
        "[section reproducibility] shared T7-eligible pair universe: "
        f"{len(shared_eligibility):,} pairs"
    )

    activity_pair, activity_correlations = activity_correlation_tables(
        activity, shared_eligibility, args.min_points
    )
    test_pair, test_overall, test_by_group = test_concordance_tables(
        tests, args.q_cutoff
    )

    activity_pair.to_csv(tables / "section_bayesian_activity_pairs.csv.gz", index=False)
    activity_correlations.to_csv(
        tables / "section_bayesian_activity_correlations.csv", index=False
    )
    shared_eligibility.to_csv(
        tables / "section_bayesian_shared_t7_eligibility.csv.gz", index=False
    )
    pd.concat(tests.values(), ignore_index=True).to_csv(
        tables / "section_bayesian_mean_negative_control_tests.csv.gz",
        index=False,
    )
    test_pair.to_csv(
        tables / "section_bayesian_test_concordance_pairs.csv.gz", index=False
    )
    test_overall.to_csv(
        tables / "section_bayesian_test_concordance_overall.csv", index=False
    )
    test_by_group.to_csv(
        tables / "section_bayesian_test_concordance_by_subclass.csv", index=False
    )
    test_by_group[
        [
            "group",
            "n_valid_ccres",
            "n_reproducible_ccres",
            "n_reproducible_significant_ccres",
            "n_reproducible_nonsignificant_ccres",
            "reproducibility",
            "n_consistent_calls",
            "call_concordance",
        ]
    ].to_csv(
        tables / "section_bayesian_reproducibility_by_celltype.csv",
        index=False,
    )

    plot_activity_correlation(
        activity_pair,
        activity_correlations,
        figures,
        args.t7_threshold,
    )
    plot_activity_correlation_violins(
        activity_correlations,
        figures,
        args.t7_threshold,
        args.min_points,
    )
    plot_test_concordance(
        test_overall,
        test_by_group,
        figures,
        args.q_cutoff,
        args.t7_threshold,
    )
    plot_reproducibility_by_celltype(
        test_overall,
        test_by_group,
        figures,
        args.q_cutoff,
        args.t7_threshold,
    )

    write_json(
        tables / "run_summary.json",
        {
            "model": EXPECTED_MODEL,
            "sections": section_metadata,
            "h5ad": str(args.h5ad),
            "activity_definition": (
                "within each posterior draw, target log_gamma minus the mean "
                "log_gamma of all seven ordinary negative controls, then "
                "averaged across draws; used for every activity correlation"
            ),
            "test_definition": (
                "within each posterior draw, target log_gamma minus the mean "
                "log_gamma of all seven ordinary negative controls"
            ),
            "reproducibility_definition": (
                "a valid target-subclass pair is reproducible when it is "
                "assigned the same significance call in both sections (both "
                "significant or both non-significant); per-cell-type "
                "reproducibility is consistent cCRE calls divided by shared "
                "T7-eligible cCREs"
            ),
            "p_right_definition": (
                "posterior fraction of target-minus-mean-control contrasts "
                "<= the effect threshold"
            ),
            "multiple_testing": (
                "BH independently within each section, using the identical "
                "shared target-subclass pair universe"
            ),
            "shared_eligibility_definition": (
                "a target-subclass pair is retained only when target T7 >= the "
                "threshold in both sections and the combined T7 of all seven "
                "ordinary negative controls >= the threshold in both sections"
            ),
            "t7_threshold": args.t7_threshold,
            "min_points_per_correlation": args.min_points,
            "effect_threshold": args.effect_threshold,
            "q_cutoff": args.q_cutoff,
            "n_common_activity_pairs": int(activity_pair["measured"].sum()),
            "n_shared_eligible_pairs": int(len(shared_eligibility)),
            "n_common_tests": int(len(test_pair)),
            "outputs": {
                "figures": [
                    "section_bayesian_activity_correlation.pdf",
                    "section_bayesian_activity_correlation.png",
                    "section_bayesian_activity_correlation_violins.pdf",
                    "section_bayesian_activity_correlation_violins.png",
                    "section_bayesian_test_concordance.pdf",
                    "section_bayesian_test_concordance.png",
                    "section_bayesian_reproducibility_by_celltype.pdf",
                    "section_bayesian_reproducibility_by_celltype.png",
                ],
                "tables_dir": str(tables),
            },
        },
    )
    log(
        f"[section reproducibility] wrote 4 figure pairs and tables to "
        f"{args.output_dir}"
    )


if __name__ == "__main__":
    main()
