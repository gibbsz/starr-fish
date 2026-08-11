#!/usr/bin/env python3
"""Scatter raw original-versus-new counts for every cCRE-cell-type pair.

Each dot is one subclass-cCRE pair. Both panels show raw transcript counts
summed over the cells of the subclass in each experiment: the T7 transcript
species on the left and the cCRE transcript species on the right. Two scopes are
drawn: every shared non-blacklisted pair, and the T7-filtered overlap universe
used by the activity comparison.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Final

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


HERE: Final[Path] = Path(__file__).resolve()
ANALYSIS_DIR: Final[Path] = HERE.parent.parent
REPO_ROOT: Final[Path] = ANALYSIS_DIR.parents[1]
ORIGIN_CODE: Final[Path] = REPO_ROOT / "revision" / "bayesian_vs_fold_change" / "code"
if str(ORIGIN_CODE) not in sys.path:
    sys.path.insert(0, str(ORIGIN_CODE))

from test_individual_negative_control_loo_empirical_fdr import (  # noqa: E402
    decode_strings,
    normalize_labels,
)


DEFAULT_ORIGIN_H5AD: Final[Path] = (
    REPO_ROOT
    / "revision"
    / "Data"
    / "scdata_5_28_2025_BRBB500gn_final_CRE_T7CRE_NEWNEW.h5ad"
)
DEFAULT_NEW_H5AD: Final[Path] = (
    REPO_ROOT
    / "revision"
    / "Data"
    / "scdata_07_29_2026_SFv8_low_dose_final_CRE_T7.h5ad"
)
DEFAULT_COMPARISON_DIR: Final[Path] = ANALYSIS_DIR / "results" / "comparison"
NEW_RESULTS: Final[Path] = REPO_ROOT / "revision" / "Bayes_NewData"
DEFAULT_NEW_BAYES: Final[Path] = NEW_RESULTS / "bayesian"
KEY: Final[list[str]] = ["group", "cre"]
MATRICES: Final[tuple[tuple[str, str, str], ...]] = (
    ("t7", "T7CRE", "T7 transcript counts"),
    ("cre", "CRE", "cCRE transcript counts"),
)
POINT_COLOR: Final[str] = "#2F6F8F"
CONTROL_COLOR: Final[str] = "#D55E00"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--origin-h5ad", type=Path, default=DEFAULT_ORIGIN_H5AD)
    parser.add_argument("--new-h5ad", type=Path, default=DEFAULT_NEW_H5AD)
    parser.add_argument("--comparison-dir", type=Path, default=DEFAULT_COMPARISON_DIR)
    parser.add_argument("--new-bayes", type=Path, default=DEFAULT_NEW_BAYES)
    parser.add_argument("--t7-threshold", type=float, default=50)
    return parser.parse_args()


def token(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def read_shared_axes(comparison_dir: Path) -> tuple[pd.Index, pd.Index]:
    """Return the shared subclasses and non-blacklisted cCREs of both fits."""
    path = comparison_dir / "tables" / "all_common_activity.csv.gz"
    if not path.exists():
        raise FileNotFoundError(f"Missing shared-pair activity table: {path}")
    frame = pd.read_csv(path, usecols=["group", "cre", "activity_pair_scope"])
    frame = frame.loc[frame["activity_pair_scope"].eq("both")]
    groups = pd.Index(sorted(frame["group"].astype(str).unique()), name="group")
    cres = pd.Index(sorted(frame["cre"].astype(str).unique()), name="cre")
    if len(frame) != len(groups) * len(cres):
        raise ValueError(
            f"{path} is not a complete subclass x cCRE product "
            f"({len(frame)} rows vs {len(groups) * len(cres)} expected)"
        )
    return groups, cres


def read_overlap_pairs(comparison_dir: Path, t7_threshold: float) -> pd.DataFrame:
    path = (
        comparison_dir
        / "tables"
        / f"overlap_t7_ge{token(t7_threshold)}_pair_comparison.csv.gz"
    )
    if not path.exists():
        raise FileNotFoundError(f"Missing overlap pair table: {path}")
    frame = pd.read_csv(
        path,
        usecols=[*KEY, "origin_target_t7_total", "new_target_t7_total"],
    )
    frame[KEY] = frame[KEY].astype(str)
    if frame.duplicated(subset=KEY).any():
        raise ValueError(f"{path} contains duplicated group/cre keys")
    return frame


def read_overlap_control_pairs(
    comparison_dir: Path, t7_threshold: float
) -> pd.DataFrame:
    """Return the negative-control pairs of the overlap-filtered subclasses."""
    path = (
        comparison_dir
        / "tables"
        / f"overlap_t7_ge{token(t7_threshold)}_negative_control_activity.csv"
    )
    if not path.exists():
        raise FileNotFoundError(f"Missing overlap negative-control table: {path}")
    frame = pd.read_csv(path, usecols=KEY)
    frame[KEY] = frame[KEY].astype(str)
    if frame.duplicated(subset=KEY).any():
        raise ValueError(f"{path} contains duplicated group/cre keys")
    return frame


def grouped_pair_counts(
    h5ad: Path, groups: pd.Index, cres: pd.Index
) -> dict[str, pd.DataFrame]:
    """Sum each obsm count matrix over the cells of every requested subclass."""
    if not h5ad.exists():
        raise FileNotFoundError(h5ad)
    totals: dict[str, pd.DataFrame] = {}
    with h5py.File(h5ad, "r") as handle:
        subclass = handle["obs"]["subclass_name"]
        categories = normalize_labels(decode_strings(subclass["categories"][...]))
        codes = subclass["codes"][...].astype(np.int64)
        valid = codes >= 0
        codes = codes[valid]
        lookup = {name: index for index, name in enumerate(categories)}
        missing_groups = sorted(set(groups.astype(str)) - set(lookup))
        if missing_groups:
            raise ValueError(f"{h5ad} is missing subclasses: {missing_groups}")
        rows = np.asarray([lookup[name] for name in groups.astype(str)], dtype=np.int64)
        for label, obsm_key, _ in MATRICES:
            matrix = handle["obsm"][obsm_key]
            missing_cres = sorted(set(cres.astype(str)) - set(matrix.keys()))
            if missing_cres:
                raise ValueError(f"{h5ad}:obsm/{obsm_key} is missing: {missing_cres}")
            values = np.empty((len(groups), len(cres)), dtype=np.float64)
            for column, cre in enumerate(cres.astype(str)):
                per_cell = matrix[cre][...].astype(np.float64, copy=False)[valid]
                grouped = np.bincount(
                    codes, weights=per_cell, minlength=len(categories)
                )
                values[:, column] = grouped[rows]
            totals[label] = pd.DataFrame(values, index=groups, columns=cres)
            print(f"[counts] {h5ad.name}: summed obsm/{obsm_key}", flush=True)
    return totals


def long_counts(
    origin: dict[str, pd.DataFrame], new: dict[str, pd.DataFrame]
) -> pd.DataFrame:
    frames = []
    for label, _, _ in MATRICES:
        for dataset, totals in (("origin", origin), ("new", new)):
            stacked = totals[label].stack()
            stacked.index.names = KEY
            frames.append(stacked.rename(f"{dataset}_{label}"))
    return pd.concat(frames, axis=1).reset_index()


def validate_t7_against_tests(counts: pd.DataFrame, overlap: pd.DataFrame) -> None:
    """Confirm the streamed T7 totals reproduce the published test-table totals."""
    merged = overlap.merge(counts, on=KEY, how="left", validate="one_to_one")
    if merged[["origin_t7", "new_t7"]].isna().any().to_numpy().any():
        raise ValueError("Overlap pairs are missing from the streamed count matrices")
    for dataset in ("origin", "new"):
        difference = np.abs(
            merged[f"{dataset}_t7"].to_numpy(float)
            - merged[f"{dataset}_target_t7_total"].to_numpy(float)
        )
        if difference.max() > 0.5:
            raise ValueError(
                f"{dataset} T7 totals disagree with the test table "
                f"(max |difference| = {difference.max():.3f})"
            )


def panel_statistics(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]
    both_zero = int(((x == 0) & (y == 0)).sum())
    positive = (x > 0) & (y > 0)
    return {
        "n_pairs": int(len(x)),
        "n_pairs_zero_in_both": both_zero,
        "pearson_r_log1p": float(pearsonr(np.log1p(x), np.log1p(y)).statistic),
        "spearman_rho_raw": float(spearmanr(x, y).statistic),
        "origin_total": float(x.sum()),
        "new_total": float(y.sum()),
        "new_over_origin_total_ratio": float(y.sum() / x.sum()) if x.sum() else np.nan,
        "median_new_over_origin_ratio": float(np.median(y[positive] / x[positive]))
        if positive.any()
        else np.nan,
    }


def plot_counts(
    counts: pd.DataFrame,
    controls: set[str],
    output_stem: Path,
    *,
    scope_title: str,
) -> dict[str, dict[str, float]]:
    dense = len(counts) > 5_000
    statistics: dict[str, dict[str, float]] = {}
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 5.8), layout="constrained")
    control_mask = counts["cre"].isin(controls).to_numpy()
    for ax, (label, _, axis_label) in zip(axes, MATRICES):
        x = counts[f"origin_{label}"].to_numpy(float)
        y = counts[f"new_{label}"].to_numpy(float)
        statistics[label] = panel_statistics(x, y)
        limit_high = float(max(x.max(), y.max())) + 1.0
        limits = (0.9, limit_high * 1.35)
        ax.plot(limits, limits, linestyle="--", linewidth=1.2, color="black", zorder=4)
        ax.scatter(
            x[~control_mask] + 1.0,
            y[~control_mask] + 1.0,
            s=7 if dense else 20,
            alpha=0.12 if dense else 0.55,
            color=POINT_COLOR,
            edgecolors="none",
            rasterized=True,
            zorder=2,
            label=f"cCRE–cell-type pairs (n={int((~control_mask).sum()):,})",
        )
        if control_mask.any():
            ax.scatter(
                x[control_mask] + 1.0,
                y[control_mask] + 1.0,
                s=9 if dense else 26,
                alpha=0.35 if dense else 0.8,
                color=CONTROL_COLOR,
                edgecolors="none",
                rasterized=True,
                zorder=3,
                label=(
                    "Negative-control cCREs "
                    f"(n={int(control_mask.sum()):,})"
                ),
            )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(limits)
        ax.set_ylim(limits)
        ax.set_aspect("equal")
        ax.set_xlabel(f"Original {axis_label} + 1")
        ax.set_ylabel(f"New low-dose {axis_label} + 1")
        ax.set_title(
            f"{axis_label}\n"
            f"Pearson r(log1p) = {statistics[label]['pearson_r_log1p']:.3f}; "
            f"Spearman ρ = {statistics[label]['spearman_rho_raw']:.3f}; "
            f"new/original total = "
            f"{statistics[label]['new_over_origin_total_ratio']:.3f}",
            fontsize=9.5,
        )
        ax.grid(color="0.90", linewidth=0.7, zorder=0)
        ax.legend(frameon=False, fontsize=7.5, loc="upper left")

    fig.suptitle(
        "Raw per-subclass count concordance, original versus new low-dose\n"
        f"{scope_title}\n{len(counts):,} cCRE–cell-type pairs",
        fontsize=10.5,
        y=1.10,
    )
    for suffix in ("png", "pdf"):
        fig.savefig(output_stem.with_suffix(f".{suffix}"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    return statistics


def main() -> None:
    args = parse_args()
    tables_dir = args.comparison_dir / "tables"
    figures_dir = args.comparison_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    groups, target_cres = read_shared_axes(args.comparison_dir)
    overlap = read_overlap_pairs(args.comparison_dir, args.t7_threshold)
    overlap_controls = read_overlap_control_pairs(
        args.comparison_dir, args.t7_threshold
    )
    controls = set(
        pd.read_csv(args.new_bayes / "negative_controls.csv")
        .iloc[:, 0]
        .astype(str)
        .tolist()
    )
    if controls & set(target_cres):
        raise ValueError("Negative controls must not appear among the target cCREs")
    cres = pd.Index(sorted(set(target_cres) | controls), name="cre")

    counts = long_counts(
        grouped_pair_counts(args.origin_h5ad, groups, cres),
        grouped_pair_counts(args.new_h5ad, groups, cres),
    )
    validate_t7_against_tests(counts, overlap)

    overlap_keys = pd.concat([overlap[KEY], overlap_controls], ignore_index=True)
    overlap_counts = counts.merge(
        overlap_keys, on=KEY, how="inner", validate="one_to_one"
    )
    if len(overlap_counts) != len(overlap_keys):
        raise ValueError("Overlap universe is not fully covered by the shared axes")

    threshold_token = token(args.t7_threshold)
    scopes = {
        "all_shared_pairs": (
            counts,
            f"All shared non-blacklisted pairs: {len(groups):,} subclasses × "
            f"{len(cres):,} cCREs ({len(target_cres):,} targets + "
            f"{len(controls)} negative controls)",
        ),
        f"overlap_t7_ge{threshold_token}": (
            overlap_counts,
            f"Overlap filter: target T7 ≥ {args.t7_threshold:g} and control T7 ≥ "
            f"{args.t7_threshold:g} in both datasets, plus the negative-control "
            "pairs of those subclasses",
        ),
    }
    manifest: dict[str, object] = {
        "origin_h5ad": str(args.origin_h5ad),
        "new_h5ad": str(args.new_h5ad),
        "n_shared_subclasses": int(len(groups)),
        "n_target_cres": int(len(target_cres)),
        "n_cres_including_controls": int(len(cres)),
        "t7_threshold": float(args.t7_threshold),
        "negative_controls": sorted(controls),
        "scopes": {},
    }
    for scope, (frame, scope_title) in scopes.items():
        stem = f"raw_count_concordance_scatter_{scope}"
        statistics = plot_counts(
            frame, controls, figures_dir / stem, scope_title=scope_title
        )
        manifest["scopes"][scope] = {
            "figure_stem": stem,
            "scope_title": scope_title,
            "panels": statistics,
        }

    values_path = tables_dir / "raw_count_concordance_pair_counts.csv.gz"
    output = counts.copy()
    output["negative_control_cre"] = output["cre"].isin(controls)
    output[f"overlap_t7_ge{threshold_token}"] = output.set_index(KEY).index.isin(
        overlap_keys.set_index(KEY).index
    )
    output.to_csv(values_path, index=False)
    manifest["values_table"] = str(values_path)

    (tables_dir / "raw_count_concordance_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
