#!/usr/bin/env python3
"""Plot matched original/new activity and mean-plus-one-SD test diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve()
ANALYSIS_DIR = HERE.parent.parent
REPO_ROOT = ANALYSIS_DIR.parents[1]
DEFAULT_COMPARISON_DIR = ANALYSIS_DIR / "results" / "comparison"
DEFAULT_SHARED_TESTS = (
    DEFAULT_COMPARISON_DIR
    / "tables"
    / "shared_pair_mean_plus_1sd_comparison_t7_ge50.csv.gz"
)
DEFAULT_ORIGIN_H5AD = (
    REPO_ROOT
    / "revision"
    / "Data"
    / "scdata_5_28_2025_BRBB500gn_final_CRE_T7CRE_NEWNEW.h5ad"
)
DEFAULT_LIBRARY_COUNTS = (
    REPO_ROOT / "STARRFISH_in_vivo" / "Data" / "SFv8_400CRE_nanopore_counts.csv"
)
DEFAULT_ATAC_PEAKS = (
    REPO_ROOT / "STARRFISH_in_vivo" / "Data" / "cre_atac_peaks.csv"
)
DEFAULT_STEM = "origin_vs_new_mean_plus_1sd_activity_heatmap_t7_ge50"
DEFAULT_OVERLAP_TESTS = (
    DEFAULT_COMPARISON_DIR / "tables" / "overlap_t7_ge50_pair_comparison.csv.gz"
)
DEFAULT_CONTROL_ACTIVITY = (
    DEFAULT_COMPARISON_DIR
    / "tables"
    / "overlap_t7_ge50_negative_control_activity.csv"
)
RUNS = ("origin", "new")
RUN_LABELS = {"origin": "Original run", "new": "New low-dose run"}

# The two test families differ only in their control reference, so the layout is
# shared and only the column names, the control-spread strip, and the labels
# change.
REFERENCE_SPECS: dict[str, dict[str, object]] = {
    "mean_plus_1sd": {
        "tests": DEFAULT_SHARED_TESTS,
        "stem": DEFAULT_STEM,
        "effect_prefix": "effect_vs_control_reference",
        "pair_sd_column": "negative_control_activity_sd_mean",
        "sd_strip_label": "Control\nSD",
        "sd_colorbar_label": "mean draw-wise SD among 7 controls",
        "universe_label": "mean+1-SD test universe",
        "effect_summary_key": "mean_effect_vs_mean_plus_1sd_reference",
    },
    "mean": {
        "tests": DEFAULT_OVERLAP_TESTS,
        "stem": "origin_vs_new_mean_control_activity_heatmap_t7_ge50",
        "effect_prefix": "effect_vs_mean_control",
        "pair_sd_column": None,
        "sd_strip_label": "Control\nspread",
        "sd_colorbar_label": "SD across the 7 control posterior-mean activities",
        "universe_label": "mean-control test universe",
        "effect_summary_key": "mean_effect_vs_mean_control_reference",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--control-reference",
        choices=tuple(REFERENCE_SPECS),
        default="mean_plus_1sd",
        help="Test family whose columns and control-spread strip are plotted.",
    )
    parser.add_argument(
        "--shared-tests",
        type=Path,
        default=None,
        help="Pair table; defaults to the table of the chosen control reference.",
    )
    parser.add_argument(
        "--negative-control-activity",
        type=Path,
        default=DEFAULT_CONTROL_ACTIVITY,
        help=(
            "Per-subclass negative-control activity table, used for the control-"
            "spread strip when --control-reference mean is selected."
        ),
    )
    parser.add_argument(
        "--restrict-calls",
        type=Path,
        default=None,
        help=(
            "Optional call table (group, cre, <basis>_call_status) restricting the "
            "displayed pairs, e.g. the BH call table of the concordance analysis."
        ),
    )
    parser.add_argument(
        "--restrict-status-column",
        default="bh_q_call_status",
        help="Status column of --restrict-calls used for the restriction.",
    )
    parser.add_argument(
        "--restrict-status",
        default="both_significant,neither_significant",
        help="Comma-separated statuses retained from --restrict-status-column.",
    )
    parser.add_argument("--origin-h5ad", type=Path, default=DEFAULT_ORIGIN_H5AD)
    parser.add_argument("--library-counts", type=Path, default=DEFAULT_LIBRARY_COUNTS)
    parser.add_argument("--atac-peaks", type=Path, default=DEFAULT_ATAC_PEAKS)
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
    parser.add_argument(
        "--stem",
        default=None,
        help="Output stem; defaults to the stem of the chosen control reference.",
    )
    parser.add_argument("--q-cutoff", type=float, default=0.05)
    parser.add_argument("--nominal-p-cutoff", type=float, default=0.05)
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def validate_shared_tests(shared: pd.DataFrame, spec: dict[str, object]) -> None:
    effect = str(spec["effect_prefix"])
    pair_sd = spec["pair_sd_column"]
    required = {"group", "cre", "common_q_call_status"}
    for run in RUNS:
        required.update(
            {
                f"{run}_activity_mean",
                f"{run}_mean_negative_control_activity_mean",
                f"{run}_{effect}_mean",
                f"{run}_{effect}_lo90",
                f"{run}_p_right",
                f"{run}_q_common_universe",
                f"{run}_significant_common_q",
                f"{run}_target_t7_total",
            }
        )
        if pair_sd is not None:
            required.add(f"{run}_{pair_sd}")
    missing = sorted(required.difference(shared.columns))
    if missing:
        raise ValueError(f"Shared test table is missing columns: {missing}")
    if shared.duplicated(["group", "cre"]).any():
        duplicated = shared.loc[
            shared.duplicated(["group", "cre"], keep=False), ["group", "cre"]
        ]
        raise ValueError(
            "Shared test table has duplicate group-cCRE pairs:\n"
            + duplicated.head(10).to_string(index=False)
        )


def subclass_numeric_order(h5ad_path: Path, observed: pd.Index) -> list[str]:
    """Match the numbered subclass order used by the reference heatmap."""
    try:
        import anndata as ad
    except ImportError as exc:  # pragma: no cover - environment-specific message
        raise ImportError(
            "Numeric subclass ordering requires anndata; run this script in the scvi "
            "environment used by the origin/new workflow."
        ) from exc

    data = ad.read_h5ad(h5ad_path, backed="r")
    try:
        source = data.obs["subclass_name"].astype(str)
    finally:
        data.file.close()
    order = pd.DataFrame(
        {
            "source": source,
            "group": source.str.replace(r"^\d+\s+", "", regex=True).str.replace(
                "/", "-", regex=False
            ),
            "prefix": pd.to_numeric(
                source.str.extract(r"^(\d+)\s+", expand=False), errors="coerce"
            ),
        }
    )
    prefixes = (
        order.dropna(subset=["prefix"])
        .assign(prefix=lambda frame: frame["prefix"].astype(int))
        .sort_values(["prefix", "group"])
        .drop_duplicates("group", keep="first")
        .set_index("group")["prefix"]
    )
    return sorted(
        observed.astype(str),
        key=lambda group: (prefixes.get(group, 10**9), str(group)),
    )


def pivot(
    shared: pd.DataFrame,
    column: str,
    row_order: list[str],
    col_order: list[str],
) -> pd.DataFrame:
    return (
        shared.pivot(index="group", columns="cre", values=column)
        .reindex(index=row_order, columns=col_order)
    )


def read_atac_mask(
    path: Path, row_order: list[str], col_order: list[str]
) -> pd.DataFrame:
    atac = pd.read_csv(path, index_col=0)
    atac.index = atac.index.astype(str).str.replace("/", "-", regex=False)
    atac.columns = atac.columns.astype(str)
    return (
        atac.reindex(index=row_order, columns=col_order)
        .gt(0.5)
        .fillna(False)
        .astype(bool)
    )


def read_library_counts(path: Path, col_order: list[str]) -> pd.Series:
    counts = pd.read_csv(path)
    if not {"CRE", "counts"}.issubset(counts.columns):
        raise ValueError(f"Expected CRE and counts columns in {path}")
    return (
        counts.assign(CRE=lambda frame: frame["CRE"].astype(str))
        .drop_duplicates("CRE", keep="first")
        .set_index("CRE")["counts"]
        .astype(float)
        .reindex(col_order)
    )


def add_boxes(ax: plt.Axes, mask: np.ndarray) -> int:
    rows, cols = np.nonzero(mask)
    boxes = [
        Rectangle((col - 0.5, row - 0.5), 1.0, 1.0)
        for row, col in zip(rows, cols)
    ]
    if boxes:
        ax.add_collection(
            PatchCollection(
                boxes,
                facecolor="none",
                edgecolor="black",
                linewidth=0.55,
                zorder=4,
            )
        )
    return int(len(boxes))


def add_markers(
    ax: plt.Axes,
    nominal: np.ndarray,
    significant: np.ndarray,
) -> tuple[int, int]:
    """Mark BH-significant cells only; nominal-only cells are counted, not drawn."""
    nominal_only_count = int(np.count_nonzero(nominal & ~significant))
    sig_rows, sig_cols = np.nonzero(significant)
    if len(sig_rows):
        ax.scatter(
            sig_cols,
            sig_rows,
            marker="*",
            s=24,
            facecolors="white",
            edgecolors="black",
            linewidths=0.3,
            zorder=6,
        )
    return nominal_only_count, int(len(sig_rows))


def run_summary(
    shared: pd.DataFrame,
    run: str,
    q_cutoff: float,
    spec: dict[str, object],
    control_sd_by_group: pd.Series,
) -> dict:
    prefix = str(spec["effect_prefix"])
    p_values = shared[f"{run}_p_right"].astype(float)
    q_values = shared[f"{run}_q_common_universe"].astype(float)
    significant = q_values.le(q_cutoff)
    target_t7 = shared[f"{run}_target_t7_total"].astype(float)
    control_sd = (
        shared["group"].map(control_sd_by_group).astype(float)
    )
    centered = (
        shared[f"{run}_activity_mean"].astype(float)
        - shared[f"{run}_mean_negative_control_activity_mean"].astype(float)
    )
    effect = shared[f"{run}_{prefix}_mean"].astype(float)
    ci90_width = (
        shared[f"{run}_{prefix}_hi90"].astype(float)
        - shared[f"{run}_{prefix}_lo90"].astype(float)
    )
    n_nominal = int(p_values.le(0.05).sum())
    return {
        "n_significant_common_bh": int(significant.sum()),
        "n_nominal_p_le_0.05": n_nominal,
        "n_effect_lo90_above_zero": int(
            shared[f"{run}_{prefix}_lo90"].astype(float).gt(0).sum()
        ),
        "min_p_right": float(p_values.min()),
        "min_q_common_universe": float(q_values.min()),
        "bh_cutoff_at_nominal_count_rank": float(
            q_cutoff * n_nominal / len(shared)
        ),
        "median_target_t7": float(target_t7.median()),
        "mean_target_t7": float(target_t7.mean()),
        "median_negative_control_sd": float(control_sd.median()),
        "mean_negative_control_sd": float(control_sd.mean()),
        "median_effect_ci90_width": float(ci90_width.median()),
        "mean_effect_ci90_width": float(ci90_width.mean()),
        "mean_activity_minus_control_mean": float(centered.mean()),
        str(spec["effect_summary_key"]): float(effect.mean()),
    }


def control_spread_by_group(
    path: Path, row_order: list[str], n_controls: int = 7
) -> dict[str, pd.Series]:
    """Return each run's SD across the control posterior means per subclass."""
    if not path.exists():
        raise FileNotFoundError(f"Missing negative-control activity table: {path}")
    controls = pd.read_csv(path)
    required = {"group", "cre", "origin_centered_activity_mean", "new_centered_activity_mean"}
    missing = sorted(required.difference(controls.columns))
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")
    controls["group"] = controls["group"].astype(str)
    counts = controls.groupby("group")["cre"].nunique()
    if not counts.eq(n_controls).all():
        bad = counts[counts.ne(n_controls)].to_dict()
        raise ValueError(f"{path} does not have {n_controls} controls per subclass: {bad}")
    return {
        run: controls.groupby("group")[f"{run}_centered_activity_mean"]
        .std(ddof=1)
        .reindex(row_order)
        .astype(float)
        for run in RUNS
    }


def pair_control_sd_by_group(
    shared: pd.DataFrame, column: str, row_order: list[str]
) -> dict[str, pd.Series]:
    """Return each run's per-subclass control SD carried by the pair table."""
    output = {}
    for run in RUNS:
        by_group = shared.groupby("group", sort=False)[f"{run}_{column}"]
        spread = by_group.max() - by_group.min()
        if spread.gt(1e-6).any():
            bad = spread[spread.gt(1e-6)].index.tolist()
            raise ValueError(f"{run} control SD varies within subclasses: {bad}")
        output[run] = by_group.first().reindex(row_order).astype(float)
    return output


def diagnostic_values(
    shared: pd.DataFrame, atac: pd.DataFrame, spec: dict[str, object]
) -> pd.DataFrame:
    prefix = str(spec["effect_prefix"])
    values = shared[["group", "cre", "common_q_call_status"]].copy()
    suffixes = [
        f"{prefix}_mean",
        f"{prefix}_lo90",
        f"{prefix}_hi90",
        "posterior_probability_above_control_reference",
        "posterior_probability_above_mean_control",
        "p_right",
        "q_common_universe",
        "significant_common_q",
        "target_t7_total",
    ]
    if spec["pair_sd_column"] is not None:
        suffixes.insert(0, str(spec["pair_sd_column"]))
    for run in RUNS:
        values[f"{run}_activity_minus_control_mean"] = (
            shared[f"{run}_activity_mean"].astype(float)
            - shared[f"{run}_mean_negative_control_activity_mean"].astype(float)
        )
        for suffix in suffixes:
            column = f"{run}_{suffix}"
            if column in shared.columns:
                values[column] = shared[column].to_numpy()
    values["new_minus_origin_centered_activity"] = (
        values["new_activity_minus_control_mean"]
        - values["origin_activity_minus_control_mean"]
    )
    if spec["pair_sd_column"] is not None:
        sd_column = str(spec["pair_sd_column"])
        values["new_minus_origin_control_sd"] = (
            values[f"new_{sd_column}"] - values[f"origin_{sd_column}"]
        )
    atac_lookup = atac.stack(future_stack=True)
    pair_index = pd.MultiIndex.from_frame(values[["group", "cre"]])
    values["atac_peak"] = atac_lookup.reindex(pair_index).fillna(False).to_numpy(bool)
    return values


def plot_heatmap(
    shared: pd.DataFrame,
    row_order: list[str],
    col_order: list[str],
    atac: pd.DataFrame,
    library_counts: pd.Series,
    q_cutoff: float,
    nominal_p_cutoff: float,
    output_pdf: Path,
    output_png: Path,
    dpi: int,
    spec: dict[str, object],
    control_sd: dict[str, pd.Series],
    universe_note: str,
) -> dict:
    centered = {
        run: pivot(
            shared.assign(
                **{
                    f"{run}_centered": (
                        shared[f"{run}_activity_mean"].astype(float)
                        - shared[f"{run}_mean_negative_control_activity_mean"].astype(float)
                    )
                }
            ),
            f"{run}_centered",
            row_order,
            col_order,
        )
        for run in RUNS
    }
    finite_values = np.concatenate(
        [matrix.to_numpy(float)[np.isfinite(matrix.to_numpy(float))] for matrix in centered.values()]
    )
    activity_limit = max(float(np.percentile(np.abs(finite_values), 99)), 1e-6)

    sd_max = max(
        float(np.percentile(np.concatenate([x.to_numpy() for x in control_sd.values()]), 99)),
        1e-6,
    )

    library_log = np.log10(1.0 + library_counts.clip(lower=0))
    t7_sums = {
        run: shared.groupby("cre")[f"{run}_target_t7_total"].sum().reindex(col_order)
        for run in RUNS
    }
    count_tracks = {
        run: np.vstack(
            [
                library_log.reindex(col_order).to_numpy(float),
                np.log10(1.0 + t7_sums[run].clip(lower=0)).to_numpy(float),
            ]
        )
        for run in RUNS
    }
    finite_counts = np.concatenate(
        [track[np.isfinite(track)] for track in count_tracks.values()]
    )
    count_norm = Normalize(
        vmin=float(finite_counts.min()), vmax=float(finite_counts.max())
    )
    activity_norm = Normalize(vmin=-activity_limit, vmax=activity_limit)
    sd_norm = Normalize(vmin=0.0, vmax=sd_max)

    width = max(14.0, 5.2 + 0.105 * len(col_order))
    height = max(10.0, 4.6 + 0.19 * len(row_order) * 2)
    fig = plt.figure(figsize=(width, height), constrained_layout=True)
    grid = fig.add_gridspec(
        4,
        4,
        height_ratios=[max(len(row_order), 1), 2.1, max(len(row_order), 1), 2.1],
        width_ratios=[max(len(col_order), 1), 2.2, 3.0, 3.0],
    )
    activity_cax = fig.add_subplot(grid[:, 2])
    sd_cax = fig.add_subplot(grid[:2, 3])
    count_cax = fig.add_subplot(grid[2:, 3])
    activity_cmap = plt.get_cmap("coolwarm").copy()
    activity_cmap.set_bad("0.9")
    annotation_cmap = plt.get_cmap("viridis").copy()
    annotation_cmap.set_bad("0.9")
    panel_details = {}

    for panel, run in enumerate(RUNS):
        heatmap_row = panel * 2
        count_row = heatmap_row + 1
        ax = fig.add_subplot(grid[heatmap_row, 0])
        sd_ax = fig.add_subplot(grid[heatmap_row, 1], sharey=ax)
        count_ax = fig.add_subplot(grid[count_row, 0], sharex=ax)
        blank_ax = fig.add_subplot(grid[count_row, 1])
        blank_ax.axis("off")

        matrix = centered[run]
        image = ax.imshow(
            np.ma.masked_invalid(matrix.to_numpy(float)),
            aspect="auto",
            cmap=activity_cmap,
            norm=activity_norm,
            interpolation="nearest",
        )
        present = np.isfinite(matrix.to_numpy(float))
        atac_values = atac.to_numpy(bool) & present
        n_atac = add_boxes(ax, atac_values)
        nominal = (
            pivot(shared, f"{run}_p_right", row_order, col_order)
            .le(nominal_p_cutoff)
            .to_numpy(bool)
            & present
        )
        significant = (
            pivot(shared, f"{run}_q_common_universe", row_order, col_order)
            .le(q_cutoff)
            .to_numpy(bool)
            & present
        )
        n_nominal_only, n_significant = add_markers(ax, nominal, significant)

        ax.set_yticks(np.arange(len(row_order)))
        ax.set_yticklabels(row_order, fontsize=6.2)
        ax.tick_params(axis="x", bottom=False, labelbottom=False)
        ax.set_ylabel("Cell subclass (numeric-prefix order)", fontsize=7)
        run_stats = run_summary(shared, run, q_cutoff, spec, control_sd[run])
        ax.set_title(
            f"{RUN_LABELS[run]}: {n_significant} BH-significant; "
            f"{n_nominal_only + n_significant} nominal p≤{nominal_p_cutoff:g}; "
            f"median T7={run_stats['median_target_t7']:.0f}; "
            f"median control SD={run_stats['median_negative_control_sd']:.2f}",
            fontsize=8.5,
        )

        sd_image = sd_ax.imshow(
            control_sd[run].to_numpy(float)[:, None],
            aspect="auto",
            cmap=annotation_cmap,
            norm=sd_norm,
            interpolation="nearest",
        )
        sd_ax.tick_params(
            axis="y", left=False, labelleft=False, right=False, labelright=False
        )
        sd_ax.set_xticks([0])
        sd_ax.set_xticklabels([str(spec["sd_strip_label"])], fontsize=6)
        sd_ax.tick_params(axis="x", bottom=False, labelbottom=True)

        count_image = count_ax.imshow(
            np.ma.masked_invalid(count_tracks[run]),
            aspect="auto",
            cmap=annotation_cmap,
            norm=count_norm,
            interpolation="nearest",
        )
        count_ax.set_yticks([0, 1])
        count_ax.set_yticklabels(
            ["Nanopore library", f"{RUN_LABELS[run]} T7"], fontsize=5.7
        )
        count_ax.set_xticks(np.arange(len(col_order)))
        if run == "new":
            count_ax.set_xticklabels(col_order, rotation=90, fontsize=4.2)
            count_ax.set_xlabel(
                "cCRE ordered by original + new T7 counts across displayed shared pairs "
                f"(n={len(col_order)})",
                fontsize=7,
            )
        else:
            count_ax.tick_params(axis="x", bottom=False, labelbottom=False)

        panel_details[run] = {
            "nominal_p_le_cutoff": int(n_nominal_only + n_significant),
            "significant_common_bh": int(n_significant),
            "atac_positive_tested_pairs": int(n_atac),
        }

    fig.colorbar(
        image,
        cax=activity_cax,
        label="posterior mean [target log_gamma − mean(log_gamma of 7 controls)]",
    )
    fig.colorbar(
        sd_image,
        cax=sd_cax,
        label=str(spec["sd_colorbar_label"]),
    )
    count_colorbar = fig.colorbar(
        count_image,
        cax=count_cax,
        label="log10(1 + count)",
    )
    count_colorbar.ax.tick_params(labelsize=6)
    fig.suptitle(
        "Original versus new low-dose run: shared T7≥50 "
        f"{spec['universe_label']}\n{universe_note}"
        f"n={len(shared):,} subclass–cCRE pairs; "
        f"★ shared-universe BH q≤{q_cutoff:g}; box = ATAC peak",
        fontsize=10,
    )
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_pdf, bbox_inches="tight")
    fig.savefig(output_png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return {
        "activity_color_limit": float(activity_limit),
        "control_sd_color_max": float(sd_max),
        "count_color_min": float(count_norm.vmin),
        "count_color_max": float(count_norm.vmax),
        "panels": panel_details,
    }


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.q_cutoff <= 1.0:
        raise ValueError("--q-cutoff must be between 0 and 1")
    if not 0.0 <= args.nominal_p_cutoff <= 1.0:
        raise ValueError("--nominal-p-cutoff must be between 0 and 1")

    spec = REFERENCE_SPECS[args.control_reference]
    shared_tests = args.shared_tests or spec["tests"]
    stem = args.stem or str(spec["stem"])
    shared = pd.read_csv(shared_tests)
    validate_shared_tests(shared, spec)
    shared["group"] = shared["group"].astype(str)
    shared["cre"] = shared["cre"].astype(str)

    restriction: dict[str, object] | None = None
    if args.restrict_calls is not None:
        statuses = [
            status.strip()
            for status in args.restrict_status.split(",")
            if status.strip()
        ]
        if not statuses:
            raise ValueError("--restrict-status must name at least one status")
        calls = pd.read_csv(args.restrict_calls)
        required = {"group", "cre", args.restrict_status_column}
        missing = sorted(required.difference(calls.columns))
        if missing:
            raise ValueError(f"{args.restrict_calls} is missing columns: {missing}")
        unknown = sorted(set(statuses) - set(calls[args.restrict_status_column]))
        if unknown:
            raise ValueError(
                f"{args.restrict_status_column} never takes the values: {unknown}"
            )
        calls["group"] = calls["group"].astype(str)
        calls["cre"] = calls["cre"].astype(str)
        keep = calls.loc[
            calls[args.restrict_status_column].isin(statuses), ["group", "cre"]
        ]
        before = len(shared)
        shared = shared.merge(keep, on=["group", "cre"], how="inner")
        if shared.empty:
            raise ValueError("No pairs survive the requested restriction")
        restriction = {
            "calls_table": str(Path(args.restrict_calls).resolve()),
            "status_column": args.restrict_status_column,
            "retained_statuses": statuses,
            "pairs_before": int(before),
            "pairs_after": int(len(shared)),
        }
        print(
            f"[restrict] {before:,} -> {len(shared):,} pairs "
            f"({args.restrict_status_column} in {statuses})",
            flush=True,
        )

    observed_rows = pd.Index(shared["group"].drop_duplicates(), dtype=str)
    row_order = subclass_numeric_order(args.origin_h5ad, observed_rows)
    t7_order = (
        shared.assign(
            combined_t7=(
                shared["origin_target_t7_total"].astype(float)
                + shared["new_target_t7_total"].astype(float)
            )
        )
        .groupby("cre")["combined_t7"]
        .sum()
        .sort_values(ascending=False, kind="stable")
    )
    col_order = t7_order.index.astype(str).tolist()
    atac = read_atac_mask(args.atac_peaks, row_order, col_order)
    library_counts = read_library_counts(args.library_counts, col_order)

    args.figures_dir.mkdir(parents=True, exist_ok=True)
    args.tables_dir.mkdir(parents=True, exist_ok=True)
    output_pdf = args.figures_dir / f"{stem}.pdf"
    output_png = args.figures_dir / f"{stem}.png"
    values_path = args.tables_dir / f"{stem}_values.csv.gz"
    manifest_path = args.tables_dir / f"{stem}_manifest.json"

    if spec["pair_sd_column"] is not None:
        control_sd = pair_control_sd_by_group(
            shared, str(spec["pair_sd_column"]), row_order
        )
    else:
        control_sd = control_spread_by_group(
            args.negative_control_activity, row_order
        )
    universe_note = (
        f"Restricted to {args.restrict_status.replace(',', ' or ')} pairs; "
        if restriction is not None
        else ""
    )

    details = plot_heatmap(
        shared,
        row_order,
        col_order,
        atac,
        library_counts,
        args.q_cutoff,
        args.nominal_p_cutoff,
        output_pdf,
        output_png,
        args.dpi,
        spec,
        control_sd,
        universe_note,
    )
    values = diagnostic_values(shared, atac, spec)
    values.to_csv(values_path, index=False)

    summaries = {
        run: run_summary(shared, run, args.q_cutoff, spec, control_sd[run])
        for run in RUNS
    }
    manifest = {
        "inputs": {
            "control_reference": args.control_reference,
            "shared_tests": str(Path(shared_tests).resolve()),
            "origin_h5ad_for_row_order": str(args.origin_h5ad.resolve()),
            "library_counts": str(args.library_counts.resolve()),
            "atac_peaks": str(args.atac_peaks.resolve()),
            "restriction": restriction,
        },
        "outputs": {
            "pdf": str(output_pdf.resolve()),
            "png": str(output_png.resolve()),
            "diagnostic_values": str(values_path.resolve()),
        },
        "universe": {
            "pairs": int(len(shared)),
            "cell_subclasses": int(len(row_order)),
            "cres": int(len(col_order)),
            "definition": (
                "pairs with target T7 >= 50 and combined seven-control T7 >= 50 "
                "in both datasets"
                + (
                    ""
                    if restriction is None
                    else f", restricted to {restriction['retained_statuses']} of "
                    f"{restriction['status_column']}"
                )
            ),
        },
        "plot": {
            "top_panel": "original run",
            "bottom_panel": "new low-dose run",
            "activity": (
                "posterior mean target log_gamma minus posterior mean of the seven "
                "ordinary negative controls"
            ),
            "control_sd_strip": str(spec["sd_colorbar_label"]),
            "nominal_marker": (
                "none; pairs with posterior p_right <= "
                f"{args.nominal_p_cutoff:g} are counted in the panel titles only"
            ),
            "significance_marker": (
                f"star: BH q <= {args.q_cutoff:g}, recomputed within the exact shared universe"
            ),
            "atac_marker": "black cell box for assay value > 0.5",
            "column_order": (
                "descending original + new target T7 summed across displayed shared pairs"
            ),
            "row_order": "subclass numeric prefix from original H5AD",
            "count_tracks": (
                "Nanopore library size and dataset-specific target T7 summed across "
                "displayed shared pairs"
            ),
            **details,
        },
        "diagnosis": {
            "original": summaries["origin"],
            "new": summaries["new"],
            "new_minus_original": {
                "mean_target_t7": float(
                    summaries["new"]["mean_target_t7"]
                    - summaries["origin"]["mean_target_t7"]
                ),
                "median_target_t7": float(
                    summaries["new"]["median_target_t7"]
                    - summaries["origin"]["median_target_t7"]
                ),
                "mean_negative_control_sd": float(
                    summaries["new"]["mean_negative_control_sd"]
                    - summaries["origin"]["mean_negative_control_sd"]
                ),
                "mean_effect_ci90_width": float(
                    summaries["new"]["mean_effect_ci90_width"]
                    - summaries["origin"]["mean_effect_ci90_width"]
                ),
                str(spec["effect_summary_key"]): float(
                    summaries["new"][str(spec["effect_summary_key"])]
                    - summaries["origin"][str(spec["effect_summary_key"])]
                ),
            },
            "bh_first_rank_cutoff": float(args.q_cutoff / len(shared)),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
