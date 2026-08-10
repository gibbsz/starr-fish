#!/usr/bin/env python3
"""Clustered heatmaps of section-specific Bootstrap and Bayesian activities."""

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
from scipy.cluster.hierarchy import leaves_list, linkage

# The shared analysis layer (analysis_utils and the plot_* modules that other
# scripts import) stays in the parent code/ directory.
import sys as _sys
from pathlib import Path as _Path
_CODE_DIR = _Path(__file__).resolve().parent.parent
if str(_CODE_DIR) not in _sys.path:
    _sys.path.insert(0, str(_CODE_DIR))

from analysis_utils import ANALYSIS_DIR, log
from plot_results import bayesian_significance, save_figure


SECTIONS = ("sec1", "sec2")
MODELS = ("Bootstrap", "Bayesian")
CALIBRATIONS = ("calibrated", "uncalibrated")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sections-dir",
        type=Path,
        default=ANALYSIS_DIR / "results" / "sections",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Optional single output directory. If omitted, calibrated plots are "
            "written to results/section_reproducibility and uncalibrated plots "
            "to results/ablation/section_reproducibility_no_calibration_all."
        ),
    )
    parser.add_argument(
        "--calibrations",
        nargs="+",
        choices=CALIBRATIONS,
        default=list(CALIBRATIONS),
    )
    parser.add_argument(
        "--max-cres",
        type=int,
        default=0,
        help="If >0, plot the most variable cCREs across all panels.",
    )
    parser.add_argument(
        "--bootstrap-log-chunk-size",
        type=int,
        default=250,
        help="Bootstrap chunks for computing uncalibrated mean log activity.",
    )
    parser.add_argument(
        "--prior-mask",
        action="store_true",
        help=(
            "Mask activity matrices to Bayesian prior-supported pairs "
            "(prior_dominated == False), section by section."
        ),
    )
    return parser.parse_args()


def default_output_dir(calibration: str) -> Path:
    if calibration == "calibrated":
        return ANALYSIS_DIR / "results" / "section_reproducibility"
    if calibration == "uncalibrated":
        return (
            ANALYSIS_DIR
            / "results"
            / "ablation"
            / "section_reproducibility_no_calibration_all"
        )
    raise ValueError(f"unsupported calibration={calibration}")


def discover_bayes_tag(bayes_dir: Path) -> str:
    return json.loads((bayes_dir / "run_manifest.json").read_text())["tag"]


def load_bootstrap_calibrated(root: Path) -> pd.DataFrame:
    path = root / "log_activity_prior_mask_vs_negative_control.csv"
    if not path.exists():
        path = root / "log_activity_vs_negative_control.csv"
    frame = pd.read_csv(path, index_col=0)
    frame.index = frame.index.astype(str)
    frame.columns = frame.columns.astype(str)
    return frame


def load_bootstrap_uncalibrated(root: Path, chunk_size: int) -> pd.DataFrame:
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
        log_sum += np.where(finite, logged, 0).sum(axis=0)
        log_count += finite.sum(axis=0)
    mean_log = np.divide(
        log_sum,
        log_count,
        out=np.full((n_groups, n_cres), np.nan, dtype=np.float64),
        where=log_count > 0,
    )
    return pd.DataFrame(mean_log, index=groups, columns=cres)


def load_bayesian_calibrated(root: Path) -> pd.DataFrame:
    tag = discover_bayes_tag(root)
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
    activity = significance.pivot(
        index="group", columns="cre", values="bayesian_effect_log"
    )
    activity.index = activity.index.astype(str)
    activity.columns = activity.columns.astype(str)
    return activity


def load_bayesian_uncalibrated(root: Path) -> pd.DataFrame:
    tag = discover_bayes_tag(root)
    with np.load(root / f"{tag}_posterior_samples.npz", allow_pickle=True) as posterior:
        log_gamma = posterior["log_gamma"].astype(np.float32)
        groups = pd.Index(posterior["group_names"].astype(str), dtype=str)
        cres = pd.Index(posterior["cre_names"].astype(str), dtype=str)
    activity = log_gamma.mean(axis=0, dtype=np.float64)
    return pd.DataFrame(activity, index=groups, columns=cres)


def load_prior_supported_mask(bayes_root: Path) -> pd.DataFrame:
    tag = discover_bayes_tag(bayes_root)
    gamma = pd.read_csv(bayes_root / f"{tag}_gamma.csv")
    mask = (
        ~gamma.assign(
            group=gamma["group"].astype(str),
            cre=gamma["cre"].astype(str),
            prior_dominated=gamma["prior_dominated"].astype(bool),
        )
        .pivot(index="group", columns="cre", values="prior_dominated")
        .astype(bool)
    )
    mask.index = mask.index.astype(str)
    mask.columns = mask.columns.astype(str)
    return mask


def load_activity(
    sections_dir: Path, calibration: str, bootstrap_chunk_size: int
) -> dict[tuple[str, str], pd.DataFrame]:
    output: dict[tuple[str, str], pd.DataFrame] = {}
    for section in SECTIONS:
        bootstrap_root = sections_dir / section / "bootstrap"
        bayesian_root = sections_dir / section / "bayesian"
        if calibration == "calibrated":
            output[("Bootstrap", section)] = load_bootstrap_calibrated(bootstrap_root)
            output[("Bayesian", section)] = load_bayesian_calibrated(bayesian_root)
        else:
            output[("Bootstrap", section)] = load_bootstrap_uncalibrated(
                bootstrap_root, bootstrap_chunk_size
            )
            output[("Bayesian", section)] = load_bayesian_uncalibrated(bayesian_root)
    return output


def apply_prior_supported_mask(
    matrices: dict[tuple[str, str], pd.DataFrame], sections_dir: Path
) -> dict[tuple[str, str], pd.DataFrame]:
    masked = {}
    for section in SECTIONS:
        prior_supported = load_prior_supported_mask(
            sections_dir / section / "bayesian"
        )
        for model in MODELS:
            frame = matrices[(model, section)]
            mask = prior_supported.reindex(
                index=frame.index.astype(str),
                columns=frame.columns.astype(str),
                fill_value=False,
            )
            masked[(model, section)] = frame.where(mask)
    return masked


def matrix_zscore(frame: pd.DataFrame) -> pd.DataFrame:
    values = frame.to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return pd.DataFrame(np.zeros(values.shape), index=frame.index, columns=frame.columns)
    center = float(np.nanmedian(values))
    scale = float(np.nanstd(values))
    if not np.isfinite(scale) or scale == 0:
        scale = 1.0
    return pd.DataFrame((values - center) / scale, index=frame.index, columns=frame.columns)


def fill_for_clustering(frame: pd.DataFrame) -> np.ndarray:
    values = frame.to_numpy(dtype=float)
    if np.isfinite(values).any():
        global_fill = float(np.nanmedian(values))
        column_fill = np.full(values.shape[1], global_fill, dtype=float)
        finite_columns = np.isfinite(values).any(axis=0)
        if finite_columns.any():
            column_fill[finite_columns] = np.nanmedian(
                values[:, finite_columns], axis=0
            )
    else:
        column_fill = np.zeros(values.shape[1], dtype=float)
        global_fill = 0.0
    missing = ~np.isfinite(values)
    if missing.any():
        values = values.copy()
        values[missing] = np.take(column_fill, np.where(missing)[1])

    row_mean = values.mean(axis=1, keepdims=True)
    row_sd = values.std(axis=1, keepdims=True)
    row_sd = np.where(row_sd > 0, row_sd, 1.0)
    z = (values - row_mean) / row_sd
    z[~np.isfinite(z)] = 0.0
    return z


def linkage_for(frame: pd.DataFrame, axis: str):
    data = frame if axis == "rows" else frame.T
    if data.shape[0] < 2:
        return None
    return linkage(fill_for_clustering(data), method="average", metric="euclidean")


def ordered_axes(consensus: pd.DataFrame) -> tuple[pd.Index, pd.Index]:
    row_link = linkage_for(consensus, "rows")
    col_link = linkage_for(consensus, "columns")
    row_order = (
        consensus.index[leaves_list(row_link)]
        if row_link is not None
        else consensus.index
    )
    col_order = (
        consensus.columns[leaves_list(col_link)]
        if col_link is not None
        else consensus.columns
    )
    return pd.Index(row_order), pd.Index(col_order)


def robust_limits(frames: list[pd.DataFrame], calibration: str) -> tuple[float, float]:
    finite = np.concatenate(
        [
            frame.to_numpy(dtype=float)[np.isfinite(frame.to_numpy(dtype=float))]
            for frame in frames
            if np.isfinite(frame.to_numpy(dtype=float)).any()
        ]
    )
    if finite.size == 0:
        return (-1.0, 1.0)
    if calibration == "calibrated":
        vmax = float(np.nanpercentile(np.abs(finite), 98))
        vmax = vmax if np.isfinite(vmax) and vmax > 0 else 1.0
        return -vmax, vmax
    vmin, vmax = np.nanpercentile(finite, [1, 99])
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin >= vmax:
        vmin, vmax = float(np.nanmin(finite)), float(np.nanmax(finite))
    if vmin == vmax:
        vmin, vmax = vmin - 1.0, vmax + 1.0
    return float(vmin), float(vmax)


def fill_and_mask_for_plot(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    clean = frame.replace([np.inf, -np.inf], np.nan)
    values = clean.to_numpy(dtype=float)
    if np.isfinite(values).any():
        fill = float(np.nanmedian(values))
    else:
        fill = 0.0
    return clean.fillna(fill), clean.isna()


def select_common_matrices(
    matrices: dict[tuple[str, str], pd.DataFrame], max_cres: int
) -> dict[tuple[str, str], pd.DataFrame]:
    keys = list(matrices)
    common_groups = matrices[keys[0]].index.astype(str)
    common_cres = matrices[keys[0]].columns.astype(str)
    for frame in matrices.values():
        common_groups = common_groups.intersection(frame.index.astype(str))
        common_cres = common_cres.intersection(frame.columns.astype(str))

    aligned = {
        key: frame.reindex(index=common_groups, columns=common_cres)
        for key, frame in matrices.items()
    }
    if max_cres and max_cres > 0 and len(common_cres) > max_cres:
        variability = pd.Series(0.0, index=common_cres)
        for frame in aligned.values():
            variability += matrix_zscore(frame).var(axis=0, skipna=True).fillna(0)
        selected = variability.sort_values(ascending=False).head(max_cres).index
        aligned = {key: frame.loc[:, selected] for key, frame in aligned.items()}
    return aligned


def consensus_matrix(matrices: dict[tuple[str, str], pd.DataFrame]) -> pd.DataFrame:
    keys = list(matrices)
    stack = np.stack(
        [matrix_zscore(matrices[key]).to_numpy(dtype=float) for key in keys],
        axis=0,
    )
    finite = np.isfinite(stack)
    consensus = np.divide(
        np.where(finite, stack, 0.0).sum(axis=0),
        finite.sum(axis=0),
        out=np.zeros(stack.shape[1:], dtype=float),
        where=finite.sum(axis=0) > 0,
    )
    return pd.DataFrame(
        consensus,
        index=matrices[keys[0]].index,
        columns=matrices[keys[0]].columns,
    )


def plot_shared_heatmap(
    matrices: dict[tuple[str, str], pd.DataFrame],
    calibration: str,
    figures: Path,
    variant: str | None = None,
) -> None:
    consensus = consensus_matrix(matrices)
    row_order, col_order = ordered_axes(consensus)
    ordered_frames = [
        matrices[(model, section)].reindex(index=row_order, columns=col_order)
        for model in MODELS
        for section in SECTIONS
    ]
    vmin, vmax = robust_limits(ordered_frames, calibration)
    cmap = "RdBu_r" if calibration == "calibrated" else "viridis"
    center = 0 if calibration == "calibrated" else None
    label = (
        "calibrated log activity"
        if calibration == "calibrated"
        else "uncalibrated log activity"
    )

    fig, axes = plt.subplots(
        len(MODELS),
        len(SECTIONS),
        figsize=(15, 11),
        constrained_layout=True,
        squeeze=False,
    )
    for row, model in enumerate(MODELS):
        for col, section in enumerate(SECTIONS):
            ax = axes[row, col]
            frame, mask = fill_and_mask_for_plot(
                matrices[(model, section)].reindex(index=row_order, columns=col_order)
            )
            sns.heatmap(
                frame,
                ax=ax,
                mask=mask,
                cmap=cmap,
                center=center,
                vmin=vmin,
                vmax=vmax,
                xticklabels=False,
                yticklabels=False,
                cbar=True,
                cbar_kws={"label": label},
                rasterized=True,
            )
            ax.set_facecolor("lightgray")
            ax.set_title(f"{model}, {section}")
            ax.set_xlabel("cCREs, hierarchical order")
            ax.set_ylabel("subclasses, hierarchical order")
    fig.suptitle(
        (
            f"{calibration.capitalize()} activity heatmaps"
            f"{' (' + variant.replace('_', ' ') + ')' if variant else ''}, "
            "shared hierarchical order"
        ),
        fontsize=16,
    )
    stem = (
        f"{calibration}_{variant}_activity_heatmap_shared_cluster"
        if variant
        else f"{calibration}_activity_heatmap_shared_cluster"
    )
    save_figure(fig, figures / stem)


def save_clustergrid(grid, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    for collection in grid.ax_heatmap.collections:
        collection.set_rasterized(True)
    grid.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    grid.savefig(stem.with_suffix(".png"), bbox_inches="tight", dpi=200)
    plt.close(grid.fig)


def plot_individual_clustermaps(
    matrices: dict[tuple[str, str], pd.DataFrame],
    calibration: str,
    figures: Path,
    variant: str | None = None,
) -> None:
    vmin, vmax = robust_limits(list(matrices.values()), calibration)
    cmap = "RdBu_r" if calibration == "calibrated" else "viridis"
    center = 0 if calibration == "calibrated" else None
    label = (
        "calibrated log activity"
        if calibration == "calibrated"
        else "uncalibrated log activity"
    )
    for model in MODELS:
        for section in SECTIONS:
            frame = matrices[(model, section)]
            row_link = linkage_for(frame, "rows")
            col_link = linkage_for(frame, "columns")
            plot_frame, mask = fill_and_mask_for_plot(frame)
            grid = sns.clustermap(
                plot_frame,
                mask=mask,
                row_linkage=row_link,
                col_linkage=col_link,
                cmap=cmap,
                center=center,
                vmin=vmin,
                vmax=vmax,
                xticklabels=False,
                yticklabels=False,
                figsize=(10, 10),
                cbar_kws={"label": label},
            )
            grid.ax_heatmap.set_facecolor("lightgray")
            grid.fig.suptitle(
                (
                    f"{calibration.capitalize()} {model} {section} activity"
                    f"{' (' + variant.replace('_', ' ') + ')' if variant else ''}"
                ),
                y=1.02,
                fontsize=14,
            )
            safe_model = model.lower()
            stem = (
                f"{calibration}_{variant}_{safe_model}_{section}_activity_clustermap"
                if variant
                else f"{calibration}_{safe_model}_{section}_activity_clustermap"
            )
            save_clustergrid(
                grid,
                figures / stem,
            )


def summarize_matrices(
    matrices_by_calibration: dict[str, dict[tuple[str, str], pd.DataFrame]]
) -> pd.DataFrame:
    rows = []
    for calibration, matrices in matrices_by_calibration.items():
        for (model, section), frame in matrices.items():
            values = frame.to_numpy(dtype=float)
            finite = values[np.isfinite(values)]
            rows.append(
                {
                    "calibration": calibration,
                    "model": model,
                    "section": section,
                    "n_subclasses": int(frame.shape[0]),
                    "n_cres": int(frame.shape[1]),
                    "n_values": int(values.size),
                    "n_finite": int(finite.size),
                    "fraction_finite": float(finite.size / values.size)
                    if values.size
                    else np.nan,
                    "median": float(np.nanmedian(finite)) if finite.size else np.nan,
                    "p01": float(np.nanpercentile(finite, 1)) if finite.size else np.nan,
                    "p99": float(np.nanpercentile(finite, 99)) if finite.size else np.nan,
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    sns.set_theme(style="white", context="paper")

    summaries_by_output: dict[Path, list[pd.DataFrame]] = {}
    calibrations_by_output: dict[Path, list[str]] = {}
    for calibration in args.calibrations:
        output_dir = args.output_dir or default_output_dir(calibration)
        figures = output_dir / "figures"
        figures.mkdir(parents=True, exist_ok=True)
        log(f"[activity heatmap] loading {calibration} activity")
        matrices = load_activity(
            args.sections_dir,
            calibration,
            args.bootstrap_log_chunk_size,
        )
        if args.prior_mask:
            matrices = apply_prior_supported_mask(matrices, args.sections_dir)
        matrices = select_common_matrices(matrices, args.max_cres)
        variant = "prior_mask" if args.prior_mask else None
        plot_shared_heatmap(matrices, calibration, figures, variant)
        plot_individual_clustermaps(matrices, calibration, figures, variant)

        summaries_by_output.setdefault(output_dir, []).append(
            summarize_matrices({calibration: matrices})
        )
        calibrations_by_output.setdefault(output_dir, []).append(calibration)

    for output_dir, summaries in summaries_by_output.items():
        tables = output_dir / "tables"
        tables.mkdir(parents=True, exist_ok=True)
        summary_name = (
            "activity_heatmap_prior_mask_matrix_summary.csv"
            if args.prior_mask
            else "activity_heatmap_matrix_summary.csv"
        )
        manifest_name = (
            "activity_heatmap_prior_mask_manifest.json"
            if args.prior_mask
            else "activity_heatmap_manifest.json"
        )
        pd.concat(summaries, ignore_index=True).to_csv(
            tables / summary_name, index=False
        )
        (tables / manifest_name).write_text(
            json.dumps(
                {
                    "sections": list(SECTIONS),
                    "models": list(MODELS),
                    "calibrations": calibrations_by_output[output_dir],
                    "prior_mask": bool(args.prior_mask),
                    "max_cres": args.max_cres,
                    "bootstrap_log_chunk_size": args.bootstrap_log_chunk_size,
                    "clustering": {
                        "method": "average",
                        "metric": "euclidean on row-standardized activity patterns",
                        "shared_order": "computed from the mean z-scored activity matrix across model/section panels",
                    },
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        log(f"[activity heatmap] wrote figures and tables to {output_dir}")


if __name__ == "__main__":
    main()
