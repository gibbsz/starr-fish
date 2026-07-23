#!/usr/bin/env python3
"""Plot sec1/sec2 activity, significance, and epigenomic reproducibility."""

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
from scipy.stats import fisher_exact, pearsonr, spearmanr
from statsmodels.stats.multitest import multipletests

from analysis_utils import ANALYSIS_DIR, REVISION_DATA, STARRFISH_DATA, log, write_json
from plot_results import bayesian_significance, save_figure, wide_to_long


MASK_SPECS = (
    ("2", "t7_total_gt10", "T7 total >10"),
    ("3", "t7_total_gt100", "T7 total >100"),
    ("4", "finite_q", "finite q"),
    ("5", "t7_cells_gt10", "T7 nonzero cells >10"),
    ("6", "t7_cells_gt100", "T7 nonzero cells >100"),
)
METHOD_ROWS = (
    ("Bootstrap", False),
    ("Bayesian", False),
    ("Bootstrap", True),
    ("Bayesian", True),
)


def make_method_name(model: str, use_prior: bool, mask_id: str) -> str:
    return f"{model} {'1+' if use_prior else ''}{mask_id}"


METHODS = tuple(
    make_method_name(model, use_prior, mask_id)
    for model, use_prior in METHOD_ROWS
    for mask_id, _mask_key, _mask_label in MASK_SPECS
)
METHOD_INFO = {
    make_method_name(model, use_prior, mask_id): {
        "model": model,
        "use_prior": use_prior,
        "mask_id": mask_id,
        "mask_key": mask_key,
        "mask_label": mask_label,
        "row_label": f"{model} + {'1+' if use_prior else ''}{mask_id}",
    }
    for model, use_prior in METHOD_ROWS
    for mask_id, mask_key, mask_label in MASK_SPECS
}
METHOD_LABELS = {
    method: f"{info['model']}\n{info['mask_id']}: {info['mask_label']}"
    if not info["use_prior"]
    else f"{info['model']}\n1+{info['mask_id']}: prior + {info['mask_label']}"
    for method, info in METHOD_INFO.items()
}
_PALETTE = plt.get_cmap("tab20").colors
METHOD_COLORS = {
    method: _PALETTE[index % len(_PALETTE)]
    for index, method in enumerate(METHODS)
}
SECTIONS = ("sec1", "sec2")
T7_TOTAL_COUNTS_CSV = REVISION_DATA / "subclass_total_t7_counts.csv"
ASSAYS = {
    "ATAC peak": STARRFISH_DATA / "cre_atac_peaks.csv",
    "Chromatin-a": STARRFISH_DATA / "cre_chromatin_state_a.csv",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sections-dir", type=Path, default=ANALYSIS_DIR / "results" / "sections"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ANALYSIS_DIR
        / "results"
        / "ablation"
        / "section_reproducibility_research_filters",
    )
    parser.add_argument("--q-cutoff", type=float, default=0.05)
    parser.add_argument("--min-points", type=int, default=5)
    parser.add_argument(
        "--activity-calibration",
        choices=["calibrated", "none"],
        default="calibrated",
        help=(
            "Activity values used for section-to-section correlations. "
            "'calibrated' uses self-cCRE and negative-control centered activity; "
            "'none' uses raw log activity/log_gamma."
        ),
    )
    parser.add_argument(
        "--bootstrap-log-chunk-size",
        type=int,
        default=250,
        help="Bootstrap chunks for computing uncalibrated mean log activity.",
    )
    return parser.parse_args()


def read_blacklist(section_dir: Path) -> set[str]:
    path = section_dir / "cre_blacklist.csv"
    if not path.exists():
        return set()
    return set(pd.read_csv(path).iloc[:, 0].astype(str))


def discover_bayes_tag(bayes_dir: Path) -> str:
    return json.loads((bayes_dir / "run_manifest.json").read_text())["tag"]


def standardize_subclass_labels(labels: pd.Index | pd.Series) -> pd.Index:
    return pd.Index(labels).astype(str).str.replace(
        r"^\d+\s+", "", regex=True
    ).str.replace("/", "-", regex=False)


def bayes_axes(bayes_dir: Path) -> tuple[pd.Index, pd.Index]:
    tag = discover_bayes_tag(bayes_dir)
    with np.load(bayes_dir / f"{tag}_posterior_samples.npz", allow_pickle=True) as posterior:
        groups = pd.Index(posterior["group_names"].astype(str), dtype=str)
        cres = pd.Index(posterior["cre_names"].astype(str), dtype=str)
    return groups, cres


def load_t7_nonzero_cell_mask(bayes_dir: Path, min_t7_pos: int) -> pd.DataFrame:
    """Return mask for subclass/cCRE pairs with n_t7_pos > min_t7_pos."""
    tag = discover_bayes_tag(bayes_dir)
    evidence = pd.read_csv(bayes_dir / f"{tag}_evidence_per_pair.csv")
    groups, cres = bayes_axes(bayes_dir)
    evidence["group"] = groups[evidence["group"].to_numpy(dtype=int)]
    evidence["cre"] = cres[evidence["cre"].to_numpy(dtype=int)]
    mask = (
        evidence.assign(t7_supported=evidence["n_t7_pos"].gt(min_t7_pos))
        .pivot(index="group", columns="cre", values="t7_supported")
        .astype(bool)
    )
    mask.index = mask.index.astype(str)
    mask.columns = mask.columns.astype(str)
    return mask


def load_t7_total_mask(
    bayes_dir: Path, section: str, min_total_t7: int
) -> pd.DataFrame:
    """Return row-wise mask for section-specific total T7 reads > threshold."""
    groups, cres = bayes_axes(bayes_dir)
    counts = pd.read_csv(T7_TOTAL_COUNTS_CSV, index_col=0)
    counts.index = standardize_subclass_labels(counts.index)
    column = f"{section}_total_t7"
    if column not in counts.columns:
        raise KeyError(f"{T7_TOTAL_COUNTS_CSV} does not contain {column}")
    supported = counts.reindex(groups)[column].fillna(0).astype(float).gt(
        min_total_t7
    )
    mask = pd.DataFrame(
        np.repeat(supported.to_numpy()[:, None], len(cres), axis=1),
        index=groups,
        columns=cres,
    )
    return mask.astype(bool)


def load_prior_supported_mask(bayes_dir: Path) -> pd.DataFrame:
    """Return mask for subclass/cCRE pairs not marked prior-dominated."""
    tag = discover_bayes_tag(bayes_dir)
    gamma = pd.read_csv(bayes_dir / f"{tag}_gamma.csv")
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


def align_mask(mask: pd.DataFrame, candidate: pd.DataFrame) -> pd.DataFrame:
    mask = mask.copy()
    mask.index = mask.index.astype(str)
    mask.columns = mask.columns.astype(str)
    return mask.reindex(
        index=candidate.index, columns=candidate.columns, fill_value=False
    ).astype(bool)


def apply_candidate_mask(
    frame: pd.DataFrame, candidate: pd.DataFrame
) -> pd.DataFrame:
    frame = frame.copy()
    frame.index = frame.index.astype(str)
    frame.columns = frame.columns.astype(str)
    frame = frame.reindex(index=candidate.index, columns=candidate.columns)
    return frame.where(candidate)


def bh_fdr_frame(pvalues: pd.DataFrame) -> pd.DataFrame:
    values = pvalues.to_numpy(dtype=float)
    qvalues = np.full(values.shape, np.nan, dtype=float)
    valid = np.isfinite(values)
    if valid.any():
        qvalues[valid] = multipletests(values[valid], method="fdr_bh")[1]
    return pd.DataFrame(qvalues, index=pvalues.index, columns=pvalues.columns)


def bootstrap_uncalibrated(
    root: Path, chunk_size: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return raw mean log bootstrap activity and q-values versus raw negative controls."""
    axes = json.loads((root / "bootstrap_axes.json").read_text())
    groups = pd.Index(axes["subclasses"], dtype=str)
    cres = pd.Index(axes["cres"], dtype=str)
    negative_controls = pd.Index(
        pd.read_csv(root / "negative_controls.csv").iloc[:, 0].astype(str)
    )
    negative_mask = cres.isin(negative_controls)
    if not negative_mask.any():
        raise ValueError(f"{root} has no negative-control cCREs in bootstrap axes")

    activity_array = np.load(root / "celltype_activity_array.npy", mmap_mode="r")
    n_boot, n_groups, n_cres = activity_array.shape
    if n_groups != len(groups) or n_cres != len(cres):
        raise ValueError(
            f"{root} axis mismatch: array={activity_array.shape}, "
            f"groups={len(groups)}, cres={len(cres)}"
    )

    log_sum = np.zeros((n_groups, n_cres), dtype=np.float64)
    log_count = np.zeros((n_groups, n_cres), dtype=np.float64)
    negative_sum = np.zeros(n_groups, dtype=np.float64)
    negative_count = np.zeros(n_groups, dtype=np.float64)
    for start in range(0, n_boot, chunk_size):
        chunk = activity_array[start : start + chunk_size]
        with np.errstate(divide="ignore", invalid="ignore"):
            logged = np.log(chunk.astype(np.float64, copy=False))
        finite = np.isfinite(logged)
        log_sum += np.where(finite, logged, 0).sum(axis=0)
        log_count += finite.sum(axis=0)

        negative_logged = logged[:, :, negative_mask]
        negative_finite = np.isfinite(negative_logged)
        negative_sum += np.where(negative_finite, negative_logged, 0).sum(
            axis=(0, 2)
        )
        negative_count += negative_finite.sum(axis=(0, 2))
    mean_log = np.divide(
        log_sum,
        log_count,
        out=np.full((n_groups, n_cres), np.nan, dtype=np.float64),
        where=log_count > 0,
    )
    negative_threshold = np.divide(
        negative_sum,
        negative_count,
        out=np.full(n_groups, np.nan, dtype=np.float64),
        where=negative_count > 0,
    )

    n_less_equal = np.zeros((n_groups, n_cres), dtype=np.float64)
    n_testable = np.zeros((n_groups, n_cres), dtype=np.float64)
    for start in range(0, n_boot, chunk_size):
        chunk = activity_array[start : start + chunk_size]
        with np.errstate(divide="ignore", invalid="ignore"):
            logged = np.log(chunk.astype(np.float64, copy=False))
        testable = np.isfinite(logged) & np.isfinite(negative_threshold)[None, :, None]
        n_less_equal += (
            (logged <= negative_threshold[None, :, None]) & testable
        ).sum(axis=0)
        n_testable += testable.sum(axis=0)
    pvalues = np.divide(
        n_less_equal,
        n_testable,
        out=np.full((n_groups, n_cres), np.nan, dtype=np.float64),
        where=n_testable > 0,
    )
    activity = pd.DataFrame(mean_log, index=groups, columns=cres)
    qvalues = bh_fdr_frame(pd.DataFrame(pvalues, index=groups, columns=cres))
    return activity, qvalues


def bayesian_uncalibrated(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return raw posterior mean log_gamma and q-values versus raw negative controls."""
    tag = discover_bayes_tag(root)
    negative_controls = pd.Index(
        pd.read_csv(root / "negative_controls.csv").iloc[:, 0].astype(str)
    )
    with np.load(root / f"{tag}_posterior_samples.npz", allow_pickle=True) as posterior:
        log_gamma = posterior["log_gamma"].astype(np.float32)
        groups = pd.Index(posterior["group_names"].astype(str), dtype=str)
        cres = pd.Index(posterior["cre_names"].astype(str), dtype=str)
    negative_mask = cres.isin(negative_controls)
    if not negative_mask.any():
        raise ValueError(f"{root} has no negative-control cCREs in Bayesian posterior")
    activity = log_gamma.mean(axis=0, dtype=np.float64)
    negative_threshold = log_gamma[:, :, negative_mask].mean(axis=(0, 2))
    pvalues = (
        log_gamma <= negative_threshold[None, :, None]
    ).mean(axis=0, dtype=np.float64)
    activity = pd.DataFrame(activity, index=groups, columns=cres)
    qvalues = bh_fdr_frame(pd.DataFrame(pvalues, index=groups, columns=cres))
    return activity, qvalues


def bootstrap_base(
    section: str, args: argparse.Namespace
) -> tuple[pd.DataFrame, pd.DataFrame]:
    root = args.sections_dir / section / "bootstrap"
    if args.activity_calibration == "none":
        activity, qvalues = bootstrap_uncalibrated(
            root, args.bootstrap_log_chunk_size
        )
    else:
        # Prefer the recomputed files that remove older count filters; filtering is
        # now handled explicitly by the named masks below.
        activity_path = root / "log_activity_prior_mask_vs_negative_control.csv"
        if not activity_path.exists():
            activity_path = root / "log_activity_vs_negative_control.csv"
        activity = pd.read_csv(activity_path, index_col=0)
        qvalue_path = root / "qvalues_prior_mask_right.csv"
        if not qvalue_path.exists():
            qvalue_path = root / "qvalues_right.csv"
        qvalues = pd.read_csv(qvalue_path, index_col=0)
    activity.index = activity.index.astype(str)
    activity.columns = activity.columns.astype(str)
    qvalues.index = qvalues.index.astype(str)
    qvalues.columns = qvalues.columns.astype(str)
    return activity, qvalues


def bayesian_base(
    section: str, args: argparse.Namespace
) -> tuple[pd.DataFrame, pd.DataFrame]:
    root = args.sections_dir / section / "bayesian"
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


def method_mask(
    *,
    model: str,
    section: str,
    args: argparse.Namespace,
    base_qvalues: pd.DataFrame,
    mask_key: str,
    use_prior: bool,
) -> pd.DataFrame:
    bayes_dir = args.sections_dir / section / "bayesian"
    run_dir = (
        args.sections_dir / section / "bootstrap"
        if model == "Bootstrap"
        else bayes_dir
    )
    candidate = pd.DataFrame(
        True, index=base_qvalues.index.astype(str), columns=base_qvalues.columns.astype(str)
    )
    blacklist = read_blacklist(run_dir)
    candidate.loc[:, candidate.columns.intersection(blacklist)] = False
    if use_prior:
        candidate &= align_mask(load_prior_supported_mask(bayes_dir), candidate)
    if mask_key == "t7_total_gt10":
        candidate &= align_mask(load_t7_total_mask(bayes_dir, section, 10), candidate)
    elif mask_key == "t7_total_gt100":
        candidate &= align_mask(load_t7_total_mask(bayes_dir, section, 100), candidate)
    elif mask_key == "finite_q":
        candidate &= base_qvalues.notna()
    elif mask_key == "t7_cells_gt10":
        candidate &= align_mask(load_t7_nonzero_cell_mask(bayes_dir, 10), candidate)
    elif mask_key == "t7_cells_gt100":
        candidate &= align_mask(load_t7_nonzero_cell_mask(bayes_dir, 100), candidate)
    else:
        raise ValueError(f"unsupported mask_key={mask_key}")
    return candidate.astype(bool)


def load_all(
    args: argparse.Namespace,
) -> tuple[dict[str, dict[str, pd.DataFrame]], dict[str, dict[str, pd.DataFrame]]]:
    activity = {method: {} for method in METHODS}
    qvalues = {method: {} for method in METHODS}
    for section in SECTIONS:
        base = {
            "Bootstrap": bootstrap_base(section, args),
            "Bayesian": bayesian_base(section, args),
        }
        for method in METHODS:
            info = METHOD_INFO[method]
            base_activity, base_qvalues = base[info["model"]]
            candidate = method_mask(
                model=info["model"],
                section=section,
                args=args,
                base_qvalues=base_qvalues,
                mask_key=info["mask_key"],
                use_prior=info["use_prior"],
            )
            activity[method][section] = apply_candidate_mask(
                base_activity, candidate
            )
            qvalues[method][section] = apply_candidate_mask(
                base_qvalues, candidate
            )
    return activity, qvalues


def align_pair(
    sec1: pd.DataFrame, sec2: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    index = sec1.index.astype(str).intersection(sec2.index.astype(str))
    columns = sec1.columns.astype(str).intersection(sec2.columns.astype(str))
    return (
        sec1.reindex(index=index, columns=columns),
        sec2.reindex(index=index, columns=columns),
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
    if (
        len(x) < min_points
        or np.unique(x).size < 2
        or np.unique(y).size < 2
    ):
        return int(len(x)), np.nan, np.nan
    return (
        int(len(x)),
        float(spearmanr(x, y).statistic),
        float(pearsonr(x, y).statistic),
    )


def activity_tables(
    activity: dict[str, dict[str, pd.DataFrame]], args: argparse.Namespace
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    long_rows = []
    correlation_rows = []
    for method in METHODS:
        sec1, sec2 = align_pair(
            activity[method]["sec1"], activity[method]["sec2"]
        )
        pair = wide_to_long(sec1, "activity_sec1").merge(
            wide_to_long(sec2, "activity_sec2"),
            on=["group", "cre"],
            how="inner",
        )
        pair["method"] = method
        pair["measured"] = np.isfinite(pair["activity_sec1"]) & np.isfinite(
            pair["activity_sec2"]
        )
        long_rows.append(pair)

        measured = pair[pair["measured"]]
        n, rho, pearson = correlation_values(
            measured["activity_sec1"],
            measured["activity_sec2"],
            args.min_points,
        )
        correlation_rows.append(
            {
                "method": method,
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
                    frame["activity_sec1"],
                    frame["activity_sec2"],
                    args.min_points,
                )
                correlation_rows.append(
                    {
                        "method": method,
                        "axis": axis,
                        "unit": unit,
                        "n_points": n,
                        "spearman": rho,
                        "pearson": pearson,
                    }
                )

    long = pd.concat(long_rows, ignore_index=True)
    correlations = pd.DataFrame(correlation_rows)
    summary = (
        correlations[correlations["axis"].ne("all_pairs")]
        .groupby(["method", "axis"], sort=False)
        .agg(
            n_units=("spearman", lambda x: int(x.notna().sum())),
            median_spearman=("spearman", "median"),
            mean_spearman=("spearman", "mean"),
            median_pearson=("pearson", "median"),
            median_n_points=("n_points", "median"),
        )
        .reset_index()
    )
    return long, correlations, summary


def plot_activity_scatter(
    long: pd.DataFrame,
    correlations: pd.DataFrame,
    figures: Path,
    activity_label: str,
) -> None:
    fig, axes = plt.subplots(
        len(METHOD_ROWS),
        len(MASK_SPECS),
        figsize=(5.2 * len(MASK_SPECS), 4.4 * len(METHOD_ROWS)),
        constrained_layout=True,
        squeeze=False,
    )
    for row_index, (model, use_prior) in enumerate(METHOD_ROWS):
        row_label = f"{model} + {'prior + ' if use_prior else ''}mask"
        for col_index, (mask_id, _mask_key, mask_label) in enumerate(MASK_SPECS):
            method = make_method_name(model, use_prior, mask_id)
            ax = axes[row_index, col_index]
            frame = long[long["method"].eq(method) & long["measured"]]
            overall = correlations[
                correlations["method"].eq(method)
                & correlations["axis"].eq("all_pairs")
            ].iloc[0]
            if frame.empty:
                ax.text(0.5, 0.5, "No measured pairs", ha="center", va="center")
                ax.set_axis_off()
                continue
            image = ax.hexbin(
                frame["activity_sec1"],
                frame["activity_sec2"],
                gridsize=55,
                mincnt=1,
                bins="log",
                cmap="viridis",
            )
            values = frame[["activity_sec1", "activity_sec2"]].to_numpy(float)
            lo, hi = np.nanpercentile(values, [0.5, 99.5])
            if np.isfinite(lo) and np.isfinite(hi) and lo < hi:
                ax.plot([lo, hi], [lo, hi], color="black", linestyle="--", linewidth=1)
                ax.set_xlim(lo, hi)
                ax.set_ylim(lo, hi)
            ax.set_xlabel(f"Section 1 {activity_label}")
            ax.set_ylabel(f"Section 2 {activity_label}")
            if row_index == 0:
                ax.set_title(f"{mask_id}: {mask_label}", fontsize=10)
            ax.text(
                0.03,
                0.97,
                f"ρ={overall.spearman:.3f}\nr={overall.pearson:.3f}\nn={int(overall.n_points):,}",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=8,
                bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
            )
            if col_index == 0:
                ax.text(
                    -0.35,
                    0.5,
                    row_label,
                    transform=ax.transAxes,
                    ha="right",
                    va="center",
                    rotation=90,
                    fontsize=11,
                    fontweight="bold",
                )
            fig.colorbar(image, ax=ax, label="log10(pair count)")
    save_figure(fig, figures / "section_activity_scatter")


def plot_activity_distributions(
    correlations: pd.DataFrame, figures: Path
) -> None:
    axes_to_plot = (
        ("within_subclass_across_ccres", "Within each subclass, across cCREs"),
        ("across_subclasses_per_ccre", "For each cCRE, across subclasses"),
    )
    model_order = ("Bootstrap", "Bayesian")
    model_colors = {"Bootstrap": "#f58518", "Bayesian": "#4c78a8"}
    mask_groups = [
        (False, mask_id, mask_label) for mask_id, _mask_key, mask_label in MASK_SPECS
    ] + [
        (True, mask_id, mask_label) for mask_id, _mask_key, mask_label in MASK_SPECS
    ]
    group_centers = np.arange(len(mask_groups), dtype=float) * 1.35
    offsets = {"Bootstrap": -0.18, "Bayesian": 0.18}
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(max(16, 1.35 * len(mask_groups)), 10),
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes)
    rng = np.random.default_rng(0)
    for ax, (axis_name, title) in zip(axes, axes_to_plot):
        nonempty_values = []
        nonempty_positions = []
        nonempty_models = []
        values_by_position = []
        for group_position, (use_prior, mask_id, _mask_label) in zip(
            group_centers, mask_groups
        ):
            for model in model_order:
                method = make_method_name(model, use_prior, mask_id)
                position = group_position + offsets[model]
                vals = (
                    correlations.loc[
                        correlations["method"].eq(method)
                        & correlations["axis"].eq(axis_name),
                        "spearman",
                    ]
                    .dropna()
                    .to_numpy(float)
                )
                values_by_position.append((position, model, vals))
                if len(vals):
                    nonempty_values.append(vals)
                    nonempty_positions.append(position)
                    nonempty_models.append(model)
        if nonempty_values:
            parts = ax.violinplot(
                nonempty_values,
                positions=nonempty_positions,
                showmedians=True,
                widths=0.28,
            )
            for body, model in zip(parts["bodies"], nonempty_models):
                body.set_facecolor(model_colors[model])
                body.set_alpha(0.35)
        for position, model, vals in values_by_position:
            ax.scatter(
                position + rng.uniform(-0.045, 0.045, len(vals)),
                vals,
                s=10,
                alpha=0.55,
                color=model_colors[model],
            )
        ax.axhline(0, color="grey", linestyle="--", linewidth=0.8)
        for boundary in (group_centers[:-1] + group_centers[1:]) / 2:
            ax.axvline(boundary, color="0.88", linewidth=0.8, zorder=0)
        ax.set_xticks(group_centers)
        ax.set_xticklabels(
            [
                f"{'prior + ' if use_prior else ''}{mask_label}"
                for use_prior, _mask_id, mask_label in mask_groups
            ],
            rotation=45,
            ha="right",
            fontsize=8,
        )
        ax.set_ylim(-1.05, 1.05)
        ax.set_ylabel("sec1/sec2 Spearman ρ")
        ax.set_title(title)
        ax.legend(
            handles=[
                plt.Line2D([0], [0], marker="o", color="w", label=model,
                           markerfacecolor=color, markersize=8)
                for model, color in model_colors.items()
            ],
            frameon=False,
            loc="upper right",
        )
    save_figure(fig, figures / "section_activity_correlation_distributions")


def significance_metrics(
    qvalues: dict[str, dict[str, pd.DataFrame]], q_cutoff: float
) -> tuple[pd.DataFrame, pd.DataFrame]:
    overall_rows = []
    group_rows = []
    for method in METHODS:
        sec1, sec2 = align_pair(qvalues[method]["sec1"], qvalues[method]["sec2"])
        tested = sec1.notna() & sec2.notna()
        sig1 = sec1.le(q_cutoff) & tested
        sig2 = sec2.le(q_cutoff) & tested
        overlap = sig1 & sig2
        union = sig1 | sig2
        n1 = int(sig1.to_numpy().sum())
        n2 = int(sig2.to_numpy().sum())
        n_overlap = int(overlap.to_numpy().sum())
        n_union = int(union.to_numpy().sum())
        overall_rows.append(
            {
                "method": method,
                "n_tested_common": int(tested.to_numpy().sum()),
                "n_sec1": n1,
                "n_sec2": n2,
                "n_overlap": n_overlap,
                "n_union": n_union,
                "overlap_over_min": n_overlap / min(n1, n2)
                if min(n1, n2)
                else np.nan,
                "jaccard": n_overlap / n_union if n_union else np.nan,
            }
        )
        for group in tested.index:
            a = sig1.loc[group]
            b = sig2.loc[group]
            n1_group = int(a.sum())
            n2_group = int(b.sum())
            overlap_group = int((a & b).sum())
            union_group = int((a | b).sum())
            group_rows.append(
                {
                    "method": method,
                    "group": group,
                    "n_tested_common": int(tested.loc[group].sum()),
                    "n_sec1": n1_group,
                    "n_sec2": n2_group,
                    "n_overlap": overlap_group,
                    "n_union": union_group,
                    "overlap_over_min": overlap_group / min(n1_group, n2_group)
                    if min(n1_group, n2_group)
                    else np.nan,
                    "jaccard": overlap_group / union_group
                    if union_group
                    else np.nan,
                }
            )
    return pd.DataFrame(overall_rows), pd.DataFrame(group_rows)


def plot_significance_reproducibility(
    overall: pd.DataFrame, by_group: pd.DataFrame, figures: Path
) -> None:
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(max(18, 0.9 * len(METHODS)), 10),
        constrained_layout=True,
    )
    x = np.arange(len(METHODS))
    width = 0.34
    ordered = overall.set_index("method").reindex(METHODS)
    axes[0].bar(
        x - width / 2,
        ordered["overlap_over_min"],
        width,
        label="overlap / min(sec1, sec2)",
        color=[METHOD_COLORS[method] for method in METHODS],
    )
    axes[0].bar(
        x + width / 2,
        ordered["jaccard"],
        width,
        label="overlap / union",
        color=[METHOD_COLORS[method] for method in METHODS],
        alpha=0.45,
    )
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(
        [
            f"{METHOD_LABELS[method]}\nsec1={int(ordered.loc[method, 'n_sec1'])}, "
            f"sec2={int(ordered.loc[method, 'n_sec2'])}, "
            f"both={int(ordered.loc[method, 'n_overlap'])}"
            for method in METHODS
        ],
        rotation=45,
        ha="right",
        fontsize=7,
    )
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("Significant-call reproducibility")
    axes[0].set_title("Overall subclass–cCRE calls")
    axes[0].legend(frameon=False, fontsize=8)

    rng = np.random.default_rng(0)
    values = [
        by_group.loc[
            by_group["method"].eq(method), "overlap_over_min"
        ].dropna().to_numpy(float)
        for method in METHODS
    ]
    nonempty_values = []
    nonempty_positions = []
    nonempty_methods = []
    for position, (method, vals) in enumerate(zip(METHODS, values)):
        if len(vals):
            nonempty_values.append(vals)
            nonempty_positions.append(position)
            nonempty_methods.append(method)
    if nonempty_values:
        parts = axes[1].violinplot(
            nonempty_values,
            positions=nonempty_positions,
            showmedians=True,
            showextrema=True,
        )
        for body, method in zip(parts["bodies"], nonempty_methods):
            body.set_facecolor(METHOD_COLORS[method])
            body.set_alpha(0.35)
    for position, (method, vals) in enumerate(zip(METHODS, values)):
        axes[1].scatter(
            position + rng.uniform(-0.06, 0.06, len(vals)),
            vals,
            s=10,
            alpha=0.55,
            color=METHOD_COLORS[method],
        )
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(
        [
            f"{METHOD_LABELS[method]}\nn={len(vals)} subclasses"
            for method, vals in zip(METHODS, values)
        ],
        rotation=45,
        ha="right",
        fontsize=7,
    )
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel("overlap / min(sec1, sec2)")
    axes[1].set_title("Per-subclass reproducibility")
    save_figure(fig, figures / "section_significance_reproducibility")


def reproducible_q_matrix(
    sec1: pd.DataFrame, sec2: pd.DataFrame, q_cutoff: float
) -> pd.DataFrame:
    sec1, sec2 = align_pair(sec1, sec2)
    tested = sec1.notna() & sec2.notna()
    significant = sec1.le(q_cutoff) & sec2.le(q_cutoff) & tested
    output = pd.DataFrame(np.nan, index=sec1.index, columns=sec1.columns)
    output[tested] = 1.0
    output[significant] = 0.0
    return output


def benchmark_one(
    qvalues: pd.DataFrame,
    assay_values: pd.DataFrame,
    method: str,
    section: str,
    assay: str,
    q_cutoff: float,
) -> pd.DataFrame:
    common_columns = qvalues.columns.intersection(assay_values.columns)
    qvalues = qvalues.loc[:, common_columns]
    rows = []
    for group in qvalues.index:
        qrow = qvalues.loc[group]
        tested = qrow.notna().to_numpy()
        significant = (qrow.le(q_cutoff) & qrow.notna()).to_numpy()
        if group in assay_values.index:
            assay_positive = (
                assay_values.loc[group].reindex(common_columns).fillna(0).gt(0.5)
            ).to_numpy() & tested
        else:
            assay_positive = np.zeros(len(common_columns), dtype=bool)
        tp = int((significant & assay_positive).sum())
        n_significant = int(significant.sum())
        n_assay = int(assay_positive.sum())
        n_tested = int(tested.sum())
        fp = n_significant - tp
        fn = n_assay - tp
        tn = n_tested - tp - fp - fn
        odds, pvalue = (
            fisher_exact([[tp, fp], [fn, tn]], alternative="greater")
            if n_tested and min(tp, fp, fn, tn) >= 0
            else (np.nan, np.nan)
        )
        rows.append(
            {
                "method": method,
                "section": section,
                "assay": assay,
                "group": group,
                "TP": tp,
                "significant": n_significant,
                "tested": n_tested,
                "assay_positive": n_assay,
                "precision": tp / n_significant if n_significant else np.nan,
                "recall": tp / n_assay if n_assay else np.nan,
                "fisher_oddsratio": odds,
                "fisher_p": pvalue,
            }
        )
    return pd.DataFrame(rows)


def epigenomic_tables(
    qvalues: dict[str, dict[str, pd.DataFrame]], q_cutoff: float
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for assay, path in ASSAYS.items():
        assay_values = pd.read_csv(path, index_col=0)
        assay_values.index = (
            assay_values.index.astype(str).str.replace("/", "-", regex=False)
        )
        assay_values.columns = assay_values.columns.astype(str)
        for method in METHODS:
            qsets = {
                "sec1": qvalues[method]["sec1"],
                "sec2": qvalues[method]["sec2"],
                "sec1&sec2": reproducible_q_matrix(
                    qvalues[method]["sec1"],
                    qvalues[method]["sec2"],
                    q_cutoff,
                ),
            }
            for section, matrix in qsets.items():
                rows.append(
                    benchmark_one(
                        matrix,
                        assay_values,
                        method,
                        section,
                        assay,
                        q_cutoff,
                    )
                )
    by_group = pd.concat(rows, ignore_index=True)
    by_group["fisher_q"] = np.nan
    valid = by_group["fisher_p"].notna()
    if valid.any():
        by_group.loc[valid, "fisher_q"] = multipletests(
            by_group.loc[valid, "fisher_p"], method="fdr_bh"
        )[1]
    summary = (
        by_group.groupby(["method", "section", "assay"], sort=False)[
            ["TP", "significant", "tested", "assay_positive"]
        ]
        .sum()
        .reset_index()
    )
    summary["precision"] = summary["TP"] / summary["significant"].replace(0, np.nan)
    summary["recall"] = summary["TP"] / summary["assay_positive"].replace(0, np.nan)
    return by_group, summary


def plot_epigenomic_overall(summary: pd.DataFrame, figures: Path) -> None:
    sections = ("sec1", "sec2", "sec1&sec2")
    model_order = ("Bootstrap", "Bayesian")
    model_colors = {"Bootstrap": "#f58518", "Bayesian": "#4c78a8"}
    mask_groups = [
        (False, mask_id, mask_label) for mask_id, _mask_key, mask_label in MASK_SPECS
    ] + [
        (True, mask_id, mask_label) for mask_id, _mask_key, mask_label in MASK_SPECS
    ]
    group_centers = np.arange(len(mask_groups), dtype=float) * 1.25
    width = 0.34
    offsets = {"Bootstrap": -width / 2, "Bayesian": width / 2}
    metrics = (
        ("precision", "Precision: matched / significant", "significant", "TP/sig"),
        ("recall", "Recall: matched / assay-positive", "assay_positive", "TP/assay+"),
    )
    fig, axes = plt.subplots(
        len(ASSAYS) * len(metrics),
        len(sections),
        figsize=(max(16, 1.45 * len(mask_groups) * len(sections)), 4.0 * len(ASSAYS) * len(metrics)),
        constrained_layout=True,
        squeeze=False,
    )
    for row_index, assay in enumerate(ASSAYS):
        for metric_index, (metric, metric_label, denominator_col, count_label) in enumerate(metrics):
            axis_row = row_index * len(metrics) + metric_index
            for column_index, section in enumerate(sections):
                ax = axes[axis_row, column_index]
                frame = (
                    summary[
                        summary["assay"].eq(assay)
                        & summary["section"].eq(section)
                    ]
                    .set_index("method")
                )
                for model in model_order:
                    values = []
                    labels = []
                    for use_prior, mask_id, _mask_label in mask_groups:
                        method = make_method_name(model, use_prior, mask_id)
                        if method in frame.index:
                            values.append(frame.loc[method, metric])
                            labels.append(
                                f"{int(frame.loc[method, 'TP'])}/"
                                f"{int(frame.loc[method, denominator_col])}"
                            )
                        else:
                            values.append(np.nan)
                            labels.append("")
                    bars = ax.bar(
                        group_centers + offsets[model],
                        values,
                        width,
                        color=model_colors[model],
                        label=model,
                    )
                    for bar, label in zip(bars, labels):
                        height = bar.get_height()
                        if not label or not np.isfinite(height):
                            continue
                        ax.text(
                            bar.get_x() + bar.get_width() / 2,
                            min(height + 0.015, 0.98),
                            label,
                            ha="center",
                            va="bottom",
                            rotation=90,
                            fontsize=5,
                        )
                for boundary in (group_centers[:-1] + group_centers[1:]) / 2:
                    ax.axvline(boundary, color="0.85", linewidth=0.8, zorder=0)
                ax.set_xticks(group_centers)
                ax.set_xticklabels(
                    [
                        f"{'prior + ' if use_prior else ''}{mask_label}"
                        for use_prior, mask_id, mask_label in mask_groups
                    ],
                    rotation=45,
                    ha="right",
                    fontsize=7,
                )
                ax.set_ylim(0, 1)
                ax.set_title(f"{assay}, {section}" if metric_index == 0 else section)
                if column_index == 0:
                    ax.set_ylabel(f"{metric_label}\n({count_label})")
                if row_index == 0 and metric_index == 0 and column_index == 0:
                    ax.legend(frameon=False, fontsize=9, title="Model")
    save_figure(fig, figures / "section_epigenomic_overlap")


def plot_epigenomic_heatmaps(by_group: pd.DataFrame, figures: Path) -> None:
    row_order = [
        (method, section)
        for method in METHODS
        for section in ("sec1", "sec2", "sec1&sec2")
    ]
    for assay in ASSAYS:
        frame = by_group[by_group["assay"].eq(assay)]
        groups = sorted(frame["group"].astype(str).unique())
        labels = [f"{method}\n{section}" for method, section in row_order]
        matrix = pd.DataFrame(index=labels, columns=groups, dtype=float)
        for method, section in row_order:
            values = frame[
                frame["method"].eq(method) & frame["section"].eq(section)
            ].set_index("group")["precision"]
            matrix.loc[f"{method}\n{section}", values.index] = values
        finite = matrix.to_numpy(float)
        vmax = float(np.nanmax(finite)) if np.isfinite(finite).any() else 1.0
        fig, ax = plt.subplots(
            figsize=(max(16, len(groups) * 0.22), 5.5),
            constrained_layout=True,
        )
        image = ax.imshow(
            matrix.to_numpy(float),
            aspect="auto",
            cmap="viridis",
            vmin=0,
            vmax=vmax,
        )
        ax.set_yticks(np.arange(len(labels)))
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xticks(np.arange(len(groups)))
        ax.set_xticklabels(groups, rotation=90, fontsize=6)
        ax.set_title(f"{assay}: precision of significant calls by subclass")
        fig.colorbar(image, ax=ax, label="matched / significant calls")
        safe_name = assay.lower().replace(" ", "_").replace("-", "_")
        save_figure(
            fig, figures / f"section_epigenomic_overlap_by_subclass_{safe_name}"
        )


def main() -> None:
    args = parse_args()
    figures = args.output_dir / "figures"
    tables = args.output_dir / "tables"
    figures.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="talk")

    activity, qvalues = load_all(args)
    activity_long, activity_correlations, activity_summary = activity_tables(
        activity, args
    )
    significance_overall, significance_by_group = significance_metrics(
        qvalues, args.q_cutoff
    )
    epigenomic_by_group, epigenomic_summary = epigenomic_tables(
        qvalues, args.q_cutoff
    )

    activity_long.to_csv(tables / "section_activity_long.csv", index=False)
    activity_correlations.to_csv(
        tables / "section_activity_correlations.csv", index=False
    )
    activity_summary.to_csv(tables / "section_activity_summary.csv", index=False)
    significance_overall.to_csv(
        tables / "section_significance_reproducibility.csv", index=False
    )
    significance_by_group.to_csv(
        tables / "section_significance_reproducibility_by_subclass.csv",
        index=False,
    )
    epigenomic_by_group.to_csv(
        tables / "section_epigenomic_overlap_by_subclass.csv", index=False
    )
    epigenomic_summary.to_csv(
        tables / "section_epigenomic_overlap_summary.csv", index=False
    )
    write_json(
        tables / "run_summary.json",
        {
            "q_cutoff": args.q_cutoff,
            "activity_calibration": args.activity_calibration,
            "significance_calibration": args.activity_calibration,
            "methods": list(METHODS),
            "sections": list(SECTIONS),
            "activity_pairs": {
                method: int(
                    activity_long[
                        activity_long["method"].eq(method)
                        & activity_long["measured"]
                    ].shape[0]
                )
                for method in METHODS
            },
        },
    )

    activity_label = (
        "uncalibrated log activity"
        if args.activity_calibration == "none"
        else "calibrated log activity"
    )
    plot_activity_scatter(activity_long, activity_correlations, figures, activity_label)
    plot_activity_distributions(activity_correlations, figures)
    plot_significance_reproducibility(
        significance_overall, significance_by_group, figures
    )
    plot_epigenomic_overall(epigenomic_summary, figures)
    plot_epigenomic_heatmaps(epigenomic_by_group, figures)
    log(
        f"[section plots] wrote {len(list(figures.glob('*.pdf')))} figure pairs "
        f"and tables to {args.output_dir}"
    )


if __name__ == "__main__":
    main()
