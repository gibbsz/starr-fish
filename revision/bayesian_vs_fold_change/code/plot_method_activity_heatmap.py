#!/usr/bin/env python3
"""Plot method-specific full-run activity heatmaps."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
import anndata as ad

from analysis_utils import (
    ANALYSIS_DIR,
    DEFAULT_H5AD,
    LIBSIZE_CSV,
    log,
    read_and_prepare_adata,
    write_json,
)
from plot_section_reproducibility import (
    bayesian_base,
    bootstrap_base,
)

METHOD_SPECS = {
    "bayesian_decoupled": {
        "label": "Bayesian decoupled",
        "kind": "bayesian",
        "root_attr": "new_bayesian_dir",
    },
    "bayesian_joint": {
        "label": "Bayesian joint",
        "kind": "bayesian",
        "root_attr": "old_bayesian_dir",
    },
    "joint_dropout": {
        "label": "Joint+dropout",
        "kind": "bayesian",
        "root_attr": "joint_dropout_bayesian_dir",
    },
    "bootstrap": {
        "label": "Bootstrap",
        "kind": "bootstrap",
        "root_attr": "bootstrap_dir",
    },
}
DEFAULT_METHOD_KEYS = ("bayesian_decoupled", "bayesian_joint", "bootstrap")
FILTER_VARIANTS = ("complete", "t7_gt_threshold", "t7_ge_threshold")
NEGATIVE_CONTROL_COLUMN = "Negative control"
DENSE_AXIS_LIMIT = 60
SPARSE_AXIS_SIZE = 0.22
SPARSE_AXIS_FONTSIZE = 7.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bootstrap-dir", type=Path, default=ANALYSIS_DIR / "results" / "bootstrap"
    )
    parser.add_argument(
        "--old-bayesian-dir",
        type=Path,
        default=ANALYSIS_DIR / "results" / "ablation" / "bayesian_joint",
    )
    parser.add_argument(
        "--new-bayesian-dir",
        type=Path,
        default=ANALYSIS_DIR / "results" / "ablation" / "bayesian_decoupled",
    )
    parser.add_argument(
        "--joint-dropout-bayesian-dir",
        type=Path,
        default=ANALYSIS_DIR / "results" / "ablation" / "bayesian_joint_dropout",
    )
    parser.add_argument(
        "--figures-dir", type=Path, default=ANALYSIS_DIR / "results" / "figures"
    )
    parser.add_argument(
        "--tables-dir", type=Path, default=ANALYSIS_DIR / "results" / "tables"
    )
    parser.add_argument("--h5ad", type=Path, default=DEFAULT_H5AD)
    parser.add_argument(
        "--activity-calibration",
        choices=["calibrated", "none"],
        default="none",
    )
    parser.add_argument(
        "--activity-centering",
        choices=["none", "posterior-alpha"],
        default="none",
        help="Use posterior mean log_gamma or posterior mean(log_gamma - alpha).",
    )
    parser.add_argument(
        "--append-negative-control",
        action="store_true",
        help="Append pooled posterior mean(log_gamma_neg - alpha_neg) as the last column.",
    )
    parser.add_argument(
        "--subtract-negative-control",
        action="store_true",
        help=(
            "Subtract posterior mean(log_gamma_neg - alpha_neg) within each "
            "subclass from posterior-alpha-centered cCRE activity."
        ),
    )
    parser.add_argument(
        "--center-by-mean-negative-controls",
        action="store_true",
        help=(
            "Center posterior mean log_gamma within each subclass by the mean "
            "of the ordinary negative-control cCREs."
        ),
    )
    parser.add_argument(
        "--append-individual-negative-controls",
        action="store_true",
        help="Append each ordinary negative-control cCRE as a separate right-hand column.",
    )
    parser.add_argument("--cmap", default="viridis")
    parser.add_argument("--library-size-csv", type=Path, default=LIBSIZE_CSV)
    parser.add_argument("--library-cmap", default="viridis")
    parser.add_argument(
        "--significance-tests",
        type=Path,
        default=None,
        help="Optional target-test table containing t7_threshold, group, cre, and q_right.",
    )
    parser.add_argument("--significance-q-cutoff", type=float, default=0.05)
    parser.add_argument(
        "--atac-peaks",
        type=Path,
        default=None,
        help=(
            "Optional subclass-cCRE ATAC assay matrix; values > 0.5 are marked "
            "for pairs present in --significance-tests."
        ),
    )
    parser.add_argument(
        "--restrict-to-on-target",
        action="store_true",
        help=(
            "Keep only on-target cCREs (any visible subclass with both an ATAC "
            "peak and a significant test) and then only the on-target subclasses "
            "within those cCREs. Requires --significance-tests and --atac-peaks."
        ),
    )
    parser.add_argument("--bootstrap-log-chunk-size", type=int, default=250)
    parser.add_argument("--t7-threshold", type=float, default=100.0)
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=tuple(METHOD_SPECS),
        default=list(DEFAULT_METHOD_KEYS),
        help="Method keys to plot.",
    )
    parser.add_argument(
        "--filter-variants",
        nargs="+",
        choices=FILTER_VARIANTS,
        default=["complete", "t7_gt_threshold"],
        help="Filtering variants to plot.",
    )
    parser.add_argument("--stem", default="section_activity_heatmap")
    parser.add_argument(
        "--dump-values",
        action="store_true",
        help=(
            "Write the exact plotted matrix of each PDF to "
            "<tables-dir>/<pdf stem>_values.csv (rows and columns in plot "
            "order; blank cells are the grey masked pairs)."
        ),
    )
    return parser.parse_args()


def selected_specs(args: argparse.Namespace) -> list[tuple[str, dict[str, str]]]:
    return [(key, METHOD_SPECS[key]) for key in args.methods]


def method_root(args: argparse.Namespace, spec: dict[str, str]) -> Path:
    return getattr(args, spec["root_attr"])


def posterior_activity(
    root: Path,
    *,
    subtract_alpha: bool,
) -> tuple[pd.DataFrame, pd.Series, Path]:
    manifest = json.loads((root / "run_manifest.json").read_text())
    posterior_path = root / f"{manifest['tag']}_posterior_samples.npz"
    with np.load(posterior_path, allow_pickle=True) as posterior:
        required = {"log_gamma", "log_gamma_neg", "alpha_neg", "group_names", "cre_names"}
        if subtract_alpha:
            required.add("alpha")
        missing = required.difference(posterior.files)
        if missing:
            raise ValueError(f"{posterior_path} is missing sites: {sorted(missing)}")
        log_gamma = posterior["log_gamma"]
        values = log_gamma.mean(axis=0, dtype=np.float64)
        if subtract_alpha:
            alpha = posterior["alpha"]
            expected = (log_gamma.shape[0], log_gamma.shape[2])
            if alpha.shape != expected:
                raise ValueError(
                    f"alpha shape {alpha.shape} does not match expected {expected}"
                )
            values -= alpha.mean(axis=0, dtype=np.float64)[None, :]
        groups = posterior["group_names"].astype(str)
        cres = posterior["cre_names"].astype(str)
        negative = (
            posterior["log_gamma_neg"].mean(axis=0, dtype=np.float64)
            - posterior["alpha_neg"].mean(dtype=np.float64)
        )
    return (
        pd.DataFrame(values, index=groups, columns=cres),
        pd.Series(negative, index=groups, name=NEGATIVE_CONTROL_COLUMN),
        posterior_path,
    )


def posterior_activity_with_ordinary_controls(
    root: Path,
    *,
    center_by_control_mean: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], Path]:
    manifest = json.loads((root / "run_manifest.json").read_text())
    posterior_path = root / f"{manifest['tag']}_posterior_samples.npz"
    with np.load(posterior_path, allow_pickle=True) as posterior:
        required = {"log_gamma", "group_names", "cre_names"}
        missing = required.difference(posterior.files)
        if missing:
            raise ValueError(f"{posterior_path} is missing sites: {sorted(missing)}")
        values = posterior["log_gamma"].mean(axis=0, dtype=np.float64)
        groups = pd.Index(posterior["group_names"].astype(str), dtype=str)
        cres = pd.Index(posterior["cre_names"].astype(str), dtype=str)

    control_path = root / "negative_controls.csv"
    if not control_path.exists():
        raise FileNotFoundError(f"missing ordinary negative controls: {control_path}")
    controls = pd.read_csv(control_path).iloc[:, 0].astype(str).tolist()
    controls = list(dict.fromkeys(controls))
    missing_controls = [cre for cre in controls if cre not in cres]
    if missing_controls:
        raise ValueError(
            f"{posterior_path} is missing negative-control cCREs: {missing_controls}"
        )
    if not controls:
        raise ValueError(f"no negative-control cCREs listed in {control_path}")

    activity = pd.DataFrame(values, index=groups, columns=cres)
    if center_by_control_mean:
        control_mean = activity.loc[:, controls].mean(axis=1)
        activity = activity.sub(control_mean, axis=0)
    return activity, activity.loc[:, controls].copy(), controls, posterior_path


def load_activity(
    args: argparse.Namespace,
) -> tuple[
    dict[str, pd.DataFrame],
    dict[str, pd.Series],
    dict[str, pd.DataFrame],
    dict[str, list[str]],
    dict[str, str],
]:
    activity = {}
    negative_activity = {}
    individual_negative_activity = {}
    individual_negative_controls = {}
    posterior_sources = {}
    for _, spec in selected_specs(args):
        label = spec["label"]
        root = method_root(args, spec)
        if spec["kind"] == "bootstrap":
            if (
                args.activity_centering == "posterior-alpha"
                or args.append_negative_control
                or args.center_by_mean_negative_controls
                or args.append_individual_negative_controls
            ):
                raise ValueError(
                    "posterior centering and negative-control columns require a "
                    "Bayesian method"
                )
            activity[label] = bootstrap_base(root, args)[0]
            continue
        if (
            args.center_by_mean_negative_controls
            or args.append_individual_negative_controls
        ):
            matrix, controls, control_names, posterior_path = (
                posterior_activity_with_ordinary_controls(
                    root,
                    center_by_control_mean=args.center_by_mean_negative_controls,
                )
            )
            activity[label] = matrix
            individual_negative_activity[label] = controls
            individual_negative_controls[label] = control_names
            posterior_sources[label] = str(posterior_path)
        elif args.activity_centering == "posterior-alpha" or args.append_negative_control:
            matrix, negative, posterior_path = posterior_activity(
                root,
                subtract_alpha=args.activity_centering == "posterior-alpha",
            )
            if args.subtract_negative_control:
                matrix = matrix.sub(negative, axis=0)
                negative = negative - negative
            activity[label] = matrix
            negative_activity[label] = negative
            posterior_sources[label] = str(posterior_path)
        else:
            activity[label] = bayesian_base(root, args)[0]
    return (
        activity,
        negative_activity,
        individual_negative_activity,
        individual_negative_controls,
        posterior_sources,
    )


def read_cre_blacklist(root: Path) -> set[str]:
    path = root / "cre_blacklist.csv"
    if not path.exists():
        return set()
    return set(pd.read_csv(path).iloc[:, 0].astype(str))


def blacklisted_cres(args: argparse.Namespace) -> tuple[set[str], dict[str, list[str]]]:
    sources = {
        spec["label"]: sorted(read_cre_blacklist(method_root(args, spec)))
        for _, spec in selected_specs(args)
    }
    blacklist = set().union(*(set(cres) for cres in sources.values()))
    return blacklist, sources


def combined_axes(matrices: dict[str, pd.DataFrame]) -> tuple[pd.Index, pd.Index]:
    combined_index = None
    combined_columns = None
    for matrix in matrices.values():
        index = pd.Index(matrix.index.astype(str))
        columns = pd.Index(matrix.columns.astype(str))
        combined_index = index if combined_index is None else combined_index.union(index)
        combined_columns = (
            columns if combined_columns is None else combined_columns.union(columns)
        )
    if combined_index is None or combined_columns is None:
        raise ValueError("no activity matrices were loaded")
    return combined_index, combined_columns


def cleaned_subclass(label: str) -> str:
    return pd.Series([label]).str.replace(r"^\d+\s+", "", regex=True).str.replace(
        "/", "-", regex=False
    ).iloc[0]


def subclass_numeric_order(h5ad: Path, observed: pd.Index) -> list[str]:
    data = ad.read_h5ad(h5ad, backed="r")
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
            "prefix": pd.to_numeric(source.str.extract(r"^(\d+)\s+", expand=False)),
        }
    )
    order = (
        order.dropna(subset=["prefix"])
        .assign(prefix=lambda x: x["prefix"].astype(int))
        .sort_values(["prefix", "group"])
        .drop_duplicates("group", keep="first")
        .set_index("group")["prefix"]
    )
    return sorted(
        observed.astype(str),
        key=lambda group: (order.get(group, 10**9), str(group)),
    )


def t7_pair_totals(
    h5ad: Path,
    candidate_index: pd.Index,
    candidate_columns: pd.Index,
) -> pd.DataFrame:
    adata = read_and_prepare_adata(h5ad)
    t7 = adata.obsm["T7CRE"].copy()
    t7.index = adata.obs["subclass"].astype(str).to_numpy()
    totals = t7.groupby(level=0, sort=False).sum()
    totals.index = totals.index.astype(str)
    totals.columns = totals.columns.astype(str)
    return totals.reindex(
        index=candidate_index.astype(str),
        columns=candidate_columns.astype(str),
        fill_value=0.0,
    ).fillna(0.0)


def t7_cre_order(pair_t7: pd.DataFrame) -> list[str]:
    total_by_cre = pair_t7.sum(axis=0)
    return (
        total_by_cre.sort_values(ascending=False, na_position="last")
        .index.astype(str)
        .tolist()
    )


def read_library_counts(path: Path) -> pd.Series:
    table = pd.read_csv(path, index_col=0)
    if "counts" not in table.columns:
        raise ValueError(f"{path} must contain a 'counts' column")
    counts = pd.to_numeric(table["counts"], errors="coerce").clip(lower=0)
    counts.index = counts.index.astype(str)
    return counts.rename("nanopore_library_size")


def read_significance_mask(
    path: Path,
    t7_threshold: float,
    rows: pd.Index,
    columns: pd.Index,
    q_cutoff: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    tests = pd.read_csv(
        path,
        usecols=["t7_threshold", "group", "cre", "q_right"],
    )
    tests = tests.loc[
        np.isclose(tests["t7_threshold"].to_numpy(float), t7_threshold)
    ].copy()
    tests["group"] = tests["group"].astype(str)
    tests["cre"] = tests["cre"].astype(str)
    q_values = tests.pivot_table(
        index="group",
        columns="cre",
        values="q_right",
        aggfunc="min",
    )
    aligned = q_values.reindex(index=rows, columns=columns)
    tested = aligned.notna()
    significant = aligned.le(q_cutoff) & tested
    return significant.astype(bool), tested.astype(bool)


def read_atac_peak_mask(
    path: Path,
    rows: pd.Index,
    columns: pd.Index,
) -> pd.DataFrame:
    atac = pd.read_csv(path, index_col=0)
    atac.index = atac.index.astype(str).str.replace("/", "-", regex=False)
    atac.columns = atac.columns.astype(str)
    return atac.reindex(index=rows, columns=columns).gt(0.5).fillna(False).astype(bool)


def finite_any_method(matrices: dict[str, pd.DataFrame]) -> pd.DataFrame:
    present = None
    for matrix in matrices.values():
        finite = pd.DataFrame(
            np.isfinite(matrix.to_numpy(float)),
            index=matrix.index,
            columns=matrix.columns,
        )
        present = finite if present is None else present | finite
    if present is None:
        raise ValueError("no matrices to summarize")
    return present


def trim_empty_axes(
    matrices: dict[str, pd.DataFrame],
) -> tuple[dict[str, pd.DataFrame], list[str], list[str], pd.DataFrame]:
    present = finite_any_method(matrices)
    keep_rows = present.any(axis=1)
    keep_cols = present.any(axis=0)
    row_order = present.index[keep_rows].astype(str).tolist()
    col_order = present.columns[keep_cols].astype(str).tolist()
    trimmed = {
        label: matrix.reindex(index=row_order, columns=col_order)
        for label, matrix in matrices.items()
    }
    return trimmed, row_order, col_order, present.loc[row_order, col_order]


def axis_scale(
    n_ticks: int,
    dense_size: float,
    dense_fontsize: float,
) -> tuple[float, float]:
    """Inches per cell and tick fontsize for one heatmap axis.

    Dense axes keep the historical compact sizing; sparse axes (on-target
    subsets) get larger cells and readable tick labels.
    """
    if n_ticks > DENSE_AXIS_LIMIT:
        return dense_size, dense_fontsize
    return SPARSE_AXIS_SIZE, SPARSE_AXIS_FONTSIZE


def on_target_axes(
    significance: pd.DataFrame,
    atac_peaks: pd.DataFrame,
    present: pd.DataFrame,
    rows: list[str],
    cols: list[str],
) -> tuple[list[str], list[str], pd.DataFrame]:
    """Select cCREs then subclasses that are ATAC-positive and significant.

    ``present`` marks visible (finite, filter-passing) pairs and is indexed by
    ``rows`` x ``cols``. Columns are kept when any visible pair in them is both
    ATAC-positive and significant; rows are then kept when any of their visible
    pairs in the surviving columns qualify.
    """
    on_target = (
        significance.reindex(index=rows, columns=cols, fill_value=False)
        & atac_peaks.reindex(index=rows, columns=cols, fill_value=False)
        & present.reindex(index=rows, columns=cols, fill_value=False)
    )
    keep_cols = [cre for cre in cols if bool(on_target[cre].any())]
    if not keep_cols:
        raise ValueError("no on-target cCREs survive the ATAC-and-significance filter")
    on_target = on_target.loc[:, keep_cols]
    keep_rows = [group for group in rows if bool(on_target.loc[group].any())]
    if not keep_rows:
        raise ValueError("no on-target subclasses survive the ATAC-and-significance filter")
    return keep_rows, keep_cols, on_target.loc[keep_rows, keep_cols]


def safe_stem(label: str) -> str:
    return (
        label.lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace("/", "_")
        .replace("+", "_plus_")
    )


def threshold_token(value: float) -> str:
    return f"{value:g}".replace("-", "minus").replace(".", "p")


def filter_variant_specs(
    args: argparse.Namespace,
    pair_t7: pd.DataFrame,
    common_rows: pd.Index,
    common_cols: pd.Index,
) -> dict[str, dict[str, object]]:
    variants = {}
    for variant in args.filter_variants:
        if variant == "complete":
            variants["complete"] = {
                "filter_label": "complete, no filtering",
                "pair_filter": "none",
                "mask": pd.DataFrame(True, index=common_rows, columns=common_cols),
            }
        elif variant == "t7_gt_threshold":
            token = threshold_token(args.t7_threshold)
            variants[f"t7_gt{token}"] = {
                "filter_label": f"subclass-cCRE total T7 > {args.t7_threshold:g}",
                "pair_filter": "gt",
                "mask": pair_t7.gt(args.t7_threshold),
            }
        elif variant == "t7_ge_threshold":
            token = threshold_token(args.t7_threshold)
            variants[f"t7_ge{token}"] = {
                "filter_label": f"subclass-cCRE total T7 >= {args.t7_threshold:g}",
                "pair_filter": "ge",
                "mask": pair_t7.ge(args.t7_threshold),
            }
    return variants


def plot_one_pdf(
    matrix: pd.DataFrame,
    label: str,
    activity_label: str,
    row_order: list[str],
    col_order: list[str],
    vmin: float,
    vmax: float,
    path: Path,
    filter_label: str,
    cmap_name: str,
    log10_library_size: pd.Series,
    log10_t7_total: pd.Series,
    library_cmap_name: str,
    individual_negative_controls: list[str] | None = None,
    significance: pd.DataFrame | None = None,
    significance_q_cutoff: float = 0.05,
    atac_peaks: pd.DataFrame | None = None,
    values_csv: Path | None = None,
) -> None:
    ordered = matrix.reindex(index=row_order, columns=col_order)
    if values_csv is not None:
        values_csv.parent.mkdir(parents=True, exist_ok=True)
        ordered.rename_axis(index="subclass", columns="cre").to_csv(values_csv)
    col_size, col_fontsize = axis_scale(len(col_order), 0.06, 3.0)
    row_size, row_fontsize = axis_scale(len(row_order), 0.09, 4.0)
    fig = plt.figure(
        figsize=(
            4.0 + col_size * len(col_order),
            4.0 + row_size * len(row_order),
        ),
        constrained_layout=True,
    )
    grid = fig.add_gridspec(
        3,
        2,
        height_ratios=[max(len(row_order), 1), 2, 2],
        width_ratios=[1, 0.022],
    )
    ax = fig.add_subplot(grid[0, 0])
    activity_colorbar_ax = fig.add_subplot(grid[0, 1])
    count_ax = fig.add_subplot(grid[1, 0], sharex=ax)
    count_colorbar_ax = fig.add_subplot(grid[2, 0])
    cmap = plt.get_cmap(cmap_name).copy()
    cmap.set_bad("0.9")
    image = ax.imshow(
        np.ma.masked_invalid(ordered.to_numpy(float)),
        aspect="auto",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
    )
    n_atac_peaks = 0
    if atac_peaks is not None:
        atac_marked = atac_peaks.reindex(
            index=row_order,
            columns=col_order,
            fill_value=False,
        ).astype(bool)
        atac_values = atac_marked.to_numpy(bool) & np.isfinite(
            ordered.to_numpy(float)
        )
        atac_rows, atac_cols = np.nonzero(atac_values)
        n_atac_peaks = int(len(atac_rows))
        boxes = [
            Rectangle((col - 0.5, row - 0.5), 1.0, 1.0)
            for row, col in zip(atac_rows, atac_cols)
        ]
        ax.add_collection(
            PatchCollection(
                boxes,
                facecolor="none",
                edgecolor="black",
                linewidth=0.65,
                zorder=4,
            )
        )
    n_significant = 0
    if significance is not None:
        marked = significance.reindex(
            index=row_order,
            columns=col_order,
            fill_value=False,
        ).astype(bool)
        marked_values = marked.to_numpy(bool) & np.isfinite(ordered.to_numpy(float))
        marked_rows, marked_cols = np.nonzero(marked_values)
        n_significant = int(len(marked_rows))
        ax.scatter(
            marked_cols,
            marked_rows,
            marker="*",
            s=8 if len(col_order) > DENSE_AXIS_LIMIT else 45,
            facecolors="white",
            edgecolors="black",
            linewidths=0.2,
            zorder=5,
        )
    ax.set_yticks(np.arange(len(row_order)))
    ax.set_yticklabels(row_order, fontsize=row_fontsize)
    ax.tick_params(axis="x", bottom=False, labelbottom=False)
    control_columns = list(individual_negative_controls or [])
    has_pooled_negative_control = (
        bool(col_order) and col_order[-1] == NEGATIVE_CONTROL_COLUMN
    )
    n_control_columns = len(control_columns) + int(has_pooled_negative_control)
    n_target_cres = len(col_order) - n_control_columns
    if n_control_columns:
        ax.axvline(n_target_cres - 0.5, color="black", linewidth=1.0)
    count_values = np.vstack(
        [
            log10_library_size.reindex(col_order).to_numpy(float),
            log10_t7_total.reindex(col_order).to_numpy(float),
        ]
    )
    count_cmap = plt.get_cmap(library_cmap_name).copy()
    count_cmap.set_bad("0.9")
    count_image = count_ax.imshow(
        np.ma.masked_invalid(count_values),
        aspect="auto",
        cmap=count_cmap,
        interpolation="nearest",
    )
    if n_control_columns:
        count_ax.axvline(n_target_cres - 0.5, color="black", linewidth=1.0)
    count_ax.set_yticks([0, 1])
    count_ax.set_yticklabels(
        ["Nanopore library size", "T7 total counts"], fontsize=6
    )
    count_ax.set_xticks(np.arange(len(col_order)))
    count_ax.set_xticklabels(col_order, rotation=90, fontsize=col_fontsize)
    count_ax.set_xlabel(
        f"cCRE ordered by total T7 counts, high to low (n={n_target_cres})"
        + (
            f"; {len(control_columns)} individual negative controls at right"
            if control_columns
            else "; pooled negative control at right"
            if has_pooled_negative_control
            else ""
        )
    )
    ax.set_ylabel("Cell subclass ordered by numeric prefix")
    title = f"{label} - {activity_label} ({filter_label})"
    if significance is not None:
        title += (
            f"\n* target versus mean controls, BH q <= {significance_q_cutoff:g} "
            f"(n={n_significant})"
        )
    if atac_peaks is not None:
        title += f"; box = ATAC peak among tested pairs (n={n_atac_peaks})"
    ax.set_title(title)
    fig.colorbar(image, cax=activity_colorbar_ax, label=activity_label)
    count_colorbar = fig.colorbar(
        count_image,
        cax=count_colorbar_ax,
        orientation="horizontal",
        label="log10(1 + count)",
    )
    count_colorbar.ax.tick_params(labelsize=6)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_heatmaps(args: argparse.Namespace) -> dict:
    if (
        args.subtract_negative_control
        and args.activity_centering != "posterior-alpha"
    ):
        raise ValueError(
            "--subtract-negative-control requires "
            "--activity-centering posterior-alpha"
        )
    if args.append_negative_control and args.append_individual_negative_controls:
        raise ValueError(
            "choose either --append-negative-control or "
            "--append-individual-negative-controls"
        )
    if args.center_by_mean_negative_controls and args.activity_centering != "none":
        raise ValueError(
            "--center-by-mean-negative-controls requires --activity-centering none"
        )
    if not 0.0 <= args.significance_q_cutoff <= 1.0:
        raise ValueError("--significance-q-cutoff must be between 0 and 1")
    if args.atac_peaks is not None and args.significance_tests is None:
        raise ValueError("--atac-peaks requires --significance-tests")
    if args.restrict_to_on_target and args.atac_peaks is None:
        raise ValueError(
            "--restrict-to-on-target requires --atac-peaks and --significance-tests"
        )
    plt.rcParams.update({
        "axes.facecolor": "white",
        "figure.facecolor": "white",
        "font.size": 8,
    })
    (
        raw,
        negative_activity,
        individual_negative_activity,
        individual_negative_controls,
        posterior_sources,
    ) = load_activity(args)
    common_rows, common_cols = combined_axes(raw)
    blacklist, blacklist_sources = blacklisted_cres(args)
    negative_controls = set()
    if args.append_negative_control:
        for _, spec in selected_specs(args):
            path = method_root(args, spec) / "negative_controls.csv"
            if path.exists():
                negative_controls.update(pd.read_csv(path).iloc[:, 0].astype(str))
    individual_control_order = list(
        dict.fromkeys(
            cre
            for _, spec in selected_specs(args)
            for cre in individual_negative_controls.get(spec["label"], [])
        )
    )
    negative_controls.update(individual_control_order)
    common_cols = pd.Index(
        [
            cre
            for cre in common_cols.astype(str)
            if cre not in blacklist and cre not in negative_controls
        ],
        dtype=str,
    )
    significance = None
    tested_pairs = None
    if args.significance_tests is not None:
        significance, tested_pairs = read_significance_mask(
            args.significance_tests,
            args.t7_threshold,
            common_rows,
            common_cols,
            args.significance_q_cutoff,
        )
    atac_peaks = (
        read_atac_peak_mask(args.atac_peaks, common_rows, common_cols)
        & tested_pairs
        if args.atac_peaks is not None
        else None
    )
    matrices = {
        label: matrix.reindex(index=common_rows, columns=common_cols)
        for label, matrix in raw.items()
    }
    t7_columns = common_cols.union(pd.Index(sorted(negative_controls), dtype=str))
    all_pair_t7 = t7_pair_totals(args.h5ad, common_rows, t7_columns)
    pair_t7 = all_pair_t7.reindex(columns=common_cols, fill_value=0.0)
    row_order = subclass_numeric_order(args.h5ad, common_rows)
    col_order = t7_cre_order(pair_t7)
    library_counts = read_library_counts(args.library_size_csv)
    log10_library_size = np.log10(1.0 + library_counts).rename(
        "log10_nanopore_library_size"
    )
    log10_t7_total = np.log10(1.0 + pair_t7.sum(axis=0).clip(lower=0)).rename(
        "log10_t7_total"
    )
    if individual_control_order:
        control_t7_total = all_pair_t7.reindex(
            columns=individual_control_order, fill_value=0.0
        ).sum(axis=0)
        log10_t7_total = pd.concat(
            [
                log10_t7_total,
                np.log10(1.0 + control_t7_total.clip(lower=0)).rename(
                    "log10_t7_total"
                ),
            ]
        )
    pooled_negative_library_count = None
    pooled_negative_t7_count = None
    if args.append_negative_control:
        if not negative_controls:
            raise ValueError(
                "cannot pool count annotations without negative-control cCREs"
            )
        pooled_negative_library_count = float(
            library_counts.reindex(sorted(negative_controls)).fillna(0.0).sum()
        )
        pooled_negative_t7_count = float(
            all_pair_t7.reindex(
                columns=sorted(negative_controls), fill_value=0.0
            ).to_numpy(float).sum()
        )
        log10_library_size.loc[NEGATIVE_CONTROL_COLUMN] = np.log10(
            1.0 + pooled_negative_library_count
        )
        log10_t7_total.loc[NEGATIVE_CONTROL_COLUMN] = np.log10(
            1.0 + pooled_negative_t7_count
        )
    if args.center_by_mean_negative_controls:
        activity_label = (
            "posterior mean [log_gamma - mean(log_gamma of 7 negative controls)]"
        )
    elif args.subtract_negative_control:
        activity_label = (
            "posterior mean [(log_gamma - alpha) - "
            "(log_gamma_neg - alpha_neg)]"
        )
    elif args.activity_centering == "posterior-alpha":
        activity_label = "posterior mean log_gamma - alpha"
    elif args.append_negative_control:
        activity_label = "posterior mean log_gamma"
    else:
        activity_label = (
            "uncalibrated log activity"
            if args.activity_calibration == "none"
            else "calibrated log activity"
        )
    variants = filter_variant_specs(args, pair_t7, common_rows, common_cols)
    individual_control_pair_t7 = all_pair_t7.reindex(
        index=common_rows,
        columns=individual_control_order,
        fill_value=0.0,
    )
    outputs = {}
    value_tables = {}
    variant_shapes = {}
    for variant, spec in variants.items():
        masked_matrices = {
            label: matrix.where(spec["mask"]).reindex(index=row_order, columns=col_order)
            for label, matrix in matrices.items()
        }
        variant_matrices, variant_rows, variant_cols, present = trim_empty_axes(
            masked_matrices
        )
        on_target_pairs = None
        if args.restrict_to_on_target:
            variant_rows, variant_cols, on_target_pairs = on_target_axes(
                significance,
                atac_peaks,
                present,
                variant_rows,
                variant_cols,
            )
            variant_matrices = {
                label: matrix.reindex(index=variant_rows, columns=variant_cols)
                for label, matrix in variant_matrices.items()
            }
            present = present.reindex(index=variant_rows, columns=variant_cols)
        display_cols = list(variant_cols)
        if args.append_negative_control:
            display_cols.append(NEGATIVE_CONTROL_COLUMN)
            variant_matrices = {
                label: matrix.assign(
                    **{
                        NEGATIVE_CONTROL_COLUMN: negative_activity[label].reindex(
                            variant_rows
                        )
                    }
                )
                for label, matrix in variant_matrices.items()
            }
        elif args.append_individual_negative_controls:
            display_cols.extend(individual_control_order)
            control_mask = individual_control_pair_t7.reindex(index=variant_rows)
            if spec["pair_filter"] == "gt":
                control_mask = control_mask.gt(args.t7_threshold)
            elif spec["pair_filter"] == "ge":
                control_mask = control_mask.ge(args.t7_threshold)
            else:
                control_mask = pd.DataFrame(
                    True,
                    index=control_mask.index,
                    columns=control_mask.columns,
                )
            variant_matrices = {
                label: pd.concat(
                    [
                        matrix,
                        individual_negative_activity[label]
                        .reindex(
                            index=variant_rows,
                            columns=individual_control_order,
                        )
                        .where(control_mask),
                    ],
                    axis=1,
                )
                for label, matrix in variant_matrices.items()
            }
        all_values = np.concatenate(
            [matrix.to_numpy(float).ravel() for matrix in variant_matrices.values()]
        )
        finite = all_values[np.isfinite(all_values)]
        if finite.size:
            vmin, vmax = tuple(np.percentile(finite, [1, 99]))
            vcenter = float(np.median(finite))
            if args.cmap.lower() == "coolwarm":
                span = max(vcenter - float(vmin), float(vmax) - vcenter, 1e-6)
                vmin, vmax = vcenter - span, vcenter + span
        else:
            vcenter = 0.0
            vmin, vmax = (
                (-1.0, 1.0)
                if args.cmap.lower() == "coolwarm"
                else (0.0, 1.0)
            )
        outputs[variant] = {}
        value_tables[variant] = {}
        for label, matrix in variant_matrices.items():
            out = args.figures_dir / f"{args.stem}_{variant}_{safe_stem(label)}.pdf"
            values_csv = (
                args.tables_dir / f"{out.stem}_values.csv"
                if args.dump_values
                else None
            )
            plot_one_pdf(
                matrix,
                label,
                activity_label,
                variant_rows,
                display_cols,
                vmin,
                vmax,
                out,
                spec["filter_label"],
                args.cmap,
                log10_library_size,
                log10_t7_total,
                args.library_cmap,
                individual_control_order
                if args.append_individual_negative_controls
                else None,
                significance,
                args.significance_q_cutoff,
                atac_peaks,
                values_csv=values_csv,
            )
            outputs[variant][label] = str(out)
            if values_csv is not None:
                value_tables[variant][label] = str(values_csv)
        variant_shapes[variant] = {
            "rows": int(len(variant_rows)),
            "columns": int(len(variant_cols)),
            "display_columns": int(len(display_cols)),
            "finite_pairs_any_method": int(present.to_numpy(bool).sum()),
            "passing_filter_pairs": int(spec["mask"].to_numpy(bool).sum()),
            "passing_t7_pairs": int(spec["mask"].to_numpy(bool).sum()),
            "passing_individual_negative_control_pairs": int(
                control_mask.to_numpy(bool).sum()
            )
            if args.append_individual_negative_controls
            else None,
            "significant_visible_target_pairs": int(
                (
                    significance.reindex(
                        index=variant_rows,
                        columns=variant_cols,
                        fill_value=False,
                    ).to_numpy(bool)
                    & present.to_numpy(bool)
                ).sum()
            )
            if significance is not None
            else None,
            "tested_visible_target_pairs": int(
                (
                    tested_pairs.reindex(
                        index=variant_rows,
                        columns=variant_cols,
                        fill_value=False,
                    ).to_numpy(bool)
                    & present.to_numpy(bool)
                ).sum()
            )
            if tested_pairs is not None
            else None,
            "atac_positive_tested_visible_pairs": int(
                (
                    atac_peaks.reindex(
                        index=variant_rows,
                        columns=variant_cols,
                        fill_value=False,
                    ).to_numpy(bool)
                    & present.to_numpy(bool)
                ).sum()
            )
            if atac_peaks is not None
            else None,
            "significant_atac_positive_visible_pairs": int(
                (
                    significance.reindex(
                        index=variant_rows,
                        columns=variant_cols,
                        fill_value=False,
                    ).to_numpy(bool)
                    & atac_peaks.reindex(
                        index=variant_rows,
                        columns=variant_cols,
                        fill_value=False,
                    ).to_numpy(bool)
                    & present.to_numpy(bool)
                ).sum()
            )
            if atac_peaks is not None
            else None,
            "on_target_pairs": int(on_target_pairs.to_numpy(bool).sum())
            if on_target_pairs is not None
            else None,
            "removed_empty_rows": int(len(row_order) - len(variant_rows)),
            "removed_empty_columns": int(len(col_order) - len(variant_cols)),
            "color_scale": {
                "vmin": float(vmin),
                "center": float(vcenter),
                "vmax": float(vmax),
                "center_definition": "median of all finite displayed values",
            },
        }

    summary = {
        "activity_calibration": args.activity_calibration,
        "activity_centering": args.activity_centering,
        "subtract_negative_control": args.subtract_negative_control,
        "center_by_mean_negative_controls": args.center_by_mean_negative_controls,
        "bayesian_activity_scale": (
            "posterior_mean_log_gamma_minus_mean_ordinary_negative_controls"
            if args.center_by_mean_negative_controls
            else "posterior_mean_log_gamma_minus_alpha_minus_negative_control"
            if args.subtract_negative_control
            else "posterior_mean_log_gamma_minus_alpha"
            if args.activity_centering == "posterior-alpha"
            else "posterior_mean_log_gamma"
            if args.append_negative_control
            else "raw_log_gamma"
        ),
        "cmap": args.cmap,
        "cmap_center": (
            "median of all finite displayed values"
            if args.cmap.lower() == "coolwarm"
            else None
        ),
        "append_negative_control": args.append_negative_control,
        "append_individual_negative_controls": (
            args.append_individual_negative_controls
        ),
        "individual_negative_control_columns": individual_control_order,
        "individual_negative_control_filter": (
            "same subclass-cCRE T7 filter as target cCREs"
            if args.append_individual_negative_controls
            else None
        ),
        "significance_markers": (
            {
                "source": str(args.significance_tests),
                "threshold_column": "t7_threshold",
                "threshold": float(args.t7_threshold),
                "q_column": "q_right",
                "q_cutoff": float(args.significance_q_cutoff),
                "definition": (
                    "target posterior log_gamma versus the draw-wise mean of all "
                    "seven ordinary negative controls; BH across eligible pairs"
                ),
                "marker": "*",
            }
            if args.significance_tests is not None
            else None
        ),
        "atac_peak_markers": (
            {
                "source": str(args.atac_peaks),
                "peak_definition": "assay value > 0.5",
                "restriction": (
                    "pairs present in the selected T7-threshold significance-test table"
                ),
                "marker": "cell box",
            }
            if args.atac_peaks is not None
            else None
        ),
        "negative_control_column": (
            "zero after centering pooled negative control against itself"
            if args.append_negative_control and args.subtract_negative_control
            else "posterior mean(log_gamma_neg - alpha_neg)"
            if args.append_negative_control
            else (
                "each ordinary control's posterior mean log_gamma minus the "
                "within-subclass mean of all ordinary controls"
            )
            if args.append_individual_negative_controls
            else None
        ),
        "restrict_to_on_target": (
            {
                "cre_rule": (
                    "keep cCREs with at least one visible ATAC-positive significant "
                    "subclass"
                ),
                "subclass_rule": (
                    "then keep subclasses with at least one visible ATAC-positive "
                    "significant pair among the retained cCREs"
                ),
            }
            if args.restrict_to_on_target
            else None
        ),
        "negative_control_cres_removed": sorted(negative_controls),
        "posterior_sources": posterior_sources,
        "library_size_annotation": {
            "source": str(args.library_size_csv),
            "column": "counts",
            "transform": "log10(1 + counts)",
            "cmap": args.library_cmap,
            "negative_control_value": (
                "log10(1 + sum of raw counts across negative-control cCREs)"
                if args.append_negative_control
                else "individual log10(1 + raw count) for each negative-control cCRE"
                if args.append_individual_negative_controls
                else None
            ),
            "pooled_negative_control_raw_count": pooled_negative_library_count,
        },
        "t7_total_annotation": {
            "source": "sum of H5AD T7CRE counts across subclasses for each cCRE",
            "transform": "log10(1 + counts)",
            "cmap": args.library_cmap,
            "normalization": "shared with nanopore library-size annotation",
            "negative_control_value": (
                "log10(1 + sum across subclasses and negative-control cCREs)"
                if args.append_negative_control
                else (
                    "individual log10(1 + count summed across subclasses) for "
                    "each negative-control cCRE"
                )
                if args.append_individual_negative_controls
                else None
            ),
            "pooled_negative_control_raw_count": pooled_negative_t7_count,
        },
        "rows": int(len(row_order)),
        "columns": int(len(col_order)),
        "methods": list(matrices),
        "method_keys": list(args.methods),
        "outputs": outputs,
        "value_tables": value_tables if args.dump_values else None,
        "variants": variant_shapes,
        "t7_threshold": args.t7_threshold,
        "filter_variants": list(args.filter_variants),
        "blacklisted_cres_removed": int(len(blacklist)),
        "blacklist_sources": blacklist_sources,
        "cell_type_order": "subclass_name numeric prefix from H5AD",
        "cre_order": "total T7 counts descending from H5AD T7CRE counts",
        "pair_t7_source": "sum of H5AD T7CRE counts by subclass-cCRE pair",
        "method_roots": {
            spec["label"]: str(method_root(args, spec))
            for _, spec in selected_specs(args)
        },
    }
    write_json(args.figures_dir / f"{args.stem}_manifest.json", summary)
    return summary


def main() -> None:
    args = parse_args()
    args.figures_dir.mkdir(parents=True, exist_ok=True)
    summary = plot_heatmaps(args)
    n_outputs = sum(len(by_method) for by_method in summary["outputs"].values())
    log(
        f"[method activity heatmap] wrote {n_outputs} PDFs with "
        f"{summary['rows']} subclasses x {summary['columns']} cCREs"
    )


if __name__ == "__main__":
    main()
