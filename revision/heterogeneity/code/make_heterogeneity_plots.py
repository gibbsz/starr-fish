#!/usr/bin/env python3
"""Bayesian heterogeneity analysis: subgroup activity vs whole-cell-type activity.

For each of the ten top-T7 cell types, compare every fitted cCRE's activity
estimated from the intact cell type with the mean activity across either five
random subsets or the annotated supertypes recorded by the fit. Annotated
supertypes use a cell-count-weighted mean for intact-fit agreement and retain
an unweighted SD as a heterogeneity measure. A second annotated-supertype
analysis calculates Lin's CCC across cCREs for every within-parent pair.

Outputs (under ``results/``):
  * ``tables/bayesian_subset_vs_whole.csv`` - pair-level agreement data;
  * ``tables/bayesian_subset_vs_whole_summary.csv`` - per-cell-type and overall metrics;
  * ``figures/bayesian_subset_mean_vs_whole.{pdf,png}`` - faceted agreement plot;
  * ``raw/combined_activity_bayesian.csv`` and ``raw/split_activity_bayesian.csv``.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

CODE_DIR = Path(__file__).resolve().parent
ANALYSIS_DIR = CODE_DIR.parent
REVISION_DIR = ANALYSIS_DIR.parent
REF_RESULTS = REVISION_DIR / "bayesian_vs_fold_change" / "results"
sys.path.insert(0, str(REVISION_DIR / "bayesian_vs_fold_change" / "code"))
sys.path.insert(0, str(CODE_DIR))

from analysis_utils import LIBSIZE_CSV, OLD_DATA_BOOTSTRAP, log, write_json  # noqa: E402

MODELS = ("bayesian", "bootstrap")
BOOT_ACTIVITY_FILE = "log_activity_vs_negative_control.csv"
GROUP_RE = re.compile(r"^(?P<subclass>.+)_group_(?P<group>\d+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--combined-bayes-dir", type=Path, default=REF_RESULTS / "bayesian"
    )
    parser.add_argument(
        "--split-bayes-dir",
        type=Path,
        default=ANALYSIS_DIR / "results" / "split" / "bayesian",
    )
    parser.add_argument(
        "--combined-bootstrap-dir", type=Path, default=OLD_DATA_BOOTSTRAP
    )
    parser.add_argument(
        "--split-bootstrap-dir",
        type=Path,
        default=ANALYSIS_DIR / "results" / "split" / "bootstrap",
    )
    parser.add_argument(
        "--model", choices=["bayesian", "bootstrap"], default="bayesian"
    )
    parser.add_argument("--outdir", type=Path, default=ANALYSIS_DIR / "results")
    parser.add_argument(
        "--calibration",
        choices=["self_cre_negctrl", "negctrl_only"],
        default="negctrl_only",
        help=(
            "Bayesian activity scale. 'negctrl_only' (default) preserves a common "
            "per-cCRE scale between the independently fitted intact and split data; "
            "'self_cre_negctrl' additionally centers each fit per cCRE."
        ),
    )
    parser.add_argument(
        "--ncols",
        type=int,
        default=5,
        help="Number of cell-type panel columns (default: 5).",
    )
    return parser.parse_args()


def discover_tag(bayes_dir: Path) -> str:
    manifest = json.loads((bayes_dir / "run_manifest.json").read_text())
    return str(manifest["tag"])


def bayesian_effect_matrix(
    posterior_path: Path, negative_controls: set[str], self_cre: bool = True
) -> pd.DataFrame:
    """Negative-control-centred log effect per group x cCRE.

    With ``self_cre=True`` (default) the posterior ``log_gamma`` is first
    self-cCRE calibrated (per-cCRE mean over all draws and groups subtracted),
    mirroring ``plot_results.bayesian_significance``. With ``self_cre=False``
    only the per-group negative-control mean is subtracted.
    """
    with np.load(posterior_path, allow_pickle=True) as posterior:
        log_gamma = posterior["log_gamma"].astype(np.float64)
        groups = posterior["group_names"].astype(str)
        cres = posterior["cre_names"].astype(str)
    if self_cre:
        log_gamma = log_gamma - log_gamma.mean(axis=(0, 1))[None, None, :]
    negative_mask = np.isin(cres, list(negative_controls))
    if not negative_mask.any():
        raise ValueError(f"{posterior_path} has no negative-control cCREs")
    negative_threshold = log_gamma[:, :, negative_mask].mean(axis=(0, 2))
    effect = log_gamma.mean(axis=0) - negative_threshold[:, None]
    return pd.DataFrame(effect, index=groups, columns=cres)


def bootstrap_effect_matrix(dir_: Path, self_cre: bool = True) -> pd.DataFrame:
    """Negative-control-centred bootstrap log activity per subclass x cCRE.

    ``self_cre=True`` reuses the run's saved ``log_activity_vs_negative_control``
    (self-cCRE calibrated + negative-control centred). ``self_cre=False``
    recomputes from the raw per-bootstrap activity array applying only
    negative-control centering, matching ``average_bootstrap_test_q`` with
    ``calibrate=None, threshold='neg_control_mean'`` (filter mask applied).
    """
    if self_cre:
        frame = pd.read_csv(dir_ / BOOT_ACTIVITY_FILE, index_col=0)
        frame.index = frame.index.astype(str)
        frame.columns = frame.columns.astype(str)
        return frame

    axes = json.loads((dir_ / "bootstrap_axes.json").read_text())
    subclasses = [str(s) for s in axes["subclasses"]]
    cres = [str(c) for c in axes["cres"]]
    negatives = set(pd.read_csv(dir_ / "negative_controls.csv")["cre"].astype(str))
    neg_idx = [i for i, c in enumerate(cres) if c in negatives]
    if not neg_idx:
        raise ValueError(f"{dir_} bootstrap axes contain no negative controls")
    fmask = pd.read_csv(dir_ / "qvalue_filter_mask.csv", index_col=0)
    fmask.index = fmask.index.astype(str)
    fmask.columns = fmask.columns.astype(str)
    fmask = fmask.reindex(index=subclasses, columns=cres).fillna(True).astype(bool)

    arr = np.load(dir_ / "celltype_activity_array.npy", mmap_mode="r")
    n_boot, n_groups, n_cres = arr.shape
    if (n_groups, n_cres) != (len(subclasses), len(cres)):
        raise ValueError("bootstrap array axes do not match bootstrap_axes.json")
    mask = fmask.to_numpy()  # (G, C) bool, True => filtered
    neg_idx = np.asarray(neg_idx)

    # Stream contiguous bootstrap-chunks (arr is C-contiguous over (boot, g, cre),
    # so arr[b0:b1] is a sequential read) accumulating nan-aware sums/counts, so
    # the 11 GB array is read once sequentially rather than strided per group.
    res_sum = np.zeros((n_groups, n_cres))
    res_cnt = np.zeros((n_groups, n_cres), dtype=np.int64)
    neg_sum = np.zeros(n_groups)
    neg_cnt = np.zeros(n_groups, dtype=np.int64)
    chunk = 1000
    with np.errstate(invalid="ignore", divide="ignore"):
        for b0 in range(0, n_boot, chunk):
            block = np.log(np.asarray(arr[b0 : b0 + chunk], dtype=np.float64))
            block[~np.isfinite(block)] = np.nan
            block[:, mask] = np.nan  # broadcast filter over the chunk's bootstraps
            valid = ~np.isnan(block)
            res_sum += np.nansum(block, axis=0)
            res_cnt += valid.sum(axis=0)
            neg_per_boot = np.nanmean(block[:, :, neg_idx], axis=2)  # (nb, G)
            neg_valid = ~np.isnan(neg_per_boot)
            neg_sum += np.nansum(neg_per_boot, axis=0)
            neg_cnt += neg_valid.sum(axis=0)
    res_df = np.where(res_cnt > 0, res_sum / np.maximum(res_cnt, 1), np.nan)
    fdc = np.where(neg_cnt > 0, neg_sum / np.maximum(neg_cnt, 1), np.nan)
    effect = res_df - fdc[:, None]
    return pd.DataFrame(effect, index=subclasses, columns=cres)


def load_bayesian(dir_: Path, self_cre: bool = True) -> pd.DataFrame:
    tag = discover_tag(dir_)
    negatives = set(
        pd.read_csv(dir_ / "negative_controls.csv")["cre"].astype(str)
    )
    return bayesian_effect_matrix(
        dir_ / f"{tag}_posterior_samples.npz", negatives, self_cre=self_cre
    )


def load_bootstrap(dir_: Path, self_cre: bool = True) -> pd.DataFrame:
    return bootstrap_effect_matrix(dir_, self_cre=self_cre)


def negative_controls(dir_: Path) -> set[str]:
    path = dir_ / "negative_controls.csv"
    if not path.exists():
        return set()
    return set(pd.read_csv(path)["cre"].astype(str))


def blacklist(dir_: Path) -> set[str]:
    path = dir_ / "cre_blacklist.csv"
    if not path.exists():
        return set()
    return set(pd.read_csv(path).iloc[:, 0].astype(str))


def split_targets(
    split_bayes_dir: Path,
) -> tuple[list[str], dict[str, list[str]], str]:
    """Return target order, subgroup membership, and grouping strategy.

    Existing random-split runs predate the explicit membership map, so retain
    a fallback that reconstructs their ``<subclass>_group_i`` labels.
    """
    manifest = json.loads((split_bayes_dir / "run_manifest.json").read_text())
    targets = [str(target) for target in manifest["split_subclasses"]]
    grouping = str(manifest.get("grouping", "random"))
    configured = manifest.get("subgroups_by_subclass")
    if configured is None:
        n_groups = int(manifest["n_groups"])
        members = {
            target: [
                f"{target}_group_{group}" for group in range(1, n_groups + 1)
            ]
            for target in targets
        }
    else:
        members = {
            target: [str(member) for member in configured[target]]
            for target in targets
        }
    empty = [target for target, labels in members.items() if not labels]
    if empty:
        raise ValueError(f"split manifest has targets without subgroups: {empty}")
    if grouping not in {"random", "supertype"}:
        raise ValueError(f"unsupported grouping={grouping!r}")
    return targets, members, grouping


def artifact_names(grouping: str, model: str = "bayesian") -> dict[str, str]:
    """Stable filenames for the legacy random and new supertype analyses."""
    if grouping == "random" and model == "bayesian":
        return {
            "table": "bayesian_subset_vs_whole.csv",
            "summary": "bayesian_subset_vs_whole_summary.csv",
            "split_raw": "split_activity_bayesian.csv",
            "figure": "bayesian_subset_mean_vs_whole",
            "manifest": "heterogeneity_manifest.json",
        }
    if grouping == "supertype" and model == "bayesian":
        return {
            "table": "bayesian_supertype_vs_whole.csv",
            "summary": "bayesian_supertype_vs_whole_summary.csv",
            "split_raw": "supertype_activity_bayesian.csv",
            "figure": "bayesian_supertype_mean_vs_whole",
            "pairwise_table": "bayesian_supertype_pairwise_ccc.csv",
            "pairwise_summary": "bayesian_supertype_pairwise_ccc_summary.csv",
            "pairwise_figure": "bayesian_supertype_pairwise_ccc",
            "pairwise_support_figure": (
                "bayesian_supertype_pairwise_ccc_vs_min_cells"
            ),
            "manifest": "heterogeneity_manifest.json",
        }
    if grouping == "random" and model == "bootstrap":
        return {
            "table": "bootstrap_subset_vs_whole.csv",
            "summary": "bootstrap_subset_vs_whole_summary.csv",
            "split_raw": "split_activity_bootstrap.csv",
            "figure": "bootstrap_subset_mean_vs_whole",
            "manifest": "bootstrap_heterogeneity_manifest.json",
        }
    if grouping == "supertype" and model == "bootstrap":
        return {
            "table": "bootstrap_supertype_vs_whole.csv",
            "summary": "bootstrap_supertype_vs_whole_summary.csv",
            "split_raw": "supertype_activity_bootstrap.csv",
            "figure": "bootstrap_supertype_mean_vs_whole",
            "pairwise_table": "bootstrap_supertype_pairwise_ccc.csv",
            "pairwise_summary": "bootstrap_supertype_pairwise_ccc_summary.csv",
            "pairwise_figure": "bootstrap_supertype_pairwise_ccc",
            "pairwise_support_figure": (
                "bootstrap_supertype_pairwise_ccc_vs_min_cells"
            ),
            "manifest": "bootstrap_heterogeneity_manifest.json",
        }
    raise ValueError(f"unsupported grouping={grouping!r}, model={model!r}")


def assemble(
    combined: dict[str, pd.DataFrame],
    split: dict[str, pd.DataFrame],
    targets: list[str],
    n_groups: int,
    drop_negatives: set[str],
) -> tuple[pd.DataFrame, dict, dict]:
    """Return long activity table plus per-model variance/diff matrices."""
    long_rows = []
    variance = {model: {} for model in MODELS}
    signed_diff = {model: {} for model in MODELS}

    for model in MODELS:
        comb = combined[model]
        spl = split[model]
        # Common non-blacklist / non-negative-control cCREs across both fits.
        cres = comb.columns.intersection(spl.columns)
        cres = [c for c in cres if c not in drop_negatives]
        for subclass in targets:
            if subclass not in comb.index:
                raise KeyError(f"{subclass!r} missing from combined {model} activity")
            member_labels = [f"{subclass}_group_{g}" for g in range(1, n_groups + 1)]
            missing = [m for m in member_labels if m not in spl.index]
            if missing:
                raise KeyError(f"missing split groups for {subclass}: {missing}")

            comb_vec = comb.loc[subclass, cres].astype(float)
            group_mat = spl.loc[member_labels, cres].astype(float)

            variance[model][subclass] = group_mat.var(axis=0, ddof=1)
            signed_diff[model][subclass] = (group_mat - comb_vec).mean(axis=0)

            for cre in cres:
                long_rows.append(
                    (model, subclass, "combined", cre, float(comb_vec[cre]))
                )
            for g, label in enumerate(member_labels, start=1):
                gvec = group_mat.loc[label]
                for cre in cres:
                    long_rows.append(
                        (model, subclass, f"group_{g}", cre, float(gvec[cre]))
                    )

    long = pd.DataFrame(
        long_rows, columns=["model", "subclass", "member", "cre", "activity"]
    )
    var_mats = {
        model: pd.DataFrame(variance[model]).T.reindex(targets) for model in MODELS
    }
    diff_mats = {
        model: pd.DataFrame(signed_diff[model]).T.reindex(targets) for model in MODELS
    }
    return long, var_mats, diff_mats


def save_figure(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), bbox_inches="tight", dpi=200)
    plt.close(fig)


def order_cres(
    activity_row: pd.Series, cres: list[str], neg_set: set[str], mode: str
) -> tuple[list[str], int]:
    """Real cCREs first, negative controls appended at the end.

    Returns the ordered cCRE list and the number of real (non-control) cCREs,
    i.e. the index where the negative-control block starts.
    """
    real = [c for c in cres if c not in neg_set]
    negs = [c for c in cres if c in neg_set]
    if mode == "activity":
        key = lambda c: activity_row.get(c, -np.inf)
        real.sort(key=key, reverse=True)
        negs.sort(key=key, reverse=True)
    else:
        real.sort()
        negs.sort()
    return real + negs, len(real)


def plot_subclass(
    subclass: str,
    combined: dict[str, pd.DataFrame],
    split: dict[str, pd.DataFrame],
    n_groups: int,
    cre_order: list[str],
    n_real: int,
    figures_dir: Path,
) -> None:
    """Two stacked panels (Bayesian, bootstrap). Per cCRE: a box over the five
    subgroups and a point for the intact-subclass estimate beside it. Negative
    controls sit at the right end, shaded and labelled."""
    members = [f"{subclass}_group_{g}" for g in range(1, n_groups + 1)]
    n = len(cre_order)
    base = np.arange(n)
    width = max(24.0, n * 0.16)
    fig, axes = plt.subplots(
        len(MODELS), 1, figsize=(width, 6 * len(MODELS) + 1), sharex=True
    )
    box_offset, pt_offset = -0.12, 0.26
    for ax, model in zip(np.atleast_1d(axes), MODELS):
        comb = combined[model]
        spl = split[model].reindex(index=members, columns=cre_order)
        comb_row = comb.reindex(index=[subclass], columns=cre_order).iloc[0]

        box_data, positions = [], []
        for i, cre in enumerate(cre_order):
            vals = spl[cre].to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            if vals.size:
                box_data.append(vals)
                positions.append(i + box_offset)
        if box_data:
            bp = ax.boxplot(
                box_data, positions=positions, widths=0.4, showfliers=False,
                patch_artist=True, manage_ticks=False,
            )
            for patch in bp["boxes"]:
                patch.set_facecolor("#f58518")
                patch.set_alpha(0.6)
            for whisk in bp["whiskers"] + bp["caps"]:
                whisk.set_linewidth(0.6)
            for med in bp["medians"]:
                med.set_color("#7a3d00")
                med.set_linewidth(0.8)

        cy = comb_row.to_numpy(dtype=float)
        finite = np.isfinite(cy)
        ax.scatter(
            base[finite] + pt_offset, cy[finite], marker="D", s=9,
            color="#4c78a8", zorder=5, linewidths=0,
        )

        if n_real < n:  # shade + separate the negative-control block
            ax.axvspan(n_real - 0.5, n - 0.5, color="#d62728", alpha=0.06, zorder=0)
            ax.axvline(n_real - 0.5, color="#d62728", ls="--", lw=0.8, zorder=1)
            ax.text(
                (n_real + n) / 2 - 0.5, 0.98, "negative\ncontrols",
                transform=ax.get_xaxis_transform(), ha="center", va="top",
                fontsize=8, color="#d62728",
            )
        ax.axhline(0, color="grey", linewidth=0.8, linestyle="--")
        ax.set_title(model.capitalize(), loc="left")
        ax.set_ylabel("log activity above\nnegative-control mean")
        ax.set_xlim(-1, n)

    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor="#f58518", alpha=0.6),
        plt.Line2D([0], [0], marker="D", color="#4c78a8", linestyle="none", markersize=6),
    ]
    axes[0].legend(
        handles, [f"{n_groups} subgroups", "combined subclass"],
        frameon=False, loc="upper right", ncol=2,
    )
    ax_bottom = np.atleast_1d(axes)[-1]
    ax_bottom.set_xticks(base)
    ax_bottom.set_xticklabels(cre_order, rotation=90, fontsize=max(2.0, min(6.0, 900 / n)))
    for i, lbl in enumerate(ax_bottom.get_xticklabels()):
        if i >= n_real:
            lbl.set_color("#d62728")
    ax_bottom.set_xlabel("cCRE")
    fig.suptitle(f"{subclass}: intact subclass vs {n_groups} random subgroups", y=0.995)
    fig.tight_layout()
    save_figure(fig, figures_dir / f"{subclass}_heterogeneity")


def plot_overview(
    var_mats: dict, diff_mats: dict, targets: list[str], figures_dir: Path
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(max(10, len(targets) * 1.1), 10))
    metrics = [("Per-cCRE subgroup variance", var_mats, False),
               ("Per-cCRE signed diff (subgroup - combined)", diff_mats, True)]
    offsets = {"bayesian": -0.18, "bootstrap": 0.18}
    colors = {"bayesian": "#4c78a8", "bootstrap": "#f58518"}
    x = np.arange(len(targets))
    for ax, (title, mats, center_line) in zip(axes, metrics):
        for model in MODELS:
            mat = mats[model].reindex(targets)
            data = [mat.loc[s].dropna().to_numpy() for s in targets]
            positions = x + offsets[model]
            box = ax.boxplot(
                data, positions=positions, widths=0.32, showfliers=False,
                patch_artist=True,
            )
            for patch in box["boxes"]:
                patch.set_facecolor(colors[model])
                patch.set_alpha(0.7)
        if center_line:
            ax.axhline(0, color="grey", linewidth=0.8, linestyle="--")
        ax.set_xticks(x)
        ax.set_xticklabels(targets, rotation=45, ha="right")
        ax.set_title(title)
    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[m], alpha=0.7) for m in MODELS]
    axes[0].legend(handles, [m.capitalize() for m in MODELS], frameon=False)
    fig.tight_layout()
    save_figure(fig, figures_dir / "heterogeneity_overview")


def within_vs_across(
    combined: dict[str, pd.DataFrame],
    var_mats: dict[str, pd.DataFrame],
    targets: list[str],
) -> pd.DataFrame:
    """Per-cCRE within- vs across-cell-type variance for each model.

    within  = mean over the 10 subclasses of the 5-subgroup variance (x-axis).
    across  = variance of the combined estimate across the same 10 subclasses.
    """
    rows = []
    for model in MODELS:
        var_mat = var_mats[model]
        cres = list(var_mat.columns)
        within = var_mat.mean(axis=0, skipna=True)
        n_within = var_mat.notna().sum(axis=0)
        comb = combined[model].reindex(index=targets, columns=cres)
        across = comb.var(axis=0, ddof=1, skipna=True)
        n_across = comb.notna().sum(axis=0)
        rows.append(
            pd.DataFrame(
                {
                    "model": model,
                    "cre": cres,
                    "within_ct_variance": within.to_numpy(),
                    "across_ct_variance": across.to_numpy(),
                    "n_subclasses_within": n_within.to_numpy(),
                    "n_celltypes_across": n_across.to_numpy(),
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def plot_within_vs_across(
    table: pd.DataFrame, counts: pd.Series, figures_dir: Path
) -> None:
    log_counts = np.log10(counts.astype(float))
    vmin, vmax = float(log_counts.min()), float(log_counts.max())
    fig, axes = plt.subplots(1, len(MODELS), figsize=(7 * len(MODELS) + 1, 6.5))
    scatter = None
    for ax, model in zip(np.atleast_1d(axes), MODELS):
        sub = table[table["model"] == model]
        x = sub["within_ct_variance"].to_numpy(dtype=float)
        y = sub["across_ct_variance"].to_numpy(dtype=float)
        c = log_counts.reindex(sub["cre"]).to_numpy(dtype=float)
        # require >=2 cell types for a defined across-variance, a positive pair,
        # and a known nanopore count for the colour.
        ok = (
            np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0) & np.isfinite(c)
            & (sub["n_celltypes_across"].to_numpy() >= 2)
        )
        x, y, c = x[ok], y[ok], c[ok]
        scatter = ax.scatter(
            x, y, c=c, s=16, alpha=0.75, cmap="viridis", vmin=vmin, vmax=vmax,
            linewidths=0,
        )
        if x.size:
            lo = float(min(x.min(), y.min()))
            hi = float(max(x.max(), y.max()))
            ax.plot([lo, hi], [lo, hi], color="black", ls="--", lw=1, label="y = x")
            n_above = int((y > x).sum())
            rho = spearman(x, y)
            ax.set_title(
                f"{model.capitalize()}  (n={x.size}, ρ={rho:.2f}, "
                f"{n_above}/{x.size} above y=x)"
            )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Within-cell-type variance\n(mean 5-subgroup variance over 10 subclasses)")
        ax.set_ylabel("Across-cell-type variance\n(combined estimate over 10 subclasses)")
        ax.legend(frameon=False, loc="upper left")
    if scatter is not None:
        cbar = fig.colorbar(scatter, ax=np.atleast_1d(axes).tolist(), fraction=0.03, pad=0.02)
        cbar.set_label("Nanopore sequencing count (log$_{10}$)")
    fig.suptitle("Per-cCRE within- vs across-cell-type activity variance")
    save_figure(fig, figures_dir / "within_vs_across_variance")


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 3:
        return np.nan
    return float(spearmanr(x, y).statistic)


def anova_variance_components(
    split: dict[str, pd.DataFrame], targets: list[str], n_groups: int, cres: list[str]
) -> pd.DataFrame:
    """Per-cCRE one-way random-effects decomposition over the 50 subgroup
    estimates (10 cell types x n_groups subgroups), cell type as random factor.

    Returns, per (model, cCRE): pooled within-cell-type residual variance
    (sigma2_within = MS_within), the bias-corrected between-cell-type variance
    (sigma2_between = max(0, (MS_between - MS_within) / n0)), the intraclass
    correlation ICC, and the F statistic. Handles the unbalanced case (NaN
    subgroups) via the general moment estimator for n0.
    """
    members = [f"{t}_group_{g}" for t in targets for g in range(1, n_groups + 1)]
    g, k = len(targets), n_groups
    rows = []
    for model in MODELS:
        mat = split[model].reindex(index=members, columns=cres)
        y = mat.to_numpy(dtype=float).reshape(g, k, len(cres))  # (celltype, subgroup, cCRE)
        finite = ~np.isnan(y)
        n_t = finite.sum(axis=1)  # (g, C)
        with np.errstate(invalid="ignore", divide="ignore"):
            sum_t = np.nansum(y, axis=1)  # (g, C)
            mean_t = sum_t / np.where(n_t > 0, n_t, np.nan)
            N = n_t.sum(axis=0)  # (C,)
            grand = np.nansum(sum_t, axis=0) / np.where(N > 0, N, np.nan)
            ss_within = np.nansum((y - mean_t[:, None, :]) ** 2, axis=(0, 1))
            ss_between = np.nansum(n_t * (mean_t - grand[None, :]) ** 2, axis=0)
            g_eff = (n_t > 0).sum(axis=0)  # (C,)
            df_within = np.where(n_t > 0, n_t - 1, 0).sum(axis=0)
            df_between = g_eff - 1
            ms_within = ss_within / np.where(df_within > 0, df_within, np.nan)
            ms_between = ss_between / np.where(df_between > 0, df_between, np.nan)
            sum_nt2 = (n_t ** 2).sum(axis=0)
            n0 = (N - sum_nt2 / np.where(N > 0, N, np.nan)) / np.where(
                g_eff > 1, g_eff - 1, np.nan
            )
            sigma2_within = ms_within
            sigma2_between = np.maximum(0.0, (ms_between - ms_within) / n0)
            icc = sigma2_between / (sigma2_between + sigma2_within)
            f_stat = ms_between / ms_within
        valid = (g_eff >= 2) & (df_within >= 1)
        rows.append(
            pd.DataFrame(
                {
                    "model": model,
                    "cre": cres,
                    "sigma2_within": np.where(valid, sigma2_within, np.nan),
                    "sigma2_between": np.where(valid, sigma2_between, np.nan),
                    "icc": np.where(valid, icc, np.nan),
                    "f_stat": np.where(valid, f_stat, np.nan),
                    "n_celltypes": g_eff,
                    "n_obs": N,
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def plot_variance_components(
    table: pd.DataFrame, counts: pd.Series, figures_dir: Path
) -> None:
    log_counts = np.log10(counts.astype(float))
    vmin, vmax = float(log_counts.min()), float(log_counts.max())
    fig, axes = plt.subplots(1, len(MODELS), figsize=(7 * len(MODELS) + 1, 6.5))
    scatter = None
    for ax, model in zip(np.atleast_1d(axes), MODELS):
        sub = table[table["model"] == model]
        x = sub["sigma2_within"].to_numpy(dtype=float)
        y = sub["sigma2_between"].to_numpy(dtype=float)
        c = log_counts.reindex(sub["cre"]).to_numpy(dtype=float)
        ok = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0) & np.isfinite(c)
        xf, yf, cf = x[ok], y[ok], c[ok]
        scatter = ax.scatter(
            xf, yf, c=cf, s=16, alpha=0.75, cmap="viridis", vmin=vmin, vmax=vmax,
            linewidths=0,
        )
        if xf.size:
            lo = float(min(xf.min(), yf.min()))
            hi = float(max(xf.max(), yf.max()))
            ax.plot([lo, hi], [lo, hi], color="black", ls="--", lw=1, label="y = x")
            med_icc = float(np.nanmedian(sub["icc"]))
            ax.set_title(
                f"{model.capitalize()}  (n={xf.size}, median ICC={med_icc:.2f}, "
                f"{int((yf > xf).sum())}/{xf.size} above y=x)"
            )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Within-cell-type variance  σ²_within\n(pooled residual MS)")
        ax.set_ylabel("Between-cell-type variance  σ²_between\n(bias-corrected)")
        ax.legend(frameon=False, loc="upper left")
    if scatter is not None:
        cbar = fig.colorbar(scatter, ax=np.atleast_1d(axes).tolist(), fraction=0.03, pad=0.02)
        cbar.set_label("Nanopore sequencing count (log$_{10}$)")
    fig.suptitle("Per-cCRE random-effects variance decomposition (cell type as factor)")
    save_figure(fig, figures_dir / "variance_components_anova")


def within_sd_vs_nn_distance(
    combined: dict[str, pd.DataFrame],
    var_mats: dict[str, pd.DataFrame],
    targets: list[str],
    scope: str,
) -> pd.DataFrame:
    """Per (cCRE, cell type): within-cell-type SD vs distance to the nearest
    other cell type's combined estimate for that cCRE.

    within_sd  = sqrt of the 5-subgroup variance (log-activity scale).
    nn_distance = min over other cell types of |combined[c, t] - combined[c, t']|.
    scope='all' compares against every cell type in the combined run;
    'selected' compares only against the other 9 split subclasses.
    """
    rows = []
    for model in MODELS:
        cres = list(var_mats[model].columns)
        comb = combined[model]
        comb.columns = comb.columns.astype(str)
        pool = comb if scope == "all" else comb.reindex(index=targets)
        sd = np.sqrt(var_mats[model])
        for t in targets:
            if t not in comb.index:
                continue
            self_vals = comb.reindex(index=[t], columns=cres).iloc[0]
            others = pool.drop(index=t, errors="ignore").reindex(columns=cres)
            nn = (others - self_vals).abs().min(axis=0, skipna=True)
            rows.append(
                pd.DataFrame(
                    {
                        "model": model,
                        "celltype": t,
                        "cre": cres,
                        "within_sd": sd.loc[t, cres].to_numpy(dtype=float),
                        "nn_distance": nn.reindex(cres).to_numpy(dtype=float),
                    }
                )
            )
    return pd.concat(rows, ignore_index=True)


def plot_within_sd_vs_nn(table: pd.DataFrame, targets: list[str], figures_dir: Path) -> None:
    palette = {t: plt.cm.tab10(i % 10) for i, t in enumerate(targets)}
    fig, axes = plt.subplots(1, len(MODELS), figsize=(7.5 * len(MODELS), 6.8))
    for ax, model in zip(np.atleast_1d(axes), MODELS):
        sub = table[table["model"] == model]
        x = sub["within_sd"].to_numpy(dtype=float)
        y = sub["nn_distance"].to_numpy(dtype=float)
        ok = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
        colors = sub["celltype"].map(palette).to_numpy()
        ax.scatter(x[ok], y[ok], s=10, alpha=0.5, c=list(colors[ok]), linewidths=0)
        xf, yf = x[ok], y[ok]
        if xf.size:
            lo = float(min(xf.min(), yf.min()))
            hi = float(max(xf.max(), yf.max()))
            ax.plot([lo, hi], [lo, hi], color="black", ls="--", lw=1)
            frac = float((yf > xf).mean())
            ax.set_title(
                f"{model.capitalize()}  (n={xf.size}, "
                f"{frac:.0%} with NN distance > within SD)"
            )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Within-cell-type SD  (√ 5-subgroup variance)")
        ax.set_ylabel("Distance to nearest other cell type\n(|combined activity difference|)")
    handles = [
        plt.Line2D([0], [0], marker="o", linestyle="none", color=palette[t], markersize=6)
        for t in targets
    ]
    fig.legend(
        handles, targets, frameon=False, loc="center left",
        bbox_to_anchor=(1.0, 0.5), fontsize=8, title="cell type",
    )
    fig.suptitle("Within-cell-type SD vs nearest-neighbour cell-type distance")
    save_figure(fig, figures_dir / "within_sd_vs_nn_distance")


def bayesian_subset_agreement(
    combined: pd.DataFrame,
    split: pd.DataFrame,
    targets: list[str],
    subgroups_by_subclass: dict[str, list[str]],
    excluded_cres: set[str],
    subgroup_weights: dict[str, dict[str, float]] | None = None,
) -> pd.DataFrame:
    """Return intact activity and subgroup summaries.

    When ``subgroup_weights`` is supplied, ``mean_subgroup_activity`` is the
    cell-count-weighted mean used for agreement with the intact fit.  The
    unweighted mean and sample SD are retained because they describe a typical
    subgroup and the between-subgroup heterogeneity, respectively.
    """
    cres = [
        cre
        for cre in combined.columns.intersection(split.columns).astype(str)
        if cre not in excluded_cres
    ]
    rows = []
    for cell_type in targets:
        if cell_type not in combined.index:
            raise KeyError(f"{cell_type!r} missing from intact Bayesian activity")
        members = subgroups_by_subclass[cell_type]
        missing = [member for member in members if member not in split.index]
        if missing:
            raise KeyError(f"missing split groups for {cell_type}: {missing}")
        subset = split.reindex(index=members, columns=cres).astype(float)
        whole = combined.reindex(index=[cell_type], columns=cres).iloc[0].astype(float)
        unweighted_mean = subset.mean(axis=0).to_numpy(dtype=float)
        if subgroup_weights is None:
            weighted_mean = unweighted_mean.copy()
        else:
            if cell_type not in subgroup_weights:
                raise KeyError(f"missing subgroup weights for {cell_type!r}")
            missing_weights = [
                member
                for member in members
                if member not in subgroup_weights[cell_type]
            ]
            if missing_weights:
                raise KeyError(
                    f"missing subgroup weights for {cell_type}: {missing_weights}"
                )
            weights = np.asarray(
                [subgroup_weights[cell_type][member] for member in members],
                dtype=float,
            )
            if not np.isfinite(weights).all() or (weights <= 0).any():
                raise ValueError(
                    f"subgroup weights for {cell_type} must be finite and positive"
                )
            values = subset.to_numpy(dtype=float)
            finite = np.isfinite(values)
            denominators = (finite * weights[:, np.newaxis]).sum(axis=0)
            numerators = np.where(finite, values, 0.0) * weights[:, np.newaxis]
            weighted_mean = np.divide(
                numerators.sum(axis=0),
                denominators,
                out=np.full(values.shape[1], np.nan, dtype=float),
                where=denominators > 0,
            )
        rows.append(
            pd.DataFrame(
                {
                    "cell_type": cell_type,
                    "cre": cres,
                    "whole_activity": whole.to_numpy(dtype=float),
                    "mean_subgroup_activity": weighted_mean,
                    "unweighted_mean_subgroup_activity": unweighted_mean,
                    "subgroup_sd": subset.std(axis=0, ddof=1).to_numpy(dtype=float),
                    "n_subgroups": subset.notna().sum(axis=0).to_numpy(dtype=int),
                }
            )
        )
    output = pd.concat(rows, ignore_index=True)
    output["difference"] = (
        output["mean_subgroup_activity"] - output["whole_activity"]
    )
    output["absolute_difference"] = output["difference"].abs()
    output["unweighted_difference"] = (
        output["unweighted_mean_subgroup_activity"] - output["whole_activity"]
    )
    output["unweighted_absolute_difference"] = (
        output["unweighted_difference"].abs()
    )
    return output


def vector_agreement_metrics(
    x: np.ndarray, y: np.ndarray
) -> dict[str, float | int]:
    """Agreement metrics for two vectors, including Lin's CCC."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if x.size == 0:
        return {
            "n": 0,
            "pearson_r": np.nan,
            "concordance_correlation": np.nan,
            "mean_error": np.nan,
            "mae": np.nan,
            "rmse": np.nan,
        }
    delta = y - x
    x_mean, y_mean = float(x.mean()), float(y.mean())
    x_var, y_var = float(x.var()), float(y.var())
    covariance = float(np.mean((x - x_mean) * (y - y_mean)))
    denominator = x_var + y_var + (x_mean - y_mean) ** 2
    concordance = 2 * covariance / denominator if denominator > 0 else np.nan
    pearson = (
        float(np.corrcoef(x, y)[0, 1])
        if x.size > 1 and x_var > 0 and y_var > 0
        else np.nan
    )
    return {
        "n": int(x.size),
        "pearson_r": pearson,
        "concordance_correlation": float(concordance),
        "mean_error": float(delta.mean()),
        "mae": float(np.abs(delta).mean()),
        "rmse": float(np.sqrt(np.mean(delta**2))),
    }


def agreement_metrics(frame: pd.DataFrame) -> dict[str, float | int]:
    """Metrics of agreement with y=x, including Lin's concordance correlation."""
    x = frame["whole_activity"].to_numpy(dtype=float)
    metrics = vector_agreement_metrics(
        x, frame["mean_subgroup_activity"].to_numpy(dtype=float)
    )
    finite_x = x[np.isfinite(x)]
    metrics["whole_activity_range"] = (
        float(finite_x.max() - finite_x.min()) if finite_x.size else np.nan
    )
    return metrics


def pairwise_supertype_agreement(
    split: pd.DataFrame,
    targets: list[str],
    subgroups_by_subclass: dict[str, list[str]],
    subgroup_weights: dict[str, dict[str, float]],
    excluded_cres: set[str],
) -> pd.DataFrame:
    """Compute agreement across cCREs for every within-parent supertype pair."""
    cres = [cre for cre in split.columns.astype(str) if cre not in excluded_cres]
    rows: list[dict[str, float | int | str]] = []
    for cell_type in targets:
        members = subgroups_by_subclass[cell_type]
        missing = [member for member in members if member not in split.index]
        if missing:
            raise KeyError(f"missing split groups for {cell_type}: {missing}")
        missing_weights = [
            member
            for member in members
            if member not in subgroup_weights.get(cell_type, {})
        ]
        if missing_weights:
            raise KeyError(
                f"missing subgroup weights for {cell_type}: {missing_weights}"
            )
        for first_index, first in enumerate(members):
            first_values = split.reindex(index=[first], columns=cres).iloc[0]
            for second in members[first_index + 1 :]:
                second_values = split.reindex(index=[second], columns=cres).iloc[0]
                metrics = vector_agreement_metrics(
                    first_values.to_numpy(dtype=float),
                    second_values.to_numpy(dtype=float),
                )
                first_cells = int(subgroup_weights[cell_type][first])
                second_cells = int(subgroup_weights[cell_type][second])
                rows.append(
                    {
                        "cell_type": cell_type,
                        "supertype_1": first,
                        "supertype_2": second,
                        "n_cells_1": first_cells,
                        "n_cells_2": second_cells,
                        "minimum_pair_cells": min(first_cells, second_cells),
                        "geometric_mean_pair_cells": float(
                            np.sqrt(first_cells * second_cells)
                        ),
                        "n_cres": metrics["n"],
                        "pearson_r": metrics["pearson_r"],
                        "concordance_correlation": metrics[
                            "concordance_correlation"
                        ],
                        "mean_difference": metrics["mean_error"],
                        "mae": metrics["mae"],
                        "rmse": metrics["rmse"],
                    }
                )
    columns = [
        "cell_type",
        "supertype_1",
        "supertype_2",
        "n_cells_1",
        "n_cells_2",
        "minimum_pair_cells",
        "geometric_mean_pair_cells",
        "n_cres",
        "pearson_r",
        "concordance_correlation",
        "mean_difference",
        "mae",
        "rmse",
    ]
    return pd.DataFrame(rows, columns=columns)


def summarize_pairwise_supertype_agreement(
    pairwise: pd.DataFrame,
    targets: list[str],
    subgroups_by_subclass: dict[str, list[str]],
) -> pd.DataFrame:
    """Summarize the pairwise-CCC distribution separately for each parent."""
    rows = []
    for cell_type in targets:
        frame = pairwise[pairwise["cell_type"] == cell_type]
        ccc = frame["concordance_correlation"].dropna()
        pearson = frame["pearson_r"].dropna()
        mae = frame["mae"].dropna()
        n_supertypes = len(subgroups_by_subclass[cell_type])
        expected_pairs = n_supertypes * (n_supertypes - 1) // 2
        if len(frame) != expected_pairs:
            raise ValueError(
                f"{cell_type} has {len(frame)} pair rows; expected {expected_pairs}"
            )
        rows.append(
            {
                "cell_type": cell_type,
                "n_supertypes": n_supertypes,
                "n_pairs": int(len(frame)),
                "median_pairwise_ccc": float(ccc.median()) if ccc.size else np.nan,
                "mean_pairwise_ccc": float(ccc.mean()) if ccc.size else np.nan,
                "q25_pairwise_ccc": float(ccc.quantile(0.25)) if ccc.size else np.nan,
                "q75_pairwise_ccc": float(ccc.quantile(0.75)) if ccc.size else np.nan,
                "min_pairwise_ccc": float(ccc.min()) if ccc.size else np.nan,
                "max_pairwise_ccc": float(ccc.max()) if ccc.size else np.nan,
                "median_pairwise_pearson_r": (
                    float(pearson.median()) if pearson.size else np.nan
                ),
                "median_pairwise_mae": float(mae.median()) if mae.size else np.nan,
            }
        )
    return pd.DataFrame(rows)


def summarize_agreement(
    table: pd.DataFrame, include_supertype_heterogeneity: bool = False
) -> pd.DataFrame:
    rows = []
    for cell_type, frame in table.groupby("cell_type", sort=False):
        row = {"cell_type": cell_type, **agreement_metrics(frame)}
        if include_supertype_heterogeneity:
            finite_sd = frame["subgroup_sd"].dropna()
            row.update(
                {
                    "median_supertype_sd": float(finite_sd.median()),
                    "mean_supertype_sd": float(finite_sd.mean()),
                    "n_cres_with_supertype_sd": int(finite_sd.size),
                }
            )
        rows.append(row)
    overall = {"cell_type": "ALL", **agreement_metrics(table)}
    if include_supertype_heterogeneity:
        finite_sd = table["subgroup_sd"].dropna()
        overall.update(
            {
                "median_supertype_sd": float(finite_sd.median()),
                "mean_supertype_sd": float(finite_sd.mean()),
                "n_cres_with_supertype_sd": int(finite_sd.size),
            }
        )
    rows.append(overall)
    return pd.DataFrame(rows)


def agreement_for_export(table: pd.DataFrame, grouping: str) -> pd.DataFrame:
    """Give exported columns terminology specific to the grouping strategy."""
    if grouping == "random":
        return table[
            [
                "cell_type",
                "cre",
                "whole_activity",
                "mean_subgroup_activity",
                "subgroup_sd",
                "n_subgroups",
                "difference",
                "absolute_difference",
            ]
        ].rename(
            columns={
                "mean_subgroup_activity": "mean_subset_activity",
                "subgroup_sd": "subset_sd",
                "n_subgroups": "n_subsets",
            }
        )
    if grouping == "supertype":
        return table[
            [
                "cell_type",
                "cre",
                "whole_activity",
                "mean_subgroup_activity",
                "unweighted_mean_subgroup_activity",
                "subgroup_sd",
                "n_subgroups",
                "difference",
                "absolute_difference",
                "unweighted_difference",
                "unweighted_absolute_difference",
            ]
        ].rename(
            columns={
                "mean_subgroup_activity": "cell_weighted_mean_supertype_activity",
                "unweighted_mean_subgroup_activity": "unweighted_mean_supertype_activity",
                "subgroup_sd": "supertype_sd",
                "n_subgroups": "n_supertypes",
                "difference": "cell_weighted_difference",
                "absolute_difference": "cell_weighted_absolute_difference",
            }
        )
    raise ValueError(f"unsupported grouping={grouping!r}")


def plot_bayesian_subset_agreement(
    table: pd.DataFrame,
    summary: pd.DataFrame,
    targets: list[str],
    cell_counts: dict[str, int],
    subgroups_by_subclass: dict[str, list[str]],
    grouping: str,
    figures_dir: Path,
    ncols: int,
    model: str = "bayesian",
) -> None:
    ncols = min(len(targets), ncols)
    nrows = int(np.ceil(len(targets) / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(4.0 * ncols, 3.8 * nrows), squeeze=False
    )
    metric_rows = summary.set_index("cell_type")

    for ax, cell_type in zip(axes.flat, targets):
        panel = table[table["cell_type"] == cell_type].copy()
        x = panel["whole_activity"].to_numpy(dtype=float)
        y = panel["mean_subgroup_activity"].to_numpy(dtype=float)
        yerr = panel["subgroup_sd"].to_numpy(dtype=float)
        valid = np.isfinite(x) & np.isfinite(y)
        x, y, yerr = x[valid], y[valid], yerr[valid]
        if x.size:
            finite_error = np.isfinite(yerr)
            error_low = y[finite_error] - yerr[finite_error]
            error_high = y[finite_error] + yerr[finite_error]
            lo = float(min(x.min(), y.min()))
            hi = float(max(x.max(), y.max()))
            if finite_error.any():
                lo = min(lo, float(error_low.min()))
                hi = max(hi, float(error_high.max()))
            pad = max(0.08 * (hi - lo), 0.08)
            lo, hi = lo - pad, hi + pad
            ax.plot([lo, hi], [lo, hi], color="#555555", ls="--", lw=1.1, zorder=1)
            if finite_error.any():
                ax.vlines(
                    x[finite_error],
                    error_low,
                    error_high,
                    color="#4c78a8",
                    linewidth=0.35,
                    alpha=0.18,
                    rasterized=True,
                    zorder=2,
                )
            ax.scatter(
                x,
                y,
                color="#4c78a8",
                s=10,
                alpha=0.6,
                linewidth=0,
                rasterized=True,
                zorder=3,
            )
            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)
        metrics = metric_rows.loc[cell_type]
        ccc = metrics["concordance_correlation"]
        mae = metrics["mae"]
        metric_text = f"CCC = {ccc:.2f}\nMAE = {mae:.2f}"
        if grouping == "supertype":
            median_sd = metrics["median_supertype_sd"]
            metric_text += (
                f"\nMedian supertype SD = {median_sd:.2f}"
                if np.isfinite(median_sd)
                else "\nMedian supertype SD = unavailable"
            )
        ax.text(
            0.04,
            0.96,
            metric_text,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
        )
        ax.set_title(
            (
                f"{cell_type}\n(n = {cell_counts[cell_type]:,} cells)"
                if grouping == "random"
                else (
                    f"{cell_type}\n(n = {cell_counts[cell_type]:,} cells; "
                    f"k = {len(subgroups_by_subclass[cell_type])} "
                    f"{'supertype' if len(subgroups_by_subclass[cell_type]) == 1 else 'supertypes'})"
                )
            ),
            fontsize=10,
        )
        ax.set_xlabel("Whole cell-type activity")
        ax.set_ylabel(
            "Mean activity across 5 subsets"
            if grouping == "random"
            else "Cell-count-weighted mean activity\nacross annotated supertypes"
        )
        if grouping == "supertype" and len(subgroups_by_subclass[cell_type]) == 1:
            ax.text(
                0.04,
                0.04,
                "One annotated supertype; SD unavailable",
                transform=ax.transAxes,
                ha="left",
                va="bottom",
                fontsize=8,
                color="#555555",
            )
        ax.set_aspect("equal", adjustable="box")

    for ax in axes.flat[len(targets) :]:
        ax.set_visible(False)

    fig.suptitle(
        (
            f"{model.capitalize()} cCRE activity: mean of 5 random cell subsets "
            "vs whole cell type"
            if grouping == "random"
            else (
                f"{model.capitalize()} cCRE activity: cell-count-weighted mean of annotated "
                "supertypes vs whole cell type"
            )
        ),
        fontsize=14,
    )
    if grouping == "supertype":
        fig.text(
            0.5,
            0.01,
            "Points: cell-count-weighted mean; vertical bars: ±1 unweighted SD "
            "across annotated supertypes (between-supertype heterogeneity, not uncertainty).",
            ha="center",
            va="bottom",
            fontsize=9,
            color="#444444",
        )
    fig.tight_layout(rect=(0, 0.04 if grouping == "supertype" else 0, 1, 0.96))
    save_figure(fig, figures_dir / artifact_names(grouping, model)["figure"])


def plot_pairwise_supertype_ccc(
    pairwise: pd.DataFrame,
    summary: pd.DataFrame,
    whole_agreement_summary: pd.DataFrame,
    targets: list[str],
    figures_dir: Path,
    model: str = "bayesian",
) -> None:
    """Plot distributions of within-parent pairwise supertype CCC values."""
    fig, ax = plt.subplots(figsize=(14, 7.2))
    summary_by_cell_type = summary.set_index("cell_type")
    whole_by_cell_type = whole_agreement_summary.set_index("cell_type")
    box_data = []
    box_positions = []
    rng = np.random.default_rng(20260811)

    for position, cell_type in enumerate(targets, start=1):
        values = pairwise.loc[
            pairwise["cell_type"] == cell_type, "concordance_correlation"
        ].dropna().to_numpy(dtype=float)
        if values.size:
            box_data.append(values)
            box_positions.append(position)
            jitter = rng.uniform(-0.16, 0.16, size=values.size)
            ax.scatter(
                position + jitter,
                values,
                s=24,
                color="#e45756",
                alpha=0.65,
                linewidth=0,
                zorder=3,
            )
            median = summary_by_cell_type.loc[cell_type, "median_pairwise_ccc"]
            ax.text(
                position,
                min(1.02, float(values.max()) + 0.06),
                f"median {median:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
                color="#7a2e2d",
            )
        else:
            ax.text(
                position,
                0.0,
                "one supertype\n(no pairs)",
                ha="center",
                va="center",
                fontsize=9,
                color="#666666",
            )

    if box_data:
        boxplot = ax.boxplot(
            box_data,
            positions=box_positions,
            widths=0.48,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "#8b1a1a", "linewidth": 1.8},
            whiskerprops={"color": "#555555", "linewidth": 1.0},
            capprops={"color": "#555555", "linewidth": 1.0},
            boxprops={"facecolor": "#f2b8b5", "edgecolor": "#555555", "alpha": 0.65},
            zorder=2,
        )
        for artist in boxplot["boxes"]:
            artist.set_zorder(2)

    for position, cell_type in enumerate(targets, start=1):
        if cell_type not in whole_by_cell_type.index:
            raise KeyError(f"missing mean-vs-whole CCC baseline for {cell_type}")
        baseline = float(
            whole_by_cell_type.loc[cell_type, "concordance_correlation"]
        )
        if not np.isfinite(baseline):
            continue
        ax.scatter(
            position,
            baseline,
            marker="D",
            s=58,
            facecolor="white",
            edgecolor="#1f77b4",
            linewidth=1.8,
            zorder=5,
        )
        ax.text(
            position,
            min(1.055, baseline + 0.025),
            f"{baseline:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
            color="#1f77b4",
            fontweight="bold",
        )

    labels = []
    for cell_type in targets:
        row = summary_by_cell_type.loc[cell_type]
        labels.append(f"{cell_type}\n({int(row['n_pairs'])} pairs)")
    ax.set_xticks(np.arange(1, len(targets) + 1), labels, rotation=32, ha="right")
    ax.axhline(1.0, color="#555555", ls="--", lw=1.0, zorder=1)
    ax.set_xlim(0.4, len(targets) + 0.6)
    finite_ccc = pairwise["concordance_correlation"].dropna().to_numpy(dtype=float)
    lower_limit = (
        max(-1.08, min(-0.08, np.floor((finite_ccc.min() - 0.05) * 10) / 10))
        if finite_ccc.size
        else -0.08
    )
    ax.set_ylim(lower_limit, 1.08)
    ax.set_ylabel("Lin concordance correlation (CCC)\nacross fitted cCREs")
    ax.set_title(
        f"{model.capitalize()} activity heterogeneity among annotated supertypes "
        "within each cell type",
        fontsize=14,
    )
    ax.grid(axis="y", color="#dddddd", linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    fig.text(
        0.5,
        0.012,
        "Red points/boxes: pairwise CCC among annotated supertypes. Blue diamonds: "
        "CCC of the cell-count-weighted supertype mean versus the intact whole cell type. "
        "Small groups may be noisier.",
        ha="center",
        va="bottom",
        fontsize=9,
        color="#444444",
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))
    save_figure(
        fig,
        figures_dir / artifact_names("supertype", model)["pairwise_figure"],
    )


def plot_pairwise_ccc_vs_minimum_cells(
    pairwise: pd.DataFrame,
    targets: list[str],
    figures_dir: Path,
    model: str = "bayesian",
) -> None:
    """Plot pairwise supertype CCC against the smaller group's cell count."""
    valid = pairwise[
        np.isfinite(pairwise["concordance_correlation"])
        & np.isfinite(pairwise["minimum_pair_cells"])
        & (pairwise["minimum_pair_cells"] > 0)
    ].copy()
    if valid.empty:
        raise ValueError("no finite supertype pairs available for cell-support plot")

    fig, ax = plt.subplots(figsize=(12.5, 7.2))
    palette = {target: plt.cm.tab10(i % 10) for i, target in enumerate(targets)}
    for cell_type in targets:
        frame = valid[valid["cell_type"] == cell_type]
        if frame.empty:
            continue
        ax.scatter(
            frame["minimum_pair_cells"],
            frame["concordance_correlation"],
            s=36,
            alpha=0.72,
            color=palette[cell_type],
            linewidth=0,
            label=cell_type,
            rasterized=True,
        )

    x = valid["minimum_pair_cells"].to_numpy(dtype=float)
    y = valid["concordance_correlation"].to_numpy(dtype=float)
    log_x = np.log10(x)
    rho, p_value = spearmanr(log_x, y)
    if np.unique(log_x).size > 1:
        slope, intercept = np.polyfit(log_x, y, deg=1)
        x_guide = np.geomspace(x.min(), x.max(), 200)
        ax.plot(
            x_guide,
            intercept + slope * np.log10(x_guide),
            color="#222222",
            lw=1.8,
            ls="--",
            label="Linear guide on log cell count",
            zorder=4,
        )

    ax.text(
        0.03,
        0.97,
        f"Spearman ρ = {rho:.2f}\np = {p_value:.2g}\nn = {len(valid)} pairs",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=11,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.85},
    )
    ax.set_xscale("log")
    ax.set_xlabel("Minimum cell count in the supertype pair (log scale)")
    ax.set_ylabel("Pairwise Lin concordance correlation (CCC)\nacross fitted cCREs")
    ax.set_ylim(min(-0.05, float(y.min()) - 0.04), min(1.02, float(y.max()) + 0.08))
    ax.set_title(
        f"{model.capitalize()} pairwise supertype agreement increases with cell support",
        fontsize=14,
    )
    ax.grid(color="#dddddd", linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    ax.legend(
        frameon=False,
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        fontsize=9,
        title="Parent cell type",
    )
    fig.text(
        0.43,
        0.012,
        "Each point is one within-parent annotated-supertype pair; Endo NN has no pair. "
        "The dashed line is a visual guide, not a fitted biological model.",
        ha="center",
        va="bottom",
        fontsize=9,
        color="#444444",
    )
    fig.tight_layout(rect=(0, 0.05, 0.82, 0.96))
    save_figure(
        fig,
        figures_dir
        / artifact_names("supertype", model)["pairwise_support_figure"],
    )


def main() -> None:
    args = parse_args()
    if args.ncols <= 0:
        raise ValueError("--ncols must be positive")

    tables = args.outdir / "tables"
    figures = args.outdir / "figures"
    raw = args.outdir / "raw"
    for d in (tables, figures, raw):
        d.mkdir(parents=True, exist_ok=True)

    if args.model == "bayesian":
        combined_dir = args.combined_bayes_dir
        split_dir = args.split_bayes_dir
        loader = load_bayesian
    else:
        combined_dir = args.combined_bootstrap_dir
        split_dir = args.split_bootstrap_dir
        loader = load_bootstrap

    targets, subgroups_by_subclass, grouping = split_targets(split_dir)
    names = artifact_names(grouping, args.model)
    self_cre = args.calibration == "self_cre_negctrl"
    log(
        f"[het] {len(targets)} split subclasses, grouping={grouping}, "
        f"{sum(map(len, subgroups_by_subclass.values()))} total groups, "
        f"model={args.model}, calibration={args.calibration}"
    )

    combined = loader(combined_dir, self_cre=self_cre)
    split = loader(split_dir, self_cre=self_cre)

    neg_set = negative_controls(split_dir) | negative_controls(combined_dir)
    black = blacklist(split_dir) | blacklist(combined_dir)
    # Blacklisted cCREs were not fitted; retain the fitted negative controls so
    # each cell-type panel shows every cCRE in the model.
    excluded = black
    assignment = pd.read_csv(split_dir / "cell_group_assignment.csv")
    subgroup_cell_counts = (
        assignment.groupby(["original_subclass", "new_subclass"], sort=False)
        .size()
        .astype(int)
        .rename("n_cells")
        .reset_index()
    )
    subgroup_weights = {
        cell_type: {
            str(row.new_subclass): int(row.n_cells)
            for row in subgroup_cell_counts[
                subgroup_cell_counts["original_subclass"] == cell_type
            ].itertuples(index=False)
        }
        for cell_type in targets
    }
    agreement = bayesian_subset_agreement(
        combined,
        split,
        targets,
        subgroups_by_subclass,
        excluded,
        subgroup_weights=subgroup_weights if grouping == "supertype" else None,
    )
    summary = summarize_agreement(
        agreement, include_supertype_heterogeneity=grouping == "supertype"
    )
    cell_counts = (
        assignment.groupby("original_subclass", sort=False)
        .size()
        .astype(int)
        .to_dict()
    )
    missing_counts = [cell_type for cell_type in targets if cell_type not in cell_counts]
    if missing_counts:
        raise KeyError(f"missing cell counts for split cell types: {missing_counts}")
    panel_targets = sorted(
        targets, key=lambda cell_type: cell_counts[cell_type], reverse=True
    )

    combined.to_csv(raw / f"combined_activity_{args.model}.csv")
    split.to_csv(raw / names["split_raw"])
    agreement_for_export(agreement, grouping).to_csv(
        tables / names["table"], index=False
    )
    summary.to_csv(tables / names["summary"], index=False)
    plot_bayesian_subset_agreement(
        agreement,
        summary,
        panel_targets,
        cell_counts,
        subgroups_by_subclass,
        grouping,
        figures,
        args.ncols,
        model=args.model,
    )
    pairwise = None
    pairwise_summary = None
    pairwise_support_association = None
    if grouping == "supertype":
        pairwise = pairwise_supertype_agreement(
            split,
            targets,
            subgroups_by_subclass,
            subgroup_weights,
            excluded,
        )
        pairwise_summary = summarize_pairwise_supertype_agreement(
            pairwise, targets, subgroups_by_subclass
        )
        valid_support = (
            pairwise[["minimum_pair_cells", "concordance_correlation"]]
            .dropna()
            .query("minimum_pair_cells > 0")
        )
        support_rho, support_p = spearmanr(
            np.log10(valid_support["minimum_pair_cells"].to_numpy(dtype=float)),
            valid_support["concordance_correlation"].to_numpy(dtype=float),
        )
        pairwise_support_association = {
            "predictor": "log10 minimum cell count of the two supertypes",
            "outcome": "pairwise Lin concordance correlation coefficient",
            "spearman_rho": float(support_rho),
            "p_value": float(support_p),
            "n_pairs": int(len(valid_support)),
            "interpretation": (
                "pairwise CCC depends on cell support; low-support pairs may mix "
                "biological heterogeneity with estimation noise"
            ),
        }
        pairwise.to_csv(tables / names["pairwise_table"], index=False)
        pairwise_summary.to_csv(
            tables / names["pairwise_summary"], index=False
        )
        plot_pairwise_supertype_ccc(
            pairwise,
            pairwise_summary,
            summary,
            panel_targets,
            figures,
            model=args.model,
        )
        plot_pairwise_ccc_vs_minimum_cells(
            pairwise,
            panel_targets,
            figures,
            model=args.model,
        )

    n_subgroups_by_subclass = {
        target: len(subgroups_by_subclass[target]) for target in targets
    }
    subgroup_noun = "random subset" if grouping == "random" else "annotated supertype"

    write_json(
        tables / names["manifest"],
        {
            "split_subclasses": targets,
            "panel_cell_types": panel_targets,
            "panel_order": "descending original cell count",
            "n_groups": (
                next(iter(n_subgroups_by_subclass.values()))
                if len(set(n_subgroups_by_subclass.values())) == 1
                else None
            ),
            "grouping": grouping,
            "subgroup_obs_column": (
                None if grouping == "random" else "supertype_name"
            ),
            "subgroups_by_subclass": subgroups_by_subclass,
            "n_subgroups_by_subclass": n_subgroups_by_subclass,
            "subgroup_cell_counts": {
                row.new_subclass: int(row.n_cells)
                for row in subgroup_cell_counts.itertuples(index=False)
            },
            "calibration": args.calibration,
            "models": [args.model],
            "combined_run_dir": str(combined_dir),
            "split_run_dir": str(split_dir),
            "combined_bayes_dir": (
                str(combined_dir) if args.model == "bayesian" else None
            ),
            "split_bayes_dir": (
                str(split_dir) if args.model == "bayesian" else None
            ),
            "combined_bootstrap_dir": (
                str(combined_dir) if args.model == "bootstrap" else None
            ),
            "split_bootstrap_dir": (
                str(split_dir) if args.model == "bootstrap" else None
            ),
            "excluded_cres": sorted(excluded),
            "panel_unit": "cell type",
            "point_unit": "cCRE",
            "cell_counts": cell_counts,
            "n_cres_per_panel": int(agreement["cre"].nunique()),
            "negative_controls_included": sorted(neg_set),
            "x_metric": "negative-control-centered activity in intact cell type",
            "y_metric": (
                "cell-count-weighted mean negative-control-centered activity "
                "across annotated supertypes"
                if grouping == "supertype"
                else (
                    "unweighted mean negative-control-centered activity across "
                    f"{subgroup_noun}s"
                )
            ),
            "aggregation": (
                "cell-count-weighted" if grouping == "supertype" else "unweighted"
            ),
            "secondary_aggregation": (
                "unweighted mean across annotated supertypes"
                if grouping == "supertype"
                else None
            ),
            "error_bar": (
                "unweighted sample standard deviation across annotated supertypes"
                if grouping == "supertype"
                else f"sample standard deviation across {subgroup_noun}s"
            ),
            "error_bar_interpretation": (
                "between-supertype heterogeneity, not posterior or fit uncertainty"
                if grouping == "supertype"
                else "between-random-subset dispersion"
            ),
            "pairwise_supertype_analysis": (
                {
                    "table": names["pairwise_table"],
                    "summary": names["pairwise_summary"],
                    "figure": f"{names['pairwise_figure']}.pdf",
                    "cell_support_figure": (
                        f"{names['pairwise_support_figure']}.pdf"
                    ),
                    "metric": "Lin concordance correlation coefficient",
                    "comparison_unit": (
                        "each unordered pair of annotated supertypes within a parent cell type"
                    ),
                    "feature_axis": "all fitted non-blacklisted cCREs",
                    "pair_weighting": "each supertype pair contributes once",
                    "baseline_marker": (
                        "per-cell-type CCC of the cell-count-weighted supertype "
                        "mean versus the intact whole-cell-type activity"
                    ),
                    "baseline_source": names["summary"],
                    "n_pairs_total": int(len(pairwise)),
                    "n_pairs_by_subclass": {
                        str(row.cell_type): int(row.n_pairs)
                        for row in pairwise_summary.itertuples(index=False)
                    },
                    "cell_support_association": pairwise_support_association,
                }
                if grouping == "supertype"
                else None
            ),
            "agreement_metrics": [
                "Lin concordance correlation",
                "mean absolute error",
                "root mean squared error",
                "mean error",
                "Pearson correlation",
            ],
        },
    )
    overall = summary.loc[summary["cell_type"] == "ALL"].iloc[0]
    log(
        f"[het] {len(targets)} cell-type panels, "
        f"{agreement['cre'].nunique()} cCREs per panel; overall "
        f"CCC={overall['concordance_correlation']:.3f}, "
        f"MAE={overall['mae']:.3f}"
    )
    log(
        f"[het] wrote {grouping} {args.model} agreement table to {tables}, "
        f"figure to {figures}"
    )
    if grouping == "supertype":
        log(
            f"[het] wrote {len(pairwise)} within-parent supertype-pair CCCs "
            f"and distribution figure; CCC vs minimum pair cells "
            f"Spearman rho={pairwise_support_association['spearman_rho']:.3f}"
        )


if __name__ == "__main__":
    main()
