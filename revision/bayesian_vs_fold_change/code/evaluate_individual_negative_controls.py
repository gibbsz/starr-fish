#!/usr/bin/env python3
"""Evaluate ordinary-fit negative controls against an all-control pooled reference."""

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

from analysis_utils import ANALYSIS_DIR, DEFAULT_H5AD, log, write_json
from compute_t7_filter_negative_control_stats import aligned_t7_totals, bh_fdr


CENTERING_LABELS = {
    "none": "Total activity",
    "posterior-alpha": "Alpha-subtracted",
}
COLORS = {
    "Total activity": "#2878B5",
    "Alpha-subtracted": "#D95F02",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ordinary-root",
        type=Path,
        required=True,
        help="Joint+dropout fit in which annotated controls are ordinary cCREs.",
    )
    parser.add_argument(
        "--pooled-root",
        type=Path,
        required=True,
        help="Existing joint+dropout fit pooling all annotated negative controls.",
    )
    parser.add_argument("--h5ad", type=Path, default=DEFAULT_H5AD)
    parser.add_argument("--t7-threshold", type=float, default=50.0)
    parser.add_argument("--q-cutoff", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=1731)
    parser.add_argument("--draw-chunk-cres", type=int, default=32)
    parser.add_argument(
        "--centerings",
        nargs="+",
        choices=list(CENTERING_LABELS),
        default=["none"],
        help="Activity versions to evaluate; the initial experiment uses total activity only.",
    )
    parser.add_argument(
        "--tables-dir", type=Path, default=ANALYSIS_DIR / "results" / "tables"
    )
    parser.add_argument(
        "--figures-dir", type=Path, default=ANALYSIS_DIR / "results" / "figures"
    )
    parser.add_argument(
        "--stem", default="joint_dropout_individual_negative_control_evaluation"
    )
    return parser.parse_args()


def posterior_path(root: Path) -> tuple[Path, dict]:
    manifest = json.loads((root / "run_manifest.json").read_text())
    return root / f"{manifest['tag']}_posterior_samples.npz", manifest


def aligned_group_indices(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    lookup = {str(name): idx for idx, name in enumerate(source.astype(str))}
    missing = [str(name) for name in target.astype(str) if str(name) not in lookup]
    if missing:
        raise ValueError(f"pooled posterior is missing groups: {missing[:5]}")
    return np.asarray([lookup[str(name)] for name in target.astype(str)], dtype=int)


def contrast_summary(
    log_gamma: np.ndarray,
    alpha: np.ndarray,
    pooled_log_gamma: np.ndarray,
    pooled_alpha: np.ndarray,
    *,
    centering: str,
    ordinary_draw_index: np.ndarray,
    pooled_draw_index: np.ndarray,
    chunk_cres: int,
) -> tuple[np.ndarray, np.ndarray]:
    n_group, n_cre = log_gamma.shape[1:]
    effects = np.full((n_group, n_cre), np.nan, dtype=np.float64)
    pvalues = np.full((n_group, n_cre), np.nan, dtype=np.float64)
    reference = pooled_log_gamma[pooled_draw_index]
    if centering == "posterior-alpha":
        reference = reference - pooled_alpha[pooled_draw_index, None]

    for start in range(0, n_cre, chunk_cres):
        stop = min(start + chunk_cres, n_cre)
        target = log_gamma[ordinary_draw_index, :, start:stop]
        if centering == "posterior-alpha":
            target = target - alpha[ordinary_draw_index, None, start:stop]
        difference = target - reference[:, :, None]
        effects[:, start:stop] = difference.mean(axis=0, dtype=np.float64)
        pvalues[:, start:stop] = (difference <= 0).mean(axis=0, dtype=np.float64)
    return effects, pvalues


def build_test_table(args: argparse.Namespace) -> tuple[pd.DataFrame, dict]:
    ordinary_path, ordinary_manifest = posterior_path(args.ordinary_root)
    pooled_path, pooled_manifest = posterior_path(args.pooled_root)
    ordinary_mode = ordinary_manifest.get("negative_control_mode")
    if ordinary_mode not in {"ordinary", "ordinary-and-pooled"}:
        raise ValueError(
            f"{args.ordinary_root} does not contain ordinary negative-control fits"
        )
    same_fit = ordinary_path.resolve() == pooled_path.resolve()
    if same_fit and ordinary_mode != "ordinary-and-pooled":
        raise ValueError(
            "a shared ordinary/pooled posterior requires ordinary-and-pooled mode"
        )

    with np.load(ordinary_path, allow_pickle=True) as posterior:
        required = {"log_gamma", "alpha", "group_names", "cre_names"}
        missing = required.difference(posterior.files)
        if missing:
            raise ValueError(f"{ordinary_path} is missing sites: {sorted(missing)}")
        log_gamma = posterior["log_gamma"].astype(np.float32)
        alpha = posterior["alpha"].astype(np.float32)
        groups = posterior["group_names"].astype(str)
        cres = posterior["cre_names"].astype(str)

    with np.load(pooled_path, allow_pickle=True) as posterior:
        required = {"log_gamma_neg", "alpha_neg", "group_names"}
        missing = required.difference(posterior.files)
        if missing:
            raise ValueError(f"{pooled_path} is missing sites: {sorted(missing)}")
        pooled_log_gamma = posterior["log_gamma_neg"].astype(np.float32)
        pooled_alpha = posterior["alpha_neg"].astype(np.float32)
        pooled_groups = posterior["group_names"].astype(str)

    if alpha.shape != (log_gamma.shape[0], log_gamma.shape[2]):
        raise ValueError("ordinary alpha axis does not match log_gamma")
    if pooled_alpha.shape != (pooled_log_gamma.shape[0],):
        raise ValueError("pooled alpha_neg axis does not match log_gamma_neg")
    pooled_log_gamma = pooled_log_gamma[
        :, aligned_group_indices(pooled_groups, groups)
    ]

    negative_controls = pd.read_csv(
        args.pooled_root / "negative_controls.csv"
    ).iloc[:, 0].astype(str).tolist()
    missing_controls = sorted(set(negative_controls) - set(cres))
    if missing_controls:
        raise ValueError(
            f"ordinary posterior is missing negative controls: {missing_controls}"
        )

    n_mc = min(log_gamma.shape[0], pooled_log_gamma.shape[0])
    if same_fit:
        if log_gamma.shape[0] != pooled_log_gamma.shape[0]:
            raise ValueError("same-fit ordinary and pooled posterior draw axes differ")
        ordinary_draw_index = np.arange(n_mc)
        pooled_draw_index = np.arange(n_mc)
        draw_pairing = "matched ordinary and pooled posterior draws from one joint fit"
    else:
        rng = np.random.default_rng(args.seed)
        ordinary_draw_index = rng.permutation(log_gamma.shape[0])[:n_mc]
        pooled_draw_index = rng.permutation(pooled_log_gamma.shape[0])[:n_mc]
        draw_pairing = (
            "independent random permutations of ordinary-fit and pooled-fit "
            "posterior draws"
        )

    group_index = pd.Index(groups, dtype=str)
    cre_index = pd.Index(cres, dtype=str)
    pair_t7 = aligned_t7_totals(args.h5ad, group_index, cre_index)
    pooled_t7 = pair_t7.reindex(columns=negative_controls, fill_value=0.0).sum(axis=1)
    eligible = pair_t7.ge(args.t7_threshold).to_numpy(bool)
    eligible &= pooled_t7.ge(args.t7_threshold).to_numpy(bool)[:, None]
    is_negative = cre_index.isin(negative_controls)

    group_grid, cre_grid = np.meshgrid(groups, cres, indexing="ij")
    frames = []
    for centering in args.centerings:
        effects, pvalues = contrast_summary(
            log_gamma,
            alpha,
            pooled_log_gamma,
            pooled_alpha,
            centering=centering,
            ordinary_draw_index=ordinary_draw_index,
            pooled_draw_index=pooled_draw_index,
            chunk_cres=args.draw_chunk_cres,
        )
        valid = eligible & np.isfinite(effects) & np.isfinite(pvalues)
        qvalues = bh_fdr(np.where(valid, pvalues, np.nan).ravel()).reshape(
            pvalues.shape
        )
        frame = pd.DataFrame(
            {
                "centering": CENTERING_LABELS[centering],
                "group": group_grid.ravel(),
                "cre": cre_grid.ravel(),
                "effect_vs_pooled_negative": effects.ravel(),
                "p_right": pvalues.ravel(),
                "q_right": qvalues.ravel(),
                "significant_q": (qvalues <= args.q_cutoff).ravel(),
                "target_t7_total": pair_t7.to_numpy(float).ravel(),
                "pooled_negative_t7_total": np.repeat(
                    pooled_t7.to_numpy(float), len(cres)
                ),
                "is_negative_control": np.tile(is_negative, len(groups)),
            }
        )
        frames.append(frame.loc[valid.ravel()].copy())

    tests = pd.concat(frames, ignore_index=True)
    metadata = {
        "ordinary_root": str(args.ordinary_root),
        "ordinary_posterior": str(ordinary_path),
        "pooled_root": str(args.pooled_root),
        "pooled_posterior": str(pooled_path),
        "negative_controls": negative_controls,
        "n_monte_carlo_draw_pairs": int(n_mc),
        "same_joint_fit": same_fit,
        "draw_pairing": draw_pairing,
    }
    return tests, metadata


def summaries(
    tests: pd.DataFrame, q_cutoff: float
) -> tuple[pd.DataFrame, pd.DataFrame]:
    controls = tests[tests["is_negative_control"]].copy()
    overall = (
        controls.groupby("centering", sort=False)
        .agg(
            tested_pairs=("q_right", "size"),
            significant_pairs=("significant_q", "sum"),
            tested_cell_types=("group", "nunique"),
            tested_controls=("cre", "nunique"),
            median_effect=("effect_vs_pooled_negative", "median"),
        )
        .reset_index()
    )
    overall["significant_proportion"] = (
        overall["significant_pairs"] / overall["tested_pairs"]
    )
    overall["q_cutoff"] = q_cutoff

    by_control = (
        controls.groupby(["centering", "cre"], sort=False)
        .agg(
            tested_pairs=("q_right", "size"),
            significant_pairs=("significant_q", "sum"),
            median_effect=("effect_vs_pooled_negative", "median"),
            median_t7=("target_t7_total", "median"),
        )
        .reset_index()
    )
    by_control["significant_proportion"] = (
        by_control["significant_pairs"] / by_control["tested_pairs"]
    )
    by_control["q_cutoff"] = q_cutoff
    return overall, by_control


def plot_diagnostics(
    tests: pd.DataFrame,
    by_control: pd.DataFrame,
    output: Path,
    q_cutoff: float,
    t7_threshold: float,
) -> None:
    sns.set_theme(context="paper", style="whitegrid")
    controls = tests[tests["is_negative_control"]].copy()
    control_order = sorted(controls["cre"].unique())
    centerings = list(dict.fromkeys(controls["centering"].astype(str)))
    palette = {centering: COLORS[centering] for centering in centerings}
    fig, axes = plt.subplots(2, 1, figsize=(10.5, 8.2), constrained_layout=True)

    ax = axes[0]
    sns.barplot(
        data=by_control,
        x="cre",
        y="significant_proportion",
        hue="centering",
        order=control_order,
        hue_order=centerings,
        palette=palette,
        errorbar=None,
        ax=ax,
    )
    lookup = {
        (str(row.centering), str(row.cre)): (
            f"{int(row.significant_pairs)}/{int(row.tested_pairs)}"
        )
        for row in by_control.itertuples(index=False)
    }
    for container, centering in zip(ax.containers, centerings):
        labels = [lookup.get((centering, cre), "") for cre in control_order]
        ax.bar_label(container, labels=labels, rotation=90, padding=2, fontsize=6)
    ax.set_xlabel("")
    ax.set_ylabel("Proportion significant")
    ax.set_title("Ground-truth-negative calls against pooled all-seven reference")
    ax.legend(title="Activity", frameon=False)

    ax = axes[1]
    sns.boxplot(
        data=controls,
        x="cre",
        y="effect_vs_pooled_negative",
        hue="centering",
        order=control_order,
        hue_order=centerings,
        palette=palette,
        showfliers=False,
        linewidth=0.8,
        ax=ax,
    )
    ax.axhline(0.0, color="black", linestyle="--", linewidth=0.9)
    ax.set_xlabel("Negative-control cCRE fitted as an ordinary cCRE")
    ax.set_ylabel("Posterior mean effect vs pooled reference")
    ax.set_title("Eligible subclass effects")
    ax.get_legend().remove()

    fig.suptitle(
        f"Individual negative-control evaluation; T7 >= {t7_threshold:g}; "
        f"BH q <= {q_cutoff:g}",
        fontsize=12,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.tables_dir.mkdir(parents=True, exist_ok=True)
    args.figures_dir.mkdir(parents=True, exist_ok=True)
    log("[individual negative controls] comparing ordinary and pooled posteriors")
    tests, metadata = build_test_table(args)
    overall, by_control = summaries(tests, args.q_cutoff)

    tests_path = args.tables_dir / f"{args.stem}_tests.csv.gz"
    controls_path = args.tables_dir / f"{args.stem}_negative_control_tests.csv"
    overall_path = args.tables_dir / f"{args.stem}_summary.csv"
    by_control_path = args.tables_dir / f"{args.stem}_by_control.csv"
    figure_path = args.figures_dir / f"{args.stem}.pdf"
    tests.to_csv(tests_path, index=False)
    tests[tests["is_negative_control"]].to_csv(controls_path, index=False)
    overall.to_csv(overall_path, index=False)
    by_control.to_csv(by_control_path, index=False)
    plot_diagnostics(
        tests,
        by_control,
        figure_path,
        args.q_cutoff,
        args.t7_threshold,
    )

    write_json(
        args.figures_dir / f"{args.stem}_manifest.json",
        {
            **metadata,
            "test": (
                "ordinary cCRE posterior activity compared with the existing "
                "shared all-seven negative-control posterior"
            ),
            "self_inclusion": (
                "each evaluated negative control also contributed to the pooled "
                "all-seven reference; leave-one-out evaluation deferred"
            ),
            "t7_filter": (
                f"ordinary cCRE-subclass T7 >= {args.t7_threshold:g} and pooled "
                f"all-seven negative-control T7 >= {args.t7_threshold:g}"
            ),
            "fdr": (
                "Benjamini-Hochberg separately for each requested activity "
                "centering over all eligible ordinary cCRE-subclass pairs"
            ),
            "activity_centerings": args.centerings,
            "q_cutoff": args.q_cutoff,
            "outputs": {
                "all_tests": str(tests_path),
                "negative_control_tests": str(controls_path),
                "summary": str(overall_path),
                "by_control": str(by_control_path),
                "figure": str(figure_path),
            },
        },
    )
    log(
        f"[individual negative controls] wrote {len(tests):,} eligible tests "
        f"and {len(tests[tests['is_negative_control']]):,} control tests"
    )


if __name__ == "__main__":
    main()
