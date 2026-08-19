#!/usr/bin/env python3
"""Compare decoupled Bayesian dropout-prior sensitivity runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from analysis_utils import (
    DEFAULT_H5AD,
    OLD_DATA_BOOTSTRAP,
    ablation_root,
    log,
    write_json,
)
from plot_method_activity_heatmap import (
    blacklisted_cres,
    combined_axes,
    read_cre_blacklist,
    t7_pair_totals,
    trim_empty_axes,
)
from plot_section_reproducibility import bayesian_base, bootstrap_base


DEFAULT_EXPERIMENTS = {
    "default_beta_1_9": ablation_root("bayesian_decoupled_dropout"),
    "moderate_beta_2_5": ablation_root("bayesian_decoupled_dropout_moderate"),
    "high_beta_5_5": ablation_root("bayesian_decoupled_dropout_high"),
    "strongly_high_beta_8_2": ablation_root("bayesian_decoupled_dropout_strongly_high"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bootstrap-dir", type=Path, default=OLD_DATA_BOOTSTRAP
    )
    parser.add_argument("--h5ad", type=Path, default=DEFAULT_H5AD)
    parser.add_argument(
        "--outdir",
        type=Path,
        default=ablation_root("dropout_prior_comparison"),
    )
    parser.add_argument("--bootstrap-log-chunk-size", type=int, default=250)
    parser.add_argument("--t7-threshold", type=float, default=100.0)
    parser.add_argument(
        "--experiment",
        action="append",
        default=[],
        metavar="LABEL=DIR",
        help="Extra or replacement experiment, e.g. moderate=results/dir.",
    )
    parser.add_argument(
        "--only-explicit-experiments",
        action="store_true",
        help="Use only --experiment values instead of the default four dirs.",
    )
    return parser.parse_args()


def discover_tag(root: Path) -> str:
    return json.loads((root / "run_manifest.json").read_text())["tag"]


def mean_log_beta_t7(root: Path) -> float:
    tag = discover_tag(root)
    with np.load(root / f"{tag}_scalar_samples.npz", allow_pickle=True) as samples:
        beta_t7 = np.asarray(samples["beta_t7"], dtype=float).reshape(-1)
    return float(np.log(beta_t7).mean())


def parse_experiments(args: argparse.Namespace) -> dict[str, Path]:
    experiments = {} if args.only_explicit_experiments else dict(DEFAULT_EXPERIMENTS)
    for item in args.experiment:
        if "=" not in item:
            raise ValueError("--experiment must be LABEL=DIR")
        label, path = item.split("=", 1)
        experiments[label] = Path(path)
    return experiments


def summarize_samples(root: Path) -> dict:
    tag = discover_tag(root)
    path = root / f"{tag}_scalar_samples.npz"
    row = {"tag": tag}
    with np.load(path, allow_pickle=True) as samples:
        for site in ("p_drop_t7", "p_drop_cre"):
            values = np.asarray(samples[site], dtype=float).reshape(-1)
            q025, median, q975 = np.percentile(values, [2.5, 50, 97.5])
            row.update(
                {
                    f"{site}_mean": float(values.mean()),
                    f"{site}_median": float(median),
                    f"{site}_q025": float(q025),
                    f"{site}_q975": float(q975),
                }
            )
    return row


def summarize_ppc(root: Path) -> dict:
    tag = discover_tag(root)
    ppc = json.loads((root / f"{tag}_ppc.json").read_text())
    row = {}
    for channel in ("t7", "cre"):
        obs = ppc[channel]["obs"]
        row[f"{channel}_obs_zero_fraction"] = float(obs["zero_fraction"])
        row[f"{channel}_rep_zero_fraction_mean"] = float(
            np.mean(ppc[channel]["rep_zero_fraction"])
        )
        row[f"{channel}_obs_mean_nonzero"] = float(obs["mean_nonzero"])
        row[f"{channel}_rep_mean_nonzero_mean"] = float(
            np.mean(ppc[channel]["rep_mean_nonzero"])
        )
    for key, value in ppc["joint"]["obs"].items():
        row[f"joint_obs_{key}"] = float(value)
    for key, value in ppc["joint"].items():
        if key == "obs":
            continue
        row[f"joint_{key}_mean"] = float(np.mean(value))
    return row


def load_gamma(root: Path) -> pd.DataFrame:
    tag = discover_tag(root)
    gamma = pd.read_csv(root / f"{tag}_gamma.csv")
    gamma["group"] = gamma["group"].astype(str)
    gamma["cre"] = gamma["cre"].astype(str)
    gamma["log_gamma_mean"] = np.log(gamma["gamma_mean"].astype(float))
    return gamma


def summarize_gamma(root: Path, baseline_gamma: pd.DataFrame | None) -> dict:
    gamma = load_gamma(root)
    row = {
        "n_gamma_pairs": int(len(gamma)),
        "prior_dominated_fraction": float(gamma["prior_dominated"].astype(bool).mean()),
        "log_gamma_mean": float(gamma["log_gamma_mean"].mean()),
        "log_gamma_median": float(gamma["log_gamma_mean"].median()),
        "log_gamma_q025": float(gamma["log_gamma_mean"].quantile(0.025)),
        "log_gamma_q975": float(gamma["log_gamma_mean"].quantile(0.975)),
    }
    if baseline_gamma is not None:
        merged = gamma.merge(
            baseline_gamma[["group", "cre", "log_gamma_mean"]],
            on=["group", "cre"],
            suffixes=("", "_baseline"),
        )
        delta = merged["log_gamma_mean"] - merged["log_gamma_mean_baseline"]
        row.update(
            {
                "delta_vs_default_mean": float(delta.mean()),
                "delta_vs_default_median": float(delta.median()),
                "delta_vs_default_q025": float(delta.quantile(0.025)),
                "delta_vs_default_q975": float(delta.quantile(0.975)),
                "log_gamma_corr_vs_default": float(
                    merged["log_gamma_mean"].corr(merged["log_gamma_mean_baseline"])
                ),
            }
        )
    return row


def summarize_manifest(root: Path) -> dict:
    manifest = json.loads((root / "run_manifest.json").read_text())
    config = manifest.get("config", {})
    dropout_prior = manifest.get("dropout_prior") or config.get("dropout_prior", {})
    return {
        "manifest_method_variant": manifest.get("method_variant"),
        "n_cells": manifest.get("n_cells"),
        "n_cres_fitted": manifest.get("n_cres_fitted"),
        "n_subclasses": manifest.get("n_subclasses"),
        "dropout_prior_label": dropout_prior.get("label"),
        "p_drop_t7_prior_mean": dropout_prior.get("p_drop_t7_mean"),
        "p_drop_cre_prior_mean": dropout_prior.get("p_drop_cre_mean"),
        "p_drop_t7_alpha": dropout_prior.get("p_drop_t7_alpha"),
        "p_drop_t7_beta": dropout_prior.get("p_drop_t7_beta"),
        "p_drop_cre_alpha": dropout_prior.get("p_drop_cre_alpha"),
        "p_drop_cre_beta": dropout_prior.get("p_drop_cre_beta"),
    }


def summarize_dirs(experiments: dict[str, Path]) -> pd.DataFrame:
    existing = {
        label: root
        for label, root in experiments.items()
        if (root / "run_manifest.json").exists()
    }
    missing = sorted(set(experiments) - set(existing))
    if missing:
        log(f"[dropout comparison] skipping missing runs: {', '.join(missing)}")
    baseline_root = existing.get("default_beta_1_9")
    baseline_gamma = load_gamma(baseline_root) if baseline_root else None
    rows = []
    for label, root in existing.items():
        row = {"label": label, "dir": str(root)}
        row.update(summarize_manifest(root))
        row.update(summarize_samples(root))
        row.update(summarize_ppc(root))
        row.update(
            summarize_gamma(
                root,
                None if label == "default_beta_1_9" else baseline_gamma,
            )
        )
        rows.append(row)
    return pd.DataFrame(rows)


def activity_matrix(root: Path, args: argparse.Namespace) -> pd.DataFrame:
    holder = SimpleNamespace(
        activity_calibration="none",
        bootstrap_log_chunk_size=args.bootstrap_log_chunk_size,
    )
    return bayesian_base(root, holder)[0] - mean_log_beta_t7(root)


def bootstrap_activity(args: argparse.Namespace) -> pd.DataFrame:
    holder = SimpleNamespace(
        activity_calibration="none",
        bootstrap_log_chunk_size=args.bootstrap_log_chunk_size,
    )
    return bootstrap_base(args.bootstrap_dir, holder)[0]


def correlation_summary(experiments: dict[str, Path], args: argparse.Namespace) -> pd.DataFrame:
    existing = {
        label: root
        for label, root in experiments.items()
        if (root / "run_manifest.json").exists()
    }
    if not existing:
        return pd.DataFrame()
    bootstrap = bootstrap_activity(args)
    bayes = {label: activity_matrix(root, args) for label, root in existing.items()}
    all_matrices = {"Bootstrap": bootstrap, **bayes}
    rows, columns = combined_axes(all_matrices)
    blacklist = set(read_cre_blacklist(args.bootstrap_dir))
    for root in existing.values():
        blacklist.update(read_cre_blacklist(root))
    columns = pd.Index([cre for cre in columns.astype(str) if cre not in blacklist], dtype=str)
    bootstrap = bootstrap.reindex(index=rows, columns=columns)
    bayes = {label: matrix.reindex(index=rows, columns=columns) for label, matrix in bayes.items()}
    pair_t7 = t7_pair_totals(args.h5ad, rows, columns)
    variants = {
        "complete": pd.DataFrame(True, index=rows, columns=columns),
        "t7_gt100": pair_t7.gt(args.t7_threshold),
    }
    records = []
    for variant, mask in variants.items():
        for label, matrix in bayes.items():
            trimmed, _, _, present = trim_empty_axes(
                {"Bootstrap": bootstrap.where(mask), label: matrix.where(mask)}
            )
            wide = pd.concat(
                [
                    trimmed["Bootstrap"].stack(dropna=False).rename("bootstrap"),
                    trimmed[label].stack(dropna=False).rename("bayesian"),
                ],
                axis=1,
            ).replace([np.inf, -np.inf], np.nan).dropna()
            records.append(
                {
                    "label": label,
                    "variant": variant,
                    "n_pairs": int(len(wide)),
                    "finite_pairs_any_method": int(present.to_numpy(bool).sum()),
                    "bayesian_activity_scale": "log_gamma - mean_log_beta_t7",
                    "pearson_bootstrap_vs_bayes": float(
                        wide["bootstrap"].corr(wide["bayesian"], method="pearson")
                    )
                    if len(wide) > 1
                    else np.nan,
                    "spearman_bootstrap_vs_bayes": float(
                        wide["bootstrap"].corr(wide["bayesian"], method="spearman")
                    )
                    if len(wide) > 1
                    else np.nan,
                }
            )
    return pd.DataFrame(records)


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    experiments = parse_experiments(args)
    summary = summarize_dirs(experiments)
    summary.to_csv(args.outdir / "dropout_prior_summary.csv", index=False)
    write_json(
        args.outdir / "dropout_prior_summary.json",
        json.loads(summary.to_json(orient="records")),
    )
    correlations = correlation_summary(experiments, args)
    correlations.to_csv(args.outdir / "dropout_prior_correlations.csv", index=False)
    write_json(
        args.outdir / "dropout_prior_correlations.json",
        json.loads(correlations.to_json(orient="records")),
    )
    log(
        "[dropout comparison] wrote summaries for "
        f"{len(summary)} available runs to {args.outdir}"
    )


if __name__ == "__main__":
    main()
