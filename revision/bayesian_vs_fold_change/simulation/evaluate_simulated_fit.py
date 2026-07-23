#!/usr/bin/env python3
"""Evaluate model estimates on simulated data against simulation truth."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(os.environ.get("TMPDIR", "/tmp")) / "matplotlib-cache"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(os.environ.get("TMPDIR", "/tmp")) / "xdg-cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CODE_DIR = Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(CODE_DIR))

from analysis_utils import write_json  # noqa: E402

SIM_DIR = Path(__file__).resolve().parent
DEFAULT_TRUTH_DIR = SIM_DIR / "results" / "joint_dropout_simulated"
DEFAULT_FIT_DIR = SIM_DIR / "results" / "joint_dropout_fit"
DEFAULT_OUTDIR = SIM_DIR / "results" / "joint_dropout_recovery"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--truth-dir", type=Path, default=DEFAULT_TRUTH_DIR)
    parser.add_argument("--fit-dir", type=Path, default=DEFAULT_FIT_DIR)
    parser.add_argument("--tag", default=None)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    return parser.parse_args()


def discover_tag(fit_dir: Path, tag: str | None) -> str:
    if tag:
        return tag
    manifest = fit_dir / "run_manifest.json"
    if manifest.exists():
        return json.loads(manifest.read_text())["tag"]
    matches = sorted(fit_dir.glob("*_gamma.csv"))
    if len(matches) == 1:
        return matches[0].name.removesuffix("_gamma.csv")
    raise FileNotFoundError(f"could not discover fit tag in {fit_dir}")


def load_truth(truth_dir: Path) -> dict[str, np.ndarray | pd.Index | float]:
    path = truth_dir / "truth_parameters.npz"
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=True) as truth:
        return {
            "group_names": pd.Index(truth["group_names"].astype(str), dtype=str),
            "cre_names": pd.Index(truth["cre_names"].astype(str), dtype=str),
            "lambda": np.asarray(truth["lambda_rate"], dtype=np.float64),
            "rho": np.asarray(truth["rho"], dtype=np.float64),
            "gamma": np.asarray(truth["gamma"], dtype=np.float64),
            "beta_t7": float(np.asarray(truth["beta_t7"])),
            "phi_t7": float(np.asarray(truth["phi_t7"])),
            "phi_cre": float(np.asarray(truth["phi_cre"])),
            "p_drop_t7": float(np.asarray(truth["p_drop_t7"])),
            "p_drop_cre": float(np.asarray(truth["p_drop_cre"])),
        }


def load_estimated_gamma(fit_dir: Path, tag: str) -> pd.DataFrame:
    path = fit_dir / f"{tag}_gamma.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    cols = [
        "group",
        "cre",
        "gamma_mean",
        "gamma_lo",
        "gamma_hi",
        "n_t7_pos",
        "n_cre_pos",
        "n_double_pos",
        "prior_dominated",
    ]
    gamma = pd.read_csv(path)
    keep = [col for col in cols if col in gamma.columns]
    return gamma[keep].assign(group=lambda x: x["group"].astype(str), cre=lambda x: x["cre"].astype(str))


def load_estimated_rho(fit_dir: Path, tag: str) -> pd.DataFrame:
    path = fit_dir / f"{tag}_rho.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path).assign(group=lambda x: x["group"].astype(str))


def posterior_lambda_summary(
    fit_dir: Path,
    tag: str,
    groups: pd.Index,
    cres: pd.Index,
) -> pd.DataFrame | None:
    path = fit_dir / f"{tag}_posterior_samples.npz"
    if not path.exists():
        return None
    with np.load(path, allow_pickle=True) as posterior:
        if not {"log_rho", "log_a", "group_names", "cre_names"}.issubset(posterior.files):
            return None
        post_groups = pd.Index(posterior["group_names"].astype(str), dtype=str)
        post_cres = pd.Index(posterior["cre_names"].astype(str), dtype=str)
        gpos = post_groups.get_indexer(groups)
        cpos = post_cres.get_indexer(cres)
        if (gpos < 0).any() or (cpos < 0).any():
            return None
        log_rho = np.asarray(posterior["log_rho"], dtype=np.float64)[:, gpos]
        log_a = np.asarray(posterior["log_a"], dtype=np.float64)[:, cpos]
    values = np.exp(log_rho[:, :, None] + log_a[:, None, :])
    mean = values.mean(axis=0)
    lo, hi = np.percentile(values, [5, 95], axis=0)
    gg, cc = np.meshgrid(np.arange(len(groups)), np.arange(len(cres)), indexing="ij")
    return pd.DataFrame(
        {
            "group": groups.to_numpy()[gg.ravel()],
            "cre": cres.to_numpy()[cc.ravel()],
            "lambda_mean": mean.ravel(),
            "lambda_lo": lo.ravel(),
            "lambda_hi": hi.ravel(),
        }
    )


def load_scalar_estimates(fit_dir: Path, tag: str, truth: dict[str, float]) -> pd.DataFrame:
    path = fit_dir / f"{tag}_scalar_samples.npz"
    if not path.exists():
        raise FileNotFoundError(path)
    rows = []
    with np.load(path, allow_pickle=True) as scalars:
        for key in ("beta_t7", "phi_t7", "phi_cre", "p_drop_t7", "p_drop_cre"):
            if key not in scalars.files:
                continue
            values = np.asarray(scalars[key], dtype=np.float64).reshape(-1)
            lo, hi = np.percentile(values, [5, 95])
            rows.append(
                {
                    "parameter": key,
                    "truth": float(truth[key]),
                    "estimate_mean": float(values.mean()),
                    "estimate_lo": float(lo),
                    "estimate_hi": float(hi),
                    "bias": float(values.mean() - truth[key]),
                    "relative_error": float((values.mean() - truth[key]) / truth[key])
                    if truth[key] != 0
                    else np.nan,
                    "covered_90": bool(lo <= truth[key] <= hi),
                }
            )
    return pd.DataFrame(rows)


def truth_pair_frame(truth: dict[str, np.ndarray | pd.Index]) -> pd.DataFrame:
    groups = truth["group_names"]
    cres = truth["cre_names"]
    gg, cc = np.meshgrid(np.arange(len(groups)), np.arange(len(cres)), indexing="ij")
    return pd.DataFrame(
        {
            "group": groups.to_numpy()[gg.ravel()],
            "cre": cres.to_numpy()[cc.ravel()],
            "gamma_true": truth["gamma"].ravel(),
            "lambda_true": truth["lambda"].ravel(),
        }
    )


def recovery_metrics(frame: pd.DataFrame, truth_col: str, estimate_col: str) -> dict:
    x = np.asarray(frame[truth_col], dtype=np.float64)
    y = np.asarray(frame[estimate_col], dtype=np.float64)
    finite = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    if finite.sum() < 2:
        return {"n": int(finite.sum())}
    lx = np.log(x[finite])
    ly = np.log(y[finite])
    diff = ly - lx
    return {
        "n": int(finite.sum()),
        "pearson_log": float(np.corrcoef(lx, ly)[0, 1]),
        "mae_log": float(np.mean(np.abs(diff))),
        "rmse_log": float(np.sqrt(np.mean(np.square(diff)))),
        "bias_log_est_minus_truth": float(diff.mean()),
        "median_fold_error": float(np.exp(np.median(np.abs(diff)))),
    }


def add_coverage(frame: pd.DataFrame, truth_col: str, lo_col: str, hi_col: str, output_col: str) -> pd.DataFrame:
    if {lo_col, hi_col}.issubset(frame.columns):
        frame[output_col] = (
            (frame[lo_col].to_numpy(dtype=float) <= frame[truth_col].to_numpy(dtype=float))
            & (frame[truth_col].to_numpy(dtype=float) <= frame[hi_col].to_numpy(dtype=float))
        )
    return frame


def save_recovery_plot(path: Path, frame: pd.DataFrame, pairs: list[tuple[str, str, str]]) -> None:
    fig, axes = plt.subplots(1, len(pairs), figsize=(4.4 * len(pairs), 4.0), squeeze=False)
    for ax, (truth_col, estimate_col, label) in zip(axes.ravel(), pairs):
        x = np.asarray(frame[truth_col], dtype=np.float64)
        y = np.asarray(frame[estimate_col], dtype=np.float64)
        finite = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
        ax.scatter(np.log(x[finite]), np.log(y[finite]), s=4, alpha=0.25, linewidths=0)
        if finite.any():
            lo = float(min(np.log(x[finite]).min(), np.log(y[finite]).min()))
            hi = float(max(np.log(x[finite]).max(), np.log(y[finite]).max()))
            pad = (hi - lo) * 0.04 if hi > lo else 0.1
            ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="#444444", lw=1)
            ax.set_xlim(lo - pad, hi + pad)
            ax.set_ylim(lo - pad, hi + pad)
        ax.set_title(label)
        ax.set_xlabel("log truth")
        ax.set_ylabel("log estimate")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(path.with_suffix(f".{ext}"), bbox_inches="tight", dpi=300)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    tag = discover_tag(args.fit_dir, args.tag)
    truth = load_truth(args.truth_dir)

    pair = truth_pair_frame(truth)
    gamma = load_estimated_gamma(args.fit_dir, tag)
    pair = pair.merge(gamma, on=["group", "cre"], how="left")
    pair = add_coverage(pair, "gamma_true", "gamma_lo", "gamma_hi", "gamma_covered_90")

    lam = posterior_lambda_summary(args.fit_dir, tag, truth["group_names"], truth["cre_names"])
    if lam is not None:
        pair = pair.merge(lam, on=["group", "cre"], how="left")
        pair = add_coverage(pair, "lambda_true", "lambda_lo", "lambda_hi", "lambda_covered_90")

    rho = load_estimated_rho(args.fit_dir, tag)
    rho_truth = pd.DataFrame({"group": truth["group_names"].to_numpy(), "rho_true": truth["rho"]})
    rho = rho_truth.merge(rho, on="group", how="left")
    rho = add_coverage(rho, "rho_true", "rho_lo", "rho_hi", "rho_covered_90")

    scalars = load_scalar_estimates(args.fit_dir, tag, truth)
    pair.to_csv(args.outdir / "per_pair_parameter_recovery.csv", index=False)
    rho.to_csv(args.outdir / "rho_recovery.csv", index=False)
    scalars.to_csv(args.outdir / "scalar_recovery.csv", index=False)

    metrics = {
        "gamma": recovery_metrics(pair, "gamma_true", "gamma_mean"),
        "rho": recovery_metrics(rho, "rho_true", "rho_mean"),
        "scalar_coverage_90": {
            row["parameter"]: bool(row["covered_90"]) for _, row in scalars.iterrows()
        },
    }
    if "lambda_mean" in pair.columns:
        metrics["lambda"] = recovery_metrics(pair, "lambda_true", "lambda_mean")
    for coverage_col in ("gamma_covered_90", "lambda_covered_90", "rho_covered_90"):
        if coverage_col in pair.columns:
            metrics[coverage_col] = float(pair[coverage_col].mean())
        elif coverage_col in rho.columns:
            metrics[coverage_col] = float(rho[coverage_col].mean())
    write_json(args.outdir / "recovery_metrics.json", metrics)

    plot_pairs = [("gamma_true", "gamma_mean", "cCRE activity gamma")]
    if "lambda_mean" in pair.columns:
        plot_pairs.append(("lambda_true", "lambda_mean", "infection rate lambda"))
    save_recovery_plot(args.outdir / "parameter_recovery_scatter", pair, plot_pairs)
    print(f"[evaluate] wrote recovery outputs to {args.outdir}", flush=True)


if __name__ == "__main__":
    main()
