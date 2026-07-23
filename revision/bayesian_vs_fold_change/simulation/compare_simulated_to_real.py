#!/usr/bin/env python3
"""Compare real and simulated STARR-FISH count statistics."""

from __future__ import annotations

import argparse
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

from analysis_utils import (  # noqa: E402
    DEFAULT_H5AD,
    cre_blacklist,
    log,
    read_and_prepare_adata,
    write_json,
)

SIM_DIR = Path(__file__).resolve().parent
DEFAULT_SIM_H5AD = SIM_DIR / "results" / "joint_dropout_simulated" / "simulated_joint_dropout.h5ad"
DEFAULT_OUTDIR = SIM_DIR / "results" / "joint_dropout_stats"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real-h5ad", type=Path, default=DEFAULT_H5AD)
    parser.add_argument("--sim-h5ad", type=Path, default=DEFAULT_SIM_H5AD)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--section", choices=["all", "sec1", "sec2"], default="all")
    parser.add_argument("--max-cells", type=int, default=None, help="Smoke testing only.")
    parser.add_argument("--max-cres", type=int, default=None, help="Smoke testing only.")
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Cell subsampling seed used when --max-cells is set.",
    )
    return parser.parse_args()


def prepare_counts(
    path: Path,
    section: str,
    max_cells: int | None,
    max_cres: int | None,
    seed: int,
):
    adata = read_and_prepare_adata(
        path, section=section, max_cells=max_cells, max_cres=max_cres, seed=seed
    )
    cre_info = adata.uns["CRE_info"].copy()
    blacklist = set(cre_blacklist(cre_info.index))
    cre_names = [name for name in cre_info.index.astype(str) if name not in blacklist]
    t7 = adata.obsm["T7CRE"].loc[:, cre_names].to_numpy(dtype=np.int64)
    cre = adata.obsm["CRE"].loc[:, cre_names].to_numpy(dtype=np.int64)
    groups = adata.obs["subclass"].astype(str).to_numpy()
    return t7, cre, groups, pd.Index(cre_names, dtype=str)


def channel_summary(values: np.ndarray) -> dict[str, float]:
    positive = values > 0
    return {
        "zero_fraction": float(1.0 - positive.mean()),
        "positive_fraction": float(positive.mean()),
        "mean": float(values.mean()),
        "variance": float(values.var()),
        "mean_nonzero": float(values[positive].mean()) if positive.any() else 0.0,
        "max": int(values.max()) if values.size else 0,
    }


def global_summary(name: str, t7: np.ndarray, cre: np.ndarray) -> pd.DataFrame:
    rows = []
    for channel, values in (("t7", t7), ("cre", cre)):
        for stat, value in channel_summary(values).items():
            rows.append({"dataset": name, "scope": channel, "stat": stat, "value": value})
    joint_masks = {
        "all_zero_fraction": (t7 == 0) & (cre == 0),
        "t7_only_fraction": (t7 > 0) & (cre == 0),
        "cre_only_fraction": (t7 == 0) & (cre > 0),
        "double_positive_fraction": (t7 > 0) & (cre > 0),
    }
    for stat, mask in joint_masks.items():
        rows.append({"dataset": name, "scope": "joint", "stat": stat, "value": float(mask.mean())})
    rows.extend(
        [
            {"dataset": name, "scope": "shape", "stat": "n_cells", "value": int(t7.shape[0])},
            {"dataset": name, "scope": "shape", "stat": "n_cres", "value": int(t7.shape[1])},
            {"dataset": name, "scope": "shape", "stat": "n_pairs", "value": int(t7.size)},
        ]
    )
    return pd.DataFrame(rows)


def per_group_summary(name: str, t7: np.ndarray, cre: np.ndarray, groups: np.ndarray) -> pd.DataFrame:
    rows = []
    for group in pd.Index(groups).unique().astype(str):
        mask = groups == group
        t7_g = t7[mask]
        cre_g = cre[mask]
        rows.append(
            {
                "dataset": name,
                "group": group,
                "n_cells": int(mask.sum()),
                "t7_positive_fraction": float((t7_g > 0).mean()),
                "cre_positive_fraction": float((cre_g > 0).mean()),
                "double_positive_fraction": float(((t7_g > 0) & (cre_g > 0)).mean()),
                "t7_mean_nonzero": float(t7_g[t7_g > 0].mean()) if (t7_g > 0).any() else 0.0,
                "cre_mean_nonzero": float(cre_g[cre_g > 0].mean()) if (cre_g > 0).any() else 0.0,
            }
        )
    return pd.DataFrame(rows)


def per_cre_summary(name: str, t7: np.ndarray, cre: np.ndarray, cre_names: pd.Index) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "dataset": name,
            "cre": cre_names.to_numpy(),
            "t7_positive_fraction": (t7 > 0).mean(axis=0),
            "cre_positive_fraction": (cre > 0).mean(axis=0),
            "double_positive_fraction": ((t7 > 0) & (cre > 0)).mean(axis=0),
            "t7_total": t7.sum(axis=0),
            "cre_total": cre.sum(axis=0),
            "t7_mean_nonzero": [
                float(col[col > 0].mean()) if (col > 0).any() else 0.0 for col in t7.T
            ],
            "cre_mean_nonzero": [
                float(col[col > 0].mean()) if (col > 0).any() else 0.0 for col in cre.T
            ],
        }
    )


def per_pair_evidence(
    name: str,
    t7: np.ndarray,
    cre: np.ndarray,
    groups: np.ndarray,
    cre_names: pd.Index,
) -> pd.DataFrame:
    frames = []
    group_index = pd.Index(groups).astype(str)
    for group in group_index.unique():
        mask = group_index == group
        t7_g = t7[mask]
        cre_g = cre[mask]
        n = int(mask.sum())
        frames.append(
            pd.DataFrame(
                {
                    "dataset": name,
                    "group": str(group),
                    "cre": cre_names.to_numpy(),
                    "n_cells": n,
                    "n_t7_pos": (t7_g > 0).sum(axis=0),
                    "n_cre_pos": (cre_g > 0).sum(axis=0),
                    "n_double_pos": ((t7_g > 0) & (cre_g > 0)).sum(axis=0),
                    "t7_total": t7_g.sum(axis=0),
                    "cre_total": cre_g.sum(axis=0),
                }
            )
        )
    out = pd.concat(frames, ignore_index=True)
    for col in ("n_t7_pos", "n_cre_pos", "n_double_pos"):
        out[f"{col}_fraction"] = out[col] / out["n_cells"].clip(lower=1)
    return out


def paired_metrics(real: pd.DataFrame, sim: pd.DataFrame, keys: list[str], columns: list[str]) -> dict:
    merged = real.merge(sim, on=keys, suffixes=("_real", "_sim"))
    metrics = {}
    for col in columns:
        x = merged[f"{col}_real"].to_numpy(dtype=float)
        y = merged[f"{col}_sim"].to_numpy(dtype=float)
        finite = np.isfinite(x) & np.isfinite(y)
        if finite.sum() < 2:
            metrics[col] = {"n": int(finite.sum())}
            continue
        diff = y[finite] - x[finite]
        pearson = (
            float(np.corrcoef(x[finite], y[finite])[0, 1])
            if np.std(x[finite]) > 0 and np.std(y[finite]) > 0
            else np.nan
        )
        metrics[col] = {
            "n": int(finite.sum()),
            "pearson": pearson,
            "mae": float(np.mean(np.abs(diff))),
            "rmse": float(np.sqrt(np.mean(np.square(diff)))),
            "bias_sim_minus_real": float(diff.mean()),
        }
    return metrics


def save_scatter(
    path: Path,
    real: pd.DataFrame,
    sim: pd.DataFrame,
    keys: list[str],
    columns: list[str],
    title_prefix: str,
) -> None:
    merged = real.merge(sim, on=keys, suffixes=("_real", "_sim"))
    fig, axes = plt.subplots(1, len(columns), figsize=(4.2 * len(columns), 3.8), squeeze=False)
    for ax, col in zip(axes.ravel(), columns):
        x = merged[f"{col}_real"].to_numpy(dtype=float)
        y = merged[f"{col}_sim"].to_numpy(dtype=float)
        ax.scatter(x, y, s=6, alpha=0.35, linewidths=0)
        finite = np.isfinite(x) & np.isfinite(y)
        if finite.any():
            lo = float(min(x[finite].min(), y[finite].min()))
            hi = float(max(x[finite].max(), y[finite].max()))
            pad = (hi - lo) * 0.04 if hi > lo else 0.01
            ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="#444444", lw=1)
            ax.set_xlim(lo - pad, hi + pad)
            ax.set_ylim(lo - pad, hi + pad)
        ax.set_title(col.replace("_", " "))
        ax.set_xlabel("real")
        ax.set_ylabel("simulated")
    fig.suptitle(title_prefix, x=0.01, ha="left")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(path.with_suffix(f".{ext}"), bbox_inches="tight", dpi=300)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    real_t7, real_cre, real_groups, real_cres = prepare_counts(
        args.real_h5ad, args.section, args.max_cells, args.max_cres, args.seed
    )
    sim_t7, sim_cre, sim_groups, sim_cres = prepare_counts(
        args.sim_h5ad, args.section, args.max_cells, args.max_cres, args.seed
    )
    common_cres = real_cres.intersection(sim_cres)
    if len(common_cres) == 0:
        raise ValueError("real and simulated H5AD files have no cCREs in common")
    real_pos = real_cres.get_indexer(common_cres)
    sim_pos = sim_cres.get_indexer(common_cres)
    real_t7, real_cre = real_t7[:, real_pos], real_cre[:, real_pos]
    sim_t7, sim_cre = sim_t7[:, sim_pos], sim_cre[:, sim_pos]

    if real_t7.shape != sim_t7.shape:
        raise ValueError(f"shape mismatch after alignment: real={real_t7.shape}, sim={sim_t7.shape}")
    if not np.array_equal(real_groups.astype(str), sim_groups.astype(str)):
        raise ValueError("real and simulated cell annotations are not aligned")

    log(f"[compare] aligned {real_t7.shape[0]:,} cells x {real_t7.shape[1]:,} cCREs")

    global_stats = pd.concat(
        [
            global_summary("real", real_t7, real_cre),
            global_summary("simulated", sim_t7, sim_cre),
        ],
        ignore_index=True,
    )
    group_stats = pd.concat(
        [
            per_group_summary("real", real_t7, real_cre, real_groups),
            per_group_summary("simulated", sim_t7, sim_cre, sim_groups),
        ],
        ignore_index=True,
    )
    cre_stats = pd.concat(
        [
            per_cre_summary("real", real_t7, real_cre, common_cres),
            per_cre_summary("simulated", sim_t7, sim_cre, common_cres),
        ],
        ignore_index=True,
    )
    pair_stats = pd.concat(
        [
            per_pair_evidence("real", real_t7, real_cre, real_groups, common_cres),
            per_pair_evidence("simulated", sim_t7, sim_cre, sim_groups, common_cres),
        ],
        ignore_index=True,
    )

    global_stats.to_csv(args.outdir / "global_stats.csv", index=False)
    group_stats.to_csv(args.outdir / "per_group_stats.csv", index=False)
    cre_stats.to_csv(args.outdir / "per_cre_stats.csv", index=False)
    pair_stats.to_csv(args.outdir / "per_pair_evidence.csv", index=False)

    group_real = group_stats[group_stats["dataset"] == "real"].drop(columns="dataset")
    group_sim = group_stats[group_stats["dataset"] == "simulated"].drop(columns="dataset")
    cre_real = cre_stats[cre_stats["dataset"] == "real"].drop(columns="dataset")
    cre_sim = cre_stats[cre_stats["dataset"] == "simulated"].drop(columns="dataset")
    pair_real = pair_stats[pair_stats["dataset"] == "real"].drop(columns="dataset")
    pair_sim = pair_stats[pair_stats["dataset"] == "simulated"].drop(columns="dataset")
    metrics = {
        "per_group": paired_metrics(
            group_real,
            group_sim,
            ["group"],
            [
                "t7_positive_fraction",
                "cre_positive_fraction",
                "double_positive_fraction",
                "t7_mean_nonzero",
                "cre_mean_nonzero",
            ],
        ),
        "per_cre": paired_metrics(
            cre_real,
            cre_sim,
            ["cre"],
            [
                "t7_positive_fraction",
                "cre_positive_fraction",
                "double_positive_fraction",
                "t7_total",
                "cre_total",
            ],
        ),
        "per_pair": paired_metrics(
            pair_real,
            pair_sim,
            ["group", "cre"],
            [
                "n_t7_pos_fraction",
                "n_cre_pos_fraction",
                "n_double_pos_fraction",
                "t7_total",
                "cre_total",
            ],
        ),
    }
    write_json(args.outdir / "real_vs_simulated_metrics.json", metrics)

    save_scatter(
        args.outdir / "per_cre_positive_fraction_scatter",
        cre_real,
        cre_sim,
        ["cre"],
        ["t7_positive_fraction", "cre_positive_fraction", "double_positive_fraction"],
        "Per-cCRE positive fractions",
    )
    save_scatter(
        args.outdir / "per_group_positive_fraction_scatter",
        group_real,
        group_sim,
        ["group"],
        ["t7_positive_fraction", "cre_positive_fraction", "double_positive_fraction"],
        "Per-subclass positive fractions",
    )
    save_scatter(
        args.outdir / "per_pair_positive_fraction_scatter",
        pair_real,
        pair_sim,
        ["group", "cre"],
        ["n_t7_pos_fraction", "n_cre_pos_fraction", "n_double_pos_fraction"],
        "Per-subclass/cCRE evidence fractions",
    )
    log(f"[compare] wrote summaries to {args.outdir}")


if __name__ == "__main__":
    main()
