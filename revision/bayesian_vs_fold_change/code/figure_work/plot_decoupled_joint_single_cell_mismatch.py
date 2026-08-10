#!/usr/bin/env python3
"""Check single-cell T7/cCRE mismatch for decoupled > joint pairs."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# The shared analysis layer (analysis_utils and the plot_* modules that other
# scripts import) stays in the parent code/ directory.
import sys as _sys
from pathlib import Path as _Path
_CODE_DIR = _Path(__file__).resolve().parent.parent
if str(_CODE_DIR) not in _sys.path:
    _sys.path.insert(0, str(_CODE_DIR))

from analysis_utils import (
    ANALYSIS_DIR,
    DEFAULT_H5AD,
    FIGURES_WORK,
    log,
    read_and_prepare_adata,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5ad", type=Path, default=DEFAULT_H5AD)
    parser.add_argument(
        "--pairs-csv",
        type=Path,
        default=(
            ANALYSIS_DIR
            / "results"
            / "figures"
            / "method_activity_decoupled_gt_joint_diagnostics_pairs.csv"
        ),
    )
    parser.add_argument(
        "--figures-dir", type=Path, default=FIGURES_WORK
    )
    parser.add_argument(
        "--stem", default="method_activity_decoupled_gt_joint_single_cell_mismatch"
    )
    return parser.parse_args()


def _vector(frame, cells: pd.Index, cre: str) -> np.ndarray:
    if not hasattr(frame, "loc"):
        raise TypeError("Expected adata.obsm count matrices to be pandas DataFrames")
    return frame.loc[cells, cre].to_numpy(dtype=float, copy=False)


def compute_contingency(adata, pairs: pd.DataFrame) -> pd.DataFrame:
    labels = adata.obs["subclass"].astype(str)
    t7 = adata.obsm["T7CRE"]
    ccre = adata.obsm["CRE"]
    available_cres = set(t7.columns.astype(str)).intersection(ccre.columns.astype(str))
    rows = []
    for group, chunk in pairs.groupby("group", sort=False):
        cells = labels.index[labels.eq(str(group))]
        for _, pair in chunk.iterrows():
            cre = str(pair["cre"])
            row = pair.to_dict()
            row["n_cells_single_cell"] = int(len(cells))
            if len(cells) == 0 or cre not in available_cres:
                row.update(
                    {
                        "t7_positive_cells": np.nan,
                        "ccre_positive_cells": np.nan,
                        "both_positive_cells": np.nan,
                        "t7_only_cells": np.nan,
                        "ccre_only_cells": np.nan,
                        "neither_cells": np.nan,
                        "either_positive_cells": np.nan,
                        "mismatch_cells": np.nan,
                        "mismatch_fraction_among_either": np.nan,
                        "co_positive_fraction_among_either": np.nan,
                    }
                )
                rows.append(row)
                continue
            t7_values = _vector(t7, cells, cre)
            ccre_values = _vector(ccre, cells, cre)
            t7_pos = t7_values > 0
            ccre_pos = ccre_values > 0
            both = t7_pos & ccre_pos
            t7_only = t7_pos & ~ccre_pos
            ccre_only = ccre_pos & ~t7_pos
            either = t7_pos | ccre_pos
            mismatch = t7_only | ccre_only
            n_either = int(either.sum())
            n_mismatch = int(mismatch.sum())
            row.update(
                {
                    "t7_positive_cells": int(t7_pos.sum()),
                    "ccre_positive_cells": int(ccre_pos.sum()),
                    "both_positive_cells": int(both.sum()),
                    "t7_only_cells": int(t7_only.sum()),
                    "ccre_only_cells": int(ccre_only.sum()),
                    "neither_cells": int((~either).sum()),
                    "either_positive_cells": n_either,
                    "mismatch_cells": n_mismatch,
                    "mismatch_fraction_among_either": (
                        n_mismatch / n_either if n_either else np.nan
                    ),
                    "co_positive_fraction_among_either": (
                        int(both.sum()) / n_either if n_either else np.nan
                    ),
                }
            )
            rows.append(row)
    out = pd.DataFrame(rows)
    out["signal_class"] = "no T7/cCRE positive cells"
    has_signal = out["either_positive_cells"].fillna(0).gt(0)
    all_mismatch = has_signal & out["both_positive_cells"].fillna(0).eq(0)
    mostly_mismatch = (
        has_signal
        & ~all_mismatch
        & out["mismatch_fraction_among_either"].fillna(0).ge(0.8)
    )
    out.loc[all_mismatch, "signal_class"] = "all positive cells mismatched"
    out.loc[mostly_mismatch, "signal_class"] = ">=80% positive cells mismatched"
    out.loc[has_signal & ~(all_mismatch | mostly_mismatch), "signal_class"] = (
        "substantial co-positive cells"
    )
    return out


def summarize(table: pd.DataFrame) -> dict:
    has_signal = table["either_positive_cells"].fillna(0).gt(0)
    return {
        "n_pairs": int(len(table)),
        "no_positive_cells": int((~has_signal).sum()),
        "has_any_positive_cell": int(has_signal.sum()),
        "both_positive_zero_among_signal": int(
            (has_signal & table["both_positive_cells"].fillna(0).eq(0)).sum()
        ),
        "fraction_signal_with_no_copositive_cells": float(
            (has_signal & table["both_positive_cells"].fillna(0).eq(0)).sum()
            / has_signal.sum()
        )
        if has_signal.any()
        else np.nan,
        "fraction_signal_mismatch_ge_0p8": float(
            (
                has_signal
                & table["mismatch_fraction_among_either"].fillna(0).ge(0.8)
            ).sum()
            / has_signal.sum()
        )
        if has_signal.any()
        else np.nan,
        "median_mismatch_fraction_among_signal": float(
            table.loc[has_signal, "mismatch_fraction_among_either"].median()
        )
        if has_signal.any()
        else np.nan,
        "median_both_positive_cells_among_signal": float(
            table.loc[has_signal, "both_positive_cells"].median()
        )
        if has_signal.any()
        else np.nan,
        "median_t7_only_cells_among_signal": float(
            table.loc[has_signal, "t7_only_cells"].median()
        )
        if has_signal.any()
        else np.nan,
        "median_ccre_only_cells_among_signal": float(
            table.loc[has_signal, "ccre_only_cells"].median()
        )
        if has_signal.any()
        else np.nan,
        "signal_class_counts": table["signal_class"].value_counts().to_dict(),
    }


def plot_mismatch(table: pd.DataFrame, output: Path) -> None:
    sns.set_theme(context="paper", style="white")
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 10.4), constrained_layout=True)
    axes = axes.ravel()
    palette = {
        "no T7/cCRE positive cells": "#7b3294",
        "all positive cells mismatched": "#d62728",
        ">=80% positive cells mismatched": "#ff7f0e",
        "substantial co-positive cells": "#2f6f8f",
    }

    ax = axes[0]
    sns.scatterplot(
        data=table,
        x="Joint",
        y="Decoupled",
        hue="signal_class",
        palette=palette,
        s=14,
        alpha=0.65,
        linewidth=0,
        ax=ax,
        rasterized=True,
    )
    lo = np.nanpercentile(table[["Joint", "Decoupled"]].to_numpy(float), 1)
    hi = np.nanpercentile(table[["Joint", "Decoupled"]].to_numpy(float), 99)
    ax.plot([lo, hi], [lo, hi], color="black", lw=0.8, alpha=0.7)
    ax.set_title("Selected decoupled > joint pairs")
    ax.legend(title="", fontsize=7, loc="best")

    ax = axes[1]
    counts = (
        table["signal_class"]
        .value_counts()
        .reindex(list(palette), fill_value=0)
        .rename_axis("class")
        .reset_index(name="n")
    )
    sns.barplot(data=counts, x="n", y="class", palette=palette, ax=ax)
    ax.set_xlabel("number of pairs")
    ax.set_ylabel("")
    ax.set_title("Single-cell overlap classes")

    ax = axes[2]
    has_signal = table[table["either_positive_cells"].fillna(0).gt(0)]
    sns.histplot(
        has_signal["mismatch_fraction_among_either"],
        bins=np.linspace(0, 1, 31),
        color="#d62728",
        ax=ax,
    )
    ax.set_xlabel("(T7-only + cCRE-only) / (T7 or cCRE positive cells)")
    ax.set_title("Mismatch fraction among pairs with any signal")

    ax = axes[3]
    long = table[
        ["both_positive_cells", "t7_only_cells", "ccre_only_cells", "either_positive_cells"]
    ].rename(
        columns={
            "both_positive_cells": "both+",
            "t7_only_cells": "T7 only",
            "ccre_only_cells": "cCRE only",
            "either_positive_cells": "either+",
        }
    )
    long = long.melt(var_name="cell class", value_name="cells")
    long["log10_cells_plus1"] = np.log10(pd.to_numeric(long["cells"], errors="coerce") + 1)
    sns.boxplot(
        data=long,
        x="cell class",
        y="log10_cells_plus1",
        color="white",
        showfliers=False,
        linewidth=0.8,
        ax=ax,
    )
    sns.stripplot(
        data=long,
        x="cell class",
        y="log10_cells_plus1",
        color="#2f6f8f",
        alpha=0.16,
        size=1.8,
        jitter=0.25,
        ax=ax,
        rasterized=True,
    )
    ax.set_xlabel("")
    ax.set_ylabel("log10(cells + 1)")
    ax.set_title("Single-cell contingency counts")

    fig.suptitle(
        "Single-cell T7/cCRE mismatch for Decoupled - Joint >= 1.5 pairs",
        fontsize=13,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.figures_dir.mkdir(parents=True, exist_ok=True)
    pairs = pd.read_csv(args.pairs_csv)
    adata = read_and_prepare_adata(args.h5ad)
    table = compute_contingency(adata, pairs)
    output_csv = args.figures_dir / f"{args.stem}_pairs.csv"
    output_pdf = args.figures_dir / f"{args.stem}.pdf"
    output_manifest = args.figures_dir / f"{args.stem}_manifest.json"
    table.to_csv(output_csv, index=False)
    plot_mismatch(table, output_pdf)
    summary = summarize(table)
    write_json(
        output_manifest,
        {
            "pairs_source": str(args.pairs_csv),
            "output_pdf": str(output_pdf),
            "output_csv": str(output_csv),
            "summary": summary,
        },
    )
    log(
        "[single-cell mismatch] wrote diagnostics; "
        f"{summary['fraction_signal_with_no_copositive_cells']:.3f} of signal pairs "
        "have no co-positive cells"
    )


if __name__ == "__main__":
    main()
