#!/usr/bin/env python3
"""Precision/recall of replicate-concordant BH calls against orthogonal assays.

The call sets come only from the BH q-values behind
`overlap_t7_ge50_significant_call_concordance_bh_q`: the original replicate, the
new low-dose replicate, their union, their intersection, and the replicate-
concordant universe, which keeps only pairs where the two replicates agree and
therefore restricts the assay-positive pairs to that same universe.

Precision, recall, the naive-precision prevalence baseline, and the one-sided
Fisher test all reuse `benchmark_assay` from
`revision/bayesian_vs_fold_change/code/plot_t7_filter_precision_recall.py`, so
the definitions are identical to the published precision-recall figure:
`assay_positive` is the assay matrix above 0.5, precision is TP/significant, and
recall is TP/assay_positive.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Final

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE: Final[Path] = Path(__file__).resolve()
ANALYSIS_DIR: Final[Path] = HERE.parent.parent
REPO_ROOT: Final[Path] = ANALYSIS_DIR.parents[1]
ORIGIN_CODE: Final[Path] = REPO_ROOT / "revision" / "bayesian_vs_fold_change" / "code"
if str(ORIGIN_CODE) not in sys.path:
    sys.path.insert(0, str(ORIGIN_CODE))

from analysis_utils import write_json  # noqa: E402
from plot_t7_filter_precision_recall import (  # noqa: E402
    ASSAYS,
    assay_positive_for_tests,
    benchmark_assay,
    read_assay,
)


DEFAULT_COMPARISON_DIR: Final[Path] = ANALYSIS_DIR / "results" / "comparison"
KEY: Final[list[str]] = ["group", "cre"]
CONCORDANT_STATUSES: Final[tuple[str, str]] = (
    "both_significant",
    "neither_significant",
)
CALL_SETS: Final[tuple[tuple[str, str], ...]] = (
    ("Original replicate", "all"),
    ("New low-dose replicate", "all"),
    ("Either replicate", "all"),
    ("Both replicates", "all"),
    ("Replicate-concordant pairs", "concordant"),
)
SET_COLORS: Final[dict[str, str]] = {
    "Original replicate": "#4477AA",
    "New low-dose replicate": "#CC6677",
    "Either replicate": "#88CCEE",
    "Both replicates": "#6E6E6E",
    "Replicate-concordant pairs": "#117733",
}
STEM: Final[str] = "replicate_concordant_bh_call_precision_recall"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison-dir", type=Path, default=DEFAULT_COMPARISON_DIR)
    parser.add_argument("--t7-threshold", type=float, default=50)
    parser.add_argument("--q-cutoff", type=float, default=0.05)
    return parser.parse_args()


def token(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def read_calls(comparison_dir: Path, t7_threshold: float) -> pd.DataFrame:
    path = (
        comparison_dir
        / "tables"
        / f"overlap_t7_ge{token(t7_threshold)}_significant_call_concordance_calls.csv.gz"
    )
    if not path.exists():
        raise FileNotFoundError(
            f"Missing call table: {path}. Run plot_call_concordance.py first."
        )
    frame = pd.read_csv(path)
    required = [
        *KEY,
        "origin_significant_bh_q",
        "new_significant_bh_q",
        "bh_q_call_status",
    ]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"{path} lacks required columns: {missing}")
    if frame.duplicated(subset=KEY).any():
        raise ValueError(f"{path} contains duplicated group/cre keys")
    frame[KEY] = frame[KEY].astype(str)
    return frame


def build_tests(calls: pd.DataFrame, t7_threshold: float) -> pd.DataFrame:
    """Encode each call set as a q_right column of 0/1 for `benchmark_assay`."""
    origin = calls["origin_significant_bh_q"].astype(bool)
    new = calls["new_significant_bh_q"].astype(bool)
    concordant = calls["bh_q_call_status"].isin(CONCORDANT_STATUSES)
    if not concordant.any():
        raise ValueError("No replicate-concordant pairs found")
    predictions = {
        "Original replicate": origin,
        "New low-dose replicate": new,
        "Either replicate": origin | new,
        "Both replicates": origin & new,
        "Replicate-concordant pairs": origin & new,
    }
    frames = []
    for label, universe in CALL_SETS:
        selected = concordant if universe == "concordant" else pd.Series(
            True, index=calls.index
        )
        significant = predictions[label].loc[selected]
        if universe == "concordant":
            agreed = (
                origin.loc[selected] == new.loc[selected]
            )
            if not bool(agreed.all()):
                raise ValueError(
                    "Concordant universe contains pairs whose replicate calls differ"
                )
        frames.append(
            pd.DataFrame(
                {
                    "group": calls.loc[selected, "group"].to_numpy(),
                    "cre": calls.loc[selected, "cre"].to_numpy(),
                    "method": label,
                    "t7_threshold": float(t7_threshold),
                    "q_right": np.where(significant.to_numpy(bool), 0.0, 1.0),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def assay_coverage(calls: pd.DataFrame, assay: pd.DataFrame) -> dict[str, int]:
    covered = calls["group"].isin(assay.index)
    return {
        "n_pairs": int(len(calls)),
        "n_pairs_in_assay_covered_groups": int(covered.sum()),
        "n_pairs_in_groups_absent_from_assay": int((~covered).sum()),
        "n_assay_positive_pairs": int(
            assay_positive_for_tests(calls, assay).sum()
        ),
    }


def plot_metrics(
    metrics: pd.DataFrame, assays: list[str], output_stem: Path, q_cutoff: float
) -> None:
    labels = [label for label, _ in CALL_SETS]
    positions = np.arange(len(labels))
    fig, axes = plt.subplots(
        2, len(assays), figsize=(5.6 * len(assays), 8.4), layout="constrained"
    )
    axes = np.atleast_2d(axes)
    if axes.shape != (2, len(assays)):
        axes = axes.reshape(2, len(assays))
    for column, assay_name in enumerate(assays):
        frame = metrics.loc[metrics["assay"].eq(assay_name)].set_index("method")
        for row, metric in enumerate(("precision", "recall")):
            ax = axes[row, column]
            values = frame.loc[labels, metric].to_numpy(float)
            ax.bar(
                positions,
                values,
                color=[SET_COLORS[label] for label in labels],
                width=0.7,
            )
            denominator = "significant" if metric == "precision" else "assay_positive"
            for position, label in zip(positions, labels):
                row_values = frame.loc[label]
                value = float(row_values[metric])
                ax.text(
                    position,
                    value if np.isfinite(value) else 0.0,
                    f"{value:.2f}\n{int(row_values['TP'])}/"
                    f"{int(row_values[denominator])}",
                    ha="center",
                    va="bottom",
                    fontsize=7.5,
                )
            if metric == "precision":
                prevalence = (
                    frame.loc[labels, "assay_positive"]
                    / frame.loc[labels, "tested"]
                ).to_numpy(float)
                ax.plot(
                    positions,
                    prevalence,
                    linestyle="--",
                    linewidth=1.2,
                    marker="_",
                    markersize=18,
                    color="black",
                    label="Naive precision (assay prevalence)",
                )
                ax.legend(frameon=False, fontsize=7.5, loc="upper left")
            finite = values[np.isfinite(values)]
            ceiling = max(float(finite.max()) if finite.size else 0.0, 0.05)
            ax.set_ylim(0, ceiling * 1.35)
            ax.set_xticks(positions, labels, rotation=25, ha="right", fontsize=8)
            ax.set_ylabel(metric.capitalize())
            ax.grid(axis="y", color="0.90", linewidth=0.7)
            ax.set_axisbelow(True)
            if row == 0:
                ax.set_title(assay_name, fontsize=10.5)
    fig.suptitle(
        "Precision and recall of BH significant calls against orthogonal assays\n"
        f"Origin-versus-new overlap universe, BH q ≤ {q_cutoff:g}; precision = "
        "TP/significant, recall = TP/assay-positive pairs",
        fontsize=10.5,
        y=1.05,
    )
    for suffix in ("png", "pdf"):
        fig.savefig(output_stem.with_suffix(f".{suffix}"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    tables_dir = args.comparison_dir / "tables"
    figures_dir = args.comparison_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    calls = read_calls(args.comparison_dir, args.t7_threshold)
    tests = build_tests(calls, args.t7_threshold)
    assays = {name: read_assay(path) for name, path in ASSAYS.items()}

    metrics = pd.concat(
        [
            benchmark_assay(tests, name, assay, args.q_cutoff)
            for name, assay in assays.items()
        ],
        ignore_index=True,
    )
    metrics.to_csv(tables_dir / f"{STEM}.csv", index=False)
    plot_metrics(metrics, list(assays), figures_dir / STEM, args.q_cutoff)

    concordant = calls.loc[calls["bh_q_call_status"].isin(CONCORDANT_STATUSES)]
    manifest = {
        "figure_stem": STEM,
        "metrics_table": str(tables_dir / f"{STEM}.csv"),
        "source_table": str(
            tables_dir
            / f"overlap_t7_ge{token(args.t7_threshold)}"
            "_significant_call_concordance_calls.csv.gz"
        ),
        "definition_source": str(ORIGIN_CODE / "plot_t7_filter_precision_recall.py"),
        "t7_threshold": float(args.t7_threshold),
        "q_cutoff": float(args.q_cutoff),
        "n_pairs_all": int(len(calls)),
        "n_pairs_concordant": int(len(concordant)),
        "concordant_status_counts": calls["bh_q_call_status"]
        .value_counts()
        .to_dict(),
        "assay_coverage": {
            name: {
                "all_pairs": assay_coverage(calls, assay),
                "concordant_pairs": assay_coverage(concordant, assay),
            }
            for name, assay in assays.items()
        },
        "metrics": metrics.to_dict(orient="records"),
    }
    write_json(tables_dir / f"{STEM}_manifest.json", manifest)
    print(
        metrics[
            [
                "assay",
                "method",
                "TP",
                "significant",
                "assay_positive",
                "tested",
                "precision",
                "recall",
                "fisher_oddsratio",
                "fisher_p",
            ]
        ].to_string(index=False),
        flush=True,
    )


if __name__ == "__main__":
    main()
