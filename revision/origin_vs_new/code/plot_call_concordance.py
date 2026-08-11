#!/usr/bin/env python3
"""Compare old/new significant calls on one control reference and two call bases.

Both bases start from the same right-tail posterior p-value of the saved pair
tables. The `raw_p` basis calls a pair significant at `p <= cutoff`, so the call
does not depend on how many pairs entered the multiple-testing correction; the
`bh_q` basis applies BH across the pairs of the table and thresholds at
`q <= cutoff`. Emitting both as a matched pair isolates the effect of the
correction from the effect of the two experiments.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Final

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

from compare_origin_vs_new import call_metrics


HERE: Final[Path] = Path(__file__).resolve()
ANALYSIS_DIR: Final[Path] = HERE.parent.parent
DEFAULT_COMPARISON_DIR: Final[Path] = ANALYSIS_DIR / "results" / "comparison"
KEY: Final[list[str]] = ["group", "cre"]
BAR_COLORS: Final[tuple[str, str, str]] = ("#4477AA", "#6E6E6E", "#CC6677")
BASES: Final[tuple[str, ...]] = ("raw_p", "bh_q")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        choices=("overlap", "mean_plus_1sd"),
        default="overlap",
        help=(
            "overlap uses the primary mean-control overlap table; mean_plus_1sd "
            "uses the draw-wise mean+1SD control reference table."
        ),
    )
    parser.add_argument("--comparison-dir", type=Path, default=DEFAULT_COMPARISON_DIR)
    parser.add_argument("--t7-threshold", type=float, default=50)
    parser.add_argument("--cutoff", type=float, default=0.05)
    return parser.parse_args()


def token(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def resolve_source(
    source: str, comparison_dir: Path, t7_threshold: float
) -> tuple[Path, str, str]:
    """Return the pair table, its control-reference label, and the output prefix."""
    threshold_token = token(t7_threshold)
    if source == "mean_plus_1sd":
        return (
            comparison_dir
            / "tables"
            / f"shared_pair_mean_plus_1sd_comparison_t7_ge{threshold_token}.csv.gz",
            "Mean + 1 SD negative-control reference",
            "shared_pair",
        )
    return (
        comparison_dir
        / "tables"
        / f"overlap_t7_ge{threshold_token}_pair_comparison.csv.gz",
        "Mean negative-control reference",
        f"overlap_t7_ge{threshold_token}",
    )


def read_pairs(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing pair comparison table: {path}")
    frame = pd.read_csv(path)
    required = [*KEY, "origin_p_right", "new_p_right"]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"{path} lacks required columns: {missing}")
    if frame.duplicated(subset=KEY).any():
        raise ValueError(f"{path} contains duplicated group/cre keys")
    if frame[["origin_p_right", "new_p_right"]].isna().any().to_numpy().any():
        raise ValueError(f"{path} has missing p-values")
    return frame


def bh_adjust(pvalues: pd.Series) -> np.ndarray:
    return multipletests(pvalues.to_numpy(float), method="fdr_bh")[1]


def validate_bh(calls: pd.DataFrame, pairs: pd.DataFrame, cutoff: float, source: Path) -> None:
    """Check the in-script BH calls reproduce the table's shared-universe calls."""
    for dataset in ("origin", "new"):
        published = f"{dataset}_significant_common_q"
        if published not in pairs.columns:
            continue
        recomputed = calls[f"{dataset}_q_bh"].le(cutoff)
        disagreements = int((recomputed != pairs[published].astype(bool)).sum())
        if disagreements:
            raise ValueError(
                f"{source}: recomputed BH calls disagree with {published} "
                f"for {disagreements} pairs"
            )


def call_status(origin: pd.Series, new: pd.Series) -> np.ndarray:
    return np.select(
        [origin & new, origin & ~new, ~origin & new],
        ["both_significant", "origin_only_significant", "new_only_significant"],
        default="neither_significant",
    )


def basis_label(basis: str, cutoff: float, n_pairs: int) -> str:
    if basis == "raw_p":
        return f"raw posterior p ≤ {cutoff:g} (no multiple-testing correction)"
    return f"BH q ≤ {cutoff:g} across these {n_pairs:,} pairs"


def plot_calls(
    metrics: dict[str, float | int],
    output_stem: Path,
    *,
    n_pairs: int,
    reference_label: str,
    call_label: str,
    t7_threshold: float,
) -> None:
    labels = ["Original only", "Both", "New only"]
    values = [
        int(metrics["origin_only_significant"]),
        int(metrics["both_significant"]),
        int(metrics["new_only_significant"]),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.8), layout="constrained")
    axes[0].bar(labels, values, color=list(BAR_COLORS))
    axes[0].set_ylabel("Significant subclass–cCRE pairs")
    axes[0].set_title(
        f"{reference_label}\nShared T7 ≥ {t7_threshold:g}; {call_label}",
        fontsize=9.5,
    )
    maximum_value = max(max(values), 1)
    axes[0].set_ylim(0, maximum_value * 1.15)
    for index, value in enumerate(values):
        axes[0].text(
            index,
            value + maximum_value * 0.02,
            f"{value:,}",
            ha="center",
            va="bottom",
        )

    matrix = np.asarray(
        [
            [metrics["neither_significant"], metrics["new_only_significant"]],
            [metrics["origin_only_significant"], metrics["both_significant"]],
        ],
        dtype=float,
    )
    image = axes[1].imshow(matrix, cmap="Blues")
    axes[1].set_xticks([0, 1], ["New not sig.", "New sig."])
    axes[1].set_yticks([0, 1], ["Original not sig.", "Original sig."])
    axes[1].set_title(
        f"n={n_pairs:,}; call concordance={metrics['call_concordance']:.3f}\n"
        f"Significant-pair Jaccard={metrics['significant_jaccard']:.3f}",
        fontsize=9.5,
    )
    for row in range(2):
        for column in range(2):
            axes[1].text(
                column,
                row,
                f"{matrix[row, column]:,.0f}",
                ha="center",
                va="center",
                color="white" if matrix[row, column] > matrix.max() / 2 else "black",
            )
    fig.colorbar(image, ax=axes[1], fraction=0.046, pad=0.04)
    for suffix in ("png", "pdf"):
        fig.savefig(output_stem.with_suffix(f".{suffix}"), dpi=300)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if not 0.0 < args.cutoff < 1.0:
        raise ValueError("--cutoff must lie strictly between 0 and 1")
    tables_dir = args.comparison_dir / "tables"
    figures_dir = args.comparison_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    source_path, reference_label, prefix = resolve_source(
        args.source, args.comparison_dir, args.t7_threshold
    )
    pairs = read_pairs(source_path)
    calls = pairs[KEY].copy()
    for dataset in ("origin", "new"):
        calls[f"{dataset}_p_right"] = pairs[f"{dataset}_p_right"]
        calls[f"{dataset}_q_bh"] = bh_adjust(pairs[f"{dataset}_p_right"])
    validate_bh(calls, pairs, args.cutoff, source_path)

    payload: dict[str, object] = {
        "source": args.source,
        "source_table": str(source_path),
        "control_reference": reference_label,
        "t7_threshold": float(args.t7_threshold),
        "cutoff": float(args.cutoff),
        "n_pairs": int(len(pairs)),
        "bases": {},
    }
    for basis in BASES:
        column = "p_right" if basis == "raw_p" else "q_bh"
        origin_calls = calls[f"origin_{column}"].le(args.cutoff)
        new_calls = calls[f"new_{column}"].le(args.cutoff)
        calls[f"origin_significant_{basis}"] = origin_calls
        calls[f"new_significant_{basis}"] = new_calls
        calls[f"{basis}_call_status"] = call_status(origin_calls, new_calls)
        metrics = call_metrics(origin_calls, new_calls)
        stem = f"{prefix}_significant_call_concordance_{basis}"
        plot_calls(
            metrics,
            figures_dir / stem,
            n_pairs=len(pairs),
            reference_label=reference_label,
            call_label=basis_label(basis, args.cutoff, len(pairs)),
            t7_threshold=args.t7_threshold,
        )
        payload["bases"][basis] = {
            "figure_stem": stem,
            "metrics": {key: float(value) for key, value in metrics.items()},
            "origin_significant_fraction": float(origin_calls.mean()),
            "new_significant_fraction": float(new_calls.mean()),
        }

    calls_path = tables_dir / f"{prefix}_significant_call_concordance_calls.csv.gz"
    calls.to_csv(calls_path, index=False)
    payload["calls_table"] = str(calls_path)
    (tables_dir / f"{prefix}_significant_call_concordance_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
