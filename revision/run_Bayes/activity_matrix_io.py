#!/usr/bin/env python3
"""Naming and readers for the per-dataset activity matrices.

The matrices are written by ``export_activity_matrix.py`` into each dataset's
``tables/`` directory and are the single source of truth for every activity
heatmap. This module owns the file-name convention so the writer and the
plotting scripts cannot drift apart, and provides the two reads the plots need:
one dataset, and the shared universe of two datasets.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

DEFAULT_STEM = "subclass_cre"
ACTIVITY_COLUMN = "centered_activity_mean"
PAIR_KEY = ["subclass", "cre"]


def stem_for(stem: str, control_sd_multiplier: float) -> str:
    """Mean-control tables keep the bare stem; stricter nulls get a suffix."""
    if control_sd_multiplier <= 0:
        return stem
    return f"{stem}_mean_plus_{control_sd_multiplier:g}sd"


def matrix_paths(
    tables_dir: Path, stem: str = DEFAULT_STEM, control_sd_multiplier: float = 0.0
) -> dict[str, Path]:
    full = stem_for(stem, control_sd_multiplier)
    return {
        "activity": tables_dir / f"{full}_activity_matrix.csv.gz",
        "target_t7": tables_dir / f"{full}_target_t7_matrix.csv.gz",
        "negative_control_activity": (
            tables_dir / f"{full}_negative_control_activity_matrix.csv.gz"
        ),
        "significance": tables_dir / f"{full}_significance.csv.gz",
        "manifest": tables_dir / f"{full}_matrix_manifest.json",
    }


@dataclass(frozen=True)
class DatasetMatrices:
    """One dataset's exported activity matrix and its test table."""

    tables_dir: Path
    stem: str
    activity: pd.DataFrame
    target_t7: pd.DataFrame
    negative_control_activity: pd.DataFrame
    significance: pd.DataFrame

    @property
    def q_column(self) -> str:
        columns = [c for c in self.significance.columns if c.startswith("q_right_t7_ge")]
        if len(columns) != 1:
            raise ValueError(f"Expected exactly one shipped q column, found {columns}")
        return columns[0]

    @property
    def control_spread(self) -> pd.Series:
        """SD across the seven control posterior-mean activities, per subclass."""
        return self.negative_control_activity.std(axis=1, ddof=1)

    def long(self) -> pd.DataFrame:
        """One row per fitted pair, activity joined onto the test columns."""
        activity = (
            self.activity.stack(future_stack=True)
            .rename(ACTIVITY_COLUMN)
            .reset_index()
        )
        activity.columns = [*PAIR_KEY, ACTIVITY_COLUMN]
        activity = activity.dropna(subset=[ACTIVITY_COLUMN])
        tests = self.significance.drop(columns=[ACTIVITY_COLUMN], errors="ignore")
        return activity.merge(tests, on=PAIR_KEY, how="inner", validate="one_to_one")


def _read_matrix(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing exported matrix: {path}. Run revision/run_Bayes/"
            "submit_activity_matrices.slurm for this dataset first."
        )
    matrix = pd.read_csv(path, index_col=0)
    matrix.index = matrix.index.astype(str)
    matrix.index.name = "subclass"
    matrix.columns = matrix.columns.astype(str)
    return matrix


def load_dataset(
    tables_dir: Path, stem: str = DEFAULT_STEM, control_sd_multiplier: float = 0.0
) -> DatasetMatrices:
    paths = matrix_paths(tables_dir, stem, control_sd_multiplier)
    significance = pd.read_csv(paths["significance"])
    missing = sorted({*PAIR_KEY, "p_right"}.difference(significance.columns))
    if missing:
        raise ValueError(f"{paths['significance']} is missing columns: {missing}")
    significance["subclass"] = significance["subclass"].astype(str)
    significance["cre"] = significance["cre"].astype(str)
    return DatasetMatrices(
        tables_dir=tables_dir,
        stem=stem_for(stem, control_sd_multiplier),
        activity=_read_matrix(paths["activity"]),
        target_t7=_read_matrix(paths["target_t7"]),
        negative_control_activity=_read_matrix(paths["negative_control_activity"]),
        significance=significance,
    )


def eligible_pairs(dataset: DatasetMatrices, t7_threshold: float) -> pd.DataFrame:
    """Pairs whose target and pooled-control T7 both clear the threshold."""
    frame = dataset.long()
    keep = frame["target_t7_total"].astype(float).ge(t7_threshold)
    if "negative_control_t7_total" in frame.columns:
        keep &= frame["negative_control_t7_total"].astype(float).ge(t7_threshold)
    return frame.loc[keep].reset_index(drop=True)


def shared_universe(
    datasets: dict[str, DatasetMatrices], t7_threshold: float
) -> pd.DataFrame:
    """Inner-join two datasets on the pairs both keep at ``t7_threshold``.

    Columns are prefixed with each dataset's key, matching the ``origin_``/``new_``
    convention the comparison figures already use.
    """
    if len(datasets) != 2:
        raise ValueError("shared_universe compares exactly two datasets")
    merged: pd.DataFrame | None = None
    for key, dataset in datasets.items():
        frame = eligible_pairs(dataset, t7_threshold).rename(
            columns=lambda column: (
                column if column in PAIR_KEY else f"{key}_{column}"
            )
        )
        merged = frame if merged is None else merged.merge(frame, on=PAIR_KEY, how="inner")
    assert merged is not None
    if merged.empty:
        raise ValueError(f"No pairs survive T7 >= {t7_threshold:g} in both datasets")
    return merged.reset_index(drop=True)
