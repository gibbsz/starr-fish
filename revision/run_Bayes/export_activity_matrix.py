#!/usr/bin/env python3
"""Export one dataset's posterior as the canonical matrices and test table.

Every activity figure in the revision reads from here rather than reopening a
439 MB posterior. This script reduces one posterior once, on the full unfiltered
universe, and writes it next to the posterior it came from:

    <dataset>/tables/<stem>_activity_matrix.csv.gz
        subclass x target cCRE -- the value the heatmaps colour: posterior mean
        target ``log_gamma`` minus the posterior mean of the seven ordinary controls
    <dataset>/tables/<stem>_beta_t7_activity_matrix.csv.gz
        subclass x *every ordinary* cCRE, controls included -- posterior mean
        ``log_gamma`` minus ``mean(log(beta_t7))``, the scale the method-comparison
        figures plot
    <dataset>/tables/<stem>_p_value_matrix.csv.gz
        subclass x target cCRE, ``p_right``, unfiltered
    <dataset>/tables/<stem>_q_value_matrix_t7_ge<k>.csv.gz
        subclass x target cCRE, BH q over this dataset's own T7 >= k pairs, NaN outside
    <dataset>/tables/<stem>_target_cre_matrix.csv.gz
        subclass x target cCRE raw ``obsm["CRE"]`` totals
    <dataset>/tables/<stem>_target_t7_matrix.csv.gz
        subclass x target cCRE raw ``obsm["T7CRE"]`` totals, the source of every T7 mask
    <dataset>/tables/<stem>_negative_control_activity_matrix.csv.gz
        subclass x 7 controls, posterior-mean log_gamma, for the control-spread strip
    <dataset>/tables/<stem>_significance.csv.gz
        one row per pair: p_right, the own-universe BH q, and the effect interval
    <dataset>/tables/<stem>_matrix_manifest.json
        provenance

Two activity scales ship because they answer different questions. The centered
matrix is the biological effect against the controls measured in the same cells, so
it is what the heatmaps and the significance calls use. The beta_t7 matrix removes
only the global T7 scale factor, leaving each method's own offset intact, which is
what makes methods comparable on one axis; it keeps the control columns because
the method-comparison scatters plot them.

``p_right`` is universe-free, so it is stored as computed. BH ``q`` is not: it
depends on which pairs are in the family, so the shipped ``q_right_t7_ge<k>``
column is BH over this dataset's own T7 >= k pairs, and any figure plotting a
different universe -- a shared original/new universe, for instance -- must re-run
BH over that universe from ``p_right``.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
BVFC_CODE = REPO_ROOT / "revision" / "bayesian_vs_fold_change" / "code"
for path in (HERE, BVFC_CODE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from baystarrfish.data import read_grouped_counts  # noqa: E402
from baystarrfish.stats import bh_fdr, negative_control_test  # noqa: E402
from test_individual_negative_control_loo_empirical_fdr import POOLED_NAME  # noqa: E402
from activity_matrix_io import (  # noqa: E402
    ACTIVITY_COLUMN,
    DEFAULT_STEM,
    DEFAULT_Q_T7_THRESHOLD,
    TARGET_CRE_COLUMN,
    call_column_for,
    matrix_paths,
    q_column_for,
    stem_for,
)

N_ORDINARY_CONTROLS = 7
SIGNIFICANCE_COLUMNS = (
    "group",
    "cre",
    "class",
    "n_cells",
    TARGET_CRE_COLUMN,
    "target_t7_total",
    "negative_control_t7_total",
    "n_negative_controls",
    "activity_mean",
    "mean_negative_control_activity_mean",
    "negative_control_activity_sd_mean",
    "control_reference_activity_mean",
    ACTIVITY_COLUMN,
    "effect_vs_control_reference_mean",
    "effect_vs_control_reference_lo90",
    "effect_vs_control_reference_hi90",
    "posterior_probability_above_control_reference",
    "p_right",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bayes-dir",
        type=Path,
        required=True,
        help="Posterior directory holding *_posterior_samples.npz and run_manifest.json.",
    )
    parser.add_argument(
        "--h5ad",
        type=Path,
        required=True,
        help="Dataset the posterior was fitted on; supplies the T7 totals.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=None,
        help="Destination (default: the posterior directory's sibling tables/).",
    )
    parser.add_argument("--stem", default=DEFAULT_STEM)
    parser.add_argument(
        "--q-t7-threshold",
        type=float,
        default=DEFAULT_Q_T7_THRESHOLD,
        help="Own-dataset universe used for the shipped BH q column.",
    )
    parser.add_argument(
        "--control-sd-multiplier",
        type=float,
        default=0.0,
        help=(
            "Raise the control reference by k draw-wise SDs. 0 is the mean-control "
            "family; 1 writes the mean+1SD test table under a suffixed stem."
        ),
    )
    parser.add_argument("--q-cutoff", type=float, default=0.05)
    parser.add_argument(
        "--check-against",
        type=Path,
        default=None,
        help="Existing per-pair test table to cross-check the exported values against.",
    )
    return parser.parse_args()


def read_single_column(path: Path) -> list[str]:
    if not path.exists():
        return []
    return pd.read_csv(path).iloc[:, 0].astype(str).tolist()


@dataclass(frozen=True)
class Posterior:
    """The ordinary-cCRE slice of one fit, plus the global T7 scale factor."""

    log_gamma: np.ndarray
    groups: np.ndarray
    cre_names: np.ndarray
    mean_log_beta_t7: float
    posterior_path: Path
    scalar_path: Path


def load_posterior(bayes_dir: Path) -> Posterior:
    """Read one fit's ``log_gamma`` draws and ``beta_t7``, dropping the pooled control."""
    manifest_path = bayes_dir / "run_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing run manifest: {manifest_path}")
    tag = json.loads(manifest_path.read_text())["tag"]
    posterior_path = bayes_dir / f"{tag}_posterior_samples.npz"
    if not posterior_path.exists():
        raise FileNotFoundError(f"Missing posterior samples: {posterior_path}")
    scalar_path = bayes_dir / f"{tag}_scalar_samples.npz"
    if not scalar_path.exists():
        raise FileNotFoundError(f"Missing scalar samples: {scalar_path}")
    with np.load(posterior_path, allow_pickle=True) as posterior:
        groups = posterior["group_names"].astype(str)
        all_cre = posterior["cre_names"].astype(str)
        ordinary = all_cre != POOLED_NAME
        log_gamma = posterior["log_gamma"][:, :, ordinary].astype(np.float32)
    with np.load(scalar_path, allow_pickle=True) as scalars:
        beta_t7 = np.asarray(scalars["beta_t7"], dtype=float).reshape(-1)
    if not np.all(beta_t7 > 0):
        raise ValueError(f"Non-positive beta_t7 draws in {scalar_path}")
    return Posterior(
        log_gamma=log_gamma,
        groups=groups,
        cre_names=all_cre[ordinary],
        mean_log_beta_t7=float(np.log(beta_t7).mean()),
        posterior_path=posterior_path,
        scalar_path=scalar_path,
    )


@dataclass(frozen=True)
class Export:
    """Everything one posterior reduces to, before the own-universe BH pass."""

    tests: pd.DataFrame
    beta_t7_activity: pd.DataFrame
    control_activity: pd.DataFrame
    posterior: Posterior


def compute_full_tests(
    bayes_dir: Path, h5ad: Path, control_sd_multiplier: float
) -> Export:
    """Run the mean-negative-control test with no T7 filter at all."""
    posterior = load_posterior(bayes_dir)
    log_gamma, groups, cre_names = (
        posterior.log_gamma,
        posterior.groups,
        posterior.cre_names,
    )
    controls = read_single_column(bayes_dir / "negative_controls.csv")
    blacklist = set(read_single_column(bayes_dir / "cre_blacklist.csv"))

    control_indices = np.flatnonzero(np.isin(cre_names, controls))
    if len(control_indices) != N_ORDINARY_CONTROLS:
        raise ValueError(
            f"Expected {N_ORDINARY_CONTROLS} ordinary negative controls; "
            f"found {len(control_indices)} in {bayes_dir}"
        )
    target_indices = np.flatnonzero(
        ~np.isin(cre_names, controls) & ~np.isin(cre_names, list(blacklist))
    )
    # One pass over the H5AD for both species; the T7 totals drive the test and
    # every downstream T7 mask, the CRE totals ship as the raw count matrix.
    counts = read_grouped_counts(h5ad, groups, cre_names, keys=("CRE", "T7CRE"))
    tests = negative_control_test(
        log_gamma,
        groups,
        cre_names,
        target_indices,
        control_indices,
        counts.totals["T7CRE"],
        counts.group_classes,
        counts.group_cell_counts,
        0.0,
        0.0,
        None,
        "Joint+dropout mean controls (unfiltered export)",
        control_sd_multiplier,
    )
    tests[ACTIVITY_COLUMN] = (
        tests["activity_mean"].astype(float)
        - tests["mean_negative_control_activity_mean"].astype(float)
    )
    tests = attach_target_cre_totals(tests, counts.frame("CRE"))

    posterior_mean = log_gamma.astype(np.float64).mean(axis=0)
    # Controls stay in: this matrix carries no test, and the method-comparison
    # scatters plot the control columns alongside the targets.
    beta_t7_activity = (
        pd.DataFrame(
            posterior_mean - posterior.mean_log_beta_t7,
            index=pd.Index(groups, name="subclass"),
            columns=cre_names,
        )
        .sort_index(axis=0)
        .sort_index(axis=1)
    )
    control_activity = (
        pd.DataFrame(
            posterior_mean[:, control_indices],
            index=pd.Index(groups, name="subclass"),
            columns=cre_names[control_indices],
        )
        .sort_index(axis=0)
        .sort_index(axis=1)
    )
    return Export(
        tests=tests,
        beta_t7_activity=beta_t7_activity,
        control_activity=control_activity,
        posterior=posterior,
    )


def attach_target_cre_totals(
    tests: pd.DataFrame, cre_totals: pd.DataFrame
) -> pd.DataFrame:
    """Join the raw cCRE counts onto the tested pairs, asserting full coverage."""
    long_totals = cre_totals.stack(future_stack=True).reset_index()
    long_totals.columns = ["group", "cre", TARGET_CRE_COLUMN]
    merged = tests.merge(
        long_totals, on=["group", "cre"], how="left", validate="one_to_one"
    )
    if merged[TARGET_CRE_COLUMN].isna().any():
        missing = merged.loc[merged[TARGET_CRE_COLUMN].isna(), ["group", "cre"]]
        raise ValueError(
            f"No cCRE counts for {len(missing)} tested pairs, e.g. "
            f"{missing.head(3).to_dict('records')}"
        )
    return merged


def add_own_universe_q(
    tests: pd.DataFrame, t7_threshold: float, q_cutoff: float
) -> tuple[pd.DataFrame, str, str, int]:
    """BH over this dataset's own T7 >= threshold pairs; NaN outside it."""
    q_column = q_column_for(t7_threshold)
    call_column = call_column_for(t7_threshold)
    eligible = tests["target_t7_total"].astype(float).ge(t7_threshold) & tests[
        "negative_control_t7_total"
    ].astype(float).ge(t7_threshold)
    tests[q_column] = np.nan
    if eligible.any():
        tests.loc[eligible, q_column] = bh_fdr(
            tests.loc[eligible, "p_right"].to_numpy(float)
        )
    tests[call_column] = tests[q_column].le(q_cutoff).fillna(False)
    return tests, q_column, call_column, int(eligible.sum())


def to_matrix(tests: pd.DataFrame, column: str) -> pd.DataFrame:
    matrix = tests.pivot(index="group", columns="cre", values=column)
    matrix.index.name = "subclass"
    matrix.columns.name = None
    return matrix.sort_index(axis=0).sort_index(axis=1)


def cross_check(tests: pd.DataFrame, reference: Path) -> dict[str, object]:
    other = pd.read_csv(reference)
    other["centered"] = (
        other["activity_mean"].astype(float)
        - other["mean_negative_control_activity_mean"].astype(float)
    )
    merged = tests.merge(
        other[["group", "cre", "centered", "p_right"]].rename(
            columns={"p_right": "p_right_reference"}
        ),
        on=["group", "cre"],
        how="inner",
    )
    if merged.empty:
        raise ValueError(f"No shared pairs with {reference}")
    return {
        "reference": str(reference),
        "shared_pairs": int(len(merged)),
        "max_abs_activity_difference": float(
            np.abs(merged[ACTIVITY_COLUMN] - merged["centered"]).max()
        ),
        "max_abs_p_right_difference": float(
            np.abs(merged["p_right"] - merged["p_right_reference"]).max()
        ),
    }


def main() -> None:
    args = parse_args()
    outdir = args.outdir or args.bayes_dir.parent / "tables"
    outdir.mkdir(parents=True, exist_ok=True)
    paths = matrix_paths(
        outdir, args.stem, args.control_sd_multiplier, args.q_t7_threshold
    )

    export = compute_full_tests(args.bayes_dir, args.h5ad, args.control_sd_multiplier)
    tests, q_column, call_column, n_eligible = add_own_universe_q(
        export.tests, args.q_t7_threshold, args.q_cutoff
    )

    activity = to_matrix(tests, ACTIVITY_COLUMN)
    matrices = {
        "activity": activity,
        "beta_t7_activity": export.beta_t7_activity,
        "p_value": to_matrix(tests, "p_right"),
        "q_value": to_matrix(tests, q_column),
        "target_cre": to_matrix(tests, TARGET_CRE_COLUMN),
        "target_t7": to_matrix(tests, "target_t7_total"),
        "negative_control_activity": export.control_activity,
    }
    for key, matrix in matrices.items():
        matrix.to_csv(paths[key], float_format="%.10g")

    columns = [
        column for column in SIGNIFICANCE_COLUMNS if column in tests.columns
    ] + [q_column, call_column]
    (
        tests[columns]
        .rename(columns={"group": "subclass"})
        .sort_values(["subclass", "cre"], kind="stable")
        .to_csv(paths["significance"], index=False, float_format="%.10g")
    )

    manifest = {
        "stem": stem_for(args.stem, args.control_sd_multiplier),
        "bayes_dir": str(args.bayes_dir.resolve()),
        "posterior": str(export.posterior.posterior_path.resolve()),
        "scalar_samples": str(export.posterior.scalar_path.resolve()),
        "h5ad": str(args.h5ad.resolve()),
        "activity_definition": (
            "posterior mean target log_gamma minus the posterior mean of the seven "
            "ordinary negative controls; alpha is not subtracted"
        ),
        "beta_t7_activity_definition": (
            "posterior mean log_gamma minus mean(log(beta_t7)); spans every ordinary "
            "cCRE, negative controls included, and carries no test"
        ),
        "mean_log_beta_t7": export.posterior.mean_log_beta_t7,
        "count_matrix_definition": (
            "raw per-cell obsm CRE / T7CRE counts summed over the cells of each "
            "subclass; cells with an unassigned subclass_name or class_name excluded"
        ),
        "control_sd_multiplier": float(args.control_sd_multiplier),
        "t7_filter": "none; every fitted target pair is exported",
        "q_column": q_column,
        "q_universe": (
            "BH over this dataset's pairs with target and pooled-control T7 >= "
            f"{args.q_t7_threshold:g}; NaN outside that universe"
        ),
        "q_cutoff": float(args.q_cutoff),
        "p_right_definition": "posterior fraction of draw-wise contrasts <= 0",
        "shape": {
            "subclasses": int(activity.shape[0]),
            "cres": int(activity.shape[1]),
            "cres_including_negative_controls": int(export.beta_t7_activity.shape[1]),
            "pairs": int(len(tests)),
            "pairs_in_q_universe": n_eligible,
        },
        "outputs": {key: str(value) for key, value in paths.items() if key != "manifest"},
    }
    if args.check_against is not None:
        manifest["cross_check"] = cross_check(tests, args.check_against)
    paths["manifest"].write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
