#!/usr/bin/env python3
"""Contract tests for the exported activity/count/significance matrices.

Two layers:

* Pure-function and synthetic-H5AD tests, which always run and are fast. They pin
  the file-name convention, the writer/reader agreement on the ``q`` column name,
  and the grouped-count aggregation semantics.
* Consistency tests against the real exported ``tables/`` directories, which skip
  when a directory has not been re-exported since the beta_t7/p/q/CRE matrices were
  added. They are the acceptance check for
  ``revision/run_Bayes/submit_activity_matrices.slurm`` and
  ``revision/Bayesian_ablation/submit_ablation_matrices.slurm``.

    pytest revision/run_Bayes/test_activity_matrix_export.py -v
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
BVFC_CODE = REPO_ROOT / "revision" / "bayesian_vs_fold_change" / "code"
for path in (HERE, BVFC_CODE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from baystarrfish.data import read_grouped_counts  # noqa: E402
from activity_matrix_io import (  # noqa: E402
    ACTIVITY_COLUMN,
    DEFAULT_STEM,
    TARGET_CRE_COLUMN,
    call_column_for,
    load_dataset,
    matrix_paths,
    q_column_for,
    stem_for,
    t7_token,
)
from export_activity_matrix import (  # noqa: E402
    add_own_universe_q,
    attach_target_cre_totals,
    to_matrix,
)

REVISION = REPO_ROOT / "revision"
#: Every directory the two export jobs write, and the stems each one carries.
#: The live ablation arms. ``bayesian_joint_dropout`` is a symlink to Bayes_OldData, so
#: including it checks the symlink resolves as well as the tables being valid. The
#: superseded hierarchical fits under Bayesian_ablation/archive/ are deliberately absent.
ABLATION_ARMS = (
    "bayesian_joint",
    "bayesian_joint_dropout",
    "bayesian_decoupled",
    "bayesian_decoupled_dropout",
)

EXPORTED = [
    (REVISION / "Bayes_OldData" / "tables", 0.0),
    (REVISION / "Bayes_OldData" / "tables", 1.0),
    (REVISION / "Bayes_NewData" / "tables", 0.0),
    (REVISION / "Bayes_NewData" / "tables", 1.0),
    *[
        (REVISION / "Bayesian_ablation" / arm / "tables", 0.0)
        for arm in ABLATION_ARMS
    ],
]


def _ids(cases: list[tuple[Path, float]]) -> list[str]:
    return [
        f"{tables.parent.name}/{stem_for(DEFAULT_STEM, sd)}" for tables, sd in cases
    ]


# --------------------------------------------------------------------------- #
# naming convention
# --------------------------------------------------------------------------- #


def test_t7_token_is_filename_safe():
    assert t7_token(50.0) == "50"
    assert t7_token(0.5) == "0p5"
    assert t7_token(100) == "100"


def test_q_filename_and_q_column_agree_on_the_token():
    paths = matrix_paths(Path("/tmp"), q_t7_threshold=50.0)
    assert paths["q_value"].name.endswith("_q_value_matrix_t7_ge50.csv.gz")
    assert q_column_for(50.0) == "q_right_t7_ge50"
    assert call_column_for(50.0) == "significant_q_t7_ge50"


def test_mean_plus_1sd_suffixes_every_output():
    plain = matrix_paths(Path("/tmp"), control_sd_multiplier=0.0)
    strict = matrix_paths(Path("/tmp"), control_sd_multiplier=1.0)
    assert set(plain) == set(strict)
    for key in plain:
        assert "mean_plus_1sd" not in plain[key].name
        assert "mean_plus_1sd" in strict[key].name


def test_every_declared_output_has_a_distinct_filename():
    paths = matrix_paths(Path("/tmp"))
    assert len({p.name for p in paths.values()}) == len(paths)


# --------------------------------------------------------------------------- #
# grouped counts, against a synthetic H5AD
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def synthetic_h5ad(tmp_path_factory) -> Path:
    """Three subclasses in two classes, four cCREs, one unassigned cell."""
    path = tmp_path_factory.mktemp("grouped") / "tiny.h5ad"
    # cell:            0   1   2   3   4   5
    subclass_codes = [0, 0, 1, 2, 2, -1]
    class_codes = [0, 0, 0, 1, 1, 0]
    cre = {
        "CRE001": [1.0, 2.0, 4.0, 8.0, 16.0, 999.0],
        "CRE002": [0.0, 0.0, 3.0, 0.0, 5.0, 999.0],
    }
    t7 = {
        "CRE001": [10.0, 20.0, 40.0, 80.0, 160.0, 999.0],
        "CRE002": [1.0, 1.0, 1.0, 1.0, 1.0, 999.0],
    }
    with h5py.File(path, "w") as handle:
        obs = handle.create_group("obs")
        sub = obs.create_group("subclass_name")
        sub.create_dataset(
            "categories",
            data=np.array(["001 Alpha", "002 Beta/Gamma", "003 Delta"], dtype="S"),
        )
        sub.create_dataset("codes", data=np.array(subclass_codes, dtype=np.int8))
        cls = obs.create_group("class_name")
        cls.create_dataset("categories", data=np.array(["01 Exc", "02 Inh"], dtype="S"))
        cls.create_dataset("codes", data=np.array(class_codes, dtype=np.int8))
        obsm = handle.create_group("obsm")
        for key, table in (("CRE", cre), ("T7CRE", t7)):
            group = obsm.create_group(key)
            for name, values in table.items():
                group.create_dataset(name, data=np.asarray(values))
    return path


def test_grouped_counts_sums_within_subclass(synthetic_h5ad):
    counts = read_grouped_counts(
        synthetic_h5ad, ["Alpha", "Beta-Gamma", "Delta"], ["CRE001", "CRE002"]
    )
    # Allen numeric prefix stripped, "/" replaced -- the label convention.
    assert list(counts.groups) == ["Alpha", "Beta-Gamma", "Delta"]
    np.testing.assert_allclose(
        counts.totals["CRE"], [[3.0, 0.0], [4.0, 3.0], [24.0, 5.0]]
    )
    np.testing.assert_allclose(
        counts.totals["T7CRE"], [[30.0, 2.0], [40.0, 1.0], [240.0, 2.0]]
    )


def test_grouped_counts_excludes_unassigned_cells(synthetic_h5ad):
    """The cell with subclass code -1 carries 999 and must not appear anywhere."""
    counts = read_grouped_counts(synthetic_h5ad, ["Alpha", "Delta"], ["CRE001"])
    assert counts.totals["CRE"].max() < 999.0
    assert counts.group_cell_counts.tolist() == [2, 2]


def test_grouped_counts_reports_group_classes(synthetic_h5ad):
    counts = read_grouped_counts(
        synthetic_h5ad, ["Alpha", "Beta-Gamma", "Delta"], ["CRE001"]
    )
    assert list(counts.group_classes) == ["Exc", "Exc", "Inh"]


def test_grouped_counts_honours_requested_axis_order(synthetic_h5ad):
    forward = read_grouped_counts(synthetic_h5ad, ["Alpha", "Delta"], ["CRE001", "CRE002"])
    reversed_ = read_grouped_counts(
        synthetic_h5ad, ["Delta", "Alpha"], ["CRE002", "CRE001"]
    )
    np.testing.assert_allclose(
        forward.totals["CRE"], reversed_.totals["CRE"][::-1, ::-1]
    )


def test_grouped_counts_rejects_unknown_axes(synthetic_h5ad):
    with pytest.raises(ValueError, match="missing requested subclasses"):
        read_grouped_counts(synthetic_h5ad, ["Nope"], ["CRE001"])
    with pytest.raises(ValueError, match="missing requested cCREs"):
        read_grouped_counts(synthetic_h5ad, ["Alpha"], ["CRE999"])


def test_grouped_counts_frame_is_labelled(synthetic_h5ad):
    frame = read_grouped_counts(synthetic_h5ad, ["Alpha"], ["CRE001"]).frame("CRE")
    assert frame.index.name == "subclass"
    assert frame.loc["Alpha", "CRE001"] == 3.0


# --------------------------------------------------------------------------- #
# the reductions the exporter applies
# --------------------------------------------------------------------------- #


def _tests_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "group": ["A", "A", "B", "B"],
            "cre": ["CRE001", "CRE002", "CRE001", "CRE002"],
            "target_t7_total": [100.0, 10.0, 60.0, 200.0],
            "negative_control_t7_total": [100.0, 100.0, 100.0, 100.0],
            "p_right": [0.001, 0.4, 0.02, 0.5],
        }
    )


def test_add_own_universe_q_is_nan_outside_the_universe():
    tests, q_column, call_column, n = add_own_universe_q(_tests_frame(), 50.0, 0.05)
    assert (q_column, call_column) == (q_column_for(50.0), call_column_for(50.0))
    assert n == 3
    # CRE002 in group A has target T7 = 10 < 50.
    outside = (tests["group"] == "A") & (tests["cre"] == "CRE002")
    assert tests.loc[outside, q_column].isna().all()
    assert not tests.loc[outside, call_column].any()
    assert tests.loc[~outside, q_column].notna().all()


def test_add_own_universe_q_bh_over_the_eligible_pairs_only():
    from baystarrfish.stats import bh_fdr

    tests, q_column, _, _ = add_own_universe_q(_tests_frame(), 50.0, 0.05)
    eligible = tests[q_column].notna()
    expected = bh_fdr(tests.loc[eligible, "p_right"].to_numpy(float))
    np.testing.assert_allclose(tests.loc[eligible, q_column].to_numpy(), expected)


def test_to_matrix_pivots_to_sorted_subclass_by_cre():
    matrix = to_matrix(_tests_frame(), "p_right")
    assert matrix.index.name == "subclass"
    assert list(matrix.index) == ["A", "B"]
    assert list(matrix.columns) == ["CRE001", "CRE002"]
    assert matrix.loc["B", "CRE001"] == 0.02


def test_attach_target_cre_totals_joins_on_the_pair_key():
    totals = pd.DataFrame(
        [[3.0, 7.0], [11.0, 13.0]],
        index=pd.Index(["A", "B"], name="subclass"),
        columns=["CRE001", "CRE002"],
    )
    merged = attach_target_cre_totals(_tests_frame(), totals)
    assert len(merged) == 4
    row = merged[(merged["group"] == "B") & (merged["cre"] == "CRE002")]
    assert row[TARGET_CRE_COLUMN].iloc[0] == 13.0


def test_attach_target_cre_totals_rejects_incomplete_coverage():
    totals = pd.DataFrame(
        [[3.0]], index=pd.Index(["A"], name="subclass"), columns=["CRE001"]
    )
    with pytest.raises(ValueError, match="No cCRE counts for 3 tested pairs"):
        attach_target_cre_totals(_tests_frame(), totals)


# --------------------------------------------------------------------------- #
# the real exports
# --------------------------------------------------------------------------- #


def _load(tables: Path, sd: float):
    manifest = matrix_paths(tables, DEFAULT_STEM, sd)["manifest"]
    if not manifest.exists():
        pytest.skip(f"not exported yet: {manifest}")
    if "mean_log_beta_t7" not in json.loads(manifest.read_text()):
        pytest.skip(f"pre-dates the beta_t7 export, re-run the export job: {manifest}")
    return load_dataset(tables, DEFAULT_STEM, sd), json.loads(manifest.read_text())


@pytest.mark.parametrize(("tables", "sd"), EXPORTED, ids=_ids(EXPORTED))
def test_matrices_share_the_subclass_axis(tables: Path, sd: float):
    dataset, _ = _load(tables, sd)
    reference = dataset.activity.index
    for name in ("beta_t7_activity", "p_value", "q_value", "target_cre", "target_t7"):
        assert getattr(dataset, name).index.equals(reference), name


@pytest.mark.parametrize(("tables", "sd"), EXPORTED, ids=_ids(EXPORTED))
def test_target_matrices_share_the_cre_axis_and_beta_t7_adds_the_controls(
    tables: Path, sd: float
):
    dataset, _ = _load(tables, sd)
    targets = dataset.activity.columns
    for name in ("p_value", "q_value", "target_cre", "target_t7"):
        assert getattr(dataset, name).columns.equals(targets), name
    extra = set(dataset.beta_t7_activity.columns) - set(targets)
    assert extra == set(dataset.negative_control_activity.columns)
    assert len(extra) == 7


@pytest.mark.parametrize(("tables", "sd"), EXPORTED, ids=_ids(EXPORTED))
def test_p_value_matrix_matches_the_long_table(tables: Path, sd: float):
    dataset, _ = _load(tables, sd)
    long = dataset.significance.set_index(["subclass", "cre"])["p_right"]
    flat = dataset.p_value.stack(future_stack=True).dropna()
    assert len(flat) == len(long)
    np.testing.assert_allclose(
        flat.sort_index().to_numpy(), long.sort_index().to_numpy(), atol=1e-12
    )


@pytest.mark.parametrize(("tables", "sd"), EXPORTED, ids=_ids(EXPORTED))
def test_q_value_matrix_is_populated_exactly_on_the_t7_universe(tables: Path, sd: float):
    dataset, _ = _load(tables, sd)
    long = dataset.significance.set_index(["subclass", "cre"])
    threshold = float(dataset.q_column.removeprefix("q_right_t7_ge").replace("p", "."))
    eligible = long["target_t7_total"].astype(float).ge(threshold) & long[
        "negative_control_t7_total"
    ].astype(float).ge(threshold)
    populated = dataset.q_value.stack(future_stack=True).dropna().index
    assert set(populated) == set(long.index[eligible])
    np.testing.assert_allclose(
        dataset.q_value.stack(future_stack=True).dropna().sort_index().to_numpy(),
        long.loc[eligible, dataset.q_column].sort_index().to_numpy(),
        atol=1e-12,
    )


@pytest.mark.parametrize(("tables", "sd"), EXPORTED, ids=_ids(EXPORTED))
def test_count_matrices_match_the_long_table(tables: Path, sd: float):
    dataset, _ = _load(tables, sd)
    long = dataset.significance.set_index(["subclass", "cre"])
    for matrix, column in (
        (dataset.target_t7, "target_t7_total"),
        (dataset.target_cre, TARGET_CRE_COLUMN),
    ):
        flat = matrix.stack(future_stack=True).dropna().sort_index()
        np.testing.assert_allclose(
            flat.to_numpy(), long[column].astype(float).sort_index().to_numpy(), atol=0.5
        )


@pytest.mark.parametrize(("tables", "sd"), EXPORTED, ids=_ids(EXPORTED))
def test_count_matrices_are_non_negative_integers(tables: Path, sd: float):
    dataset, _ = _load(tables, sd)
    for matrix in (dataset.target_cre, dataset.target_t7):
        values = matrix.to_numpy(float)
        finite = values[np.isfinite(values)]
        assert (finite >= 0).all()
        np.testing.assert_allclose(finite, np.round(finite), atol=1e-6)


@pytest.mark.parametrize(("tables", "sd"), EXPORTED, ids=_ids(EXPORTED))
def test_activity_matrix_is_the_control_centered_contrast(tables: Path, sd: float):
    dataset, _ = _load(tables, sd)
    long = dataset.significance
    expected = long["activity_mean"].astype(float) - long[
        "mean_negative_control_activity_mean"
    ].astype(float)
    np.testing.assert_allclose(long[ACTIVITY_COLUMN].astype(float), expected, atol=1e-9)
    flat = dataset.activity.stack(future_stack=True).dropna().sort_index()
    np.testing.assert_allclose(
        flat.to_numpy(),
        long.set_index(["subclass", "cre"])[ACTIVITY_COLUMN]
        .astype(float)
        .sort_index()
        .to_numpy(),
        atol=1e-9,
    )


@pytest.mark.parametrize(("tables", "sd"), EXPORTED, ids=_ids(EXPORTED))
def test_beta_t7_activity_is_the_centered_activity_up_to_a_per_subclass_shift(
    tables: Path, sd: float
):
    """The two scales differ by the control mean, which is constant within a subclass."""
    dataset, _ = _load(tables, sd)
    targets = dataset.activity.columns
    difference = dataset.beta_t7_activity[targets] - dataset.activity
    spread = difference.max(axis=1) - difference.min(axis=1)
    assert float(spread.max()) < 1e-5


def test_beta_t7_activity_reproduces_the_posterior():
    """Recompute log_gamma.mean(0) - log(beta_t7).mean() straight from the .npz."""
    bayes_dir = REVISION / "Bayes_OldData" / "bayesian"
    tables = REVISION / "Bayes_OldData" / "tables"
    dataset, manifest = _load(tables, 0.0)
    tag = json.loads((bayes_dir / "run_manifest.json").read_text())["tag"]
    with np.load(bayes_dir / f"{tag}_posterior_samples.npz", allow_pickle=True) as npz:
        groups = npz["group_names"].astype(str)
        cres = npz["cre_names"].astype(str)
        mean_log_gamma = npz["log_gamma"].astype(np.float64).mean(axis=0)
    with np.load(bayes_dir / f"{tag}_scalar_samples.npz", allow_pickle=True) as npz:
        correction = float(np.log(np.asarray(npz["beta_t7"], float).reshape(-1)).mean())
    assert correction == pytest.approx(manifest["mean_log_beta_t7"], abs=1e-12)
    expected = (
        pd.DataFrame(mean_log_gamma - correction, index=groups, columns=cres)
        .sort_index(axis=0)
        .sort_index(axis=1)
    )
    got = dataset.beta_t7_activity
    expected = expected.loc[got.index, got.columns]
    np.testing.assert_allclose(got.to_numpy(), expected.to_numpy(), atol=1e-6)


# --------------------------------------------------------------------------- #
# end-to-end: the CLI on a synthetic fit
# --------------------------------------------------------------------------- #

N_SYNTHETIC_CONTROLS = 7
SYNTHETIC_TARGETS = ("CRE001", "CRE002", "CRE003", "CRE004")
SYNTHETIC_CONTROLS = tuple(f"CRE9{i:02d}" for i in range(N_SYNTHETIC_CONTROLS))
SYNTHETIC_GROUPS = ("Alpha", "Beta-Gamma", "Delta")


@pytest.fixture(scope="module")
def synthetic_fit(tmp_path_factory) -> tuple[Path, Path, Path]:
    """A miniature (bayes_dir, h5ad, outdir) the real CLI can run against.

    Small enough to be a unit test, complete enough that every branch of the
    exporter runs: 7 ordinary controls, 4 targets, one blacklisted cCRE present in
    ``cre_blacklist.csv`` but absent from the posterior (the real situation), and a
    T7 spread that puts some pairs inside the q universe and some outside.
    """
    root = tmp_path_factory.mktemp("fit")
    bayes_dir = root / "bayesian"
    bayes_dir.mkdir()
    cres = np.array(SYNTHETIC_TARGETS + SYNTHETIC_CONTROLS)
    groups = np.array(SYNTHETIC_GROUPS)
    rng = np.random.default_rng(0)
    n_draws = 200
    # Targets sit above the controls, by a margin that grows across the target axis.
    means = np.zeros((len(groups), len(cres)))
    means[:, : len(SYNTHETIC_TARGETS)] = np.array([0.0, 0.5, 1.0, 2.0])
    log_gamma = means[None, :, :] + 0.2 * rng.standard_normal(
        (n_draws, len(groups), len(cres))
    )
    np.savez(
        bayes_dir / "synthetic_svi_posterior_samples.npz",
        log_gamma=log_gamma.astype(np.float32),
        group_names=groups,
        cre_names=cres,
    )
    np.savez(
        bayes_dir / "synthetic_svi_scalar_samples.npz",
        beta_t7=np.full(n_draws, 0.25) * np.exp(0.01 * rng.standard_normal(n_draws)),
    )
    (bayes_dir / "run_manifest.json").write_text(json.dumps({"tag": "synthetic_svi"}))
    pd.DataFrame({"cre": list(SYNTHETIC_CONTROLS)}).to_csv(
        bayes_dir / "negative_controls.csv", index=False
    )
    # Present in the blacklist, absent from the posterior -- as in the real fits.
    pd.DataFrame({"cre": ["CRE777"]}).to_csv(
        bayes_dir / "cre_blacklist.csv", index=False
    )

    h5ad = root / "synthetic.h5ad"
    n_cells = 300
    codes = np.repeat(np.arange(len(groups)), n_cells // len(groups)).astype(np.int8)
    with h5py.File(h5ad, "w") as handle:
        obs = handle.create_group("obs")
        sub = obs.create_group("subclass_name")
        sub.create_dataset("categories", data=np.array(list(groups), dtype="S"))
        sub.create_dataset("codes", data=codes)
        cls = obs.create_group("class_name")
        cls.create_dataset("categories", data=np.array(["Exc", "Inh"], dtype="S"))
        cls.create_dataset("codes", data=(codes == 2).astype(np.int8))
        obsm = handle.create_group("obsm")
        for key, scale in (("CRE", 1.0), ("T7CRE", 4.0)):
            group = obsm.create_group(key)
            for index, cre in enumerate(cres):
                # cCRE 0 stays under the T7 >= 50 universe; the rest clear it.
                per_cell = np.full(n_cells, 0.02 if index == 0 else 1.0) * scale
                group.create_dataset(cre, data=per_cell)
    outdir = root / "tables"
    return bayes_dir, h5ad, outdir


@pytest.fixture(scope="module")
def synthetic_export(synthetic_fit) -> tuple[dict, Path]:
    """Run the real CLI once and return (manifest, outdir)."""
    import subprocess

    bayes_dir, h5ad, outdir = synthetic_fit
    result = subprocess.run(
        [
            sys.executable,
            str(HERE / "export_activity_matrix.py"),
            "--bayes-dir", str(bayes_dir),
            "--h5ad", str(h5ad),
            "--outdir", str(outdir),
            "--q-t7-threshold", "50",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout), outdir


def test_cli_writes_every_declared_output(synthetic_export):
    manifest, outdir = synthetic_export
    paths = matrix_paths(outdir, DEFAULT_STEM, 0.0, 50.0)
    for key, path in paths.items():
        assert path.exists(), f"{key} -> {path}"
    assert set(manifest["outputs"]) == set(paths) - {"manifest"}


def test_cli_manifest_records_both_activity_scales(synthetic_export):
    manifest, _ = synthetic_export
    assert "activity_definition" in manifest
    assert "beta_t7_activity_definition" in manifest
    # beta_t7 = 0.25 -> log(0.25) ~ -1.386
    assert manifest["mean_log_beta_t7"] == pytest.approx(np.log(0.25), abs=0.01)
    assert manifest["shape"]["cres"] == len(SYNTHETIC_TARGETS)
    assert manifest["shape"]["cres_including_negative_controls"] == len(
        SYNTHETIC_TARGETS
    ) + N_SYNTHETIC_CONTROLS


def test_cli_output_is_readable_by_load_dataset(synthetic_export):
    _, outdir = synthetic_export
    dataset = load_dataset(outdir)
    assert list(dataset.activity.index) == sorted(SYNTHETIC_GROUPS)
    assert list(dataset.activity.columns) == sorted(SYNTHETIC_TARGETS)
    assert list(dataset.beta_t7_activity.columns) == sorted(
        SYNTHETIC_TARGETS + SYNTHETIC_CONTROLS
    )
    assert dataset.negative_control_activity.shape == (
        len(SYNTHETIC_GROUPS),
        N_SYNTHETIC_CONTROLS,
    )


def test_cli_q_universe_excludes_the_low_t7_cre(synthetic_export):
    """CRE001 was given 1/50th the T7 of the others and must fall out of the BH family."""
    manifest, outdir = synthetic_export
    dataset = load_dataset(outdir)
    assert dataset.q_value["CRE001"].isna().all()
    assert dataset.q_value.drop(columns=["CRE001"]).notna().all().all()
    assert manifest["shape"]["pairs_in_q_universe"] == len(SYNTHETIC_GROUPS) * (
        len(SYNTHETIC_TARGETS) - 1
    )


def test_cli_recovers_the_planted_effect_ordering(synthetic_export):
    """Activity was planted as 0 < 0.5 < 1 < 2 above the controls; recover that order."""
    _, outdir = synthetic_export
    activity = load_dataset(outdir).activity
    ordered = activity.mean(axis=0)[["CRE001", "CRE002", "CRE003", "CRE004"]]
    assert list(ordered.sort_values().index) == ["CRE001", "CRE002", "CRE003", "CRE004"]
    np.testing.assert_allclose(ordered.to_numpy(), [0.0, 0.5, 1.0, 2.0], atol=0.1)


def test_cli_count_matrices_match_the_planted_counts(synthetic_export):
    _, outdir = synthetic_export
    dataset = load_dataset(outdir)
    # 100 cells per subclass, 1.0 CRE and 4.0 T7 per cell, except CRE001 at 1/50th.
    assert dataset.target_cre.loc["Alpha", "CRE002"] == pytest.approx(100.0)
    assert dataset.target_t7.loc["Alpha", "CRE002"] == pytest.approx(400.0)
    assert dataset.target_cre.loc["Alpha", "CRE001"] == pytest.approx(2.0)
    assert dataset.target_t7.loc["Alpha", "CRE001"] == pytest.approx(8.0)


def test_cli_mean_plus_1sd_writes_a_separate_stem(synthetic_fit, synthetic_export):
    import subprocess

    bayes_dir, h5ad, outdir = synthetic_fit
    result = subprocess.run(
        [
            sys.executable,
            str(HERE / "export_activity_matrix.py"),
            "--bayes-dir", str(bayes_dir),
            "--h5ad", str(h5ad),
            "--outdir", str(outdir),
            "--q-t7-threshold", "50",
            "--control-sd-multiplier", "1",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    strict = load_dataset(outdir, DEFAULT_STEM, 1.0)
    plain = load_dataset(outdir, DEFAULT_STEM, 0.0)
    # Same activity matrix; a stricter null can only raise p_right.
    pd.testing.assert_frame_equal(strict.activity, plain.activity)
    assert (strict.p_value.to_numpy() >= plain.p_value.to_numpy() - 1e-12).all()


# --------------------------------------------------------------------------- #
# the ablation arms share the production parameterisation
#
# These are the checks that would have caught the class of error this suite exists
# for: four arms that looked interchangeable with the production fit, and were not,
# because they used a different activity parameterisation with the negative controls
# collapsed onto one shared trajectory. Manifest fields and control spread are both
# asserted, because either alone can be misleading -- a manifest can record a flag the
# model never applied, and spread alone does not say which parameterisation ran.
# --------------------------------------------------------------------------- #

ABLATION_DIR = REVISION / "Bayesian_ablation"
HIERARCHICAL_SCALAR_SITES = frozenset(
    {"mu_alpha", "sigma_alpha", "sigma_eta", "sigma_delta", "alpha_neg", "log_gamma_neg"}
)


def _arm_manifest(arm: str) -> dict:
    path = ABLATION_DIR / arm / "bayesian" / "run_manifest.json"
    if not path.exists():
        pytest.skip(f"not fitted yet: {path}")
    return json.loads(path.read_text())


@pytest.mark.parametrize("arm", ABLATION_ARMS)
def test_arm_records_the_direct_parameterisation(arm: str):
    config = _arm_manifest(arm)["config"]
    assert config.get("activity_model") == "direct", (
        f"{arm} was fitted with activity_model={config.get('activity_model')!r}; "
        "the arms must share the production parameterisation to be comparable"
    )


@pytest.mark.parametrize("arm", ABLATION_ARMS)
def test_arm_treats_negative_controls_as_ordinary(arm: str):
    """No cCRE may be a shared in-model control; direct activity forbids the mask."""
    config = _arm_manifest(arm)["config"]
    assert not config.get("negative_control_cre"), (
        f"{arm} pooled {config.get('negative_control_cre')} in-model"
    )
    mode = config.get("negative_control_mode")
    assert mode in (None, "ordinary"), f"{arm} used negative_control_mode={mode!r}"


@pytest.mark.parametrize("arm", ABLATION_ARMS)
def test_arm_saved_direct_scalars_and_no_hierarchical_ones(arm: str):
    manifest = _arm_manifest(arm)
    tag = manifest["tag"]
    with np.load(ABLATION_DIR / arm / "bayesian" / f"{tag}_scalar_samples.npz") as handle:
        sites = set(handle.files)
    assert {"mu_gamma", "sigma_gamma"} <= sites, (
        f"{arm} is missing the direct activity scalars; saved: {sorted(sites)}"
    )
    leaked = sites & HIERARCHICAL_SCALAR_SITES
    assert not leaked, f"{arm} carries hierarchical sites {sorted(leaked)}"


@pytest.mark.parametrize("arm", ABLATION_ARMS)
def test_arm_negative_controls_have_real_spread(arm: str):
    """Free controls vary; pooled ones are identical to floating-point noise.

    The hierarchical arms scored exactly 0.000 here, which is what made the
    mean+1SD null degenerate and the two fits incomparable.
    """
    tables = ABLATION_DIR / arm / "tables"
    path = matrix_paths(tables, DEFAULT_STEM, 0.0)["negative_control_activity"]
    if not path.exists():
        pytest.skip(f"not exported yet: {path}")
    controls = pd.read_csv(path, index_col=0)
    spread = (controls.max(axis=1) - controls.min(axis=1)).median()
    assert spread > 0.1, (
        f"{arm} median across-control range is {spread:.3e}; the controls are pooled, "
        "not free -- this arm is not the production parameterisation"
    )


def test_archive_is_not_reachable_through_the_resolver():
    """The superseded hierarchical fits must not be loadable as live arms."""
    sys.path.insert(0, str(BVFC_CODE))
    from analysis_utils import RELOCATED_ABLATION_ARMS, ablation_root

    assert set(ABLATION_ARMS) == set(RELOCATED_ABLATION_ARMS)
    for retired in ("bayesian_decoupled_no_dropout", "nonsense_arm"):
        with pytest.raises(KeyError):
            ablation_root(retired)
