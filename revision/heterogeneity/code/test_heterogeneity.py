from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib.collections as mcollections
import numpy as np
import pandas as pd
import pytest

CODE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE_DIR))

import make_heterogeneity_plots as plots
from relabel import relabel_subclasses_from_obs


def test_relabel_subclasses_from_obs_preserves_untargeted_cells() -> None:
    subclasses = np.array(["A", "A", "A", "B", "B", "C"])
    supertypes = np.array(
        ["001 A_1", "002 A_2", "002 A_2", "003 B_1", "003 B_1", "004 C_1"]
    )

    labels, assignment, members = relabel_subclasses_from_obs(
        subclasses, supertypes, ["A", "B"]
    )

    assert labels.tolist() == ["A_1", "A_2", "A_2", "B_1", "B_1", "C"]
    assert members == {"A": ["A_1", "A_2"], "B": ["B_1"]}
    assert assignment.columns.tolist() == [
        "position",
        "original_subclass",
        "source_subgroup",
        "new_subclass",
    ]
    assert assignment["position"].tolist() == [0, 1, 2, 3, 4]


@pytest.mark.parametrize(
    ("subclasses", "supertypes", "message"),
    [
        (
            np.array(["A", "A"], dtype=object),
            np.array(["001 A_1", None], dtype=object),
            "missing subgroup labels",
        ),
        (
            np.array(["A", "A"], dtype=object),
            np.array(["001 A_1", "999 A_1"], dtype=object),
            "collapse after standardization",
        ),
        (
            np.array(["A", "B"], dtype=object),
            np.array(["001 Shared", "001 Shared"], dtype=object),
            "multiple target subclasses",
        ),
        (
            np.array(["A", "C"], dtype=object),
            np.array(["001 C", "002 C_1"], dtype=object),
            "collide with untouched subclasses",
        ),
    ],
)
def test_relabel_subclasses_from_obs_rejects_invalid_annotations(
    subclasses: np.ndarray, supertypes: np.ndarray, message: str
) -> None:
    targets = sorted(set(subclasses) - {"C"})
    with pytest.raises(ValueError, match=message):
        relabel_subclasses_from_obs(subclasses, supertypes, targets)


def agreement_fixture() -> tuple[
    pd.DataFrame, pd.DataFrame, list[str], dict[str, list[str]]
]:
    combined = pd.DataFrame(
        [[1.0, 3.0], [2.0, 4.0]],
        index=["A", "B"],
        columns=["c1", "c2"],
    )
    split = pd.DataFrame(
        [[0.0, 2.0], [2.0, 4.0], [2.5, 4.5]],
        index=["A_1", "A_2", "B_1"],
        columns=["c1", "c2"],
    )
    targets = ["A", "B"]
    members = {"A": ["A_1", "A_2"], "B": ["B_1"]}
    return combined, split, targets, members


def test_random_agreement_remains_unweighted_and_backward_compatible() -> None:
    combined, split, targets, members = agreement_fixture()
    table = plots.bayesian_subset_agreement(
        combined, split, targets, members, excluded_cres=set()
    )

    a = table[table["cell_type"] == "A"].set_index("cre")
    assert a.loc["c1", "mean_subgroup_activity"] == pytest.approx(1.0)
    assert a.loc["c1", "subgroup_sd"] == pytest.approx(np.sqrt(2.0))
    assert a.loc["c1", "n_subgroups"] == 2

    b = table[table["cell_type"] == "B"].set_index("cre")
    assert b.loc["c1", "mean_subgroup_activity"] == pytest.approx(2.5)
    assert np.isnan(b.loc["c1", "subgroup_sd"])
    assert b.loc["c1", "n_subgroups"] == 1

    exported = plots.agreement_for_export(table, "random")
    assert exported.columns.tolist() == [
        "cell_type",
        "cre",
        "whole_activity",
        "mean_subset_activity",
        "subset_sd",
        "n_subsets",
        "difference",
        "absolute_difference",
    ]


def test_supertype_agreement_uses_cell_counts_but_retains_heterogeneity() -> None:
    combined, split, targets, members = agreement_fixture()
    weights = {"A": {"A_1": 1, "A_2": 3}, "B": {"B_1": 2}}
    table = plots.bayesian_subset_agreement(
        combined,
        split,
        targets,
        members,
        excluded_cres=set(),
        subgroup_weights=weights,
    )

    a = table[table["cell_type"] == "A"].set_index("cre")
    assert a.loc["c1", "mean_subgroup_activity"] == pytest.approx(1.5)
    assert a.loc["c1", "unweighted_mean_subgroup_activity"] == pytest.approx(1.0)
    assert a.loc["c1", "subgroup_sd"] == pytest.approx(np.sqrt(2.0))
    assert a.loc["c1", "difference"] == pytest.approx(0.5)
    assert a.loc["c1", "unweighted_difference"] == pytest.approx(0.0)

    b = table[table["cell_type"] == "B"].set_index("cre")
    assert b.loc["c1", "mean_subgroup_activity"] == pytest.approx(2.5)
    assert np.isnan(b.loc["c1", "subgroup_sd"])

    exported = plots.agreement_for_export(table, "supertype")
    assert {
        "cell_weighted_mean_supertype_activity",
        "unweighted_mean_supertype_activity",
        "supertype_sd",
        "n_supertypes",
        "cell_weighted_difference",
        "unweighted_difference",
    } <= set(exported.columns)

    summary = plots.summarize_agreement(
        table, include_supertype_heterogeneity=True
    ).set_index("cell_type")
    assert summary.loc["A", "median_supertype_sd"] == pytest.approx(np.sqrt(2.0))
    assert np.isnan(summary.loc["B", "median_supertype_sd"])


def test_pairwise_supertype_ccc_is_computed_within_each_parent() -> None:
    _, split, targets, members = agreement_fixture()
    weights = {"A": {"A_1": 1, "A_2": 3}, "B": {"B_1": 2}}

    pairwise = plots.pairwise_supertype_agreement(
        split, targets, members, weights, excluded_cres=set()
    )

    assert len(pairwise) == 1
    pair = pairwise.iloc[0]
    assert pair["cell_type"] == "A"
    assert {pair["supertype_1"], pair["supertype_2"]} == {"A_1", "A_2"}
    assert pair["n_cres"] == 2
    assert pair["concordance_correlation"] == pytest.approx(1.0 / 3.0)
    assert pair["pearson_r"] == pytest.approx(1.0)
    assert pair["minimum_pair_cells"] == 1

    summary = plots.summarize_pairwise_supertype_agreement(
        pairwise, targets, members
    ).set_index("cell_type")
    assert summary.loc["A", "n_pairs"] == 1
    assert summary.loc["A", "median_pairwise_ccc"] == pytest.approx(1.0 / 3.0)
    assert summary.loc["B", "n_pairs"] == 0
    assert np.isnan(summary.loc["B", "median_pairwise_ccc"])


def test_pairwise_ccc_plot_includes_mean_vs_whole_baseline(monkeypatch) -> None:
    _, split, targets, members = agreement_fixture()
    weights = {"A": {"A_1": 1, "A_2": 3}, "B": {"B_1": 2}}
    pairwise = plots.pairwise_supertype_agreement(
        split, targets, members, weights, excluded_cres=set()
    )
    pairwise_summary = plots.summarize_pairwise_supertype_agreement(
        pairwise, targets, members
    )
    whole_summary = pd.DataFrame(
        {
            "cell_type": ["A", "B"],
            "concordance_correlation": [0.9, 0.95],
        }
    )
    captured: dict[str, object] = {}

    def capture(fig, stem: Path) -> None:
        captured["fig"] = fig
        captured["stem"] = stem

    monkeypatch.setattr(plots, "save_figure", capture)
    plots.plot_pairwise_supertype_ccc(
        pairwise,
        pairwise_summary,
        whole_summary,
        targets,
        Path("unused"),
    )

    fig = captured["fig"]
    labels = {text.get_text() for text in fig.axes[0].texts}
    assert {"0.90", "0.95"} <= labels
    assert any("Blue diamonds" in text.get_text() for text in fig.texts)
    assert captured["stem"] == Path("unused/bayesian_supertype_pairwise_ccc")


def test_pairwise_ccc_cell_support_plot_uses_log_scale(monkeypatch) -> None:
    pairwise = pd.DataFrame(
        {
            "cell_type": ["A", "A", "B"],
            "minimum_pair_cells": [10, 100, 1000],
            "concordance_correlation": [0.1, 0.3, 0.5],
        }
    )
    captured: dict[str, object] = {}

    def capture(fig, stem: Path) -> None:
        captured["fig"] = fig
        captured["stem"] = stem

    monkeypatch.setattr(plots, "save_figure", capture)
    plots.plot_pairwise_ccc_vs_minimum_cells(
        pairwise, ["A", "B"], Path("unused")
    )

    fig = captured["fig"]
    assert fig.axes[0].get_xscale() == "log"
    assert "Spearman" in fig.axes[0].texts[0].get_text()
    assert captured["stem"] == Path(
        "unused/bayesian_supertype_pairwise_ccc_vs_min_cells"
    )


def test_supertype_plot_keeps_points_when_sd_is_undefined(monkeypatch) -> None:
    combined, split, targets, members = agreement_fixture()
    table = plots.bayesian_subset_agreement(
        combined,
        split,
        targets,
        members,
        excluded_cres=set(),
        subgroup_weights={"A": {"A_1": 1, "A_2": 2}, "B": {"B_1": 2}},
    )
    summary = plots.summarize_agreement(
        table, include_supertype_heterogeneity=True
    )
    captured: dict[str, object] = {}

    def capture(fig, stem: Path) -> None:
        captured["fig"] = fig
        captured["stem"] = stem

    monkeypatch.setattr(plots, "save_figure", capture)
    plots.plot_bayesian_subset_agreement(
        table,
        summary,
        targets,
        {"A": 3, "B": 2},
        members,
        "supertype",
        Path("unused"),
        ncols=2,
    )

    fig = captured["fig"]
    b_axis = fig.axes[1]
    scatters = [
        collection
        for collection in b_axis.collections
        if isinstance(collection, mcollections.PathCollection)
    ]
    error_bars = [
        collection
        for collection in b_axis.collections
        if isinstance(collection, mcollections.LineCollection)
    ]
    assert sum(len(scatter.get_offsets()) for scatter in scatters) == 2
    assert error_bars == []
    assert captured["stem"] == Path("unused/bayesian_supertype_mean_vs_whole")
    assert "cell-count-weighted mean" in fig._suptitle.get_text()
    assert any("between-supertype heterogeneity" in text.get_text() for text in fig.texts)


def test_split_targets_supports_legacy_and_explicit_manifests(tmp_path: Path) -> None:
    legacy = tmp_path / "legacy"
    legacy.mkdir()
    (legacy / "run_manifest.json").write_text(
        json.dumps({"split_subclasses": ["A"], "n_groups": 2})
    )
    assert plots.split_targets(legacy) == (
        ["A"],
        {"A": ["A_group_1", "A_group_2"]},
        "random",
    )

    annotated = tmp_path / "annotated"
    annotated.mkdir()
    (annotated / "run_manifest.json").write_text(
        json.dumps(
            {
                "split_subclasses": ["A"],
                "grouping": "supertype",
                "subgroups_by_subclass": {"A": ["A_1", "A_2"]},
            }
        )
    )
    assert plots.split_targets(annotated) == (
        ["A"],
        {"A": ["A_1", "A_2"]},
        "supertype",
    )


def test_bootstrap_negctrl_only_effect_matrix(tmp_path: Path) -> None:
    axes = {"subclasses": ["A_1", "A_2"], "cres": ["c1", "c2", "neg"]}
    (tmp_path / "bootstrap_axes.json").write_text(json.dumps(axes))
    pd.Series(["neg"], name="cre").to_csv(
        tmp_path / "negative_controls.csv", index=False
    )
    pd.DataFrame(False, index=axes["subclasses"], columns=axes["cres"]).to_csv(
        tmp_path / "qvalue_filter_mask.csv"
    )
    activity = np.array(
        [
            [[2.0, 4.0, 1.0], [3.0, 6.0, 1.5]],
            [[4.0, 8.0, 2.0], [6.0, 12.0, 3.0]],
        ]
    )
    np.save(tmp_path / "celltype_activity_array.npy", activity)

    observed = plots.bootstrap_effect_matrix(tmp_path, self_cre=False)
    mean_log = np.log(activity).mean(axis=0)
    expected = mean_log - mean_log[:, [2]]

    np.testing.assert_allclose(observed.to_numpy(), expected)


def test_bootstrap_artifact_names_are_separate_from_bayesian() -> None:
    bootstrap = plots.artifact_names("supertype", "bootstrap")
    bayesian = plots.artifact_names("supertype", "bayesian")
    assert bootstrap["figure"] == "bootstrap_supertype_mean_vs_whole"
    assert bootstrap["table"] == "bootstrap_supertype_vs_whole.csv"
    assert bootstrap["manifest"] != bayesian["manifest"]
    assert bootstrap["pairwise_figure"] == "bootstrap_supertype_pairwise_ccc"
    assert bayesian["pairwise_table"] == "bayesian_supertype_pairwise_ccc.csv"
    assert bayesian["pairwise_support_figure"].endswith("_vs_min_cells")
