"""CountData validation, path overriding, and the negative-control modes.

No test here touches the 3.3 GB input: the parts that need it are covered by the
golden-diff run in README_BAYSTARRFISH.md. What is covered here is everything
that can silently mis-assemble the arrays before the model ever sees them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from baystarrfish.data import CountData, paths
from baystarrfish.data.controls import (
    POOLED_NEGATIVE_CONTROL_NAME,
    build_pooled_negative_control,
)

N_CELL, N_CRE = 30, 4
CRE_NAMES = [f"CRE{i:03d}" for i in range(1, N_CRE + 1)]


def _data(rng, **overrides):
    kwargs = dict(
        t7=rng.poisson(0.3, (N_CELL, N_CRE)),
        cre=rng.poisson(0.1, (N_CELL, N_CRE)),
        subclass=np.array(["a", "b", "c"] * (N_CELL // 3)),
        class_=np.array(["A", "A", "B"] * (N_CELL // 3)),
        lib_size_log=np.log1p(rng.poisson(50, N_CRE)),
        cre_names=list(CRE_NAMES),
        negative_control_mask=None,
        negative_controls=[CRE_NAMES[-1]],
        negative_control_mode="ordinary",
    )
    kwargs.update(overrides)
    return CountData(**kwargs)


def test_shapes_are_reported(rng):
    data = _data(rng)
    assert (data.n_cells, data.n_cre) == (N_CELL, N_CRE)
    assert (data.n_subclasses, data.n_classes) == (3, 2)


@pytest.mark.parametrize(
    "override, message",
    [
        ({"cre": np.zeros((N_CELL, N_CRE + 1))}, "equal shape"),
        ({"cre_names": CRE_NAMES[:-1]}, "cCRE names"),
        ({"lib_size_log": np.zeros(N_CRE + 2)}, "lib_size_log"),
        ({"subclass": np.array(["a"])}, "subclass"),
        ({"negative_control_mask": np.zeros(N_CRE + 1, bool)}, "negative_control_mask"),
        ({"negative_control_mode": "nonsense"}, "negative_control_mode"),
    ],
)
def test_mismatched_inputs_are_rejected_at_construction(rng, override, message):
    with pytest.raises(ValueError, match=message):
        _data(rng, **override)


def test_to_run_kwargs_matches_the_fit_signature(rng):
    import inspect

    from baystarrfish.inference.run import run_model

    kwargs = _data(rng).to_run_kwargs()
    parameters = inspect.signature(run_model).parameters
    assert set(kwargs) <= set(parameters)
    for required in ("t7", "cre", "subclass_labels", "class_labels",
                     "lib_size_log", "cre_names"):
        assert required in kwargs


def test_side_tables_cover_what_write_fit_records(rng):
    data = _data(
        rng,
        cre_info=pd.DataFrame({"labeling_type": ["x"] * N_CRE}, index=CRE_NAMES),
        subclass_cell_counts=pd.Series([10, 10, 10], index=["a", "b", "c"], name="n_cells"),
        blacklist=["CRE099"],
    )
    assert set(data.side_tables()) == {
        "cre_info", "cre_blacklist", "negative_controls", "subclass_cell_counts"
    }


def test_pooled_control_sums_the_constituents_and_masks_only_the_new_column(rng):
    t7 = rng.poisson(1.0, (N_CELL, N_CRE))
    cre = rng.poisson(1.0, (N_CELL, N_CRE))
    counts = np.array([10.0, 20.0, 30.0, 40.0])
    mask = np.array([False, False, True, True])
    t7_out, cre_out, counts_out, names, model_mask, provenance = (
        build_pooled_negative_control(
            t7, cre, counts, list(CRE_NAMES), mask, [CRE_NAMES[2], CRE_NAMES[3]]
        )
    )
    assert names[-1] == POOLED_NEGATIVE_CONTROL_NAME
    np.testing.assert_array_equal(t7_out[:, -1], t7[:, mask].sum(axis=1))
    np.testing.assert_array_equal(cre_out[:, -1], cre[:, mask].sum(axis=1))
    assert counts_out[-1] == 70.0
    # Only the appended column is pooled in-model; the originals stay ordinary.
    assert model_mask[-1] and not model_mask[:-1].any()
    assert provenance["constituent_cre"] == [CRE_NAMES[2], CRE_NAMES[3]]


def test_pooled_control_refuses_when_there_are_no_controls(rng):
    with pytest.raises(ValueError, match="without negative controls"):
        build_pooled_negative_control(
            np.zeros((2, N_CRE)), np.zeros((2, N_CRE)), np.zeros(N_CRE),
            list(CRE_NAMES), np.zeros(N_CRE, bool), [],
        )


def test_paths_honour_the_environment_override(monkeypatch, tmp_path):
    monkeypatch.setenv(paths.ENV_REPO_ROOT, str(tmp_path))
    assert paths.repo_root() == tmp_path
    assert paths.starrfish_root() == tmp_path / "STARRFISH_in_vivo"
    monkeypatch.setenv(paths.ENV_DATA_ROOT, str(tmp_path / "elsewhere"))
    assert paths.libsize_csv().parent == tmp_path / "elsewhere"
    monkeypatch.setenv(paths.ENV_H5AD, str(tmp_path / "input.h5ad"))
    assert paths.default_h5ad() == tmp_path / "input.h5ad"


def test_paths_are_evaluated_on_access_not_at_import(monkeypatch, tmp_path):
    """The legacy upper-case names must follow the override too."""
    monkeypatch.setenv(paths.ENV_REPO_ROOT, str(tmp_path))
    assert paths.REPO_ROOT == tmp_path


def test_section_labels_invert_the_zscan_numbering():
    from baystarrfish.data import section_labels

    labels = section_labels(["Conv_zscan2_cell1", "Conv_zscan1_cell2"])
    assert list(labels) == ["sec1", "sec2"]


def test_section_labels_reject_unparseable_names():
    from baystarrfish.data import section_labels

    with pytest.raises(ValueError, match="cannot assign section"):
        section_labels(["something_else"])


def test_standardize_obs_strips_allen_prefixes_and_slashes():
    import anndata as ad

    from baystarrfish.data import standardize_obs

    adata = ad.AnnData(
        np.zeros((2, 1)),
        obs=pd.DataFrame(
            {"subclass_name": ["006 L2/3 IT CTX", "022 Pvalb"],
             "class_name": ["01 IT-ET Glut", "02 GABA"]},
            index=["c1", "c2"],
        ),
    )
    standardize_obs(adata)
    assert list(adata.obs["subclass"]) == ["L2-3 IT CTX", "Pvalb"]
    assert list(adata.obs["class"]) == ["IT-ET Glut", "GABA"]


def test_standardize_obs_reports_the_missing_column():
    import anndata as ad

    from baystarrfish.data import standardize_obs

    adata = ad.AnnData(np.zeros((1, 1)), obs=pd.DataFrame(index=["c1"]))
    with pytest.raises(KeyError, match="class_name"):
        standardize_obs(adata)
