"""Fit serialisation round-trips, and the CountData contract.

The layout written here is the one every existing ``results/`` tree already has,
so these tests are as much about not breaking readers as about writers.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from baystarrfish.data import CountData
from baystarrfish.io import (
    decode_strings,
    fit_tag,
    jsonable,
    load_gamma,
    load_posterior_samples,
    read_fit,
    write_fit,
)

TAG = "subclass_joint_copy_number_dropout_svi"
N_GROUP, N_CRE, N_DRAW, N_CELL = 3, 5, 20, 40
CRE_NAMES = [f"CRE{i:03d}" for i in range(1, N_CRE + 1)]
GROUPS = [f"sub{i}" for i in range(N_GROUP)]


@pytest.fixture
def count_data(rng):
    return CountData(
        t7=rng.poisson(0.3, (N_CELL, N_CRE)),
        cre=rng.poisson(0.1, (N_CELL, N_CRE)),
        subclass=np.array([GROUPS[i % N_GROUP] for i in range(N_CELL)]),
        class_=np.array(["cls"] * N_CELL),
        lib_size_log=np.log1p(rng.poisson(50, N_CRE)),
        cre_names=list(CRE_NAMES),
        negative_control_mask=None,
        negative_controls=[CRE_NAMES[-1]],
        negative_control_mode="ordinary",
        blacklist=["CRE099"],
        cre_info=pd.DataFrame({"labeling_type": ["x"] * N_CRE}, index=CRE_NAMES),
        subclass_cell_counts=pd.Series([14, 13, 13], index=GROUPS, name="n_cells"),
        section="all",
    )


@pytest.fixture
def result(rng):
    return {
        "summary": {
            "gamma": pd.DataFrame(
                {"group": GROUPS * N_CRE, "cre": CRE_NAMES * N_GROUP,
                 "gamma_mean": rng.normal(size=N_GROUP * N_CRE)}
            ),
            "rho": pd.DataFrame({"group": GROUPS, "rho_mean": rng.normal(size=N_GROUP)}),
        },
        "evidence": {
            "per_pair": pd.DataFrame({"group": GROUPS, "n": [1, 2, 3]}),
            "totals": {"n_double_pos": np.int64(7)},
        },
        "ppc": {"t7": {"obs": {"zero_fraction": np.float64(0.99)}}},
        "diagnostics": {"kmax": 6, "losses": np.array([9e5, 1e5, 5e4])},
        "scalar_samples": {"beta_t7": rng.normal(size=N_DRAW)},
        "posterior_samples": {
            "log_gamma": rng.normal(size=(N_DRAW, N_GROUP, N_CRE)).astype(np.float64)
        },
        "group_names": GROUPS,
        "cre_names": CRE_NAMES,
        "config": {"level": "subclass"},
    }


def test_write_fit_produces_the_published_directory_layout(tmp_path, result, count_data):
    write_fit(result, tmp_path, TAG, data=count_data, input_path="pyproject.toml",
              manifest_extra={"method_variant": "demo"})
    # Exactly the file set of results/bayesian/ from the production run.
    assert {p.name for p in tmp_path.iterdir()} == {
        "cre_info.csv", "cre_blacklist.csv", "negative_controls.csv",
        "run_manifest.json", "subclass_cell_counts.csv",
        f"{TAG}_diagnostics.json", f"{TAG}_evidence_per_pair.csv",
        f"{TAG}_evidence_totals.json", f"{TAG}_gamma.csv", f"{TAG}_losses.npy",
        f"{TAG}_posterior_samples.npz", f"{TAG}_ppc.json", f"{TAG}_result.pkl",
        f"{TAG}_rho.csv", f"{TAG}_scalar_samples.npz",
    }


def test_round_trip(tmp_path, result, count_data):
    gamma_in = result["summary"]["gamma"].copy()
    draws_in = result["posterior_samples"]["log_gamma"].copy()
    write_fit(result, tmp_path, TAG, data=count_data)

    assert fit_tag(tmp_path) == TAG
    pd.testing.assert_frame_equal(load_gamma(tmp_path), gamma_in)
    post = load_posterior_samples(tmp_path)
    np.testing.assert_allclose(post["log_gamma"], draws_in.astype(np.float32))
    assert list(post["cre_names"]) == CRE_NAMES
    assert list(post["group_names"]) == GROUPS
    assert set(read_fit(tmp_path)) >= {"summary", "evidence", "config"}


def test_posterior_draws_are_stored_as_float32(tmp_path, result, count_data):
    """A 444 MB block per production fit; float64 would double it for no gain."""
    write_fit(result, tmp_path, TAG, data=count_data)
    assert load_posterior_samples(tmp_path)["log_gamma"].dtype == np.float32


def test_posterior_is_popped_so_the_pickle_stays_small(tmp_path, result, count_data):
    write_fit(result, tmp_path, TAG, data=count_data)
    assert "posterior_samples" not in result
    assert "posterior_samples" not in read_fit(tmp_path)


def test_sites_argument_avoids_materialising_the_big_block(tmp_path, result, count_data):
    write_fit(result, tmp_path, TAG, data=count_data)
    names_only = load_posterior_samples(tmp_path, sites=[])
    assert set(names_only) == {"group_names", "cre_names"}
    with pytest.raises(KeyError, match="no site"):
        load_posterior_samples(tmp_path, sites=["not_a_site"])


def test_losses_become_a_npy_plus_scalar_diagnostics(tmp_path, result, count_data):
    write_fit(result, tmp_path, TAG, data=count_data)
    diagnostics = json.loads((tmp_path / f"{TAG}_diagnostics.json").read_text())
    assert diagnostics["loss_start"] == 9e5
    assert diagnostics["loss_end"] == 5e4
    assert diagnostics["loss_all_finite"] is True
    assert "losses" not in diagnostics  # a 30k-step trace has no place in JSON
    assert np.load(tmp_path / f"{TAG}_losses.npy").shape == (3,)


def test_decoupled_shape_writes_both_loss_traces_and_the_lambda_summary(
    tmp_path, result, count_data, rng
):
    result["diagnostics"] = {"losses_t7": np.array([5.0, 1.0]),
                             "losses_cre": np.array([7.0, 2.0])}
    result["t7_evidence"] = result["evidence"]
    result["log_lambda_mean"] = rng.normal(size=(N_GROUP, N_CRE))
    result["log_lambda_sd"] = np.abs(rng.normal(size=(N_GROUP, N_CRE)))
    result["infection_posterior_samples"] = {
        "log_lambda": rng.normal(size=(N_DRAW, N_GROUP, N_CRE))
    }
    write_fit(result, tmp_path, TAG, data=count_data)
    for name in (f"{TAG}_losses_t7.npy", f"{TAG}_losses_cre.npy",
                 f"{TAG}_log_lambda_summary.npz",
                 f"{TAG}_infection_posterior_samples.npz",
                 f"{TAG}_t7_evidence_per_pair.csv"):
        assert (tmp_path / name).exists(), name
    diagnostics = json.loads((tmp_path / f"{TAG}_diagnostics.json").read_text())
    assert diagnostics["loss_t7_start"] == 5.0
    assert diagnostics["loss_cre_end"] == 2.0


def test_manifest_records_the_shapes_and_provenance(tmp_path, result, count_data):
    write_fit(result, tmp_path, TAG, data=count_data, input_path="pyproject.toml",
              manifest_extra={"method_variant": "demo"})
    manifest = json.loads((tmp_path / "run_manifest.json").read_text())
    assert manifest["tag"] == TAG
    assert manifest["n_cells"] == N_CELL
    assert manifest["n_cres_fitted"] == N_CRE
    assert manifest["n_subclasses"] == N_GROUP
    assert manifest["negative_control_mode"] == "ordinary"
    assert manifest["method_variant"] == "demo"
    assert manifest["input"]["size_bytes"] > 0


def test_jsonable_flattens_numpy_scalars_and_arrays():
    out = jsonable({"a": np.int64(3), "b": np.float32(1.5), "c": np.arange(3),
                    "d": [np.float64(2.0)]})
    assert out == {"a": 3, "b": pytest.approx(1.5), "c": [0, 1, 2], "d": [2.0]}
    json.dumps(out)  # must not raise


def test_decode_strings_handles_bytes_and_objects():
    got = decode_strings(np.array([b"CRE001", "CRE002", np.bytes_(b"CRE003")], dtype=object))
    assert list(got) == ["CRE001", "CRE002", "CRE003"]


def test_atomic_save_leaves_no_partial_file(tmp_path):
    from baystarrfish.io import atomic_save_array

    target = tmp_path / "nested" / "array.npy"
    atomic_save_array(target, np.arange(5))
    assert not list(target.parent.glob("*.tmp"))
    np.testing.assert_array_equal(np.load(target), np.arange(5))
