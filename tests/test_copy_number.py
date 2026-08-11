"""Recovering the latent AAV copy number as a cell x cCRE matrix.

The matrix is built by a collapse-and-scatter shortcut (baseline for the all-zero
pairs, unique patterns for the rest). The central test here computes the same
thing pair by pair with no shortcut and demands equality, because the shortcut is
what makes the feature usable and also what could silently mis-assign a row.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from baystarrfish.inference.copy_number import (
    REQUIRED_POSTERIOR_SITES,
    infer_copy_number,
    infer_copy_number_from_fit,
    load_copy_number_draws,
)
from baystarrfish.inference.posterior_k import posterior_k_expectation, posterior_k_moments

N_DRAW, N_GROUP, N_CRE, N_CELL, KMAX = 40, 3, 5, 60, 25
CRE_NAMES = [f"CRE{i:03d}" for i in range(1, N_CRE + 1)]
GROUP_NAMES = [f"sub{i}" for i in range(N_GROUP)]


@pytest.fixture
def draws(rng):
    return {
        "rho": np.exp(rng.normal(-1.2, 0.2, size=(N_DRAW, N_GROUP))),
        "a": np.exp(rng.normal(0.0, 0.3, size=(N_DRAW, N_CRE))),
        "log_gamma": rng.normal(0.5, 0.3, size=(N_DRAW, N_GROUP, N_CRE)),
        "beta_t7": np.exp(rng.normal(1.2, 0.1, size=N_DRAW)),
        "phi_t7": np.full(N_DRAW, 3.0),
        "phi_cre": np.full(N_DRAW, 2.5),
        "group_names": np.array(GROUP_NAMES, dtype=object),
        "cre_names": np.array(CRE_NAMES, dtype=object),
    }


@pytest.fixture
def observations(rng):
    group = np.array([GROUP_NAMES[i % N_GROUP] for i in range(N_CELL)])
    t7 = rng.poisson(0.6, size=(N_CELL, N_CRE))
    cre = rng.poisson(0.4, size=(N_CELL, N_CRE))
    return t7, cre, group


def _brute_force(t7, cre, group, draws, dtype=np.float64):
    """E[k|obs] computed one pair at a time, with no collapse and no scatter."""
    group_index = np.array([GROUP_NAMES.index(g) for g in group])
    rows, cols = np.meshgrid(np.arange(len(group)), np.arange(N_CRE), indexing="ij")
    flat = posterior_k_expectation(
        t7[rows.ravel(), cols.ravel()],
        cre[rows.ravel(), cols.ravel()],
        group_index[rows.ravel()],
        cols.ravel(),
        draws,
        KMAX,
    )
    return flat.reshape(len(group), N_CRE).astype(dtype)


def test_matrix_equals_the_pair_by_pair_computation(draws, observations):
    t7, cre, group = observations
    got = infer_copy_number(t7, cre, group, draws, kmax=KMAX, cre_names=CRE_NAMES,
                            dtype=np.float64, verbose=False)
    np.testing.assert_allclose(got.copies, _brute_force(t7, cre, group, draws), rtol=1e-12)


def test_the_all_zero_baseline_is_used_where_nothing_was_observed(draws, observations):
    """Cells of the same type with no counts must share the same expectation."""
    t7, cre, group = observations
    t7 = np.zeros_like(t7)
    cre = np.zeros_like(cre)
    matrix = infer_copy_number(t7, cre, group, draws, kmax=KMAX, cre_names=CRE_NAMES,
                               dtype=np.float64, verbose=False)
    for name in GROUP_NAMES:
        rows = matrix.copies[group == name]
        np.testing.assert_allclose(rows, np.broadcast_to(rows[0], rows.shape), rtol=1e-12)
    # Different cell types have different infection rates, so different baselines.
    per_group = np.array([matrix.copies[group == name][0] for name in GROUP_NAMES])
    assert len(np.unique(per_group.round(12), axis=0)) == N_GROUP
    # An unobserved pair is not zero copies: it is the posterior given no signal.
    assert (matrix.copies > 0).all()


def test_shape_and_labels(draws, observations):
    t7, cre, group = observations
    names = [f"cell{i}" for i in range(N_CELL)]
    matrix = infer_copy_number(t7, cre, group, draws, kmax=KMAX, cre_names=CRE_NAMES,
                               obs_names=names, verbose=False)
    assert matrix.copies.shape == (N_CELL, N_CRE) == (matrix.n_cells, matrix.n_cre)
    frame = matrix.to_frame()
    assert list(frame.columns) == CRE_NAMES
    assert list(frame.index) == names
    np.testing.assert_allclose(matrix.total_per_cell(), matrix.copies.sum(axis=1))


def test_more_signal_means_more_inferred_copies(draws, observations):
    """Monotonicity is the one behaviour a reader will assume; check it holds."""
    _, _, group = observations
    quiet = np.zeros((N_CELL, N_CRE), dtype=np.int64)
    loud = np.full((N_CELL, N_CRE), 30, dtype=np.int64)
    kwargs = dict(kmax=KMAX, cre_names=CRE_NAMES, dtype=np.float64, verbose=False)
    low = infer_copy_number(quiet, quiet, group, draws, **kwargs).copies
    high = infer_copy_number(loud, loud, group, draws, **kwargs).copies
    assert (high > low).all()


def test_dropout_raises_the_inferred_copies_for_a_zero_observation(draws, observations):
    """With dropout, a zero is weaker evidence of no infection, so E[k] rises."""
    _, _, group = observations
    zeros = np.zeros((N_CELL, N_CRE), dtype=np.int64)
    kwargs = dict(kmax=KMAX, cre_names=CRE_NAMES, dtype=np.float64, verbose=False)
    without = infer_copy_number(zeros, zeros, group, draws, **kwargs).copies
    with_dropout = infer_copy_number(
        zeros, zeros, group,
        {**draws, "p_drop_t7": np.full(N_DRAW, 0.5), "p_drop_cre": np.full(N_DRAW, 0.5)},
        **kwargs,
    ).copies
    assert (with_dropout > without).all()


def test_sd_is_returned_only_when_asked_and_is_positive(draws, observations):
    t7, cre, group = observations
    assert infer_copy_number(t7, cre, group, draws, kmax=KMAX, cre_names=CRE_NAMES,
                             verbose=False).sd is None
    matrix = infer_copy_number(t7, cre, group, draws, kmax=KMAX, cre_names=CRE_NAMES,
                               return_sd=True, verbose=False)
    assert matrix.sd.shape == matrix.copies.shape
    assert (matrix.sd > 0).all()


def test_sd_matches_the_law_of_total_variance(draws):
    """Var[k|obs] = E_d[Var(k|obs,theta)] + Var_d(E[k|obs,theta]), not just one term."""
    from scipy.special import gammaln

    from baystarrfish.inference.posterior_k import channel_logprob

    t7 = np.array([0, 3, 11])
    cre = np.array([0, 1, 7])
    group_idx = np.array([0, 1, 2])
    cre_idx = np.array([0, 1, 2])
    got = posterior_k_moments(t7, cre, group_idx, cre_idx, draws, KMAX)

    k = np.arange(KMAX + 1, dtype=float)
    first, second = [], []
    for d in range(N_DRAW):
        lam = draws["rho"][d, group_idx] * draws["a"][d, cre_idx]
        gamma = np.exp(draws["log_gamma"][d, group_idx, cre_idx])
        logp = (
            k * np.log(lam)[:, None] - lam[:, None] - gammaln(k + 1)
            + channel_logprob(t7[:, None], k, draws["beta_t7"][d], draws["phi_t7"][d])
            + channel_logprob(cre[:, None], k, gamma[:, None], draws["phi_cre"][d])
        )
        w = np.exp(logp - logp.max(axis=-1, keepdims=True))
        w /= w.sum(axis=-1, keepdims=True)
        first.append((w * k).sum(axis=-1))
        second.append((w * k**2).sum(axis=-1))
    first, second = np.array(first), np.array(second)
    np.testing.assert_allclose(got.mean, first.mean(axis=0), rtol=1e-12)
    want_sd = np.sqrt((second - first**2).mean(axis=0) + first.var(axis=0))
    np.testing.assert_allclose(got.sd, want_sd, rtol=1e-12)


def test_chunking_does_not_change_the_answer(draws, observations):
    t7, cre, group = observations
    kwargs = dict(kmax=KMAX, cre_names=CRE_NAMES, dtype=np.float64, verbose=False)
    a = infer_copy_number(t7, cre, group, draws, chunk=3, **kwargs).copies
    b = infer_copy_number(t7, cre, group, draws, chunk=10_000, **kwargs).copies
    np.testing.assert_array_equal(a, b)


def test_a_reordered_cre_axis_is_realigned_not_assumed(draws, observations):
    """The data's column order need not match the fit's; it must be mapped."""
    t7, cre, group = observations
    order = np.array([3, 0, 4, 1, 2])
    shuffled = [CRE_NAMES[i] for i in order]
    straight = infer_copy_number(t7, cre, group, draws, kmax=KMAX, cre_names=CRE_NAMES,
                                 dtype=np.float64, verbose=False).copies
    reordered = infer_copy_number(t7[:, order], cre[:, order], group, draws, kmax=KMAX,
                                  cre_names=shuffled, dtype=np.float64, verbose=False).copies
    np.testing.assert_allclose(reordered, straight[:, order], rtol=1e-12)


def test_unknown_cre_or_cell_type_is_reported(draws, observations):
    t7, cre, group = observations
    with pytest.raises(ValueError, match="absent from the fit"):
        infer_copy_number(t7, cre, group, draws, kmax=KMAX,
                          cre_names=["nope"] + CRE_NAMES[1:], verbose=False)
    with pytest.raises(ValueError, match="cell type"):
        infer_copy_number(t7, cre, np.array(["ghost"] * N_CELL), draws, kmax=KMAX,
                          cre_names=CRE_NAMES, verbose=False)


def test_mismatched_shapes_are_rejected(draws, observations):
    t7, cre, group = observations
    with pytest.raises(ValueError, match="equal shape"):
        infer_copy_number(t7, cre[:, :-1], group, draws, kmax=KMAX,
                          cre_names=CRE_NAMES, verbose=False)
    with pytest.raises(ValueError, match="cCRE names"):
        infer_copy_number(t7, cre, group, draws, kmax=KMAX,
                          cre_names=CRE_NAMES[:-1], verbose=False)
    with pytest.raises(ValueError, match="labels for"):
        infer_copy_number(t7, cre, group[:-1], draws, kmax=KMAX,
                          cre_names=CRE_NAMES, verbose=False)


def test_write_npz_round_trips(tmp_path, draws, observations):
    t7, cre, group = observations
    matrix = infer_copy_number(t7, cre, group, draws, kmax=KMAX, cre_names=CRE_NAMES,
                               obs_names=[f"cell{i}" for i in range(N_CELL)],
                               return_sd=True, verbose=False)
    path = matrix.write_npz(tmp_path / "copies.npz")
    with np.load(path, allow_pickle=True) as handle:
        np.testing.assert_allclose(handle["copies"], matrix.copies)
        np.testing.assert_allclose(handle["sd"], matrix.sd)
        assert list(handle["cre_names"]) == CRE_NAMES
        assert int(handle["kmax"]) == KMAX
        assert len(handle["obs_names"]) == N_CELL


# ---- loading from a fit directory ----------------------------------------- #


def _write_fit_dir(tmp_path, draws, *, sites, dropout=False, kmax=KMAX):
    tag = "subclass_joint_copy_number_svi"
    payload = {name: draws[name] for name in sites}
    payload["group_names"] = np.array(GROUP_NAMES, dtype=object)
    payload["cre_names"] = np.array(CRE_NAMES, dtype=object)
    np.savez_compressed(tmp_path / f"{tag}_posterior_samples.npz", **payload)
    scalars = {name: draws[name] for name in ("beta_t7", "phi_t7", "phi_cre")}
    if dropout:
        scalars["p_drop_t7"] = np.full(N_DRAW, 0.1)
        scalars["p_drop_cre"] = np.full(N_DRAW, 0.1)
    np.savez(tmp_path / f"{tag}_scalar_samples.npz", **scalars)
    (tmp_path / "run_manifest.json").write_text(json.dumps({
        "tag": tag,
        "section": "all",
        "negative_control_mode": "ordinary",
        "config": {
            "kmax": kmax, "level": "subclass",
            "infection_model": "copy_number_dropout" if dropout else "copy_number",
        },
    }))
    return tag


def _log_draws(draws):
    return {**draws, "log_rho": np.log(draws["rho"]), "log_a": np.log(draws["a"])}


def test_load_from_a_fit_directory(tmp_path, draws):
    _write_fit_dir(tmp_path, _log_draws(draws), sites=REQUIRED_POSTERIOR_SITES)
    loaded, manifest = load_copy_number_draws(tmp_path)
    np.testing.assert_allclose(loaded["rho"], draws["rho"], rtol=1e-6)
    np.testing.assert_allclose(loaded["a"], draws["a"], rtol=1e-6)
    assert manifest["config"]["kmax"] == KMAX
    assert "p_drop_t7" not in loaded


def test_a_log_gamma_only_fit_explains_what_to_refit(tmp_path, draws):
    """The production fit is exactly this case; the message must be actionable."""
    _write_fit_dir(tmp_path, _log_draws(draws), sites=("log_gamma",))
    with pytest.raises(KeyError) as excinfo:
        load_copy_number_draws(tmp_path)
    message = str(excinfo.value)
    assert "log_rho" in message and "log_a" in message
    assert "--posterior-sites log_gamma log_rho log_a" in message


def test_dropout_parameters_are_carried_through(tmp_path, draws):
    _write_fit_dir(tmp_path, _log_draws(draws), sites=REQUIRED_POSTERIOR_SITES, dropout=True)
    loaded, _ = load_copy_number_draws(tmp_path)
    assert loaded["p_drop_t7"].shape == (N_DRAW,)


def test_a_dropout_fit_missing_its_dropout_scalars_is_refused(tmp_path, draws):
    tag = _write_fit_dir(tmp_path, _log_draws(draws), sites=REQUIRED_POSTERIOR_SITES)
    manifest = json.loads((tmp_path / "run_manifest.json").read_text())
    manifest["config"]["infection_model"] = "copy_number_dropout"
    (tmp_path / "run_manifest.json").write_text(json.dumps(manifest))
    del tag
    with pytest.raises(KeyError, match="lack p_drop_t7"):
        load_copy_number_draws(tmp_path)


def test_from_fit_reads_kmax_and_level_from_the_manifest(tmp_path, draws, observations, rng):
    from baystarrfish.data import CountData

    t7, cre, group = observations
    _write_fit_dir(tmp_path, _log_draws(draws), sites=REQUIRED_POSTERIOR_SITES)
    data = CountData(
        t7=t7, cre=cre, subclass=group, class_=np.array(["cls"] * N_CELL),
        lib_size_log=np.log1p(rng.poisson(50, N_CRE)), cre_names=list(CRE_NAMES),
        negative_control_mask=None, negative_controls=[], negative_control_mode="ordinary",
        obs_names=np.array([f"cell{i}" for i in range(N_CELL)], dtype=object),
    )
    matrix = infer_copy_number_from_fit(data, tmp_path, verbose=False)
    assert matrix.kmax == KMAX
    assert matrix.level == "subclass"
    assert list(matrix.to_frame().index) == list(data.obs_names)
    np.testing.assert_allclose(
        matrix.copies, _brute_force(t7, cre, group, draws, np.float32), rtol=1e-6
    )


def test_from_fit_refuses_a_manifest_without_kmax(tmp_path, draws, observations, rng):
    from baystarrfish.data import CountData

    t7, cre, group = observations
    _write_fit_dir(tmp_path, _log_draws(draws), sites=REQUIRED_POSTERIOR_SITES, kmax=None)
    data = CountData(
        t7=t7, cre=cre, subclass=group, class_=np.array(["cls"] * N_CELL),
        lib_size_log=np.log1p(rng.poisson(50, N_CRE)), cre_names=list(CRE_NAMES),
        negative_control_mask=None, negative_controls=[], negative_control_mode="ordinary",
    )
    with pytest.raises(KeyError, match="no kmax"):
        infer_copy_number_from_fit(data, tmp_path, verbose=False)


def test_pandas_frame_is_labelled_even_without_obs_names(draws, observations):
    t7, cre, group = observations
    frame = infer_copy_number(t7, cre, group, draws, kmax=KMAX, cre_names=CRE_NAMES,
                              verbose=False).to_frame()
    assert isinstance(frame, pd.DataFrame)
    assert list(frame.columns) == CRE_NAMES
    assert frame.shape == (N_CELL, N_CRE)


def test_thinning_the_posterior_leaves_the_mean_close(draws, observations):
    """max_draws trades Monte Carlo error for wall clock; check it is a fair trade."""
    t7, cre, group = observations
    kwargs = dict(kmax=KMAX, cre_names=CRE_NAMES, dtype=np.float64, verbose=False)
    full = infer_copy_number(t7, cre, group, draws, **kwargs).copies
    thinned = infer_copy_number(t7, cre, group, draws, max_draws=N_DRAW // 2, **kwargs).copies
    assert np.abs(thinned - full).max() < 0.05 * full.max()


def test_thinning_is_deterministic_and_evenly_spaced(draws, observations):
    t7, cre, group = observations
    kwargs = dict(kmax=KMAX, cre_names=CRE_NAMES, max_draws=7, dtype=np.float64, verbose=False)
    a = infer_copy_number(t7, cre, group, draws, **kwargs).copies
    b = infer_copy_number(t7, cre, group, draws, **kwargs).copies
    np.testing.assert_array_equal(a, b)  # no rng, so no seed to get wrong


def test_max_draws_above_the_draw_count_is_a_no_op(draws, observations):
    t7, cre, group = observations
    kwargs = dict(kmax=KMAX, cre_names=CRE_NAMES, dtype=np.float64, verbose=False)
    full = infer_copy_number(t7, cre, group, draws, **kwargs).copies
    asked_too_many = infer_copy_number(t7, cre, group, draws, max_draws=10 * N_DRAW,
                                       **kwargs).copies
    np.testing.assert_array_equal(asked_too_many, full)


def test_max_draws_must_be_positive(draws, observations):
    t7, cre, group = observations
    with pytest.raises(ValueError, match="max_draws must be positive"):
        infer_copy_number(t7, cre, group, draws, kmax=KMAX, cre_names=CRE_NAMES,
                          max_draws=0, verbose=False)


def test_float_count_matrices_are_handled(draws, observations):
    """AnnData obsm hands back floats; the fit casts them, so this must too.

    Regression: np.stack promoted the (group, cre, t7, cre) pattern array to
    float, and the index columns then could not be used as indices.
    """
    t7, cre, group = observations
    kwargs = dict(kmax=KMAX, cre_names=CRE_NAMES, dtype=np.float64, verbose=False)
    integral = infer_copy_number(t7, cre, group, draws, **kwargs).copies
    floating = infer_copy_number(
        t7.astype(np.float32), cre.astype(np.float64), group, draws, **kwargs
    ).copies
    np.testing.assert_array_equal(floating, integral)


def test_a_grid_too_small_for_the_infection_rate_warns(draws, observations):
    """E[k] over a truncated grid is biased low with no sign of it in the number."""
    t7, cre, group = observations
    hot = {**draws, "rho": draws["rho"] * 500.0}  # rates far past a tiny kmax
    with pytest.warns(RuntimeWarning, match="truncated and biased low"):
        infer_copy_number(t7, cre, group, hot, kmax=3, cre_names=CRE_NAMES,
                          verbose=False)


def test_no_warning_when_the_grid_comfortably_covers_the_rates(draws, observations):
    import warnings

    t7, cre, group = observations
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        infer_copy_number(t7, cre, group, draws, kmax=KMAX, cre_names=CRE_NAMES,
                          verbose=False)


def test_infection_probability_is_a_probability(draws, observations):
    t7, cre, group = observations
    m = infer_copy_number(t7, cre, group, draws, kmax=KMAX, cre_names=CRE_NAMES,
                          return_probability=True, dtype=np.float64, verbose=False)
    assert m.p_infected.shape == m.copies.shape
    assert (m.p_infected >= 0).all() and (m.p_infected <= 1).all()


def test_probability_matches_an_explicit_sum_over_k(draws):
    """P(k>=1|obs) = E_theta[1 - P(k=0|obs,theta)] -- check against a hand sum."""
    from scipy.special import gammaln

    from baystarrfish.inference.posterior_k import channel_logprob

    t7 = np.array([0, 3, 11]); cre = np.array([0, 1, 7])
    idx = np.array([0, 1, 2])
    got = posterior_k_moments(t7, cre, idx, idx, draws, KMAX)

    k = np.arange(KMAX + 1, dtype=float)
    p0 = []
    for d in range(N_DRAW):
        lam = draws["rho"][d, idx] * draws["a"][d, idx]
        gamma = np.exp(draws["log_gamma"][d, idx, idx])
        logp = (
            k * np.log(lam)[:, None] - lam[:, None] - gammaln(k + 1)
            + channel_logprob(t7[:, None], k, draws["beta_t7"][d], draws["phi_t7"][d])
            + channel_logprob(cre[:, None], k, gamma[:, None], draws["phi_cre"][d])
        )
        w = np.exp(logp - logp.max(axis=-1, keepdims=True))
        w /= w.sum(axis=-1, keepdims=True)
        p0.append(w[:, 0])
    np.testing.assert_allclose(got.p_infected, 1.0 - np.array(p0).mean(axis=0), rtol=1e-12)


def test_a_positive_count_forces_certainty_of_infection(draws, observations):
    """k=0 is a point mass at zero counts, so any nonzero read implies k>=1."""
    _, _, group = observations
    loud = np.full((N_CELL, N_CRE), 4, dtype=np.int64)
    m = infer_copy_number(loud, loud, group, draws, kmax=KMAX, cre_names=CRE_NAMES,
                          return_probability=True, dtype=np.float64, verbose=False)
    np.testing.assert_allclose(m.p_infected, 1.0, atol=1e-12)


def test_probability_and_mean_are_different_questions(draws, observations):
    """A pair can be probably-infected yet expected to carry few copies."""
    _, _, group = observations
    one = np.ones((N_CELL, N_CRE), dtype=np.int64)
    m = infer_copy_number(one, one, group, draws, kmax=KMAX, cre_names=CRE_NAMES,
                          return_probability=True, dtype=np.float64, verbose=False)
    assert np.allclose(m.p_infected, 1.0, atol=1e-12)
    assert (m.copies > 1.0).any()  # certain of infection, uncertain of the count


def test_probability_is_absent_unless_requested(draws, observations):
    t7, cre, group = observations
    m = infer_copy_number(t7, cre, group, draws, kmax=KMAX, cre_names=CRE_NAMES,
                          verbose=False)
    assert m.p_infected is None
    with pytest.raises(ValueError, match="was not computed"):
        m.matrix("p_infected")
    with pytest.raises(ValueError, match="unknown matrix"):
        m.matrix("nonsense")


def test_write_csv_round_trips_every_matrix(tmp_path, draws, observations):
    t7, cre, group = observations
    m = infer_copy_number(t7, cre, group, draws, kmax=KMAX, cre_names=CRE_NAMES,
                          obs_names=[f"cell{i}" for i in range(N_CELL)],
                          return_sd=True, return_probability=True, verbose=False)
    for which in ("copies", "sd", "p_infected"):
        for suffix in (".csv", ".csv.gz"):
            path = m.write_csv(tmp_path / f"{which}{suffix}", which, decimals=6)
            back = pd.read_csv(path, index_col="cell")
            assert list(back.columns) == CRE_NAMES
            assert list(back.index) == [f"cell{i}" for i in range(N_CELL)]
            np.testing.assert_allclose(back.to_numpy(), m.matrix(which), atol=1e-6)


def test_write_csv_block_boundary_keeps_one_header(tmp_path, draws, observations):
    """Blocked writing must not repeat the header or drop rows."""
    import gzip

    t7, cre, group = observations
    m = infer_copy_number(t7, cre, group, draws, kmax=KMAX, cre_names=CRE_NAMES,
                          verbose=False)
    path = m.write_csv(tmp_path / "c.csv.gz", "copies")
    with gzip.open(path, "rt") as handle:
        lines = handle.read().splitlines()
    assert len(lines) == N_CELL + 1
    assert sum(line.startswith("cell,") for line in lines) == 1


# ---- the Gamma-conjugate per-cell activity --------------------------------- #


def test_activity_matches_the_conjugate_formula(draws):
    """gamma * E[G], G | cre,k ~ Gamma(phi+cre, phi+k*gamma), over P(k|obs)."""
    from scipy.special import gammaln

    from baystarrfish.inference.posterior_k import channel_logprob

    t7 = np.array([0, 2, 9]); cre = np.array([0, 1, 6])
    idx = np.array([0, 1, 2])
    got = posterior_k_moments(t7, cre, idx, idx, draws, KMAX)

    k = np.arange(KMAX + 1, dtype=float)
    per_cell = []
    for i in range(3):
        vals = []
        for d in range(N_DRAW):
            lam = draws["rho"][d, i] * draws["a"][d, i]
            g = np.exp(draws["log_gamma"][d, i, i]); phi = draws["phi_cre"][d]
            lp = (k*np.log(lam) - lam - gammaln(k+1)
                  + channel_logprob(np.array([[t7[i]]]), k, draws["beta_t7"][d], draws["phi_t7"][d])[0]
                  + channel_logprob(np.array([[cre[i]]]), k, g, phi)[0])
            w = np.exp(lp - lp.max()); w /= w.sum()
            vals.append(g * (w * (phi + cre[i]) / (phi + k * g)).sum())
        per_cell.append(np.mean(vals))
    np.testing.assert_allclose(got.activity, per_cell, rtol=1e-12)


def test_activity_is_finite_and_positive_even_with_no_counts(draws):
    """cre=0 returns a small positive number, not the ratio's hard zero."""
    zeros = np.zeros(3, dtype=np.int64)
    idx = np.array([0, 1, 2])
    got = posterior_k_moments(zeros, zeros, idx, idx, draws, KMAX)
    assert np.isfinite(got.activity).all()
    assert (got.activity > 0).all()


def test_activity_approaches_the_moment_estimator_for_large_counts(draws):
    """With many counts the prior washes out and it relaxes to cre/k."""
    t7 = np.array([400]); cre = np.array([400]); idx = np.array([0])
    got = posterior_k_moments(t7, cre, idx, idx, draws, KMAX)
    ratio = cre[0] / got.mean[0]
    assert abs(got.activity[0] - ratio) / ratio < 0.25


def test_activity_shrinks_toward_the_cell_type_activity(draws):
    """A one-count cell is pulled below its raw ratio; a silent one below that."""
    idx = np.array([0, 0])
    got = posterior_k_moments(np.array([3, 3]), np.array([1, 0]), idx, idx, draws, KMAX)
    assert got.activity[0] > got.activity[1] > 0


def test_matrix_carries_activity_only_when_asked(draws, observations):
    t7, cre, group = observations
    plain = infer_copy_number(t7, cre, group, draws, kmax=KMAX, cre_names=CRE_NAMES,
                              verbose=False)
    assert plain.activity is None
    with pytest.raises(ValueError, match="return_activity"):
        plain.matrix("activity")
    full = infer_copy_number(t7, cre, group, draws, kmax=KMAX, cre_names=CRE_NAMES,
                             return_activity=True, verbose=False)
    assert full.activity.shape == full.copies.shape
    assert np.isfinite(full.activity).all() and (full.activity > 0).all()


def test_activity_survives_the_npz_round_trip(tmp_path, draws, observations):
    t7, cre, group = observations
    m = infer_copy_number(t7, cre, group, draws, kmax=KMAX, cre_names=CRE_NAMES,
                          return_activity=True, verbose=False)
    with np.load(m.write_npz(tmp_path / "cn.npz"), allow_pickle=True) as h:
        np.testing.assert_allclose(h["activity"], m.activity)


# ---- activity normalised by the negative-control reference ----------------- #


def test_a_constant_baseline_reduces_to_a_plain_division(draws):
    """With ``b`` fixed across draws, ``E[X/b] = E[X]/b`` -- exactly, not nearly.

    This is the one case where the per-draw and two-stage forms must agree, so it
    pins the arithmetic without the draw-to-draw variation obscuring it.
    """
    idx = np.repeat(np.arange(N_GROUP), 4)
    cre_idx = np.tile(np.arange(4), N_GROUP)
    t7 = np.tile(np.array([0, 2, 5, 9]), N_GROUP)
    cre = np.tile(np.array([0, 1, 3, 7]), N_GROUP)
    offsets = np.array([0.4, -0.7, 1.1])[:N_GROUP]
    log_baseline = np.tile(offsets, (N_DRAW, 1))

    got = posterior_k_moments(t7, cre, idx, cre_idx, draws, KMAX,
                              log_baseline=log_baseline)
    np.testing.assert_allclose(
        got.activity_normalized, got.activity / np.exp(offsets[idx]), rtol=1e-12
    )


def test_the_ratio_is_formed_inside_the_draw_not_after(draws, rng):
    """A draw-varying baseline must NOT equal activity / mean(baseline).

    ``gamma`` and ``b`` share the scale factors that make either one arbitrary,
    so their per-draw ratio is a different -- and better determined -- quantity
    than the ratio of their averages. If this ever passes, the division has
    silently moved outside the draw loop.
    """
    idx = np.repeat(np.arange(N_GROUP), 4)
    cre_idx = np.tile(np.arange(4), N_GROUP)
    t7 = np.tile(np.array([0, 2, 5, 9]), N_GROUP)
    cre = np.tile(np.array([0, 1, 3, 7]), N_GROUP)
    log_baseline = rng.normal(0.2, 0.8, size=(N_DRAW, N_GROUP))

    got = posterior_k_moments(t7, cre, idx, cre_idx, draws, KMAX,
                              log_baseline=log_baseline)
    two_stage = got.activity / np.exp(log_baseline).mean(axis=0)[idx]
    assert np.all(np.isfinite(got.activity_normalized))
    assert not np.allclose(got.activity_normalized, two_stage, rtol=1e-3)


def test_an_ineligible_cell_type_stays_nan(draws):
    idx = np.repeat(np.arange(N_GROUP), 3)
    cre_idx = np.tile(np.arange(3), N_GROUP)
    counts = np.tile(np.array([0, 2, 4]), N_GROUP)
    log_baseline = np.zeros((N_DRAW, N_GROUP))
    log_baseline[:, 1] = np.nan  # no eligible control reference for group 1

    got = posterior_k_moments(counts, counts, idx, cre_idx, draws, KMAX,
                              log_baseline=log_baseline)
    assert np.all(np.isnan(got.activity_normalized[idx == 1]))
    assert np.all(np.isfinite(got.activity_normalized[idx != 1]))
    # The other outputs are unaffected: only the normalisation is missing.
    assert np.isfinite(got.mean).all() and np.isfinite(got.activity).all()


def test_a_mismatched_baseline_is_rejected_rather_than_broadcast(draws):
    idx = np.zeros(3, dtype=np.int64)
    with pytest.raises(ValueError, match="log_baseline has shape"):
        posterior_k_moments(idx, idx, idx, idx, draws, KMAX,
                            log_baseline=np.zeros((N_DRAW // 2, N_GROUP)))
    with pytest.raises(ValueError, match="cell types but"):
        posterior_k_moments(idx, idx, np.full(3, 2), idx, draws, KMAX,
                            log_baseline=np.zeros((N_DRAW, 1)))


def test_normalized_matrix_scatters_the_same_way_the_others_do(draws, observations):
    """The collapse-and-scatter shortcut must carry the new field correctly.

    Controls are given equal, draw-constant activity by construction, so the
    reference is known and the whole matrix must equal activity / b.
    """
    t7, cre, group = observations
    controls = CRE_NAMES[-2:]
    control_idx = [CRE_NAMES.index(name) for name in controls]
    flat = dict(draws)
    flat["log_gamma"] = draws["log_gamma"].copy()
    flat["log_gamma"][:, :, control_idx] = 0.25   # b = exp(0.25) everywhere

    matrix = infer_copy_number(
        t7, cre, group, flat, kmax=KMAX, cre_names=CRE_NAMES,
        return_activity=True, return_activity_normalized=True,
        negative_control_cre=controls, negative_control_t7_threshold=0.0,
        verbose=False,
    )
    assert matrix.activity_normalized.shape == matrix.copies.shape
    np.testing.assert_allclose(
        matrix.activity_normalized, matrix.activity / np.exp(0.25), rtol=2e-6
    )


def test_a_cell_type_below_the_t7_threshold_is_nan_across_the_row(draws, observations):
    t7, cre, group = observations
    controls = CRE_NAMES[-2:]
    # Nothing reaches a pooled control T7 this high, so no cell type qualifies.
    matrix = infer_copy_number(
        t7, cre, group, draws, kmax=KMAX, cre_names=CRE_NAMES,
        return_activity_normalized=True, negative_control_cre=controls,
        negative_control_t7_threshold=1e9, verbose=False,
    )
    assert np.all(np.isnan(matrix.activity_normalized))
    assert np.isfinite(matrix.copies).all()


def test_normalization_needs_controls_and_reports_unknown_ones(draws, observations):
    t7, cre, group = observations
    with pytest.raises(ValueError, match="needs negative_control_cre"):
        infer_copy_number(t7, cre, group, draws, kmax=KMAX, cre_names=CRE_NAMES,
                          return_activity_normalized=True, verbose=False)
    with pytest.raises(ValueError, match="not columns of the fit"):
        infer_copy_number(t7, cre, group, draws, kmax=KMAX, cre_names=CRE_NAMES,
                          return_activity_normalized=True,
                          negative_control_cre=["CRE999"], verbose=False)


def test_normalized_activity_survives_the_npz_round_trip(tmp_path, draws, observations):
    t7, cre, group = observations
    m = infer_copy_number(
        t7, cre, group, draws, kmax=KMAX, cre_names=CRE_NAMES,
        return_activity_normalized=True, negative_control_cre=CRE_NAMES[-2:],
        negative_control_t7_threshold=0.0, verbose=False,
    )
    assert m.activity is None
    with pytest.raises(ValueError, match="return_activity_normalized"):
        m.matrix("activity")
    with np.load(m.write_npz(tmp_path / "cn.npz"), allow_pickle=True) as h:
        np.testing.assert_allclose(
            h["activity_normalized"], m.matrix("activity_normalized")
        )
