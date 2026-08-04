"""The weighted collapse -- the trick that makes 408,621 x 400 tractable.

If this is wrong the model silently fits a different dataset, so the central
test reconstructs the naive per-cell log-likelihood and demands equality.
"""

from __future__ import annotations

import numpy as np

from baystarrfish.model.collapse import (
    CollapsedStats,
    build_sufficient_stats,
    choose_kmax,
    count_kmax_truncated,
    kmax_tail_mass,
    summarize_evidence,
)
from baystarrfish.model.likelihood import marginal_loglik


def _stats(counts, group, n_group, n_cre):
    return build_sufficient_stats(counts, group, n_group, n_cre,
                                  class_of_group=None, n_class=None)


def test_weighted_collapse_reproduces_the_naive_loglik(toy_counts):
    """Sum of weight * per-row loglik must equal the per-(cell, cCRE) sum."""
    t7, cre = toy_counts["t7"], toy_counts["cre"]
    group, n_group, n_cre = toy_counts["group"], toy_counts["n_group"], toy_counts["n_cre"]
    stats = _stats({"t7": t7, "cre": cre}, group, n_group, n_cre)

    rho = np.exp(np.linspace(-2.0, -1.0, n_group))
    a = np.exp(np.linspace(-0.3, 0.3, n_cre))
    gamma_matrix = np.exp(np.linspace(0.1, 1.0, n_group * n_cre)).reshape(n_group, n_cre)

    lam = rho[np.asarray(stats.group)] * a[np.asarray(stats.cre)]
    gamma = gamma_matrix[np.asarray(stats.group), np.asarray(stats.cre)]
    per_row = np.asarray(marginal_loglik(stats, lam, 4.0, 3.0, gamma, 2.5, kmax=30))
    collapsed_total = float((np.asarray(stats.weight) * per_row).sum())

    # Naive: every (cell, cCRE) pair as its own weight-1 row.
    cells, cres = np.meshgrid(np.arange(t7.shape[0]), np.arange(n_cre), indexing="ij")
    flat_group = group[cells.ravel()]
    flat_cre = cres.ravel()
    naive = CollapsedStats(
        group=flat_group,
        cre=flat_cre,
        counts={"t7": t7.ravel(), "cre": cre.ravel()},
        weight=np.ones(t7.size, dtype=np.int64),
        n_per_group=np.bincount(group, minlength=n_group),
        n_group=n_group,
        n_cre=n_cre,
        channels=("t7", "cre"),
    )
    lam_flat = rho[flat_group] * a[flat_cre]
    gamma_flat = gamma_matrix[flat_group, flat_cre]
    naive_total = float(
        np.asarray(marginal_loglik(naive, lam_flat, 4.0, 3.0, gamma_flat, 2.5, kmax=30)).sum()
    )
    np.testing.assert_allclose(collapsed_total, naive_total, rtol=1e-10)


def test_collapse_conserves_every_observation(toy_counts):
    t7, cre = toy_counts["t7"], toy_counts["cre"]
    stats = _stats({"t7": t7, "cre": cre}, toy_counts["group"],
                   toy_counts["n_group"], toy_counts["n_cre"])
    assert float(np.asarray(stats.weight).sum()) == t7.size
    # Total counts per channel survive the collapse.
    for name, matrix in (("t7", t7), ("cre", cre)):
        got = float((np.asarray(stats.weight) * np.asarray(stats.counts[name])).sum())
        assert got == float(matrix.sum())


def test_collapse_is_much_smaller_than_the_naive_table(toy_counts):
    stats = _stats({"t7": toy_counts["t7"], "cre": toy_counts["cre"]},
                   toy_counts["group"], toy_counts["n_group"], toy_counts["n_cre"])
    assert len(stats.weight) < toy_counts["t7"].size


def test_evidence_counts_double_positive_pairs(toy_counts):
    t7, cre = toy_counts["t7"], toy_counts["cre"]
    stats = _stats({"t7": t7, "cre": cre}, toy_counts["group"],
                   toy_counts["n_group"], toy_counts["n_cre"])
    evidence = summarize_evidence(stats)
    assert evidence["totals"]["n_double_pos"] == int(((t7 > 0) & (cre > 0)).sum())


def test_kmax_truncation_leaves_negligible_tail_mass():
    lam = np.array([0.01, 0.5, 3.0])
    kmax = choose_kmax(float(lam.max()), max_count=50, beta_t7=4.0)
    assert kmax_tail_mass(lam, kmax) < 1e-8


def test_count_kmax_truncated_flags_rows_that_need_a_bigger_grid(toy_counts):
    stats = _stats({"t7": toy_counts["t7"], "cre": toy_counts["cre"]},
                   toy_counts["group"], toy_counts["n_group"], toy_counts["n_cre"])
    generous = count_kmax_truncated(stats, kmax=200, beta_t7=4.0)
    stingy = count_kmax_truncated(stats, kmax=1, beta_t7=4.0)
    assert generous == 0
    assert stingy >= generous
