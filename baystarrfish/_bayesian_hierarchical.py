"""Bayesian hierarchical infection models for STARR-FISH.

Generative model (see docs/superpowers/specs and the plan file):

    latent copies   k_{ij}  ~ Poisson(lambda_{s,j}),  lambda_{s,j} = rho_s * a_j
    T7 channel      t7_{ij} | k ~ NB(mean = k * beta_t7,    disp = phi_t7)   (k=0 => 0)
    CRE channel     cre_{ij}| k ~ NB(mean = k * gamma_{s,j}, disp = phi_cre)  (k=0 => 0)

with a two-level (class -> subclass) hierarchy on the per-cell-type infection
rate ``rho`` and on the per-(cell-type, CRE) enhancer activity ``gamma`` and an
informative ``lib_size`` prior on the per-CRE library abundance ``a``.

The discrete latent ``k`` is marginalised analytically (``logsumexp`` over a
truncated grid ``0..Kmax``), mirroring ``T7CRE_*_DistributionEM.expectation_step``
in :mod:`STARRFISH.utils`, so the model is a continuous-parameter target that
NumPyro can fit by SVI or NUTS.

Scalability comes from collapsing the (cells x CREs) observation array to weighted
unique ``(group, cre, *counts)`` rows: the ~99.85% all-zero pairs share an
identical marginal within a cell type and collapse to a single weighted row.

Nothing in this module loads STARR-FISH data or runs long jobs; it is a pure
numerical library driven by :meth:`STARRFISH.utils.STARRFISH.bayesian_activity_test`.

The module also provides a binary-infection alternative. It replaces the latent
Poisson copy number with a shared Bernoulli infection event whose probability is
``1 - exp(-rho_s * a_j)``. Conditional on infection, each observed channel follows
an NB2 distribution; without infection, both channels are exactly zero.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

import jax

# Tiny infection rates and -inf point masses make float32 logsumexp fragile;
# enable double precision before any array is created.
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from jax.scipy.special import logsumexp

import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, SVI, Trace_ELBO, Predictive
from numpyro.infer.autoguide import AutoNormal, AutoLowRankMultivariateNormal
from numpyro.infer.initialization import init_to_value


# --------------------------------------------------------------------------- #
# Collapsed sufficient statistics
# --------------------------------------------------------------------------- #
@dataclass
class CollapsedStats:
    """Weighted unique-row representation of the observation array.

    Each row is one distinct ``(group, cre, *channel_counts)`` pattern; ``weight``
    is the number of cells contributing it. The likelihood is the ``weight``-
    weighted sum of per-row marginal log-likelihoods.

    Attributes
    ----------
    group : (M,) int
        Group index per row (class or subclass, depending on granularity).
    cre : (M,) int
        CRE column index per row, in ``[0, n_cre)``.
    counts : dict[str, (M,) int]
        Observed count per channel (``"t7"`` and optionally ``"cre"``).
    weight : (M,) int
        Number of cells with this exact pattern.
    n_per_group : (n_group,) int
        Cell count per group.
    class_of_group : (n_group,) int or None
        For subclass-level stats, the parent class index of each subclass; None
        when ``group`` already is the class.
    n_group, n_cre, n_class : int
        Dimensions.
    channels : tuple[str, ...]
        Channel names present, in order.
    """

    group: np.ndarray
    cre: np.ndarray
    counts: Mapping[str, np.ndarray]
    weight: np.ndarray
    n_per_group: np.ndarray
    n_group: int
    n_cre: int
    channels: tuple
    class_of_group: np.ndarray | None = None
    n_class: int | None = None

    def to_jax(self) -> "CollapsedStats":
        """Return a copy with array fields moved to JAX device arrays."""
        return CollapsedStats(
            group=jnp.asarray(self.group),
            cre=jnp.asarray(self.cre),
            counts={k: jnp.asarray(v) for k, v in self.counts.items()},
            weight=jnp.asarray(self.weight, dtype=jnp.float64),
            n_per_group=jnp.asarray(self.n_per_group),
            n_group=self.n_group,
            n_cre=self.n_cre,
            channels=self.channels,
            class_of_group=None if self.class_of_group is None else jnp.asarray(self.class_of_group),
            n_class=self.n_class,
        )


def _nonzero_coords(mat) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(rows, cols, values)`` of the nonzero entries of ``mat``.

    Accepts dense ``np.ndarray``, ``pandas.DataFrame`` or any object exposing a
    SciPy-sparse ``.tocoo()``; values are cast to ``int64``.
    """
    if hasattr(mat, "tocoo"):  # scipy sparse
        coo = mat.tocoo()
        return coo.row.astype(np.int64), coo.col.astype(np.int64), coo.data.astype(np.int64)
    arr = mat.values if isinstance(mat, pd.DataFrame) else np.asarray(mat)
    rows, cols = np.nonzero(arr)
    return rows.astype(np.int64), cols.astype(np.int64), arr[rows, cols].astype(np.int64)


def build_sufficient_stats(
    channel_mats: Mapping[str, "np.ndarray | pd.DataFrame"],
    group_idx: np.ndarray,
    n_group: int,
    n_cre: int,
    class_of_group: np.ndarray | None = None,
    n_class: int | None = None,
) -> CollapsedStats:
    """Collapse cell-level count matrices to weighted unique rows.

    Parameters
    ----------
    channel_mats : ordered mapping name -> (n_cells, n_cre) integer matrix
        e.g. ``{"t7": t7, "cre": cre}`` (joint) or ``{"t7": t7}`` (T7-only).
        Dense, DataFrame or SciPy-sparse are all accepted.
    group_idx : (n_cells,) int
        Group (class or subclass) index per cell, in ``[0, n_group)``.
    n_group, n_cre : int
        Dimensions.
    class_of_group, n_class : optional
        Parent-class map for subclass-level stats (enables the nested hierarchy).

    Returns
    -------
    CollapsedStats
        Rows include the single all-zero ``(group, cre, 0, ...)`` pattern per
        ``(group, cre)`` carrying weight ``N_group - n_nonzero_cells``.
    """
    channels = tuple(channel_mats.keys())
    group_idx = np.asarray(group_idx).astype(np.int64)
    n_cells = group_idx.shape[0]

    # 1. Union of nonzero (cell, cre) positions across channels. The CRE-index
    # column is named "cre_idx" to avoid colliding with a "cre" channel name.
    frames = []
    for name, mat in channel_mats.items():
        rows, cols, vals = _nonzero_coords(mat)
        frames.append(pd.DataFrame({"cell": rows, "cre_idx": cols, name: vals}))
    # Outer-join channels on (cell, cre_idx); missing entries are true zeros.
    merged = frames[0]
    for f in frames[1:]:
        merged = merged.merge(f, on=["cell", "cre_idx"], how="outer")
    for name in channels:
        merged[name] = merged[name].fillna(0).astype(np.int64)
    merged["group"] = group_idx[merged["cell"].to_numpy()]

    # 2. Group nonzero patterns -> weights.
    key_cols = ["group", "cre_idx", *channels]
    nz = (
        merged.groupby(key_cols, sort=False).size().reset_index(name="weight")
        if len(merged)
        else pd.DataFrame(columns=[*key_cols, "weight"])
    )

    # 3. Per-(group, cre) count of cells with any nonzero -> all-zero remainder.
    if len(merged):
        nonzero_cells = merged.groupby(["group", "cre_idx"], sort=False).size().reset_index(name="n_nz")
    else:
        nonzero_cells = pd.DataFrame(columns=["group", "cre_idx", "n_nz"])
    n_per_group = np.bincount(group_idx, minlength=n_group).astype(np.int64)

    # Full (group, cre) grid; all-zero weight = N_group - n_nonzero_cells.
    grid = pd.MultiIndex.from_product(
        [np.arange(n_group), np.arange(n_cre)], names=["group", "cre_idx"]
    ).to_frame(index=False)
    grid = grid.merge(nonzero_cells, on=["group", "cre_idx"], how="left")
    grid["n_nz"] = grid["n_nz"].fillna(0).astype(np.int64)
    grid["weight"] = n_per_group[grid["group"].to_numpy()] - grid["n_nz"].to_numpy()
    allzero = grid.loc[grid["weight"] > 0, ["group", "cre_idx", "weight"]].copy()
    for name in channels:
        allzero[name] = np.int64(0)

    rows = pd.concat([nz[[*key_cols, "weight"]], allzero[[*key_cols, "weight"]]], ignore_index=True)
    rows = rows.astype(np.int64)

    return CollapsedStats(
        group=rows["group"].to_numpy(),
        cre=rows["cre_idx"].to_numpy(),
        counts={name: rows[name].to_numpy() for name in channels},
        weight=rows["weight"].to_numpy(),
        n_per_group=n_per_group,
        n_group=n_group,
        n_cre=n_cre,
        channels=channels,
        class_of_group=None if class_of_group is None else np.asarray(class_of_group).astype(np.int64),
        n_class=n_class,
    )


def summarize_evidence(stats: CollapsedStats) -> dict:
    """Compute per-(group, cre) and global evidence counts.

    Returns a dict with a per-pair :class:`pandas.DataFrame` (``"per_pair"``)
    holding ``n_t7_pos``, ``n_cre_pos`` (if present), ``n_double_pos`` and
    ``n_total``, plus scalar totals and observed zero fractions matching the
    pre-fit audit described in the plan.
    """
    # "cre_idx" holds the CRE column index; channel counts keep their own names
    # ("t7", "cre") so a "cre" channel never collides with the index column.
    df = pd.DataFrame({"group": np.asarray(stats.group), "cre_idx": np.asarray(stats.cre),
                       "weight": np.asarray(stats.weight)})
    for name in stats.channels:
        df[name] = np.asarray(stats.counts[name])

    pos = {f"n_{name}_pos": (df[name] > 0) for name in stats.channels}
    agg_cols = {}
    for col, mask in pos.items():
        df[col] = np.where(mask, df["weight"], 0)
        agg_cols[col] = "sum"
    if {"t7", "cre"} <= set(stats.channels):
        df["n_double_pos"] = np.where((df["t7"] > 0) & (df["cre"] > 0), df["weight"], 0)
        agg_cols["n_double_pos"] = "sum"
    df["n_total"] = df["weight"]
    agg_cols["n_total"] = "sum"

    per_pair = df.groupby(["group", "cre_idx"], sort=True).agg(agg_cols).reset_index()
    per_pair = per_pair.rename(columns={"cre_idx": "cre"})

    total_cells = int(stats.n_per_group.sum())
    totals = {
        "n_cells": total_cells,
        "n_cre": int(stats.n_cre),
        "n_group": int(stats.n_group),
        "n_pairs": int(stats.n_group * stats.n_cre),
    }
    denom = total_cells * stats.n_cre
    for name in stats.channels:
        n_pos = int(per_pair[f"n_{name}_pos"].sum())
        totals[f"n_{name}_pos"] = n_pos
        totals[f"{name}_zero_fraction"] = float(1.0 - n_pos / denom)
    if "n_double_pos" in per_pair:
        totals["n_double_pos"] = int(per_pair["n_double_pos"].sum())
        totals["n_union_nonzero"] = int(
            np.where((df[[*stats.channels]].to_numpy() > 0).any(axis=1), df["weight"], 0).sum()
        )
    return {"per_pair": per_pair, "totals": totals}


def choose_kmax(lam_max: float, max_count: int, beta_t7: float,
                poisson_tol: float = 1e-8, floor: int = 5, cap: int = 60) -> int:
    """Pick a copy-number truncation ``Kmax``.

    Large enough that (a) the Poisson(``lam_max``) tail beyond ``Kmax`` is below
    ``poisson_tol`` and (b) ``Kmax * beta_t7`` covers the bulk of observed counts,
    but capped at ``cap`` so the (n_rows x Kmax) likelihood tensors fit in GPU
    memory. A few extreme-count outliers may be truncated; the caller should log
    how many (see :func:`count_kmax_truncated`). Validate with :func:`kmax_tail_mass`.
    """
    from scipy.stats import poisson

    k_poisson = int(poisson.ppf(1.0 - poisson_tol, max(lam_max, 1e-12)))
    k_counts = int(np.ceil(2.0 * max_count / max(beta_t7, 1e-6)))
    return int(min(cap, max(floor, k_poisson, k_counts)))


def count_kmax_truncated(stats: CollapsedStats, kmax: int, beta_t7: float) -> int:
    """Number of cells whose observed counts imply a copy number above ``kmax``
    (i.e. ``max(t7, cre)/beta_t7 > kmax``) and are therefore modelled imperfectly."""
    counts = np.maximum.reduce([np.asarray(stats.counts[c]) for c in stats.channels])
    mask = counts > kmax * max(beta_t7, 1e-6)
    return int(np.asarray(stats.weight)[mask].sum())


def kmax_tail_mass(lam: np.ndarray, kmax: int) -> float:
    """Max over groups/CREs of Poisson(``lam``) tail mass above ``kmax``."""
    from scipy.stats import poisson

    return float(np.max(poisson.sf(kmax, np.asarray(lam))))


# --------------------------------------------------------------------------- #
# Marginal likelihood over the latent copy number k
# --------------------------------------------------------------------------- #
def _nb2_logprob(count, mean, conc):
    """NB2 log-pmf with mean ``mean`` (>0) and dispersion ``conc`` (var = mu + mu^2/conc)."""
    rate = conc / mean
    return dist.GammaPoisson(concentration=conc, rate=rate).log_prob(count)


def _channel_logprob(obs, k, per_copy, phi, p_drop=None):
    """log P(obs | k) for one NB channel over a k-grid.

    Parameters
    ----------
    obs : (M, 1) array
    k : (1, K) array of integer copy numbers including 0
    per_copy : (M, 1) array, the per-copy mean (beta_t7 or gamma)
    phi : scalar dispersion
    p_drop : optional scalar measurement-dropout probability for k > 0
    """
    safe_k = jnp.where(k == 0, 1.0, k)          # avoid mean=0 at k=0
    mean = per_copy * safe_k                     # (M, K)
    nb = _nb2_logprob(obs, mean, phi)            # finite everywhere (mean>0)
    if p_drop is not None:
        log_keep = jnp.log1p(-p_drop)
        log_drop = jnp.log(p_drop)
        nb = jnp.where(
            obs == 0,
            jnp.logaddexp(log_drop, log_keep + nb),
            log_keep + nb,
        )
    point_mass = jnp.where(obs == 0, 0.0, -jnp.inf)  # (M, 1) broadcast over K
    return jnp.where(k == 0, point_mass, nb)


def marginal_loglik(stats: CollapsedStats, lam, beta_t7, phi_t7,
                    gamma=None, phi_cre=None, kmax: int = 30,
                    p_drop_t7=None, p_drop_cre=None):
    """Per-row marginal log-likelihood, ``logsumexp_k`` over ``0..kmax``.

    Parameters
    ----------
    stats : CollapsedStats (JAX arrays)
    lam : (M,) expected copies for each row, ``rho[group] * a[cre]``.
    beta_t7, phi_t7 : scalars
    gamma : (M,) per-copy CRE mean for each row, or None for a T7-only model.
    phi_cre : scalar or None
    kmax : int

    Returns
    -------
    (M,) array of marginal log-likelihoods (un-weighted).
    """
    k = jnp.arange(0, kmax + 1)                  # (K,)
    k_row = k[None, :]
    log_pk = dist.Poisson(lam[:, None]).log_prob(k_row)            # (M, K)

    t7 = stats.counts["t7"][:, None]
    ll = log_pk + _channel_logprob(t7, k_row, beta_t7, phi_t7, p_drop_t7)

    if gamma is not None:
        cre = stats.counts["cre"][:, None]
        ll = ll + _channel_logprob(cre, k_row, gamma[:, None], phi_cre, p_drop_cre)

    return logsumexp(ll, axis=1)                  # (M,)


def gauss_hermite_rule(n_points: int = 7) -> tuple[np.ndarray, np.ndarray]:
    """Return nodes and log-weights for E[f(Z)], Z ~ Normal(0, 1)."""
    if n_points <= 0:
        raise ValueError("n_points must be positive")
    nodes, weights = np.polynomial.hermite.hermgauss(n_points)
    nodes = np.sqrt(2.0) * nodes
    log_weights = np.log(weights) - 0.5 * np.log(np.pi)
    return nodes.astype(np.float64), log_weights.astype(np.float64)


def cre_marginal_loglik(
    stats: CollapsedStats,
    log_lambda_mean,
    log_lambda_sd,
    gh_nodes,
    gh_log_weights,
    gamma,
    phi_cre,
    p_drop_cre,
    kmax: int = 30,
):
    """CRE-only marginal likelihood integrating over T7-derived log infection.

    ``log_lambda_mean`` and ``log_lambda_sd`` are row-level summaries of the T7
    posterior for log(rho_s * a_j). The expectation over that posterior is
    approximated by Gauss-Hermite quadrature.
    """
    k = jnp.arange(0, kmax + 1)
    log_lambda = log_lambda_mean[:, None] + log_lambda_sd[:, None] * gh_nodes[None, :]
    lam = jnp.exp(log_lambda)
    log_pk = dist.Poisson(lam[:, :, None]).log_prob(k[None, None, :])

    cre = stats.counts["cre"][:, None, None]
    channel_ll = _channel_logprob(
        cre, k[None, None, :], gamma[:, None, None], phi_cre, p_drop_cre
    )
    ll_given_lambda = logsumexp(log_pk + channel_ll, axis=2)
    return logsumexp(gh_log_weights[None, :] + ll_given_lambda, axis=1)


def binary_infection_loglik(stats: CollapsedStats, infection_rate, beta_t7, phi_t7,
                            gamma=None, phi_cre=None):
    """Per-row likelihood after marginalizing a shared binary infection event.

    ``infection_rate`` is the positive infection hazard ``rho[group] * a[cre]``.
    The corresponding infection probability uses the complementary-log-log link:
    ``p_infected = 1 - exp(-infection_rate)``.

    If uninfected, every observed channel is exactly zero. If infected, T7 and
    CRE counts are conditionally independent NB2 variables and may themselves be
    zero, making this a shared-gate zero-inflated negative-binomial model.
    """
    rate = jnp.maximum(infection_rate, jnp.finfo(jnp.float64).tiny)
    log_p_uninfected = -rate
    log_p_infected = jnp.log(-jnp.expm1(-rate))

    all_zero = stats.counts["t7"] == 0
    infected_ll = _nb2_logprob(stats.counts["t7"], beta_t7, phi_t7)
    if gamma is not None:
        all_zero = all_zero & (stats.counts["cre"] == 0)
        infected_ll = infected_ll + _nb2_logprob(stats.counts["cre"], gamma, phi_cre)

    uninfected_ll = jnp.where(all_zero, log_p_uninfected, -jnp.inf)
    return jnp.logaddexp(uninfected_ll, log_p_infected + infected_ll)


# --------------------------------------------------------------------------- #
# Priors / model configuration
# --------------------------------------------------------------------------- #
@dataclass
class ModelPriors:
    """Hyperparameters of the priors (sensible rare-infection defaults)."""
    mu_rho_loc: float = -6.0
    mu_rho_scale: float = 2.0
    sigma_u_scale: float = 1.0          # class-level infection sd
    sigma_w_scale: float = 1.0          # subclass-level infection sd
    tau_a_scale: float = 0.5            # abundance noise around lib_size
    beta_t7_loc: float = 0.0            # LogNormal(loc, scale) for per-copy T7
    beta_t7_scale: float = 1.0
    phi_t7_scale: float = 5.0           # HalfNormal dispersion
    phi_cre_scale: float = 5.0
    mu_alpha_scale: float = 3.0         # CRE baseline activity mean
    sigma_alpha_scale: float = 2.0
    sigma_eta_scale: float = 1.0        # class-level activity sd
    sigma_delta_scale: float = 1.0      # subclass-level activity sd
    mu_gamma_scale: float = 3.0         # direct activity global mean
    sigma_gamma_scale: float = 2.0      # direct activity global sd
    p_drop_t7_alpha: float = 1.0
    p_drop_t7_beta: float = 9.0
    p_drop_cre_alpha: float = 1.0
    p_drop_cre_beta: float = 9.0


def _sample_abundance(lib_size_centered, priors: ModelPriors):
    """Latent log-abundance with informative ``lib_size`` prior, mean-zero constrained."""
    n_cre = lib_size_centered.shape[0]
    tau_a = numpyro.sample("tau_a", dist.HalfNormal(priors.tau_a_scale))
    eps_raw = numpyro.sample("eps_a_raw", dist.Normal(0.0, 1.0).expand([n_cre]).to_event(1))
    log_a = lib_size_centered + tau_a * eps_raw
    log_a = log_a - jnp.mean(log_a)              # fix shared scale: mean(log a) = 0
    return numpyro.deterministic("log_a", log_a)


def _sample_t7_params(priors: ModelPriors):
    beta_t7 = numpyro.sample("beta_t7", dist.LogNormal(priors.beta_t7_loc, priors.beta_t7_scale))
    phi_t7 = numpyro.sample("phi_t7", dist.HalfNormal(priors.phi_t7_scale))
    return beta_t7, phi_t7


def _sample_t7_dropout(priors: ModelPriors):
    return numpyro.sample(
        "p_drop_t7", dist.Beta(priors.p_drop_t7_alpha, priors.p_drop_t7_beta)
    )


def _sample_cre_dropout(priors: ModelPriors):
    return numpyro.sample(
        "p_drop_cre", dist.Beta(priors.p_drop_cre_alpha, priors.p_drop_cre_beta)
    )


def _sample_infection_classlevel(n_group, priors: ModelPriors):
    """log_rho_g = mu_rho + sigma_u * u_raw_g  (group == class)."""
    mu_rho = numpyro.sample("mu_rho", dist.Normal(priors.mu_rho_loc, priors.mu_rho_scale))
    sigma_u = numpyro.sample("sigma_u", dist.HalfNormal(priors.sigma_u_scale))
    u_raw = numpyro.sample("u_raw", dist.Normal(0.0, 1.0).expand([n_group]).to_event(1))
    log_rho = mu_rho + sigma_u * u_raw
    return numpyro.deterministic("log_rho", log_rho)


def _sample_infection_subclasslevel(n_subclass, n_class, class_of_subclass, priors: ModelPriors):
    """log_rho_s = mu_rho + sigma_u * u_class[class(s)] + sigma_w * w_raw_s.

    ``n_class`` must be a concrete Python int (static shape); deriving it from a
    traced ``class_of_subclass`` array would make ``.expand([n_class])`` dynamic.
    """
    mu_rho = numpyro.sample("mu_rho", dist.Normal(priors.mu_rho_loc, priors.mu_rho_scale))
    sigma_u = numpyro.sample("sigma_u", dist.HalfNormal(priors.sigma_u_scale))
    sigma_w = numpyro.sample("sigma_w", dist.HalfNormal(priors.sigma_w_scale))
    u_raw = numpyro.sample("u_raw", dist.Normal(0.0, 1.0).expand([n_class]).to_event(1))
    w_raw = numpyro.sample("w_raw", dist.Normal(0.0, 1.0).expand([n_subclass]).to_event(1))
    log_rho = mu_rho + sigma_u * u_raw[class_of_subclass] + sigma_w * w_raw
    return numpyro.deterministic("log_rho", log_rho)


def _sample_activity_direct(n_group, n_cre, priors: ModelPriors, negative_control_mask=None):
    """Exchangeable raw activity with no cCRE/class/subclass decomposition."""
    if negative_control_mask is not None:
        raise ValueError(
            "direct activity requires ordinary negative controls; pooled/shared "
            "negative-control parameters are not supported"
        )
    mu_gamma = numpyro.sample(
        "mu_gamma", dist.Normal(0.0, priors.mu_gamma_scale)
    )
    sigma_gamma = numpyro.sample(
        "sigma_gamma", dist.HalfNormal(priors.sigma_gamma_scale)
    )
    gamma_raw = numpyro.sample(
        "gamma_raw",
        dist.Normal(0.0, 1.0).expand([n_group, n_cre]).to_event(2),
    )
    return numpyro.deterministic(
        "log_gamma", mu_gamma + sigma_gamma * gamma_raw
    )


def _sample_activity_classlevel(
    n_group,
    n_cre,
    priors: ModelPriors,
    negative_control_mask=None,
    activity_model="hierarchical",
):
    """log_gamma_{g,j} = alpha_j + eta_{g,j}; controls share alpha/eta by group."""
    if activity_model == "direct":
        return _sample_activity_direct(
            n_group, n_cre, priors, negative_control_mask
        )
    if activity_model != "hierarchical":
        raise ValueError(f"unsupported activity_model={activity_model}")
    mu_alpha = numpyro.sample("mu_alpha", dist.Normal(0.0, priors.mu_alpha_scale))
    sigma_alpha = numpyro.sample("sigma_alpha", dist.HalfNormal(priors.sigma_alpha_scale))
    alpha_raw = numpyro.sample("alpha_raw", dist.Normal(0.0, 1.0).expand([n_cre]).to_event(1))
    alpha = mu_alpha + sigma_alpha * alpha_raw
    sigma_eta = numpyro.sample("sigma_eta", dist.HalfNormal(priors.sigma_eta_scale))
    eta_raw = numpyro.sample("eta_raw", dist.Normal(0.0, 1.0).expand([n_group, n_cre]).to_event(2))
    eta = sigma_eta * eta_raw
    if negative_control_mask is not None:
        mask = jnp.asarray(negative_control_mask, dtype=bool)
        alpha_neg = numpyro.sample("alpha_neg", dist.Normal(mu_alpha, sigma_alpha))
        eta_neg_raw = numpyro.sample(
            "eta_neg_raw", dist.Normal(0.0, 1.0).expand([n_group]).to_event(1)
        )
        eta_neg = numpyro.deterministic("eta_neg", sigma_eta * eta_neg_raw)
        alpha = jnp.where(mask, alpha_neg, alpha)
        eta = jnp.where(mask[None, :], eta_neg[:, None], eta)
        numpyro.deterministic("log_gamma_neg", alpha_neg + eta_neg)
    alpha = numpyro.deterministic("alpha", alpha)
    eta = numpyro.deterministic("eta", eta)
    log_gamma = alpha[None, :] + eta
    return numpyro.deterministic("log_gamma", log_gamma)


def _sample_activity_subclasslevel(
    n_subclass,
    n_class,
    n_cre,
    class_of_subclass,
    priors: ModelPriors,
    negative_control_mask=None,
    activity_model="hierarchical",
):
    """log_gamma_{s,j} = alpha_j + eta_{class(s),j} + delta_{s,j}.

    Negative-control cCREs share one alpha, one class-level pattern, and one
    subclass-level pattern, so they vary by class/subclass but not by
    negative-control cCRE identity.
    """
    if activity_model == "direct":
        return _sample_activity_direct(
            n_subclass, n_cre, priors, negative_control_mask
        )
    if activity_model != "hierarchical":
        raise ValueError(f"unsupported activity_model={activity_model}")
    mu_alpha = numpyro.sample("mu_alpha", dist.Normal(0.0, priors.mu_alpha_scale))
    sigma_alpha = numpyro.sample("sigma_alpha", dist.HalfNormal(priors.sigma_alpha_scale))
    alpha_raw = numpyro.sample("alpha_raw", dist.Normal(0.0, 1.0).expand([n_cre]).to_event(1))
    alpha = mu_alpha + sigma_alpha * alpha_raw
    sigma_eta = numpyro.sample("sigma_eta", dist.HalfNormal(priors.sigma_eta_scale))
    eta_raw = numpyro.sample("eta_raw", dist.Normal(0.0, 1.0).expand([n_class, n_cre]).to_event(2))
    eta = sigma_eta * eta_raw
    sigma_delta = numpyro.sample("sigma_delta", dist.HalfNormal(priors.sigma_delta_scale))
    delta_raw = numpyro.sample("delta_raw", dist.Normal(0.0, 1.0).expand([n_subclass, n_cre]).to_event(2))
    delta = sigma_delta * delta_raw
    if negative_control_mask is not None:
        mask = jnp.asarray(negative_control_mask, dtype=bool)
        alpha_neg = numpyro.sample("alpha_neg", dist.Normal(mu_alpha, sigma_alpha))
        eta_neg_raw = numpyro.sample(
            "eta_neg_raw", dist.Normal(0.0, 1.0).expand([n_class]).to_event(1)
        )
        eta_neg = numpyro.deterministic("eta_neg", sigma_eta * eta_neg_raw)
        delta_neg_raw = numpyro.sample(
            "delta_neg_raw", dist.Normal(0.0, 1.0).expand([n_subclass]).to_event(1)
        )
        delta_neg = numpyro.deterministic("delta_neg", sigma_delta * delta_neg_raw)
        alpha = jnp.where(mask, alpha_neg, alpha)
        eta = jnp.where(mask[None, :], eta_neg[:, None], eta)
        delta = jnp.where(mask[None, :], delta_neg[:, None], delta)
        numpyro.deterministic(
            "log_gamma_neg",
            alpha_neg + eta_neg[class_of_subclass] + delta_neg,
        )
    alpha = numpyro.deterministic("alpha", alpha)
    eta = numpyro.deterministic("eta", eta)
    delta = numpyro.deterministic("delta", delta)
    log_gamma = alpha[None, :] + eta[class_of_subclass, :] + delta
    return numpyro.deterministic("log_gamma", log_gamma)


def _obs_factor(
    stats: CollapsedStats,
    lam_row,
    beta_t7,
    phi_t7,
    gamma_row,
    phi_cre,
    kmax,
    p_drop_t7=None,
    p_drop_cre=None,
):
    """Add the weighted marginal log-likelihood to the joint density."""
    ll = marginal_loglik(
        stats,
        lam_row,
        beta_t7,
        phi_t7,
        gamma_row,
        phi_cre,
        kmax,
        p_drop_t7=p_drop_t7,
        p_drop_cre=p_drop_cre,
    )
    numpyro.factor("obs", jnp.sum(stats.weight * ll))


def _cre_conditional_obs_factor(
    stats: CollapsedStats,
    log_lambda_mean_row,
    log_lambda_sd_row,
    gh_nodes,
    gh_log_weights,
    gamma_row,
    phi_cre,
    p_drop_cre,
    kmax,
):
    """Add the CRE-only conditional likelihood to the joint density."""
    ll = cre_marginal_loglik(
        stats,
        log_lambda_mean_row,
        log_lambda_sd_row,
        gh_nodes,
        gh_log_weights,
        gamma_row,
        phi_cre,
        p_drop_cre,
        kmax,
    )
    numpyro.factor("obs", jnp.sum(stats.weight * ll))


def _binary_obs_factor(stats: CollapsedStats, infection_rate_row, beta_t7, phi_t7,
                       gamma_row, phi_cre):
    """Add the weighted shared-gate zero-inflated NB likelihood."""
    ll = binary_infection_loglik(
        stats, infection_rate_row, beta_t7, phi_t7, gamma_row, phi_cre)
    numpyro.factor("obs", jnp.sum(stats.weight * ll))


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
def model_t7_classlevel(stats: CollapsedStats, lib_size_centered, kmax: int,
                        negative_control_mask=None,
                        priors: ModelPriors = ModelPriors(), observe: bool = True):
    """Stage-1 T7-only infection calibration (group == class). Fits rho, a, beta_t7, phi_t7.

    ``observe=False`` skips the (M x Kmax) likelihood factor — used to cheaply replay
    the deterministic sites under posterior draws without re-materialising the tensor.
    """
    log_a = _sample_abundance(lib_size_centered, priors)
    log_rho = _sample_infection_classlevel(stats.n_group, priors)
    beta_t7, phi_t7 = _sample_t7_params(priors)
    if observe:
        lam_row = jnp.exp(log_rho[stats.group] + log_a[stats.cre])
        _obs_factor(stats, lam_row, beta_t7, phi_t7, None, None, kmax)


def model_t7_full(stats: CollapsedStats, lib_size_centered, kmax: int,
                  negative_control_mask=None,
                  priors: ModelPriors = ModelPriors(), observe: bool = True):
    """T7-only infection calibration at subclass granularity."""
    assert stats.class_of_group is not None, "model_t7_full needs class_of_group on stats"
    n_class = int(stats.n_class)
    log_a = _sample_abundance(lib_size_centered, priors)
    log_rho = _sample_infection_subclasslevel(
        stats.n_group, n_class, stats.class_of_group, priors
    )
    beta_t7, phi_t7 = _sample_t7_params(priors)
    if observe:
        lam_row = jnp.exp(log_rho[stats.group] + log_a[stats.cre])
        _obs_factor(stats, lam_row, beta_t7, phi_t7, None, None, kmax)


def model_t7_full_dropout(stats: CollapsedStats, lib_size_centered, kmax: int,
                          negative_control_mask=None,
                          priors: ModelPriors = ModelPriors(), observe: bool = True):
    """T7-only subclass infection calibration with one global T7 dropout rate."""
    assert stats.class_of_group is not None, "model_t7_full_dropout needs class_of_group on stats"
    n_class = int(stats.n_class)
    log_a = _sample_abundance(lib_size_centered, priors)
    log_rho = _sample_infection_subclasslevel(
        stats.n_group, n_class, stats.class_of_group, priors
    )
    beta_t7, phi_t7 = _sample_t7_params(priors)
    p_drop_t7 = _sample_t7_dropout(priors)
    if observe:
        lam_row = jnp.exp(log_rho[stats.group] + log_a[stats.cre])
        _obs_factor(
            stats, lam_row, beta_t7, phi_t7, None, None, kmax,
            p_drop_t7=p_drop_t7,
        )


def model_classlevel(stats: CollapsedStats, lib_size_centered, kmax: int,
                     negative_control_mask=None,
                     priors: ModelPriors = ModelPriors(), activity_model="hierarchical",
                     observe: bool = True):
    """Stage-2 joint CRE+T7 model at class granularity (group == class)."""
    log_a = _sample_abundance(lib_size_centered, priors)
    log_rho = _sample_infection_classlevel(stats.n_group, priors)
    beta_t7, phi_t7 = _sample_t7_params(priors)
    log_gamma = _sample_activity_classlevel(
        stats.n_group, stats.n_cre, priors, negative_control_mask, activity_model
    )
    phi_cre = numpyro.sample("phi_cre", dist.HalfNormal(priors.phi_cre_scale))
    if observe:
        lam_row = jnp.exp(log_rho[stats.group] + log_a[stats.cre])
        gamma_row = jnp.exp(log_gamma[stats.group, stats.cre])
        _obs_factor(stats, lam_row, beta_t7, phi_t7, gamma_row, phi_cre, kmax)


def model_full(stats: CollapsedStats, lib_size_centered, kmax: int,
               negative_control_mask=None,
               priors: ModelPriors = ModelPriors(), activity_model="hierarchical",
               observe: bool = True):
    """Stage-3 full model: subclass nested in class (group == subclass)."""
    assert stats.class_of_group is not None, "model_full needs class_of_group on stats"
    n_class = int(stats.n_class)
    log_a = _sample_abundance(lib_size_centered, priors)
    log_rho = _sample_infection_subclasslevel(stats.n_group, n_class, stats.class_of_group, priors)
    beta_t7, phi_t7 = _sample_t7_params(priors)
    log_gamma = _sample_activity_subclasslevel(
        stats.n_group, n_class, stats.n_cre, stats.class_of_group, priors,
        negative_control_mask, activity_model)
    phi_cre = numpyro.sample("phi_cre", dist.HalfNormal(priors.phi_cre_scale))
    if observe:
        lam_row = jnp.exp(log_rho[stats.group] + log_a[stats.cre])
        gamma_row = jnp.exp(log_gamma[stats.group, stats.cre])
        _obs_factor(stats, lam_row, beta_t7, phi_t7, gamma_row, phi_cre, kmax)


def model_classlevel_dropout(stats: CollapsedStats, lib_size_centered, kmax: int,
                             negative_control_mask=None,
                             priors: ModelPriors = ModelPriors(), activity_model="hierarchical",
                             observe: bool = True):
    """Joint class model with global T7 and CRE zero-inflated dropout rates."""
    log_a = _sample_abundance(lib_size_centered, priors)
    log_rho = _sample_infection_classlevel(stats.n_group, priors)
    beta_t7, phi_t7 = _sample_t7_params(priors)
    p_drop_t7 = _sample_t7_dropout(priors)
    log_gamma = _sample_activity_classlevel(
        stats.n_group, stats.n_cre, priors, negative_control_mask, activity_model
    )
    phi_cre = numpyro.sample("phi_cre", dist.HalfNormal(priors.phi_cre_scale))
    p_drop_cre = _sample_cre_dropout(priors)
    if observe:
        lam_row = jnp.exp(log_rho[stats.group] + log_a[stats.cre])
        gamma_row = jnp.exp(log_gamma[stats.group, stats.cre])
        _obs_factor(
            stats,
            lam_row,
            beta_t7,
            phi_t7,
            gamma_row,
            phi_cre,
            kmax,
            p_drop_t7=p_drop_t7,
            p_drop_cre=p_drop_cre,
        )


def model_full_dropout(stats: CollapsedStats, lib_size_centered, kmax: int,
                       negative_control_mask=None,
                       priors: ModelPriors = ModelPriors(), activity_model="hierarchical",
                       observe: bool = True):
    """Joint subclass model with global T7 and CRE zero-inflated dropout rates."""
    assert stats.class_of_group is not None, "model_full_dropout needs class_of_group on stats"
    n_class = int(stats.n_class)
    log_a = _sample_abundance(lib_size_centered, priors)
    log_rho = _sample_infection_subclasslevel(stats.n_group, n_class, stats.class_of_group, priors)
    beta_t7, phi_t7 = _sample_t7_params(priors)
    p_drop_t7 = _sample_t7_dropout(priors)
    log_gamma = _sample_activity_subclasslevel(
        stats.n_group, n_class, stats.n_cre, stats.class_of_group, priors,
        negative_control_mask, activity_model)
    phi_cre = numpyro.sample("phi_cre", dist.HalfNormal(priors.phi_cre_scale))
    p_drop_cre = _sample_cre_dropout(priors)
    if observe:
        lam_row = jnp.exp(log_rho[stats.group] + log_a[stats.cre])
        gamma_row = jnp.exp(log_gamma[stats.group, stats.cre])
        _obs_factor(
            stats,
            lam_row,
            beta_t7,
            phi_t7,
            gamma_row,
            phi_cre,
            kmax,
            p_drop_t7=p_drop_t7,
            p_drop_cre=p_drop_cre,
        )


def model_cre_conditional_subclass(
    stats: CollapsedStats,
    log_lambda_mean,
    log_lambda_sd,
    gh_nodes,
    gh_log_weights,
    kmax: int,
    negative_control_mask=None,
    priors: ModelPriors = ModelPriors(),
    observe: bool = True,
):
    """CRE-only subclass activity model conditioned on T7 infection posterior."""
    assert stats.class_of_group is not None, "model_cre_conditional_subclass needs class_of_group"
    n_class = int(stats.n_class)
    log_gamma = _sample_activity_subclasslevel(
        stats.n_group, n_class, stats.n_cre, stats.class_of_group, priors, negative_control_mask
    )
    phi_cre = numpyro.sample("phi_cre", dist.HalfNormal(priors.phi_cre_scale))
    p_drop_cre = _sample_cre_dropout(priors)
    if observe:
        gamma_row = jnp.exp(log_gamma[stats.group, stats.cre])
        _cre_conditional_obs_factor(
            stats,
            log_lambda_mean[stats.group, stats.cre],
            log_lambda_sd[stats.group, stats.cre],
            gh_nodes,
            gh_log_weights,
            gamma_row,
            phi_cre,
            p_drop_cre,
            kmax,
        )


def model_cre_conditional_subclass_no_dropout(
    stats: CollapsedStats,
    log_lambda_mean,
    log_lambda_sd,
    gh_nodes,
    gh_log_weights,
    kmax: int,
    negative_control_mask=None,
    priors: ModelPriors = ModelPriors(),
    observe: bool = True,
):
    """CRE-only subclass activity model conditioned on T7 posterior, without dropout."""
    assert stats.class_of_group is not None, "model_cre_conditional_subclass_no_dropout needs class_of_group"
    n_class = int(stats.n_class)
    log_gamma = _sample_activity_subclasslevel(
        stats.n_group, n_class, stats.n_cre, stats.class_of_group, priors, negative_control_mask
    )
    phi_cre = numpyro.sample("phi_cre", dist.HalfNormal(priors.phi_cre_scale))
    if observe:
        gamma_row = jnp.exp(log_gamma[stats.group, stats.cre])
        _cre_conditional_obs_factor(
            stats,
            log_lambda_mean[stats.group, stats.cre],
            log_lambda_sd[stats.group, stats.cre],
            gh_nodes,
            gh_log_weights,
            gamma_row,
            phi_cre,
            None,
            kmax,
        )


def model_binary_t7_classlevel(stats: CollapsedStats, lib_size_centered, kmax: int,
                               negative_control_mask=None,
                               priors: ModelPriors = ModelPriors(), observe: bool = True):
    """T7-only class model with a shared binary infection event."""
    log_a = _sample_abundance(lib_size_centered, priors)
    log_rho = _sample_infection_classlevel(stats.n_group, priors)
    beta_t7, phi_t7 = _sample_t7_params(priors)
    if observe:
        infection_rate_row = jnp.exp(log_rho[stats.group] + log_a[stats.cre])
        _binary_obs_factor(stats, infection_rate_row, beta_t7, phi_t7, None, None)


def model_binary_classlevel(stats: CollapsedStats, lib_size_centered, kmax: int,
                            negative_control_mask=None,
                            priors: ModelPriors = ModelPriors(), activity_model="hierarchical",
                            observe: bool = True):
    """Joint class model with a shared binary infection event."""
    log_a = _sample_abundance(lib_size_centered, priors)
    log_rho = _sample_infection_classlevel(stats.n_group, priors)
    beta_t7, phi_t7 = _sample_t7_params(priors)
    log_gamma = _sample_activity_classlevel(
        stats.n_group, stats.n_cre, priors, negative_control_mask, activity_model
    )
    phi_cre = numpyro.sample("phi_cre", dist.HalfNormal(priors.phi_cre_scale))
    if observe:
        infection_rate_row = jnp.exp(log_rho[stats.group] + log_a[stats.cre])
        gamma_row = jnp.exp(log_gamma[stats.group, stats.cre])
        _binary_obs_factor(stats, infection_rate_row, beta_t7, phi_t7, gamma_row, phi_cre)


def model_binary_full(stats: CollapsedStats, lib_size_centered, kmax: int,
                      negative_control_mask=None,
                      priors: ModelPriors = ModelPriors(), activity_model="hierarchical",
                      observe: bool = True):
    """Joint subclass model with a shared binary infection event."""
    assert stats.class_of_group is not None, "model_binary_full needs class_of_group on stats"
    n_class = int(stats.n_class)
    log_a = _sample_abundance(lib_size_centered, priors)
    log_rho = _sample_infection_subclasslevel(stats.n_group, n_class, stats.class_of_group, priors)
    beta_t7, phi_t7 = _sample_t7_params(priors)
    log_gamma = _sample_activity_subclasslevel(
        stats.n_group, n_class, stats.n_cre, stats.class_of_group, priors,
        negative_control_mask, activity_model)
    phi_cre = numpyro.sample("phi_cre", dist.HalfNormal(priors.phi_cre_scale))
    if observe:
        infection_rate_row = jnp.exp(log_rho[stats.group] + log_a[stats.cre])
        gamma_row = jnp.exp(log_gamma[stats.group, stats.cre])
        _binary_obs_factor(stats, infection_rate_row, beta_t7, phi_t7, gamma_row, phi_cre)


MODELS = {
    ("class", "t7"): model_t7_classlevel,
    ("subclass", "t7"): model_t7_full,
    ("class", "joint"): model_classlevel,
    ("subclass", "joint"): model_full,
}

COPY_NUMBER_DROPOUT_MODELS = {
    ("class", "t7"): model_t7_classlevel,
    ("subclass", "t7"): model_t7_full_dropout,
    ("class", "joint"): model_classlevel_dropout,
    ("subclass", "joint"): model_full_dropout,
}

BINARY_INFECTION_MODELS = {
    ("class", "t7"): model_binary_t7_classlevel,
    ("class", "joint"): model_binary_classlevel,
    ("subclass", "joint"): model_binary_full,
}

MODEL_FAMILIES = {
    "copy_number": MODELS,
    "copy_number_dropout": COPY_NUMBER_DROPOUT_MODELS,
    "binary": BINARY_INFECTION_MODELS,
}


# --------------------------------------------------------------------------- #
# Method-of-moments initialisation (avoids needing to run the torch EM)
# --------------------------------------------------------------------------- #
def init_from_moments(stats: CollapsedStats, lib_size_centered, priors: ModelPriors,
                      level: str, channel: str, negative_control_mask=None,
                      activity_model: str = "hierarchical") -> dict:
    """Crude method-of-moments init for the raw (non-centered) sites.

    Anchors ``beta_t7`` at the mean T7 among T7-positive rows (k~=1 under rare
    infection), ``rho`` from per-group T7-positive fractions, and CRE baseline
    ``alpha`` from mean CRE among CRE-positive rows. Returned dict is suitable
    for :func:`numpyro.infer.initialization.init_to_value`.
    """
    w = np.asarray(stats.weight)
    t7 = np.asarray(stats.counts["t7"])
    grp = np.asarray(stats.group)
    cre_idx = np.asarray(stats.cre)
    lib = np.asarray(lib_size_centered)

    pos = t7 > 0
    beta_t7 = float(np.average(t7[pos], weights=w[pos])) if pos.any() else 1.0
    beta_t7 = max(beta_t7, 1.0)

    # per-group T7-positive fraction -> rho_g ~ mean_j[ -log(1-p_{g,j}) / a_j ]
    a_j = np.exp(lib)  # mean(log a)=0 => a_j ~ relative abundance
    n_per_group = np.asarray(stats.n_per_group)
    pos_weight = np.where(pos, w, 0.0)
    n_pos_gj = np.zeros((stats.n_group, stats.n_cre))
    np.add.at(n_pos_gj, (grp, cre_idx), pos_weight)
    p_gj = n_pos_gj / np.maximum(n_per_group[:, None], 1)
    lam_gj = -np.log1p(-np.clip(p_gj, 0, 1 - 1e-9))
    with np.errstate(divide="ignore", invalid="ignore"):
        rho_g = np.nanmean(np.where(a_j[None, :] > 0, lam_gj / a_j[None, :], np.nan), axis=1)
    rho_g = np.clip(np.nan_to_num(rho_g, nan=np.exp(priors.mu_rho_loc)), 1e-6, None)
    log_rho_g = np.log(rho_g)
    mu_rho = float(np.mean(log_rho_g))

    init = {
        "tau_a": np.float64(priors.tau_a_scale * 0.5),
        "eps_a_raw": np.zeros(stats.n_cre),
        "beta_t7": np.float64(beta_t7),
        "phi_t7": np.float64(2.0),
        "mu_rho": np.float64(mu_rho),
        "sigma_u": np.float64(0.5),
        "u_raw": np.zeros(stats.n_group if level == "class" else int(stats.n_class)),
    }
    if level == "subclass":
        init["sigma_w"] = np.float64(0.5)
        init["w_raw"] = np.zeros(stats.n_group)

    if channel == "joint":
        cre = np.asarray(stats.counts["cre"])
        cpos = cre > 0
        gamma0 = float(np.average(cre[cpos], weights=w[cpos])) if cpos.any() else beta_t7
        if activity_model == "direct":
            if negative_control_mask is not None:
                raise ValueError(
                    "direct activity requires ordinary negative controls"
                )
            init.update({
                "mu_gamma": np.float64(np.log(max(gamma0, 1.0))),
                "sigma_gamma": np.float64(1.0),
                "gamma_raw": np.zeros((stats.n_group, stats.n_cre)),
                "phi_cre": np.float64(2.0),
            })
        else:
            if activity_model != "hierarchical":
                raise ValueError(f"unsupported activity_model={activity_model}")
            init.update({
                "mu_alpha": np.float64(np.log(max(gamma0, 1.0))),
                "sigma_alpha": np.float64(1.0),
                "alpha_raw": np.zeros(stats.n_cre),
                "sigma_eta": np.float64(0.5),
                "phi_cre": np.float64(2.0),
            })
            if level == "class":
                init["eta_raw"] = np.zeros((stats.n_group, stats.n_cre))
            else:
                init["eta_raw"] = np.zeros((int(stats.n_class), stats.n_cre))
                init["sigma_delta"] = np.float64(0.5)
                init["delta_raw"] = np.zeros((stats.n_group, stats.n_cre))
            if negative_control_mask is not None:
                init["alpha_neg"] = np.float64(np.log(max(gamma0, 1.0)))
                if level == "class":
                    init["eta_neg_raw"] = np.zeros(stats.n_group)
                else:
                    init["eta_neg_raw"] = np.zeros(int(stats.n_class))
                    init["delta_neg_raw"] = np.zeros(stats.n_group)
    return init


# --------------------------------------------------------------------------- #
# Inference drivers
# --------------------------------------------------------------------------- #
def fit_svi(model, stats: CollapsedStats, lib_size_centered, kmax: int,
            priors: ModelPriors = ModelPriors(), *, negative_control_mask=None,
            init_values: dict | None = None,
            num_steps: int = 20000, lr: float = 5e-3, guide: str = "AutoNormal",
            num_posterior: int = 1000, seed: int = 0):
    """Fit by stochastic variational inference.

    Returns ``(samples, info)`` where ``samples`` is a dict of posterior draws
    (drawn from the fitted guide) and ``info`` holds ``losses``, ``params`` and
    the ``guide`` object.
    """
    init_loc = init_to_value(values=init_values) if init_values else None
    guide_cls = {"AutoNormal": AutoNormal,
                 "AutoLowRankMultivariateNormal": AutoLowRankMultivariateNormal}[guide]
    guide_obj = guide_cls(model, init_loc_fn=init_loc) if init_loc else guide_cls(model)

    svi = SVI(model, guide_obj, numpyro.optim.Adam(lr), loss=Trace_ELBO())
    result = svi.run(jax.random.PRNGKey(seed), num_steps, stats, lib_size_centered,
                     kmax, negative_control_mask, priors)

    pred = Predictive(guide_obj, params=result.params, num_samples=num_posterior)
    samples = pred(jax.random.PRNGKey(seed + 1))
    # Recover deterministic sites by replaying the model on guide draws.
    samples = _add_deterministics(model, samples, stats, lib_size_centered, kmax,
                                  negative_control_mask, priors, seed + 2)
    return samples, {"losses": np.asarray(result.losses), "params": result.params, "guide": guide_obj}


def fit_nuts(model, stats: CollapsedStats, lib_size_centered, kmax: int,
             priors: ModelPriors = ModelPriors(), *, negative_control_mask=None,
             init_values: dict | None = None,
             num_warmup: int = 1000, num_samples: int = 1000, num_chains: int = 2,
             seed: int = 0):
    """Fit by NUTS. Feasible at class granularity; avoid on the full subclass model."""
    init_strategy = init_to_value(values=init_values) if init_values else None
    kernel = NUTS(model, init_strategy=init_strategy) if init_strategy else NUTS(model)
    mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples,
                num_chains=num_chains, progress_bar=True)
    mcmc.run(jax.random.PRNGKey(seed), stats, lib_size_centered, kmax,
             negative_control_mask, priors)
    # NUTS get_samples() omits deterministic sites; materialise them by replay.
    samples = _add_deterministics(model, mcmc.get_samples(), stats, lib_size_centered,
                                  kmax, negative_control_mask, priors, seed + 1)
    return samples, {"mcmc": mcmc}


def _add_deterministics(model, samples, stats, lib_size_centered, kmax,
                        negative_control_mask, priors, seed):
    """Replay the model under posterior draws to materialise deterministic sites."""
    from functools import partial
    from numpyro.handlers import substitute, trace, seed as seed_handler

    # observe=False skips the (M x Kmax) likelihood factor, so the vmap over all
    # posterior draws only materialises the cheap deterministic transforms
    # (log_rho/log_a/log_gamma) instead of a (n_draws x M x Kmax) tensor.
    det_model = partial(model, observe=False)

    def one(draw):
        tr = trace(seed_handler(substitute(det_model, draw), jax.random.PRNGKey(seed))).get_trace(
            stats, lib_size_centered, kmax, negative_control_mask, priors)
        return {k: site["value"] for k, site in tr.items() if site["type"] == "deterministic"}

    det = jax.vmap(one)(dict(samples))
    out = dict(samples)
    out.update(det)
    return out


# --------------------------------------------------------------------------- #
# Posterior summaries and predictive checks
# --------------------------------------------------------------------------- #
def _ci(arr, axis=0, lo=5, hi=95):
    return np.percentile(arr, lo, axis=axis), np.percentile(arr, hi, axis=axis)


def summarize_posterior(samples: Mapping[str, np.ndarray], stats: CollapsedStats,
                        evidence: dict, cre_names: Sequence[str],
                        group_names: Sequence[str], level: str,
                        prior_dominated_basis: str = "double") -> dict:
    """Posterior means + 90% CIs for the scientific parameters, joined to evidence.

    Returns a dict of tidy DataFrames. Every per-(group, cre) row carries its
    evidence counts and a ``prior_dominated`` flag (no double-positive support),
    so prior/hierarchy-driven estimates are never mistaken for measured ones.
    """
    out = {}

    if "log_rho" in samples:
        rho = np.exp(np.asarray(samples["log_rho"]))      # (n_draws, n_group)
        lo, hi = _ci(rho)
        out["rho"] = pd.DataFrame({
            "group": list(group_names),
            "rho_mean": rho.mean(0), "rho_lo": lo, "rho_hi": hi,
            "n_cells": np.asarray(stats.n_per_group),
        })

    if "log_gamma" in samples:
        g = np.exp(np.asarray(samples["log_gamma"]))      # (n_draws, n_group, n_cre)
        gm = g.mean(0); glo, ghi = _ci(g)
        n_group, n_cre = gm.shape
        gg, cc = np.meshgrid(np.arange(n_group), np.arange(n_cre), indexing="ij")
        gamma = pd.DataFrame({
            "group_idx": gg.ravel(), "cre_idx": cc.ravel(),
            "group": np.asarray(group_names)[gg.ravel()],
            "cre": np.asarray(cre_names)[cc.ravel()],
            "gamma_mean": gm.ravel(), "gamma_lo": glo.ravel(), "gamma_hi": ghi.ravel(),
            "ci_width": (ghi - glo).ravel(),
        })
        # vectorised join of evidence counts on integer (group, cre) keys
        ev_cols = [
            col for col in ("n_t7_pos", "n_cre_pos", "n_double_pos", "n_total")
            if col in evidence["per_pair"]
        ]
        ev = evidence["per_pair"][["group", "cre", *ev_cols]].rename(
            columns={"group": "group_idx", "cre": "cre_idx"})
        gamma = gamma.merge(ev, on=["group_idx", "cre_idx"], how="left")
        for c in ev_cols:
            gamma[c] = gamma[c].fillna(0).astype(np.int64)
        if prior_dominated_basis == "cre":
            gamma["prior_dominated"] = gamma.get("n_cre_pos", pd.Series(0, index=gamma.index)) == 0
        else:
            gamma["prior_dominated"] = gamma.get("n_double_pos", pd.Series(0, index=gamma.index)) == 0
        out["gamma"] = gamma.drop(columns=["group_idx", "cre_idx"])

    if "delta" in samples:
        d = np.asarray(samples["delta"])                  # (n_draws, n_subclass, n_cre)
        out["delta_mean"] = pd.DataFrame(d.mean(0), index=list(group_names), columns=list(cre_names))

    return out


def summarize_binary_infection(samples: Mapping[str, np.ndarray],
                               cre_names: Sequence[str],
                               group_names: Sequence[str]) -> pd.DataFrame:
    """Summarize the explicit binary infection hazard/probability for each pair.

    Computation is group-chunked to avoid materializing a full
    ``draw x group x cCRE`` tensor for subclass fits.
    """
    log_rho = np.asarray(samples["log_rho"])
    log_a = np.asarray(samples["log_a"])
    frames = []
    for group_idx, group_name in enumerate(group_names):
        rate = np.exp(log_rho[:, group_idx, None] + log_a)
        probability = -np.expm1(-rate)
        rate_lo, rate_hi = _ci(rate)
        prob_lo, prob_hi = _ci(probability)
        frames.append(pd.DataFrame({
            "group": group_name,
            "cre": list(cre_names),
            "infection_rate_mean": rate.mean(0),
            "infection_rate_lo": rate_lo,
            "infection_rate_hi": rate_hi,
            "infection_probability_mean": probability.mean(0),
            "infection_probability_lo": prob_lo,
            "infection_probability_hi": prob_hi,
        }))
    return pd.concat(frames, ignore_index=True)


def posterior_predictive_check(samples: Mapping[str, np.ndarray], stats: CollapsedStats,
                               lib_size_centered, kmax: int, level: str,
                               n_draws: int = 100, seed: int = 0,
                               infection_model: str = "copy_number") -> dict:
    """Compare observed vs posterior-predictive zero-fraction & mean-nonzero per channel.

    Draws the selected latent infection state, then channel counts, and aggregates
    the same statistics computed on the observed collapsed table.
    """
    rng = np.random.default_rng(seed)
    n_total = float(np.asarray(stats.weight).sum())

    obs = {}
    for name in stats.channels:
        c = np.asarray(stats.counts[name]); w = np.asarray(stats.weight)
        obs[name] = {"zero_fraction": float(w[c == 0].sum() / n_total),
                     "mean_nonzero": float(np.average(c[c > 0], weights=w[c > 0])) if (c > 0).any() else 0.0}

    log_rho = np.asarray(samples["log_rho"]); log_a = np.asarray(samples["log_a"])
    beta = np.asarray(samples["beta_t7"]); phi_t7 = np.asarray(samples["phi_t7"])
    p_drop_t7 = np.asarray(samples["p_drop_t7"]) if "p_drop_t7" in samples else None
    p_drop_cre = np.asarray(samples["p_drop_cre"]) if "p_drop_cre" in samples else None
    has_cre = "log_gamma" in samples
    if has_cre:
        log_gamma = np.asarray(samples["log_gamma"]); phi_cre = np.asarray(samples["phi_cre"])
    grp = np.asarray(stats.group); cre_idx = np.asarray(stats.cre); w = np.asarray(stats.weight)

    draw_ids = rng.integers(0, log_rho.shape[0], size=n_draws)
    rep = {name: {"zero_fraction": [], "mean_nonzero": []} for name in stats.channels}
    for d in draw_ids:
        infection_rate = np.exp(log_rho[d][grp] + log_a[d][cre_idx])
        if infection_model in {"copy_number", "copy_number_dropout"}:
            latent_multiplier = rng.poisson(infection_rate)
        else:
            p_infected = -np.expm1(-infection_rate)
            latent_multiplier = rng.binomial(1, p_infected)
        for name in stats.channels:
            if name == "t7":
                mean = beta[d] * latent_multiplier; phi = phi_t7[d]
                p_drop = None if p_drop_t7 is None else p_drop_t7[d]
            else:
                mean = np.exp(log_gamma[d][grp, cre_idx]) * latent_multiplier; phi = phi_cre[d]
                p_drop = None if p_drop_cre is None else p_drop_cre[d]
            sim = _nb2_sample(rng, mean, phi)
            if p_drop is not None:
                drop = rng.binomial(1, p_drop, size=sim.shape).astype(bool)
                sim = np.where((latent_multiplier > 0) & drop, 0, sim)
            sim = np.where(latent_multiplier == 0, 0, sim)
            rep[name]["zero_fraction"].append(w[sim == 0].sum() / n_total)
            nz = sim > 0
            rep[name]["mean_nonzero"].append(np.average(sim[nz], weights=w[nz]) if nz.any() else 0.0)

    summary = {}
    for name in stats.channels:
        summary[name] = {
            "obs": obs[name],
            "rep_zero_fraction": (float(np.mean(rep[name]["zero_fraction"])), *(_ci(np.array(rep[name]["zero_fraction"])))),
            "rep_mean_nonzero": (float(np.mean(rep[name]["mean_nonzero"])), *(_ci(np.array(rep[name]["mean_nonzero"])))),
        }
    return summary


def _nb2_sample(rng, mean, conc):
    """Sample NB2(mean, conc) via Gamma-Poisson; ``mean`` array, ``conc`` scalar."""
    mean = np.where(mean <= 0, 1e-9, mean)
    rate = conc / mean
    lam = rng.gamma(shape=conc, scale=1.0 / rate)
    return rng.poisson(lam)


def init_cre_from_moments(
    stats: CollapsedStats,
    priors: ModelPriors,
    negative_control_mask=None,
) -> dict:
    """Method-of-moments initial values for the CRE-only conditional model."""
    w = np.asarray(stats.weight)
    cre = np.asarray(stats.counts["cre"])
    cpos = cre > 0
    gamma0 = float(np.average(cre[cpos], weights=w[cpos])) if cpos.any() else 1.0
    gamma0 = max(gamma0, 1.0)
    init = {
        "mu_alpha": np.float64(np.log(gamma0)),
        "sigma_alpha": np.float64(1.0),
        "alpha_raw": np.zeros(stats.n_cre),
        "sigma_eta": np.float64(0.5),
        "eta_raw": np.zeros((int(stats.n_class), stats.n_cre)),
        "sigma_delta": np.float64(0.5),
        "delta_raw": np.zeros((stats.n_group, stats.n_cre)),
        "phi_cre": np.float64(2.0),
        "p_drop_cre": np.float64(
            priors.p_drop_cre_alpha / (priors.p_drop_cre_alpha + priors.p_drop_cre_beta)
        ),
    }
    if negative_control_mask is not None:
        init["alpha_neg"] = np.float64(np.log(gamma0))
        init["eta_neg_raw"] = np.zeros(int(stats.n_class))
        init["delta_neg_raw"] = np.zeros(stats.n_group)
    return init


def _prepare_grouping(subclass_labels, class_labels, level: str):
    sub = np.asarray(subclass_labels).astype(str)
    cls = np.asarray(class_labels).astype(str)
    sub_cats, sub_idx = np.unique(sub, return_inverse=True)
    cls_cats, cls_idx_cell = np.unique(cls, return_inverse=True)
    mapping = pd.DataFrame({"sub": sub_idx, "cls": cls_idx_cell}).drop_duplicates()
    if mapping["sub"].duplicated().any():
        raise ValueError("subclass does not nest cleanly within class")
    class_of_sub = mapping.sort_values("sub")["cls"].to_numpy().astype(np.int64)

    if level == "class":
        return cls_idx_cell, len(cls_cats), list(cls_cats), None, None
    return sub_idx, len(sub_cats), list(sub_cats), class_of_sub, len(cls_cats)


def summarize_log_lambda_posterior(samples: Mapping[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Compress T7 posterior draws of log(rho_s * a_j) to mean/sd per pair."""
    log_rho = np.asarray(samples["log_rho"], dtype=np.float64)
    log_a = np.asarray(samples["log_a"], dtype=np.float64)
    n_group = log_rho.shape[1]
    n_cre = log_a.shape[1]
    mean = np.empty((n_group, n_cre), dtype=np.float64)
    sd = np.empty((n_group, n_cre), dtype=np.float64)
    for start in range(0, n_group, 32):
        stop = min(start + 32, n_group)
        values = log_rho[:, start:stop, None] + log_a[:, None, :]
        mean[start:stop] = values.mean(axis=0)
        sd[start:stop] = values.std(axis=0)
    return mean, sd


def summarize_lognormal_infection(
    log_lambda_mean: np.ndarray,
    log_lambda_sd: np.ndarray,
    cre_names: Sequence[str],
    group_names: Sequence[str],
) -> pd.DataFrame:
    """Tidy infection-rate summary from lognormal pairwise approximation."""
    z05 = -1.6448536269514722
    z95 = 1.6448536269514722
    rate_mean = np.exp(log_lambda_mean + 0.5 * np.square(log_lambda_sd))
    rate_lo = np.exp(log_lambda_mean + z05 * log_lambda_sd)
    rate_hi = np.exp(log_lambda_mean + z95 * log_lambda_sd)
    gg, cc = np.meshgrid(
        np.arange(len(group_names)), np.arange(len(cre_names)), indexing="ij"
    )
    return pd.DataFrame({
        "group": np.asarray(group_names)[gg.ravel()],
        "cre": np.asarray(cre_names)[cc.ravel()],
        "log_infection_rate_mean": log_lambda_mean.ravel(),
        "log_infection_rate_sd": log_lambda_sd.ravel(),
        "infection_rate_mean": rate_mean.ravel(),
        "infection_rate_lo": rate_lo.ravel(),
        "infection_rate_hi": rate_hi.ravel(),
    })


def _weighted_channel_summary(counts, weight):
    return {
        "zero_fraction": float(weight[counts == 0].sum() / weight.sum()),
        "mean_nonzero": (
            float(np.average(counts[counts > 0], weights=weight[counts > 0]))
            if (counts > 0).any()
            else 0.0
        ),
    }


def _weighted_joint_summary(t7, cre, weight):
    total = float(weight.sum())
    return {
        "all_zero_fraction": float(weight[(t7 == 0) & (cre == 0)].sum() / total),
        "t7_only_fraction": float(weight[(t7 > 0) & (cre == 0)].sum() / total),
        "cre_only_fraction": float(weight[(t7 == 0) & (cre > 0)].sum() / total),
        "double_positive_fraction": float(weight[(t7 > 0) & (cre > 0)].sum() / total),
    }


def _rep_interval(values):
    values = np.asarray(values, dtype=np.float64)
    return (float(values.mean()), *(_ci(values)))


def posterior_predictive_check_decoupled(
    t7_samples: Mapping[str, np.ndarray],
    cre_samples: Mapping[str, np.ndarray],
    stats: CollapsedStats,
    n_draws: int = 100,
    seed: int = 0,
) -> dict:
    """Posterior predictive checks for the decoupled T7 and CRE stages."""
    rng = np.random.default_rng(seed)
    weight = np.asarray(stats.weight, dtype=np.float64)
    t7_obs = np.asarray(stats.counts["t7"])
    cre_obs = np.asarray(stats.counts["cre"])
    grp = np.asarray(stats.group)
    cre_idx = np.asarray(stats.cre)

    obs = {
        "t7": _weighted_channel_summary(t7_obs, weight),
        "cre": _weighted_channel_summary(cre_obs, weight),
        "joint": _weighted_joint_summary(t7_obs, cre_obs, weight),
    }
    rep = {
        "t7": {"zero_fraction": [], "mean_nonzero": []},
        "cre": {"zero_fraction": [], "mean_nonzero": []},
        "joint": {key: [] for key in obs["joint"]},
    }

    log_rho = np.asarray(t7_samples["log_rho"])
    log_a = np.asarray(t7_samples["log_a"])
    beta = np.asarray(t7_samples["beta_t7"])
    phi_t7 = np.asarray(t7_samples["phi_t7"])
    p_drop_t7 = np.asarray(t7_samples["p_drop_t7"]) if "p_drop_t7" in t7_samples else None
    log_gamma = np.asarray(cre_samples["log_gamma"])
    phi_cre = np.asarray(cre_samples["phi_cre"])
    p_drop_cre = np.asarray(cre_samples["p_drop_cre"]) if "p_drop_cre" in cre_samples else None

    t7_draws = rng.integers(0, log_rho.shape[0], size=n_draws)
    cre_draws = rng.integers(0, log_gamma.shape[0], size=n_draws)
    for d_t7, d_cre in zip(t7_draws, cre_draws):
        lam = np.exp(log_rho[d_t7][grp] + log_a[d_t7][cre_idx])
        k_t7 = rng.poisson(lam)
        k_cre = rng.poisson(lam)

        sim_t7 = _nb2_sample(rng, beta[d_t7] * k_t7, phi_t7[d_t7])
        if p_drop_t7 is not None:
            drop_t7 = rng.binomial(1, p_drop_t7[d_t7], size=sim_t7.shape).astype(bool)
            sim_t7 = np.where((k_t7 > 0) & drop_t7, 0, sim_t7)
        sim_t7 = np.where(k_t7 == 0, 0, sim_t7)

        gamma = np.exp(log_gamma[d_cre][grp, cre_idx])
        sim_cre = _nb2_sample(rng, gamma * k_cre, phi_cre[d_cre])
        if p_drop_cre is not None:
            drop_cre = rng.binomial(1, p_drop_cre[d_cre], size=sim_cre.shape).astype(bool)
            sim_cre = np.where((k_cre > 0) & drop_cre, 0, sim_cre)
        sim_cre = np.where(k_cre == 0, 0, sim_cre)

        for name, sim in (("t7", sim_t7), ("cre", sim_cre)):
            summary = _weighted_channel_summary(sim, weight)
            for key, value in summary.items():
                rep[name][key].append(value)
        joint = _weighted_joint_summary(sim_t7, sim_cre, weight)
        for key, value in joint.items():
            rep["joint"][key].append(value)

    return {
        name: {
            "obs": obs[name],
            **{f"rep_{key}": _rep_interval(values) for key, values in rep[name].items()},
        }
        for name in ("t7", "cre", "joint")
    }


def run_decoupled_model(
    t7,
    cre,
    subclass_labels,
    class_labels,
    lib_size_log,
    cre_names,
    *,
    kmax=None,
    priors: "ModelPriors" = None,
    steps_t7=20000,
    steps_cre=20000,
    lr=5e-3,
    guide="AutoNormal",
    num_posterior=1000,
    seed=0,
    verbose=True,
    negative_control_mask=None,
    infection_quadrature_points: int = 7,
    posterior_sites_to_return=None,
    dropout_model: str = "zero_inflated",
) -> dict:
    """Fit the two-stage T7 infection / CRE activity model."""
    if dropout_model not in {"zero_inflated", "none"}:
        raise ValueError("dropout_model must be 'zero_inflated' or 'none'")
    priors = priors or ModelPriors()
    t7 = np.asarray(t7).astype(np.int64)
    cre = np.asarray(cre).astype(np.int64)
    n_cre = len(cre_names)
    if negative_control_mask is not None:
        negative_control_mask = np.asarray(negative_control_mask, dtype=bool)
        if negative_control_mask.shape != (n_cre,):
            raise ValueError("negative_control_mask must have shape (n_cre,)")
        if not negative_control_mask.any():
            negative_control_mask = None
    lib_centered = np.asarray(lib_size_log, dtype=np.float64)
    lib_centered = lib_centered - lib_centered.mean()

    group_idx, n_group, group_names, class_of_group, n_class = _prepare_grouping(
        subclass_labels, class_labels, "subclass"
    )
    t7_stats = build_sufficient_stats(
        {"t7": t7}, group_idx, n_group, n_cre,
        class_of_group=class_of_group, n_class=n_class,
    )
    cre_stats = build_sufficient_stats(
        {"cre": cre}, group_idx, n_group, n_cre,
        class_of_group=class_of_group, n_class=n_class,
    )
    joint_stats = build_sufficient_stats(
        {"t7": t7, "cre": cre}, group_idx, n_group, n_cre,
        class_of_group=class_of_group, n_class=n_class,
    )
    t7_evidence = summarize_evidence(t7_stats)
    cre_evidence = summarize_evidence(cre_stats)
    joint_evidence = summarize_evidence(joint_stats)
    if verbose:
        print("[run_decoupled_model] joint evidence audit:", joint_evidence["totals"])
        print(
            "[run_decoupled_model] collapsed rows: "
            f"t7={len(t7_stats.weight)} cre={len(cre_stats.weight)} "
            f"joint={len(joint_stats.weight)}"
        )

    init_t7 = init_from_moments(t7_stats, lib_centered, priors, "subclass", "t7")
    if dropout_model == "zero_inflated":
        init_t7["p_drop_t7"] = np.float64(
            priors.p_drop_t7_alpha / (priors.p_drop_t7_alpha + priors.p_drop_t7_beta)
        )
    if kmax is None:
        pp = t7_evidence["per_pair"]
        p_t7 = pp["n_t7_pos"].to_numpy() / np.maximum(
            t7_stats.n_per_group[pp["group"].to_numpy()], 1
        )
        lam_max = (
            float(np.max(-np.log1p(-np.clip(p_t7, 0, 1 - 1e-9))))
            if len(p_t7)
            else 0.1
        )
        kmax = choose_kmax(lam_max, int(t7.max()), float(init_t7["beta_t7"]))
    n_trunc = count_kmax_truncated(t7_stats, kmax, float(init_t7["beta_t7"]))
    if verbose:
        print(f"[run_decoupled_model] T7 Kmax={kmax}; T7 rows implying k>Kmax: {n_trunc}")

    t7_model = model_t7_full_dropout if dropout_model == "zero_inflated" else model_t7_full
    t7_samples, t7_info = fit_svi(
        t7_model,
        t7_stats.to_jax(),
        lib_centered,
        kmax,
        priors,
        init_values=init_t7,
        num_steps=steps_t7,
        lr=lr,
        guide=guide,
        num_posterior=num_posterior,
        seed=seed,
    )
    log_lambda_mean, log_lambda_sd = summarize_log_lambda_posterior(t7_samples)
    gh_nodes, gh_log_weights = gauss_hermite_rule(infection_quadrature_points)
    log_lambda_mean_j = jnp.asarray(log_lambda_mean)
    log_lambda_sd_j = jnp.asarray(log_lambda_sd)
    gh_nodes_j = jnp.asarray(gh_nodes)
    gh_log_weights_j = jnp.asarray(gh_log_weights)

    cre_conditional = (
        model_cre_conditional_subclass
        if dropout_model == "zero_inflated"
        else model_cre_conditional_subclass_no_dropout
    )

    def cre_model(stats, lib_size_centered, kmax, negative_control_mask, priors, observe=True):
        return cre_conditional(
            stats,
            log_lambda_mean_j,
            log_lambda_sd_j,
            gh_nodes_j,
            gh_log_weights_j,
            kmax,
            negative_control_mask,
            priors,
            observe,
        )

    init_cre = init_cre_from_moments(cre_stats, priors, negative_control_mask)
    if dropout_model == "none":
        init_cre.pop("p_drop_cre", None)
    cre_samples, cre_info = fit_svi(
        cre_model,
        cre_stats.to_jax(),
        lib_centered,
        kmax,
        priors,
        negative_control_mask=negative_control_mask,
        init_values=init_cre,
        num_steps=steps_cre,
        lr=lr,
        guide=guide,
        num_posterior=num_posterior,
        seed=seed + 10_000,
    )

    t7_summary = summarize_posterior(
        t7_samples, t7_stats, t7_evidence, cre_names, group_names, "subclass"
    )
    cre_summary = summarize_posterior(
        cre_samples, cre_stats, cre_evidence, cre_names, group_names, "subclass",
        prior_dominated_basis="cre",
    )
    summary = {
        "rho": t7_summary["rho"],
        "infection": summarize_lognormal_infection(log_lambda_mean, log_lambda_sd, cre_names, group_names),
        "gamma": cre_summary["gamma"],
    }
    if "delta_mean" in cre_summary:
        summary["delta_mean"] = cre_summary["delta_mean"]

    ppc = posterior_predictive_check_decoupled(
        t7_samples, cre_samples, joint_stats, seed=seed
    )
    rate_mean = np.exp(log_lambda_mean + 0.5 * np.square(log_lambda_sd))
    diagnostics = {
        "losses_t7": np.asarray(t7_info["losses"]),
        "losses_cre": np.asarray(cre_info["losses"]),
        "loss_t7_all_finite": bool(np.isfinite(t7_info["losses"]).all()),
        "loss_cre_all_finite": bool(np.isfinite(cre_info["losses"]).all()),
        "max_infection_rate_mean": float(np.max(rate_mean)),
        "max_log_infection_rate_sd": float(np.max(log_lambda_sd)),
        "kmax_tail_mass_at_mean_rate": kmax_tail_mass(rate_mean, kmax),
        "n_kmax_truncated_t7": n_trunc,
    }

    scalar_sites_t7 = [
        "beta_t7", "phi_t7", "p_drop_t7", "mu_rho", "sigma_u", "sigma_w", "tau_a",
    ]
    scalar_sites_cre = [
        "phi_cre", "p_drop_cre", "mu_alpha", "sigma_alpha", "sigma_eta",
        "sigma_delta", "alpha_neg", "log_gamma_neg",
    ]
    scalar_samples = {
        key: np.asarray(t7_samples[key]) for key in scalar_sites_t7 if key in t7_samples
    }
    scalar_samples.update({
        key: np.asarray(cre_samples[key]) for key in scalar_sites_cre if key in cre_samples
    })

    posterior_samples = {}
    if posterior_sites_to_return:
        requested_sites = list(posterior_sites_to_return)
        if "all" in requested_sites:
            posterior_samples = {k: np.asarray(v) for k, v in cre_samples.items()}
        else:
            if verbose:
                missing_sites = sorted(set(requested_sites) - set(cre_samples))
                if missing_sites:
                    print(f"[run_decoupled_model] requested CRE posterior sites not found: {missing_sites}")
            posterior_samples = {
                key: np.asarray(cre_samples[key])
                for key in requested_sites
                if key in cre_samples
            }

    infection_posterior_samples = {
        key: np.asarray(t7_samples[key])
        for key in ("log_rho", "log_a", "beta_t7", "phi_t7", "p_drop_t7")
        if key in t7_samples
    }
    negative_control_cre = []
    if negative_control_mask is not None:
        negative_control_cre = np.asarray(cre_names)[negative_control_mask].tolist()

    result = {
        "summary": summary,
        "evidence": joint_evidence,
        "t7_evidence": t7_evidence,
        "cre_evidence": cre_evidence,
        "ppc": ppc,
        "diagnostics": diagnostics,
        "scalar_samples": scalar_samples,
        "posterior_samples": posterior_samples,
        "infection_posterior_samples": infection_posterior_samples,
        "log_lambda_mean": log_lambda_mean,
        "log_lambda_sd": log_lambda_sd,
        "kmax": kmax,
        "group_names": group_names,
        "cre_names": list(cre_names),
        "config": dict(
            level="subclass",
            channel="cre",
            method="svi",
            model_variant="decoupled_t7_cre",
            infection_model="copy_number",
            dropout_model=dropout_model,
            kmax=kmax,
            steps_t7=steps_t7,
            steps_cre=steps_cre,
            lr=lr,
            guide=guide,
            num_posterior=num_posterior,
            seed=seed,
            infection_quadrature_points=infection_quadrature_points,
            negative_control_cre=negative_control_cre,
        ),
    }
    return result


# --------------------------------------------------------------------------- #
# Array-level driver (shared by the STARRFISH wrapper and the CLI runner)
# --------------------------------------------------------------------------- #
def run_model(t7, cre, subclass_labels, class_labels, lib_size_log, cre_names, *,
              level="class", channel="joint", method="svi", kmax=None,
              priors: "ModelPriors" = None, num_steps=20000, lr=5e-3,
              guide="AutoNormal", num_warmup=1000, num_samples=1000, num_chains=2,
              num_posterior=1000, seed=0, verbose=True, negative_control_mask=None,
              infection_model="copy_number", activity_model="hierarchical",
              posterior_sites_to_return=None) -> dict:
    """Fit the model from plain arrays (no STARRFISH object, no 97GB pickle).

    Parameters
    ----------
    t7, cre : (n_cells, n_cre) integer count matrices (cre may be ignored for channel='t7').
    subclass_labels, class_labels : (n_cells,) cell-type label arrays (subclass nested in class).
    lib_size_log : (n_cre,) log1p library abundance aligned to ``cre_names`` (centered internally).
    cre_names : sequence of CRE identifiers, length n_cre, in the count-matrix column order.

    Returns the same result dict described in ``STARRFISH.bayesian_activity_test``.
    """
    if infection_model not in MODEL_FAMILIES:
        raise ValueError(
            f"unsupported infection_model={infection_model}; available {sorted(MODEL_FAMILIES)}")
    model_family = MODEL_FAMILIES[infection_model]
    if activity_model not in {"hierarchical", "direct"}:
        raise ValueError(
            f"unsupported activity_model={activity_model}; available ['direct', 'hierarchical']"
        )
    if (level, channel) not in model_family:
        raise ValueError(
            f"unsupported (level, channel)=({level}, {channel}); available {sorted(model_family)}")
    priors = priors or ModelPriors()

    t7 = np.asarray(t7).astype(np.int64)
    cre = np.asarray(cre).astype(np.int64)
    n_cre = len(cre_names)
    if negative_control_mask is not None:
        negative_control_mask = np.asarray(negative_control_mask, dtype=bool)
        if negative_control_mask.shape != (n_cre,):
            raise ValueError("negative_control_mask must have shape (n_cre,)")
        if not negative_control_mask.any():
            negative_control_mask = None
    if activity_model == "direct" and negative_control_mask is not None:
        raise ValueError(
            "direct activity requires ordinary negative controls; pass no "
            "negative_control_mask"
        )
    lib_centered = np.asarray(lib_size_log, dtype=np.float64)
    lib_centered = lib_centered - lib_centered.mean()

    sub = np.asarray(subclass_labels).astype(str)
    cls = np.asarray(class_labels).astype(str)
    sub_cats, sub_idx = np.unique(sub, return_inverse=True)
    cls_cats, cls_idx_cell = np.unique(cls, return_inverse=True)
    mapping = pd.DataFrame({"sub": sub_idx, "cls": cls_idx_cell}).drop_duplicates()
    if mapping["sub"].duplicated().any():
        raise ValueError("subclass does not nest cleanly within class")
    class_of_sub = mapping.sort_values("sub")["cls"].to_numpy().astype(np.int64)

    if level == "class":
        group_idx, n_group, group_names = cls_idx_cell, len(cls_cats), list(cls_cats)
        class_of_group, n_class = None, None
    else:
        group_idx, n_group, group_names = sub_idx, len(sub_cats), list(sub_cats)
        class_of_group, n_class = class_of_sub, len(cls_cats)

    channels = {"t7": t7} if channel == "t7" else {"t7": t7, "cre": cre}
    stats = build_sufficient_stats(channels, group_idx, n_group, n_cre,
                                   class_of_group=class_of_group, n_class=n_class)
    evidence = summarize_evidence(stats)
    if verbose:
        print("[run_model] evidence audit:", evidence["totals"])
        print(f"[run_model] collapsed rows: {len(stats.weight)} (naive cells*cre = {t7.size})")

    init_values = init_from_moments(
        stats,
        lib_centered,
        priors,
        level,
        channel,
        negative_control_mask=negative_control_mask,
        activity_model=activity_model,
    )
    is_copy_number = infection_model in {"copy_number", "copy_number_dropout"}
    if infection_model == "copy_number_dropout":
        init_values["p_drop_t7"] = np.float64(
            priors.p_drop_t7_alpha / (priors.p_drop_t7_alpha + priors.p_drop_t7_beta)
        )
        if channel == "joint":
            init_values["p_drop_cre"] = np.float64(
                priors.p_drop_cre_alpha
                / (priors.p_drop_cre_alpha + priors.p_drop_cre_beta)
            )
    if is_copy_number:
        if kmax is None:
            pp = evidence["per_pair"]
            p_t7 = pp["n_t7_pos"].to_numpy() / np.maximum(
                stats.n_per_group[pp["group"].to_numpy()], 1)
            lam_max = float(np.max(-np.log1p(-np.clip(p_t7, 0, 1 - 1e-9)))) if len(p_t7) else 0.1
            kmax = choose_kmax(
                lam_max, int(max(t7.max(), cre.max())), float(init_values["beta_t7"]))
        n_trunc = count_kmax_truncated(stats, kmax, float(init_values["beta_t7"]))
    else:
        # Retain the common model call signature; no copy-number grid is used.
        kmax = 1
        n_trunc = 0
    if verbose:
        if is_copy_number:
            print(f"[run_model] Kmax={kmax}; cells with counts implying k>Kmax (truncated): {n_trunc}")
        else:
            print("[run_model] binary infection model; no copy-number truncation")

    model = model_family[(level, channel)]
    if channel == "joint":
        model = partial(model, activity_model=activity_model)
    sj = stats.to_jax()
    if method == "svi":
        samples, info = fit_svi(model, sj, lib_centered, kmax, priors,
                                negative_control_mask=negative_control_mask, init_values=init_values,
                                num_steps=num_steps, lr=lr, guide=guide,
                                num_posterior=num_posterior, seed=seed)
        diagnostics = {"losses": np.asarray(info["losses"])}
    else:
        samples, info = fit_nuts(model, sj, lib_centered, kmax, priors,
                                 negative_control_mask=negative_control_mask, init_values=init_values,
                                 num_warmup=num_warmup, num_samples=num_samples,
                                 num_chains=num_chains, seed=seed)
        diagnostics = {"method": "nuts"}

    summary = summarize_posterior(samples, stats, evidence, cre_names, group_names, level)
    if infection_model == "binary":
        summary["infection"] = summarize_binary_infection(samples, cre_names, group_names)
    ppc = posterior_predictive_check(
        samples, sj, lib_centered, kmax, level, seed=seed, infection_model=infection_model)

    rho_mean = np.exp(np.asarray(samples["log_rho"]).mean(0))
    a_mean = np.exp(np.asarray(samples["log_a"]).mean(0))
    max_infection_rate = float(np.max(rho_mean[:, None] * a_mean[None, :]))
    diagnostics["max_infection_rate"] = max_infection_rate
    diagnostics["max_infection_probability"] = float(-np.expm1(-max_infection_rate))
    diagnostics["kmax_tail_mass"] = (
        kmax_tail_mass(max_infection_rate, kmax) if is_copy_number else None
    )
    diagnostics["n_kmax_truncated"] = n_trunc

    scalar_sites = ["beta_t7", "phi_t7", "phi_cre", "mu_rho", "sigma_u", "sigma_w",
                    "tau_a", "mu_alpha", "sigma_alpha", "sigma_eta", "sigma_delta",
                    "mu_gamma", "sigma_gamma",
                    "alpha_neg", "log_gamma_neg", "p_drop_t7", "p_drop_cre"]
    scalar_samples = {k: np.asarray(samples[k]) for k in scalar_sites if k in samples}

    negative_control_cre = []
    if negative_control_mask is not None:
        negative_control_cre = np.asarray(cre_names)[negative_control_mask].tolist()

    posterior_samples = {}
    if posterior_sites_to_return:
        requested_sites = list(posterior_sites_to_return)
        if "all" in requested_sites:
            posterior_samples = {k: np.asarray(v) for k, v in samples.items()}
        else:
            if verbose:
                missing_sites = sorted(set(requested_sites) - set(samples))
                if missing_sites:
                    print(f"[run_model] requested posterior sites not found: {missing_sites}")
            posterior_samples = {
                k: np.asarray(samples[k])
                for k in requested_sites
                if k in samples
            }

    reported_kmax = kmax if is_copy_number else None
    result = {"summary": summary, "evidence": evidence, "ppc": ppc, "diagnostics": diagnostics,
            "scalar_samples": scalar_samples, "kmax": reported_kmax, "group_names": group_names,
            "cre_names": list(cre_names),
            "config": dict(level=level, channel=channel, method=method, kmax=reported_kmax,
                           infection_model=infection_model, activity_model=activity_model,
                           num_steps=num_steps, lr=lr, guide=guide, num_warmup=num_warmup,
                           num_samples=num_samples, num_chains=num_chains, seed=seed,
                           negative_control_cre=negative_control_cre)}
    if posterior_samples:
        result["posterior_samples"] = posterior_samples
    return result
