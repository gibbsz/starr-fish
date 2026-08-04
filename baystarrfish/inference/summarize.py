"""Posterior summaries: tidy per-(cell type, cCRE) and per-cell-type tables.

``summarize_posterior`` produces the ``gamma`` / ``rho`` / ``delta_mean`` frames
written by :func:`baystarrfish.io.write_fit`, including the ``prior_dominated``
flag that marks pairs with no double-positive (T7>0 and cCRE>0) support -- those
posteriors are the prior, not evidence, and must not be interpreted as activity.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from ..model.collapse import CollapsedStats


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


__all__ = [
    "summarize_posterior",
    "summarize_binary_infection",
    "summarize_log_lambda_posterior",
    "summarize_lognormal_infection",
]
