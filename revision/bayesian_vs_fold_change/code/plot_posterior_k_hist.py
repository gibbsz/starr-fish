#!/usr/bin/env python
"""Histograms of the posterior latent copy number k (NOT the prior rho*a).

k is analytically marginalized in the fit (logsumexp over 0..kmax). We
reconstruct its posterior per observation from the fitted draws and the observed
T7/CRE counts, on the SAME k-grid used at fit time (kmax=60):

    P(k | t7, cre) proportional to
        Poisson(k; rho_s a_j) * NB2(t7 | k beta_t7, phi_t7) * NB2(cre | k gamma_sj, phi_cre)

and report the posterior mean E[k|obs] = mean over draws of sum_k k P(k|obs).

Two figures:
  1. percell_total_k_hist        - per-cell total inferred copies  sum_j E[k_ij].
  2. percell_perccre_k_hist       - per (cell,cCRE) inferred copies E[k_ij].
"""
from __future__ import annotations

import sys
from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import gammaln

ANALYSIS_DIR = Path(__file__).resolve().parent.parent
RESULTS = ANALYSIS_DIR / "results" / "ablation" / "bayesian_full_posterior"
FIGDIR = ANALYSIS_DIR / "figures"
H5AD = ANALYSIS_DIR.parents[0] / "Data" / "scdata_5_28_2025_BRBB500gn_final_CRE_T7CRE_NEWNEW.h5ad"
TAG = "subclass_joint_copy_number_svi"
KMAX = 60
CHUNK = 400  # pairs per k-grid chunk

INK, MUTED, GRID, ACCENT, MEDIAN = "#1b1e28", "#6b7280", "#e5e7eb", "#2166ac", "#b2182b"
plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 300, "font.size": 11,
    "axes.edgecolor": MUTED, "axes.linewidth": 0.8, "axes.labelcolor": INK,
    "text.color": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False,
})


def nb2_logpmf(count, mean, conc):
    """NB2 log-pmf: mean `mean`, var = mean + mean^2/conc. Matches _nb2_logprob."""
    p = conc / (conc + mean)
    return (gammaln(count + conc) - gammaln(conc) - gammaln(count + 1)
            + conc * np.log(p) + count * np.log1p(-p))


def channel_logprob(obs, k, per_copy, phi):
    """log P(obs | k) for one NB channel, k a (K,) grid. Matches _channel_logprob."""
    safe_k = np.where(k == 0, 1.0, k)
    mean = per_copy * safe_k                       # (..., K)
    nb = nb2_logpmf(obs, mean, phi)
    point = np.where(obs == 0, 0.0, -np.inf)       # (..., 1) over K
    return np.where(k == 0, point, nb)


def ek_for_pairs(s_idx, j_idx, t7v, crev, draws):
    """Posterior-mean E[k|obs] for a set of observations (vectorized, chunked).

    draws: dict of (rho, a, log_gamma, beta_t7, phi_t7, phi_cre) posterior arrays.
    Returns (P,) array of E[k|obs] averaged over draws.
    """
    rho, a, log_gamma = draws["rho"], draws["a"], draws["log_gamma"]
    beta, phi_t7, phi_cre = draws["beta_t7"], draws["phi_t7"], draws["phi_cre"]
    k = np.arange(0, KMAX + 1, dtype=np.float64)               # (K,)
    lgk1 = gammaln(k + 1)
    out = np.empty(len(s_idx), dtype=np.float64)
    D = rho.shape[0]
    for start in range(0, len(s_idx), CHUNK):
        sl = slice(start, start + CHUNK)
        s, j = s_idx[sl], j_idx[sl]
        t7 = t7v[sl].astype(np.float64)[None, :, None]         # (1,C,1)
        cre = crev[sl].astype(np.float64)[None, :, None]
        lam = (rho[:, s] * a[:, j])[:, :, None]                # (D,C,1)
        gam = np.exp(log_gamma[:, s, j])[:, :, None]           # (D,C,1)
        b = beta[:, None, None]; pt7 = phi_t7[:, None, None]; pcre = phi_cre[:, None, None]
        logpk = k * np.log(lam) - lam - lgk1                   # (D,C,K)
        logpost = logpk
        logpost = logpost + channel_logprob(t7, k, b, pt7)
        logpost = logpost + channel_logprob(cre, k, gam, pcre)
        m = logpost.max(axis=-1, keepdims=True)
        w = np.exp(logpost - m)
        w /= w.sum(axis=-1, keepdims=True)
        ek_d = (w * k).sum(axis=-1)                            # (D,C)
        out[sl] = ek_d.mean(axis=0)
    return out


def load_counts(cre_names):
    print(f"[load] {H5AD} (backed)", flush=True)
    adata = ad.read_h5ad(H5AD, backed="r")
    try:
        t7 = adata.obsm["T7CRE"]; cre = adata.obsm["CRE"]
        t7.columns = t7.columns.astype(str); cre.columns = cre.columns.astype(str)
        t7 = np.rint(t7.loc[:, cre_names].to_numpy()).astype(np.int32)
        cre = np.rint(cre.loc[:, cre_names].to_numpy()).astype(np.int32)
        sub = (adata.obs["subclass_name"].astype(str)
               .str.replace(r"^\d+\s+", "", regex=True)
               .str.replace("/", "-", regex=False).to_numpy())
    finally:
        adata.file.close()
    return t7, cre, sub


def main():
    FIGDIR.mkdir(exist_ok=True)
    post = np.load(RESULTS / f"{TAG}_posterior_samples.npz", allow_pickle=True)
    scal = np.load(RESULTS / f"{TAG}_scalar_samples.npz")
    groups = [str(g) for g in post["group_names"]]
    cre_names = [str(c) for c in post["cre_names"]]
    gidx = {g: i for i, g in enumerate(groups)}
    draws = {
        "rho": np.exp(post["log_rho"]).astype(np.float64),      # (D,S)
        "a": np.exp(post["log_a"]).astype(np.float64),          # (D,J)
        "log_gamma": post["log_gamma"].astype(np.float64),      # (D,S,J)
        "beta_t7": scal["beta_t7"].astype(np.float64),
        "phi_t7": scal["phi_t7"].astype(np.float64),
        "phi_cre": scal["phi_cre"].astype(np.float64),
    }
    S, J = len(groups), len(cre_names)

    t7, cre, sub = load_counts(cre_names)
    if not np.isin(sub, groups).all():
        missing = sorted(set(sub) - set(groups))
        raise ValueError(f"{len(missing)} cell subclasses absent from fit: {missing[:5]}")
    cell_group = np.array([gidx[s] for s in sub], dtype=np.int64)   # (n_cells,)
    n_cells = t7.shape[0]
    print(f"[data] {n_cells:,} cells x {J} cCREs", flush=True)

    # --- all-zero baseline E[k|t7=0,cre=0] for every (s,j) ---
    print("[k] computing zero-observation baseline (S*J grid)...", flush=True)
    ss, jj = np.meshgrid(np.arange(S), np.arange(J), indexing="ij")
    ek_zero = ek_for_pairs(
        ss.ravel(), jj.ravel(),
        np.zeros(S * J, np.int32), np.zeros(S * J, np.int32), draws
    ).reshape(S, J)
    baseline_per_subclass = ek_zero.sum(axis=1)                 # (S,)

    # --- nonzero observations: unique (s,j,t7,cre) patterns ---
    rows, cols = np.nonzero((t7 > 0) | (cre > 0))
    t7v, crev = t7[rows, cols], cre[rows, cols]
    sidx = cell_group[rows]
    print(f"[data] {len(rows):,} nonzero (cell,cCRE) entries", flush=True)
    pat = np.stack([sidx, cols, t7v, crev], axis=1)
    uniq, inv = np.unique(pat, axis=0, return_inverse=True)
    print(f"[k] computing {len(uniq):,} unique nonzero patterns...", flush=True)
    ek_uniq = ek_for_pairs(uniq[:, 0], uniq[:, 1], uniq[:, 2], uniq[:, 3], draws)
    ek_entry = ek_uniq[inv]                                     # E[k|obs] per nonzero entry

    # --- Figure 1: per-cell total inferred copies ---
    total = baseline_per_subclass[cell_group].astype(np.float64)
    corr = ek_entry - ek_zero[sidx, cols]
    np.add.at(total, rows, corr)

    med_tot = np.median(total)
    fig, ax = plt.subplots(figsize=(7.6, 5.0))
    lo = max(total.min(), 1e-3)
    bins = np.logspace(np.log10(lo), np.log10(total.max()), 60)
    ax.hist(total, bins=bins, color=ACCENT, alpha=0.85, edgecolor="white", linewidth=0.3)
    ax.axvline(med_tot, color=MEDIAN, linewidth=1.8, linestyle="--",
               label=f"median = {med_tot:.2g}")
    ax.set_xscale("log")
    ax.set_xlabel(r"posterior total inferred AAV copies per cell  ($\sum_j E[k_{ij}\,|\,\mathrm{obs}]$)")
    ax.set_ylabel("number of cells")
    ax.set_title("Posterior latent copy number: per-cell total", color=INK, loc="left")
    ax.grid(True, axis="y", color=GRID, linewidth=0.6, alpha=0.7); ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIGDIR / f"percell_total_k_hist.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] percell_total_k_hist  median={med_tot:.3g} "
          f"range {total.min():.3g}-{total.max():.3g}", flush=True)

    # --- Figure 2: per (cell,cCRE) inferred copies ---
    # zero entries per (s,j): n_cells_in_s - n_nonzero_cells(s,j)
    n_per_sub = np.bincount(cell_group, minlength=S)
    nz_per_pair = np.zeros((S, J), np.int64)
    np.add.at(nz_per_pair, (sidx, cols), 1)
    zero_w = (n_per_sub[:, None] - nz_per_pair).ravel()
    zero_val = ek_zero.ravel()
    m = zero_w > 0
    vals = np.concatenate([ek_entry, zero_val[m]])
    wts = np.concatenate([np.ones(len(ek_entry)), zero_w[m].astype(float)])

    order = np.argsort(vals); c = np.cumsum(wts[order])
    med_pair = vals[order][np.searchsorted(c, c[-1] / 2)]
    frac_ge1 = wts[vals >= 1].sum() / wts.sum()
    fig, ax = plt.subplots(figsize=(7.6, 5.0))
    lo = max(vals[vals > 0].min(), 1e-6)
    bins = np.logspace(np.log10(lo), np.log10(vals.max()), 60)
    ax.hist(vals, bins=bins, weights=wts, color=ACCENT, alpha=0.85,
            edgecolor="white", linewidth=0.3)
    ax.axvline(med_pair, color=MEDIAN, linewidth=1.8, linestyle="--",
               label=f"pair-weighted median = {med_pair:.2g}")
    ax.axvline(1.0, color=MUTED, linewidth=1.0, linestyle=":", label="1 copy")
    ax.set_xscale("log")
    ax.set_xlabel(r"posterior inferred AAV copies, per cell per cCRE  ($E[k_{ij}\,|\,\mathrm{obs}]$)")
    ax.set_ylabel("cell-cCRE pairs")
    ax.set_title("Posterior latent copy number: per cell, per cCRE", color=INK, loc="left")
    ax.grid(True, axis="y", color=GRID, linewidth=0.6, alpha=0.7); ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIGDIR / f"percell_perccre_k_hist.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] percell_perccre_k_hist  median={med_pair:.3g} "
          f"frac>=1={frac_ge1:.3%} total_pairs={int(wts.sum()):,}", flush=True)


if __name__ == "__main__":
    main()
