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

from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np

from baystarrfish.data.paths import default_h5ad
from baystarrfish.inference.posterior_k import posterior_k_expectation

ANALYSIS_DIR = Path(__file__).resolve().parent.parent
RESULTS = ANALYSIS_DIR / "results" / "ablation" / "bayesian_full_posterior"
FIGDIR = ANALYSIS_DIR / "figures"
H5AD = default_h5ad()
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


def ek_for_pairs(s_idx, j_idx, t7v, crev, draws):
    """Posterior-mean E[k|obs] on the fit-time k-grid."""
    return posterior_k_expectation(
        t7v, crev, s_idx, j_idx, draws, kmax=KMAX, chunk=CHUNK
    )


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
    # TAG above is the no-dropout fit, so these are normally absent. Passing them
    # when present is what makes this script correct for a dropout fit too --
    # evaluating a dropout posterior without them silently uses a different model.
    for site in ("p_drop_t7", "p_drop_cre"):
        if site in scal.files:
            draws[site] = scal[site].astype(np.float64)
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
