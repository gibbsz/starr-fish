#!/usr/bin/env python
"""Histogram of the posterior expected per-cell, per-cCRE AAV copy number.

For a cell in subclass ``s`` and cCRE ``j`` the latent copy count is
Poisson(rho_s * a_j); its posterior mean is ``lambda_{s,j} = E[rho_s * a_j]``.
The histogram runs over all 328 x 389 subclass-cCRE combinations, each weighted
by the cells in that subclass (so it is the distribution over the
408,621 x 389 cell-cCRE pairs).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ANALYSIS_DIR = Path(__file__).resolve().parent.parent
RESULTS = ANALYSIS_DIR / "results" / "ablation" / "bayesian_full_posterior"
FIGDIR = ANALYSIS_DIR / "figures"
TAG = "subclass_joint_copy_number_svi"

INK = "#1b1e28"
MUTED = "#6b7280"
GRID = "#e5e7eb"
ACCENT = "#2166ac"
MEDIAN = "#b2182b"
plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 300, "font.size": 11,
    "axes.edgecolor": MUTED, "axes.linewidth": 0.8, "axes.labelcolor": INK,
    "text.color": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False,
})


def main() -> None:
    FIGDIR.mkdir(exist_ok=True)
    post = np.load(RESULTS / f"{TAG}_posterior_samples.npz", allow_pickle=True)
    groups = [str(g) for g in post["group_names"]]
    rho = np.exp(post["log_rho"])        # (draws, S)
    a = np.exp(post["log_a"])            # (draws, J)
    D = rho.shape[0]
    lam = (rho.T @ a) / D                # E[rho_s * a_j] -> (S, J)

    rho_df = pd.read_csv(RESULTS / f"{TAG}_rho.csv").set_index("group")
    n_cells = rho_df.loc[groups, "n_cells"].to_numpy(dtype=float)   # (S,)
    weights = np.repeat(n_cells[:, None], lam.shape[1], axis=1)     # (S, J)

    lam_f = lam.ravel()
    w_f = weights.ravel()

    # cell-weighted median over cell-cCRE pairs
    order = np.argsort(lam_f)
    csum = np.cumsum(w_f[order])
    med = lam_f[order][np.searchsorted(csum, csum[-1] / 2)]

    fig, ax = plt.subplots(figsize=(7.6, 5.0))
    bins = np.logspace(np.log10(lam_f.min()), np.log10(lam_f.max()), 60)
    ax.hist(lam_f, bins=bins, weights=w_f,
            color=ACCENT, alpha=0.85, edgecolor="white", linewidth=0.3)
    ax.axvline(med, color=MEDIAN, linewidth=1.8, linestyle="--",
               label=f"pair-weighted median = {med:.2g}")
    ax.axvline(1.0, color=MUTED, linewidth=1.0, linestyle=":", label="1 copy")
    ax.set_xscale("log")
    ax.set_xlabel(r"expected AAV copies per cell per cCRE  ($\rho_s \cdot a_j$)")
    ax.set_ylabel("cell-cCRE pairs")
    ax.set_title("Posterior expected per-cell, per-cCRE infected AAV copy number",
                 color=INK, loc="left")
    ax.grid(True, axis="y", color=GRID, linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIGDIR / f"percell_perccre_aav_copies_hist.{ext}",
                    bbox_inches="tight")
    plt.close(fig)
    frac_ge1 = w_f[lam_f >= 1].sum() / w_f.sum()
    print(f"[fig] percell_perccre_aav_copies_hist  "
          f"(median={med:.3g}, range {lam_f.min():.2g}-{lam_f.max():.2g}, "
          f"frac>=1 copy={frac_ge1:.2%})")


if __name__ == "__main__":
    main()
