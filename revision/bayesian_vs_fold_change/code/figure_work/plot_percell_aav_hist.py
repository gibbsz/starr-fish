#!/usr/bin/env python
"""Histogram of the posterior expected per-cell infected AAV copy number.

Under the copy-number infection model a cell in subclass ``s`` carries, for each
cCRE ``j``, a latent Poisson(rho_s * a_j) copy count. The total AAV copies per
cell is therefore Poisson(rho_s * sum_j a_j) with posterior mean ``rho_s * A``,
``A = sum_j a_j``. Each of the 408,621 cells inherits its subclass value; the
histogram is weighted by cells per subclass.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# parents[2]: this file lives in code/figure_work/, so the analysis root is
# two levels up from the code directory.
ANALYSIS_DIR = Path(__file__).resolve().parents[2]
RESULTS = ANALYSIS_DIR / "results" / "ablation" / "bayesian_full_posterior"
FIGDIR = ANALYSIS_DIR / "results" / "figures" / "work"
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
    A = np.exp(post["log_a"]).sum(axis=1)                 # (draws,) total abundance
    copies = np.exp(post["log_rho"]) * A[:, None]         # (draws, subclass)
    copies_mean = copies.mean(axis=0)                     # per-subclass posterior mean

    rho_df = pd.read_csv(RESULTS / f"{TAG}_rho.csv").set_index("group")
    n_cells = rho_df.loc[groups, "n_cells"].to_numpy(dtype=float)

    # cell-weighted median of the per-cell copy number
    order = np.argsort(copies_mean)
    csum = np.cumsum(n_cells[order])
    med = copies_mean[order][np.searchsorted(csum, csum[-1] / 2)]

    fig, ax = plt.subplots(figsize=(7.6, 5.0))
    bins = np.logspace(np.log10(copies_mean.min()), np.log10(copies_mean.max()), 45)
    ax.hist(copies_mean, bins=bins, weights=n_cells,
            color=ACCENT, alpha=0.85, edgecolor="white", linewidth=0.4)
    ax.axvline(med, color=MEDIAN, linewidth=1.8, linestyle="--",
               label=f"cell-weighted median = {med:.0f}")
    ax.set_xscale("log")
    ax.set_xlabel("expected AAV copies per cell  " r"($\rho_s \cdot \sum_j a_j$)")
    ax.set_ylabel("number of cells")
    ax.set_title("Posterior expected per-cell infected AAV copy number",
                 color=INK, loc="left")
    ax.grid(True, axis="y", color=GRID, linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIGDIR / f"percell_aav_copies_hist.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] percell_aav_copies_hist  "
          f"(median={med:.1f}, range {copies_mean.min():.1f}-{copies_mean.max():.1f}, "
          f"{int(n_cells.sum()):,} cells)")


if __name__ == "__main__":
    main()
