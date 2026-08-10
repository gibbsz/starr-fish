#!/usr/bin/env python
"""Sanity-plot posterior rho / a against observed total T7 counts.

Three figures land in ``revision/bayesian_vs_fold_change/figures/``:

1. ``rho_vs_t7_subclass``  - per-subclass infection rate rho vs total T7 reads.
2. ``a_across_cres``       - per-cCRE abundance a, rank-sorted, with 90% CI.
3. ``a_vs_t7_cre``         - per-cCRE abundance a vs total T7 reads (all cells).
"""
from __future__ import annotations

from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import ScalarFormatter
from scipy.stats import spearmanr, pearsonr

# parents[2]: this file lives in code/figure_work/, so the analysis root is
# two levels up from the code directory.
ANALYSIS_DIR = Path(__file__).resolve().parents[2]
RESULTS = ANALYSIS_DIR / "results" / "ablation" / "bayesian_full_posterior"
FIGDIR = ANALYSIS_DIR / "results" / "figures" / "work"
H5AD = (
    ANALYSIS_DIR.parents[0]
    / "Data"
    / "scdata_5_28_2025_BRBB500gn_final_CRE_T7CRE_NEWNEW.h5ad"
)
TAG = "subclass_joint_copy_number_svi"

# --- brand-neutral, colorblind-safe marks (single sequential hue + one accent) ---
INK = "#1b1e28"
MUTED = "#6b7280"
GRID = "#e5e7eb"
ACCENT = "#2166ac"     # cool primary for marks
FIT = "#b2182b"        # warm accent for the fit line
plt.rcParams.update({
    "figure.dpi": 130,
    "savefig.dpi": 300,
    "font.size": 11,
    "axes.edgecolor": MUTED,
    "axes.linewidth": 0.8,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def clean_subclass(raw: pd.Series) -> pd.Series:
    """Replicate analysis_utils.standardize_obs subclass cleaning."""
    return (
        raw.astype(str)
        .str.replace(r"^\d+\s+", "", regex=True)
        .str.replace("/", "-", regex=False)
    )


def load_t7_totals(cre_names: list[str]) -> tuple[pd.Series, pd.Series]:
    """Return (total T7 per subclass, total T7 per cCRE) over the fitted cCREs."""
    print(f"[load] {H5AD} (backed)")
    adata = ad.read_h5ad(H5AD, backed="r")
    try:
        t7 = adata.obsm["T7CRE"]
        if not isinstance(t7, pd.DataFrame):
            raise TypeError("expected obsm['T7CRE'] to be a DataFrame")
        t7.columns = t7.columns.astype(str)
        t7 = t7.loc[:, cre_names].to_numpy(dtype=np.float64)  # (cells, 389)
        subclass = clean_subclass(adata.obs["subclass_name"]).to_numpy()
    finally:
        adata.file.close()

    per_cre = pd.Series(t7.sum(axis=0), index=cre_names, name="t7_total")
    per_cell_total = t7.sum(axis=1)
    per_subclass = (
        pd.DataFrame({"subclass": subclass, "t7": per_cell_total})
        .groupby("subclass")["t7"]
        .sum()
    )
    return per_subclass, per_cre


def annotate_corr(ax, x, y, log_x=False, log_y=False) -> None:
    lx = np.log10(x) if log_x else np.asarray(x, dtype=float)
    ly = np.log10(y) if log_y else np.asarray(y, dtype=float)
    ok = np.isfinite(lx) & np.isfinite(ly)
    rho, _ = spearmanr(x[ok], y[ok])
    r, _ = pearsonr(lx[ok], ly[ok])
    ax.text(
        0.03, 0.97,
        f"Spearman $\\rho$ = {rho:.2f}\nPearson r = {r:.2f}"
        + ("  (log-log)" if log_x and log_y else ""),
        transform=ax.transAxes, va="top", ha="left", fontsize=10, color=INK,
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=GRID, lw=0.8),
    )


def main() -> None:
    FIGDIR.mkdir(exist_ok=True)

    post = np.load(RESULTS / f"{TAG}_posterior_samples.npz", allow_pickle=True)
    cre_names = [str(c) for c in post["cre_names"]]
    log_a = post["log_a"]                       # (draws, 389)
    a = np.exp(log_a)
    a_mean = a.mean(axis=0)
    a_lo, a_hi = np.percentile(a, [5, 95], axis=0)

    rho_df = pd.read_csv(RESULTS / f"{TAG}_rho.csv")   # group, rho_mean/lo/hi, n_cells

    t7_subclass, t7_cre = load_t7_totals(cre_names)

    # ---------- Figure 1: rho vs total T7 per subclass ----------
    rho_df = rho_df.assign(t7_total=rho_df["group"].map(t7_subclass))
    rho_df = rho_df.dropna(subset=["t7_total"])
    rho_df = rho_df[rho_df["t7_total"] > 0]
    rho_df = rho_df.assign(t7_per_cell=rho_df["t7_total"] / rho_df["n_cells"])

    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    yerr = np.vstack([
        rho_df["rho_mean"] - rho_df["rho_lo"],
        rho_df["rho_hi"] - rho_df["rho_mean"],
    ])
    ax.errorbar(
        rho_df["t7_per_cell"], rho_df["rho_mean"], yerr=yerr,
        fmt="none", ecolor=GRID, elinewidth=0.8, zorder=1,
    )
    sc = ax.scatter(
        rho_df["t7_per_cell"], rho_df["rho_mean"],
        c=rho_df["n_cells"], cmap="coolwarm",
        norm=plt.matplotlib.colors.LogNorm(),
        s=26, edgecolor="white", linewidth=0.4, zorder=2,
    )
    cb = fig.colorbar(sc, ax=ax, pad=0.02)
    cb.set_label("cells per subclass", color=INK)
    cb.outline.set_edgecolor(MUTED)
    ax.set_xscale("log")
    ax.set_xlabel("T7 reads per cell in subclass (fitted cCREs)")
    ax.set_ylabel(r"posterior infection rate $\rho$ (mean $\pm$ 90% CI)")
    ax.set_title("Per-subclass infection rate vs T7 per cell", color=INK, loc="left")
    ax.grid(True, color=GRID, linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    annotate_corr(ax, rho_df["t7_per_cell"].values, rho_df["rho_mean"].values, log_x=True)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIGDIR / f"rho_vs_t7_subclass.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] rho_vs_t7_subclass  ({len(rho_df)} subclasses)")

    # ---------- Figure 2: a across cCREs, rank-sorted ----------
    order = np.argsort(a_mean)[::-1]
    rank = np.arange(1, len(order) + 1)
    fig, ax = plt.subplots(figsize=(9.0, 5.0))
    ax.fill_between(
        rank, a_lo[order], a_hi[order],
        color=ACCENT, alpha=0.18, linewidth=0, label="90% CI",
    )
    ax.plot(rank, a_mean[order], color=ACCENT, linewidth=1.6, label="posterior mean")
    ax.set_yscale("log")
    ax.set_xlabel("cCRE rank (by abundance)")
    ax.set_ylabel(r"posterior abundance $a = e^{\log a}$")
    ax.set_title(
        r"Per-cCRE library abundance $a$ across 389 fitted cCREs",
        color=INK, loc="left",
    )
    ax.grid(True, color=GRID, linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIGDIR / f"a_across_cres.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] a_across_cres  ({len(order)} cCREs)")

    # ---------- Figure 3: a vs total T7 per cCRE ----------
    a_df = pd.DataFrame({"cre": cre_names, "a_mean": a_mean, "a_lo": a_lo, "a_hi": a_hi})
    a_df = a_df.assign(t7_total=a_df["cre"].map(t7_cre))
    a_df = a_df[a_df["t7_total"] > 0]

    fig, ax = plt.subplots(figsize=(6.8, 5.6))
    ax.errorbar(
        a_df["t7_total"], a_df["a_mean"],
        yerr=np.vstack([a_df["a_mean"] - a_df["a_lo"], a_df["a_hi"] - a_df["a_mean"]]),
        fmt="none", ecolor=GRID, elinewidth=0.7, zorder=1,
    )
    ax.scatter(
        a_df["t7_total"], a_df["a_mean"],
        s=22, color=ACCENT, edgecolor="white", linewidth=0.4, zorder=2,
    )
    # log-log OLS fit for reference
    lx, ly = np.log10(a_df["t7_total"]), np.log10(a_df["a_mean"])
    slope, intercept = np.polyfit(lx, ly, 1)
    xs = np.linspace(lx.min(), lx.max(), 100)
    ax.plot(10**xs, 10**(slope * xs + intercept), color=FIT, linewidth=1.6,
            label=f"OLS slope = {slope:.2f}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("total T7 reads for cCRE (all cells)")
    ax.set_ylabel(r"posterior abundance $a$ (mean $\pm$ 90% CI)")
    ax.set_title("Per-cCRE abundance vs total T7", color=INK, loc="left")
    ax.grid(True, color=GRID, linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="lower right")
    annotate_corr(ax, a_df["t7_total"].values, a_df["a_mean"].values, log_x=True, log_y=True)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIGDIR / f"a_vs_t7_cre.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] a_vs_t7_cre  ({len(a_df)} cCREs)")


if __name__ == "__main__":
    main()
