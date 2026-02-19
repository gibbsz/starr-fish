"""Visualization functions for barcode-enhancer decomposition."""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import squareform


def _upper_tri(D):
    """Extract upper triangle of a distance matrix."""
    return squareform(D, checks=False)


def plot_distance_scatter(D_activity, D_barcode, D_enhancer, save_dir=None):
    """Scatter plots of flattened distance matrices with regression lines.

    Produces two panels:
      - D_activity vs D_barcode
      - D_activity vs D_enhancer
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    da = _upper_tri(D_activity)
    db = _upper_tri(D_barcode)
    de = _upper_tri(D_enhancer)

    # Subsample for plotting if too many points
    n_pairs = len(da)
    if n_pairs > 10000:
        idx = np.random.default_rng(0).choice(n_pairs, 10000, replace=False)
    else:
        idx = np.arange(n_pairs)

    # Panel 1: activity vs barcode
    ax = axes[0]
    ax.scatter(db[idx], da[idx], s=1, alpha=0.3, rasterized=True)
    # Regression line
    m, b = np.polyfit(db[idx], da[idx], 1)
    x_line = np.linspace(db[idx].min(), db[idx].max(), 100)
    ax.plot(x_line, m * x_line + b, "r-", linewidth=2)
    r = np.corrcoef(db, da)[0, 1]
    ax.set_xlabel("Barcode distance")
    ax.set_ylabel("Activity distance")
    ax.set_title(f"Activity vs Barcode distance\nr = {r:.4f}")

    # Panel 2: activity vs enhancer
    ax = axes[1]
    ax.scatter(de[idx], da[idx], s=1, alpha=0.3, rasterized=True)
    m, b = np.polyfit(de[idx], da[idx], 1)
    x_line = np.linspace(de[idx].min(), de[idx].max(), 100)
    ax.plot(x_line, m * x_line + b, "r-", linewidth=2)
    r = np.corrcoef(de, da)[0, 1]
    ax.set_xlabel("Enhancer distance")
    ax.set_ylabel("Activity distance")
    ax.set_title(f"Activity vs Enhancer distance\nr = {r:.4f}")

    plt.tight_layout()
    if save_dir:
        path = os.path.join(save_dir, "mantel_distance_scatter.pdf")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved: {path}")
    plt.close(fig)
    return fig


def plot_variance_histogram(results_df, save_dir=None):
    """Histogram of prop_barcode across cell types."""
    fig, ax = plt.subplots(figsize=(7, 5))
    valid = results_df.dropna(subset=["prop_barcode"])
    ax.hist(valid["prop_barcode"], bins=30, edgecolor="black", alpha=0.7,
            color="steelblue")
    ax.axvline(valid["prop_barcode"].median(), color="red", linestyle="--",
               label=f"Median = {valid['prop_barcode'].median():.3f}")
    ax.set_xlabel("Proportion of variance explained by barcode")
    ax.set_ylabel("Number of cell types")
    ax.set_title("Barcode variance component across cell types")
    ax.legend()
    plt.tight_layout()
    if save_dir:
        path = os.path.join(save_dir, "variance_barcode_histogram.pdf")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved: {path}")
    plt.close(fig)
    return fig


def plot_volcano(results_df, fdr_alpha=0.05, save_dir=None):
    """Volcano-style plot: prop_barcode vs -log10(p_adj)."""
    fig, ax = plt.subplots(figsize=(7, 5))
    valid = results_df.dropna(subset=["prop_barcode", "p_adj"])

    neg_log_p = -np.log10(valid["p_adj"].clip(lower=1e-300))
    colors = np.where(valid["p_adj"] < fdr_alpha, "red", "grey")

    ax.scatter(valid["prop_barcode"], neg_log_p, c=colors, s=15, alpha=0.7)
    ax.axhline(-np.log10(fdr_alpha), color="blue", linestyle="--", alpha=0.5,
               label=f"FDR = {fdr_alpha}")
    ax.set_xlabel("Proportion of variance (barcode)")
    ax.set_ylabel("-log10(adjusted p-value)")
    ax.set_title("Barcode effect significance across cell types")
    ax.legend()
    plt.tight_layout()
    if save_dir:
        path = os.path.join(save_dir, "variance_volcano.pdf")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved: {path}")
    plt.close(fig)
    return fig


def plot_variance_stacked_bar(results_df, top_n=50, save_dir=None):
    """Stacked bar chart of variance components for top_n cell types.

    Sorted by enhancer proportion.
    """
    valid = results_df.dropna(subset=["prop_barcode"]).copy()
    valid = valid.sort_values("prop_enhancer", ascending=False).head(top_n)

    fig, ax = plt.subplots(figsize=(14, 5))
    x = np.arange(len(valid))
    w = 0.8

    ax.bar(x, valid["prop_enhancer"].values, w, label="Enhancer",
           color="steelblue")
    ax.bar(x, valid["prop_barcode"].values, w,
           bottom=valid["prop_enhancer"].values, label="Barcode",
           color="coral")
    ax.bar(x, valid["prop_noise"].values, w,
           bottom=(valid["prop_enhancer"] + valid["prop_barcode"]).values,
           label="Noise", color="lightgrey")

    ax.set_xlabel("Cell type (sorted by enhancer proportion)")
    ax.set_ylabel("Proportion of variance")
    ax.set_title(f"Variance decomposition (top {top_n} cell types)")
    ax.legend(loc="upper right")
    if "cell_type" in valid.columns:
        ax.set_xticks(x)
        ax.set_xticklabels(valid["cell_type"].values, rotation=90, fontsize=6)
    else:
        ax.set_xticks([])
    plt.tight_layout()
    if save_dir:
        path = os.path.join(save_dir, "variance_stacked_bar.pdf")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved: {path}")
    plt.close(fig)
    return fig


def plot_variance_violin(results_df, save_dir=None):
    """Violin plot comparing variance component proportions."""
    valid = results_df.dropna(subset=["prop_barcode"]).copy()
    melted = pd.melt(
        valid[["prop_barcode", "prop_enhancer", "prop_noise"]],
        var_name="Component", value_name="Proportion",
    )
    melted["Component"] = melted["Component"].map({
        "prop_barcode": "Barcode",
        "prop_enhancer": "Enhancer",
        "prop_noise": "Noise",
    })

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.violinplot(data=melted, x="Component", y="Proportion", ax=ax,
                   palette=["coral", "steelblue", "lightgrey"], inner="box")
    ax.set_title("Variance component proportions across cell types")
    ax.set_ylabel("Proportion of variance")
    plt.tight_layout()
    if save_dir:
        path = os.path.join(save_dir, "variance_violin.pdf")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved: {path}")
    plt.close(fig)
    return fig


def plot_all_variance(results_df, fdr_alpha=0.05, save_dir=None):
    """Generate all variance decomposition plots."""
    plot_variance_histogram(results_df, save_dir=save_dir)
    plot_volcano(results_df, fdr_alpha=fdr_alpha, save_dir=save_dir)
    plot_variance_stacked_bar(results_df, save_dir=save_dir)
    plot_variance_violin(results_df, save_dir=save_dir)
