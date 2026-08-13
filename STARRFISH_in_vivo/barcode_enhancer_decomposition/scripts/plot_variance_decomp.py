# %% 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
# %%
in_vitro_20CRE = pd.read_csv("/gpfs/commons/groups/ren_lab/guojiezhong/starr-fish/STARRFISH_in_vivo/barcode_enhancer_decomposition/results/invitro_20CRE/blup.csv")
in_vitro_300CRE = pd.read_csv("/gpfs/commons/groups/ren_lab/guojiezhong/starr-fish/STARRFISH_in_vivo/barcode_enhancer_decomposition/results/invitro_300CRE/blup.csv")
in_vivo = pd.read_csv("/gpfs/commons/groups/ren_lab/guojiezhong/starr-fish/STARRFISH_in_vivo/barcode_enhancer_decomposition/results/variance_decomp/blup.csv")
# %%
def plot_variance_decomposition(df_plot, variance=False, fig=None, ax=None):
    """
    Plots variance decomposition using Seaborn with support for pre-calculated CIs.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))

    # --- 1. Data Preparation: Melt to Long Form ---
    # We convert wide columns (prop_enhancer, prop_barcode, etc.) into rows
    df_long = df_plot.melt(
        id_vars=["construct"], 
        value_vars=["prop_enhancer", "prop_barcode", "prop_noise"],
        var_name="Component_Key", 
        value_name="Proportion"
    )

    # Create a nice label map for the Legend
    label_map = {
        "prop_enhancer": "Enhancer Signal",
        "prop_barcode":  "Barcode Artifact",
        "prop_noise":    "Noise / Residual"
    }
    df_long["Component"] = df_long["Component_Key"].map(label_map)

    # Define the color palette to match your original scheme
    palette = {
        "Enhancer Signal": "#2ca02c", # Green
        "Barcode Artifact": "#d62728", # Red
        "Noise / Residual": "#7f7f7f"  # Grey
    }

    # --- 2. Plotting with Seaborn ---
    # We enforce 'order' to ensure lines are in the correct order
    sns.lineplot(
        data=df_long,
        x="construct",
        y="Proportion",
        hue="Component",
        palette=palette,
        marker="o",
        markersize=8,
        linewidth=2,
        alpha=0.9,
        ax=ax
    )

    # --- 3. Add Pre-calculated Error Bars ---
    if variance:
        # Map the component names to their corresponding CI column in original df
        ci_lookup = {
            "Enhancer Signal": "CI95_enhancer",
            "Barcode Artifact": "CI95_barcode",
            "Noise / Residual": None  # Usually no CI for noise
        }

        # Seaborn stores groups of bars in ax.containers
        # We iterate through them to place error bars on the correct hue group
        for container in ax.containers:
            # Get the label for this group of bars (e.g., "Enhancer Signal")
            # Note: container.get_label() might be standardized, so we check the first patch
            # But safer to rely on the legend order or palette keys if standard
            # Best way in recent MPL/Seaborn: use the label assigned to the container
            label = container.get_label()
            
            # Find the corresponding CI column
            ci_col = ci_lookup.get(label)

            if ci_col and ci_col in df_plot.columns:
                # Get the error values in the correct order (matching the x-axis)
                # Note: df_plot is already sorted by the user before passing in
                errors = df_plot[ci_col].values
                
                # Get x and y coordinates from the bars themselves
                x_coords = [patch.get_x() + patch.get_width() / 2 for patch in container]
                y_coords = [patch.get_height() for patch in container]

                # Draw the error bars
                ax.errorbar(
                    x=x_coords, 
                    y=y_coords, 
                    yerr=errors, 
                    fmt='none',    # No marker
                    c='black',     # Black error bars
                    capsize=3, 
                    elinewidth=1.5
                )

    # --- 4. Formatting ---
    ax.set_ylabel('Proportion of Activity Explained')
    ax.set_xlabel('') # constructs are self-explanatory
    ax.set_title('Activity Decomposition by Component (Ranked by Enhancer Signal)')
    
    # Rotate x-labels
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=10)
    
    # Legend cleanup
    ax.legend(title=None, loc='upper right')
    
    # Aesthetics
    ax.set_ylim(0, 1.1)
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    if fig:
        fig.tight_layout()
        
    return fig, ax
# 1. Prepare and Sort Data
# Sort by 'prop_enhancer' descending so the strongest biological signals are first
df_plot = in_vitro_20CRE[in_vitro_20CRE['cell_type']=='July'].sort_values("prop_enhancer", ascending=False).copy()

# 5. Save and Show
save_path = "/gpfs/commons/groups/ren_lab/guojiezhong/starr-fish/STARRFISH_in_vivo/barcode_enhancer_decomposition/results/plots/invitro_20CRE_variance_grouped_bar.pdf"
fig, ax = plt.subplots(figsize=(14, 6))
fig, ax = plot_variance_decomposition(df_plot, fig=fig, ax=ax)
fig.savefig(save_path, dpi=300)
fig.show()


# %% for 300 CRE
df_plot_300 = in_vitro_300CRE[in_vitro_300CRE['cell_type']=='300CRE_1'].sort_values("prop_enhancer", ascending=False).copy()
save_path_300 = "/gpfs/commons/groups/ren_lab/guojiezhong/starr-fish/STARRFISH_in_vivo/barcode_enhancer_decomposition/results/plots/invitro_300CRE_variance_grouped_bar.pdf"
fig_300, ax_300 = plt.subplots(figsize=(30, 6))
fig_300, ax_300 = plot_variance_decomposition(df_plot_300, fig=fig_300, ax=ax_300)
fig_300.savefig(save_path_300, dpi=300)
fig_300.show()

# %%
def plot_variance_qc(df, output_prefix="qc_plot", show=True, figsize=(12, 6)):
    """
    Generates QC plots for Variance Decomposition:
    1. Stacked Barplot of Variance Components (Ranked)
    2. Scatter Plot of Noise vs. Cell Number
    
    Args:
        df (pd.DataFrame): Dataframe containing 'prop_enhancer', 'prop_barcode', 
                           'prop_noise', 'cell_type', and 'cell_number'.
        output_prefix (str): Path prefix for saving files (e.g., 'results/figures/invivo').
        show (bool): Whether to call plt.show() at the end.
        
    Returns:
        tuple: (fig1, fig2) The matplotlib figure objects.
    """
    
    # --- Plot 1: Stacked Barplot of Variance Decomposition ---
    # Sort by Enhancer proportion for better visualization
    df_sorted = df.copy()
    
    fig1, ax1 = plt.subplots(figsize=figsize)
    
    # Bottom: Enhancer (Green)
    p1 = ax1.bar(df_sorted['cell_type'], df_sorted['prop_enhancer'],
                 label='Enhancer', color='#2ca02c', alpha=0.9, width=0.8)

    # Middle: Barcode (Red)
    p2 = ax1.bar(df_sorted['cell_type'], df_sorted['prop_barcode'],
                 bottom=df_sorted['prop_enhancer'],
                 label='Barcode', color='#d62728', alpha=0.8, width=0.8)

    # Top: Noise (Grey)
    bottom_noise = df_sorted['prop_enhancer'] + df_sorted['prop_barcode']
    p3 = ax1.bar(df_sorted['cell_type'], df_sorted['prop_noise'],
                 bottom=bottom_noise,
                 label='Noise', color='#7f7f7f', alpha=0.5, width=0.8)
    
    ax1.set_ylabel('Proportion of Variance')
    ax1.set_title('Variance Decomposition across Cell Types')
    ax1.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
    
    # Adjust x-tick visibility based on number of cell types
    if len(df_sorted) > 50:
        # If too many cell types, show every 5th or 10th label to avoid clutter
        step = max(1, len(df_sorted) // 50)
        ax1.set_xticks(range(0, len(df_sorted), step))
        ax1.set_xticklabels(df_sorted['cell_type'].iloc[::step], rotation=90, fontsize=8)
    else:
        ax1.set_xticks(range(len(df_sorted)))
        ax1.set_xticklabels(df_sorted['cell_type'], rotation=90, fontsize=10)
        
    ax1.set_xlim(-0.5, len(df_sorted) - 0.5)
    ax1.set_ylim(0, 1)
    
    fig1.tight_layout()
    
    # Save Plot 1
    save_path1 = f"{output_prefix}_variance_stacked.pdf"
    fig1.savefig(save_path1, dpi=300)
    print(f"Saved: {save_path1}")

    # --- Plot 2: Noise vs Cell Number ---
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    
    # Check if 'cell_number' exists to avoid errors
    if 'cell_number' in df.columns:
        # log transform cell_number
        df['cell_number_log'] = np.log10(df['cell_number'] + 1)  # add 1 to avoid log(0)
        sns.scatterplot(data=df, x='cell_number_log', y='prop_noise', 
                        ax=ax2, color='black', alpha=0.6, s=50)
        
        # Add trend line
        sns.regplot(data=df, x='cell_number_log', y='prop_noise', 
                    scatter=False, ax=ax2, color='red', 
                    line_kws={'linestyle': '--', 'alpha': 0.8})
        
        ax2.set_xlabel('Number of Cells (log10 scale)')
        ax2.set_ylabel('Proportion of Noise Variance')
        ax2.set_title('QC: Relationship between Cell Number and Noise')
            
        ax2.grid(True, linestyle='--', alpha=0.5)
        ax2.set_ylim(0, 1.1)
        
        fig2.tight_layout()
        
        # Save Plot 2
        save_path2 = f"{output_prefix}_noise_vs_cells.pdf"
        fig2.savefig(save_path2, dpi=300)
        print(f"Saved: {save_path2}")
    else:
        print("Warning: 'cell_number' column not found. Skipping Plot 2.")
        ax2.text(0.5, 0.5, "Missing 'cell_number' data", ha='center', va='center')

    if show:
        plt.show()
        
    return fig1, fig2

# %% plot variance_decomp results
var_decomp_in_vitro_20CRE = pd.read_csv("/gpfs/commons/groups/ren_lab/guojiezhong/starr-fish/STARRFISH_in_vivo/barcode_enhancer_decomposition/results/invitro_20CRE/variance_decomp_results.csv", index_col=0)
var_decomp_in_vitro_20CRE = var_decomp_in_vitro_20CRE[var_decomp_in_vitro_20CRE["cell_type"]!="All_celltypes"]
fig1, fig2 = plot_variance_qc(var_decomp_in_vitro_20CRE, figsize=(4, 6),
                              output_prefix="/gpfs/commons/groups/ren_lab/guojiezhong/starr-fish/STARRFISH_in_vivo/barcode_enhancer_decomposition/results/plots/invitro_20CRE", show=True)

# %% plot variance_decomp results
var_decomp_in_vitro_300CRE = pd.read_csv("/gpfs/commons/groups/ren_lab/guojiezhong/starr-fish/STARRFISH_in_vivo/barcode_enhancer_decomposition/results/invitro_300CRE/variance_decomp_results.csv", index_col=0)
var_decomp_in_vitro_300CRE = var_decomp_in_vitro_300CRE[var_decomp_in_vitro_300CRE["cell_type"]!="All_celltypes"]
fig1, fig2 = plot_variance_qc(var_decomp_in_vitro_300CRE, figsize=(4, 6),
                              output_prefix="/gpfs/commons/groups/ren_lab/guojiezhong/starr-fish/STARRFISH_in_vivo/barcode_enhancer_decomposition/results/plots/invitro_300CRE", show=True)
# %%
# show variance_decomp results for each cell type in vivo
var_decomp_in_vivo = pd.read_csv("/gpfs/commons/groups/ren_lab/guojiezhong/starr-fish/STARRFISH_in_vivo/barcode_enhancer_decomposition/results/variance_decomp/variance_decomp_results.csv", index_col=0)
# only keep barcodes and enhancers that both has structure
var_decomp_in_vivo = var_decomp_in_vivo[var_decomp_in_vivo["barcode_kernel_has_structure"] & var_decomp_in_vivo["enhancer_kernel_has_structure"]]
# filter for residual_qq_correlation > 0.95
var_decomp_in_vivo = var_decomp_in_vivo[var_decomp_in_vivo["residual_qq_correlation"] > 0.95]
# %%
# read the cell number of cell types
cell_numbers = pd.read_csv("/gpfs/commons/groups/ren_lab/guojiezhong/starr-fish/STARRFISH_in_vivo/results/expr3/celltype_number.csv")
var_decomp_in_vivo['cell_number'] = var_decomp_in_vivo['cell_type'].map(cell_numbers.set_index('subclass')['count'])
# %%
# --- Plot 1: Stacked Barplot of Variance Decomposition ---
# Sort by Enhancer proportion for better visualization
df_sorted = var_decomp_in_vivo.sort_values('cell_number', ascending=False)
fig1, fig2 = plot_variance_qc(df_sorted[df_sorted["cell_type"]=="All_celltypes"],
                              figsize=(3, 6), 
                              output_prefix="/gpfs/commons/groups/ren_lab/guojiezhong/starr-fish/STARRFISH_in_vivo/barcode_enhancer_decomposition/results/plots/invivo_all", show=True)
fig1, fig2 = plot_variance_qc(df_sorted[df_sorted["cell_type"]!="All_celltypes"],
                              figsize=(12, 6), 
                              output_prefix="/gpfs/commons/groups/ren_lab/guojiezhong/starr-fish/STARRFISH_in_vivo/barcode_enhancer_decomposition/results/plots/invivo", show=True)

# %%
def plot_noise_vs_activity(blup_df, activity_matrix_path, cell_type=None,
                           fig=None, ax=None, figsize=(7, 6)):
    """
    Scatter plot of per-construct noise proportion vs. in-vitro activity.

    Args:
        blup_df (pd.DataFrame): DataFrame from blup.csv with columns
            'construct', 'cell_type', 'prop_noise'.
        activity_matrix_path (str): Path to activity matrix CSV (rows = cell
            types, columns = constructs).
        cell_type (str | None): If given, restrict to that cell type; if None,
            plot all cell types with separate colours.
        fig, ax: Optional existing matplotlib Figure/Axes to draw into.
        figsize (tuple): Figure size when fig/ax are not provided.

    Returns:
        (fig, ax, merged_df)
    """
    # --- 1. Load and melt activity matrix ---
    act = pd.read_csv(activity_matrix_path, index_col=0)
    act_long = act.reset_index().melt(
        id_vars=act.index.name or "index",
        var_name="construct",
        value_name="activity"
    )
    act_long = act_long.rename(columns={act_long.columns[0]: "cell_type"})

    # --- 2. Merge with blup_df ---
    merged = blup_df.merge(act_long, on=["construct", "cell_type"], how="inner")

    if cell_type is not None:
        merged = merged[merged["cell_type"] == cell_type]

    if merged.empty:
        raise ValueError("No overlapping (construct, cell_type) pairs found after merge.")

    # --- 3. Plot ---
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    n_celltypes = merged["cell_type"].nunique()
    if n_celltypes == 1 or cell_type is not None:
        sns.scatterplot(data=merged, x="activity", y="prop_noise",
                        ax=ax, color="steelblue", alpha=0.75, s=60)
        sns.regplot(data=merged, x="activity", y="prop_noise",
                    scatter=False, ax=ax, color="red",
                    line_kws={"linestyle": "--", "alpha": 0.8})
    else:
        sns.scatterplot(data=merged, x="activity", y="prop_noise",
                        hue="cell_type", ax=ax, alpha=0.75, s=60)
        for ct, grp in merged.groupby("cell_type"):
            sns.regplot(data=grp, x="activity", y="prop_noise",
                        scatter=False, ax=ax,
                        line_kws={"linestyle": "--", "alpha": 0.6})

    # Pearson r annotation
    from scipy.stats import pearsonr
    r, p = pearsonr(merged["activity"], merged["prop_noise"])
    ax.annotate(f"r = {r:.2f}\np = {p:.3g}", xy=(0.05, 0.93),
                xycoords="axes fraction", fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))

    ax.set_xlabel("In-vitro Activity")
    ax.set_ylabel("Noise Proportion")
    title = "Noise vs. Activity"
    if cell_type:
        title += f" ({cell_type})"
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.4)

    if fig:
        fig.tight_layout()

    return fig, ax, merged


# %% noise vs activity — 20CRE (all cell types)
blup_20CRE = pd.read_csv("/gpfs/commons/groups/ren_lab/guojiezhong/starr-fish/STARRFISH_in_vivo/barcode_enhancer_decomposition/results/invitro_20CRE/blup.csv")
act_20CRE_path = "/gpfs/commons/groups/ren_lab/guojiezhong/starr-fish/STARRFISH_in_vitro/results/activity_matrix_20CRE.csv"

fig_nva, ax_nva, _ = plot_noise_vs_activity(blup_20CRE, act_20CRE_path)
fig_nva.savefig("/gpfs/commons/groups/ren_lab/guojiezhong/starr-fish/STARRFISH_in_vivo/barcode_enhancer_decomposition/results/plots/invitro_20CRE_noise_vs_activity.pdf", dpi=300)
fig_nva.show()

# %% noise vs activity — 300CRE (all cell types)
blup_300CRE = pd.read_csv("/gpfs/commons/groups/ren_lab/guojiezhong/starr-fish/STARRFISH_in_vivo/barcode_enhancer_decomposition/results/invitro_300CRE/blup.csv")
act_300CRE_path = "/gpfs/commons/groups/ren_lab/guojiezhong/starr-fish/STARRFISH_in_vitro/results/activity_matrix_300CRE.csv"

fig_nva_300, ax_nva_300, _ = plot_noise_vs_activity(blup_300CRE, act_300CRE_path)
fig_nva_300.savefig("/gpfs/commons/groups/ren_lab/guojiezhong/starr-fish/STARRFISH_in_vivo/barcode_enhancer_decomposition/results/plots/invitro_300CRE_noise_vs_activity.pdf", dpi=300)
fig_nva_300.show()

# %%
# plot the violin of barcode vs enhancer variance proportion
fig, ax = plt.subplots(figsize=(6, 6))
df_violin = var_decomp_in_vivo.melt(id_vars=["cell_type"], value_vars=["prop_enhancer", "prop_barcode"], var_name="Component", value_name="Proportion")
sns.violinplot(data=df_violin, x="Component", y="Proportion", ax=ax, palette={"prop_enhancer": "#2ca02c", "prop_barcode": "#d62728"})
ax.set_title("Distribution of Variance Proportions across Cell Types")
ax.set_xlabel("")
ax.set_xticklabels(["Enhancer", "Barcode"])
ax.grid(axis='y', linestyle='--', alpha=0.5)
fig.tight_layout()
save_path_violin = "/gpfs/commons/groups/ren_lab/guojiezhong/starr-fish/STARRFISH_in_vivo/barcode_enhancer_decomposition/results/plots/invivo_enhancer_barcode_violin.pdf"
fig.savefig(save_path_violin, dpi=300)
fig.show()
# %%
