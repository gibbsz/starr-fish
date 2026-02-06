"""
STARRFISH: Single-cell Transcriptomic And Regulatory Region Readout For Identifying Specificity of cis-regulatory elements and their combinatorial signatures in cells at High-resolution

A Python package for analyzing STARR-FISH data.
"""

# Core utilities and analysis
from .utils import (
    STARRFISH,
    T7CRE_Joint_DistributionEM,
    T7CRE_Split_DistributionEM,
    DataPaths,
    fit_glm,
    glm,
    cre_deseq2,
    calculate_fold_change,
    col_corr,
    row_corr,
    cross_talk_fisher_test,
    cross_talk_corr_test,
    fit_sklearn_gauss_mixture,
    fit_stan,
    motif_enrichment,
)

# Plotting functions
from .plots import (
    plot_text,
    fetch_data,
    fetch_data_p,
    subplot_cre_corr_compare,
    subplot_cre_corr,
    plot_atac_cre_corr_compare,
    plot_pval_compare,
    plot_qval_compare,
    plot_atac_celltype_corr_compare,
    scatter_plot_with_margin_density_by_celltype,
    box_plot_by_celltype,
    scatter_plot_with_margin_density_by_cre,
    plot_celltype_activity_distribution_compare,
    plot_celltype_activity_atac_distribution_compare,
    plot_cre_activity_distribution_compare,
    plot_cre_activity_atac_distribution_compare,
    plot_all_and_save,
    negative_control_regression_plot,
    cre_corr_dotplot,
    cre_corr_heatmap,
    celltype_corr_dotplot,
    draw_custom_dendrogram,
    cre_pval_dotplot,
    cre_proportion_dotplot,
    celltype_pval_dotplot,
    plot_q_value_celltype_reproducibility,
    plot_q_value_cre_reproducibility,
    get_pr_df,
    plot_bar,
    plot_grouped_clustermap,
    average_foldchange_specificity_test,
    average_foldchange_specificity_t_test,
    q_value_correction,
    plot_reproducibility,
)

# Genomic tracks visualization
from .tracksClass import (
    VType,
    VlineType,
    VhighlightType,
    MultiDict,
    PlotTracks,
)

# Preprocessing utilities
from .get_preprocess_utils import (
    download_motif,
    join_peaks,
    query_motif,
    get_motif,
    create_peak_motif,
    zip_zarr,
    unzip_zarr,
    add_atpm,
)

# VAE models
from .starr_fish_vae import (
    DecoderInfectionRateVI,
    STARRFISHVAE,
    STARRFISHVI,
)

__version__ = "1.0.0"
__author__ = "Guojie Zhong"

__all__ = [
    # Core classes
    "STARRFISH",
    "T7CRE_Joint_DistributionEM",
    "T7CRE_Split_DistributionEM",
    "DataPaths",
    # Core functions
    "fit_glm",
    "glm",
    "cre_deseq2",
    "calculate_fold_change",
    "col_corr",
    "row_corr",
    "cross_talk_fisher_test",
    "cross_talk_corr_test",
    "fit_sklearn_gauss_mixture",
    "fit_stan",
    "motif_enrichment",
    # Plotting functions
    "plot_text",
    "fetch_data",
    "fetch_data_p",
    "subplot_cre_corr_compare",
    "subplot_cre_corr",
    "plot_atac_cre_corr_compare",
    "plot_pval_compare",
    "plot_qval_compare",
    "plot_atac_celltype_corr_compare",
    "scatter_plot_with_margin_density_by_celltype",
    "box_plot_by_celltype",
    "scatter_plot_with_margin_density_by_cre",
    "plot_celltype_activity_distribution_compare",
    "plot_celltype_activity_atac_distribution_compare",
    "plot_cre_activity_distribution_compare",
    "plot_cre_activity_atac_distribution_compare",
    "plot_all_and_save",
    "negative_control_regression_plot",
    "cre_corr_dotplot",
    "cre_corr_heatmap",
    "celltype_corr_dotplot",
    "draw_custom_dendrogram",
    "cre_pval_dotplot",
    "cre_proportion_dotplot",
    "celltype_pval_dotplot",
    "plot_q_value_celltype_reproducibility",
    "plot_q_value_cre_reproducibility",
    "get_pr_df",
    "plot_bar",
    "plot_grouped_clustermap",
    "average_foldchange_specificity_test",
    "average_foldchange_specificity_t_test",
    "q_value_correction",
    "plot_reproducibility",
    # Tracks visualization
    "VType",
    "VlineType",
    "VhighlightType",
    "MultiDict",
    "PlotTracks",
    # Preprocessing
    "download_motif",
    "join_peaks",
    "query_motif",
    "get_motif",
    "create_peak_motif",
    "zip_zarr",
    "unzip_zarr",
    "add_atpm",
    # VAE models
    "DecoderInfectionRateVI",
    "STARRFISHVAE",
    "STARRFISHVI",
]
