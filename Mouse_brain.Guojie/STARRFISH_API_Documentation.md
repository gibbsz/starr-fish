# STARRFISH Package API Documentation

This document provides a comprehensive overview of the STARRFISH package for analyzing spatial transcriptomics data with CRE (cis-regulatory element) activity measurements.

## Table of Contents

1. [Installation](#installation)
2. [Package Structure](#package-structure)
3. [Core Class: STARRFISH](#core-class-starrfish)
   - [Initialization and I/O](#initialization-and-io)
   - [Data Getters](#data-getters)
   - [Data Processing Methods](#data-processing-methods)
   - [Control Methods](#control-methods)
   - [Statistical Testing Methods](#statistical-testing-methods)
   - [Analysis Methods](#analysis-methods)
4. [Additional Modules](#additional-modules)
   - [Plotting Functions](#plotting-functions)
   - [Preprocessing Utilities](#preprocessing-utilities)
   - [Genomic Tracks Visualization](#genomic-tracks-visualization)
   - [VAE Models](#vae-models)

---

## Installation

To install the STARRFISH package:

```bash
# From the source directory
cd /gpfs/commons/groups/ren_lab/guojiezhong/starr-fish/Mouse_brain.Guojie
pip install -e .
```

Or install directly from the package directory:

```bash
pip install /gpfs/commons/groups/ren_lab/guojiezhong/starr-fish/Mouse_brain.Guojie
```

---

## Package Structure

The STARRFISH package consists of the following modules:

- **`STARRFISH.utils`**: Core STARRFISH class and statistical functions
- **`STARRFISH.plots`**: Plotting and visualization functions
- **`STARRFISH.get_preprocess_utils`**: Data preprocessing utilities
- **`STARRFISH.tracksClass`**: Genomic tracks visualization classes
- **`STARRFISH.starr_fish_vae`**: Variational autoencoder models for STARR-FISH data

All main functions and classes can be imported directly from the STARRFISH package:

```python
from STARRFISH import STARRFISH, plot_all_and_save, PlotTracks
```

---

## Core Class: STARRFISH

## Initialization and I/O

### `__init__(adata, cre_tag='obsm:CRE', t7_tag='obsm:T7CRE', ...)`
Initialize STARRFISH object for analyzing spatial transcriptomics data with CRE activity.

**Parameters:**
- `adata`: AnnData object or path to .h5ad file containing spatial transcriptomics data
- `cre_tag`: Tag to access CRE expression data in adata (default: 'obsm:CRE')
- `t7_tag`: Tag to access T7-CRE expression data (default: 'obsm:T7CRE')
- `celltype_tag`: Tag to access cell type annotations (default: 'obs:subclass')
- `spatial_tag`: Tag to access spatial coordinates (default: 'obsm:X_spatial')
- `creinfo_tag`: Tag to access CRE metadata (default: 'uns:CRE_info')
- `atac_cpm`: DataFrame or path to ATAC-seq CPM data by cell type
- `atac_counts`: DataFrame or path to ATAC-seq count data by cell type
- `lib_size`: DataFrame or path to library size data for normalization
- `log_lib_size`: Whether to log-transform library size (default: True)
- `blacklist_cre`: List of CRE IDs to exclude from analysis

**Description:** Loads spatial transcriptomics data, processes cell type annotations, loads ATAC-seq data, and prepares library size normalization factors. Handles different cell type granularities (subclass, class, region).

---

### `save(path, overwrite_adata=False)`
Save STARRFISH object to file.

**Parameters:**
- `path`: Path to save the STARRFISH object (pickle file)
- `overwrite_adata`: Whether to overwrite existing AnnData file (default: False)

**Description:** Saves the STARRFISH object to a pickle file while optionally saving the AnnData object separately to an .h5ad file.

---

### `load(path, adata=None)` [static method]
Load STARRFISH object from file.

**Parameters:**
- `path`: Path to the saved STARRFISH pickle file
- `adata`: AnnData object or path to .h5ad file to load

**Returns:** STARRFISH object with data

**Description:** Loads a previously saved STARRFISH object from pickle file and optionally loads the associated AnnData object.

---

### `load_adata(adata_path)`
Load AnnData object from file.

**Parameters:**
- `adata_path`: Path to the .h5ad file

**Description:** Loads an AnnData object from disk and stores it in the STARRFISH object.

---

### `load_cpm(cpm_path, attr_to_add='atac_cpm')`
Load CPM (counts per million) data from CSV file.

**Parameters:**
- `cpm_path`: Path to CSV file containing CPM data
- `attr_to_add`: Attribute name to store the loaded CPM data (default: 'atac_cpm')

**Description:** Loads and processes CPM data, matches it to CRE information, and stores as a transposed DataFrame.

---

### `load_libsize(lib_size_path, log_transform=True)`
Load library size data for CRE normalization.

**Parameters:**
- `lib_size_path`: Path to CSV file containing library size data
- `log_transform`: Whether to apply log1p transformation to library sizes (default: True)

**Description:** Loads library size data and optionally log-transforms. Stores both raw and processed versions.

---

## Data Getters

### `get_tag(tag)`
Retrieve data from AnnData object using tag notation.

**Parameters:**
- `tag`: Tag in format 'attribute:key' (e.g., 'obs:subclass', 'obsm:CRE')

**Returns:** pd.DataFrame or pd.Series or None

**Description:** Parses tag string to access nested attributes in AnnData object.

---

### `get_cre_expression()`
Get CRE expression data for all cells.

**Returns:** pd.DataFrame with cells as rows and CREs as columns

**Description:** Retrieves CRE expression data from the location specified by self.cre_tag.

---

### `get_t7_expression()`
Get T7-CRE expression data for all cells.

**Returns:** pd.DataFrame or None

**Description:** T7-CRE is used to measure transfection/infection efficiency. Returns None if T7 data unavailable.

---

### `get_rna_expression()`
Get RNA expression data for all cells.

**Returns:** pd.DataFrame with cells as rows and genes as columns

**Description:** Retrieves raw RNA expression data from 'obsm:X_raw' in the AnnData object.

---

### `get_celltypes(celltype_tag=None)`
Get cell type annotations for all cells.

**Parameters:**
- `celltype_tag`: Tag specifying cell type location (default: uses self.celltype_tag)

**Returns:** pd.Series with cell IDs as index and cell type labels as values

**Description:** Retrieves cell type annotations from the AnnData object.

---

### `get_creinfo()`
Get metadata information for all CREs.

**Returns:** pd.DataFrame with CRE IDs as index and metadata columns

**Description:** Contains information such as enhancer coordinates, target genes, and cell type specificity.

---

## Data Processing Methods

### `get_k_nearest_neighbors(cell_id, k=10, spatial_tag='obsm:X_spatial')`
Find k nearest neighbor cells based on spatial coordinates.

**Parameters:**
- `cell_id`: Cell identifier to find neighbors for
- `k`: Number of nearest neighbors to find (default: 10)
- `spatial_tag`: Tag specifying spatial coordinate location (default: 'obsm:X_spatial')

**Returns:** pd.DataFrame with neighbor cell IDs as index and columns for distance, X, and Y coordinates

**Description:** Uses Euclidean distance to find spatial neighbors. The query cell itself is excluded.

---

### `get_cre_expression_normalized(cell_types_to_use=None, normalize_by_cell_rna=True, normalize_by_volume=True, log_transform=False)`
Get normalized CRE expression data.

**Parameters:**
- `cell_types_to_use`: List of cell types to include (default: all cell types)
- `normalize_by_cell_rna`: Normalize by RNA content per cell (default: True)
- `normalize_by_volume`: Normalize by cell volume (default: True)
- `log_transform`: Apply log1p transformation (default: False)

**Returns:** Tuple of (normalized CRE expression matrix, cell type labels)

**Description:** Performs cell-level normalization to account for differences in RNA content and/or cell volume.

---

### `get_cre_celltypes(celltypes, celltype_tag=None)`
Get CRE expression data for specific cell types.

**Parameters:**
- `celltypes`: List of cell type labels to filter for
- `celltype_tag`: Tag specifying cell type location (default: uses self.celltype_tag)

**Returns:** Tuple of (filtered CRE expression matrix, cell type labels)

**Description:** Filters cells to include only those matching the specified cell types.

---

### `get_cre_rna_celltypes(celltypes)`
Get both CRE and RNA expression data for specific cell types.

**Parameters:**
- `celltypes`: List of cell type labels to filter for

**Returns:** Tuple of (CRE expression matrix, RNA expression matrix, cell type labels)

**Description:** Filters cells and returns both CRE and RNA data for downstream normalization or analysis.

---

## Control Methods

### `get_negative_control_cres()`
Get negative control CREs.

**Returns:** pd.Index of CRE IDs labeled as negative controls

**Description:** Negative controls are CREs that should not show cell type-specific activity and are used as background references.

---

### `get_positive_control_cres(cell_type, use='define')`
Get positive control CREs for a given cell type.

**Parameters:**
- `cell_type`: Cell type label to find positive control CREs for
- `use`: Method to define positive controls: 'define', 'atac-peak', 'h3k27ac-peak', 'h3k4me1-peak', 'chromatin-a', or 'chromatin-o' (default: 'define')

**Returns:** pd.Index of CRE IDs or None

**Description:** Returns CREs expected to show activity in the specified cell type based on various epigenomic evidence.

---

### `get_positive_control_celltypes(cre, use='define')`
Get positive control cell types for a given CRE.

**Parameters:**
- `cre`: CRE ID to find positive control cell types for
- `use`: Method to define positive controls (default: 'define')

**Returns:** pd.Series or pd.Index of cell type labels or None

**Description:** Returns cell types where the given CRE should show activity based on epigenomic evidence.

---

### `get_atac_z_cres(cell_type, z=2)`
Get CREs with ATAC-seq signal above z-score threshold for a cell type.

**Parameters:**
- `cell_type`: Cell type label to query ATAC-seq data for
- `z`: Z-score threshold for ATAC signal (default: 2)

**Returns:** pd.Index of CRE IDs or None

**Description:** Identifies CREs with strong chromatin accessibility by computing z-scores of log-transformed ATAC CPM values.

---

## Statistical Testing Methods

### `fisher_exact_test(cell_types_to_use=None, activate_threshold=2, infect_threshold=1)`
Perform Fisher's exact test for CRE activation enrichment in cell types.

**Parameters:**
- `cell_types_to_use`: List of cell types to include (default: all)
- `activate_threshold`: Expression threshold to consider a CRE as activated (default: 2)
- `infect_threshold`: T7 expression threshold to consider a cell as infected (default: 1)

**Returns:** Dictionary containing 'activity' DataFrame with p-values, 'config', and statistics

**Description:** Tests whether each CRE shows significant enrichment of activation in each cell type using Fisher's exact test. Results are cached.

---

### `fold_change_test(cell_types_to_use=None, normalize_by_cell_rna=False, ...)`
Compute fold-change enrichment of CRE activity across cell types with extensive normalization options.

**Parameters:**
- `cell_types_to_use`: Cell types to include in analysis
- `normalize_by_cell_rna`: Normalize by RNA content per cell (default: False)
- `normalize_by_cell_volume`: Normalize by cell volume (default: False)
- `normalize_by_cell_t7`: Normalize by T7 expression per cell (default: False)
- `filter_by_cell_t7`: Filter cells with T7 expression below threshold
- `normalize_by_celltype_rna`: Normalize by median RNA per cell type (default: False)
- `normalize_by_celltype_volume`: Normalize by median volume per cell type (default: False)
- `normalize_by_negative_control`: Normalize by negative control CRE expression (default: False)
- `normalize_by_infected_cell`: Normalize by infected cell fraction (default: False)
- `normalize_by_celltype_t7`: Normalize by median T7 per cell type (default: False)
- `normalize_by_total_cre`: Normalize by total CRE expression (default: False)
- `normalize_by_libsize`: Normalize by library size (default: False)
- `filter_zero_counts`: Filter out zero counts (default: False)
- `log_transform`: Apply log transformation at cell level (default: False)
- `binarize_t7`: Convert T7 to binary infected/uninfected (default: False)
- `bootstrap_number`: Number of bootstrap iterations
- `bootstrap_to_fixed_sample_size`: Resample to fixed sample size
- `calculate_fdc`: Calculate fold discovery curve (default: False)
- `fill_nan`: Fill NaN values with 0 (default: True)
- `n_jobs`: Number of parallel jobs (default: 256)
- `load_stored`: Load cached results if available (default: True)

**Returns:** Dictionary containing 'activity' DataFrame with fold-change values, 'config', 'bootstrap_std', and statistics

**Description:** Computes fold-change of each CRE in each cell type relative to other cell types, with extensive normalization and bootstrap options. Results are cached.

---

### `average_bootstrap_test(cell_types_to_use, normalize_by_cell_rna=False, ...)`
Compute average CRE activity per cell type with bootstrap confidence intervals.

**Parameters:**
- `cell_types_to_use`: Cell types to include in analysis
- `normalize_by_cell_rna`: Normalize by RNA content per cell (default: False)
- `normalize_by_cell_volume`: Normalize by cell volume (default: False)
- `normalize_by_cell_t7`: Normalize by T7 expression per cell (default: False)
- `normalize_by_celltype_rna`: Normalize by median RNA per cell type (default: False)
- `normalize_by_celltype_volume`: Normalize by median volume per cell type (default: False)
- `normalize_by_negative_control`: Normalize by negative control CRE (default: False)
- `normalize_by_celltype_t7`: Normalize by median T7 per cell type (default: False)
- `filter_by_cell_t7`: Filter cells with T7 expression below threshold
- `normalize_by_libsize`: Normalize by library size (default: False)
- `log_transform`: Apply log transformation (default: False)
- `bootstrap_number`: Number of bootstrap iterations
- `bootstrap_to_fixed_sample_size`: Resample to fixed sample size
- `bootstrap_to_fixed_pct`: Resample to fixed percentage of cells
- `fill_nan`: Fill NaN values with 0 (default: True)
- `n_jobs`: Number of parallel jobs (default: 256)
- `load_stored`: Load cached results if available (default: True)

**Returns:** Dictionary containing 'activity' DataFrame with average expression, 'bootstrap_std', and 'config'

**Description:** Similar to fold_change_test but computes average expression within each cell type rather than fold-changes. Bootstrap resampling is performed within each cell type. Results are cached.

---

## Analysis Methods

### `scvi(use_model='STARRFISHVI', model_args=None, train_args=None)`
Run single-cell variational inference (scVI) to model CRE activity.

**Parameters:**
- `use_model`: Model to use: 'STARRFISHVI' or 'SCVI' (default: 'STARRFISHVI')
- `model_args`: Arguments to pass to model initialization
- `train_args`: Arguments to pass to model training

**Returns:** Dictionary containing 'model', 'latent' representations, and 'config'

**Description:** Uses variational inference to learn low-dimensional representations of CRE activity while accounting for technical variation. STARRFISHVI is optimized for STARR-FISH data.

---

### `corr_atac_cpm(cell_types_to_use=None, cres_to_use=None, acvitity_df=None, ...)`
Compute correlation between STARR-FISH activity and ATAC-seq chromatin accessibility.

**Parameters:**
- `cell_types_to_use`: Cell types to include in correlation
- `cres_to_use`: CREs to include in correlation
- `acvitity_df`: Custom activity matrix (default: uses stored results)
- `log_atac`: Log-transform ATAC-seq values (default: False)
- `log_activity`: Log-transform activity values (default: False)
- `filter_by_atac_z_threshold`: Filter CREs by ATAC-seq z-score threshold
- `filter_by_atac_raw_threshold`: Filter CREs by raw ATAC-seq threshold
- `filter_by_negative_control_z_threshold`: Filter by negative control z-score
- `attr_to_use`: Attribute name for ATAC data (default: 'atac_cpm')

**Returns:** Tuple of (correlation matrix, p-value matrix) for cell types x CREs

**Description:** Computes Pearson correlation between STARR-FISH measured enhancer activity and ATAC-seq chromatin accessibility across cell types to validate CRE activity predictions.

---

### `corr_starrfish(activity_df1, activity_df2, cell_types_to_use=None, log_activity=False)` [static method]
Compute correlation between two STARR-FISH activity matrices.

**Parameters:**
- `activity_df1`: First activity matrix (cell types x CREs)
- `activity_df2`: Second activity matrix (cell types x CREs)
- `cell_types_to_use`: Cell types to include in correlation
- `log_activity`: Log-transform activity values before correlation (default: False)

**Returns:** Tuple of (correlation matrix, p-value matrix) for CREs x CREs

**Description:** Static method to compare CRE activity patterns between different STARR-FISH experiments or analysis methods. Useful for validating reproducibility or comparing normalization strategies.

---

## Usage Example

```python
import scanpy as sc
from STARRFISH import STARRFISH

# Initialize STARRFISH object
starrfish = STARRFISH(
    adata='path/to/data.h5ad',
    atac_cpm='path/to/atac_cpm.csv',
    celltype_tag='obs:subclass'
)

# Get CRE expression data
cre_expr = starrfish.get_cre_expression()

# Get positive control CREs for a cell type
pos_cres = starrfish.get_positive_control_cres('Excitatory neurons')

# Run fold-change test
results = starrfish.fold_change_test(
    normalize_by_cell_rna=True,
    normalize_by_cell_volume=True,
    bootstrap_number=1000
)

# Get activity matrix
activity = results['activity']

# Correlate with ATAC-seq
corr, pval = starrfish.corr_atac_cpm(acvitity_df=activity)

# Save results
starrfish.save('starrfish_analysis.pkl')
```

---

## Additional Modules

### Plotting Functions

The `STARRFISH.plots` module provides comprehensive visualization functions for STARR-FISH data analysis.

**Key Functions:**

```python
from STARRFISH import (
    plot_all_and_save,
    cre_corr_dotplot,
    cre_pval_dotplot,
    plot_atac_cre_corr_compare,
    plot_reproducibility,
    negative_control_regression_plot,
)

# Generate comprehensive analysis plots
plot_all_and_save(
    obj1=starrfish1,
    obj2=starrfish2,
    cell_types_to_use=cell_types,
    cres_to_use=cres,
    test_method='fold_change_test',
    test_configs=configs,
    save_dir='results/'
)

# Create correlation dotplot
cre_corr_dotplot(
    obj=starrfish,
    cres_to_use=cres,
    cell_types_to_use=cell_types,
    mods=['CRE', 'T7'],
    test_method='fold_change_test',
    test_configs=configs
)

# Visualize negative control regression
negative_control_regression_plot(
    obj=starrfish,
    cell_types_to_check=cell_types
)
```

**Available Plotting Functions:**
- `plot_text()`: Add text annotations to plots
- `subplot_cre_corr()`: Create CRE correlation subplots
- `scatter_plot_with_margin_density_by_celltype()`: Scatter plots with marginal densities
- `box_plot_by_celltype()`: Box plots grouped by cell type
- `plot_celltype_activity_distribution_compare()`: Compare activity distributions
- `draw_custom_dendrogram()`: Create custom hierarchical clustering dendrograms
- `cre_proportion_dotplot()`: Visualize CRE proportion across cell types
- `plot_q_value_celltype_reproducibility()`: Assess reproducibility of q-values
- `plot_grouped_clustermap()`: Create grouped clustermaps
- `q_value_correction()`: Apply FDR correction to p-values
- `get_pr_df()`: Generate precision-recall dataframes

---

### Preprocessing Utilities

The `STARRFISH.get_preprocess_utils` module provides tools for preprocessing genomic data and preparing inputs for STARR-FISH analysis.

**Key Functions:**

```python
from STARRFISH import (
    download_motif,
    join_peaks,
    query_motif,
    get_motif,
    create_peak_motif,
    add_atpm,
)

# Download motif database
download_motif(
    motif_url='http://example.com/motifs.bed',
    index_url='http://example.com/motifs.bed.tbi',
    motif_dir='data/motifs'
)

# Join peak files
joined_peaks = join_peaks(
    peak_bed='peaks.bed',
    reference_peaks='reference_peaks.bed'
)

# Query motifs in peaks
motif_results = query_motif(
    peak_bed='peaks.bed',
    motif_bed='motifs.bed'
)

# Get motif information for peaks
motif_info = get_motif(
    peak_file='peaks.bed',
    motif_file='motifs.bed',
    assembly='mm10'
)

# Create peak-motif zarr file
create_peak_motif(
    peak_motif_bed='peak_motifs.bed',
    output_zarr='peak_motifs.zarr',
    peak_bed='peaks.bed'
)

# Add ATAC-seq TPM data
add_atpm(
    zarr_file='data.zarr',
    bed_file='atac_peaks.bed',
    celltype='Excitatory neurons'
)
```

**Available Functions:**
- `download_motif()`: Download and index motif databases
- `join_peaks()`: Merge and join peak files
- `query_motif()`: Query motif occurrences in peaks
- `get_motif()`: Extract motif information for genomic regions
- `create_peak_motif()`: Create peak-motif association zarr files
- `zip_zarr()`: Compress zarr files
- `unzip_zarr()`: Decompress zarr files
- `add_atpm()`: Add ATAC-seq TPM data to zarr files

---

### Genomic Tracks Visualization

The `STARRFISH.tracksClass` module provides classes for creating publication-quality genomic browser tracks.

**Key Classes:**

```python
from STARRFISH import PlotTracks, VlineType, VhighlightType

# Create PlotTracks object
tracks = PlotTracks(
    figsize=(12, 8),
    TrackHeight=0.5,
    trackNum=5
)

# Add genomic tracks
tracks.add_track(
    data=bigwig_file,
    track_type='bigwig',
    label='ATAC-seq',
    color='blue'
)

# Add vertical line annotation
vline = VlineType(position=100000, color='red', linestyle='--')
tracks.add_vline(vline)

# Add highlight region
highlight = VhighlightType(start=95000, end=105000, color='yellow', alpha=0.3)
tracks.add_highlight(highlight)

# Plot the tracks
tracks.plot(
    chrom='chr1',
    start=90000,
    end=110000
)
```

**Available Classes:**
- `PlotTracks`: Main class for creating genomic tracks visualizations
- `VType`: Base class for vertical annotations
- `VlineType`: Vertical line annotations
- `VhighlightType`: Highlighted region annotations
- `MultiDict`: Ordered dictionary for managing multiple track properties

**Features:**
- Support for BigWig, BED, BAM, and custom data formats
- Flexible track styling and coloring
- Vertical annotations and region highlighting
- Publication-quality figure export

---

### VAE Models

The `STARRFISH.starr_fish_vae` module provides variational autoencoder models specifically designed for STARR-FISH data.

**Key Classes:**

```python
from STARRFISH import STARRFISHVAE, STARRFISHVI, DecoderInfectionRateVI

# Initialize STARRFISH VAE model
model = STARRFISHVAE(
    adata=adata,
    n_latent=10,
    n_layers=2,
    infection_rate_prior=True
)

# Train the model
model.train(
    max_epochs=200,
    lr=1e-3,
    use_gpu=True
)

# Get latent representation
latent = model.get_latent_representation()

# Get normalized expression
normalized_expr = model.get_normalized_expression()

# Initialize STARRFISH VI model (Variational Inference)
vi_model = STARRFISHVI(
    adata=adata,
    n_latent=10,
    modality_weights='equal',
    modality_penalty='kl'
)

# Train VI model
vi_model.train(max_epochs=200)
```

**Available Classes:**
- `STARRFISHVAE`: Variational autoencoder for STARR-FISH data
- `STARRFISHVI`: Variational inference model for multi-modal data
- `DecoderInfectionRateVI`: Custom decoder accounting for infection rate heterogeneity

**Features:**
- Account for infection rate heterogeneity across cells
- Multi-modal integration of CRE and RNA expression
- Batch effect correction
- Latent space representation for dimensionality reduction
- Integration with scanpy/AnnData workflows

---

## Notes

- All statistical test methods cache results to avoid recomputation
- Most methods support extensive customization through normalization and filtering options
- The package uses NumPy-style docstrings compatible with Sphinx documentation generation
- For detailed method signatures and parameters, refer to the source code documentation
- The package is designed to integrate seamlessly with scanpy/AnnData workflows
- Parallel processing is supported for computationally intensive methods

---

## Citation

If you use STARRFISH in your research, please cite:

```
Gibbs, Z., Zhong, G. et al. (2026). STARRFISH [Under submission].
```

---

**Generated:** 2026-01-09
**Version:** STARRFISH Package v1.0.0
**Location:** `/gpfs/commons/groups/ren_lab/guojiezhong/starr-fish/Mouse_brain.Guojie/STARRFISH/`
