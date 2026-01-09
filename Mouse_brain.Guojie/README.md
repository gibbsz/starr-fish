# STARRFISH Package

**S**ingle-cell **T**ranscriptomic **A**nd **R**egulatory **R**egion **R**eadout **F**or **I**dentifying **S**pecificity of cis-regulatory elements and their combinatorial signatures in cells at **H**igh-resolution

A comprehensive Python package for analyzing STARR-FISH data to measure cis-regulatory element (CRE) activity at single-cell resolution.

## Features

- **Single-cell CRE Activity Measurement**: Quantify enhancer activity in individual cells
- **Cell Type-Specific Analysis**: Identify CREs with cell type-specific regulatory functions
- **Statistical Testing**: Multiple methods for detecting significant CRE-cell type associations
- **Normalization & Quality Control**: Extensive options for data normalization and filtering
- **Integration with ATAC-seq**: Validate CRE activity predictions using chromatin accessibility
- **Visualization Tools**: Comprehensive plotting functions for publication-quality figures
- **VAE Models**: Deep learning models for dimensionality reduction and batch correction
- **Genomic Tracks**: Create browser-like visualizations of CRE activity

## Installation

### From Source

```bash
cd ./starr-fish/Mouse_brain.Guojie/
conda env create -f STARRFISH.yaml
conda activate STARRFISH
pip install -e .
```

### Dependencies

The package requires Python 3.8+ and the following dependencies:
- numpy >= 1.20.0
- pandas >= 1.3.0
- scipy >= 1.7.0
- matplotlib >= 3.4.0
- seaborn >= 0.11.0
- scanpy >= 1.8.0
- anndata >= 0.8.0
- scikit-learn >= 0.24.0
- statsmodels >= 0.13.0
- scvi-tools >= 0.16.0
- pybedtools >= 0.9.0
- pysam >= 0.19.0
- zarr >= 2.10.0

**Optional dependencies:**
- pystan >= 3.0.0 (for Bayesian modeling via Stan)
  - **Note:** Not compatible with Python 3.13 due to `pysimdjson` dependency
  - Install with: `pip install STARRFISH[stan]`
  - Alternative: Use `model='sklearn_gaussian_mixture'` instead of Stan

## Quick Start

```python
from STARRFISH import STARRFISH

# Load STARR-FISH data
starrfish = STARRFISH(
    adata='data/starr_fish_data.h5ad',
    atac_cpm='data/atac_cpm.csv',
    celltype_tag='obs:subclass'
)

# Get CRE expression for all cells
cre_expr = starrfish.get_cre_expression()

# Run fold-change test with normalization
results = starrfish.fold_change_test(
    normalize_by_cell_rna=True,
    normalize_by_cell_volume=True,
    bootstrap_number=1000
)

# Get activity matrix
activity = results['activity']

# Correlate with ATAC-seq
corr, pval = starrfish.corr_atac_cpm(acvitity_df=activity)

# Save analysis
starrfish.save('starrfish_analysis.pkl')
```

## Package Structure

```
STARRFISH/
├── __init__.py              # Package initialization and exports
├── utils.py                 # Core STARRFISH class and statistical functions
├── plots.py                 # Visualization and plotting functions
├── get_preprocess_utils.py  # Data preprocessing utilities
├── tracksClass.py           # Genomic tracks visualization
└── starr_fish_vae.py        # VAE models for STARR-FISH data
```

## Documentation

Comprehensive API documentation is available in `STARRFISH_API_Documentation.md`, including:
- Detailed method descriptions
- Parameter specifications
- Usage examples
- Best practices

## Analysis Notebooks

Example Jupyter notebooks demonstrating key analyses:

- `analysis.figure_3.overview_of_data.ipynb` - Data quality and reproducibility assessment
- `analysis.figure_4.cCRE_activity.ipynb` - Cell type-specific CRE activity analysis
- `analysis.figure_S5.abc_expression.ipynb` - Validation with ABC Atlas gene expression
- `analysis.figure_S7.infection_rate.ipynb` - Infection rate analysis and filtering

## Key Functionality

### Statistical Testing Methods

1. **Fisher's Exact Test**: Test CRE activation enrichment in cell types
2. **Fold-Change Test**: Compute fold-change enrichment with bootstrap confidence intervals
3. **Average Bootstrap Test**: Calculate average CRE activity per cell type

### Normalization Options

- Cell-level normalization (RNA content, volume, T7 expression)
- Cell type-level normalization (median RNA, volume, T7)
- Negative control normalization
- Library size normalization
- Infected cell fraction normalization

### Data Integration

- ATAC-seq correlation analysis
- Multi-modal data integration via VAE
- Cross-experiment reproducibility assessment
- Integration with scanpy/AnnData ecosystem

### Visualization

- Correlation heatmaps and dotplots
- Cell type-specific activity profiles
- Genomic browser tracks
- Quality control plots
- Reproducibility assessments

## Data Format

STARRFISH uses AnnData objects with the following structure:

```python
AnnData object
├── .obs                    # Cell metadata (cell types, spatial coords, etc.)
├── .obsm
│   ├── 'CRE'              # CRE expression matrix (cells x CREs)
│   ├── 'T7CRE'            # T7-CRE expression (infection marker)
│   ├── 'X_raw'            # Raw RNA expression
│   └── 'X_spatial'        # Spatial coordinates
├── .uns
│   └── 'CRE_info'         # CRE metadata (genomic coords, target genes)
└── .var                   # Gene metadata
```

## Performance

STARRFISH is optimized for large-scale datasets:
- Parallel processing support (configurable with `n_jobs` parameter)
- Result caching to avoid recomputation
- Efficient bootstrap resampling
- Memory-efficient data structures

## Citation

If you use STARRFISH in your research, please cite:

```
Gibbs, Z., Zhong, G. et al. (2026). STARRFISH [Under submission].
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

For questions, issues, or contributions:
- **Author**: Guojie Zhong
- **Lab**: Ren Lab
- **Institution**: The Jackson Laboratory / Columbia University
- **Issues**: Please report bugs and feature requests via GitHub Issues

## Version History

- **v1.0.0** (2026-01-09): Initial release
  - Core STARRFISH class implementation
  - Statistical testing methods
  - Plotting and visualization tools
  - VAE models
  - Package structure and documentation
