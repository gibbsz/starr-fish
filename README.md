# STARR-FISH

STARR-FISH (self transcribing active regulatory region fluorescence *in situ* hybridization) is an imaging-based reporter assay that measures the transcriptional capacity of enhancer reporter constructs in a multiplexed, spatially resolved manner at single-molecule resolution. This repository includes:
- scripts for processing raw image data (smFISH or MERFISH) ⟶ `image_processing`
- custom code for enhancer activity quantification of *in vitro* STARR-FISH data ⟶ `STARR-FISH_in_vitro`
- a full analysis package that includes a comprehensive set of tools for activity quantification, statistical testing for cell type-intrinsic activity, integration with epigenomic data, and data visualization (see accompanying `README` for installation and usage details) ⟶ `STARR-FISH_in_vivo`
- example notebooks for analysis related to the STARR-FISH manuscript (submitted) ⟶ `STARR-FISH_manuscript`
- **BAYSTARRFISH**, the Bayesian hierarchical model of cCRE activity: it treats the unobserved number of infecting AAV genomes as a latent variable and marginalises it out, rather than taking a fold change over a background that is 99.85% zeros (see [`README_BAYSTARRFISH.md`](README_BAYSTARRFISH.md)) ⟶ `baystarrfish`
