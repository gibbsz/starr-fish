# Image processing

These scripts can be utilized to process raw MERFISH and/or STARR-FISH data in `zarr` format. This includes:
- Deconvolution and spot fitting
- Drift correction/image registration
- Decoding molecule identities (from combinatorial imaging experiments)

## How to implement these scripts for your data

Generally, users should only need to adjust the following file paths to use these processing pipelines:
1. `master_analysis_folder`: where the relevant files for analysis are stored
2. `lib_fl`: codebook file (csv format) used for MERFISH decoding
3. `psf_file`: point spread function calculated according to your imaging setup
4. `flat_field_tag`: keyword for the flat field files calculated for your imaging setup
5. `master_data_folder`: where the raw data (zarr format) is stored
6. `save_folder`: where the fitted spot files will be stored

## Running the worker scripts

The required functions for running the worker scripts are included in `ioMicro.py`. The following dependencies are required in your environment: