#!/usr/bin/env python3
"""Extract the 20CRE/in vitro activity matrix from GLM fitting.

Runs glm_fit on July and Sept experiments (lines 289-290 of glm.py),
then saves a CRE x experiment activity matrix for downstream analysis.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import statsmodels.api as sm
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from glm import preprocess_experiment, glm_fit, glm_fit_total

# Change to STARRFISH_in_vitro directory for relative data paths
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Preprocess experiments (same as glm.py lines 275-282)
print("Preprocessing July experiment...")
july_experiment = preprocess_experiment(
    enhancer_file='data/SFv4_T7_July_enhancer_cbg.csv',
    vector_file='data/SFv4_T7_July_T7_cbg.csv',
    nanopore_file='data/SFv4_T7_20CRE_nanopore_counts',
    rna_seq_file='data/Bulk_sequencing/RNA_sequencing/CRE20_T7_July_featureCounts_output.txt',
    dna_seq_file='data/Bulk_sequencing/DNA_sequencing/CRE20_July_featureCounts_output.txt',
    bed_file='data/20CRE.bed',
)

print("Preprocessing Sept experiment...")
sept_experiment = preprocess_experiment(
    enhancer_file='data/SFv4_T7_Sept_enhancer_cbg.csv',
    vector_file='data/SFv4_T7_Sept_T7_cbg.csv',
    nanopore_file='data/SFv4_T7_20CRE_nanopore_counts',
    rna_seq_file='data/Bulk_sequencing/RNA_sequencing/CRE20_T7_Sept_featureCounts_output.txt',
    dna_seq_file='data/Bulk_sequencing/DNA_sequencing/CRE20_Sept_featureCounts_output.txt',
    bed_file='data/20CRE.bed',
)

print("Preprocessing 300CRE experiment 1 ...")
CRE_300_experiment1 = preprocess_experiment(
    enhancer_file='data/SFv6_cell_by_CRE_01_04_2023.csv', 
    vector_file=None, 
    nanopore_file='data/SFv6_300CRE_nanopore_counts',
    bed_file='data/STARR-FISH_300_library.bed'
)

print("Preprocessing 300CRE experiment 2 ...")
CRE_300_experiment2 = preprocess_experiment(
    enhancer_file='data/SFv6_cell_by_CRE_03_19_2023.csv', 
    vector_file=None, 
    nanopore_file='data/SFv6_300CRE_nanopore_counts',
    bed_file='data/STARR-FISH_300_library.bed'
)

print("Preprocessing WTC11 ...")
CRE_300_experiment3 = preprocess_experiment(
    enhancer_file='data/SFv6_WTC11_D3_enhancer_cbg.csv', 
    vector_file=None, 
    nanopore_file='data/SFv6_300CRE_nanopore_counts',
    bed_file='data/STARR-FISH_300_library.bed'
)

# GLM fit (same as glm.py lines 289-290)
print("Fitting GLM (July)...")
july_experiment = glm_fit(july_experiment, family=sm.families.Gaussian(), norm_by_total=False)
print("Fitting GLM (Sept)...")
sept_experiment = glm_fit(sept_experiment, family=sm.families.Gaussian(), norm_by_total=False)
print("Fitting GLM (300CRE) 1...")
CRE_300_experiment1 = glm_fit_total(CRE_300_experiment1, family=sm.families.Gaussian(), norm_by_vector=False, norm_by_nanopore=True, key_add='glm_fit_total_result_nanopore')
print("Fitting GLM (300CRE) 2...")
CRE_300_experiment2 = glm_fit_total(CRE_300_experiment2, family=sm.families.Gaussian(), norm_by_vector=False, norm_by_nanopore=True, key_add='glm_fit_total_result_nanopore')
print("Fitting GLM (300CRE) 3...")
CRE_300_experiment3 = glm_fit_total(CRE_300_experiment3, family=sm.families.Gaussian(), norm_by_vector=False, norm_by_nanopore=True, key_add='glm_fit_total_result_nanopore')
    
# Extract STARR-FISH Activity values into a matrix
# Rows = experiments ("cell types"), Columns = CREs
july_activity = july_experiment['glm_fit_result']['STARR-FISH Activity']
sept_activity = sept_experiment['glm_fit_result']['STARR-FISH Activity']
CRE_300_activity1 = CRE_300_experiment1['glm_fit_total_result_nanopore']['STARR-FISH Activity']*100 # Scale by 100 to match the scale of the 20CRE experiments (since they are not normalized by nanopore counts)
CRE_300_activity2 = CRE_300_experiment2['glm_fit_total_result_nanopore']['STARR-FISH Activity']*100
CRE_300_activity3 = CRE_300_experiment3['glm_fit_total_result_nanopore']['STARR-FISH Activity']*1000000

activity_matrix_20CRE = pd.DataFrame({
    'July': july_activity,
    'Sept': sept_activity,
}).T  # Shape: (2 experiments) x (20 CREs)

activity_matrix_300CRE = pd.DataFrame({
    '300CRE_1': CRE_300_activity1,
    '300CRE_2': CRE_300_activity2,
    '300CRE_3': CRE_300_activity3,
}).T  # Shape: (3 experiments) x (300 CREs)

print(f"\nActivity matrix shape: {activity_matrix_20CRE.shape}")
print(activity_matrix_20CRE)

print(f"\nActivity matrix shape: {activity_matrix_300CRE.shape}")
print(activity_matrix_300CRE)

# Save
outpath = os.path.join(os.path.dirname(__file__), 'results', 'activity_matrix_20CRE.csv')
os.makedirs(os.path.dirname(outpath), exist_ok=True)
activity_matrix_20CRE.to_csv(outpath)
print(f"\nSaved activity matrix to {outpath}")

outpath = os.path.join(os.path.dirname(__file__), 'results', 'activity_matrix_300CRE.csv')
os.makedirs(os.path.dirname(outpath), exist_ok=True)
activity_matrix_300CRE.to_csv(outpath)
print(f"\nSaved activity matrix to {outpath}")
