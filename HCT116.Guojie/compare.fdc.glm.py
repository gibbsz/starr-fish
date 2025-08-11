# %% compare fold change test and glm
from fdc import preprocess_experiment as preprocess_experiment_fdc
from glm import preprocess_experiment as preprocess_experiment_glm
from glm import glm_fit, glm_fit_total
import statsmodels.api as sm
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
PWD = os.path.dirname(os.path.abspath(__file__))
sys.path.append(f'{PWD}/../Mouse_brain.Guojie/')
from utils import STARRFISH
# %%
july_glm = preprocess_experiment_glm(enhancer_file='data/SFv4_T7_July_enhancer_cbg.csv', vector_file='data/SFv4_T7_July_T7_cbg.csv', nanopore_file='data/SFv4_T7_20CRE_nanopore_counts',
                                     rna_seq_file='data/Bulk_sequencing/RNA_sequencing/CRE20_T7_July_featureCounts_output.txt',
                                     dna_seq_file='data/Bulk_sequencing/DNA_sequencing/CRE20_July_featureCounts_output.txt', 
                                     bed_file='data/20CRE.bed')
july_glm = glm_fit(july_glm, family=sm.families.Gaussian(), norm_by_total=False)
july_glm = glm_fit_total(july_glm, family=sm.families.Gaussian(), norm_by_vector=False, norm_by_nanopore=True, key_add='glm_fit_total_result_nanopore')
july_glm = glm_fit_total(july_glm, family=sm.families.Gaussian(), norm_by_vector=True, norm_by_nanopore=False, key_add='glm_fit_total_result_T7')
# %%
july_fdc = preprocess_experiment_fdc(enhancer_file='data/SFv4_T7_July_enhancer_cbg.csv', vector_file='data/SFv4_T7_July_T7_cbg.csv', nanopore_file='data/SFv4_T7_20CRE_nanopore_counts',
                                     rna_seq_file='data/Bulk_sequencing/RNA_sequencing/CRE20_T7_July_featureCounts_output.txt',
                                     dna_seq_file='data/Bulk_sequencing/DNA_sequencing/CRE20_July_featureCounts_output.txt', 
                                     bed_file='data/20CRE.bed')
starrfish_july = STARRFISH(july_fdc, atac_cpm=None, atac_counts=None, lib_size=None)
# set library size
lib_size = pd.DataFrame({'counts': starrfish_july.adata.var['nanopore'].values}, index=starrfish_july.adata.var_names)
starrfish_july.lib_size = lib_size.copy()
# %%
average_bootstrap_test_config = {
    'cell_types_to_use': None,
    'normalize_by_cell_rna': False,
    'normalize_by_cell_volume': False,
    'normalize_by_cell_t7': False,  # normalize by T7, filter cells with T7 < 4
    'normalize_by_celltype_rna': False,
    'normalize_by_celltype_volume': False,
    'normalize_by_celltype_t7': True,  # normalize by T7
    'filter_by_cell_t7': None,
    'normalize_by_negative_control': False,  # normalize by negative control
    'normalize_by_libsize': False,
    'log_transform': False,
    'bootstrap_number': 10000,
    'bootstrap_to_fixed_pct': 1.0,
    'bootstrap_to_fixed_sample_size': None,
    'load_stored': True,
    'n_jobs': 28,
}
threshold = 'neg_control_dist'
res1 = starrfish_july.average_bootstrap_test(**average_bootstrap_test_config)
res_q1, res_df1 = starrfish_july.average_bootstrap_test_q(res1, threshold=threshold, norm='libsize', tail='right')
# %%
# compare the results
fig, ax = plt.subplots(figsize=(8, 6))
# compare res_df1 vs july_glm
toplot = pd.DataFrame({
    'FDC': np.exp(res_df1.loc['CellLine']),
    'GLM': july_glm['glm_fit_result']['STARR-FISH Activity']
})
sns.scatterplot(data=toplot, x='FDC', y='GLM', ax=ax)
# label the CRE001
from adjustText import adjust_text
texts = []
for i in range(len(toplot)):
    texts.append(ax.text(toplot['FDC'][i], toplot['GLM'][i], toplot.index[i], fontsize=9))
adjust_text(texts)
# regression line
slope, intercept = np.polyfit(toplot['FDC'], toplot['GLM'], 1)
ax.plot(toplot['FDC'], slope * toplot['FDC'] + intercept, color='red')
# print slope and intercept
ax.text(0.1, 0.9, f'slope: {slope:.2f}\nintercept: {intercept:.2f}', transform=ax.transAxes, fontsize=10)
# %%
fig, ax = plt.subplots(figsize=(8, 6))
sns.scatterplot(y=july_glm['enhancer']['CRE001'], x=july_glm['vector']['CRE001'], ax=ax)

# %%
