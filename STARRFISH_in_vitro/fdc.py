# %%
import statsmodels.api as sm
import statsmodels.formula.api as smf
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy
import scanpy as sc
import warnings
warnings.filterwarnings('ignore')
# prepare the utils.py from STARRFISH_in_vivo
import sys
import os
PWD = os.path.dirname(os.path.abspath(__file__))
sys.path.append(f'{PWD}/../STARRFISH_in_vivo/')
from utils import STARRFISH
# %%
def preprocess_experiment(enhancer_file, vector_file=None, nanopore_file=None, rna_seq_file=None, dna_seq_file=None, bed_file=None):
    enhancer = pd.read_csv(enhancer_file)
    if vector_file is None:
        # use enhancer as vector as a place holder, so that I don't have to change the code
        vector = enhancer.copy()
    else:
        vector = pd.read_csv(vector_file)
    if 'masks' in enhancer.columns and 'fov' in enhancer.columns:
        enhancer.index = enhancer['masks'].astype(str) + '_' + enhancer['fov'].astype(str)
        vector.index = vector['masks'].astype(str) + '_' + vector['fov'].astype(str)
    else:
        # infer mask and fov from the index
        enhancer['fov'] = enhancer.index.str.split('-').str[0]
        enhancer['masks'] = enhancer.index.str.split('-').str[1]
        vector['fov'] = vector.index.str.split('-').str[0]
        vector['masks'] = vector.index.str.split('-').str[1]
    # find common index and filter
    common_index = enhancer.index.intersection(vector.index)
    enhancer = enhancer.loc[common_index]
    vector = vector.loc[common_index]
    # drop mask and fov columns
    enhancer_drop = enhancer.copy()
    vector_drop = vector.copy()
    for col in ['masks', 'fov', 'total transcripts']:
        if col in enhancer.columns:
            enhancer_drop.drop(columns=col, inplace=True)
        if col in vector.columns:
            vector_drop.drop(columns=col, inplace=True)
    adata = sc.AnnData(X = np.zeros_like(enhancer_drop.values), 
                       obs=enhancer[['masks', 'fov']], 
                       var=pd.DataFrame(index=enhancer_drop.columns))
    if 'total transcripts' in enhancer.columns:
        adata.obs['total transcripts'] = enhancer['total transcripts'].values
    adata.obsm['CRE'] = enhancer_drop
    adata.obsm['T7CRE'] = vector_drop
    adata.obsm['X_raw'] = np.zeros_like(enhancer_drop.values)
    if nanopore_file is not None:
        nanopore = pd.read_csv(nanopore_file, sep=' ', skipinitialspace=True, header=None)
        nanopore.set_index(1, inplace=True)
        # fullfill the index to CRE names
        cre_names = enhancer_drop.columns[enhancer_drop.columns != 'total transcripts']
        adata.var['nanopore'] = nanopore.reindex(cre_names, fill_value=0)
    if rna_seq_file is not None:
        rna_counts = pd.read_csv(rna_seq_file, sep='\t', skipinitialspace=True, skiprows=1)
        rna_counts.set_index('Geneid', inplace=True)
        adata.var['rna_counts'] = rna_counts.iloc[:, -1]
    if dna_seq_file is not None:
        dna_counts = pd.read_csv(dna_seq_file, sep='\t', skipinitialspace=True, skiprows=1)
        dna_counts.set_index('Geneid', inplace=True)
        adata.var['dna_counts'] = dna_counts.iloc[:, -1]
    if bed_file is not None:
        ccre_names = pd.read_csv(bed_file, header=None, sep='\t')
        ccre_names = ccre_names.astype(str)
        ccre_names.index = ['CRE' + str(i+1).zfill(3) for i in range(len(ccre_names))]
        ccre_names['name'] = (ccre_names[0] + ":" + ccre_names[1] + "-" + ccre_names[2])
        ccre_names.columns = ['chr', 'start', 'end', 'name']
        ccre_names = ccre_names.reindex(adata.var_names, fill_value=pd.NA)
        adata.var = pd.concat([adata.var, ccre_names], axis=1)
    adata.obs['class'] = 'CellLine'

    adata.obs['subclass'] = 'CellLine'
    adata.obs['volm'] = 0
    adata.uns['CRE_info'] = adata.var.copy()
    adata.uns['CRE_info']['labeling_type'] = 'positive control'
    adata.uns['CRE_info']['labeling_type'][pd.isna(adata.uns['CRE_info']['chr'])] = 'negative control'
    return adata
# %%
if __name__ == '__main__':
    # %%
    july_experiment = preprocess_experiment(enhancer_file='data/SFv4_T7_July_enhancer_cbg.csv', vector_file='data/SFv4_T7_July_T7_cbg.csv', nanopore_file='data/SFv4_T7_20CRE_nanopore_counts',
                                            rna_seq_file='data/Bulk_sequencing/RNA_sequencing/CRE20_T7_July_featureCounts_output.txt',
                                            dna_seq_file='data/Bulk_sequencing/DNA_sequencing/CRE20_July_featureCounts_output.txt', 
                                            bed_file='data/20CRE.bed')
    sept_experiment = preprocess_experiment(enhancer_file='data/SFv4_T7_Sept_enhancer_cbg.csv', vector_file='data/SFv4_T7_Sept_T7_cbg.csv', nanopore_file='data/SFv4_T7_20CRE_nanopore_counts',
                                            rna_seq_file='data/Bulk_sequencing/RNA_sequencing/CRE20_T7_Sept_featureCounts_output.txt',
                                            dna_seq_file='data/Bulk_sequencing/DNA_sequencing/CRE20_Sept_featureCounts_output.txt', 
                                            bed_file='data/20CRE.bed')
    CRE_300_experiment1 = preprocess_experiment(enhancer_file='data/SFv6_cell_by_CRE_01_04_2023.csv', vector_file=None, nanopore_file='data/SFv6_300CRE_nanopore_counts',
                                                bed_file='data/STARR-FISH_300_library.bed')
    CRE_300_experiment2 = preprocess_experiment(enhancer_file='data/SFv6_cell_by_CRE_03_19_2023.csv', vector_file=None, nanopore_file='data/SFv6_300CRE_nanopore_counts',
                                                bed_file='data/STARR-FISH_300_library.bed')
    # %%
    starrfish_july = STARRFISH(july_experiment, atac_cpm=None, atac_counts=None, lib_size=None)
    starrfish_sept = STARRFISH(sept_experiment, atac_cpm=None, atac_counts=None, lib_size=None)
    starrfish_300_1 = STARRFISH(CRE_300_experiment1, atac_cpm=None, atac_counts=None, lib_size=None)
    starrfish_300_2 = STARRFISH(CRE_300_experiment2, atac_cpm=None, atac_counts=None, lib_size=None)
    # set library size
    lib_size = pd.DataFrame({'counts': starrfish_july.adata.var['nanopore'].values}, index=starrfish_july.adata.var_names)
    starrfish_july.lib_size = lib_size.copy()
    starrfish_sept.lib_size = lib_size.copy()
    starrfish_300_1.lib_size = lib_size.copy()
    starrfish_300_2.lib_size = lib_size.copy()
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
        'bootstrap_to_fixed_pct': 0.5,
        'bootstrap_to_fixed_sample_size': None,
        'load_stored': True,
        'n_jobs': 64,
    }
    threshold = 'neg_control_dist'
    res1 = starrfish_july.average_bootstrap_test(**average_bootstrap_test_config)
    res_q1, res_df1 = starrfish_july.average_bootstrap_test_q(res1, threshold=threshold, norm='libsize', tail='right')

    res2 = starrfish_sept.average_bootstrap_test(**average_bootstrap_test_config)
    res_q2, res_df2 = starrfish_sept.average_bootstrap_test_q(res2, threshold=threshold, norm='libsize', tail='right')
    # %%
    # plot q value
    compare = pd.DataFrame({
        'July': res_q1.loc['CellLine'],
        'Sept': res_q2.loc['CellLine']
    })
    compare