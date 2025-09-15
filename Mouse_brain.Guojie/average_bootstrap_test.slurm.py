# %%
import warnings
warnings.filterwarnings("ignore", category=SyntaxWarning, module="docrep")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import sys
import os
import re
import pandas as pd
import scanpy as sc
import scvi
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# add current path to sys.path
try:
    PWD = os.path.dirname(os.path.abspath(__file__))
except NameError:
    PWD = '/share/vault/Users/gz2294/starr-fish/Mouse_brain.Guojie'
sys.path.append(f'{PWD}/')
os.chdir(PWD)
from utils import STARRFISH

scvi.settings.seed = 0
print("Last run with scvi-tools version:", scvi.__version__)
# %% helper function to reload
def reload(starrfish):
    import importlib
    import utils
    import starr_fish_vae
    importlib.reload(utils)
    importlib.reload(starr_fish_vae)
    from utils import STARRFISH
    from starr_fish_vae import STARRFISHVI
    starrfish.__class__ = STARRFISH
    return starrfish

def drop_test(starrfish, test_method):
    if hasattr(starrfish, f'{test_method}_configs'):
        delattr(starrfish, f'{test_method}_configs')
    if hasattr(starrfish, f'{test_method}_results'):
        delattr(starrfish, f'{test_method}_results')
    return starrfish

def preprocess(adata_path):
    if type(adata_path) is str:
        adata = sc.read_h5ad(adata_path)
    elif type(adata_path) is sc.AnnData:
        adata = adata_path
    # operate fov, it is the index names
    adata.obs['fov'] = adata.obs.index.str.split('--').str[0]
    # change adata2 obs subclass_name to subclass
    adata.obs['subclass'] = adata.obs['subclass_name'].str.replace('^[0-9]+ ', '', regex=True)
    adata.obs['class'] = adata.obs['class_name'].str.replace('^[0-9]+ ', '', regex=True)
    # change '' best_subclass to its label
    adata.uns['CRE_info']['best_subclass'][adata.uns['CRE_info']['best_subclass'] == ''] = adata.uns['CRE_info']['label'][adata.uns['CRE_info']['best_subclass'] == ''].copy()
    # process enh
    chrom = []
    start = []
    end = []
    for i in adata.uns['CRE_info']['enh']:
        if i.startswith('chr'):
            chrom.append(i.split(':')[0])
            start.append(int(re.split('−|-', i.split(':')[1])[0]))
            end.append(int(re.split('−|-', i.split(':')[1])[1]))
        else:
            chrom.append(i)
            start.append('')
            end.append('')
    adata.uns['CRE_info']['Chrom'] = chrom
    adata.uns['CRE_info']['Start'] = start
    adata.uns['CRE_info']['End'] = end
    # convert start and end to str
    adata.uns['CRE_info']['Chrom'] = adata.uns['CRE_info']['Chrom'].astype(str)
    adata.uns['CRE_info']['Start'] = adata.uns['CRE_info']['Start'].astype(str)
    adata.uns['CRE_info']['End'] = adata.uns['CRE_info']['End'].astype(str)
    # rename enh
    adata.uns['CRE_info']['enh'] = adata.uns['CRE_info']['Chrom'] + ':' + adata.uns['CRE_info']['Start'].astype(str) + '-' + adata.uns['CRE_info']['End'].astype(str)
    # rename best_subclass
    adata.uns['CRE_info']['best_subclass'] = adata.uns['CRE_info']['best_subclass'].str.replace('_', ' ')
    adata.uns['CRE_info'].index = ['CRE' + str(i+1).zfill(3) for i in range(len(adata.uns['CRE_info']))]
    adata.obsm['CRE'] = adata.obsm['CRE'][adata.uns['CRE_info'].index]
    if 'T7CRE' in adata.obsm.keys():
        adata.obsm['T7CRE'] = adata.obsm['T7CRE'][adata.uns['CRE_info'].index]
    return adata
# %%
# adata3 = preprocess(f'{PWD}/Data/scdata_5_28_2025_BRBB500gn_final_CRE_T7CRE.h5ad')
# # split the adata3 into two parts based on two sections
# adata3.obs['section'] = (adata3.obsm['X_spatial'][:, 0] >= -1900).astype(int)
# adata3_sec1 = adata3[adata3.obs['section'] == 0].copy()
# adata3_sec2 = adata3[adata3.obs['section'] == 1].copy()
# # make two STARRFISH objects
# starrfish3_sec1 = STARRFISH(adata3_sec1)
# starrfish3_sec2 = STARRFISH(adata3_sec2)
starrfish3_sec1 = STARRFISH.load('results/starrfish3_sec1.pkl')
# get subclass name and subclass transform
subclass_annotation = pd.read_excel(f'Data/abc_atlas/allen_institute_nominature.xlsx')
subclass_annotation['subclass'] = subclass_annotation['subclass_id_label'].str.replace('^[0-9]+ ', '', regex=True)
subclass_annotation['subclass'] = subclass_annotation['subclass'].str.replace('/', '-', regex=True)
subclass_to_subclass_name = subclass_annotation['subclass_id_label'].groupby(subclass_annotation['subclass']).first().to_dict()
subclass_name_to_subclass = subclass_annotation['subclass'].groupby(subclass_annotation['subclass_id_label']).first().to_dict()
cre_blacklist = ['CRE061', 'CRE143', 'CRE001']
cre_whitelist = starrfish3_sec1.get_creinfo().index[~starrfish3_sec1.get_creinfo().index.isin(cre_blacklist)]
# read in the mismatching barcode CREs
mismatching_cres = pd.read_csv('Data/AAV_ONT_Barcode_Counts_vs_Mismatch_Percentage.csv', index_col=0)
cre_whitelist = cre_whitelist[~cre_whitelist.isin(mismatching_cres.index[mismatching_cres['MismatchPercent'] > 20])]
cre_blacklist = np.unique(cre_blacklist + mismatching_cres.index[mismatching_cres['MismatchPercent'] > 20].tolist()).tolist()
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
    'bootstrap_to_fixed_pct': 1,
    'bootstrap_to_fixed_sample_size': None,
    'load_stored': True,
    'n_jobs': 62,
}
infected_cells_threshold = 5
threshold = 'neg_control_mean'
starrfish3_sec1 = STARRFISH.load('results/starrfish3_sec1.bak.pkl')
res1 = starrfish3_sec1.average_bootstrap_test(**average_bootstrap_test_config)
del starrfish3_sec1
starrfish3_sec1 = STARRFISH.load('results/starrfish3_sec1.pkl')
to_filter_sec1 = (starrfish3_sec1.get_cre_expression() > 0).groupby(starrfish3_sec1.get_celltypes()).sum() < infected_cells_threshold
to_filter_sec1[cre_blacklist] = True
res_q1, res_df1, _ = starrfish3_sec1.average_bootstrap_test_q(res1, threshold=threshold, norm='T7', tail='both', to_filter=to_filter_sec1, calibrate=None)

starrfish3_sec2 = STARRFISH.load('results/starrfish3_sec2.bak.pkl')
res2 = starrfish3_sec2.average_bootstrap_test(**average_bootstrap_test_config)
del starrfish3_sec2
starrfish3_sec2 = STARRFISH.load('results/starrfish3_sec2.pkl')
to_filter_sec2 = (starrfish3_sec2.get_cre_expression() > 0).groupby(starrfish3_sec2.get_celltypes()).sum() < infected_cells_threshold
to_filter_sec2[cre_blacklist] = True
res_q2, res_df2, _ = starrfish3_sec2.average_bootstrap_test_q(res2, threshold=threshold, norm='T7', tail='both', to_filter=to_filter_sec2, calibrate=None)

starrfish3 = STARRFISH.load('results/starrfish3.bak.pkl')
res = starrfish3.average_bootstrap_test(**average_bootstrap_test_config)
del starrfish3
starrfish3 = STARRFISH.load('results/starrfish3.pkl')
to_filter = (starrfish3.get_cre_expression() > 0).groupby(starrfish3.get_celltypes()).sum() < infected_cells_threshold
to_filter[cre_blacklist] = True
res_q, res_df, _ = starrfish3.average_bootstrap_test_q(res, threshold=threshold, norm='T7', tail='both', to_filter=to_filter, calibrate=None)


# %%
# regress on the total T7 counts ~ log fold change of each CRE/Cell Type pair
# fist visualize the result
toplot = pd.DataFrame({'value': res_df.values.flatten(), 
                       'T7 total': np.tile(starrfish3.get_t7_expression().sum().values, res_df.shape[0])})
fig, ax = plt.subplots(figsize=(4, 4))
sns.scatterplot(data=toplot, x='T7 total', y='value', alpha=0.1, ax=ax)
toplot_avg = res_df.apply(np.nanmean, axis=0)
sns.scatterplot(x=starrfish3.get_t7_expression().sum().values, y=toplot_avg.values, color='red', ax=ax, alpha=0.5)
ax.set_xscale('log')
ax.set_xlabel('Total T7 counts (per CRE)')
ax.set_ylabel('Average Bootstrap Test Statistic')
# %% check cell type bias
toplot = pd.DataFrame({'value': res_df.values.flatten(), 
                       'T7 total': np.repeat(starrfish3.get_t7_expression().sum(axis=1).groupby(starrfish3.get_celltypes()).sum().values, res_df.shape[1])})
fig, ax = plt.subplots(figsize=(4, 4))
sns.scatterplot(data=toplot, x='T7 total', y='value', alpha=0.1, ax=ax)
toplot_avg = res_df.apply(np.nanmean, axis=1)
sns.scatterplot(x=starrfish3.get_t7_expression().sum(axis=1).groupby(starrfish3.get_celltypes()).sum().values, y=toplot_avg.values, color='red', ax=ax, alpha=0.5)
ax.set_xscale('log')
ax.set_xlabel('Total T7 counts (per Cell Type)')
ax.set_ylabel('Average Bootstrap Test Statistic')



# %%
# heatmap after correction
from plots import plot_grouped_clustermap
cre_info = starrfish3.get_creinfo().copy()
cre_info['best_subclass'] = 'CRE'
cre_info.loc[starrfish3.get_negative_control_cres(), 'best_subclass'] = 'Negative Control'

_, final_order = plot_grouped_clustermap(res_df.loc[pd.isna(res_df).any(axis=1), pd.isna(res_df).any(axis=0)], cre_info, 'All', figsize=(15, 8))
# %% dot plot after correction
threshold = 'neg_control_mean'
to_filter = (starrfish3.get_cre_expression() > 0).groupby(starrfish3.get_celltypes()).sum() < 0
to_filter[cre_blacklist] = True
res_q, res_q_right, res_q_left, res_df, res_df_fdc = starrfish3.average_bootstrap_test_q(res, threshold=threshold, norm='T7', tail='all', 
                                                                                         to_filter=to_filter, calibrate='self-CRE')
# %%
to_filter_sec1 = (starrfish3_sec1.get_cre_expression() > 0).groupby(starrfish3_sec1.get_celltypes()).sum() < 0
to_filter_sec1[cre_blacklist] = True
res_q1, res_q1_right, res_q1_left, res_df1, res_df1_fdc = starrfish3.average_bootstrap_test_q(res1, threshold=threshold, norm='T7', tail='all',
                                                                                              to_filter=to_filter_sec1, calibrate='self-CRE')
# %%
to_filter_sec2 = (starrfish3_sec2.get_cre_expression() > 0).groupby(starrfish3_sec2.get_celltypes()).sum() < 0
to_filter_sec2[cre_blacklist] = True
res_q2, res_q2_right, res_q2_left, res_df2, res_df2_fdc = starrfish3.average_bootstrap_test_q(res2, threshold=threshold, norm='T7', tail='all',
                                                                                              to_filter=to_filter_sec2, calibrate='self-CRE')

# %% check overlap of significant CREs between sections
res_q2_overlap = res_q2_right[(~res_q2_right.isna()) & (~res_q1_right.isna())].copy()
res_q1_overlap = res_q1_right[(~res_q2_right.isna()) & (~res_q1_right.isna())].copy()
overlap_df = pd.DataFrame(index=res_q2_overlap.index.intersection(res_q1_overlap.index), columns=['sec1', 'sec2', 'overlap'])
overlap_df['sec1'] = (res_q1_overlap.loc[overlap_df.index] <= 0.05).sum(axis=1)
overlap_df['sec2'] = (res_q2_overlap.loc[overlap_df.index] <= 0.05).sum(axis=1)
overlap_df['overlap'] = ((res_q1_overlap.loc[overlap_df.index] <= 0.05) & (res_q2_overlap.loc[overlap_df.index] <= 0.05)).sum(axis=1)
overlap_df['percentage'] = overlap_df['overlap'] / np.minimum(overlap_df['sec1'], overlap_df['sec2'])
overlap_df = overlap_df.sort_values('percentage', ascending=False)
overlap_df['celltype_n'] = starrfish3.get_celltypes().value_counts().loc[overlap_df.index]
fig, ax = plt.subplots(figsize=(4, 4))
sns.scatterplot(data=overlap_df, x='celltype_n', y='percentage', ax=ax)
ax.set_xscale('log')


# %% get some numbers of reproducibility in each CRE
res_q1_overlap = res_q1_overlap.loc[overlap_df.index]
res_q2_overlap = res_q2_overlap.loc[overlap_df.index]
overlap_cre_df = pd.DataFrame(index=res_q1_overlap.columns, columns=['no_na', 'sec1', 'sec2', 'overlap', 'percentage'])
# for each CRE, check if the test result is exactly the same
for cre in res_q1_overlap.columns:
    overlap_cre_df.loc[cre, 'no_na'] = sum(~pd.isna(res_q1_overlap[cre]))
    overlap_cre_df.loc[cre, 'sec1'] = sum(res_q1_overlap[cre] <= 0.05)
    overlap_cre_df.loc[cre, 'sec2'] = sum(res_q2_overlap[cre] <= 0.05)
    overlap_cre_df.loc[cre, 'overlap'] = sum((res_q1_overlap[cre] <= 0.05) & (res_q2_overlap[cre] <= 0.05))
    if overlap_cre_df.loc[cre, 'sec1'] == 0 and overlap_cre_df.loc[cre, 'sec2'] == 0:
        if overlap_cre_df.loc[cre, 'no_na'] > 0:
            # all non-significant
            overlap_cre_df.loc[cre, 'percentage'] = 1
        else:
            # all NA
            overlap_cre_df.loc[cre, 'percentage'] = -1
    else:
        overlap_cre_df.loc[cre, 'percentage'] = overlap_cre_df.loc[cre, 'overlap'] / np.maximum(overlap_cre_df.loc[cre, 'sec1'], overlap_cre_df.loc[cre, 'sec2'])
sum(overlap_cre_df['percentage'] > 0)



# %%
from plots import celltype_pval_dotplot
cell_types_to_use = starrfish3.get_celltypes().value_counts().index[starrfish3.get_celltypes().value_counts()>=1000]
cres_to_use = res_q.columns[np.nanmin(res_q.loc[cell_types_to_use], axis=0) < 0.05].union(starrfish3.get_negative_control_cres())
cres_to_use = cres_to_use[~cres_to_use.isin(cre_blacklist)]
fig, final_order = celltype_pval_dotplot(res_q_right, res_df, cres_to_use, cell_types_to_use,
                                         positive_control_info=cre_info, significant_cutoff=0.05, z_norm=False,
                                         figsize=(15, 45))
fig
# %%
fig, final_order = celltype_pval_dotplot(res_q1_right, res_df1, pd.Index(final_order), cell_types_to_use, reorder_cres=False,
                                         positive_control_info=cre_info, significant_cutoff=0.05, z_norm=False,
                                         figsize=(15, 45))
fig
# %%
fig, final_order = celltype_pval_dotplot(res_q2_right, res_df2, pd.Index(final_order), cell_types_to_use, reorder_cres=False,
                                         positive_control_info=cre_info, significant_cutoff=0.05, z_norm=False,
                                         figsize=(15, 45))
fig
# %% reproducibility
cre_corr, celltype_corr = starrfish3.corr_starrfish(res_df1_fdc, res_df2_fdc)
cre_corr['libsize'] = starrfish3.lib_size['counts'].loc[cre_corr.index]
celltype_corr['celltype_sec1'] = starrfish3_sec1.get_celltypes().value_counts().loc[celltype_corr.index].values
celltype_corr['celltype_sec2'] = starrfish3_sec2.get_celltypes().value_counts().loc[celltype_corr.index].values
celltype_corr['celltype_n'] = np.minimum(celltype_corr['celltype_sec1'], celltype_corr['celltype_sec2'])
# %% plot
n_cre_threshold = 5
n_celltype_threshold = 5
fig, ax = plt.subplots(figsize=(4, 4))
sns.scatterplot(data=celltype_corr[(celltype_corr['pearson_p'] > 0.05) & (celltype_corr['effect_n'] >= n_cre_threshold)], x='celltype_n', y='pearson', color='blue', ax=ax)
sns.scatterplot(data=celltype_corr[(celltype_corr['pearson_p'] <= 0.05) & (celltype_corr['effect_n'] >= n_cre_threshold)], x='celltype_n', y='pearson', color='red', ax=ax)
ax.set_xscale('log')
fig, ax = plt.subplots(figsize=(4, 4))
sns.scatterplot(data=cre_corr[(cre_corr['pearson_p'] > 0.05) & (cre_corr['effect_n'] >= n_celltype_threshold)], x='libsize', y='pearson', color='blue', ax=ax)
sns.scatterplot(data=cre_corr[(cre_corr['pearson_p'] <= 0.05) & (cre_corr['effect_n'] >= n_celltype_threshold)], x='libsize', y='pearson', color='red', ax=ax)
reproducible_celltypes = celltype_corr.index[(celltype_corr['pearson_p'] <= 0.05) & (celltype_corr['effect_n'] >= n_cre_threshold)]
ax.set_xscale('log')




# %%
# check ATAC peaks
res_df_atac = res_df.copy()
atac_peaks = pd.read_csv('Data/cre_atac_peaks.csv', index_col=0)
atac_peaks = atac_peaks.loc[res_df_atac.index.intersection(atac_peaks.index), res_df_atac.columns.intersection(atac_peaks.columns)] >= 0.5
res_df_atac = res_df_atac.loc[atac_peaks.index, atac_peaks.columns]
atac_peaks[res_df_atac.isna()] = False
res_df_atac[res_q_right > 0.05] = np.nan
overlap = res_df_atac.loc[atac_peaks.index, atac_peaks.columns][atac_peaks].notna().sum().sum()
precision = overlap / atac_peaks.sum().sum()
recall = overlap / res_df_atac.notna().sum().sum()
precision, recall
# %%
res_df_atac = res_df.copy()
atac_peaks = pd.read_csv('Data/cre_chromatin_state_a.csv', index_col=0)
atac_peaks = atac_peaks.loc[res_df_atac.index.intersection(atac_peaks.index), res_df_atac.columns.intersection(atac_peaks.columns)] >= 0.5
res_df_atac = res_df_atac.loc[atac_peaks.index, atac_peaks.columns]
atac_peaks[res_df_atac.isna()] = False
res_df_atac[res_q_right > 0.05] = np.nan
overlap = res_df_atac.loc[atac_peaks.index, atac_peaks.columns][atac_peaks].notna().sum().sum()
precision = overlap / atac_peaks.sum().sum()
recall = overlap / res_df_atac.notna().sum().sum()
precision, recall
# %% correlation to ATAC
cre_corr, celltype_corr = starrfish3.corr_atac_cpm(None, None, res_df)
cre_corr['libsize'] = starrfish3.lib_size['counts'].loc[cre_corr.index]
celltype_corr['celltype_n'] = starrfish3.get_celltypes().value_counts().loc[celltype_corr.index].values

# %% plot
n_cre_threshold = 0
n_celltype_threshold = 0
fig, ax = plt.subplots(figsize=(4, 4))
sns.scatterplot(data=celltype_corr[(celltype_corr['spearman_p'] > 0.05) & (celltype_corr['effect_n'] >= n_cre_threshold)], x='celltype_n', y='spearman', color='blue', ax=ax)
sns.scatterplot(data=celltype_corr[(celltype_corr['spearman_p'] <= 0.05) & (celltype_corr['effect_n'] >= n_cre_threshold)], x='celltype_n', y='spearman', color='red', ax=ax)
ax.set_xscale('log')
fig, ax = plt.subplots(figsize=(4, 4))
sns.scatterplot(data=cre_corr[(cre_corr['spearman_p'] > 0.05) & (cre_corr['effect_n'] >= n_celltype_threshold)], x='libsize', y='spearman', color='blue', ax=ax)
sns.scatterplot(data=cre_corr[(cre_corr['spearman_p'] <= 0.05) & (cre_corr['effect_n'] >= n_celltype_threshold)], x='libsize', y='spearman', color='red', ax=ax)
reproducible_celltypes = celltype_corr.index[(celltype_corr['spearman_p'] <= 0.05) & (celltype_corr['effect_n'] >= n_cre_threshold)]


# %%
