# %%
from turtle import st
import warnings
warnings.filterwarnings("ignore", category=SyntaxWarning, module="docrep")
import scvi
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
import scanpy as sc
from scipy.stats import pearsonr, spearmanr, ttest_ind
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, dendrogram, leaves_list, optimal_leaf_ordering
from scipy.spatial.distance import squareform
from adjustText import adjust_text
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)
scvi.settings.seed = 0
print("Last run with scvi-tools version:", scvi.__version__)
# add current path to sys.path
import sys
import os
try:
    PWD = os.path.dirname(os.path.abspath(__file__))
except NameError:
    PWD = '/share/vault/Users/gz2294/starr-fish/Mouse_brain.Guojie'
sys.path.append(f'{PWD}/')
os.chdir(PWD)
from utils import STARRFISH
import re
import statsmodels.api as sm
from statsmodels.stats import multitest
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize, LinearSegmentedColormap, to_rgb
from matplotlib.colorbar import ColorbarBase
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.cm import ScalarMappable
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
starrfish3_sec2 = STARRFISH.load('results/starrfish3_sec2.pkl')
starrfish3 = STARRFISH.load('results/starrfish3.pkl')
# %% define cell types to use for filtered data
negative_control_cres = starrfish3_sec1.get_negative_control_cres()
cell_types_counts1 = starrfish3_sec1.get_celltypes().value_counts()
cell_types_counts2 = starrfish3_sec2.get_celltypes().value_counts()
cell_types_to_use_1 = cell_types_counts1[cell_types_counts1 > 500].index
cell_types_to_use_2 = cell_types_counts2[cell_types_counts2 > 500].index
cell_types_to_use = cell_types_to_use_1.intersection(cell_types_to_use_2)
# check the negative control counts for those cell types
negative_control_counts1 = starrfish3_sec1.get_cre_expression()[negative_control_cres].groupby(starrfish3_sec1.get_celltypes()).sum()
negative_control_counts2 = starrfish3_sec2.get_cre_expression()[negative_control_cres].groupby(starrfish3_sec2.get_celltypes()).sum()
negative_control_sum_counts1 = starrfish3_sec1.get_cre_expression()[starrfish3_sec1.get_negative_control_cres()].sum(axis=1).groupby(starrfish3_sec1.get_celltypes()).sum()
negative_control_sum_counts2 = starrfish3_sec2.get_cre_expression()[starrfish3_sec2.get_negative_control_cres()].sum(axis=1).groupby(starrfish3_sec2.get_celltypes()).sum()
common_cell_types_sum_20_nc = negative_control_sum_counts1[negative_control_sum_counts1 > 20].index.intersection(negative_control_sum_counts2[negative_control_sum_counts2 > 20].index)
# define the cell types by the negative control counts > 50
cell_types_to_use_nc_1 = negative_control_sum_counts1[negative_control_sum_counts1 > 10].index
cell_types_to_use_nc_2 = negative_control_sum_counts2[negative_control_sum_counts2 > 10].index
cell_types_to_use_nc = cell_types_to_use_nc_1.intersection(cell_types_to_use_nc_2)
target_cres = starrfish3_sec1.get_creinfo().index[starrfish3_sec1.get_creinfo()['best_subclass'].isin(cell_types_to_use_nc_2)]
len(cell_types_to_use), len(cell_types_to_use_nc), len(cell_types_to_use_nc_2), len(target_cres)




# %%
# correlation of cell type counts
cell_type_counts = pd.DataFrame(index=cell_types_counts2.index.intersection(cell_types_counts1.index), columns=['Exp2', 'Exp3'])
cell_type_counts['Sec1'] = cell_types_counts1[cell_type_counts.index]
cell_type_counts['Sec2'] = cell_types_counts2[cell_type_counts.index]
fig, ax = plt.subplots(figsize=(5, 4))
sns.scatterplot(x=cell_type_counts['Sec1'], y=cell_type_counts['Sec2'], ax=ax, alpha=0.5)
ax.set_xlabel('Cell type counts in Sec1')
ax.set_ylabel('Cell type counts in Sec2')
# calculate the correlation
to_test = cell_type_counts.index[cell_type_counts['Sec1'] > 0].intersection(cell_type_counts.index[cell_type_counts['Sec2'] > 0])
corr, p_value = pearsonr(np.log(cell_type_counts['Sec1'].loc[to_test]), np.log(cell_type_counts['Sec2'].loc[to_test]))
# plot text
ax.text(0.05, 0.95, f'Pearson r: {corr:.2f}\np-value: {p_value:.2e}', transform=ax.transAxes, fontsize=12, verticalalignment='top')
ax.set_xscale('log')
ax.set_yscale('log')



# %%
# first check the correlation of transcripts and T7 per cell type.
fig, ax = plt.subplots(ncols=2, figsize=(12, 6))
cre_counts_sec1 = starrfish3_sec1.get_cre_expression().sum(axis=1)
cre_counts_sec2 = starrfish3_sec2.get_cre_expression().sum(axis=1)
t7_counts_sec1 = starrfish3_sec1.get_t7_expression().sum(axis=1)
t7_counts_sec2 = starrfish3_sec2.get_t7_expression().sum(axis=1)
cre_celltype_sec1 = cre_counts_sec1.groupby(starrfish3_sec1.get_tag('obs:subclass')).mean()
cre_celltype_sec2 = cre_counts_sec2.groupby(starrfish3_sec2.get_tag('obs:subclass')).mean()
t7_celltype_sec1 = t7_counts_sec1.groupby(starrfish3_sec1.get_tag('obs:subclass')).mean()
t7_celltype_sec2 = t7_counts_sec2.groupby(starrfish3_sec2.get_tag('obs:subclass')).mean()
common_celltypes = t7_celltype_sec1.index.intersection(t7_celltype_sec2.index)
sns.scatterplot(x=t7_celltype_sec1.loc[common_celltypes], y=t7_celltype_sec2.loc[common_celltypes], ax=ax[0], alpha=0.5)
to_test = t7_celltype_sec1.index[t7_celltype_sec1 > 0].intersection(t7_celltype_sec2.index[t7_celltype_sec2 > 0])
corr, p_value = pearsonr(np.log(t7_celltype_sec1.loc[to_test]), np.log(t7_celltype_sec2.loc[to_test]))
ax[0].text(0.05, 0.95, f'Pearson r: {corr:.2f}\np-value: {p_value:.2e}', transform=ax[0].transAxes, fontsize=12, verticalalignment='top')
ax[0].set_xscale('log')
ax[0].set_yscale('log')
ax[0].set_xlabel('Average T7 counts per cell type in Sec1')
ax[0].set_ylabel('Average T7 counts per cell type in Sec2')
sns.scatterplot(x=cre_celltype_sec1.loc[common_celltypes], y=cre_celltype_sec2.loc[common_celltypes], ax=ax[1], alpha=0.5)
to_test = cre_celltype_sec1.index[cre_celltype_sec1 > 0].intersection(cre_celltype_sec2.index[cre_celltype_sec2 > 0])
corr, p_value = pearsonr(np.log(cre_celltype_sec1.loc[to_test]), np.log(cre_celltype_sec2.loc[to_test]))
ax[1].text(0.05, 0.95, f'Pearson r: {corr:.2f}\np-value: {p_value:.2e}', transform=ax[1].transAxes, fontsize=12, verticalalignment='top')
ax[1].set_xlabel('Average CRE counts per cell type in Sec1')
ax[1].set_ylabel('Average CRE counts per cell type in Sec2')
ax[1].set_xscale('log')
ax[1].set_yscale('log')




# %% test plot the negative control CREs/T7 counts per cell type
fig, ax = plt.subplots(ncols=3, figsize=(15, 4))
for p in ['CRE', 'T7', 'CRE/T7']:
    if p == 'CRE':
        counts_sec1 = starrfish3_sec1.get_cre_expression()[negative_control_cres].sum(axis=1).groupby(starrfish3_sec1.get_tag('obs:subclass')).mean()
        counts_sec2 = starrfish3_sec2.get_cre_expression()[negative_control_cres].sum(axis=1).groupby(starrfish3_sec2.get_tag('obs:subclass')).mean()
        ax_work = ax[0]
    elif p == 'T7':
        counts_sec1 = starrfish3_sec1.get_t7_expression()[negative_control_cres].sum(axis=1).groupby(starrfish3_sec1.get_tag('obs:subclass')).mean()
        counts_sec2 = starrfish3_sec2.get_t7_expression()[negative_control_cres].sum(axis=1).groupby(starrfish3_sec2.get_tag('obs:subclass')).mean()
        ax_work = ax[1]
    elif p == 'CRE/T7':
        counts_sec1 = starrfish3_sec1.get_cre_expression()[negative_control_cres].sum(axis=1).groupby(starrfish3_sec1.get_tag('obs:subclass')).mean() / \
                        starrfish3_sec1.get_t7_expression()[negative_control_cres].sum(axis=1).groupby(starrfish3_sec1.get_tag('obs:subclass')).mean()
        counts_sec2 = starrfish3_sec2.get_cre_expression()[negative_control_cres].sum(axis=1).groupby(starrfish3_sec2.get_tag('obs:subclass')).mean() / \
                        starrfish3_sec2.get_t7_expression()[negative_control_cres].sum(axis=1).groupby(starrfish3_sec2.get_tag('obs:subclass')).mean()
        ax_work = ax[2]
    common_celltypes = counts_sec1.index.intersection(counts_sec2.index)
    to_test = counts_sec1.index[(counts_sec1 > 0) & np.isfinite(counts_sec1)].intersection(counts_sec2.index[(counts_sec2 > 0) & np.isfinite(counts_sec2)])
    corr, p_value = pearsonr(np.log(counts_sec1.loc[to_test]), np.log(counts_sec2.loc[to_test]))
    sns.scatterplot(x=counts_sec1.loc[common_celltypes], y=counts_sec2.loc[common_celltypes], ax=ax_work, alpha=0.5)
    ax_work.text(0.05, 0.95, f'Pearson r: {corr:.2f}\np-value: {p_value:.2e}', transform=ax_work.transAxes, fontsize=12, verticalalignment='top')
    ax_work.set_xscale('log')
    ax_work.set_yscale('log')
    ax_work.set_xlabel(f'Average {p} counts per cell type in Sec1')
    ax_work.set_ylabel(f'Average {p} counts per cell type in Sec2')




# %% run the average bootstrap test for T7
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
    'n_jobs': 128,
}
threshold = 'neg_control'
res1 = starrfish3_sec1.average_bootstrap_test(**average_bootstrap_test_config)
res_q1, res_df1 = starrfish3_sec1.average_bootstrap_test_q(res1, threshold=threshold, norm='T7', tail='right')
res2 = starrfish3_sec2.average_bootstrap_test(**average_bootstrap_test_config)
res_q2, res_df2 = starrfish3_sec2.average_bootstrap_test_q(res2, threshold=threshold, norm='T7', tail='right')
res = starrfish3.average_bootstrap_test(**average_bootstrap_test_config)
res_q, res_df = starrfish3.average_bootstrap_test_q(res, threshold=threshold, norm='T7', tail='right')



# %% run the fold change test for T7
fold_change_test_config = {"cell_types_to_use": None,
                           "normalize_by_cell_rna": False,
                           "normalize_by_cell_volume": False,
                           "normalize_by_cell_t7": False, # normalize by T7
                           'filter_by_cell_t7': None,
                           "normalize_by_celltype_rna": False,
                           "normalize_by_celltype_volume": False,
                           "normalize_by_celltype_t7": True, # normalize by T7
                           "normalize_by_negative_control": True, # normalize by negative control
                           'normalize_by_total_cre': False,
                           "normalize_by_infected_cell": False,
                           "normalize_by_libsize": False,
                           "log_transform": False,
                           "filter_zero_counts": False,
                           "bootstrap_number": 10000,
                           'load_stored': True,}
res1 = starrfish3_sec1.fold_change_test(**fold_change_test_config)
res2 = starrfish3_sec2.fold_change_test(**fold_change_test_config)




# %% save the results
starrfish3_sec1.save('results/starrfish3_sec1.pkl')
starrfish3_sec2.save('results/starrfish3_sec2.pkl')
# %%
# check if res_q1 and res_q2 are the same
cell_type = 'Endo NN'
norm = 'T7'
fig, ax = plt.subplots(ncols=3, figsize=(15, 4))
def plot_volcano(res_q, res_df, res_array, cell_type, starrfish3_obj, ax):
    # log then average
    res_array = np.log(res_array)
    # assign inf to NaN
    res_array[np.isinf(res_array)] = np.nan
    neg_control_array = res_array[:, :, res_df.columns.isin(starrfish3_obj.get_negative_control_cres())]
    neg_control_array = np.nanmean(neg_control_array, axis=2)
    # cap to min 1e-5
    res_q = res_q.clip(lower=1e-5)
    cre_ct1 = starrfish3_obj.get_cre_expression().loc[starrfish3_obj.get_celltypes() == cell_type].copy()
    t7_ct1 = starrfish3_obj.get_t7_expression().loc[starrfish3_obj.get_celltypes() == cell_type].copy()
    # filter out cells with low t7 counts
    total_neg_control_cre1 = cre_ct1[starrfish3_obj.get_negative_control_cres()].sum(axis=0).sum()
    total_neg_control_t71 = t7_ct1[starrfish3_obj.get_negative_control_cres()].sum(axis=0).sum()
    # calculate the fold change
    # calculate the fold change
    if threshold == '0':
        fdc_u = 0
        fdc_l = fdc_u
    elif threshold == 'total':
        if norm == 'T7':
            fdc_u = np.log(cre_ct1.sum(axis=0).sum() / t7_ct1.sum(axis=0).sum())
        elif norm == 'libsize':
            fdc_u = np.log(cre_ct1.sum() / starrfish3_obj.lib_size['counts'].sum())
        fdc_l = fdc_u
    elif threshold == 'total_dist':
        # use the distribution of total CREs to set the threshold
        fdc = np.nanmean(res_array[:, res_df.index == cell_type])
        fdc_std = np.nanstd(res_array[:, res_df.index == cell_type])
        fdc_u = fdc + 2 * fdc_std
        fdc_l = fdc - 2 * fdc_std
    elif threshold == 'neg_control':
        if norm == 'T7':
            fdc_u = np.log(total_neg_control_cre1 / total_neg_control_t71)
        elif norm == 'libsize':
            fdc_u = np.log(total_neg_control_cre1 / starrfish3_obj.lib_size['counts'][starrfish3_obj.get_negative_control_cres()].sum())
        fdc_l = fdc_u
    elif threshold == 'neg_control_dist':
        # use the distribution of negative control CREs to set the threshold
        fdc = np.nanmean(neg_control_array[:, res_df.index == cell_type])
        fdc_std = np.nanstd(neg_control_array[:, res_df.index == cell_type])
        fdc_u = fdc + 2 * fdc_std
        fdc_l = fdc - 2 * fdc_std
    fdc = (fdc_u + fdc_l) / 2
    # fdc = 0
    sns.scatterplot(x=res_df.loc[cell_type].values.flatten(), y=-np.log10(res_q.loc[cell_type].values.flatten().astype(float)), ax=ax, alpha=0.5)
    ax.axhline(y=-np.log10(0.05), color='red', linestyle='--')
    ax.axvline(x=fdc, color='black', linestyle='--')
    ax.axvline(x=fdc_u, color='grey', linestyle='--')
    ax.axvline(x=fdc_l, color='grey', linestyle='--')
    ax.set_xlabel('Fold change (log)')
    ax.set_ylabel('-log10(q-value)')
    # mark the negative control CREs
    neg_control_cres = starrfish3_obj.get_negative_control_cres()
    positive_control_cres = starrfish3_obj.get_positive_control_cres(cell_type, use='atac-peak')
    sns.scatterplot(x=res_df.loc[cell_type, neg_control_cres].values.flatten(), y=-np.log10(res_q.loc[cell_type, neg_control_cres].values.flatten().astype(float)), ax=ax, color='orange', label='Negative control CREs')
    sns.scatterplot(x=res_df.loc[cell_type, positive_control_cres].values.flatten(), y=-np.log10(res_q.loc[cell_type, positive_control_cres].values.flatten().astype(float)), ax=ax, color='red', label='Positive control CREs')
    # plot positive controls
    texts = []
    for cre in positive_control_cres:
        texts.append(ax.text(res_df.loc[cell_type, cre], -np.log10(res_q.loc[cell_type, cre]), cre, fontsize=8, ha='right', va='bottom', color='red'))
    adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle='->', color='black', lw=0.5))
    ax.legend()
plot_volcano(res_q1, res_df1, res1['celltype_activity_array'], cell_type, starrfish3_sec1, ax[0])
plot_volcano(res_q2, res_df2, res2['celltype_activity_array'], cell_type, starrfish3_sec2, ax[1])
plot_volcano(res_q, res_df, res['celltype_activity_array'], cell_type, starrfish3, ax[2])



# %% plot q-value for all cell types
common_celltypes = res_q1.index.intersection(res_q2.index)
res_compare = pd.DataFrame(index=common_celltypes, columns=['Sec1', 'Sec2', 'All', 
                                                            'Common', 'Percentage',
                                                            'Common_sec1', 'Percentage_sec1',
                                                            'Common_sec2', 'Percentage_sec2'])
q_cutoff = 0.05
for cell_type in common_celltypes:
    res_compare.loc[cell_type, 'Sec1'] = (res_q1.loc[cell_type] <= q_cutoff).sum()
    res_compare.loc[cell_type, 'Sec2'] = (res_q2.loc[cell_type] <= q_cutoff).sum()
    res_compare.loc[cell_type, 'All'] = (res_q.loc[cell_type] <= q_cutoff).sum()
    res_compare.loc[cell_type, 'Common'] = res_q1.loc[cell_type].index[res_q1.loc[cell_type] <= q_cutoff].intersection(res_q2.loc[cell_type].index[res_q2.loc[cell_type] <= q_cutoff]).shape[0]
    res_compare.loc[cell_type, 'Percentage'] = res_compare.loc[cell_type, 'Common'] / np.minimum(res_compare.loc[cell_type, 'Sec1'], res_compare.loc[cell_type, 'Sec2'])
    res_compare.loc[cell_type, 'Common_sec1'] = res_q1.loc[cell_type].index[res_q1.loc[cell_type] <= q_cutoff].intersection(res_q.loc[cell_type].index[res_q.loc[cell_type] <= q_cutoff]).shape[0]
    res_compare.loc[cell_type, 'Percentage_sec1'] = res_compare.loc[cell_type, 'Common_sec1'] / np.minimum(res_compare.loc[cell_type, 'Sec1'], res_compare.loc[cell_type, 'All'])
    res_compare.loc[cell_type, 'Common_sec2'] = res_q2.loc[cell_type].index[res_q2.loc[cell_type] <= q_cutoff].intersection(res_q.loc[cell_type].index[res_q.loc[cell_type] <= q_cutoff]).shape[0]
    res_compare.loc[cell_type, 'Percentage_sec2'] = res_compare.loc[cell_type, 'Common_sec2'] / np.minimum(res_compare.loc[cell_type, 'Sec2'], res_compare.loc[cell_type, 'All'])
# %% plot the percentage vs number of cells
res_compare['Cell_counts1'] = starrfish3_sec1.get_celltypes().value_counts().loc[common_celltypes].values
res_compare['Cell_counts2'] = starrfish3_sec2.get_celltypes().value_counts().loc[common_celltypes].values
res_compare['Cell_counts'] = starrfish3.get_celltypes().value_counts().loc[common_celltypes].values
fig, ax = plt.subplots(ncols=2, figsize=(10, 4))
# Create separate plots for NaN and non-NaN values
mask_valid = ~res_compare['Percentage_sec1'].isna()
mask_nan = res_compare['Percentage_sec1'].isna()
# Plot valid values with coolwarm palette
if mask_valid.any():
    sns.scatterplot(x=res_compare.loc[mask_valid, 'Cell_counts1'], 
                         y=res_compare.loc[mask_valid, 'Cell_counts'],
                         hue=res_compare.loc[mask_valid, 'Percentage_sec1'], 
                         palette='coolwarm', ax=ax[0], alpha=0.5)
# Plot NaN values in grey
if mask_nan.any():
    ax[0].scatter(res_compare.loc[mask_nan, 'Cell_counts1'], 
               res_compare.loc[mask_nan, 'Cell_counts'], edgecolors='none', s=20,  # Adjust size as needed
               color='grey', alpha=0.5, label='NA')
ax[0].set_xscale('log')
ax[0].set_yscale('log')
ax[0].set_xlabel('Cell counts in Sec1')
ax[0].set_ylabel('Cell counts in All')

mask_valid = ~res_compare['Percentage_sec2'].isna()
mask_nan = res_compare['Percentage_sec2'].isna()
# Plot valid values with coolwarm palette
if mask_valid.any():
    sns.scatterplot(x=res_compare.loc[mask_valid, 'Cell_counts2'], 
                         y=res_compare.loc[mask_valid, 'Cell_counts'],
                         hue=res_compare.loc[mask_valid, 'Percentage_sec2'], 
                         palette='coolwarm', ax=ax[1], alpha=0.5)
# Plot NaN values in grey
if mask_nan.any():
    ax[1].scatter(res_compare.loc[mask_nan, 'Cell_counts2'], 
               res_compare.loc[mask_nan, 'Cell_counts'], edgecolors='none', s=20,  # Adjust size as needed
               color='grey', alpha=0.5, label='NA')
ax[1].set_xscale('log')
ax[1].set_yscale('log')
ax[1].set_xlabel('Cell counts in Sec2')
ax[1].set_ylabel('Cell counts in All')
# %%
# %% plot the percentage vs number of cells
res_compare['Cell_counts1'] = starrfish3_sec1.get_celltypes().value_counts().loc[common_celltypes].values
res_compare['Cell_counts2'] = starrfish3_sec2.get_celltypes().value_counts().loc[common_celltypes].values
res_compare['Cell_counts'] = starrfish3.get_celltypes().value_counts().loc[common_celltypes].values
fig, ax = plt.subplots(figsize=(10, 4))
# Create separate plots for NaN and non-NaN values
mask_valid = ~res_compare['Percentage_sec1'].isna()
mask_nan = res_compare['Percentage_sec1'].isna()
# Plot valid values with coolwarm palette
if mask_valid.any():
    sns.scatterplot(x=res_compare.loc[mask_valid, 'Cell_counts1'], 
                         y=res_compare.loc[mask_valid, 'Cell_counts'],
                         hue=res_compare.loc[mask_valid, 'Percentage_sec1'], 
                         palette='coolwarm', ax=ax[0], alpha=0.5)
# Plot NaN values in grey
if mask_nan.any():
    ax.scatter(res_compare.loc[mask_nan, 'Cell_counts1'], 
               res_compare.loc[mask_nan, 'Cell_counts'], edgecolors='none', s=20,  # Adjust size as needed
               color='grey', alpha=0.5, label='NA')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('Cell counts in Sec1')
ax.set_ylabel('Cell counts in All')