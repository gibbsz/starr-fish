# %%
import warnings
warnings.filterwarnings("ignore", category=SyntaxWarning, module="docrep")
from scipy import stats
import scvi
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import TwoSlopeNorm
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
starrfish2 = STARRFISH.load('results/starrfish2.pkl')
starrfish3 = STARRFISH.load('results/starrfish3.pkl')
# get subclass name and subclass transform
subclass_annotation = pd.read_excel(f'Data/abc_atlas/allen_institute_nominature.xlsx')
subclass_annotation['subclass'] = subclass_annotation['subclass_id_label'].str.replace('^[0-9]+ ', '', regex=True)
subclass_annotation['subclass'] = subclass_annotation['subclass'].str.replace('/', '-', regex=True)
subclass_to_subclass_name = subclass_annotation['subclass_id_label'].groupby(subclass_annotation['subclass']).first().to_dict()
subclass_name_to_subclass = subclass_annotation['subclass'].groupby(subclass_annotation['subclass_id_label']).first().to_dict()
cre_blacklist = ['CRE061', 'CRE143', 'CRE001']
cre_whitelist = starrfish3_sec1.get_creinfo().index[~starrfish3_sec1.get_creinfo().index.isin(cre_blacklist)]
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




# %% T7 correlation with AAV libaray size
t7_counts = starrfish3.get_t7_expression().sum(axis=0)
t7_counts = t7_counts.loc[starrfish3.lib_size.index]  # align with library size
fig, ax = plt.subplots(ncols = 2, figsize=(12, 6))
sns.scatterplot(x=starrfish3.lib_size['counts'], y=t7_counts, ax=ax[0], alpha=0.5)
ax[0].set_xlabel('AAV library size')
ax[0].set_ylabel('Total T7 counts in all cells')
# log scale
t7_counts_log = np.log1p(t7_counts)
sns.scatterplot(x=np.log(starrfish3.lib_size['counts']), y=t7_counts_log, ax=ax[1], alpha=0.5)
ax[1].set_xlabel('Log(AAV library size)')
ax[1].set_ylabel('Log(T7 counts)')
# draw correlation line
from scipy.stats import linregress
slope, intercept, r_value, p_value, std_err = linregress(starrfish3.lib_size['counts'], t7_counts)
x = np.linspace(starrfish3.lib_size['counts'].min(), starrfish3.lib_size['counts'].max(), 100)
y = slope * x + intercept
ax[0].plot(x, y, color='red', label=f'Correlation: {r_value:.2f}, p-value: {p_value:.2e}')
ax[0].legend()
# draw correlation line for log scale
slope_log, intercept_log, r_value_log, p_value_log, std_err_log = linregress(np.log(starrfish3.lib_size['counts']), t7_counts_log)
x_log = np.linspace(np.log(starrfish3.lib_size['counts']).min(), np.log(starrfish3.lib_size['counts']).max(), 100)
y_log = slope_log * x_log + intercept_log
ax[1].plot(x_log, y_log, color='red', label=f'Correlation: {r_value_log:.2f}, p-value: {p_value_log:.2e}')
ax[1].legend()  
# mark negative control cres
sns.scatterplot(x=starrfish3.lib_size['counts'].loc[negative_control_cres], y=t7_counts.loc[negative_control_cres], ax=ax[0], alpha=0.5, color='orange')
sns.scatterplot(x=np.log(starrfish3.lib_size['counts'].loc[negative_control_cres]), y=np.log(t7_counts.loc[negative_control_cres]), ax=ax[1], alpha=0.5, color='orange')



# %%
# correlation of cell type counts
cell_type_counts = pd.DataFrame(index=cell_types_counts2.index.intersection(cell_types_counts1.index).intersection(starrfish2.get_celltypes()), columns=['Sec1', 'Sec2', 'Exp2', 'Exp3'])
cell_type_counts['Exp2'] = starrfish2.get_celltypes().value_counts().loc[cell_type_counts.index]
cell_type_counts['Exp3'] = starrfish3.get_celltypes().value_counts().loc[cell_type_counts.index]
cell_type_counts['Sec1'] = cell_types_counts1[cell_type_counts.index]
cell_type_counts['Sec2'] = cell_types_counts2[cell_type_counts.index]
fig, ax = plt.subplots(ncols=2, figsize=(10, 4))
sns.scatterplot(x=cell_type_counts['Sec1'], y=cell_type_counts['Sec2'], ax=ax[0], alpha=0.5)
ax[0].set_xlabel('Cell type counts in Sec1')
ax[0].set_ylabel('Cell type counts in Sec2')
# calculate the correlation
to_test = cell_type_counts.index[cell_type_counts['Sec1'] > 0].intersection(cell_type_counts.index[cell_type_counts['Sec2'] > 0])
corr, p_value = pearsonr(np.log(cell_type_counts['Sec1'].loc[to_test]), np.log(cell_type_counts['Sec2'].loc[to_test]))
# plot text
ax[0].text(0.05, 0.95, f'Pearson r: {corr:.2f}\np-value: {p_value:.2e}', transform=ax[0].transAxes, fontsize=12, verticalalignment='top')
ax[0].set_xscale('log')
ax[0].set_yscale('log')
sns.scatterplot(x=cell_type_counts['Exp2'], y=cell_type_counts['Exp3'], ax=ax[1], alpha=0.5)
ax[1].set_xlabel('Cell type counts in Exp2')
ax[1].set_ylabel('Cell type counts in Exp3')
# calculate the correlation
to_test = cell_type_counts.index[cell_type_counts['Exp2'] > 0].intersection(cell_type_counts.index[cell_type_counts['Exp3'] > 0])
corr, p_value = pearsonr(np.log(cell_type_counts['Exp2'].loc[to_test]), np.log(cell_type_counts['Exp3'].loc[to_test]))
# plot text
ax[1].text(0.05, 0.95, f'Pearson r: {corr:.2f}\np-value: {p_value:.2e}', transform=ax[1].transAxes, fontsize=12, verticalalignment='top')
ax[1].set_xscale('log')
ax[1].set_yscale('log')




# %%
# first check the correlation of transcripts and T7 per cell type.
fig, ax = plt.subplots(ncols=2, figsize=(12, 6))
cre_counts_sec1 = starrfish3_sec1.get_cre_expression().sum(axis=1)
cre_counts_sec2 = starrfish3_sec2.get_cre_expression().sum(axis=1)
t7_counts_sec1 = starrfish3_sec1.get_t7_expression().sum(axis=1)
t7_counts_sec2 = starrfish3_sec2.get_t7_expression().sum(axis=1)
cre_celltype_sec1 = cre_counts_sec1.groupby(starrfish3_sec1.get_tag('obs:subclass_name')).mean()
cre_celltype_sec2 = cre_counts_sec2.groupby(starrfish3_sec2.get_tag('obs:subclass_name')).mean()
t7_celltype_sec1 = t7_counts_sec1.groupby(starrfish3_sec1.get_tag('obs:subclass_name')).mean()
t7_celltype_sec2 = t7_counts_sec2.groupby(starrfish3_sec2.get_tag('obs:subclass_name')).mean()
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
    ax_work.set_xlabel(f'Average NC {p} counts per cell type in Sec1')
    ax_work.set_ylabel(f'Average NC {p} counts per cell type in Sec2')



# %% Heatmap of CRE counts and T7 counts per cell type
cell_types_to_use = cell_type_counts.index[(cell_type_counts['Sec1'] > 1000) & (cell_type_counts['Sec2'] > 1000) & 
                                           (cell_type_counts['Exp2'] > 1000) & (cell_type_counts['Exp3'] > 1000)].map(subclass_to_subclass_name)
cre_anno = pd.DataFrame(data = 0, index=cell_types_to_use.to_list() + ['Negative Control'], columns=starrfish3_sec1.get_creinfo().index)
for i in cell_types_to_use:
    cres = starrfish3_sec1.get_positive_control_cres(subclass_name_to_subclass[i], use='atac-peak')
    cre_anno.loc[i, cres] = 1
cre_anno.loc['Negative Control', starrfish3_sec1.get_negative_control_cres()] = 1

def plot_heatmap(expression_mat_sec1, expression_mat_sec2, cell_type_sec1, cell_type_sec2, cell_types_to_use, cre_anno, log=False):
    cre_celltype_sec1 = expression_mat_sec1.groupby(cell_type_sec1).mean()
    cre_celltype_sec2 = expression_mat_sec2.groupby(cell_type_sec2).mean()
    # sort cre by lib size
    
    fig, ax = plt.subplots(nrows=4, figsize=(24, 12), height_ratios=[1, 1, 0.1, 0.1])
    toplot1 = cre_celltype_sec1.loc[cell_types_to_use, cre_whitelist].copy()
    toplot2 = cre_celltype_sec2.loc[cell_types_to_use, cre_whitelist].copy()
    if log:
        toplot1 = np.log1p(toplot1)
        toplot2 = np.log1p(toplot2)
    # toplot1 = toplot1.div(toplot1.max(axis=0), axis=1)
    # toplot2 = toplot2.div(toplot2.max(axis=0), axis=1)
    sns.heatmap(toplot1, cmap='coolwarm', ax=ax[0])
    ax[0].set_xticks([])
    sns.heatmap(toplot2, cmap='coolwarm', ax=ax[1])
    ax[1].set_xticks([])
    # mark library size
    sns.heatmap(starrfish3_sec1.lib_size['counts'].loc[cre_anno.columns.intersection(cre_whitelist)].values.reshape(1, -1), cmap='coolwarm', ax=ax[2])
    ax[2].set_xticks([])
    # mark negative control
    sns.heatmap(cre_anno.loc[['Negative Control'], cre_whitelist], cmap='coolwarm', ax=ax[3])
    ax[3].set_xticks([])

cell_type_sec1 = starrfish3_sec1.get_tag('obs:subclass_name')
cell_type_sec2 = starrfish3_sec2.get_tag('obs:subclass_name')
plot_heatmap(starrfish3_sec1.get_cre_expression(), starrfish3_sec2.get_cre_expression(), cell_type_sec1, cell_type_sec2, cell_types_to_use, cre_anno)
plot_heatmap(starrfish3_sec1.get_t7_expression(), starrfish3_sec2.get_t7_expression(), cell_type_sec1, cell_type_sec2, cell_types_to_use, cre_anno)
plot_heatmap(starrfish3_sec1.get_cre_expression()>0, starrfish3_sec2.get_cre_expression()>0, cell_type_sec1, cell_type_sec2, cell_types_to_use, cre_anno)
plot_heatmap(starrfish3_sec1.get_t7_expression()>0, starrfish3_sec2.get_t7_expression()>0, cell_type_sec1, cell_type_sec2, cell_types_to_use, cre_anno)



# %% correlation of T7/CRE counts
# cell_types_to_use = cre_celltype_sec1.index.intersection(cre_celltype_sec2.index)
cres_to_use = cre_whitelist
cell_types_to_use = cell_type_counts.index[(cell_type_counts['Sec1'] > 1000) & (cell_type_counts['Sec2'] > 1000) & 
                                           (cell_type_counts['Exp2'] > 1000) & (cell_type_counts['Exp3'] > 1000)].map(subclass_to_subclass_name)
# # pick the cres that have target in any of the 5 cell types
# cres_to_use = starrfish3_sec1.lib_size.index[starrfish3_sec1.lib_size['counts'] >= 7].intersection(cre_whitelist)
# cres_to_use = []
# for i in cell_types_to_use:
#     cres_to_use.extend(starrfish3_sec1.get_positive_control_cres(subclass_name_to_subclass[i], use='atac-peak'))
# cres_to_use = np.unique(cres_to_use)
def plot_corr(activity_df1, activity_df2, cell_types_to_use, cres_to_use, cre_anno, log=False):
    cre_corr, celltype_corr = starrfish3_sec1.corr_starrfish(activity_df1=activity_df1.loc[cell_types_to_use, cres_to_use],
                                                             activity_df2=activity_df2.loc[cell_types_to_use, cres_to_use],
                                                             log_activity=log)
    cre_corr['lib_size'] = starrfish3_sec1.lib_size['counts'].loc[cre_corr.index]
    celltype_corr['cell_type_size_sec1'] = starrfish3_sec1.get_celltypes().value_counts().loc[celltype_corr.index.map(subclass_name_to_subclass)].values
    celltype_corr['cell_type_size_sec2'] = starrfish3_sec2.get_celltypes().value_counts().loc[celltype_corr.index.map(subclass_name_to_subclass)].values
    celltype_corr['cell_type_size'] = celltype_corr[['cell_type_size_sec1', 'cell_type_size_sec2']].min(axis=1)
    fig, ax = plt.subplots(nrows=2, figsize=(5, 10))
    sns.scatterplot(x=cre_corr.loc[cres_to_use, 'lib_size'], y=cre_corr.loc[cres_to_use, 'pearson'], ax=ax[0], alpha=0.5)
    sns.scatterplot(x=celltype_corr.loc[cell_types_to_use, 'cell_type_size'], y=celltype_corr.loc[cell_types_to_use, 'pearson'], ax=ax[1], alpha=0.5)
    ax[0].axhline(y=0.6, color='red', linestyle='--')
    ax[1].axhline(y=0.6, color='red', linestyle='--')
    ax[1].set_xscale('log')
    ax[0].set_xlabel('log(lib_size)')
    # mark negative control CREs
    neg_control_cres = cre_anno.loc['Negative Control'].index[cre_anno.loc['Negative Control'] == 1]
    neg_control_cres = neg_control_cres[neg_control_cres.isin(cre_corr.index)]
    sns.scatterplot(x=cre_corr.loc[neg_control_cres, 'lib_size'], y=cre_corr.loc[neg_control_cres, 'pearson'], ax=ax[0], color='orange', label='Negative Control CREs')

plot_corr(starrfish3_sec1.get_cre_expression().groupby(starrfish3_sec1.get_tag('obs:subclass_name')).sum(), 
          starrfish3_sec2.get_cre_expression().groupby(starrfish3_sec2.get_tag('obs:subclass_name')).sum(), 
          cell_types_to_use, cres_to_use, cre_anno, log=False)
plot_corr(starrfish3_sec1.get_t7_expression().groupby(starrfish3_sec1.get_tag('obs:subclass_name')).sum(), 
          starrfish3_sec2.get_t7_expression().groupby(starrfish3_sec2.get_tag('obs:subclass_name')).sum(), 
          cell_types_to_use, cres_to_use, cre_anno, log=False)
plot_corr((starrfish3_sec1.get_cre_expression() > 0).groupby(starrfish3_sec1.get_tag('obs:subclass_name')).mean(), 
          (starrfish3_sec2.get_cre_expression() > 0).groupby(starrfish3_sec2.get_tag('obs:subclass_name')).mean(), 
          cell_types_to_use, cres_to_use, cre_anno)
plot_corr((starrfish3_sec1.get_t7_expression() > 0).groupby(starrfish3_sec1.get_tag('obs:subclass_name')).mean(), 
          (starrfish3_sec2.get_t7_expression() > 0).groupby(starrfish3_sec2.get_tag('obs:subclass_name')).mean(), 
          cell_types_to_use, cres_to_use, cre_anno)
# %%
cell_types_to_use = cell_type_counts.index[(cell_type_counts['Sec1'] > 1000) & (cell_type_counts['Sec2'] > 1000) & 
                                           (cell_type_counts['Exp2'] > 1000) & (cell_type_counts['Exp3'] > 1000)].map(subclass_to_subclass_name)
# pick the CREs with at least 2^7 lib sizes
cres_to_use = starrfish3_sec1.lib_size.index[starrfish3_sec1.lib_size['counts'] >= 7]
# use the cell types with ≥ 0.6 correlation for both CRE and T7
df1 = starrfish3_sec1.get_cre_expression().groupby(starrfish3_sec1.get_tag('obs:subclass_name')).mean().loc[cell_types_to_use]
df2 = starrfish3_sec2.get_cre_expression().groupby(starrfish3_sec2.get_tag('obs:subclass_name')).mean().loc[cell_types_to_use]
cre_corr, celltype_corr = starrfish3_sec1.corr_starrfish(activity_df1=df1[cres_to_use], activity_df2=df2[cres_to_use])
# get the cell types with correlation > 0.6
# cell_types_to_use = celltype_corr.index[celltype_corr['pearson'] > 0.6].intersection(cell_types_to_use)
plot_corr(starrfish3_sec1.get_cre_expression().groupby(starrfish3_sec1.get_tag('obs:subclass_name')).mean().loc[cell_types_to_use] / 
          starrfish3_sec1.get_t7_expression().groupby(starrfish3_sec1.get_tag('obs:subclass_name')).mean().loc[cell_types_to_use], 
          starrfish3_sec2.get_cre_expression().groupby(starrfish3_sec2.get_tag('obs:subclass_name')).mean().loc[cell_types_to_use] /
          starrfish3_sec2.get_t7_expression().groupby(starrfish3_sec2.get_tag('obs:subclass_name')).mean().loc[cell_types_to_use], 
          cell_types_to_use, cres_to_use, cre_anno)
plot_corr((starrfish3_sec1.get_cre_expression() > 0).groupby(starrfish3_sec1.get_tag('obs:subclass_name')).mean().loc[cell_types_to_use] / 
          (starrfish3_sec1.get_t7_expression() > 0).groupby(starrfish3_sec1.get_tag('obs:subclass_name')).mean().loc[cell_types_to_use], 
          (starrfish3_sec2.get_cre_expression() > 0).groupby(starrfish3_sec2.get_tag('obs:subclass_name')).mean().loc[cell_types_to_use] /
          (starrfish3_sec2.get_t7_expression() > 0).groupby(starrfish3_sec2.get_tag('obs:subclass_name')).mean().loc[cell_types_to_use], 
          cell_types_to_use, cres_to_use, cre_anno)



# %% for each CRE/cell type, filter by T7 proportion
t7_infected_cells_sec1 = (starrfish3_sec1.get_t7_expression() > 0).groupby(starrfish3_sec1.get_tag('obs:subclass_name')).sum()
t7_infected_cells_sec2 = (starrfish3_sec2.get_t7_expression() > 0).groupby(starrfish3_sec2.get_tag('obs:subclass_name')).sum()
cre_infected_cells_sec1 = (starrfish3_sec1.get_cre_expression() > 0).groupby(starrfish3_sec1.get_tag('obs:subclass_name')).sum()
cre_infected_cells_sec2 = (starrfish3_sec2.get_cre_expression() > 0).groupby(starrfish3_sec2.get_tag('obs:subclass_name')).sum()
infected_cells_threshold = 0
def plot_corr_by_t7_infected_cells(activity_df1, activity_df2, infected_cells_sec1, infected_cells_sec2, 
                                   celltype_counts_sec1, celltype_counts_sec2, infected_cells_threshold, log=False):
    celltype_corr = pd.DataFrame(index=activity_df1.index.intersection(activity_df2.index), columns=['pearson', 'p_value', 'n_cres', 'n_cells'])
    for cell_type in activity_df1.index.intersection(activity_df2.index):
        cres_to_use = infected_cells_sec1.loc[cell_type].index[infected_cells_sec1.loc[cell_type] >= infected_cells_threshold].intersection(
            infected_cells_sec2.loc[cell_type].index[infected_cells_sec2.loc[cell_type] >= infected_cells_threshold]
        ).intersection(cre_whitelist)
        celltype_corr.loc[cell_type, 'n_cres'] = cres_to_use.size
        # if number of cres_to_use smaller than 2
        if cres_to_use.size < 2:
            continue
        if log:
            corr, p = pearsonr(np.log1p(activity_df1.loc[cell_type, cres_to_use]), np.log1p(activity_df2.loc[cell_type, cres_to_use]))
        else:
            corr, p = pearsonr(activity_df1.loc[cell_type, cres_to_use], activity_df2.loc[cell_type, cres_to_use])
        celltype_corr.loc[cell_type, 'pearson'] = corr
        celltype_corr.loc[cell_type, 'p_value'] = p
        celltype_corr.loc[cell_type, 'n_cells'] = np.minimum(
            celltype_counts_sec1.loc[cell_type],
            celltype_counts_sec2.loc[cell_type])
    cre_corr = pd.DataFrame(index=activity_df1.columns.intersection(activity_df2.columns), columns=['pearson', 'p_value', 'n_celltypes', 'lib_size'])
    for cre in activity_df1.columns.intersection(activity_df2.columns).intersection(cre_whitelist):
        celltypes_to_use = infected_cells_sec1[cre].index[infected_cells_sec1[cre] >= infected_cells_threshold].intersection(
            infected_cells_sec2[cre].index[infected_cells_sec2[cre] >= infected_cells_threshold]
        ).intersection(activity_df1.index).intersection(activity_df2.index)
        cre_corr.loc[cre, 'n_celltypes'] = celltypes_to_use.size
        if celltypes_to_use.size < 2:
            continue
        if log:
            corr, p = pearsonr(np.log1p(activity_df1.loc[celltypes_to_use, cre]), np.log1p(activity_df2.loc[celltypes_to_use, cre]))
        else:
            corr, p = pearsonr(activity_df1.loc[celltypes_to_use, cre], activity_df2.loc[celltypes_to_use, cre])
        cre_corr.loc[cre, 'pearson'] = corr
        cre_corr.loc[cre, 'p_value'] = p
        cre_corr.loc[cre, 'lib_size'] = starrfish3_sec1.lib_size['counts'].loc[cre]
    # plot n_cells with regard to n_cres, colored by pearson
    fig, ax = plt.subplots(figsize=(6, 6))
    # set mid point to 0.4
    norm = TwoSlopeNorm(vmin=celltype_corr['pearson'].min(), vcenter=0.4, vmax=celltype_corr['pearson'].max())
    sns.scatterplot(x=celltype_corr['n_cells'], y=celltype_corr['n_cres'], hue=celltype_corr['pearson'], ax=ax, palette='coolwarm', hue_norm=norm)
    ax.set_xscale('log')
    ax.set_xlabel('Number of Cells')
    ax.set_ylabel(f'Number of CREs with infected cells ≥ {infected_cells_threshold}')
    # plot lib size with regard to n_celltypes, colored by pearson
    fig, ax = plt.subplots(figsize=(6, 6))
    # set mid point to 0.4
    norm = TwoSlopeNorm(vmin=cre_corr['pearson'].min(), vcenter=0.4, vmax=cre_corr['pearson'].max())
    sns.scatterplot(y=cre_corr['n_celltypes'], x=cre_corr['lib_size'], hue=cre_corr['pearson'], ax=ax, palette='coolwarm', hue_norm=norm)
    ax.set_ylabel(f'Number of Cell Types with infected cells ≥ {infected_cells_threshold}')
    ax.set_xlabel('log(Library Size)')
    return celltype_corr, cre_corr

celltype_corr, cre_corr = plot_corr_by_t7_infected_cells((starrfish3_sec1.get_cre_expression()).groupby(starrfish3_sec1.get_tag('obs:subclass_name')).mean(),
                                                         (starrfish3_sec2.get_cre_expression()).groupby(starrfish3_sec2.get_tag('obs:subclass_name')).mean(),
                                                         cre_infected_cells_sec1, cre_infected_cells_sec2, 
                                                         starrfish3_sec1.get_tag('obs:subclass_name').value_counts(), 
                                                         starrfish3_sec2.get_tag('obs:subclass_name').value_counts(), 
                                                         infected_cells_threshold, log=False)
# fig, ax = plt.subplots(figsize=(6, 6))
# sns.scatterplot(x=celltype_corr.loc[celltype_corr.index[celltype_corr['n_cres'] >= 50], 'n_cells'],
#                 y=celltype_corr.loc[celltype_corr.index[celltype_corr['n_cres'] >= 50], 'pearson'], ax=ax)
# ax.set_xscale('log')
# ax.set_xlabel('Cell type size')
# ax.set_ylabel('Pearson correlation')
# fig, ax = plt.subplots(figsize=(6, 6))
# sns.scatterplot(x=cre_corr.loc[cre_corr.index[cre_corr['n_celltypes'] >= 20], 'n_celltypes'],
#                 y=cre_corr.loc[cre_corr.index[cre_corr['n_celltypes'] >= 20], 'pearson'], ax=ax)
# ax.set_xlabel('Number of Cell Types')
# ax.set_ylabel('Pearson correlation')
# violin plot
cell_type_n_threshold = 20
cre_n_threshold = 50
fig, ax = plt.subplots(figsize=(6, 6))
toplot1 = pd.DataFrame(celltype_corr['pearson'][celltype_corr['n_cres'] >= cre_n_threshold])
toplot2 = pd.DataFrame(cre_corr['pearson'][cre_corr['n_celltypes'] >= cell_type_n_threshold])
toplot1['metric'] = f'within cell type correlation\n{sum(celltype_corr["n_cres"] >= cre_n_threshold)} cell types'
toplot2['metric'] = f'across cell type correlation\n{sum(cre_corr["n_celltypes"] >= cell_type_n_threshold)} CREs'
toplot = pd.concat((toplot1, toplot2))
sns.violinplot(x=toplot['metric'], y=toplot['pearson'], hue=toplot['metric'], ax=ax)





# %% 
# apply the infection rate observed in Expr3 to Expr2
t7_infected_cells_expr3 = (starrfish3.get_t7_expression() > 0).groupby(starrfish3.get_tag('obs:subclass_name')).sum()
t7_infected_cells_expr2 = t7_infected_cells_expr3.copy()
cre_infected_cells_expr3 = (starrfish3.get_cre_expression() > 0).groupby(starrfish3.get_tag('obs:subclass_name')).sum()
cre_infected_cells_expr2 = (starrfish2.get_cre_expression() > 0).groupby(starrfish2.get_tag('obs:subclass_name')).sum()
infected_cells_threshold = 10
celltype_corr, cre_corr = plot_corr_by_t7_infected_cells((starrfish2.get_cre_expression()).groupby(starrfish2.get_tag('obs:subclass_name')).mean(),
                                                         (starrfish3.get_cre_expression()).groupby(starrfish3.get_tag('obs:subclass_name')).mean(),
                                                         cre_infected_cells_expr2, cre_infected_cells_expr3,
                                                         starrfish2.get_tag('obs:subclass_name').value_counts(),
                                                         starrfish3.get_tag('obs:subclass_name').value_counts(),
                                                         infected_cells_threshold, log=False)
# violin plot
cell_type_n_threshold = 20
cre_n_threshold = 20
fig, ax = plt.subplots(figsize=(6, 6))
toplot1 = pd.DataFrame(celltype_corr['pearson'][celltype_corr['n_cres'] >= cre_n_threshold])
toplot2 = pd.DataFrame(cre_corr['pearson'][cre_corr['n_celltypes'] >= cell_type_n_threshold])
toplot1['metric'] = f'within cell type correlation\n{sum(celltype_corr["n_cres"] >= cre_n_threshold)} cell types'
toplot2['metric'] = f'across cell type correlation\n{sum(cre_corr["n_celltypes"] >= cell_type_n_threshold)} CREs'
toplot = pd.concat((toplot1, toplot2))
sns.violinplot(x=toplot['metric'], y=toplot['pearson'], hue=toplot['metric'], ax=ax)
ax.set_ylabel('Pearson correlation')





# %% define the CREs and Cell Type matric to keep
infected_cells_threshold = 5
to_filter = (starrfish3.get_cre_expression() > 0).groupby(starrfish3.get_celltypes()).sum() < infected_cells_threshold
to_filter_sec1 = (starrfish3_sec1.get_cre_expression() > 0).groupby(starrfish3_sec1.get_celltypes()).sum() < infected_cells_threshold
to_filter_sec2 = (starrfish3_sec2.get_cre_expression() > 0).groupby(starrfish3_sec2.get_celltypes()).sum() < infected_cells_threshold
to_filter[cre_blacklist] = True
to_filter_sec1[cre_blacklist] = True
to_filter_sec2[cre_blacklist] = True





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
    'n_jobs': 28,
}
threshold = 'neg_control'
res1 = starrfish3_sec1.average_bootstrap_test(**average_bootstrap_test_config)
res_q1, res_df1 = starrfish3_sec1.average_bootstrap_test_q(res1, threshold=threshold, norm='T7', tail='both', to_filter=to_filter_sec1)
res2 = starrfish3_sec2.average_bootstrap_test(**average_bootstrap_test_config)
res_q2, res_df2 = starrfish3_sec2.average_bootstrap_test_q(res2, threshold=threshold, norm='T7', tail='both', to_filter=to_filter_sec2)
res = starrfish3.average_bootstrap_test(**average_bootstrap_test_config)
res_q, res_df = starrfish3.average_bootstrap_test_q(res, threshold=threshold, norm='T7', tail='both', to_filter=to_filter)




# %% test the correlation of bootstrapping test results
cre_corr, celltype_corr = starrfish3.corr_starrfish(activity_df1=res_df1, activity_df2=res_df2, log_activity=False)
# %% plot the cre_corr versus effect_n
fig, ax = plt.subplots(ncols=2, figsize=(12, 6))
sns.scatterplot(data=cre_corr, x='effect_n', y='pearson', ax=ax[0])
ax[0].set_title('Across Cell Type Correlation vs Effect Cell Types')
ax[0].set_xlabel('Cell Types')
ax[0].set_ylabel('CRE Correlation')
sns.scatterplot(data=celltype_corr, x='effect_n', y='pearson', ax=ax[1])
ax[1].set_title('Within Cell Type Correlation vs Effect CREs')
ax[1].set_xlabel('CREs')
ax[1].set_ylabel('Cell Type Correlation')
# plot violin plot
cell_type_n_threshold = 50
cre_n_threshold = 50
fig, ax = plt.subplots(figsize=(6, 6))
toplot1 = pd.DataFrame(celltype_corr['pearson'][celltype_corr['effect_n'] >= cre_n_threshold])
toplot2 = pd.DataFrame(cre_corr['pearson'][cre_corr['effect_n'] >= cell_type_n_threshold])
toplot1['metric'] = f'within cell type correlation\n{sum(celltype_corr['effect_n']>= cre_n_threshold)} cell types'
toplot2['metric'] = f'across cell type correlation\n{sum(cre_corr['effect_n']>= cell_type_n_threshold)} CREs'
toplot = pd.concat((toplot1, toplot2))
sns.violinplot(x=toplot['metric'], y=toplot['pearson'], hue=toplot['metric'], ax=ax)





# %% correlation with ATAC_cpm
cre_corr, celltype_corr = starrfish3.corr_atac_cpm(None, None, acvitity_df=res_df, log_atac=True)
# %% plot the cre_corr versus effect_n
fig, ax = plt.subplots(ncols=2, figsize=(12, 6))
sns.scatterplot(data=cre_corr, x='effect_n', y='pearson', ax=ax[0])
ax[0].set_title('Across Cell Type Correlation vs Effect Cell Types')
ax[0].set_xlabel('Cell Types')
ax[0].set_ylabel('CRE Correlation')
sns.scatterplot(data=celltype_corr, x='effect_n', y='pearson', ax=ax[1])
ax[1].set_title('Within Cell Type Correlation vs Effect CREs')
ax[1].set_xlabel('CREs')
ax[1].set_ylabel('Cell Type Correlation')
# plot violin plot
cell_type_n_threshold = 50
cre_n_threshold = 50
fig, ax = plt.subplots(figsize=(6, 6))
toplot1 = pd.DataFrame(celltype_corr['pearson'][celltype_corr['effect_n'] >= cre_n_threshold])
toplot2 = pd.DataFrame(cre_corr['pearson'][cre_corr['effect_n'] >= cell_type_n_threshold])
toplot1['metric'] = f'within cell type correlation\n{sum(celltype_corr['effect_n']>= cre_n_threshold)} cell types'
toplot2['metric'] = f'across cell type correlation\n{sum(cre_corr['effect_n']>= cell_type_n_threshold)} CREs'
toplot = pd.concat((toplot1, toplot2))
sns.violinplot(x=toplot['metric'], y=toplot['pearson'], hue=toplot['metric'], ax=ax)




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
                           "n_jobs": 28,
                           'load_stored': True,}
res1 = starrfish3_sec1.fold_change_test(**fold_change_test_config)
res2 = starrfish3_sec2.fold_change_test(**fold_change_test_config)
res = starrfish3.fold_change_test(**fold_change_test_config)



# %% save the results
starrfish3_sec1.save('results/starrfish3_sec1.pkl')
starrfish3_sec2.save('results/starrfish3_sec2.pkl')
starrfish3.save('results/starrfish3.pkl')
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
def plot_q_value_reproducibility(res_q1, res_q2, res_q):
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
    # plot the percentage vs number of cells
    res_compare['Cell_counts1'] = starrfish3_sec1.get_celltypes().value_counts().loc[common_celltypes].values
    res_compare['Cell_counts2'] = starrfish3_sec2.get_celltypes().value_counts().loc[common_celltypes].values
    res_compare['Cell_counts'] = starrfish3.get_celltypes().value_counts().loc[common_celltypes].values
    fig, ax = plt.subplots(ncols=3, figsize=(12, 4))
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

    mask_valid = ~res_compare['Percentage'].isna()
    mask_nan = res_compare['Percentage'].isna()
    # Plot valid values with coolwarm palette
    if mask_valid.any():
        sns.scatterplot(x=res_compare.loc[mask_valid, 'Cell_counts1'], 
                        y=res_compare.loc[mask_valid, 'Cell_counts2'],
                        hue=res_compare.loc[mask_valid, 'Percentage'], 
                        palette='coolwarm', ax=ax[2], alpha=0.5)
    # Plot NaN values in grey
    if mask_nan.any():
        ax[2].scatter(res_compare.loc[mask_nan, 'Cell_counts1'], 
                    res_compare.loc[mask_nan, 'Cell_counts2'], edgecolors='none', s=20,  # Adjust size as needed
                    color='grey', alpha=0.5, label='NA')
    ax[2].set_xscale('log')
    ax[2].set_yscale('log')
    ax[2].set_xlabel('Cell counts in Sec1')
    ax[2].set_ylabel('Cell counts in Sec2')
    # plot violin plot
    fig, ax = plt.subplots(ncols=3, figsize=(12, 4), gridspec_kw={'wspace': 0.4})
    sns.violinplot(y=res_compare['Percentage_sec1'], ax=ax[0])
    sns.violinplot(y=res_compare['Percentage_sec2'], ax=ax[1])
    sns.violinplot(y=res_compare['Percentage'], ax=ax[2])
    return res_compare
res_compare = plot_q_value_reproducibility(res_q1, res_q2, res_q)





# %% fisher exact test significant reproducibility
infected_cells_threshold = 5
to_filter = (starrfish3.get_cre_expression() > 0).groupby(starrfish3.get_celltypes()).sum() < infected_cells_threshold
to_filter_sec1 = (starrfish3_sec1.get_cre_expression() > 0).groupby(starrfish3_sec1.get_celltypes()).sum() < infected_cells_threshold
to_filter_sec2 = (starrfish3_sec2.get_cre_expression() > 0).groupby(starrfish3_sec2.get_celltypes()).sum() < infected_cells_threshold
to_filter[cre_blacklist] = True
to_filter_sec1[cre_blacklist] = True
to_filter_sec2[cre_blacklist] = True
def fisher_exact_test(starrfish_obj, baseline = 'total'):
    t7_infected = (starrfish_obj.get_t7_expression() > 0).groupby(starrfish_obj.get_celltypes()).sum()
    cre_infected = (starrfish_obj.get_cre_expression() > 0).groupby(starrfish_obj.get_celltypes()).sum()
    total_t7_infected = (starrfish_obj.get_t7_expression().sum(axis=1) > 0).groupby(starrfish_obj.get_celltypes()).sum()
    total_cre_infected = (starrfish_obj.get_cre_expression().sum(axis=1) > 0).groupby(starrfish_obj.get_celltypes()).sum()
    neg_t7_infected = (starrfish_obj.get_t7_expression()[starrfish_obj.get_negative_control_cres()].sum(axis=1) > 0).groupby(starrfish_obj.get_celltypes()).sum()
    neg_cre_infected = (starrfish_obj.get_cre_expression()[starrfish_obj.get_negative_control_cres()].sum(axis=1) > 0).groupby(starrfish_obj.get_celltypes()).sum()
    if baseline == 'total':
        fisher_data = pd.DataFrame({
            'T7 Infected': t7_infected.values.flatten(),
            'CRE Infected': cre_infected.values.flatten(),
            'Baseline T7 Infected': total_t7_infected.values.flatten().repeat(t7_infected.shape[1]),
            'Baseline CRE Infected': total_cre_infected.values.flatten().repeat(cre_infected.shape[1])
        })
    elif baseline == 'neg_control':
        fisher_data = pd.DataFrame({
            'T7 Infected': t7_infected.values.flatten(),
            'CRE Infected': cre_infected.values.flatten(),
            'Baseline T7 Infected': neg_t7_infected.values.flatten().repeat(t7_infected.shape[1]),
            'Baseline CRE Infected': neg_cre_infected.values.flatten().repeat(cre_infected.shape[1])
        })
    # vectorized fisher exact test
    fisher_results = fisher_data.apply(
        lambda row: pd.Series(stats.fisher_exact([
            [row['T7 Infected'], row['Baseline T7 Infected']],
            [row['CRE Infected'], row['Baseline CRE Infected']]
        ]), index=['Odds Ratio', 'p-value']), 
        axis=1
    )
    fisher_odd = pd.DataFrame(fisher_results['Odds Ratio'].values.reshape(t7_infected.shape),
                              index=t7_infected.index, columns=t7_infected.columns)
    fisher_p = pd.DataFrame(fisher_results['p-value'].values.reshape(t7_infected.shape),
                            index=t7_infected.index, columns=t7_infected.columns)
    return fisher_odd, fisher_p
odd_sec1, p_sec1 = fisher_exact_test(starrfish3_sec1, baseline = 'neg_control')
odd_sec2, p_sec2 = fisher_exact_test(starrfish3_sec2, baseline = 'neg_control')
odd, p = fisher_exact_test(starrfish3, baseline = 'neg_control')
p_sec1[to_filter_sec1], odd_sec1[to_filter_sec1] = np.nan, np.nan
p_sec2[to_filter_sec2], odd_sec2[to_filter_sec2] = np.nan, np.nan
p[to_filter], odd[to_filter] = np.nan, np.nan
# transform to q value
q_sec1 = p_sec1.values.flatten().copy()
q_sec1[~np.isnan(q_sec1)] = multitest.multipletests(q_sec1[~np.isnan(q_sec1)], method='fdr_bh')[1]
q_sec1 = pd.DataFrame(q_sec1.reshape(p_sec1.shape), index=p_sec1.index, columns=p_sec1.columns)
q_sec2 = p_sec2.values.flatten().copy()
q_sec2[~np.isnan(q_sec2)] = multitest.multipletests(q_sec2[~np.isnan(q_sec2)], method='fdr_bh')[1]
q_sec2 = pd.DataFrame(q_sec2.reshape(p_sec2.shape), index=p_sec2.index, columns=p_sec2.columns)
q = p.values.flatten().copy()
q[~np.isnan(q)] = multitest.multipletests(q[~np.isnan(q)], method='fdr_bh')[1]
q = pd.DataFrame(q.reshape(p.shape), index=p.index, columns=p.columns)
res_compare = plot_q_value_reproducibility(q_sec1, q_sec2, q)






# %% correlation to atac_cpm
cre_corr, celltype_corr = starrfish3.corr_atac_cpm(None, None, odd, log_atac=True, log_activity=True)
fig, ax = plt.subplots(ncols=2, figsize=(12, 6))
sns.scatterplot(data=cre_corr, x='effect_n', y='pearson', ax=ax[0])
ax[0].set_title('Across Cell Type Correlation vs Effect Cell Types')
ax[0].set_xlabel('Cell Types')
ax[0].set_ylabel('CRE Correlation')
sns.scatterplot(data=celltype_corr, x='effect_n', y='pearson', ax=ax[1])
ax[1].set_title('Within Cell Type Correlation vs Effect CREs')
ax[1].set_xlabel('CREs')
ax[1].set_ylabel('Cell Type Correlation')
# plot violin plot
cell_type_n_threshold = 50
cre_n_threshold = 50
fig, ax = plt.subplots(figsize=(6, 6))
toplot1 = pd.DataFrame(celltype_corr['pearson'][celltype_corr['effect_n'] >= cre_n_threshold])
toplot2 = pd.DataFrame(cre_corr['pearson'][cre_corr['effect_n'] >= cell_type_n_threshold])
toplot1['metric'] = f'within cell type correlation\n{sum(celltype_corr['effect_n']>= cre_n_threshold)} cell types'
toplot2['metric'] = f'across cell type correlation\n{sum(cre_corr['effect_n']>= cell_type_n_threshold)} CREs'
toplot = pd.concat((toplot1, toplot2))
sns.violinplot(x=toplot['metric'], y=toplot['pearson'], hue=toplot['metric'], ax=ax)





# %%
cre_corr, celltype_corr = starrfish3.corr_atac_cpm(None, None, res_df, log_atac=True)
fig, ax = plt.subplots(ncols=2, figsize=(6, 6), gridspec_kw={'wspace': 0.4})
sns.violinplot(celltype_corr['pearson'], ax=ax[0])
ax[0].set_title('Cell Type Correlation with ATAC')
sns.violinplot(cre_corr['pearson'], ax=ax[1])
ax[1].set_title('CRE Correlation with ATAC')
plt.show()




# %%
from plots import cre_pval_dotplot
cell_types_to_use = starrfish3.get_celltypes().value_counts().index[starrfish3.get_celltypes().value_counts()>=1000]
infected_cells_threshold = 5
to_filter = (starrfish3.get_cre_expression() > 0).groupby(starrfish3.get_celltypes()).sum() < infected_cells_threshold
to_filter[cre_blacklist] = True
q_toplot = res_q.copy()
q_toplot[to_filter] = np.nan
res_toplot = res_df.copy()
res_toplot[to_filter] = np.nan
cres_to_use = q_toplot.columns[np.nanmin(q_toplot.loc[cell_types_to_use], axis=0) < 0.05]
cre_pval_dotplot(q_toplot, res_toplot, cres_to_use, cell_types_to_use, None, figsize=(18, 24))
# %%
