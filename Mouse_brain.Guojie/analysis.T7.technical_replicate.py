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
# read in the mismatching barcode CREs
mismatching_cres = pd.read_csv('Data/AAV_ONT_Barcode_Counts_vs_Mismatch_Percentage.csv', index_col=0)
cre_whitelist = cre_whitelist[~cre_whitelist.isin(mismatching_cres.index[mismatching_cres['MismatchPercent'] > 20])]
cre_blacklist = np.unique(cre_blacklist + mismatching_cres.index[mismatching_cres['MismatchPercent'] > 20].tolist()).tolist()


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
fig.savefig('results/expr3/T7_libsize.pdf')




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
fig.savefig('results/expr3/sec1_sec2_cell_type_counts.pdf')



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
fig.savefig('results/expr3/sec1_sec2_cre_t7_counts.pdf')



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
cell_types_to_use = cre_celltype_sec1.index.intersection(cre_celltype_sec2.index)
cres_to_use = cre_whitelist
# cell_types_to_use = cell_type_counts.index[(cell_type_counts['Sec1'] > 1000) & (cell_type_counts['Sec2'] > 1000) & 
#                                            (cell_type_counts['Exp2'] > 1000) & (cell_type_counts['Exp3'] > 1000)].map(subclass_to_subclass_name)
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
    # plot significant cres
    significant_cres = cre_corr.index[(cre_corr['pearson_p'] < 0.05) & (cre_corr['pearson'] > 0)].intersection(cres_to_use)
    sns.scatterplot(x=cre_corr.loc[significant_cres, 'lib_size'], y=cre_corr.loc[significant_cres, 'pearson'], ax=ax[0], alpha=0.5, label='Significant CREs', color='red')
    significant_celltypes = celltype_corr.index[(celltype_corr['pearson_p'] < 0.05) & (celltype_corr['pearson'] > 0)].intersection(cell_types_to_use)
    sns.scatterplot(x=celltype_corr.loc[significant_celltypes, 'cell_type_size'], y=celltype_corr.loc[significant_celltypes, 'pearson'], ax=ax[1], alpha=0.5, label='Significant Cell Types', color='red')
    # plot not significant cres
    not_significant_cres = cre_corr.index[(cre_corr['pearson_p'] >= 0.05) | (cre_corr['pearson'] <= 0)].intersection(cres_to_use)
    sns.scatterplot(x=cre_corr.loc[not_significant_cres, 'lib_size'], y=cre_corr.loc[not_significant_cres, 'pearson'], ax=ax[0], alpha=0.5, color='blue')
    not_significant_celltypes = celltype_corr.index[(celltype_corr['pearson_p'] >= 0.05) | (celltype_corr['pearson'] <= 0)].intersection(cell_types_to_use)
    sns.scatterplot(x=celltype_corr.loc[not_significant_celltypes, 'cell_type_size'], y=celltype_corr.loc[not_significant_celltypes, 'pearson'], ax=ax[1], alpha=0.5, color='blue')
    # ax[0].axhline(y=0.6, color='red', linestyle='--')
    # ax[1].axhline(y=0.6, color='red', linestyle='--')
    ax[1].set_xscale('log')
    ax[0].set_xlabel('log(lib_size)')
    # mark negative control CREs
    # neg_control_cres = cre_anno.loc['Negative Control'].index[cre_anno.loc['Negative Control'] == 1]
    # neg_control_cres = neg_control_cres[neg_control_cres.isin(cre_corr.index)]
    # sns.scatterplot(x=cre_corr.loc[neg_control_cres, 'lib_size'], y=cre_corr.loc[neg_control_cres, 'pearson'], ax=ax[0], color='orange', label='Negative Control CREs')
    return fig

fig1 = plot_corr(starrfish3_sec1.get_cre_expression().groupby(starrfish3_sec1.get_tag('obs:subclass_name')).sum(), 
                 starrfish3_sec2.get_cre_expression().groupby(starrfish3_sec2.get_tag('obs:subclass_name')).sum(), 
                 cell_types_to_use, cres_to_use, cre_anno, log=False)
fig2 = plot_corr(starrfish3_sec1.get_t7_expression().groupby(starrfish3_sec1.get_tag('obs:subclass_name')).sum(), 
                 starrfish3_sec2.get_t7_expression().groupby(starrfish3_sec2.get_tag('obs:subclass_name')).sum(), 
                 cell_types_to_use, cres_to_use, cre_anno, log=False)
fig3 = plot_corr((starrfish3_sec1.get_cre_expression() > 0).groupby(starrfish3_sec1.get_tag('obs:subclass_name')).mean(), 
                 (starrfish3_sec2.get_cre_expression() > 0).groupby(starrfish3_sec2.get_tag('obs:subclass_name')).mean(), 
                 cell_types_to_use, cres_to_use, cre_anno)
fig4 = plot_corr((starrfish3_sec1.get_t7_expression() > 0).groupby(starrfish3_sec1.get_tag('obs:subclass_name')).mean(), 
                 (starrfish3_sec2.get_t7_expression() > 0).groupby(starrfish3_sec2.get_tag('obs:subclass_name')).mean(), 
                 cell_types_to_use, cres_to_use, cre_anno)
fig1.savefig("results/expr3/sec1_sec2.cre.counts.corr.pdf")
fig2.savefig("results/expr3/sec1_sec2.t7.counts.corr.pdf")
fig3.savefig("results/expr3/sec1_sec2.cre.proportion.corr.pdf")
fig4.savefig("results/expr3/sec1_sec2.t7.proportion.corr.pdf")
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
infected_cells_threshold = 5
def plot_corr_by_infected_cells(activity_df1, activity_df2, infected_cells_sec1, infected_cells_sec2, 
                                   celltype_counts_sec1, celltype_counts_sec2, infected_cells_threshold, log=False):
    celltype_corr = pd.DataFrame(index=activity_df1.index.intersection(activity_df2.index), columns=['pearson', 'p_value', 'n_cres', 'n_cells'])
    for cell_type in activity_df1.index.intersection(activity_df2.index):
        cres_to_use = infected_cells_sec1.loc[cell_type].index[infected_cells_sec1.loc[cell_type] >= infected_cells_threshold].intersection(
            infected_cells_sec2.loc[cell_type].index[infected_cells_sec2.loc[cell_type] >= infected_cells_threshold]
        ).intersection(cre_whitelist)
        cres_to_use = cres_to_use[~pd.isna(activity_df1.loc[cell_type, cres_to_use])]
        cres_to_use = cres_to_use[~pd.isna(activity_df2.loc[cell_type, cres_to_use])]
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
        celltypes_to_use = celltypes_to_use[~pd.isna(activity_df1.loc[celltypes_to_use, cre])]
        celltypes_to_use = celltypes_to_use[~pd.isna(activity_df2.loc[celltypes_to_use, cre])]
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
    fig1, ax = plt.subplots(figsize=(6, 6))
    # set mid point to 0.4
    norm = TwoSlopeNorm(vmin=celltype_corr['pearson'].min(), vcenter=0.4, vmax=celltype_corr['pearson'].max())
    sns.scatterplot(x=celltype_corr['n_cells'], y=celltype_corr['n_cres'], hue=celltype_corr['pearson'], ax=ax, palette='coolwarm', hue_norm=norm)
    ax.set_xscale('log')
    ax.set_xlabel('Number of Cells')
    ax.set_ylabel(f'Number of CREs with infected cells ≥ {infected_cells_threshold}')
    # plot lib size with regard to n_celltypes, colored by pearson
    fig2, ax = plt.subplots(figsize=(6, 6))
    # set mid point to 0.4
    norm = TwoSlopeNorm(vmin=cre_corr['pearson'].min(), vcenter=0.4, vmax=cre_corr['pearson'].max())
    sns.scatterplot(y=cre_corr['n_celltypes'], x=cre_corr['lib_size'], hue=cre_corr['pearson'], ax=ax, palette='coolwarm', hue_norm=norm)
    ax.set_ylabel(f'Number of Cell Types with infected cells ≥ {infected_cells_threshold}')
    ax.set_xlabel('log(Library Size)')
    return celltype_corr, cre_corr, fig1, fig2
# %%
celltype_corr, cre_corr, fig1, fig2 = \
    plot_corr_by_infected_cells((starrfish3_sec1.get_cre_expression()).groupby(starrfish3_sec1.get_tag('obs:subclass_name')).mean(),
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
fig1.savefig("results/expr3/sec1_sec2.cre_n>5.cre.within.celltype.corr.pdf")
fig2.savefig("results/expr3/sec1_sec2.cre_n>5.cre.across.celltype.corr.pdf")
fig.savefig("results/expr3/sec1_sec2.cre_n>5.cre.violin.plot.pdf")
# %%
toplot = pd.DataFrame()
for infected_cells_threshold in [0, 5, 10, 20]:
    celltype_corr, cre_corr, fig1, fig2 = \
        plot_corr_by_infected_cells((starrfish3_sec1.get_t7_expression()).groupby(starrfish3_sec1.get_tag('obs:subclass_name')).mean(),
                                    (starrfish3_sec2.get_t7_expression()).groupby(starrfish3_sec2.get_tag('obs:subclass_name')).mean(),
                                    cre_infected_cells_sec1, cre_infected_cells_sec2, 
                                    starrfish3_sec1.get_tag('obs:subclass_name').value_counts(), 
                                    starrfish3_sec2.get_tag('obs:subclass_name').value_counts(),
                                    infected_cells_threshold, log=False)
    # violin plot
    cell_type_n_threshold = 10
    cre_n_threshold = 10
    toplot1 = pd.DataFrame(celltype_corr['pearson'][celltype_corr['n_cres'] >= cre_n_threshold])
    toplot2 = pd.DataFrame(cre_corr['pearson'][cre_corr['n_celltypes'] >= cell_type_n_threshold])
    toplot1['metric'] = f'Cells ≥ {infected_cells_threshold}'
    toplot2['metric'] = f'Cells ≥ {infected_cells_threshold}'
    toplot1['Corr_type'] = 'Cell type wise'
    toplot2['Corr_type'] = 'CRE wise'
    toplot = pd.concat((toplot, toplot1, toplot2), ignore_index=True)
fig, ax = plt.subplots(figsize=(12, 6))
sns.violinplot(x=toplot['metric'], y=toplot['pearson'], hue=toplot['Corr_type'], ax=ax)
fig.savefig("results/expr3/sec1_sec2.violin.plot.multiple_thresholds.t7.pdf")



# %% 
# apply the infection rate observed in Expr3 to Expr2
t7_infected_cells_expr3 = (starrfish3.get_t7_expression() > 0).groupby(starrfish3.get_tag('obs:subclass_name')).sum()
t7_infected_cells_expr2 = t7_infected_cells_expr3.copy()
cre_infected_cells_expr3 = (starrfish3.get_cre_expression() > 0).groupby(starrfish3.get_tag('obs:subclass_name')).sum()
cre_infected_cells_expr2 = (starrfish2.get_cre_expression() > 0).groupby(starrfish2.get_tag('obs:subclass_name')).sum()
infected_cells_threshold = 5
celltype_corr, cre_corr, fig1, fig2 = \
    plot_corr_by_infected_cells((starrfish2.get_cre_expression()).groupby(starrfish2.get_tag('obs:subclass_name')).mean(),
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
fig1.savefig("results/expr3/expr3_expr2.cre_n>5.cre.within.celltype.corr.pdf")
fig2.savefig("results/expr3/expr3_expr2.cre_n>5.cre.across.celltype.corr.pdf")
fig.savefig("results/expr3/expr3_expr2.cre_n>5.cre.violin.plot.pdf")




# %% define the CREs and Cell Type matric to keep
infected_cells_threshold = 5
to_filter = (starrfish3.get_cre_expression() > 0).groupby(starrfish3.get_celltypes()).sum() < infected_cells_threshold
to_filter_sec1 = (starrfish3_sec1.get_cre_expression() > 0).groupby(starrfish3_sec1.get_celltypes()).sum() < infected_cells_threshold
to_filter_sec2 = (starrfish3_sec2.get_cre_expression() > 0).groupby(starrfish3_sec2.get_celltypes()).sum() < infected_cells_threshold
to_filter[cre_blacklist] = True
to_filter_sec1[cre_blacklist] = True
to_filter_sec2[cre_blacklist] = True


# %% run the pseudo bulk bootstrap test for T7
pseudo_bulk_glm_test_config = {
    'cell_types_to_use': None,
    'variate': 'T7',
    'norm_by_volm': False,
    'volm_covariate': False,  # normalize by T7, filter cells with T7 < 4
    'rna_covariate': False,
    'filter_infected_cells': False,
    'positive_x_or_y': False,  # normalize by T7
    'only_keep_positive_x': False,
    'only_keep_positive_y': False,  # normalize by negative control
    'pseudo_bulk_size': None,
    'pseudo_bulk_percentage': 1,
    'pseudo_bulk_number': 10000,
    'replace': True,
    'multiprocess_threads': 128,
}
res1 = starrfish3_sec1.pseudo_bulk_glm_test(**pseudo_bulk_glm_test_config)
res2 = starrfish3_sec2.pseudo_bulk_glm_test(**pseudo_bulk_glm_test_config)
res = starrfish3.pseudo_bulk_glm_test(**pseudo_bulk_glm_test_config)

starrfish3_sec1.save('results/starrfish3_sec1.pkl')
starrfish3_sec2.save('results/starrfish3_sec2.pkl')
starrfish3.save('results/starrfish3.pkl')

pseudo_bulk_glm_test_config['pseudo_bulk_size'] = 50
res1 = starrfish3_sec1.pseudo_bulk_glm_test(**pseudo_bulk_glm_test_config)
res2 = starrfish3_sec2.pseudo_bulk_glm_test(**pseudo_bulk_glm_test_config)
res = starrfish3.pseudo_bulk_glm_test(**pseudo_bulk_glm_test_config)

starrfish3_sec1.save('results/starrfish3_sec1.pkl')
starrfish3_sec2.save('results/starrfish3_sec2.pkl')
starrfish3.save('results/starrfish3.pkl')

pseudo_bulk_glm_test_config['pseudo_bulk_size'] = 100
res1 = starrfish3_sec1.pseudo_bulk_glm_test(**pseudo_bulk_glm_test_config)
res2 = starrfish3_sec2.pseudo_bulk_glm_test(**pseudo_bulk_glm_test_config)
res = starrfish3.pseudo_bulk_glm_test(**pseudo_bulk_glm_test_config)

starrfish3_sec1.save('results/starrfish3_sec1.pkl')
starrfish3_sec2.save('results/starrfish3_sec2.pkl')
starrfish3.save('results/starrfish3.pkl')


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
    'bootstrap_to_fixed_pct': 1,
    'bootstrap_to_fixed_sample_size': None,
    'load_stored': True,
    'n_jobs': 128,
}
threshold = 'neg_control_mean'
res1 = starrfish3_sec1.average_bootstrap_test(**average_bootstrap_test_config)
res_q1, res_df1 = starrfish3_sec1.average_bootstrap_test_q(res1, threshold=threshold, norm='T7', tail='both', to_filter=to_filter_sec1, calibrate=None)
res2 = starrfish3_sec2.average_bootstrap_test(**average_bootstrap_test_config)
res_q2, res_df2 = starrfish3_sec2.average_bootstrap_test_q(res2, threshold=threshold, norm='T7', tail='both', to_filter=to_filter_sec2, calibrate=None)
res = starrfish3.average_bootstrap_test(**average_bootstrap_test_config)
res_q, res_df = starrfish3.average_bootstrap_test_q(res, threshold=threshold, norm='T7', tail='both', to_filter=to_filter, calibrate=None)




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
cell_type_n_threshold = 10
cre_n_threshold = 10
fig, ax = plt.subplots(figsize=(6, 6))
toplot1 = pd.DataFrame(celltype_corr['pearson'][celltype_corr['effect_n'] >= cre_n_threshold])
toplot2 = pd.DataFrame(cre_corr['pearson'][cre_corr['effect_n'] >= cell_type_n_threshold])
toplot1['metric'] = f'Cell Type wise correlation\n{sum(celltype_corr['effect_n']>= cre_n_threshold)} cell types'
toplot2['metric'] = f'CRE wise correlation\n{sum(cre_corr['effect_n']>= cell_type_n_threshold)} CREs'
toplot = pd.concat((toplot1, toplot2))
sns.violinplot(x=toplot['metric'], y=toplot['pearson'], hue=toplot['metric'], ax=ax)
# %%
t7_infected_cells_sec1 = (starrfish3_sec1.get_t7_expression() > 0).groupby(starrfish3_sec1.get_tag('obs:subclass')).sum()
t7_infected_cells_sec2 = (starrfish3_sec2.get_t7_expression() > 0).groupby(starrfish3_sec2.get_tag('obs:subclass')).sum()
cre_infected_cells_sec1 = (starrfish3_sec1.get_cre_expression() > 0).groupby(starrfish3_sec1.get_tag('obs:subclass')).sum()
cre_infected_cells_sec2 = (starrfish3_sec2.get_cre_expression() > 0).groupby(starrfish3_sec2.get_tag('obs:subclass')).sum()
infected_cells_threshold = 0
celltype_corr, cre_corr, fig1, fig2 = \
    plot_corr_by_infected_cells(res_df1, res_df2,
                                cre_infected_cells_sec1, cre_infected_cells_sec2, 
                                starrfish3_sec1.get_tag('obs:subclass').value_counts(), 
                                starrfish3_sec2.get_tag('obs:subclass').value_counts(),
                                infected_cells_threshold, log=False)




# %% plot q-value for all cell types
from plots import plot_q_value_celltype_reproducibility
res_compare = plot_q_value_celltype_reproducibility(res_q1, res_q2, res_q, 
                                                    starrfish3_sec1.get_celltypes().value_counts(),
                                                    starrfish3_sec2.get_celltypes().value_counts(),
                                                    starrfish3.get_celltypes().value_counts())





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
cell_type_n_threshold = 20
cre_n_threshold = 50
fig, ax = plt.subplots(figsize=(6, 6))
toplot1 = pd.DataFrame(celltype_corr['pearson'][celltype_corr['effect_n'] >= cre_n_threshold])
toplot2 = pd.DataFrame(cre_corr['pearson'][cre_corr['effect_n'] >= cell_type_n_threshold])
toplot1['metric'] = f'within cell type correlation\n{sum(celltype_corr['effect_n']>= cre_n_threshold)} cell types'
toplot2['metric'] = f'across cell type correlation\n{sum(cre_corr['effect_n']>= cell_type_n_threshold)} CREs'
toplot = pd.concat((toplot1, toplot2))
sns.violinplot(x=toplot['metric'], y=toplot['pearson'], hue=toplot['metric'], ax=ax)







# %% fisher exact test significant reproducibility
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
# %%
infected_cells_threshold = 5
to_filter = (starrfish3.get_cre_expression() > 0).groupby(starrfish3.get_celltypes()).sum() < infected_cells_threshold
to_filter_sec1 = (starrfish3_sec1.get_cre_expression() > 0).groupby(starrfish3_sec1.get_celltypes()).sum() < infected_cells_threshold
to_filter_sec2 = (starrfish3_sec2.get_cre_expression() > 0).groupby(starrfish3_sec2.get_celltypes()).sum() < infected_cells_threshold
to_filter[cre_blacklist] = True
to_filter_sec1[cre_blacklist] = True
to_filter_sec2[cre_blacklist] = True
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
res_compare = plot_q_value_celltype_reproducibility(q_sec1, q_sec2, q)






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




# %% run the fold change test for T7
fold_change_test_config = {"cell_types_to_use": None,
                           "normalize_by_cell_rna": False,
                           "normalize_by_cell_volume": False,
                           "normalize_by_cell_t7": False, # normalize by T7
                           'filter_by_cell_t7': None,
                           "normalize_by_celltype_rna": False,
                           "normalize_by_celltype_volume": False,
                           "normalize_by_celltype_t7": True, # normalize by T7
                           "normalize_by_negative_control": False, # normalize by negative control
                           'normalize_by_total_cre': False,
                           "normalize_by_infected_cell": False,
                           "normalize_by_libsize": False,
                           "log_transform": False,
                           "filter_zero_counts": False,
                           "bootstrap_number": 10000,
                           "n_jobs": 196,
                           'load_stored': True,}
res1 = starrfish3_sec1.fold_change_test(**fold_change_test_config)
res2 = starrfish3_sec2.fold_change_test(**fold_change_test_config)
res = starrfish3.fold_change_test(**fold_change_test_config)





# %% check reproducibility of cell type bias
from plots import plot_q_value_cre_reproducibility
q_res1 = res1['qvalue_activity'].copy()
q_res2 = res2['qvalue_activity'].copy()
q_res = res['qvalue_activity'].copy()
# filter out to filter
infected_cells_threshold = 5
to_filter = (starrfish3.get_cre_expression() > 0).groupby(starrfish3.get_celltypes()).sum() < infected_cells_threshold
to_filter_sec1 = (starrfish3_sec1.get_cre_expression() > 0).groupby(starrfish3_sec1.get_celltypes()).sum() < infected_cells_threshold
to_filter_sec2 = (starrfish3_sec2.get_cre_expression() > 0).groupby(starrfish3_sec2.get_celltypes()).sum() < infected_cells_threshold
to_filter[cre_blacklist] = True
to_filter_sec1[cre_blacklist] = True
to_filter_sec2[cre_blacklist] = True
q_res1[to_filter_sec1] = pd.NA
q_res2[to_filter_sec2] = pd.NA
q_res[to_filter] = pd.NA
res_compare = plot_q_value_cre_reproducibility(q_res1, q_res2, q_res, 0.05)
res_compare = plot_q_value_celltype_reproducibility(q_res1, q_res2, q_res, 0.05)
# %% filter based on cell types
cell_type_counts_df = pd.DataFrame(index=starrfish3.get_celltypes().unique(), columns=['Sec1', 'Sec2', 'All'])
cell_type_counts_df['Sec1'] = starrfish3_sec1.get_celltypes().value_counts().reindex(cell_type_counts_df.index, fill_value=0)
cell_type_counts_df['Sec2'] = starrfish3_sec2.get_celltypes().value_counts().reindex(cell_type_counts_df.index, fill_value=0)
cell_type_counts_df['All'] = starrfish3.get_celltypes().value_counts().reindex(cell_type_counts_df.index, fill_value=0)
cell_type_size_thresholds = np.concatenate((np.linspace(0, 90, 10), np.linspace(100, 900, 9), np.linspace(1000, 10000, 10)))
cell_type_size_reproducity = pd.DataFrame(index=cell_type_size_thresholds, columns=['Sec1-All', 'Sec2-All', 'Sec1-Sec2', '#Sec1-All', '#Sec2-All', '#Sec1-Sec2'])
for threshold in cell_type_size_thresholds:
    cell_types_to_use = cell_type_counts_df.index[(cell_type_counts_df > threshold).all(axis=1)]
    res_compare = plot_q_value_cre_reproducibility(q_res1.loc[cell_types_to_use], q_res2.loc[cell_types_to_use], q_res.loc[cell_types_to_use], 0.05, plot=False)
    to_compare = res_compare[(res_compare[['Sec1', 'All']] > 0).all(axis=1)]
    cell_type_size_reproducity.loc[threshold, 'Sec1-All'] = to_compare['Common_sec1'].sum() / np.minimum(to_compare['Sec1'].sum(), to_compare['All'].sum())
    cell_type_size_reproducity.loc[threshold, '#Sec1-All'] = to_compare['Common_sec1'].sum()
    to_compare = res_compare[(res_compare[['Sec2', 'All']] > 0).all(axis=1)]
    cell_type_size_reproducity.loc[threshold, 'Sec2-All'] = to_compare['Common_sec2'].sum() / np.minimum(to_compare['Sec2'].sum(), to_compare['All'].sum())
    cell_type_size_reproducity.loc[threshold, '#Sec2-All'] = to_compare['Common_sec2'].sum()
    to_compare = res_compare[(res_compare[['Sec1', 'Sec2']] > 0).all(axis=1)]
    cell_type_size_reproducity.loc[threshold, 'Sec1-Sec2'] = to_compare['Common'].sum() / np.minimum(to_compare['Sec1'].sum(), to_compare['Sec2'].sum())
    cell_type_size_reproducity.loc[threshold, '#Sec1-Sec2'] = to_compare['Common'].sum()
# plot
fig, ax = plt.subplots(figsize=(8, 6))
sns.lineplot(x=cell_type_size_reproducity.index, y=cell_type_size_reproducity['Sec1-All'], ax=ax, label='Sec1-All', alpha=0.7)
sns.lineplot(x=cell_type_size_reproducity.index, y=cell_type_size_reproducity['Sec2-All'], ax=ax, label='Sec2-All', alpha=0.7)
sns.lineplot(x=cell_type_size_reproducity.index, y=cell_type_size_reproducity['Sec1-Sec2'], ax=ax, label='Sec1-Sec2', alpha=0.7)
# Determine consistent hue range for all scatter plots
hue_min = min(cell_type_size_reproducity['#Sec1-All'].min(), 
              cell_type_size_reproducity['#Sec2-All'].min(),
              cell_type_size_reproducity['#Sec1-Sec2'].min())
hue_max = max(cell_type_size_reproducity['#Sec1-All'].max(), 
              cell_type_size_reproducity['#Sec2-All'].max(),
              cell_type_size_reproducity['#Sec1-Sec2'].max())
# Create scatter plots with coolwarm colormap
scatter1 = ax.scatter(cell_type_size_reproducity.index, cell_type_size_reproducity['Sec1-All'], 
                     c=cell_type_size_reproducity['#Sec1-All'], cmap='coolwarm', 
                     vmin=hue_min, vmax=hue_max, alpha=0.7, s=50)
scatter2 = ax.scatter(cell_type_size_reproducity.index, cell_type_size_reproducity['Sec2-All'], 
                     c=cell_type_size_reproducity['#Sec2-All'], cmap='coolwarm', 
                     vmin=hue_min, vmax=hue_max, alpha=0.7, s=50)
scatter3 = ax.scatter(cell_type_size_reproducity.index, cell_type_size_reproducity['Sec1-Sec2'], 
                     c=cell_type_size_reproducity['#Sec1-Sec2'], cmap='coolwarm', 
                     vmin=hue_min, vmax=hue_max, alpha=0.7, s=50)
# Add colorbar
cbar = plt.colorbar(scatter1, ax=ax)
cbar.set_label('Count of Reproducible CRE-Cell Type pairs')
ax.set_title('Cell Type Size ~ Reproducibility')
ax.set_xlabel('Cell Type Size Threshold')
ax.set_ylabel('Reproducibility')
ax.set_xscale('log')
ax.legend()






# %% Try fisher exact test of cell type specificity
# for a particular CRE, test the proportion of infected cells / CREs and in each cell types
def fisher_exact_test_cell_type_specificity(starrfish_obj, infected_cells_threshold=5):
    cre_celltypes = (starrfish_obj.get_cre_expression() > 0).groupby(starrfish_obj.get_celltypes()).sum()
    t7_celltypes = (starrfish_obj.get_t7_expression() > 0).groupby(starrfish_obj.get_celltypes()).sum()
    cre_all = (starrfish_obj.get_cre_expression() > 0).sum()
    t7_all = (starrfish_obj.get_t7_expression() > 0).sum()
    odds_df = pd.DataFrame(index=cre_celltypes.index, columns=cre_celltypes.columns)
    p_df = pd.DataFrame(index=cre_celltypes.index, columns=cre_celltypes.columns)
    for cre in cre_celltypes.columns:
        for cell_type in starrfish_obj.get_celltypes().unique():
            contingency_table = pd.DataFrame({
                'All': [t7_all.loc[cre], t7_celltypes.loc[cell_type, cre]],
                'CellType': [cre_all.loc[cre], cre_celltypes.loc[cell_type, cre]]
            })
            odds_ratio, p_value = stats.fisher_exact(contingency_table, alternative='greater')
            odds_df.loc[cell_type, cre] = odds_ratio
            p_df.loc[cell_type, cre] = p_value
    # filter out the results
    to_filter = cre_celltypes < infected_cells_threshold
    odds_df_filter = odds_df.copy()
    p_df_filter = p_df.copy()
    odds_df_filter[to_filter] = pd.NA
    p_df_filter[to_filter] = pd.NA
    # calculate q-value
    q_df = pd.DataFrame(multitest.multipletests(p_df.values.flatten(), method='fdr_bh')[1].reshape(p_df.shape),
                        index=p_df.index, columns=p_df.columns)
    # For filtered data, only apply correction to non-NA values
    p_values_flat = p_df_filter.values.flatten()
    valid_mask = ~pd.isna(p_values_flat)
    q_values_flat = np.full_like(p_values_flat, np.nan, dtype=float)
    if valid_mask.any():
        q_values_flat[valid_mask] = multitest.multipletests(p_values_flat[valid_mask], method='fdr_bh')[1]
    q_df_filter = pd.DataFrame(q_values_flat.reshape(p_df_filter.shape),
                                index=p_df_filter.index, columns=p_df_filter.columns)
    return odds_df, q_df, p_df, odds_df_filter, q_df_filter, p_df_filter
odds_df1, q_df1, _, odds_df_filter1, q_df_filter1, _ = fisher_exact_test_cell_type_specificity(starrfish3_sec1, infected_cells_threshold=5)
odds_df2, q_df2, _, odds_df_filter2, q_df_filter2, _ = fisher_exact_test_cell_type_specificity(starrfish3_sec2, infected_cells_threshold=5)
odds_df, q_df, _, odds_df_filter, q_df_filter, _ = fisher_exact_test_cell_type_specificity(starrfish3, infected_cells_threshold=5)
# %%
res_compare = plot_q_value_cre_reproducibility(q_df1, q_df2, q_df)
res_compare = plot_q_value_celltype_reproducibility(q_df1, q_df2, q_df)
# %% filter based on cell types
cell_type_counts_df = pd.DataFrame(index=starrfish3.get_celltypes().unique(), columns=['Sec1', 'Sec2', 'All'])
cell_type_counts_df['Sec1'] = starrfish3_sec1.get_celltypes().value_counts().reindex(cell_type_counts_df.index, fill_value=0)
cell_type_counts_df['Sec2'] = starrfish3_sec2.get_celltypes().value_counts().reindex(cell_type_counts_df.index, fill_value=0)
cell_type_counts_df['All'] = starrfish3.get_celltypes().value_counts().reindex(cell_type_counts_df.index, fill_value=0)
cell_type_size_thresholds = np.concatenate((np.linspace(0, 90, 10), np.linspace(100, 900, 9), np.linspace(1000, 10000, 10)))
cell_type_size_reproducity = pd.DataFrame(index=cell_type_size_thresholds, columns=['Sec1-All', 'Sec2-All', 'Sec1-Sec2'])
for threshold in cell_type_size_thresholds:
    cell_types_to_use = cell_type_counts_df.index[(cell_type_counts_df > threshold).all(axis=1)]
    res_compare = plot_q_value_cre_reproducibility(q_df1.loc[cell_types_to_use], q_df2.loc[cell_types_to_use], q_df.loc[cell_types_to_use], 0.05, plot=False)
    to_compare = res_compare[(res_compare[['Sec1', 'All']] > 0).all(axis=1)]
    cell_type_size_reproducity.loc[threshold, 'Sec1-All'] = to_compare['Common_sec1'].sum() / np.minimum(to_compare['Sec1'].sum(), to_compare['All'].sum())
    cell_type_size_reproducity.loc[threshold, '#Sec1-All'] = to_compare['Common_sec1'].sum()
    to_compare = res_compare[(res_compare[['Sec2', 'All']] > 0).all(axis=1)]
    cell_type_size_reproducity.loc[threshold, 'Sec2-All'] = to_compare['Common_sec2'].sum() / np.minimum(to_compare['Sec2'].sum(), to_compare['All'].sum())
    cell_type_size_reproducity.loc[threshold, '#Sec2-All'] = to_compare['Common_sec2'].sum()
    to_compare = res_compare[(res_compare[['Sec1', 'Sec2']] > 0).all(axis=1)]
    cell_type_size_reproducity.loc[threshold, 'Sec1-Sec2'] = to_compare['Common'].sum() / np.minimum(to_compare['Sec1'].sum(), to_compare['Sec2'].sum())
    cell_type_size_reproducity.loc[threshold, '#Sec1-Sec2'] = to_compare['Common'].sum()
# plot
fig, ax = plt.subplots(figsize=(8, 6))
sns.lineplot(x=cell_type_size_reproducity.index, y=cell_type_size_reproducity['Sec1-All'], ax=ax, label='Sec1-All', alpha=0.7)
sns.lineplot(x=cell_type_size_reproducity.index, y=cell_type_size_reproducity['Sec2-All'], ax=ax, label='Sec2-All', alpha=0.7)
sns.lineplot(x=cell_type_size_reproducity.index, y=cell_type_size_reproducity['Sec1-Sec2'], ax=ax, label='Sec1-Sec2', alpha=0.7)
# Determine consistent hue range for all scatter plots
hue_min = min(cell_type_size_reproducity['#Sec1-All'].min(), 
              cell_type_size_reproducity['#Sec2-All'].min(),
              cell_type_size_reproducity['#Sec1-Sec2'].min())
hue_max = max(cell_type_size_reproducity['#Sec1-All'].max(), 
              cell_type_size_reproducity['#Sec2-All'].max(),
              cell_type_size_reproducity['#Sec1-Sec2'].max())
# Create scatter plots with coolwarm colormap
scatter1 = ax.scatter(cell_type_size_reproducity.index, cell_type_size_reproducity['Sec1-All'], 
                     c=cell_type_size_reproducity['#Sec1-All'], cmap='coolwarm', 
                     vmin=hue_min, vmax=hue_max, alpha=0.7, s=50)
scatter2 = ax.scatter(cell_type_size_reproducity.index, cell_type_size_reproducity['Sec2-All'], 
                     c=cell_type_size_reproducity['#Sec2-All'], cmap='coolwarm', 
                     vmin=hue_min, vmax=hue_max, alpha=0.7, s=50)
scatter3 = ax.scatter(cell_type_size_reproducity.index, cell_type_size_reproducity['Sec1-Sec2'], 
                     c=cell_type_size_reproducity['#Sec1-Sec2'], cmap='coolwarm', 
                     vmin=hue_min, vmax=hue_max, alpha=0.7, s=50)
# Add colorbar
cbar = plt.colorbar(scatter1, ax=ax)
cbar.set_label('Count of Reproducible CRE-Cell Type pairs')
ax.set_title('Cell Type Size ~ Reproducibility')
ax.set_xlabel('Cell Type Size Threshold')
ax.set_ylabel('Reproducibility')
ax.set_xscale('log')
ax.legend()





# %% dot plot of some biology
from plots import cre_pval_dotplot, celltype_pval_dotplot
cell_types_to_use = starrfish3.get_celltypes().value_counts().index[starrfish3.get_celltypes().value_counts()>=1000]
infected_cells_threshold = 5
to_filter = (starrfish3.get_cre_expression() > 0).groupby(starrfish3.get_celltypes()).sum() < infected_cells_threshold
to_filter[cre_blacklist] = True




# %% first the average fold change test
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
    'n_jobs': 128,
}
threshold = 'neg_control_mean'
res = starrfish3.average_bootstrap_test(**average_bootstrap_test_config)
res_q, res_df = starrfish3.average_bootstrap_test_q(res, threshold=threshold, norm='T7', tail='both', to_filter=to_filter)
cres_to_use = res_q.columns[np.nanmin(res_q.loc[cell_types_to_use], axis=0) < 0.05].union(starrfish3.get_negative_control_cres())
cres_to_use = cres_to_use[~cres_to_use.isin(cre_blacklist)]
cre_info = starrfish3.get_creinfo().copy()
cre_info['best_subclass'] = 'CRE'
cre_info.loc[negative_control_cres, 'best_subclass'] = 'Negative Control'
# %%
fig, final_order = celltype_pval_dotplot(res_q, res_df, cres_to_use, cell_types_to_use,
                                         positive_control_info=cre_info, significant_cutoff=0.05, z_norm=False,
                                         figsize=(15, 40))
fig
# %% check the global effect
global_cre_t7 = (starrfish3.get_cre_expression().sum(axis=0) / starrfish3.get_t7_expression().sum(axis=0)).loc[cres_to_use]
global_cre_t7 = global_cre_t7.sort_values()
fix, ax = plt.subplots(figsize=(5, 5))
sns.barplot(np.log(global_cre_t7), ax=ax)
# flip axis
ax.set_ylabel('Log2(Fold Change)')
ax.set_xlabel('CREs')
ax.set_xticks([])
ax.set_title('Global CRE/T7 Ratio')
# %% plot the other two sections
res1 = starrfish3_sec1.average_bootstrap_test(**average_bootstrap_test_config)
res_q1, res_df1 = starrfish3_sec1.average_bootstrap_test_q(res1, threshold=threshold, norm='T7', tail='both', to_filter=to_filter_sec1)
res2 = starrfish3_sec2.average_bootstrap_test(**average_bootstrap_test_config)
res_q2, res_df2 = starrfish3_sec2.average_bootstrap_test_q(res2, threshold=threshold, norm='T7', tail='both', to_filter=to_filter_sec2)
cre_info = starrfish3.get_creinfo().copy()
cre_info['best_subclass'] = 'CRE'
cre_info.loc[negative_control_cres, 'best_subclass'] = 'Negative Control'
# %%
fig, final_order = celltype_pval_dotplot(res_q1, res_df1, pd.Index(final_order), cell_types_to_use,
                                         positive_control_info=cre_info, significant_cutoff=0.05, 
                                         z_norm=False, reorder_cres=False, figsize=(15, 40))
fig
# %%
fig, final_order = celltype_pval_dotplot(res_q2, res_df2, pd.Index(final_order), cell_types_to_use,
                                         positive_control_info=cre_info, significant_cutoff=0.05, 
                                         z_norm=False, reorder_cres=False, figsize=(15, 40))
fig
# %%
fig, ax = plt.subplots(ncols=3, figsize=(18, 6))
sns.scatterplot(y=res_df.apply(np.nanmean, axis=0), x=starrfish3.get_t7_expression()[starrfish3.get_celltypes().isin(cell_types_to_use)].sum(), ax=ax[0])
ax[0].set_ylabel('Mean CRE/T7 fold change across cell types')
ax[0].set_xlabel('Total T7 expression')
ax[0].set_xscale('log')
ax[0].set_title('CRE/T7 ~ T7 for each CRE')
sns.scatterplot(y=res_df.apply(np.nanmean, axis=0), x=starrfish3.get_cre_expression()[starrfish3.get_celltypes().isin(cell_types_to_use)].sum(), ax=ax[1])
ax[1].set_ylabel('Mean CRE/T7 fold change across cell types')
ax[1].set_xlabel('Total CRE expression')
ax[1].set_xscale('log')
ax[1].set_title('CRE/T7 ~ CRE for each CRE')
sns.scatterplot(y=starrfish3.get_cre_expression()[starrfish3.get_celltypes().isin(cell_types_to_use)].sum(), x=starrfish3.get_t7_expression()[starrfish3.get_celltypes().isin(cell_types_to_use)].sum(), ax=ax[2])
ax[2].set_ylabel('Total CRE expression')
ax[2].set_xlabel('Total T7 expression')
ax[2].set_xscale('log')
ax[2].set_yscale('log')
ax[2].set_title('T7 ~ CRE for each CRE')


# %% next fisher exact test
odd, p = fisher_exact_test(starrfish3, baseline = 'neg_control')
p[to_filter], odd[to_filter] = np.nan, np.nan
# %% transform to q value
q = p.copy()
q[to_filter] = np.nan
q = q.values.flatten().copy()
q[~np.isnan(q)] = multitest.multipletests(q[~np.isnan(q)], method='fdr_bh')[1]
q = pd.DataFrame(q.reshape(p.shape), index=p.index, columns=p.columns)
q_toplot = q.copy()
q_toplot[to_filter] = np.nan
res_toplot = odd.copy()
res_toplot[to_filter] = np.nan
cres_to_use = q_toplot.columns[np.nanmin(q_toplot.loc[cell_types_to_use], axis=0) < 0.05].union(starrfish3.get_negative_control_cres()[~pd.isna(q_toplot[starrfish3.get_negative_control_cres()]).all()])
cre_info = starrfish3.get_creinfo().copy()
cre_info['best_subclass'] = 'CRE'
cre_info.loc[negative_control_cres, 'best_subclass'] = 'Negative Control'
fig = celltype_pval_dotplot(q_toplot, res_toplot, cres_to_use, cell_types_to_use,
                            positive_control_info=cre_info, significant_cutoff=0.05, z_norm=False, activity_log=True,
                            figsize=(15, 30))
fig




# %% Next chapter, do precision analysis
from plots import get_pr_df, plot_bar
# %% get precision for ATAC and histone modifications
starrfish3.load_cpm('Data/ATAC_cpm_peakBysubclass.csv', attr_to_add='atac_cpm')
starrfish3.load_cpm('Data/H3K4me1_cpm_peak_pad_500_Bysubclass.csv', attr_to_add='h3k4me1_cpm')
starrfish3.load_cpm('Data/H3K9me3_cpm_peak_pad_500_Bysubclass.csv', attr_to_add='h3k9me3_cpm')
starrfish3.load_cpm('Data/H3K27ac_cpm_peak_pad_500_Bysubclass.csv', attr_to_add='h3k27ac_cpm')
starrfish3.load_cpm('Data/H3K27me3_cpm_peak_pad_500_Bysubclass.csv', attr_to_add='h3k27me3_cpm')
# add chromatin state data
chromatin_o = pd.read_csv('Data/cre_chromatin_state_o.csv', index_col=0)
chromatin_a = pd.read_csv('Data/cre_chromatin_state_a.csv', index_col=0)
starrfish3.chromatin_o = (chromatin_o.copy() + chromatin_a.copy()) / 2
starrfish3.chromatin_a = chromatin_a.copy()
# %% fold change test, cell type specificity
fold_change_test_config = {"cell_types_to_use": None,
                           "normalize_by_cell_rna": False,
                           "normalize_by_cell_volume": False,
                           "normalize_by_cell_t7": False, # normalize by T7
                           'filter_by_cell_t7': None,
                           "normalize_by_celltype_rna": False,
                           "normalize_by_celltype_volume": False,
                           "normalize_by_celltype_t7": True, # normalize by T7
                           "normalize_by_negative_control": False, # normalize by negative control
                           'normalize_by_total_cre': False,
                           "normalize_by_infected_cell": False,
                           "normalize_by_libsize": False,
                           "log_transform": False,
                           "filter_zero_counts": False,
                           "bootstrap_number": 10000,
                           "n_jobs": 196,
                           'load_stored': True,}
res = starrfish3.fold_change_test(**fold_change_test_config)
p = res['pvalue_activity'].copy()
q = p.copy()
q[to_filter] = np.nan
q = q.values.flatten().copy().astype(float)
q[~np.isnan(q)] = multitest.multipletests(q[~np.isnan(q)], method='fdr_bh')[1]
q = pd.DataFrame(q.reshape(p.shape), index=p.index, columns=p.columns)
q_toplot = q.copy()
res_toplot = res['celltype_activity'].copy()
cell_types_to_use = starrfish3.get_celltypes().value_counts().index[starrfish3.get_celltypes().value_counts()>=1000]
q_toplot[to_filter] = np.nan
res_toplot[to_filter] = np.nan
cres_to_use = q_toplot.columns[np.nanmin(q_toplot.loc[cell_types_to_use], axis=0) < 0.05].union(starrfish3.get_negative_control_cres())
cres_to_use = cres_to_use[~cres_to_use.isin(cre_blacklist)]
cre_info = starrfish3.get_creinfo().copy()
cre_info['best_subclass'] = 'CRE'
cre_info.loc[negative_control_cres, 'best_subclass'] = 'Negative Control'
fig, cre_orders = cre_pval_dotplot(q_toplot, res_toplot, cres_to_use, cell_types_to_use, cre_info, figsize=(15, 15))
fig
# %% plot sec1
res = starrfish3_sec1.fold_change_test(**fold_change_test_config)
p = res['pvalue_activity'].copy()
q = p.copy()
q[to_filter] = np.nan
q = q.values.flatten().copy().astype(float)
q[~np.isnan(q)] = multitest.multipletests(q[~np.isnan(q)], method='fdr_bh')[1]
q = pd.DataFrame(q.reshape(p.shape), index=p.index, columns=p.columns)
q_toplot = q.copy()
res_toplot = res['celltype_activity'].copy()
cell_types_to_use = starrfish3.get_celltypes().value_counts().index[starrfish3.get_celltypes().value_counts()>=1000]
q_toplot[to_filter] = np.nan
res_toplot[to_filter] = np.nan
cre_info = starrfish3.get_creinfo().copy()
cre_info['best_subclass'] = 'CRE'
cre_info.loc[negative_control_cres, 'best_subclass'] = 'Negative Control'
fig, cre_orders = cre_pval_dotplot(q_toplot, res_toplot, pd.Index(cre_orders), cell_types_to_use, cre_info, reorder_cres=False, figsize=(15, 15))
fig
# %% plot sec2
res = starrfish3_sec2.fold_change_test(**fold_change_test_config)
p = res['pvalue_activity'].copy()
q = p.copy()
q[to_filter] = np.nan
q = q.values.flatten().copy().astype(float)
q[~np.isnan(q)] = multitest.multipletests(q[~np.isnan(q)], method='fdr_bh')[1]
q = pd.DataFrame(q.reshape(p.shape), index=p.index, columns=p.columns)
q_toplot = q.copy()
res_toplot = res['celltype_activity'].copy()
cell_types_to_use = starrfish3.get_celltypes().value_counts().index[starrfish3.get_celltypes().value_counts()>=1000]
q_toplot[to_filter] = np.nan
res_toplot[to_filter] = np.nan
cre_info = starrfish3.get_creinfo().copy()
cre_info['best_subclass'] = 'CRE'
cre_info.loc[negative_control_cres, 'best_subclass'] = 'Negative Control'
fig, cre_orders = cre_pval_dotplot(q_toplot, res_toplot, pd.Index(cre_orders), cell_types_to_use, cre_info, reorder_cres=False, figsize=(15, 15))
fig



# %% get precision recall
pr_df1 = get_pr_df(qvalue_df=q_toplot.loc[cell_types_to_use].copy(), cell_types_to_use=cell_types_to_use,
                   starrfish_obj=starrfish3, metric = ['atac_cpm', 'h3k4me1_cpm', 'h3k27ac_cpm'], z_cutoffs=[2.0])
pr_df2 = get_pr_df(qvalue_df=q_toplot.loc[cell_types_to_use].copy(), cell_types_to_use=cell_types_to_use,
                   starrfish_obj=starrfish3, 
                   metric=['chromatin_o', 'chromatin_a'], z_cutoffs=[0.5])
# pr_df3 = get_pr_df(qvalue_df=q_toplot.loc[cell_types_to_use_nc_2].copy(), cell_types_to_use=cell_types_to_use,
#                    starrfish_obj=starrfish3, 
#                    metric=['snapatac2_de_fc'], z_cutoffs=[2])
# pr_df2 = pd.concat((pr_df2, pr_df3), axis=0, ignore_index=True)
pr_df2 = pr_df2.sort_values(by=['cell_type_rank']).reset_index(drop=True)
pr_df1 = pr_df1[pr_df1['cell_type'].isin(pr_df2['cell_type'])].copy()
pr_df2 = pr_df2[pr_df2['cell_type'].isin(pr_df1['cell_type'])].copy()
df_bar = pr_df1[(pr_df1['z_cutoff'] == 2.0)].copy()
# add a column for overall precision
df_bar_all1 = pd.DataFrame({'cell_type': 'ALL',
    'precision': df_bar.groupby(['mod'])['correct'].sum() / df_bar.groupby(['mod'])['all_pred'].sum(),
    'correct': df_bar.groupby(['mod'])['correct'].sum(),
    'all_pred': df_bar.groupby(['mod'])['all_pred'].sum(),
    'target': df_bar.groupby(['mod'])['target'].sum(),
})
df_bar_all1['recall'] = df_bar_all1['correct'].astype(str) + '/' + df_bar_all1['all_pred'].astype(str)
df_bar_all1['mod'] = df_bar_all1.index
df_bar = df_bar[df_bar['target'] >= 2].copy()
fig, ax = plot_bar(df_bar, legend_loc=(0.95, 0.75), figsize=(6, 6), flip_axis=True, fontsize=6)
df_bar = pr_df2.copy()
df_bar_all2 = pd.DataFrame({'cell_type': 'ALL',
    'precision': df_bar.groupby(['mod'])['correct'].sum() / df_bar.groupby(['mod'])['all_pred'].sum(),
    'correct': df_bar.groupby(['mod'])['correct'].sum(),
    'all_pred': df_bar.groupby(['mod'])['all_pred'].sum(),
    'target': df_bar.groupby(['mod'])['target'].sum(),
})
df_bar_all2['recall'] = df_bar_all2['correct'].astype(str) + '/' + df_bar_all2['all_pred'].astype(str)
df_bar = df_bar[df_bar['target'] >= 2].copy()
df_bar_all2['mod'] = df_bar_all2.index
fig, ax = plot_bar(df_bar, figsize=(6, 6), flip_axis=True, fontsize=6)
# ALL cell type
df_bar_all = pd.concat([df_bar_all1, df_bar_all2], axis=0, ignore_index=True)
fig, ax = plot_bar(df_bar_all, figsize=(6, 6), flip_axis=True, fontsize=6)





# %% visualize some example
have_target_cres = q_toplot.columns[np.nanmin(q_toplot.loc[cell_types_to_use], axis=0) < 0.05]
for cre in have_target_cres:
    # rank by q-value
    cre_q_values = q_toplot.loc[cell_types_to_use, cre]
    cre_q_values = cre_q_values[cre_q_values <= 0.05] 
    # order by rank
    cre_q_values = cre_q_values.sort_values(ascending=True)
    cell_types_to_visualize = cre_q_values.index
    fig = starrfish3.plot_gene(
        cre, average_by_celltype=False, # if true, all cells from same cell type will have same value
        cell_types_to_visualize=cell_types_to_visualize, # only visualize some cell types
        scale_size_by='counts', # scale size by "counts": normalized counts; or "celltype_number": number of cells in the cell type
        log=True, transpose=-1, flipx=-1, sz_max=50,
        cell_types_to_use=cell_types_to_use)










# %% fisher test, cell type specificity
odds_df, _, p_df, _, _, _ = fisher_exact_test_cell_type_specificity(starrfish3, infected_cells_threshold=5)
# %%
q = p_df.copy()
q[to_filter] = np.nan
q = q.values.flatten().copy().astype(float)
q[~np.isnan(q)] = multitest.multipletests(q[~np.isnan(q)], method='fdr_bh')[1]
q = pd.DataFrame(q.reshape(p_df.shape), index=p_df.index, columns=p_df.columns)
q_toplot = q.copy()
res_toplot = odds_df.copy()
cell_types_to_use = starrfish3.get_celltypes().value_counts().index[starrfish3.get_celltypes().value_counts()>=1000]
infected_cells_threshold = 5
to_filter = (starrfish3.get_cre_expression() > 0).groupby(starrfish3.get_celltypes()).sum() < infected_cells_threshold
to_filter[cre_blacklist] = True
q_toplot[to_filter] = np.nan
res_toplot[to_filter] = np.nan
cres_to_use = q_toplot.columns[np.nanmin(q_toplot.loc[cell_types_to_use], axis=0) < 0.05].union(starrfish3.get_negative_control_cres())
cre_info = starrfish3.get_creinfo().copy()
cre_info['best_subclass'] = 'CRE'
cre_info.loc[negative_control_cres, 'best_subclass'] = 'Negative Control'
fig = cre_pval_dotplot(q_toplot, res_toplot, cres_to_use, cell_types_to_use, cre_info, figsize=(15, 30))
fig






# %% get precision recall
pr_df1 = get_pr_df(qvalue_df=q_toplot.loc[cell_types_to_use].copy(), cell_types_to_use=cell_types_to_use,
                   starrfish_obj=starrfish3, metric = ['atac_cpm', 'h3k4me1_cpm', 'h3k27ac_cpm'], z_cutoffs=[2.0])
pr_df2 = get_pr_df(qvalue_df=q_toplot.loc[cell_types_to_use].copy(), cell_types_to_use=cell_types_to_use,
                   starrfish_obj=starrfish3, 
                   metric=['chromatin_o', 'chromatin_a'], z_cutoffs=[0.5])
# pr_df3 = get_pr_df(qvalue_df=q_toplot.loc[cell_types_to_use_nc_2].copy(), cell_types_to_use=cell_types_to_use,
#                    starrfish_obj=starrfish3, 
#                    metric=['snapatac2_de_fc'], z_cutoffs=[2])
# pr_df2 = pd.concat((pr_df2, pr_df3), axis=0, ignore_index=True)
pr_df2 = pr_df2.sort_values(by=['cell_type_rank']).reset_index(drop=True)
pr_df1 = pr_df1[pr_df1['cell_type'].isin(pr_df2['cell_type'])].copy()
pr_df2 = pr_df2[pr_df2['cell_type'].isin(pr_df1['cell_type'])].copy()
df_bar = pr_df1[(pr_df1['z_cutoff'] == 2.0)].copy()
# add a column for overall precision
df_bar_all1 = pd.DataFrame({'cell_type': 'ALL',
    'precision': df_bar.groupby(['mod'])['correct'].sum() / df_bar.groupby(['mod'])['all_pred'].sum(),
    'correct': df_bar.groupby(['mod'])['correct'].sum(),
    'all_pred': df_bar.groupby(['mod'])['all_pred'].sum(),
    'target': df_bar.groupby(['mod'])['target'].sum(),
})
df_bar_all1['recall'] = df_bar_all1['correct'].astype(str) + '/' + df_bar_all1['all_pred'].astype(str)
df_bar_all1['mod'] = df_bar_all1.index
df_bar = df_bar[df_bar['target'] >= 2].copy()
fig, ax = plot_bar(df_bar, legend_loc=(0.95, 0.75), figsize=(6, 6), flip_axis=True, fontsize=6)
df_bar = pr_df2.copy()
df_bar_all2 = pd.DataFrame({'cell_type': 'ALL',
    'precision': df_bar.groupby(['mod'])['correct'].sum() / df_bar.groupby(['mod'])['all_pred'].sum(),
    'correct': df_bar.groupby(['mod'])['correct'].sum(),
    'all_pred': df_bar.groupby(['mod'])['all_pred'].sum(),
    'target': df_bar.groupby(['mod'])['target'].sum(),
})
df_bar_all2['recall'] = df_bar_all2['correct'].astype(str) + '/' + df_bar_all2['all_pred'].astype(str)
df_bar = df_bar[df_bar['target'] >= 2].copy()
df_bar_all2['mod'] = df_bar_all2.index
fig, ax = plot_bar(df_bar, figsize=(6, 6), flip_axis=True, fontsize=6)
# ALL cell type
df_bar_all = pd.concat([df_bar_all1, df_bar_all2], axis=0, ignore_index=True)
fig, ax = plot_bar(df_bar_all, figsize=(6, 6), flip_axis=True, fontsize=6)





# %% save the results
starrfish3_sec1.save('results/starrfish3_sec1.pkl')
starrfish3_sec2.save('results/starrfish3_sec2.pkl')
starrfish3.save('results/starrfish3.pkl')


