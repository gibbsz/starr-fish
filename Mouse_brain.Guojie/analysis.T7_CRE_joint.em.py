# %% Fixed version with numerical stability
import torch
import torch.nn as nn
import torch.optim as optim
import time
import numpy as np
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
from scipy.optimize import minimize
from scipy.special import logsumexp
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
starrfish3_sec1 = STARRFISH.load('results/starrfish3_sec1.pkl')
starrfish3_sec2 = STARRFISH.load('results/starrfish3_sec2.pkl')
# %%
from utils import T7CRE_Joint_DistributionEM
# %% for cre in t7_counts.columns:
cell_counts_sec1 = starrfish3_sec1.get_celltypes().value_counts()
cell_counts_sec2 = starrfish3_sec2.get_celltypes().value_counts()
common_celltypes = cell_counts_sec1.index[cell_counts_sec1 >= 1000].intersection(cell_counts_sec2.index[cell_counts_sec2 >= 1000])

cre_blacklist = ['CRE061', 'CRE143', 'CRE001']
mismatching_cres = pd.read_csv('Data/AAV_ONT_Barcode_Counts_vs_Mismatch_Percentage.csv', index_col=0)
cre_blacklist = np.unique(cre_blacklist + mismatching_cres.index[mismatching_cres['MismatchPercent'] > 20].tolist()).tolist()

common_cres = starrfish3_sec1.lib_size.index[starrfish3_sec1.lib_size['counts'] >= 6]
common_cres = common_cres[~common_cres.isin(cre_blacklist)]

t7_counts_sec1 = starrfish3_sec1.get_t7_expression()[starrfish3_sec1.get_celltypes().isin(common_celltypes)][common_cres]
t7_counts_sec2 = starrfish3_sec2.get_t7_expression()[starrfish3_sec2.get_celltypes().isin(common_celltypes)][common_cres]

cre_counts_sec1 = starrfish3_sec1.get_cre_expression().loc[t7_counts_sec1.index, common_cres]
cre_counts_sec2 = starrfish3_sec2.get_cre_expression().loc[t7_counts_sec2.index, common_cres]

celltypes_sec1 = starrfish3_sec1.get_celltypes().loc[t7_counts_sec1.index]
celltypes_sec2 = starrfish3_sec2.get_celltypes().loc[t7_counts_sec2.index]
# %%
em_model = T7CRE_Joint_DistributionEM(device='cuda', use_x0=True)
x0_sec1, x1_sec1, x2_sec1 = em_model.fit(celltypes_sec1.values, t7_counts_sec1, cre_counts_sec1, dim=40, max_iter=1000, x0_prior='zero_percentage',
                                         x0_checkpoint_path=f'{PWD}/results/expr3/t7cre_em.joint.x0.sec1.npy',
                                         x1_checkpoint_path=f'{PWD}/results/expr3/t7cre_em.joint.x1.sec1.npy',
                                         x2_checkpoint_path=f'{PWD}/results/expr3/t7cre_em.joint.x2.sec1.npy')
x0_sec2, x1_sec2, x2_sec2 = em_model.fit(celltypes_sec2.values, t7_counts_sec2, cre_counts_sec2, dim=40, max_iter=1000, x0_prior='zero_percentage',
                                         x0_checkpoint_path=f'{PWD}/results/expr3/t7cre_em.joint.x0.sec2.npy',
                                         x1_checkpoint_path=f'{PWD}/results/expr3/t7cre_em.joint.x1.sec2.npy',
                                         x2_checkpoint_path=f'{PWD}/results/expr3/t7cre_em.joint.x2.sec2.npy')
# %%
np.save(f'{PWD}/results/expr3/t7cre_em.joint.x0.sec1.npy', x0_sec1)
np.save(f'{PWD}/results/expr3/t7cre_em.joint.x0.sec2.npy', x0_sec2)
np.save(f'{PWD}/results/expr3/t7cre_em.joint.x1.sec1.npy', x1_sec1)
np.save(f'{PWD}/results/expr3/t7cre_em.joint.x1.sec2.npy', x1_sec2)
np.save(f'{PWD}/results/expr3/t7cre_em.joint.x2.sec1.npy', x2_sec1)
np.save(f'{PWD}/results/expr3/t7cre_em.joint.x2.sec2.npy', x2_sec2)
# %%
x0_sec1 = np.load(f'{PWD}/results/expr3/t7cre_em.joint.x0.sec1.npy')
x0_sec2 = np.load(f'{PWD}/results/expr3/t7cre_em.joint.x0.sec2.npy')
x1_sec1 = np.load(f'{PWD}/results/expr3/t7cre_em.joint.x1.sec1.npy')
x1_sec2 = np.load(f'{PWD}/results/expr3/t7cre_em.joint.x1.sec2.npy')
x2_sec1 = np.load(f'{PWD}/results/expr3/t7cre_em.joint.x2.sec1.npy')
x2_sec2 = np.load(f'{PWD}/results/expr3/t7cre_em.joint.x2.sec2.npy')
# %%
# check infection rate differences
x0_df_sec1 = pd.DataFrame(torch.nn.functional.softplus(torch.from_numpy(x0_sec1)), index=common_celltypes, columns=t7_counts_sec1.columns)
x0_df_sec2 = pd.DataFrame(torch.nn.functional.softplus(torch.from_numpy(x0_sec2)), index=common_celltypes, columns=t7_counts_sec2.columns)
x1_r_sec1 = torch.nn.functional.softplus(torch.tensor(x1_sec1[0]))
x1_p_sec1 = torch.nn.functional.sigmoid(torch.tensor(x1_sec1[1]))
x1_r_sec2 = torch.nn.functional.softplus(torch.tensor(x1_sec2[0]))
x1_p_sec2 = torch.nn.functional.sigmoid(torch.tensor(x1_sec2[1]))
x2_r_df_sec1 = pd.DataFrame(torch.nn.functional.softplus(torch.from_numpy(x2_sec1[:, :, 0])), index=common_celltypes, columns=t7_counts_sec1.columns)
x2_r_df_sec2 = pd.DataFrame(torch.nn.functional.softplus(torch.from_numpy(x2_sec2[:, :, 0])), index=common_celltypes, columns=t7_counts_sec2.columns)
x2_p_df_sec1 = pd.DataFrame(torch.nn.functional.sigmoid(torch.from_numpy(x2_sec1[:, :, 1])), index=common_celltypes, columns=t7_counts_sec1.columns)
x2_p_df_sec2 = pd.DataFrame(torch.nn.functional.sigmoid(torch.from_numpy(x2_sec2[:, :, 1])), index=common_celltypes, columns=t7_counts_sec2.columns)
# %%
x1_mean_sec1 = x1_r_sec1 * (1-x1_p_sec1) / x1_p_sec1
x1_mean_sec2 = x1_r_sec2 * (1-x1_p_sec2) / x1_p_sec2
x1_var_sec1 = x1_r_sec1 * (1-x1_p_sec1) / (x1_p_sec1**2)
x1_var_sec2 = x1_r_sec2 * (1-x1_p_sec2) / (x1_p_sec2**2)
x2_mean_df_sec1 = x2_r_df_sec1 * (1-x2_p_df_sec1) / x2_p_df_sec1
x2_mean_df_sec2 = x2_r_df_sec2 * (1-x2_p_df_sec2) / x2_p_df_sec2
x2_var_df_sec1 = x2_r_df_sec1 * (1-x2_p_df_sec1) / (x2_p_df_sec1**2)
x2_var_df_sec2 = x2_r_df_sec2 * (1-x2_p_df_sec2) / (x2_p_df_sec2**2)
x2_log_mean_df_sec1 = np.log(x2_mean_df_sec1)
x2_log_mean_df_sec2 = np.log(x2_mean_df_sec2)
x2_log_var_df_sec1 = np.log(x2_var_df_sec1)
x2_log_var_df_sec2 = np.log(x2_var_df_sec2)


# %% filter by cell number ≥ 5
infected_cells_threshold = 5
to_filter_sec1 = (starrfish3_sec1.get_cre_expression() > 0).groupby(starrfish3_sec1.get_celltypes()).sum() < infected_cells_threshold
to_filter_sec2 = (starrfish3_sec2.get_cre_expression() > 0).groupby(starrfish3_sec2.get_celltypes()).sum() < infected_cells_threshold
to_filter_sec1[cre_blacklist] = True
to_filter_sec2[cre_blacklist] = True



# %% do cre and cell type corr
x2_log_mean_df_sec1[to_filter_sec1] = np.nan
x2_log_mean_df_sec2[to_filter_sec2] = np.nan
cre_corr, celltype_corr = starrfish3_sec1.corr_starrfish(x2_log_mean_df_sec1, x2_log_mean_df_sec2)



# %%
cres_to_use = cre_corr[cre_corr['pearson_p'] <= 0.05].index
cre_corr, celltype_corr = starrfish3_sec1.corr_atac_cpm(acvitity_df=x2_log_mean_df_sec1[cres_to_use], cell_types_to_use=x2_log_mean_df_sec1.index)
sum(cre_corr['spearman_p'] <= 0.05), sum(celltype_corr['pearson_p'] <= 0.05)
cres_to_use = cre_corr[cre_corr['spearman_p'] <= 0.05].index



# %% visualize best cell type-wise corr
fig, ax = plt.subplots(figsize=(4, 4))
cre = 'CRE132'
sns.scatterplot(x=x2_log_mean_df_sec1[cre], y=starrfish3_sec1.atac_cpm[cre], color='blue', ax=ax)
# plot negative control
sns.scatterplot(x=x2_log_mean_df_sec1.loc[starrfish3_sec1.get_positive_control_celltypes(cre, use='atac-peak').intersection(x2_log_mean_df_sec1.index), cre],
                y=starrfish3_sec1.atac_cpm.loc[starrfish3_sec1.get_positive_control_celltypes(cre, use='atac-peak').intersection(x2_log_mean_df_sec2.index), cre], color='red', ax=ax)
ax.set_xlabel('Activity Sec1 (log)')
ax.set_ylabel('ATAC cpm')



# %%
cre_corr['libsize'] = starrfish3_sec1.lib_size['counts'].loc[cre_corr.index]
celltype_corr['celltype_sec1'] = starrfish3_sec1.get_celltypes().value_counts().loc[celltype_corr.index].values
celltype_corr['celltype_sec2'] = starrfish3_sec2.get_celltypes().value_counts().loc[celltype_corr.index].values
celltype_corr['celltype_n'] = np.minimum(celltype_corr['celltype_sec1'], celltype_corr['celltype_sec2'])
# %% plot
n_cre_threshold = 10
n_celltype_threshold = 10
fig, ax = plt.subplots(figsize=(4, 4))
sns.scatterplot(data=celltype_corr[(celltype_corr['pearson_p'] > 0.05) & (celltype_corr['effect_n'] >= n_cre_threshold)], x='celltype_n', y='pearson', color='blue', ax=ax)
sns.scatterplot(data=celltype_corr[(celltype_corr['pearson_p'] <= 0.05) & (celltype_corr['effect_n'] >= n_cre_threshold)], x='celltype_n', y='pearson', color='red', ax=ax)
ax.set_xscale('log')
fig, ax = plt.subplots(figsize=(4, 4))
sns.scatterplot(data=cre_corr[(cre_corr['pearson_p'] > 0.05) & (cre_corr['effect_n'] >= n_celltype_threshold)], x='libsize', y='pearson', color='blue', ax=ax)
sns.scatterplot(data=cre_corr[(cre_corr['pearson_p'] <= 0.05) & (cre_corr['effect_n'] >= n_celltype_threshold)], x='libsize', y='pearson', color='red', ax=ax)




# %% visualize best cell type-wise corr
fig, ax = plt.subplots(figsize=(4, 4))
celltype = 'OB Eomes Ms4a15 Glut'
sns.scatterplot(x=x2_log_mean_df_sec1.loc[celltype], y=x2_log_mean_df_sec2.loc[celltype], color='blue', ax=ax)
# plot negative control
sns.scatterplot(x=x2_log_mean_df_sec1.loc[celltype, x2_log_mean_df_sec1.columns.intersection(starrfish3_sec1.get_negative_control_cres())],
                y=x2_log_mean_df_sec2.loc[celltype, x2_log_mean_df_sec1.columns.intersection(starrfish3_sec1.get_negative_control_cres())], color='orange', ax=ax)
sns.scatterplot(x=x2_log_mean_df_sec1.loc[celltype, x2_log_mean_df_sec1.columns.intersection(starrfish3_sec1.get_positive_control_cres(celltype, use='atac-peak'))],
                y=x2_log_mean_df_sec2.loc[celltype, x2_log_mean_df_sec1.columns.intersection(starrfish3_sec1.get_positive_control_cres(celltype, use='atac-peak'))], color='red', ax=ax)
sns.scatterplot(x=x2_log_mean_df_sec1.loc[celltype, x2_log_mean_df_sec1.columns.intersection(cre_blacklist)],
                y=x2_log_mean_df_sec2.loc[celltype, x2_log_mean_df_sec1.columns.intersection(cre_blacklist)], color='green', ax=ax)
ax.set_xlabel('Activity Sec1 (log)')
ax.set_ylabel('Activity Sec2 (log)')
# %% visualize best cell type-wise corr
fig, ax = plt.subplots(figsize=(4, 4))
sns.scatterplot(x=x2_log_mean_df_sec1.loc[celltype], y=x2_log_mean_df_sec2.loc[celltype], 
                hue=starrfish3_sec1.lib_size['counts'].loc[x2_log_mean_df_sec1.columns], ax=ax)
ax.set_xlabel('Activity Sec1 (log)')
ax.set_ylabel('Activity Sec2 (log)')
fig, ax = plt.subplots(figsize=(4, 4))
sns.scatterplot(x=x2_log_mean_df_sec1.loc[celltype], 
                y=starrfish3_sec1.lib_size['counts'].loc[x2_log_mean_df_sec1.columns], ax=ax)
ax.set_xlabel('Activity Sec1 (log)')
ax.set_ylabel('Library Size (counts)')

# %% take out the CREs that didn't align well
cres_mismatch = x2_log_mean_df_sec1.columns[(np.abs(x2_log_mean_df_sec1.loc[celltype] - x2_log_mean_df_sec2.loc[celltype]) > 1) & (~x2_log_mean_df_sec1.loc[celltype].isna()) & (~x2_log_mean_df_sec2.loc[celltype].isna())]
cres_match = x2_log_mean_df_sec1.columns[(np.abs(x2_log_mean_df_sec1.loc[celltype] - x2_log_mean_df_sec2.loc[celltype]) <= 1) | (x2_log_mean_df_sec1.loc[celltype].isna()) | (x2_log_mean_df_sec2.loc[celltype].isna())]
# plot the variance
fig, ax = plt.subplots(figsize=(4, 4))
sns.scatterplot(x=x2_log_mean_df_sec1.loc[celltype, cres_mismatch], y=x2_log_mean_df_sec2.loc[celltype, cres_mismatch], color='red', ax=ax)
sns.scatterplot(x=x2_log_mean_df_sec1.loc[celltype, cres_match], y=x2_log_mean_df_sec2.loc[celltype, cres_match], color='blue', ax=ax)
ax.set_xlabel('Activity Sec1 (log) mean')
ax.set_ylabel('Activity Sec2 (log) mean')
# plot their variances
fig, ax = plt.subplots(ncols=2, figsize=(8, 4))
sns.scatterplot(x=x2_log_mean_df_sec1.loc[celltype, cres_mismatch], y=x2_log_var_df_sec1.loc[celltype, cres_mismatch], color='red', ax=ax[0], alpha=0.5)
sns.scatterplot(x=x2_log_mean_df_sec2.loc[celltype, cres_mismatch], y=x2_log_var_df_sec2.loc[celltype, cres_mismatch], color='red', ax=ax[1], alpha=0.5)
sns.scatterplot(x=x2_log_mean_df_sec1.loc[celltype, cres_match], y=x2_log_var_df_sec1.loc[celltype, cres_match], color='blue', ax=ax[0], alpha=0.5)
sns.scatterplot(x=x2_log_mean_df_sec2.loc[celltype, cres_match], y=x2_log_var_df_sec2.loc[celltype, cres_match], color='blue', ax=ax[1], alpha=0.5)
ax[0].set_xlabel('Activity Sec1 (log) mean')
ax[0].set_ylabel('Activity Sec1 (log) variance')
ax[1].set_xlabel('Activity Sec2 (log) mean')
ax[1].set_ylabel('Activity Sec2 (log) variance')



# %% visualize best cell type-wise corr
fig, ax = plt.subplots(figsize=(4, 4))
cre = 'CRE155'
sns.scatterplot(x=x2_log_mean_df_sec1[cre], y=x2_log_mean_df_sec2[cre], color='blue', ax=ax)
# plot negative control
sns.scatterplot(x=x2_log_mean_df_sec1.loc[starrfish3_sec1.get_positive_control_celltypes(cre, use='atac-peak').intersection(x2_log_mean_df_sec1.index), cre],
                y=x2_log_mean_df_sec2.loc[starrfish3_sec1.get_positive_control_celltypes(cre, use='atac-peak').intersection(x2_log_mean_df_sec2.index), cre], color='red', ax=ax)
ax.set_xlabel('Activity Sec1 (log)')
ax.set_ylabel('Activity Sec2 (log)')



# %% directly plot x2_log_mean_df_sec1
from plots import plot_grouped_clustermap
cre_info = starrfish3_sec1.get_creinfo().copy()
cre_info['best_subclass'] = 'CRE'
cre_info.loc[starrfish3_sec1.get_negative_control_cres(), 'best_subclass'] = 'Negative Control'

_, final_order = plot_grouped_clustermap(x2_log_mean_df_sec1, cre_info, 'Section 1', figsize=(15, 8))
_, _ = plot_grouped_clustermap(x2_log_mean_df_sec2, cre_info, 'Section 2', final_order=final_order, figsize=(15, 8))
# %% filter to reproducible results
nonreproducible = np.abs(x2_log_mean_df_sec1 - x2_log_mean_df_sec2) > 1
x2_log_mean_df_sec1_filtered = x2_log_mean_df_sec1.copy()
x2_log_mean_df_sec2_filtered = x2_log_mean_df_sec2.copy()
x2_log_mean_df_sec1_filtered[nonreproducible] = np.nan
x2_log_mean_df_sec2_filtered[nonreproducible] = np.nan
_, final_order = plot_grouped_clustermap(x2_log_mean_df_sec1_filtered, cre_info, 'Section 1 Filtered', figsize=(15, 8))
_, _ = plot_grouped_clustermap(x2_log_mean_df_sec2_filtered, cre_info, 'Section 2 Filtered', final_order=final_order, figsize=(15, 8))
# %% do cre and cell type corr
cre_corr, celltype_corr = starrfish3_sec1.corr_starrfish(x2_log_mean_df_sec1_filtered, x2_log_mean_df_sec2_filtered)
# %%
cre_corr['libsize'] = starrfish3_sec1.lib_size['counts'].loc[cre_corr.index]
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




# %% dot plot
from plots import celltype_pval_dotplot
cre_info = starrfish3_sec1.get_creinfo().copy()
for cre in cre_info.index:
    positive_cts = starrfish3_sec1.get_positive_control_celltypes(cre, use='chromatin-a')
    if positive_cts is not None:
        cre_info.loc[cre, 'best_subclass'] = ';'.join(positive_cts)
    else:
        cre_info.loc[cre, 'best_subclass'] = 'CRE'
# cre_info['best_subclass'] = 'CRE'
cre_info.loc[starrfish3_sec1.get_negative_control_cres(), 'best_subclass'] = 'Negative Control'
# design a test to compare CRE activity in each cell type to Negative Control
# res_df = x2_log_mean_df_sec1.copy()
res_df = x2_log_mean_df_sec1_filtered.copy()
res_q = pd.DataFrame(1.0, index=res_df.index, columns=res_df.columns)
# remove to_filter
res_df[cre_blacklist] = np.nan
negative_control_mean = res_df[starrfish3_sec1.get_negative_control_cres().intersection(res_df.columns)].apply(np.nanmean, axis=1)
negative_control_std = res_df[starrfish3_sec1.get_negative_control_cres().intersection(res_df.columns)].apply(np.nanstd, axis=1)
negative_control_upper = negative_control_mean + 2 * negative_control_std
negative_control_lower = negative_control_mean - 2 * negative_control_std
# for each cell type, anything between negative control upper and lower will be marked as not significant
for ct in res_df.index:
    if ct in negative_control_upper.index and ct in negative_control_lower.index:
        if np.isnan(negative_control_upper[ct]) or np.isnan(negative_control_lower[ct]):
            res_df.loc[ct, :] = np.nan
            res_q.loc[ct, :] = np.nan
        else:
            # calculate z-score and p-value of normal distribution with regard to negative control
            z_scores = (res_df.loc[ct] - negative_control_mean[ct]) / negative_control_std[ct]
            p_values = z_scores.apply(stats.norm.cdf)
            res_q.loc[ct] = np.minimum(p_values, 1-p_values)
    # remove irreproduciable results
    if ct in x2_log_mean_df_sec1.index and ct in x2_log_mean_df_sec2.index:
        irr = x2_log_mean_df_sec1.columns[np.abs(x2_log_mean_df_sec1.loc[ct] - x2_log_mean_df_sec2.loc[ct]) > 1]
        irr = x2_log_mean_df_sec1.columns[np.isnan(x2_log_mean_df_sec1.loc[ct]) | np.isnan(x2_log_mean_df_sec2.loc[ct])].union(irr)
        res_df.loc[ct, irr] = np.nan
        res_q.loc[ct, irr] = np.nan
    else:
        res_df.loc[ct, :] = np.nan
        res_q.loc[ct, :] = np.nan
celltypes_to_use = starrfish3_sec1.get_celltypes().value_counts().index[starrfish3_sec1.get_celltypes().value_counts() >= 1000].intersection(res_q.index)
cres_to_use = res_q.columns[np.nanmin(res_q.loc[celltypes_to_use], axis=0) < 0.05].union(starrfish3_sec1.get_negative_control_cres())
cres_to_use = cres_to_use[~cres_to_use.isin(cre_blacklist)]
fig, final_order = celltype_pval_dotplot(res_q, res_df, cres_to_use, celltypes_to_use,
                                         positive_control_info=cre_info, significant_cutoff=0.05, z_norm=False,
                                         figsize=(15, 20))
fig




# %%
# res_df[res_q > 0.05] = np.nan
res_df = x2_log_mean_df_sec1_filtered.copy()
atac_peaks = pd.read_csv('Data/cre_atac_peaks.csv', index_col=0)
atac_peaks = atac_peaks.loc[res_df.index.intersection(atac_peaks.index), res_df.columns.intersection(atac_peaks.columns)] >= 0.5
res_df = res_df.loc[atac_peaks.index, atac_peaks.columns]
atac_peaks[res_df.isna()] = False
# res_df[res_df < 1] = np.nan
res_df[res_q > 0.05] = np.nan
overlap = res_df.loc[atac_peaks.index, atac_peaks.columns][atac_peaks].notna().sum().sum()
precision = overlap / atac_peaks.sum().sum()
recall = overlap / res_df.notna().sum().sum()
precision, recall
# %%
res_df = x2_log_mean_df_sec1_filtered.copy()
atac_peaks = pd.read_csv('Data/cre_chromatin_state_a.csv', index_col=0)
atac_peaks = atac_peaks.loc[res_df.index.intersection(atac_peaks.index), res_df.columns.intersection(atac_peaks.columns)] >= 0.5
res_df = res_df.loc[atac_peaks.index, atac_peaks.columns]
atac_peaks[res_df.isna()] = False
# res_df[res_df < 1] = np.nan
res_df[res_q > 0.05] = np.nan
overlap = res_df.loc[atac_peaks.index, atac_peaks.columns][atac_peaks].notna().sum().sum()
precision = overlap / atac_peaks.sum().sum()
recall = overlap / res_df.notna().sum().sum()
precision, recall


# %% compare with pseudo_glm_df
pseudo_glm_df = pd.read_csv('results/expr3/pseudo_glm_df.csv', index_col=0)
pseudo_glm_df = pseudo_glm_df.loc[res_df.index, res_df.columns]
# for the matched cell types, calculate overlap
overlap_df = pd.DataFrame(index=res_df.index, columns=['pseudo_glm_pos_n', 'pseudo_glm_neg_n', 'em_pos_n', 'em_neg_n', 'overlap'])
for ct in overlap_df.index:
    overlap_df.loc[ct, 'pseudo_glm_pos_n'] = ((pseudo_glm_df.loc[ct].notna()) & (pseudo_glm_df.loc[ct] > 0)).sum()
    overlap_df.loc[ct, 'pseudo_glm_neg_n'] = ((pseudo_glm_df.loc[ct].notna()) & (pseudo_glm_df.loc[ct] < 0)).sum()
    overlap_df.loc[ct, 'em_pos_n'] = ((res_df.loc[ct].notna()) & (res_df.loc[ct] > 0)).sum()
    overlap_df.loc[ct, 'em_neg_n'] = ((res_df.loc[ct].notna()) & (res_df.loc[ct] < 0)).sum()
    overlap_df.loc[ct, 'overlap_pos'] = ((pseudo_glm_df.loc[ct].notna()) & (pseudo_glm_df.loc[ct] > 0) & (res_df.loc[ct].notna()) & (res_df.loc[ct] > 0)).sum()
    overlap_df.loc[ct, 'overlap_neg'] = ((pseudo_glm_df.loc[ct].notna()) & (pseudo_glm_df.loc[ct] < 0) & (res_df.loc[ct].notna()) & (res_df.loc[ct] < 0)).sum()
# %% plot
overlap_df['overlap_pos_pct'] = overlap_df['overlap_pos'] / np.minimum(overlap_df['pseudo_glm_pos_n'], overlap_df['em_pos_n'])
overlap_df['overlap_neg_pct'] = overlap_df['overlap_neg'] / np.minimum(overlap_df['pseudo_glm_neg_n'], overlap_df['em_neg_n'])
fig, ax = plt.subplots(figsize=(8, 6))
sns.scatterplot(data=overlap_df, x='pseudo_glm_pos_n', y='em_pos_n', size='overlap_pos_pct', ax=ax)
# add text, don't overlap, use adjust_text package
texts = []
for i in overlap_df.index:
    texts.append(ax.text(overlap_df.loc[i, 'pseudo_glm_pos_n'], overlap_df.loc[i, 'em_pos_n'], i + ': ' + '{:.2f}'.format(overlap_df.loc[i, 'overlap_pos_pct']), fontsize=9))
adjust_text(texts, ax=ax)
# %%
fig, ax = plt.subplots(figsize=(8, 6))
sns.scatterplot(data=overlap_df, x='pseudo_glm_neg_n', y='em_neg_n', size='overlap_neg_pct', ax=ax)
# add text, don't overlap, use adjust_text package
texts = []
for i in overlap_df.index:
    texts.append(ax.text(overlap_df.loc[i, 'pseudo_glm_neg_n'], overlap_df.loc[i, 'em_neg_n'], i + ': ' + '{:.2f}'.format(overlap_df.loc[i, 'overlap_neg_pct']), fontsize=9))
adjust_text(texts, ax=ax)
# %%
