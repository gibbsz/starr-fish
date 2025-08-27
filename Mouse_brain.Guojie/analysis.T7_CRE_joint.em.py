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
from utils import T7CRE_DistributionEM
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
em_model = T7CRE_DistributionEM(device='cuda:4', use_x0=True)
x0_sec1, x1_sec1, x2_sec1 = em_model.fit(celltypes_sec1.values, t7_counts_sec1, cre_counts_sec1, dim=20)
x0_sec2, x1_sec2, x2_sec2 = em_model.fit(celltypes_sec2.values, t7_counts_sec2, cre_counts_sec2, dim=20)
# %%
np.save(f'{PWD}/results/expr3/t7cre_em.x0.sec1.npy', x0_sec1)
np.save(f'{PWD}/results/expr3/t7cre_em.x0.sec2.npy', x0_sec2)
np.save(f'{PWD}/results/expr3/t7cre_em.x1.sec1.npy', x1_sec1)
np.save(f'{PWD}/results/expr3/t7cre_em.x1.sec2.npy', x1_sec2)
np.save(f'{PWD}/results/expr3/t7cre_em.x2.sec1.npy', x2_sec1)
np.save(f'{PWD}/results/expr3/t7cre_em.x2.sec2.npy', x2_sec2)
# %%
# check infection rate differences
x0_df_sec1 = pd.DataFrame(torch.nn.functional.softplus(torch.from_numpy(x0_sec1)), index=common_celltypes, columns=t7_counts_sec1.columns)
x0_df_sec2 = pd.DataFrame(torch.nn.functional.softplus(torch.from_numpy(x0_sec2)), index=common_celltypes, columns=t7_counts_sec2.columns)
x2_r_df_sec1 = pd.DataFrame(torch.nn.functional.softplus(torch.from_numpy(x2_sec1[:, :, 0])), index=common_celltypes, columns=t7_counts_sec1.columns)
x2_r_df_sec2 = pd.DataFrame(torch.nn.functional.softplus(torch.from_numpy(x2_sec2[:, :, 0])), index=common_celltypes, columns=t7_counts_sec2.columns)
x2_p_df_sec1 = pd.DataFrame(torch.nn.functional.sigmoid(torch.from_numpy(x2_sec1[:, :, 1])), index=common_celltypes, columns=t7_counts_sec1.columns)
x2_p_df_sec2 = pd.DataFrame(torch.nn.functional.sigmoid(torch.from_numpy(x2_sec2[:, :, 1])), index=common_celltypes, columns=t7_counts_sec2.columns)
# %%
x2_mean_df_sec1 = x2_r_df_sec1 * (1-x2_p_df_sec1) / x2_p_df_sec1
x2_mean_df_sec2 = x2_r_df_sec2 * (1-x2_p_df_sec2) / x2_p_df_sec2
# %% do cre and cell type corr
cre_corr, celltype_corr = starrfish3_sec1.corr_starrfish(x2_mean_df_sec1, x2_mean_df_sec2)
# %%
cre_corr['libsize'] = starrfish3_sec1.lib_size['counts'].loc[cre_corr.index]
celltype_corr['celltype_sec1'] = starrfish3_sec1.get_celltypes().value_counts().loc[celltype_corr.index].values
celltype_corr['celltype_sec2'] = starrfish3_sec2.get_celltypes().value_counts().loc[celltype_corr.index].values
celltype_corr['celltype_n'] = np.minimum(celltype_corr['celltype_sec1'], celltype_corr['celltype_sec2'])
# %% plot
fig, ax = plt.subplots(figsize=(4, 4))
sns.scatterplot(data=celltype_corr, x='celltype_n', y='pearson', ax=ax)
ax.set_xscale('log')


# # %%
# # analysis
# x0_df_sec1 = pd.read_csv(f'{PWD}/results/expr3/t7cre_mle_separate_nox0.x0.sec1.csv', index_col=0)
# x0_df_sec2 = pd.read_csv(f'{PWD}/results/expr3/t7cre_mle_separate_nox0.x0.sec2.csv', index_col=0)
# x1_df_sec1 = pd.read_csv(f'{PWD}/results/expr3/t7cre_mle_separate_nox0.x1.sec1.csv', index_col=0)
# x1_df_sec2 = pd.read_csv(f'{PWD}/results/expr3/t7cre_mle_separate_nox0.x1.sec2.csv', index_col=0)
# x2_r_df_sec1 = pd.read_csv(f'{PWD}/results/expr3/t7cre_mle_separate_nox0.x2.r.sec1.csv', index_col=0)
# x2_r_df_sec2 = pd.read_csv(f'{PWD}/results/expr3/t7cre_mle_separate_nox0.x2.r.sec2.csv', index_col=0)
# x2_p_df_sec1 = pd.read_csv(f'{PWD}/results/expr3/t7cre_mle_separate_nox0.x2.p.sec1.csv', index_col=0)
# x2_p_df_sec2 = pd.read_csv(f'{PWD}/results/expr3/t7cre_mle_separate_nox0.x2.p.sec2.csv', index_col=0)
# # %%
# # make sure no extreme values are used
# cres_to_use = x1_df_sec1.index[x1_df_sec1['total_counts'].between(-np.exp(5), np.exp(5)) & x1_df_sec2['total_counts'].between(-np.exp(5), np.exp(5)) & x1_df_sec1['logits'].between(-np.exp(5), np.exp(5)) & x1_df_sec2['logits'].between(-np.exp(5), np.exp(5))]
# # calculate average of the negative binomial distribution
# sec1_r = np.exp(x2_r_df_sec1[cres_to_use])
# sec1_p = 1 / (1+np.exp(-x2_p_df_sec1[cres_to_use]))
# sec2_r = np.exp(x2_r_df_sec2[cres_to_use])
# sec2_p = 1 / (1+np.exp(-x2_p_df_sec2[cres_to_use]))
# mu_sec1 = sec1_r * (1-sec1_p) / sec1_p
# mu_sec2 = sec2_r * (1-sec2_p) / sec2_p
# # %% 
# cell_type_counts = starrfish3.get_celltypes().value_counts()
# cell_types_counts1 = starrfish3_sec1.get_celltypes().value_counts()
# cell_types_counts2 = starrfish3_sec2.get_celltypes().value_counts()
# cell_types_to_use_1 = cell_types_counts1[cell_types_counts1 > 1000].index
# cell_types_to_use_2 = cell_types_counts2[cell_types_counts2 > 1000].index
# cell_types_to_use = cell_types_to_use_1.intersection(cell_types_to_use_2)

# # %%
# cre_corr, celltype_corr = starrfish3_sec1.corr_starrfish(mu_sec1, mu_sec2, cell_types_to_use=cell_types_to_use)
# cre_corr['libsize'] = starrfish3_sec1.lib_size['counts'].loc[cre_corr.index].values
# celltype_corr['celltype_counts_sec1'] = starrfish3_sec1.get_celltypes().value_counts().loc[celltype_corr.index].values
# celltype_corr['celltype_counts_sec2'] = starrfish3_sec2.get_celltypes().value_counts().loc[celltype_corr.index].values
# celltype_corr['celltype_counts'] = celltype_corr[['celltype_counts_sec1', 'celltype_counts_sec2']].min(axis=1)
# # %%
# fig, ax = plt.subplots(1, 2, figsize=(10, 5))
# sns.scatterplot(data=cre_corr, x='libsize', y='spearman', ax=ax[0])
# sns.scatterplot(data=celltype_corr, x='celltype_counts', y='spearman', ax=ax[1])
# ax[1].set_xscale('log')
# # %%
# sns.scatterplot(x=mu_sec1['CRE062'].loc[cell_types_to_use], y=mu_sec2['CRE386'].loc[cell_types_to_use])
# # %%

# %%
