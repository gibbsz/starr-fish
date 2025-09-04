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
# run the pseudo bulk bootstrap test for T7
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
    'transform_x_y': 'log',
    'fix_intercept': None, # can be None, negative_control_x, total_x or negative_control_y, total_y
    'pseudo_bulk_size': [100, 200, 400, 600, 800, 1000, 1200, 1400, 1600, 2000, 2400, 2800, 3200, 3600, 4000],
    'pseudo_bulk_percentage': None,
    'pseudo_bulk_number': [100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100],
    'replace': True,
    'multiprocess_threads': 96,
}
# %%
cre_blacklist = ['CRE061', 'CRE143', 'CRE001']
mismatching_cres = pd.read_csv('Data/AAV_ONT_Barcode_Counts_vs_Mismatch_Percentage.csv', index_col=0)
cre_blacklist = np.unique(cre_blacklist + mismatching_cres.index[mismatching_cres['MismatchPercent'] > 20].tolist()).tolist()
# %%
save = False
infected_cells_threshold = 5
starrfish3_sec1 = STARRFISH.load('results/starrfish3_sec1.pkl')
cell_counts1 = starrfish3_sec1.get_celltypes().value_counts()
res1 = starrfish3_sec1.pseudo_bulk_glm_test(**pseudo_bulk_glm_test_config)
res1_summary = res1['result'].copy()
res1['pseudo_bulk_adata'].uns['CRE_info'] = starrfish3_sec1.adata.uns['CRE_info'].copy()
res1['pseudo_bulk_adata'].obs.index.name = None
res1 = STARRFISH(res1['pseudo_bulk_adata'])
res1.adata.obs['percentage'] = res1.adata.obs['percentage'].astype(float)
if save:
    res1.save('results/starrfish3_sec1_pseudo_bulk.pkl', overwrite_adata=True)
    starrfish3_sec1.save('results/starrfish3_sec1.pkl')
to_filter_sec1 = (starrfish3_sec1.get_cre_expression() > 0).groupby(starrfish3_sec1.get_celltypes()).sum() < infected_cells_threshold
to_filter_sec1[cre_blacklist] = True
del starrfish3_sec1

starrfish3_sec2 = STARRFISH.load('results/starrfish3_sec2.pkl')
cell_counts2 = starrfish3_sec2.get_celltypes().value_counts()
res2 = starrfish3_sec2.pseudo_bulk_glm_test(**pseudo_bulk_glm_test_config)
res2_summary = res2['result'].copy()
res2['pseudo_bulk_adata'].uns['CRE_info'] = starrfish3_sec2.adata.uns['CRE_info'].copy()
res2['pseudo_bulk_adata'].obs.index.name = None
res2 = STARRFISH(res2['pseudo_bulk_adata'])
res2.adata.obs['percentage'] = res2.adata.obs['percentage'].astype(float)
if save:
    res2.save('results/starrfish3_sec2_pseudo_bulk.pkl', overwrite_adata=True)
    starrfish3_sec2.save('results/starrfish3_sec2.pkl')
to_filter_sec2 = (starrfish3_sec2.get_cre_expression() > 0).groupby(starrfish3_sec2.get_celltypes()).sum() < infected_cells_threshold
to_filter_sec2[cre_blacklist] = True
del starrfish3_sec2

starrfish3 = STARRFISH.load('results/starrfish3.pkl')
cell_counts = starrfish3.get_celltypes().value_counts()
res = starrfish3.pseudo_bulk_glm_test(**pseudo_bulk_glm_test_config)
res_summary = res['result'].copy()
res['pseudo_bulk_adata'].uns['CRE_info'] = starrfish3.adata.uns['CRE_info'].copy()
res['pseudo_bulk_adata'].obs.index.name = None
res = STARRFISH(res['pseudo_bulk_adata'])
res.adata.obs['percentage'] = res.adata.obs['percentage'].astype(float)
if save:
    res.save('results/starrfish3_pseudo_bulk.pkl', overwrite_adata=True)
    starrfish3.save('results/starrfish3.pkl')
to_filter = (starrfish3.get_cre_expression() > 0).groupby(starrfish3.get_celltypes()).sum() < infected_cells_threshold
to_filter[cre_blacklist] = True
del starrfish3
# %%
res = STARRFISH.load('results/starrfish3_pseudo_bulk.pkl')
res1 = STARRFISH.load('results/starrfish3_sec1_pseudo_bulk.pkl')
res2 = STARRFISH.load('results/starrfish3_sec2_pseudo_bulk.pkl')
# %% check the results
res1_summary_filter = res1_summary['coef'].copy()
res2_summary_filter = res2_summary['coef'].copy()
res1_summary_filter[to_filter_sec1] = np.nan
res2_summary_filter[to_filter_sec2] = np.nan
# %%
cre_corr, celltype_corr = res1.corr_starrfish(res1_summary['coef'], res2_summary['coef'])
# %% plot cell type corr
cre_corr['libsize'] = res1.lib_size['counts'].loc[cre_corr.index]
celltype_corr['celltype_sec1'] = cell_counts1.loc[celltype_corr.index].values
celltype_corr['celltype_sec2'] = cell_counts2.loc[celltype_corr.index].values
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
reproducible_celltypes = celltype_corr.index[(celltype_corr['pearson_p'] <= 0.05) & (celltype_corr['effect_n'] >= n_cre_threshold)]



# %% visualize best cell type-wise corr
fig, ax = plt.subplots(figsize=(4, 4))
celltype = 'OB-in Frmd7 Gaba'
sns.scatterplot(x=res1_summary['coef'].loc[celltype], y=res2_summary['coef'].loc[celltype], color='blue', ax=ax)
# plot negative control
sns.scatterplot(x=res1_summary['coef'].loc[celltype, res1.get_negative_control_cres()], 
                y=res2_summary['coef'].loc[celltype, res1.get_negative_control_cres()], color='orange', ax=ax)
sns.scatterplot(x=res1_summary['coef'].loc[celltype, res1.get_positive_control_cres(celltype, use='atac-peak')], 
                y=res2_summary['coef'].loc[celltype, res1.get_positive_control_cres(celltype, use='atac-peak')], color='red', ax=ax)
sns.scatterplot(x=res1_summary['coef'].loc[celltype, cre_blacklist], 
                y=res2_summary['coef'].loc[celltype, cre_blacklist], color='green', ax=ax)
ax.set_xlabel(f'Section 1 {celltype}')
ax.set_ylabel(f'Section 2 {celltype}')
# %% check CRE363, strongest CRE
fig, ax = plt.subplots(ncols=2, figsize=(8, 4))
ct = 'OB-in Frmd7 Gaba'
cre = 'CRE216'
# plot regression line
sns.scatterplot(x=np.log1p(res1.get_t7_expression()[(res1.get_celltypes() == ct)][cre]),
                y=np.log1p(res1.get_cre_expression()[(res1.get_celltypes() == ct)][cre]), alpha=0.5,
                hue=res1.adata.obs[(res1.get_celltypes() == ct)]['size'], palette='coolwarm',
                ax=ax[0])
sns.regplot(x=np.log1p(res1.get_t7_expression()[(res1.get_celltypes() == ct)][cre]),
            y=np.log1p(res1.get_cre_expression()[(res1.get_celltypes() == ct)][cre]), ax=ax[0], scatter=False, color='red')
sns.regplot(x=np.log(res1.get_t7_expression()[(res1.get_celltypes() == ct)][cre][(res1.get_t7_expression()[(res1.get_celltypes() == ct)][cre] > 0) & (res1.get_cre_expression()[(res1.get_celltypes() == ct)][cre] > 0)]),
            y=np.log(res1.get_cre_expression()[(res1.get_celltypes() == ct)][cre][(res1.get_t7_expression()[(res1.get_celltypes() == ct)][cre] > 0) & (res1.get_cre_expression()[(res1.get_celltypes() == ct)][cre] > 0)]), ax=ax[0], color='blue', scatter=False)
sns.scatterplot(x=np.log1p(res2.get_t7_expression()[(res2.get_celltypes() == ct)][cre]),
                y=np.log1p(res2.get_cre_expression()[(res2.get_celltypes() == ct)][cre]), alpha=0.5,
                hue=res2.adata.obs[(res2.get_celltypes() == ct)]['size'], palette='coolwarm',
                ax=ax[1])
sns.regplot(x=np.log1p(res2.get_t7_expression()[(res2.get_celltypes() == ct)][cre]),
            y=np.log1p(res2.get_cre_expression()[(res2.get_celltypes() == ct)][cre]), ax=ax[1], scatter=False, color='red')
sns.regplot(x=np.log(res2.get_t7_expression()[(res2.get_celltypes() == ct)][cre][(res2.get_t7_expression()[(res2.get_celltypes() == ct)][cre] > 0) & (res2.get_cre_expression()[(res2.get_celltypes() == ct)][cre] > 0)]),
            y=np.log(res2.get_cre_expression()[(res2.get_celltypes() == ct)][cre][(res2.get_t7_expression()[(res2.get_celltypes() == ct)][cre] > 0) & (res2.get_cre_expression()[(res2.get_celltypes() == ct)][cre] > 0)]), ax=ax[1], color='blue', scatter=False)
ax[0].set_xlabel('T7 Pseudo bulk Expression')
ax[0].set_ylabel('CRE Pseudo bulk Expression')
ax[1].set_xlabel('T7 Pseudo bulk Expression')
ax[1].set_ylabel('CRE Pseudo bulk Expression')




# %% visualize best CRE-wise corr: CRE298 or CRE210
fig, ax = plt.subplots(ncols=2, figsize=(8, 4))
cre = 'CRE298'
sns.scatterplot(x=res1_summary['coef'][cre], y=res2_summary['coef'][cre], color='blue', ax=ax[0])
# plot negative control
sns.scatterplot(x=res1_summary['coef'].loc[res1.get_positive_control_celltypes(cre, use='atac-peak'), cre], 
                y=res2_summary['coef'].loc[res1.get_positive_control_celltypes(cre, use='atac-peak'), cre], color='red', ax=ax[0])
# plot cell type size
# sns.scatterplot(x=res1_summary['coef'][cre], y=res2_summary['coef'][cre],
#                 hue=np.log(np.minimum(cell_counts1.loc[res1_summary['coef'].index],
#                                        cell_counts2.loc[res2_summary['coef'].index])), ax=ax[1])
sns.scatterplot(x=res1_summary['coef'][cre], hue=res2_summary['coef'][cre],
                y=np.log(np.minimum(cell_counts1.loc[res1_summary['coef'].index],
                                       cell_counts2.loc[res2_summary['coef'].index])), ax=ax[1])



# %% check the regression on all the CREs
fig, ax = plt.subplots(ncols=3, figsize=(27, 9))
ct = 'Oligo NN'
data = pd.DataFrame({'x': np.log(res1.get_t7_expression()[(res1.get_celltypes() == ct)].values.flatten()),
                     'y': np.log(res1.get_cre_expression()[(res1.get_celltypes() == ct)].values.flatten()),
                     'cre': np.repeat(res1.get_cre_expression().columns, res1.get_t7_expression()[(res1.get_celltypes() == ct)].shape[0]),
                     'size': res1.adata.obs[(res1.get_celltypes() == ct)]['size'].tolist() * res1.get_cre_expression()[(res1.get_celltypes() == ct)].shape[1]})
data['libsize'] = res1.lib_size['counts'].loc[data['cre']].values
# drop NaN
data['x'][np.isinf(data['x'])] = np.nan
data['y'][np.isinf(data['y'])] = np.nan
data = data.dropna()

data['label'] = pd.NA
data['label'][(data['cre'].isin(['CRE363']))] = 'CRE363'
data['label'][(data['cre'].isin(res1.get_negative_control_cres()))] = 'Negative Control'
# plot regression line
sns.scatterplot(data, x='x', y='y', hue='size', palette='coolwarm', alpha=0.2, ax=ax[0])
sns.regplot(data, x='x', y='y', ax=ax[0], scatter=False)
ax[0].set_xlabel('T7 Pseudo bulk Expression')
ax[0].set_ylabel('CRE Pseudo bulk Expression')

# plot regression line
sns.scatterplot(data, x='x', y='y', hue='libsize', palette='coolwarm', alpha=0.2, ax=ax[1])
sns.regplot(data, x='x', y='y', ax=ax[1], scatter=False)
ax[1].set_xlabel('T7 Pseudo bulk Expression')
ax[1].set_ylabel('CRE Pseudo bulk Expression')


# plot regression line
sns.scatterplot(data, x='x', y='y', hue='label', alpha=0.2, ax=ax[2])
sns.regplot(data, x='x', y='y', ax=ax[2], scatter=False, color='green')
sns.regplot(data[data['label'] == 'Negative Control'], x='x', y='y', ax=ax[2], scatter=False, color='blue')
ax[2].set_xlabel('T7 Pseudo bulk Expression')
ax[2].set_ylabel('CRE Pseudo bulk Expression')



# %% dot plot
from plots import celltype_pval_dotplot
cre_info = res.get_creinfo().copy()
# for cre in cre_info.index:
#     positive_cts = res.get_positive_control_celltypes(cre, use='atac-peak')
#     if positive_cts is not None:
#         cre_info.loc[cre, 'best_subclass'] = ';'.join(positive_cts)
#     else:
#         cre_info.loc[cre, 'best_subclass'] = 'CRE'
cre_info['best_subclass'] = 'CRE'
cre_info.loc[res.get_negative_control_cres(), 'best_subclass'] = 'Negative Control'
# design a test to compare CRE activity in each cell type to Negative Control
res_df = res_summary['coef'].copy()
res_q = pd.DataFrame(1.0, index=res_df.index, columns=res_df.columns)
# remove to_filter
res_df[cre_blacklist] = np.nan
negative_control_mean = res_df[res.get_negative_control_cres()].apply(np.nanmean, axis=1)
negative_control_std = res_df[res.get_negative_control_cres()].apply(np.nanstd, axis=1)
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
    if ct in res1_summary['coef'].index and ct in res2_summary['coef'].index:
        irr = res1_summary['coef'].columns[np.abs(res1_summary['coef'].loc[ct] - res2_summary['coef'].loc[ct]) > 0.2]
        irr = res1_summary['coef'].columns[np.isnan(res1_summary['coef'].loc[ct]) | np.isnan(res2_summary['coef'].loc[ct])].union(irr)
        res_df.loc[ct, irr] = np.nan
        res_q.loc[ct, irr] = np.nan
    else:
        res_df.loc[ct, :] = np.nan
celltypes_to_use = reproducible_celltypes
cres_to_use = res_q.columns[np.nanmin(res_q.loc[celltypes_to_use], axis=0) < 0.05].union(res.get_negative_control_cres())
cres_to_use = cres_to_use[~cres_to_use.isin(cre_blacklist)]
fig, final_order = celltype_pval_dotplot(res_q, res_df, cres_to_use, celltypes_to_use,
                                         positive_control_info=cre_info, significant_cutoff=0.05, z_norm=False,
                                         figsize=(15, 30))
fig




# %%
res_df[res_q > 0.05] = np.nan
atac_peaks = pd.read_csv('Data/cre_atac_peaks.csv', index_col=0)
atac_peaks = atac_peaks.loc[celltypes_to_use.intersection(atac_peaks.index), cres_to_use.intersection(atac_peaks.columns)] >= 0.5
overlap = res_df.loc[atac_peaks.index, atac_peaks.columns][atac_peaks].notna().sum().sum()
precision = overlap / atac_peaks.sum().sum()
precision
# %%
atac_peaks = pd.read_csv('Data/cre_chromatin_state_a.csv', index_col=0)
atac_peaks = atac_peaks.loc[celltypes_to_use.intersection(atac_peaks.index), cres_to_use.intersection(atac_peaks.columns)] >= 0.5
overlap = res_df.loc[atac_peaks.index, atac_peaks.columns][atac_peaks].notna().sum().sum()
precision = overlap / atac_peaks.sum().sum()
precision

