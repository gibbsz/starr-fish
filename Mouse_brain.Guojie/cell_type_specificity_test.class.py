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
import pickle
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
# split the adata3 into two parts based on two sections
# adata3.obs['section'] = (adata3.obsm['X_spatial'][:, 0] >= -1900).astype(int)
# adata3_sec1 = adata3[adata3.obs['section'] == 0].copy()
# adata3_sec2 = adata3[adata3.obs['section'] == 1].copy()
# make two STARRFISH objects
# starrfish3_sec1 = STARRFISH(adata3_sec1, celltype_tag='obs:class')
# starrfish3_sec2 = STARRFISH(adata3_sec2, celltype_tag='obs:class')
# starrfish3 = STARRFISH(adata3, celltype_tag='obs:class')

starrfish3_sec1 = STARRFISH.load(f'{PWD}/results/starrfish3_sec1.class.pkl')
starrfish3_sec2 = STARRFISH.load(f'{PWD}/results/starrfish3_sec2.class.pkl')
starrfish3 = STARRFISH.load(f'{PWD}/results/starrfish3.class.pkl')

# %% define the CREs and Cell Type matric to keep
infected_cells_threshold = 5
to_filter = (starrfish3.get_cre_expression() > 0).groupby(starrfish3.get_celltypes()).sum() < infected_cells_threshold
to_filter_sec1 = (starrfish3_sec1.get_cre_expression() > 0).groupby(starrfish3_sec1.get_celltypes()).sum() < infected_cells_threshold
to_filter_sec2 = (starrfish3_sec2.get_cre_expression() > 0).groupby(starrfish3_sec2.get_celltypes()).sum() < infected_cells_threshold
cre_blacklist = ['CRE061', 'CRE143', 'CRE001']
cre_whitelist = starrfish3_sec1.get_creinfo().index[~starrfish3_sec1.get_creinfo().index.isin(cre_blacklist)]
# read in the mismatching barcode CREs
mismatching_cres = pd.read_csv('Data/AAV_ONT_Barcode_Counts_vs_Mismatch_Percentage.csv', index_col=0)
cre_whitelist = cre_whitelist[~cre_whitelist.isin(mismatching_cres.index[mismatching_cres['MismatchPercent'] > 20])]
cre_blacklist = np.unique(cre_blacklist + mismatching_cres.index[mismatching_cres['MismatchPercent'] > 20].tolist()).tolist()
to_filter[cre_blacklist] = True
to_filter_sec1[cre_blacklist] = True
to_filter_sec2[cre_blacklist] = True



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
                           "n_jobs": 24,
                           'load_stored': True,}
res1 = starrfish3_sec1.fold_change_test(**fold_change_test_config)
#starrfish3_sec1.save('results/starrfish3_sec1.class.pkl')
res2 = starrfish3_sec2.fold_change_test(**fold_change_test_config)
# starrfish3_sec2.save('results/starrfish3_sec2.class.pkl')
res = starrfish3.fold_change_test(**fold_change_test_config)
# starrfish3.save('results/starrfish3.class.pkl')


# %% get activity from average_bootstrap method
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
    'n_jobs': 36,
}
starrfish3_sec1_avg = STARRFISH.load(f'{PWD}/results/starrfish3_sec1.bak.class.pkl')
starrfish3_sec2_avg = STARRFISH.load(f'{PWD}/results/starrfish3_sec2.bak.class.pkl')
starrfish3_avg = STARRFISH.load(f'{PWD}/results/starrfish3.bak.class.pkl')
res1_avg = starrfish3_sec1_avg.average_bootstrap_test(**average_bootstrap_test_config)
res2_avg = starrfish3_sec2_avg.average_bootstrap_test(**average_bootstrap_test_config)
res_avg = starrfish3_avg.average_bootstrap_test(**average_bootstrap_test_config)
del starrfish3_sec1_avg, starrfish3_sec2_avg, starrfish3_avg
# %% compare the two methods
from plots import average_foldchange_specificity_test, q_value_correction
p_mat_rank_test, p_mat_frequentist = average_foldchange_specificity_test(res_avg, res)
# sec1 
p_mat_rank_test1, p_mat_frequentist1 = average_foldchange_specificity_test(res1_avg, res1)
# sec2
p_mat_rank_test2, p_mat_frequentist2 = average_foldchange_specificity_test(res2_avg, res2)
# %% q-value correction
p_mat_rank_test_filter = p_mat_rank_test.copy()
p_mat_rank_test_filter[to_filter] = np.nan
q_mat_rank_test = q_value_correction(p_mat_rank_test_filter)

p_mat_frequentist_filter = p_mat_frequentist.copy()
p_mat_frequentist_filter[to_filter] = np.nan
q_mat_frequentist = q_value_correction(p_mat_frequentist_filter)

p_mat_rank_test1_filter = p_mat_rank_test1.copy()
p_mat_rank_test1_filter[to_filter_sec1] = np.nan
q_mat_rank_test1 = q_value_correction(p_mat_rank_test1_filter)

p_mat_frequentist1_filter = p_mat_frequentist1.copy()
p_mat_frequentist1_filter[to_filter_sec1] = np.nan
q_mat_frequentist1 = q_value_correction(p_mat_frequentist1_filter)

p_mat_rank_test2_filter = p_mat_rank_test2.copy()
p_mat_rank_test2_filter[to_filter_sec2] = np.nan
q_mat_rank_test2 = q_value_correction(p_mat_rank_test2_filter)

p_mat_frequentist2_filter = p_mat_frequentist2.copy()
p_mat_frequentist2_filter[to_filter_sec2] = np.nan
q_mat_frequentist2 = q_value_correction(p_mat_frequentist2_filter)


# %% 
q_res1 = q_mat_frequentist1.copy()
q_res2 = q_mat_frequentist2.copy()
q_res = q_mat_frequentist.copy()
activity_res1 = res1_avg['celltype_activity'].copy()
activity_res1[to_filter_sec1] = np.nan
activity_res2 = res2_avg['celltype_activity'].copy()
activity_res2[to_filter_sec2] = np.nan
activity_res = res_avg['celltype_activity'].copy()
activity_res[to_filter] = np.nan



# %% define reproducible cell types
reproducible_celltypes = starrfish3.get_celltypes().value_counts().index[starrfish3.get_celltypes().value_counts()>=1000].intersection(
    starrfish3_sec1.get_celltypes().value_counts().index[starrfish3_sec1.get_celltypes().value_counts()>=1000]).intersection(
    starrfish3_sec2.get_celltypes().value_counts().index[starrfish3_sec2.get_celltypes().value_counts()>=1000]
)
# remove one of them based on figure 4 results
reproducible_celltypes = reproducible_celltypes[~reproducible_celltypes.isin(['Ependymal NN'])]


# %% pearson correlation of between the two sections
cre_corr, celltype_corr = starrfish3.corr_starrfish(activity_res1, activity_res2)
cre_corr['libsize'] = starrfish3.lib_size['counts'].loc[cre_corr.index]
celltype_corr['celltype_sec1'] = starrfish3_sec1.get_celltypes().value_counts().reindex(celltype_corr.index).fillna(0).values
celltype_corr['celltype_sec2'] = starrfish3_sec2.get_celltypes().value_counts().reindex(celltype_corr.index).fillna(0).values
celltype_corr['celltype_full'] = starrfish3.get_celltypes().value_counts().reindex(celltype_corr.index).fillna(0).values
celltype_corr['celltype_n'] = np.minimum(celltype_corr['celltype_sec1'], celltype_corr['celltype_sec2'])
# %% plot
n_cre_threshold = 10
n_celltype_threshold = 10
fig, ax = plt.subplots(figsize=(4, 4))
sns.scatterplot(data=cre_corr[(cre_corr['pearson_p'] > 0.05) & (cre_corr['effect_n'] >= n_celltype_threshold)], x='libsize', y='pearson', color='blue', ax=ax)
sns.scatterplot(data=cre_corr[(cre_corr['pearson_p'] <= 0.05) & (cre_corr['effect_n'] >= n_celltype_threshold)], x='libsize', y='pearson', color='red', ax=ax)
ax.set_xscale('log')
fig.savefig('results/expr3/class/reproducibility_by_cre_pearson_sec1_sec2.pdf')
# %% plot pearson R in violin
fig, ax = plt.subplots(figsize=(2, 4))
sns.violinplot(data=cre_corr[(cre_corr['effect_n'] >= n_celltype_threshold)], y='pearson', ax=ax)
fig.savefig('results/expr3/class/reproducibility_by_cre_pearson_violin_sec1_sec2.pdf')



# %%
from plots import plot_q_value_cre_reproducibility
res_compare = plot_q_value_cre_reproducibility(q_res1, q_res2, q_res, starrfish3.lib_size, 0.05)
# define reproducible CREs
reproducible_cres = res_compare.index[(res_compare[['Common', 'Common_sec1', 'Common_sec2']] > 0).any(axis=1)]
# %% plot of sec1 vs sec2 and overlap
fig, ax = plt.subplots(ncols=3, figsize=(12, 4))
# Determine common size and hue ranges
size_cols = ['Common', 'Common_sec1', 'Common_sec2']
hue_cols = ['Percentage', 'Percentage_sec1', 'Percentage_sec2']
size_range = (res_compare[size_cols].min().min(), res_compare[size_cols].max().max())
hue_range = (res_compare[hue_cols].min().min(), res_compare[hue_cols].max().max())

sns.scatterplot(data=res_compare, x='Sec1', y='Sec2', size='Common', hue='Percentage',
                palette='coolwarm', ax=ax[0], alpha=0.7, legend=False,
                sizes=(20, 200), size_norm=size_range, hue_norm=hue_range)
sns.scatterplot(data=res_compare, x='Sec1', y='All', size='Common_sec1', hue='Percentage_sec1',
                palette='coolwarm', ax=ax[1], alpha=0.7, legend=False,
                sizes=(20, 200), size_norm=size_range, hue_norm=hue_range)
sns.scatterplot(data=res_compare, x='Sec2', y='All', size='Common_sec2', hue='Percentage_sec2',
                palette='coolwarm', ax=ax[2], alpha=0.7,
                sizes=(20, 200), size_norm=size_range, hue_norm=hue_range)

# Customize legend titles
handles, labels = ax[2].get_legend_handles_labels()
legend = ax[2].legend(handles, labels, bbox_to_anchor=(1.05, 1), loc='upper left')
legend.set_title('')
for text in legend.get_texts():
    if text.get_text() in [str(i) for i in range(10)]:  # size legend items
        legend.get_texts()[legend.get_texts().index(text)].set_text('')
fig.savefig('results/expr3/class/reproducibility_by_cre_percentage_all.pdf', bbox_inches='tight')


# %% plot number of reproducible CREs
reproducible_cres_sec1_sec2 = res_compare.index[res_compare['Common'] > 0]
reproducible_cres_sec1_all = res_compare.index[res_compare['Common_sec1'] > 0]
reproducible_cres_sec2_all = res_compare.index[res_compare['Common_sec2'] > 0]
fig, ax = plt.subplots(figsize=(4, 4))
venn_labels = {'Sec1-Sec2': set(reproducible_cres_sec1_sec2),
               'Sec2-All': set(reproducible_cres_sec1_all),
               'Sec1-All': set(reproducible_cres_sec2_all)}
from matplotlib_venn import venn3
venn3(subsets=venn_labels.values(), set_labels=venn_labels.keys(), ax=ax)
fig.savefig('results/expr3/class/cre_venn.pdf', bbox_inches='tight')



# %% dot plot of some biology
from plots import cre_pval_dotplot
negative_control_cres = starrfish3.get_negative_control_cres()
cre_info = starrfish3.get_creinfo().copy()
cre_info['best_subclass'] = 'CRE'
cre_info.loc[negative_control_cres, 'best_subclass'] = 'Negative Control'
# plot
celltypes_to_use = starrfish3.get_celltypes().value_counts().index[starrfish3.get_celltypes().value_counts()>=1000]
cres_to_use = q_res.columns[np.nanmin(q_res.loc[celltypes_to_use], axis=0) < 0.05].union(starrfish3.get_negative_control_cres())
cres_to_use = cres_to_use[~cres_to_use.isin(cre_blacklist)]
fig, cre_orders = cre_pval_dotplot(q_res, activity_res, cres_to_use, celltypes_to_use, cre_info, reorder_cres=True, figsize=(15, 25))
fig.savefig('results/expr3/class/cre_pval_dotplot_all.pdf')
fig
# %%
fig, cre_orders = cre_pval_dotplot(q_res1, activity_res1, pd.Index(cre_orders), celltypes_to_use, cre_info, reorder_cres=False, figsize=(15, 25))
fig.savefig('results/expr3/class/cre_pval_dotplot_1.pdf')
fig
# %%
fig, cre_orders = cre_pval_dotplot(q_res2, activity_res2, pd.Index(cre_orders), celltypes_to_use, cre_info, reorder_cres=False, figsize=(15, 25))
fig.savefig('results/expr3/class/cre_pval_dotplot_2.pdf')
fig



# %% visualize some example
have_target_cres = q_res.columns[np.nanmin(q_res.loc[celltypes_to_use], axis=0) < 0.05]
for cre in have_target_cres:
    # rank by q-value
    cre_q_values = q_res.loc[celltypes_to_use, cre]
    cre_q_values = cre_q_values[cre_q_values <= 0.05] 
    # order by rank
    cre_q_values = cre_q_values.sort_values(ascending=True)
    cell_types_to_visualize = cre_q_values.index
    fig = starrfish3.plot_gene(
        cre, average_by_celltype=False, # if true, all cells from same cell type will have same value
        cell_types_to_visualize=cell_types_to_visualize, # only visualize some cell types
        scale_size_by='counts', # scale size by "counts": normalized counts; or "celltype_number": number of cells in the cell type
        log=True, transpose=-1, flipx=-1, sz_max=50,
        cell_types_to_use=celltypes_to_use)
    fig.savefig(f'results/expr3/class/cre_significant_celltypes/{cre}.pdf')

# %%
