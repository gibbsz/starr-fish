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
from matplotlib.ticker import LogLocator, LogFormatter
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
    PWD = '/gpfs/commons/groups/ren_lab/guojiezhong/starr-fish/Mouse_brain.Guojie'
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
starrfish3 = STARRFISH.load('results/starrfish3.gauss.pkl')
# %% get subclass name and subclass transform
subclass_annotation = pd.read_excel(f'Data/abc_atlas/allen_institute_nominature.xlsx')
subclass_annotation['subclass'] = subclass_annotation['subclass_id_label'].str.replace('^[0-9]+ ', '', regex=True)
subclass_annotation['subclass'] = subclass_annotation['subclass'].str.replace('/', '-', regex=True)
subclass_to_subclass_name = subclass_annotation['subclass_id_label'].groupby(subclass_annotation['subclass']).first().to_dict()
subclass_name_to_subclass = subclass_annotation['subclass'].groupby(subclass_annotation['subclass_id_label']).first().to_dict()
cre_blacklist = ['CRE061', 'CRE143', 'CRE001']
# add CREs that dividable by 3 to blacklist
every_third_blacklist = [f'CRE{i:03d}' for i in range(1, 400) if i % 3 == 1]
# cre_blacklist += every_third_blacklist
cre_whitelist = starrfish3_sec1.get_creinfo().index[~starrfish3_sec1.get_creinfo().index.isin(cre_blacklist)]
# read in the mismatching barcode CREs
mismatching_cres = pd.read_csv('Data/AAV_ONT_Barcode_Counts_vs_Mismatch_Percentage.csv', index_col=0)
cre_whitelist = cre_whitelist[~cre_whitelist.isin(mismatching_cres.index[mismatching_cres['MismatchPercent'] > 20])]
cre_blacklist = np.unique(cre_blacklist + mismatching_cres.index[mismatching_cres['MismatchPercent'] > 20].tolist()).tolist()
# assign to starrfish3
starrfish3_sec1.blacklist_cre = cre_blacklist
starrfish3_sec2.blacklist_cre = cre_blacklist
starrfish3.blacklist_cre = cre_blacklist


# %% define the CREs and Cell Type matric to keep
infected_cells_threshold = 5
to_filter = (starrfish3.get_cre_expression() > 0).groupby(starrfish3.get_celltypes()).sum() < infected_cells_threshold
to_filter_sec1 = (starrfish3_sec1.get_cre_expression() > 0).groupby(starrfish3_sec1.get_celltypes()).sum() < infected_cells_threshold
to_filter_sec2 = (starrfish3_sec2.get_cre_expression() > 0).groupby(starrfish3_sec2.get_celltypes()).sum() < infected_cells_threshold
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
                           "normalize_by_negative_control": True, # normalize by negative control
                           'normalize_by_total_cre': False,
                           "normalize_by_infected_cell": False,
                           "normalize_by_libsize": False,
                           "log_transform": False,
                           "filter_zero_counts": False,
                           "bootstrap_number": 10000,
                           "n_jobs": 72,
                           'load_stored': True,
                           'fill_nan': False}
res1 = starrfish3_sec1.fold_change_test(**fold_change_test_config)
# starrfish3_sec1.save('results/starrfish3_sec1.pkl')
res2 = starrfish3_sec2.fold_change_test(**fold_change_test_config)
# starrfish3_sec2.save('results/starrfish3_sec2.pkl')
res = starrfish3.fold_change_test(**fold_change_test_config)
# starrfish3.save('results/starrfish3.gauss.pkl')


# # # %% get activity from average_bootstrap method
# with open('results/starrfish3.average_bootstrap.pkl', 'rb') as f:
#     res_avg = pickle.load(f)
# with open('results/starrfish3_sec1.average_bootstrap.pkl', 'rb') as f:
#     res1_avg = pickle.load(f)
# with open('results/starrfish3_sec2.average_bootstrap.pkl', 'rb') as f:
#     res2_avg = pickle.load(f)
# # # %% compare the two methods
# from plots import average_foldchange_specificity_test, q_value_correction
# p_mat_rank_test, p_mat_frequentist = average_foldchange_specificity_test(res_avg, res)
# # sec1 
# p_mat_rank_test1, p_mat_frequentist1 = average_foldchange_specificity_test(res1_avg, res1)
# # sec2
# p_mat_rank_test2, p_mat_frequentist2 = average_foldchange_specificity_test(res2_avg, res2)
# # %% q-value correction
# p_mat_rank_test_filter = p_mat_rank_test.copy()
# p_mat_rank_test_filter[to_filter] = np.nan
# q_mat_rank_test = q_value_correction(p_mat_rank_test_filter)

# p_mat_frequentist_filter = p_mat_frequentist.copy()
# p_mat_frequentist_filter[to_filter] = np.nan
# q_mat_frequentist = q_value_correction(p_mat_frequentist_filter)

# p_mat_rank_test1_filter = p_mat_rank_test1.copy()
# p_mat_rank_test1_filter[to_filter_sec1] = np.nan
# q_mat_rank_test1 = q_value_correction(p_mat_rank_test1_filter)

# p_mat_frequentist1_filter = p_mat_frequentist1.copy()
# p_mat_frequentist1_filter[to_filter_sec1] = np.nan
# q_mat_frequentist1 = q_value_correction(p_mat_frequentist1_filter)

# p_mat_rank_test2_filter = p_mat_rank_test2.copy()
# p_mat_rank_test2_filter[to_filter_sec2] = np.nan
# q_mat_rank_test2 = q_value_correction(p_mat_rank_test2_filter)

# p_mat_frequentist2_filter = p_mat_frequentist2.copy()
# p_mat_frequentist2_filter[to_filter_sec2] = np.nan
# q_mat_frequentist2 = q_value_correction(p_mat_frequentist2_filter)

# %% recalculate p-value
def recalculate_activity_pvalue(res, to_filter=None, bootstrap_threshold=0.8):
    activity_res = res['celltype_activity'].copy()
    activity_array = res['activity_array'].copy()
    # apply filter
    if to_filter is not None:
        activity_res[to_filter] = np.nan
    # drop the nan or inf results
    activity_res[np.isfinite(activity_res) == False] = np.nan
    # figure out failed nan values
    p_value_mat = np.ones(activity_res.shape) * np.nan
    for i, celltype in enumerate(activity_res.index):
        for j, cre in enumerate(activity_res.columns):
            if np.isfinite(activity_res.iloc[i, j]):
                # get bootstrap values
                boot_values = activity_array[:, i, j]
                boot_values = boot_values[np.isfinite(boot_values)]
                if len(boot_values) >= bootstrap_threshold * activity_array.shape[0]:
                    # perform frequentist p-value calculation
                    p_value_mat[i, j] = np.sum(activity_res.iloc[i, j] <= boot_values) / len(boot_values)
    # calibrate p-values
    q_res = p_value_mat.flatten().copy().astype(float)
    q_res[~np.isnan(q_res)] = multitest.multipletests(q_res[~np.isnan(q_res)], method='fdr_bh')[1]
    q_res = pd.DataFrame(q_res.reshape(activity_res.shape), index=activity_res.index, columns=activity_res.columns)
    return activity_res, q_res

# %% check reproducibility of cell type specificity
activity_res1 = res1['celltype_activity'].copy()
activity_res2 = res2['celltype_activity'].copy()
activity_res = res['celltype_activity'].copy()
# filter out to filter
infected_cells_threshold = 5
to_filter = (starrfish3.get_cre_expression() > 0).groupby(starrfish3.get_celltypes()).sum() < infected_cells_threshold
to_filter_sec1 = (starrfish3_sec1.get_cre_expression() > 0).groupby(starrfish3_sec1.get_celltypes()).sum() < infected_cells_threshold
to_filter_sec2 = (starrfish3_sec2.get_cre_expression() > 0).groupby(starrfish3_sec2.get_celltypes()).sum() < infected_cells_threshold
to_filter[cre_blacklist] = True
to_filter_sec1[cre_blacklist] = True
to_filter_sec2[cre_blacklist] = True
# recalculate q-values
activity_res1, q_res1 = recalculate_activity_pvalue(res1, to_filter=to_filter_sec1)
activity_res2, q_res2 = recalculate_activity_pvalue(res2, to_filter=to_filter_sec2)
# find consistent reproducible q-values
q_res1_filter = q_res1.reindex(activity_res.index).copy()
q_res2_filter = q_res2.reindex(activity_res.index).copy()
q_consistent = ((q_res1_filter <= 0.05) & (q_res2_filter <= 0.05)) | ((q_res1_filter > 0.05) & (q_res2_filter > 0.05))
activity_res, q_res = recalculate_activity_pvalue(res, to_filter=to_filter)
# q_res[~q_consistent] = np.nan

# %% 
# q_res1 = q_mat_frequentist1.copy()
# q_res2 = q_mat_frequentist2.copy()
# q_res = q_mat_frequentist.copy()
# activity_res1 = res1_avg['celltype_activity'].copy()
# activity_res1[to_filter_sec1] = np.nan
# activity_res2 = res2_avg['celltype_activity'].copy()
# activity_res2[to_filter_sec2] = np.nan
# activity_res = res_avg['celltype_activity'].copy()
# activity_res[to_filter] = np.nan



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
fig.savefig('results/expr3/specificity_norm_neg_control/reproducibility_by_cre_pearson_sec1_sec2.pdf')
# %% plot pearson R in violin
fig, ax = plt.subplots(figsize=(2, 4))
sns.violinplot(data=cre_corr[(cre_corr['effect_n'] >= n_celltype_threshold)], y='pearson', ax=ax)
fig.savefig('results/expr3/specificity_norm_neg_control/reproducibility_by_cre_pearson_violin_sec1_sec2.pdf')




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
fig.savefig('results/expr3/specificity_norm_neg_control/reproducibility_by_cre_percentage_all.pdf', bbox_inches='tight')


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


# %% filter based on cell types
cell_type_counts_df = pd.DataFrame(index=starrfish3.get_celltypes().unique(), columns=['Sec1', 'Sec2', 'All'])
cell_type_counts_df['Sec1'] = starrfish3_sec1.get_celltypes().value_counts().reindex(cell_type_counts_df.index, fill_value=0)
cell_type_counts_df['Sec2'] = starrfish3_sec2.get_celltypes().value_counts().reindex(cell_type_counts_df.index, fill_value=0)
cell_type_counts_df['All'] = starrfish3.get_celltypes().value_counts().reindex(cell_type_counts_df.index, fill_value=0)
cell_type_size_thresholds = np.concatenate((np.linspace(0, 90, 10), np.linspace(100, 900, 9), np.linspace(1000, 8000, 8)))
cell_type_size_reproducity = pd.DataFrame(index=cell_type_size_thresholds, columns=['Sec1-All', 'Sec2-All', 'Sec1-Sec2', '#Sec1-All', '#Sec2-All', '#Sec1-Sec2'])
for threshold in cell_type_size_thresholds:
    cell_types_to_use = cell_type_counts_df.index[(cell_type_counts_df > threshold).all(axis=1)]
    res_compare = plot_q_value_cre_reproducibility(q_res1.loc[cell_types_to_use], q_res2.loc[cell_types_to_use], q_res.loc[cell_types_to_use], starrfish3.lib_size, 0.05, plot=False)
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
fig.savefig('results/expr3/specificity_norm_neg_control/reproducibility_by_cre_threshold_celltype.pdf')


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
# cres_to_use = cres_to_use.intersection(reproducible_cres)
fig, cre_orders = cre_pval_dotplot(q_res, activity_res, cres_to_use, celltypes_to_use, cre_info, reorder_cres=True, figsize=(15, 30))
fig.savefig('results/expr3/specificity_norm_neg_control/cre_pval_dotplot_all.pdf')
fig
# %%
fig, cre_orders = cre_pval_dotplot(q_res1, activity_res1, pd.Index(cre_orders), celltypes_to_use, cre_info, reorder_cres=False, figsize=(15, 15))
fig.savefig('results/expr3/specificity_norm_neg_control/cre_pval_dotplot_1.pdf')
fig
# %%
fig, cre_orders = cre_pval_dotplot(q_res2, activity_res2, pd.Index(cre_orders), celltypes_to_use, cre_info, reorder_cres=False, figsize=(15, 15))
fig.savefig('results/expr3/specificity_norm_neg_control/cre_pval_dotplot_2.pdf')
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
        norm_by_negative_control_cell_type_sum=True,
        scale_size_by='counts', # scale size by "counts": normalized counts; or "celltype_number": number of cells in the cell type
        log=True, transpose=-1, flipx=-1, sz_max=50,
        cell_types_to_use=celltypes_to_use)
    fig.savefig(f'results/expr3/specificity_norm_neg_control/cre_significant_celltypes/{cre}.pdf')


# %% plot correlation with three modalities
cell_types_to_use = starrfish3.get_celltypes().value_counts().index[starrfish3.get_celltypes().value_counts()>=1000]
cre_corr, celltype_corr = starrfish3.corr_atac_cpm(cell_types_to_use=reproducible_celltypes, cres_to_use=None, acvitity_df = activity_res)
cre_corr['libsize'] = starrfish3.lib_size['counts'].loc[cre_corr.index]
celltype_corr['celltype_sec1'] = starrfish3_sec1.get_celltypes().value_counts().reindex(celltype_corr.index).fillna(0).values
celltype_corr['celltype_sec2'] = starrfish3_sec2.get_celltypes().value_counts().reindex(celltype_corr.index).fillna(0).values
celltype_corr['celltype_full'] = starrfish3.get_celltypes().value_counts().reindex(celltype_corr.index).fillna(0).values
celltype_corr['celltype_n'] = np.minimum(celltype_corr['celltype_sec1'], celltype_corr['celltype_sec2'])
# plot cre_corr scatter
n_cre_threshold = 10
n_celltype_threshold = 10
fig, ax = plt.subplots(figsize=(4, 4))
sns.scatterplot(data=cre_corr[(cre_corr['pearson_p'] > 0.05) & (cre_corr['effect_n'] >= n_celltype_threshold)], x='libsize', y='pearson', color='blue', ax=ax)
sns.scatterplot(data=cre_corr[(cre_corr['pearson_p'] <= 0.05) & (cre_corr['effect_n'] >= n_celltype_threshold)], x='libsize', y='pearson', color='red', ax=ax)
fig, ax = plt.subplots(figsize=(2, 4))
sns.violinplot(data=cre_corr[(cre_corr['effect_n'] >= n_celltype_threshold)], y='pearson', ax=ax)



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

# %% get precision recall
pr_df1 = get_pr_df(qvalue_df=q_res.loc[reproducible_celltypes, reproducible_cres].copy(), cell_types_to_use=reproducible_celltypes,
                   starrfish_obj=starrfish3, metric = ['atac_cpm', 'h3k4me1_cpm', 'h3k27ac_cpm'], z_cutoffs=[0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0])
pr_df2 = get_pr_df(qvalue_df=q_res.loc[reproducible_celltypes, reproducible_cres].copy(), cell_types_to_use=reproducible_celltypes,
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
fig.savefig('results/expr3/specificity_norm_neg_control/precision_recall_all.pdf')

# %%
overall_precision = pr_df1.groupby(['mod', 'z_cutoff'])['correct'].sum() / pr_df1.groupby(['mod', 'z_cutoff'])['all_pred'].sum()
toplot = pd.DataFrame(overall_precision).reset_index()
toplot.rename(columns={0: 'precision'}, inplace=True)
sns.lmplot(data=toplot, x='z_cutoff', y='precision', hue='mod', scatter=True, lowess=True)
# %% screen how many cres can reach certain z-score cutoff
def get_num_cres(z_cutoffs, cell_types_to_use, metric = ['ATAC_cpm', 'H3K4me1_cpm', 'H3K9me3_cpm', 'H3K27ac_cpm', 'H3K27me3_cpm'],):
    mod_cpms = {}
    for mod in metric:
        mod_cpm = pd.read_csv(f'Data/CRE_CPM_matrices/{mod}_peak_pad_500_Bysubclass.csv', index_col=0)
        cell_types_to_use = [ct for ct in cell_types_to_use if ct in mod_cpm.columns]
        mod_cpms[mod] = mod_cpm
    num_cres = {}
    res_dfs = {}
    for mod in metric:
        res_dfs[mod] = pd.DataFrame(index=z_cutoffs, columns=cell_types_to_use)
        # append another cell type, which is 'ALL'
        res_dfs[mod]['ALL'] = 0
    for z in z_cutoffs:
        for mod in metric:
            mod_cpm = mod_cpms[mod][cell_types_to_use] * 10
            # log transform
            mod_cpm = np.log1p(mod_cpm.astype(float))
            mod_cpm_z = mod_cpm.sub(mod_cpm.mean(axis=1), axis=0).div(mod_cpm.std(axis=1), axis=0)
            for celltype in cell_types_to_use:
                num_cres = (mod_cpm_z[celltype] >= z).sum()
                res_dfs[mod].loc[z, celltype] = num_cres
            # overall
            num_cres = (mod_cpm_z.max(axis=1) >= z).sum()
            res_dfs[mod].loc[z, 'ALL'] = num_cres
    return res_dfs
# %%
z_cutoffs = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
num_cres_dfs = get_num_cres(z_cutoffs, reproducible_celltypes)
# %% plot overall results
toplot = pd.DataFrame()
for mod in num_cres_dfs.keys():
    df_mod = num_cres_dfs[mod].reset_index().melt(id_vars='index', var_name='cell_type', value_name='num_cres')
    df_mod['mod'] = mod
    toplot = pd.concat([toplot, df_mod], axis=0, ignore_index=True)
fig, ax = plt.subplots(figsize=(6, 6))
palette = {'ATAC_cpm': '#A6CEE3', 'H3K4me1_cpm': '#B2DF8A', 'H3K9me3_cpm': '#FB8072',
           'H3K27ac_cpm': '#FDB462', 'H3K27me3_cpm': '#CAB2D6'}
sns.lineplot(data=toplot[(toplot['cell_type'] == 'ALL') & (toplot['mod'].isin(['ATAC_cpm', 'H3K4me1_cpm', 'H3K27ac_cpm']))], 
             x='index', y='num_cres', hue='mod', marker='o', ax=ax, palette=palette)
# plot in log scale, add more ticks
ax.set_xticks(z_cutoffs)
ax.set_xlabel('Z-score Cutoff')
ax.set_yscale('log')
ax.set_yticks([1e3, 1e4, 2e4, 5e4, 1e5, 2e5, 5e5, 1e6])
fig.savefig('results/expr3/specificity_norm_neg_control/num_cres_above_zscore_cutoff_cortex.pdf')
# %%
