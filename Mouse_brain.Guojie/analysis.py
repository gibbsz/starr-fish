# implement of starrfish vae
# %%
import scvi
import numpy as np
import warnings
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
PWD = os.path.dirname(os.path.abspath(__file__))
sys.path.append(f'{PWD}/')
from utils import STARRFISH
import re
import statsmodels.api as sm
from statsmodels.stats import multitest
# %% helper function to reload
def reload(starrfish):
    import importlib
    import utils
    importlib.reload(utils)
    from utils import STARRFISH
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
    return adata
# %% preprocess and load data
# load data and form STARRFISH object
load = True
if not load:
    adata1 = preprocess(f'{PWD}/Data/scdata_12_11NoT7_BRBB500gn_withCRE_final.h5ad')
    adata2 = preprocess(f'{PWD}/Data/scdata_03_14_BRBB500gn_withCRE_final.h5ad')
adata_cpm = 'Data/ATAC_cpm_peakBysubclass.csv'
if os.path.exists('results/starrfish1.pkl') and load:
    starrfish1 = STARRFISH.load('results/starrfish1.pkl')
else:
    starrfish1 = STARRFISH(adata1, atac_cpm=adata_cpm)
if os.path.exists('results/starrfish2.pkl') and load:
    starrfish2 = STARRFISH.load('results/starrfish2.pkl')
else:
    starrfish2 = STARRFISH(adata2, atac_cpm=adata_cpm)
if os.path.exists('results/starrfish1_filtered.pkl') and load:
    starrfish1_filtered = STARRFISH.load('results/starrfish1_filtered.pkl')
else:
    starrfish1_filtered = STARRFISH(adata1[(adata1.obsm['CRE'] > 0).sum(axis=1) > 0], atac_cpm=adata_cpm)
if os.path.exists('results/starrfish2_filtered.pkl') and load:
    starrfish2_filtered = STARRFISH.load('results/starrfish2_filtered.pkl')
else:
    starrfish2_filtered = STARRFISH(adata2[(adata2.obsm['CRE'] > 0).sum(axis=1) > 0], atac_cpm=adata_cpm)
# %% reload starrfish object, if update utils.py
starrfish1_filtered = reload(starrfish1_filtered)
starrfish2_filtered = reload(starrfish2_filtered)
# %% drop existing test results, if any, specified by to_drop
to_drop = '' # drop nothing
starrfish1_filtered = drop_test(starrfish1_filtered, to_drop)
starrfish2_filtered = drop_test(starrfish2_filtered, to_drop)
# %%
# define CREs to use
lib_size = starrfish2_filtered.lib_size['counts']
# fold to average lib_size
lib_size_fold = lib_size / lib_size.mean()
# remove CREs with less than 5 fold enrichment
cres_to_use_libsize_high = lib_size_fold[(lib_size > 35)].index
# remove CRE217
cres_to_use_libsize_high = cres_to_use_libsize_high[cres_to_use_libsize_high != 'CRE217']
non_negative_control_cres = lib_size.index[~lib_size.index.isin(starrfish2_filtered.get_negative_control_cres())]
non_negative_control_cres_libsize_high = non_negative_control_cres.intersection(cres_to_use_libsize_high)
negative_control_cres = starrfish1_filtered.get_negative_control_cres()
negative_control_cres_libsize_high = negative_control_cres.intersection(cres_to_use_libsize_high)
len(cres_to_use_libsize_high), lib_size.loc[cres_to_use_libsize_high].min()
# %%
# define cell types to use for filtered data
cell_types_counts1 = starrfish1_filtered.get_celltypes().value_counts()
cell_types_counts2 = starrfish2_filtered.get_celltypes().value_counts()
cell_types_to_use_1 = cell_types_counts1[cell_types_counts1 > 500].index
cell_types_to_use_2 = cell_types_counts2[cell_types_counts2 > 500].index
cell_types_to_use = cell_types_to_use_1.intersection(cell_types_to_use_2)
# check the negative control counts for those cell types
negative_control_counts1 = starrfish1_filtered.get_cre_expression()[negative_control_cres].groupby(starrfish1_filtered.get_celltypes()).sum()
negative_control_counts2 = starrfish2_filtered.get_cre_expression()[negative_control_cres].groupby(starrfish2_filtered.get_celltypes()).sum()
negative_control_sum_counts1 = starrfish1_filtered.get_cre_expression()[starrfish1_filtered.get_negative_control_cres()].sum(axis=1).groupby(starrfish1_filtered.get_celltypes()).sum()
negative_control_sum_counts2 = starrfish2_filtered.get_cre_expression()[starrfish2_filtered.get_negative_control_cres()].sum(axis=1).groupby(starrfish2_filtered.get_celltypes()).sum()
# define the cell types by the negative control counts > 50
cell_types_to_use_nc_1 = negative_control_sum_counts1[negative_control_sum_counts1 > 40].index
cell_types_to_use_nc_2 = negative_control_sum_counts2[negative_control_sum_counts2 > 40].index
cell_types_to_use_nc = cell_types_to_use_nc_1.intersection(cell_types_to_use_nc_2)
target_cres = starrfish2_filtered.get_creinfo().index[starrfish2_filtered.get_creinfo()['best_subclass'].isin(cell_types_to_use_nc_2)]
len(cell_types_to_use), len(cell_types_to_use_nc), len(cell_types_to_use_nc_2), len(target_cres)
# %% fold change test
for ct in [None, cell_types_to_use.tolist(), cell_types_to_use_nc.tolist(), cell_types_to_use_1.tolist(), cell_types_to_use_nc_1.tolist()]: 
    fold_change_test_config = {
        "cell_types_to_use": ct,
        "normalize_by_cell_rna": False, "normalize_by_cell_volume": False, 
        "normalize_by_celltype_rna": False, "normalize_by_celltype_volume": False,
        "normalize_by_negative_control": True, 
        "normalize_by_infected_cell": False, "normalize_by_libsize": False,
        "log_transform": False, "rank_transform": None,
        "filter_zero_counts": False,
        "bootstrap_number": 10000, "n_jobs": 296}
    res=starrfish1.fold_change_test(**fold_change_test_config)
    res=starrfish1_filtered.fold_change_test(**fold_change_test_config)
for ct in [None, cell_types_to_use.tolist(), cell_types_to_use_nc.tolist(), cell_types_to_use_2.tolist(), cell_types_to_use_nc_2.tolist()]:
    fold_change_test_config = {
        "cell_types_to_use": ct,
        "normalize_by_cell_rna": False, "normalize_by_cell_volume": False, 
        "normalize_by_celltype_rna": False, "normalize_by_celltype_volume": False,
        "normalize_by_negative_control": True, 
        "normalize_by_infected_cell": False, "normalize_by_libsize": False,
        "log_transform": False, "rank_transform": None,
        "filter_zero_counts": False,
        "bootstrap_number": 10000, "n_jobs": 296}
    res=starrfish2.fold_change_test(**fold_change_test_config)
    res=starrfish2_filtered.fold_change_test(**fold_change_test_config)
# do for cells with high nc
# %% negative control regression test, failed
# neg_controls_to_check = starrfish2_filtered.get_negative_control_cres()
# neg_controls_to_check = neg_controls_to_check[neg_controls_to_check != 'CRE334'].tolist()
# neg_controls_to_check.append('sum')
# neg_control_regression_test_config = {
#     'cell_types_to_use': None,
#     'negative_control': neg_controls_to_check,
#     'normalize_by_cell_rna': False,
#     'normalize_by_cell_volume': False,
#     'normalize_by_celltype_rna': False,
#     'normalize_by_celltype_volume': False,
#     'log_transform': True,
# }
# res2 = starrfish2_filtered.neg_control_regression_test(**neg_control_regression_test_config)
# %% scvi
scvi_test_config = {'use_model': 'SCVI'}
adata2_mvi = starrfish2.scvi(**scvi_test_config)
adata2_mvi = preprocess(adata2_mvi)
starrfish2_mvi = STARRFISH(adata2_mvi, atac_cpm='Data/ATAC_cpm_peakBysubclass.csv')
starrfish2_mvi
# %% save the test results
starrfish1 = reload(starrfish1)
starrfish1.save('results/starrfish1.pkl')
starrfish1_filtered = reload(starrfish1_filtered)
starrfish1_filtered.save('results/starrfish1_filtered.pkl')
starrfish2 = reload(starrfish2)
starrfish2.save('results/starrfish2.pkl')
starrfish2_filtered = reload(starrfish2_filtered)
starrfish2_filtered.save('results/starrfish2_filtered.pkl')
# %% plot figures
def plot_text(ax, toplot, x, y, adjust=True):
    texts = []
    for i, row in toplot.iterrows():
        if row['significant']:
            texts.append(ax.text(row[x], row[y], i, fontsize=8))
    if adjust:
        adjust_text(texts, force_points=0.2, force_text=0.2, expand_points=(1, 1), expand_text=(1, 1),
                    arrowprops=dict(arrowstyle="-", color='black', lw=0.5), ax=ax)

def fetch_data(obj: STARRFISH, test_method, test_configs, normalize_by_lib_size=False) -> pd.DataFrame:
    if test_method == 'fold_change':
        if test_configs is None:
            test_configs = obj.fold_change_test_configs[0]
        test_results = obj.fold_change_test(**test_configs)['celltype_activity']
    elif test_method == 'fisher_exact':
        if test_configs is None:
            test_configs = obj.fisher_exact_test_configs[0]
        test_results = obj.fisher_exact_test(**test_configs)['recall']
    elif test_method == 'fisher_exact_cre':
        if test_configs is None:
            test_configs = obj.fisher_exact_cre_test_configs[0]
        test_results = obj.fisher_exact_cre_test(**test_configs)['recall']
    elif test_method == 'glm':
        if test_configs is None:
            test_configs = obj.glm_test_configs[0]
        test_results = obj.glm_test(**test_configs)['coef']
    elif test_method == 'pseudo_bulk_glm':
        if test_configs is None:
            test_configs = obj.pseudo_bulk_glm_test_configs[0]
        test_results = obj.pseudo_bulk_glm_test(**test_configs)['result']['coef']
        # change anything that is NaN to 0
        test_results = test_results.fillna(0)
        # change anything that is negative to 0
        test_results = test_results.where(test_results > 0, 0)
    elif test_method == 'scvi':
        if test_configs is None:
            test_configs = obj.scvi_configs[0]
        test_results = obj.scvi(**test_configs)
        # average by cell types
        test_results = test_results.obsm['activity'].groupby(test_results.obs['subclass']).mean()
        # only return CREs
        test_results = test_results.loc[:, obj.adata.uns['CRE_info'].index]
    elif test_method == "mixture_model":
        if test_configs is None:
            test_configs = obj.mixture_model_test_configs[0]
        test_results = obj.mixture_model_test(**test_configs)['activity_df']
    # normalize by library size if test_method is not scvi
    if normalize_by_lib_size and not test_method in ['fisher_exact', 'fisher_exact_cre', 'scvi', 'mixture_model']:
        lib_size = np.log1p(obj.lib_size.values + 0.5).reshape(1, -1)
        to_norm = 1 / lib_size
        test_results = test_results * to_norm
        # drop nan and inf cols
        # to_keep = obj.lib_size[obj.lib_size['counts'] >= 10].index
        # test_results = test_results.loc[:, to_keep]
    return test_results

def fetch_data_p(obj: STARRFISH, test_method, test_configs):
    if test_method == 'fold_change':
        if test_configs is None:
            test_configs = obj.fold_change_test_configs[0]
        test_results = obj.fold_change_test(**test_configs)['cre_info']
    elif test_method == 'fisher_exact':
        if test_configs is None:
            test_configs = obj.fisher_exact_test_configs[0]
        test_results = obj.fisher_exact_test(**test_configs)['cre_info']
    elif test_method == 'fisher_exact_cre':
        if test_configs is None:
            test_configs = obj.fisher_exact_cre_test_configs[0]
        test_results = obj.fisher_exact_cre_test(**test_configs)['cre_info']
    elif test_method == 'atac_ontarget_cre':
        if test_configs is None:
            test_configs = obj.atac_ontarget_cre_test_configs[0]
        test_results = obj.atac_ontarget_cre_test(**test_configs)
    elif test_method == 'glm':
        if test_configs is None:
            test_configs = obj.glm_test_configs[0]
        test_results = obj.glm_test(**test_configs)['coef']
    elif test_method == 'pseudo_bulk_glm':
        if test_configs is None:
            test_configs = obj.pseudo_bulk_glm_test_configs[0]
        test_results = obj.pseudo_bulk_glm_test(**test_configs)['result']['pvalue']
        # change anything that is NaN to 0
        test_results = test_results.fillna(1)
        # get cre_info
        cre_info1 = obj.get_creinfo()
        for i in cre_info1.index:
            if cre_info1.loc[i, 'best_subclass'] in test_results.index:
                cre_info1.loc[i, 'p_value'] = test_results.loc[cre_info1.loc[i, 'best_subclass'], i]
        test_results = cre_info1
    elif test_method == 'scvi':
        if test_configs is None:
            test_configs = obj.scvi_configs[0]
        test_results = obj.scvi(**test_configs)
    return test_results

def subplot_cre_corr_compare(ax, cre_corr1, cre_corr2, method, corr_or_p, title=None):
    toplot = pd.DataFrame(index=cre_corr1.index, columns=['corr1', 'pval1', 'corr2', 'pval2'])
    toplot['corr1'] = cre_corr1[f'{method}']
    toplot['pval1'] = cre_corr1[f'{method}_p']
    toplot['corr2'] = cre_corr2[f'{method}']
    toplot['pval2'] = cre_corr2[f'{method}_p']
    # mark significant points and print their names
    toplot['significant'] = (toplot['pval1'] <= 0.05) & (toplot['pval2'] <= 0.05)
    toplot['significant1'] = (toplot['pval1'] <= 0.05)
    toplot['significant2'] = (toplot['pval2'] <= 0.05)
    sns.scatterplot(x=toplot[f'{corr_or_p}1'], y=toplot[f'{corr_or_p}2'], alpha=0.5, ax=ax)
    if corr_or_p != 'pval' or toplot['significant'].sum() <= 15:
        plot_text(ax, toplot, f'{corr_or_p}1', f'{corr_or_p}2')
    # scatter a line on the pval == 0.05
    if corr_or_p == 'pval':
        ax.plot([0.05, 0.05], [0, 1], color='red', linestyle='--')
        ax.plot([0, 1], [0.05, 0.05], color='red', linestyle='--')
        # plot text that count the number of points above the line
        ax.text(0.06, 1, f'{toplot['significant1'].sum()} significant', fontsize=8, color='red')
        ax.text(0.8, 0.06, f'{toplot['significant2'].sum()} significant', fontsize=8, color='red')
        ax.text(0.4, 0.4, f'{toplot['significant'].sum()} both significant', fontsize=8, color='red')
    ax.set_xlabel(f'Experiment 1 {corr_or_p}')
    ax.set_ylabel(f'Experiment 2 {corr_or_p}')
    if title is not None:
        ax.set_title(title)
        
def subplot_cre_corr(ax, cre_corr, corr_or_p, title=None):
    toplot = pd.DataFrame(index=cre_corr.index, columns=['corr1', 'pval1', 'corr2', 'pval2'])
    toplot['corr1'] = cre_corr[f'pearson']
    toplot['pval1'] = cre_corr[f'pearson_p']
    toplot['corr2'] = cre_corr[f'spearman']
    toplot['pval2'] = cre_corr[f'spearman_p']
    # mark significant points and print their names
    toplot['significant'] = (toplot['pval1'] <= 0.05) & (toplot['pval2'] <= 0.05)
    toplot['significant1'] = (toplot['pval1'] <= 0.05)
    toplot['significant2'] = (toplot['pval2'] <= 0.05)
    sns.scatterplot(x=toplot[f'{corr_or_p}1'], y=toplot[f'{corr_or_p}2'], alpha=0.5, ax=ax)
    if corr_or_p != 'pval' or toplot['significant'].sum() <= 15:
        plot_text(ax, toplot, f'{corr_or_p}1', f'{corr_or_p}2')
    # scatter a line on the pval == 0.05
    if corr_or_p == 'pval':
        ax.plot([0.05, 0.05], [0, 1], color='red', linestyle='--')
        ax.plot([0, 1], [0.05, 0.05], color='red', linestyle='--')
        # plot text that count the number of points above the line
        ax.text(0.06, 1, f'{toplot['significant1'].sum()} significant', fontsize=8, color='red')
        ax.text(0.8, 0.06, f'{toplot['significant2'].sum()} significant', fontsize=8, color='red')
        ax.text(0.4, 0.4, f'{toplot['significant'].sum()} both significant', fontsize=8, color='red')
    ax.set_xlabel(f'Pearson {corr_or_p}')
    ax.set_ylabel(f'Spearman {corr_or_p}')
    if title is not None:
        ax.set_title(title)

def plot_atac_cre_corr_compare(obj1: STARRFISH, obj2: STARRFISH, 
                               cell_types_to_use, test_method, test_configs, 
                               log_activity=False, log_atac=True):
    test_results1 = fetch_data(obj1, test_method, test_configs)
    test_results2 = fetch_data(obj2, test_method, test_configs)
    if cell_types_to_use is not None:
        cell_types_to_use = cell_types_to_use[cell_types_to_use.isin(test_results1.index)]
        cell_types_to_use = cell_types_to_use[cell_types_to_use.isin(test_results2.index)]
        test_results1 = test_results1.loc[cell_types_to_use]
        test_results2 = test_results2.loc[cell_types_to_use]
        # corr with atac
        cell_types_to_use = cell_types_to_use.intersection(obj1.atac_cpm.index)
    if cell_types_to_use is None:
        cell_types_to_use1 = obj1.get_celltypes().value_counts()
        cell_types_to_use1 = cell_types_to_use1[cell_types_to_use1 > 50].index
    else:
        cell_types_to_use1 = cell_types_to_use
    cre_corr1, celltype_corr1 = obj1.corr_atac_cpm(
        cell_types_to_use=cell_types_to_use1, 
        acvitity_df=test_results1,
        log_activity=log_activity, log_atac=log_atac
    )
    if cell_types_to_use is None:
        cell_types_to_use2 = obj2.get_celltypes().value_counts()
        cell_types_to_use2 = cell_types_to_use2[cell_types_to_use2 > 50].index
    else:
        cell_types_to_use2 = cell_types_to_use
    cre_corr2, celltype_corr2 = obj2.corr_atac_cpm(
        cell_types_to_use=cell_types_to_use2, 
        acvitity_df=test_results2,
        log_activity=log_activity, log_atac=log_atac
    )
    # add expr1 corr with expr2
    cre_corr, celltype_corr = STARRFISH.corr_starrfish(test_results1, test_results2, log_activity=log_activity)
    fig, ax = plt.subplots(ncols=4, nrows=2, figsize=(24, 10))
    title = f'Spearman Correlation with ATAC'
    if cell_types_to_use is not None:
        title += f' in {len(cell_types_to_use)} cell types'
    subplot_cre_corr_compare(ax[0, 0], cre_corr1, cre_corr2, 'spearman', 'corr', title)
    subplot_cre_corr_compare(ax[1, 0], cre_corr1, cre_corr2, 'spearman', 'pval')
    title = f'Pearson Correlation with ATAC'
    if cell_types_to_use is not None:
        title += f' in {len(cell_types_to_use)} cell types'
    subplot_cre_corr_compare(ax[0, 1], cre_corr1, cre_corr2, 'pearson', 'corr', title)
    subplot_cre_corr_compare(ax[1, 1], cre_corr1, cre_corr2, 'pearson', 'pval')
    title = f'Fisher exact test with ATAC'
    if cell_types_to_use is not None:
        title += f' in {len(cell_types_to_use)} cell types'
    subplot_cre_corr_compare(ax[0, 2], cre_corr1, cre_corr2, 'fisher', 'corr', title)
    subplot_cre_corr_compare(ax[1, 2], cre_corr1, cre_corr2, 'fisher', 'pval')
    # plot corr itself
    subplot_cre_corr(ax[0, 3], cre_corr, 'corr', title='Correlation between experiments')
    subplot_cre_corr(ax[1, 3], cre_corr, 'pval', title='Correlation P-value between experiments')
    # create another figure to show correlation vs log library size
    cre_corr['lib_size'] = np.log10(obj1.lib_size['counts'].loc[cre_corr.index])
    cre_corr['lib_size'] = np.log10(obj1.lib_size['counts'].loc[cre_corr.index])
    cre_corr['significant'] = (cre_corr['spearman_p'] <= 0.05) & (cre_corr['pearson_p'] <= 0.05)
    fig2, ax2 = plt.subplots(ncols=2, nrows=1, figsize=(12, 5))
    sns.scatterplot(x=cre_corr['lib_size'], y=cre_corr['spearman'], alpha=0.5, ax=ax2[0])
    plot_text(ax2[0], cre_corr, 'lib_size', 'spearman')
    ax2[0].set_xlabel('CRE library size (log10)')
    ax2[0].set_ylabel('Spearman correlation between experiments')
    sns.scatterplot(x=cre_corr['lib_size'], y=cre_corr['pearson'], alpha=0.5, ax=ax2[1])
    plot_text(ax2[1], cre_corr, 'lib_size', 'pearson')
    ax2[1].set_xlabel('CRE library size (log10)')
    ax2[1].set_ylabel('Pearson correlation between experiments')
    plt.close(fig)
    plt.close(fig2)
    return fig, fig2, cre_corr1, cre_corr2, cre_corr

def plot_pval_compare(obj1: STARRFISH, obj2: STARRFISH, test_method, test_configs):
    test_results1 = fetch_data_p(obj1, test_method, test_configs)
    test_results2 = fetch_data_p(obj2, test_method, test_configs)
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 5))
    toplot = pd.DataFrame(index=test_results1.index, columns=['pval1', 'pval2'])
    # cap the p-value at 10e-5, anything below that is set to 10e-5
    test_results1['p_value'] = test_results1['p_value'].clip(lower=1e-5)
    test_results2['p_value'] = test_results2['p_value'].clip(lower=1e-5)
    toplot['pval1'] = test_results1['p_value']
    toplot['pval2'] = test_results2['p_value']
    # mark significant points and print their names
    toplot['significant'] = (toplot['pval1'] <= 0.05) & (toplot['pval2'] <= 0.05)
    # log transform p-values
    toplot['pval1'] = -np.log10(toplot['pval1'])
    toplot['pval2'] = -np.log10(toplot['pval2'])
    sns.scatterplot(x=toplot['pval1'], y=toplot['pval2'], alpha=0.5, ax=ax)
    plot_text(ax, toplot, 'pval1', 'pval2', adjust=False)
    # plot a line on the pval == 0.05
    ax.axvline(-np.log10(0.05), color='red', linestyle='--', alpha=0.5)
    ax.axhline(-np.log10(0.05), color='red', linestyle='--', alpha=0.5)
    ax.set_xlabel(f'-Experiment 1 pval (log10)')
    ax.set_ylabel(f'-Experiment 2 pval (log10)')
    # plot the number of significant points
    ax.text(0.8, 0.1, f'{(test_results1['p_value'] <= 0.05).sum()} significant', fontsize=8, color='red', transform=ax.transAxes)
    ax.text(0.1, 0.9, f'{(test_results2['p_value'] <= 0.05).sum()} significant', fontsize=8, color='red', transform=ax.transAxes)
    plt.close(fig)
    return fig

def plot_qval_compare(obj1: STARRFISH, obj2: STARRFISH, test_method, test_configs):
    test_results1 = fetch_data_p(obj1, test_method, test_configs)
    test_results2 = fetch_data_p(obj2, test_method, test_configs)
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 5))
    toplot = pd.DataFrame(index=test_results1.index, columns=['qval1', 'qval2'])
    # cap the p-value at 10e-5, anything below that is set to 10e-5
    test_results1['q_value'] = test_results1['q_value'].clip(lower=1e-5)
    test_results2['q_value'] = test_results2['q_value'].clip(lower=1e-5)
    toplot['qval1'] = test_results1['q_value']
    toplot['qval2'] = test_results2['q_value']
    # mark significant points and print their names
    toplot['significant'] = (toplot['qval1'] <= 0.05) & (toplot['qval2'] <= 0.05)
    # log transform p-values
    toplot['qval1'] = -np.log10(toplot['qval1'])
    toplot['qval2'] = -np.log10(toplot['qval2'])
    sns.scatterplot(x=toplot['qval1'], y=toplot['qval2'], alpha=0.5, ax=ax)
    plot_text(ax, toplot, 'qval1', 'qval2', adjust=False)
    # plot a line on the pval == 0.05
    ax.axvline(-np.log10(0.05), color='red', linestyle='--', alpha=0.5)
    ax.axhline(-np.log10(0.05), color='red', linestyle='--', alpha=0.5)
    ax.set_xlabel(f'-Experiment 1 qval (log10)')
    ax.set_ylabel(f'-Experiment 2 qval (log10)')
    plt.close(fig)
    return fig

def plot_atac_celltype_corr_compare(obj1: STARRFISH, obj2: STARRFISH, 
                                    cell_types_to_use, test_method, test_configs,
                                    log_activity=False, log_atac=True):
    normalize_by_lib_size = True
    if test_method == 'fold_change':
        if test_configs['normalize_by_negative_control'] or test_configs["filter_zero_counts"]:
            normalize_by_lib_size = False
    test_results1 = fetch_data(obj1, test_method, test_configs, normalize_by_lib_size=normalize_by_lib_size)
    test_results2 = fetch_data(obj2, test_method, test_configs, normalize_by_lib_size=normalize_by_lib_size)
    if cell_types_to_use is not None:
        cell_types_to_use = cell_types_to_use[cell_types_to_use.isin(test_results1.index)]
        cell_types_to_use = cell_types_to_use[cell_types_to_use.isin(test_results2.index)]
        test_results1 = test_results1.loc[cell_types_to_use]
        test_results2 = test_results2.loc[cell_types_to_use]
        # corr with atac
        cell_types_to_use = cell_types_to_use.intersection(obj1.atac_cpm.index)
    if cell_types_to_use is None:
        cell_types_to_use1 = obj1.get_celltypes().value_counts().index
    else:
        cell_types_to_use1 = cell_types_to_use
    cre_corr1, celltype_corr1 = obj1.corr_atac_cpm(
        cell_types_to_use=cell_types_to_use1, 
        acvitity_df=test_results1,
        log_activity=log_activity, log_atac=log_atac
    )
    if cell_types_to_use is None:
        cell_types_to_use2 = obj2.get_celltypes().value_counts().index
    else:
        cell_types_to_use2 = cell_types_to_use
    cre_corr2, celltype_corr2 = obj2.corr_atac_cpm(
        cell_types_to_use=cell_types_to_use2, 
        acvitity_df=test_results2,
        log_activity=log_activity, log_atac=log_atac
    )
    # add expr1 corr with expr2
    cre_corr, celltype_corr = STARRFISH.corr_starrfish(test_results1, test_results2, log_activity=log_activity)
    fig, ax = plt.subplots(ncols=4, nrows=2, figsize=(24, 10))
    title = f'Spearman Correlation with ATAC'
    if cell_types_to_use is not None:
        title += f' in {len(cell_types_to_use)} cell types'
    subplot_cre_corr_compare(ax[0, 0], celltype_corr1, celltype_corr2, 'spearman', 'corr', title)
    subplot_cre_corr_compare(ax[1, 0], celltype_corr1, celltype_corr2, 'spearman', 'pval')
    title = f'Pearson Correlation with ATAC'
    if cell_types_to_use is not None:
        title += f' in {len(cell_types_to_use)} cell types'
    subplot_cre_corr_compare(ax[0, 1], celltype_corr1, celltype_corr2, 'pearson', 'corr', title)
    subplot_cre_corr_compare(ax[1, 1], celltype_corr1, celltype_corr2, 'pearson', 'pval')
    title = f'Fisher exact test with ATAC'
    if cell_types_to_use is not None:
        title += f' in {len(cell_types_to_use)} cell types'
    subplot_cre_corr_compare(ax[0, 2], celltype_corr1, celltype_corr2, 'fisher', 'corr', title)
    subplot_cre_corr_compare(ax[1, 2], celltype_corr1, celltype_corr2, 'fisher', 'pval')
    # plot corr itself
    subplot_cre_corr(ax[0, 3], celltype_corr, 'corr', title='Correlation between experiments')
    subplot_cre_corr(ax[1, 3], celltype_corr, 'pval', title='Correlation P-value between experiments')
    # create another figure to show correlation vs cell counts
    celltype_corr['cell_count1'] = np.log10(obj1.get_celltypes().value_counts().loc[celltype_corr.index])
    celltype_corr['cell_count2'] = np.log10(obj2.get_celltypes().value_counts().loc[celltype_corr.index])
    celltype_corr['significant'] = (celltype_corr['spearman_p'] <= 0.05) & (celltype_corr['pearson_p'] <= 0.05)
    fig2, ax2 = plt.subplots(ncols=2, nrows=1, figsize=(12, 5))
    sns.scatterplot(x=celltype_corr['cell_count1'], y=celltype_corr['cell_count2'], palette='Spectral',
                    hue=celltype_corr['pearson'], alpha=0.5, ax=ax2[0])
    ax2[0].set_xlabel('Experiment 1 cell count (log10)')
    ax2[0].set_ylabel('Experiment 2 cell count (log10)')
    sns.scatterplot(x=celltype_corr['cell_count1'], y=celltype_corr['cell_count2'], palette='Spectral',
                    hue=celltype_corr['spearman'], alpha=0.5, ax=ax2[1])
    ax2[1].set_xlabel('Experiment 1 cell count (log10)')
    ax2[1].set_ylabel('Experiment 2 cell count (log10)')
    plt.close(fig)
    plt.close(fig2)
    return fig, fig2, cre_corr1, cre_corr2, cre_corr

def scatter_plot_with_margin_density_by_celltype(test_results1, test_results2, cell_types_to_use,
                                                 ncol, nrow, x_lab='Expr 1 activity', y_lab='Expr 2 acvitiy',
                                                 cell_type_counts1=None, cell_type_counts2=None, log=True,
                                                 filter_zero=True, hist=True, contour=True, 
                                                 negative_control_results1=None, negative_control_results2=None,
                                                 positive_control_info1=None, positive_control_info2=None,
                                                 show_mean_std=True, show_positive_control=True):
    fig = plt.figure(figsize=(ncol * 5, nrow * 5))
    gs = GridSpec(
        3*nrow, 3*ncol,  # 4*rows, 4*columns for layout control
        figure=fig,
        width_ratios=[1, 0.2, 0.25] * ncol,  # Adjust column widths, 0.05 is spacing
        height_ratios=[0.2, 1, 0.25] * nrow,  # Adjust row heights, 0.05 is spacing
        hspace=0.05,
        wspace=0.05
    )
    for i, cell_type in enumerate(cell_types_to_use):
        toplot1 = test_results1.loc[cell_type]
        toplot2 = test_results2.loc[cell_type]
        col_idx = i // ncol
        row_idx = i % ncol
        ax = fig.add_subplot(gs[3*col_idx + 1, 3*row_idx])
        ax_up = fig.add_subplot(gs[3*col_idx, 3*row_idx])
        ax_right = fig.add_subplot(gs[3*col_idx + 1, 3*row_idx+1])
        if filter_zero:
            to_keep = (toplot1 > 0) & (toplot2 > 0)
            toplot1 = toplot1[to_keep]
            toplot2 = toplot2[to_keep]
        # plot main scatter plot
        sns.scatterplot(x=toplot1, y=toplot2, alpha=0.5, ax=ax)
        # if contour is True, plot contour
        if contour:
            sns.kdeplot(x=toplot1, y=toplot2, ax=ax, fill=True, alpha=0.5)
        if hist:
            # # plot histogram, no edge
            sns.histplot(toplot1, ax=ax_up, bins=50, kde=True, legend=False, alpha=0.2, edgecolor=None)
            sns.histplot(y=toplot2, ax=ax_right, bins=50, kde=True, legend=False, alpha=0.2, edgecolor=None)
        else:
            # plot kde, no histogram
            sns.kdeplot(toplot1, ax=ax_up, fill=True, legend=False, alpha=0.2)
            sns.kdeplot(y=toplot2, ax=ax_right, fill=True, legend=False, alpha=0.2)
        # add lines of mean and std
        mean1 = toplot1.mean()
        std1 = toplot1.std()
        mean2 = toplot2.mean()
        std2 = toplot2.std()
        # get max of up and left axis
        max_y = ax.get_ylim()[1]
        min_y = ax.get_ylim()[0]
        max_x = ax.get_xlim()[1]
        min_x = ax.get_xlim()[0]
        # set limits of up and left axis
        ax_up.set_xlim(min_x, max_x)
        ax_right.set_ylim(min_y, max_y)
        # plot text 
        # plot mean and std
        if show_mean_std:
            ax_up.text(mean1, ax_up.get_ylim()[1] * 0.75, 
                    f'{sum((toplot1 < mean1 + 2*std1) & (toplot1>mean1-2*std1))} CREs', 
                    fontsize=8, ha='center')
            ax_right.text(ax_right.get_xlim()[1] * 0.75, mean2, 
                        f'{sum((toplot2 < mean2 + 2*std2) & (toplot2>mean2-2*std2))} CREs',
                        fontsize=8, va='center', rotation=270)
            ax_up.axvline(mean1, color='red', linestyle='--', label='mean', alpha=0.5)
            ax_up.axvline(mean1 + 2*std1, color='blue', linestyle='--', label='std', alpha=0.5)
            ax_up.axvline(mean1 - 2*std1, color='blue', linestyle='--', label='std', alpha=0.5)
            ax_right.axhline(mean2, color='red', linestyle='--', label='mean', alpha=0.5)
            ax_right.axhline(mean2 + 2*std2, color='blue', linestyle='--', label='std', alpha=0.5)
            ax_right.axhline(mean2 - 2*std2, color='blue', linestyle='--', label='std', alpha=0.5)
            # plot mean and std in the main plot
            ax.axvline(mean1, color='red', linestyle='--', label='mean', alpha=0.5)
            ax.axvline(mean1 + 2*std1, color='blue', linestyle='--', label='std', alpha=0.5)
            ax.axvline(mean1 - 2*std1, color='blue', linestyle='--', label='std', alpha=0.5)
            ax.axhline(mean2, color='red', linestyle='--', label='mean', alpha=0.5)
            ax.axhline(mean2 + 2*std2, color='blue', linestyle='--', label='std', alpha=0.5)
            ax.axhline(mean2 - 2*std2, color='blue', linestyle='--', label='std', alpha=0.5)
            # show negative control mean in the main plot, if provided
            if negative_control_results1 is not None:
                neg_mean1 = negative_control_results1.loc[cell_type].mean()
                neg_max1 = negative_control_results1.loc[cell_type].max()
                ax.axvline(neg_mean1, color='grey', linestyle='--', label='neg mean', alpha=0.5)
                ax.axvline(neg_max1, color='green', linestyle='--', label='neg std', alpha=0.5)
            if negative_control_results2 is not None:
                neg_mean2 = negative_control_results2.loc[cell_type].mean()
                neg_max2 = negative_control_results2.loc[cell_type].max()
                ax.axhline(neg_mean2, color='grey', linestyle='--', label='neg mean', alpha=0.5)
                ax.axhline(neg_max2, color='green', linestyle='--', label='neg std', alpha=0.5)
        if positive_control_info1 is not None and show_positive_control:
            pos_cres1 = positive_control_info1.index[positive_control_info1['best_subclass'] == cell_type].intersection(toplot1.index)
            # scatter them, color with red
            sns.scatterplot(x=test_results1.loc[cell_type, pos_cres1], y=test_results2.loc[cell_type, pos_cres1], color='red', ax=ax, label='pos control')
            if hist:
                # plot histogram of positive control
                sns.histplot(test_results1.loc[cell_type, pos_cres1], ax=ax_up, bins=50, kde=True, legend=False, alpha=0.2, edgecolor=None)
                sns.histplot(y=test_results2.loc[cell_type, pos_cres1], ax=ax_right, bins=50, kde=True, legend=False, alpha=0.2, edgecolor=None)
            else:
                # plot kde of positive control
                sns.kdeplot(test_results1.loc[cell_type, pos_cres1], ax=ax_up, fill=True, legend=False, alpha=0.2)
                sns.kdeplot(y=test_results2.loc[cell_type, pos_cres1], ax=ax_right, fill=True, legend=False, alpha=0.2)
        if positive_control_info2 is not None and show_positive_control:
            pos_cres2 = positive_control_info2.index[positive_control_info2['best_subclass'] == cell_type].intersection(toplot2.index)
            # scatter them, color with red
            sns.scatterplot(x=test_results1.loc[cell_type, pos_cres2], y=test_results2.loc[cell_type, pos_cres2], color='red', ax=ax, label='pos control')
            if hist:
                # plot histogram of positive control
                sns.histplot(test_results1.loc[cell_type, pos_cres2], ax=ax_up, bins=50, kde=True, legend=False, alpha=0.2, edgecolor=None)
                sns.histplot(y=test_results2.loc[cell_type, pos_cres2], ax=ax_right, bins=50, kde=True, legend=False, alpha=0.2, edgecolor=None)
            else:
                # plot kde of positive control
                sns.kdeplot(test_results1.loc[cell_type, pos_cres2], ax=ax_up, fill=True, legend=False, alpha=0.2)
                sns.kdeplot(y=test_results2.loc[cell_type, pos_cres2], ax=ax_right, fill=True, legend=False, alpha=0.2)
        # check number of CREs that are both above and below 2 std from the mean
        cre_big1 = toplot1[(toplot1 >= mean1 + 2*std1)]
        cre_big2 = toplot2[(toplot2 >= mean2 + 2*std2)]
        cre_small1 = toplot1[(toplot1 <= mean1 - 2*std1)]
        cre_small2 = toplot2[(toplot2 <= mean2 - 2*std2)]
        cre_big = toplot1[(toplot1 >= mean1 + 2*std1) & (toplot2 >= mean2 + 2*std2)]
        if show_mean_std:
            ax.text(mean1 + 2*std1, mean2 + 2*std2, f'{len(cre_big)} CREs', fontsize=8, ha='left') if len(cre_big) > 0 else None
            ax_up.text(mean1 + 2*std1, ax_up.get_ylim()[1] * 0.75, f'{len(cre_big1)} CREs', fontsize=8, ha='center') if len(cre_big1) > 0 else None
            ax_up.text(mean1 - 2*std1, ax_up.get_ylim()[1] * 0.75, f'{len(cre_small1)} CREs', fontsize=8, ha='center') if len(cre_small1) > 0 else None
            ax_right.text(ax_right.get_xlim()[1] * 0.75, mean2 + 2*std2, f'{len(cre_big2)} CREs', fontsize=8, va='center', rotation=270) if len(cre_big2) > 0 else None
            ax_right.text(ax_right.get_xlim()[1] * 0.75, mean2 - 2*std2, f'{len(cre_small2)} CREs', fontsize=8, va='center', rotation=270) if len(cre_small2) > 0 else None
        ax_up.set_title(cell_type)
        # add spearman and pearson correlation
        tokeep = (np.isfinite(toplot1) & np.isfinite(toplot2))
        if tokeep.sum() >= 2:
            pearson = pearsonr(x=toplot1[tokeep].astype(float), y=toplot2[tokeep].astype(float))
            spearman = spearmanr(a=toplot1[tokeep].astype(float), b=toplot2[tokeep].astype(float))
        else:
            pearson = (np.nan, np.nan)
            spearman = (np.nan, np.nan)
        # if pearson or spearman is significant, color the text red
        sig = []
        if pearson[1] <= 0.05 or spearman[1] <= 0.05:
            color = 'red'
            sig += [cell_type]
        else:
            color = 'black'
        ax.text(0.5, 0.9, 
                f'Pearson: {pearson[0]:.2f} ({pearson[1]:.2e})\nSpearman: {spearman[0]:.2f} ({spearman[1]:.2e})',
                fontsize=8, ha='center', va='center', transform=ax.transAxes, color=color)
        ax_right.set_ylabel('')
        ax_up.set_xlabel('')
        ax_right.set_yticks([])
        ax_up.set_xticks([])
        ax_x_lab = x_lab
        ax_y_lab = y_lab
        if cell_type_counts1 is not None:
            ax_x_lab += f' ({cell_type_counts1[cell_type]} cells)'
        if cell_type_counts2 is not None:
            ax_y_lab += f' ({cell_type_counts2[cell_type]} cells)'
        if log:
            ax_x_lab += ' (log10)'
            ax_y_lab += ' (log10)'
        ax.set_xlabel(ax_x_lab)
        ax.set_ylabel(ax_y_lab)
        leg = ax.get_legend()
        if leg is not None:
            leg.remove()
    fig.tight_layout()
    plt.close(fig)
    print(f'{len(sig)} cell types significant in pearson or spearman')
    print(f'{sig}')
    return fig

def box_plot_by_celltype(test_results1, test_results2, cell_types_to_use, 
                         ncol, nrow, z_threshold=2, x_lab='Expr 1 activity', y_lab='Expr 2 acvitiy',
                         cell_type_counts1=None, cell_type_counts2=None, log=True,
                         filter_zero=True, negative_control_results1=None,
                         filter_by_negative_control_z=None,
                         filter_by_test_results2=None):
    # plot test_results1 on binary version of test_results2
    fig = plt.figure(figsize=(ncol * 5, nrow * 5))
    gs = GridSpec(
        2*nrow, 2*ncol,  # 4*rows, 4*columns for layout control
        figure=fig,
        width_ratios=[1, 0.2] * ncol,  # Adjust column widths, 0.05 is spacing
        height_ratios=[1, 0.2] * nrow,  # Adjust row heights, 0.05 is spacing
        hspace=0.05,
        wspace=0.05
    )
    sig = []
    for i, cell_type in enumerate(cell_types_to_use):
        toplot1 = test_results1.loc[cell_type]
        toplot2 = test_results2.loc[cell_type]
        col_idx = i // ncol
        row_idx = i % ncol
        ax = fig.add_subplot(gs[2*col_idx, 2*row_idx])
        if filter_zero:
            to_keep = (toplot1 > 0) & (toplot2 > 0)
            toplot1 = toplot1[to_keep]
            toplot2 = toplot2[to_keep]
        # get category by z_threshold of toplot2
        toplot2_z = (toplot2 - toplot2.mean()) / toplot2.std()
        # binary version of toplot2
        toplot2_bin = toplot2_z.apply(lambda x: 'z_score ≥ 2' if x >= z_threshold else 'z_score < 2')
        # plot box plot
        if np.isfinite(toplot1).sum() == 0:
            continue
        # show negative control mean in the main plot, if provided
        if negative_control_results1 is not None:
            neg_mean1 = negative_control_results1.loc[cell_type].mean()
            neg_max1 = negative_control_results1.loc[cell_type].max()
            neg_std1 = negative_control_results1.loc[cell_type].std()
            ax.axhline(neg_mean1, color='grey', linestyle='--', label='neg mean', alpha=0.5)
            if filter_by_negative_control_z is not None:
                # filter out cres that are not in the z_threshold
                tokeep = (toplot1 > neg_mean1 + filter_by_negative_control_z * neg_std1)
                toplot1 = toplot1[tokeep]
                toplot2_bin = toplot2_bin[tokeep]
        if filter_by_test_results2 is not None:
            # filter out cres that are not in the z_threshold
            tokeep = (toplot2 > filter_by_test_results2)
            toplot1 = toplot1[tokeep]
            toplot2_bin = toplot2_bin[tokeep]
        sns.boxplot(x=toplot2_bin, y=toplot1, ax=ax, palette='Set2', showfliers=False)
        # do student t-test
        # only do test if there are more than 1 sample in each group
        if len(toplot1[toplot2_bin == 'z_score ≥ 2']) > 1 and len(toplot1[toplot2_bin == 'z_score < 2']) > 1:
            ttest = ttest_ind(toplot1[toplot2_bin == 'z_score ≥ 2'].astype(float), toplot1[toplot2_bin == 'z_score < 2'].astype(float))
        else:
            ttest = None
        # add t-test result to the plot
        # get max of up and left axis
        max_y = ax.get_ylim()[1]
        # check number of CREs that are both above and below 2 std from the mean
        cre_big = sum(toplot2_bin == 'z_score ≥ 2')
        cre_small = sum(toplot2_bin == 'z_score < 2')
        ax.text('z_score ≥ 2', max_y * 0.8, f'{cre_big} CREs', fontsize=8, ha='left')
        ax.text('z_score < 2', max_y * 0.8, f'{cre_small} CREs', fontsize=8, ha='left')
        ax.set_title(cell_type)
        # add t-test result
        if ttest is not None:
            if ttest.pvalue <= 0.05:
                color = 'red'
                sig += [cell_type]
            else:
                color = 'black'
            ax.text(0.5, 0.9, 
                    f't-test: {ttest.statistic:.2f} ({ttest.pvalue:.2e})', color=color,
                    fontsize=8, ha='center', va='center', transform=ax.transAxes)
        ax_x_lab = x_lab
        ax_y_lab = y_lab
        if cell_type_counts1 is not None:
            ax_x_lab += f' ({cell_type_counts1[cell_type]} cells)'
        if cell_type_counts2 is not None:
            ax_y_lab += f' ({cell_type_counts2[cell_type]} cells)'
        if log:
            ax_x_lab += ' (log10)'
            ax_y_lab += ' (log10)'
        # reverse the axis
        ax.set_xlabel(ax_y_lab)
        ax.set_ylabel(ax_x_lab)
    fig.tight_layout()
    plt.close(fig)
    print(f'{len(sig)} cell types have significant difference in activity between z_score ≥ 2 and z_score < 2')
    print(f'{sig}')
    return fig

def scatter_plot_with_margin_density_by_cre(test_results1, test_results2, cres_to_use,
                                            ncol, nrow, x_lab='Expr 1 activity', y_lab='Expr 2 acvitiy',
                                            log1=True, log2=True, filter_zero=True, hist=True, contour=True,
                                            lib_size_counts1=None, lib_size_counts2=None,
                                            positive_control_info1=None, positive_control_info2=None):
    fig = plt.figure(figsize=(ncol * 5, nrow * 5))
    gs = GridSpec(
        3*nrow, 3*ncol,  # 3*rows, 3*columns for layout control
        figure=fig,
        width_ratios=[1, 0.2, 0.25] * ncol,  # Adjust column widths, 0.05 is spacing
        height_ratios=[0.2, 1, 0.25] * nrow,  # Adjust row heights, 0.05 is spacing
        hspace=0.05,
        wspace=0.05
    )
    for i, cre in enumerate(cres_to_use):
        toplot1 = test_results1[cre]
        toplot2 = test_results2[cre]
        col_idx = i // ncol
        row_idx = i % ncol
        ax = fig.add_subplot(gs[3*col_idx + 1, 3*row_idx])
        ax_up = fig.add_subplot(gs[3*col_idx, 3*row_idx])
        ax_right = fig.add_subplot(gs[3*col_idx + 1, 3*row_idx+1])
        if filter_zero:
            to_keep = (toplot1 > 0) & (toplot2 > 0)
            toplot1 = toplot1[to_keep]
            toplot2 = toplot2[to_keep]
        # plot main scatter plot
        sns.scatterplot(x=toplot1, y=toplot2, alpha=0.5, ax=ax)
        # if contour is True, plot contour
        if contour:
            sns.kdeplot(x=toplot1, y=toplot2, ax=ax, fill=True, alpha=0.5)
        if hist:
            # # plot histogram, no edge
            sns.histplot(toplot1, ax=ax_up, bins=50, kde=True, legend=False, alpha=0.2, edgecolor=None)
            sns.histplot(y=toplot2, ax=ax_right, bins=50, kde=True, legend=False, alpha=0.2, edgecolor=None)
        else:
            # plot kde, no histogram
            sns.kdeplot(toplot1, ax=ax_up, fill=True, legend=False, alpha=0.2)
            sns.kdeplot(y=toplot2, ax=ax_right, fill=True, legend=False, alpha=0.2)
        # add lines of mean and std
        mean1 = toplot1.mean()
        std1 = toplot1.std()
        mean2 = toplot2.mean()
        std2 = toplot2.std()
        # get max of up and left axis
        max_y = ax.get_ylim()[1]
        min_y = ax.get_ylim()[0]
        max_x = ax.get_xlim()[1]
        min_x = ax.get_xlim()[0]
        # set limits of up and left axis
        ax_up.set_xlim(min_x, max_x)
        ax_right.set_ylim(min_y, max_y)
        # plot text 
        ax_up.text(mean1, ax_up.get_ylim()[1] * 0.75, 
                   f'{sum((toplot1 < mean1 + 2*std1) & (toplot1>mean1-2*std1))} CellTypes', 
                   fontsize=8, ha='center')
        ax_right.text(ax_right.get_xlim()[1] * 0.75, mean2, 
                     f'{sum((toplot2 < mean2 + 2*std2) & (toplot2>mean2-2*std2))} CellTypes',
                     fontsize=8, va='center', rotation=270)
        # plot mean and std
        ax_up.axvline(mean1, color='red', linestyle='--', label='mean', alpha=0.5)
        ax_up.axvline(mean1 + 2*std1, color='blue', linestyle='--', label='std', alpha=0.5)
        ax_up.axvline(mean1 - 2*std1, color='blue', linestyle='--', label='std', alpha=0.5)
        ax_right.axhline(mean2, color='red', linestyle='--', label='mean', alpha=0.5)
        ax_right.axhline(mean2 + 2*std2, color='blue', linestyle='--', label='std', alpha=0.5)
        ax_right.axhline(mean2 - 2*std2, color='blue', linestyle='--', label='std', alpha=0.5)
        # plot mean and std in the main plot
        ax.axvline(mean1, color='red', linestyle='--', label='mean', alpha=0.5)
        ax.axvline(mean1 + 2*std1, color='blue', linestyle='--', label='std', alpha=0.5)
        ax.axvline(mean1 - 2*std1, color='blue', linestyle='--', label='std', alpha=0.5)
        ax.axhline(mean2, color='red', linestyle='--', label='mean', alpha=0.5)
        ax.axhline(mean2 + 2*std2, color='blue', linestyle='--', label='std', alpha=0.5)
        ax.axhline(mean2 - 2*std2, color='blue', linestyle='--', label='std', alpha=0.5)
        # plot positive_controls
        if positive_control_info1 is not None:
            pos_celltype = positive_control_info1.loc[cre, 'best_subclass']
            if pos_celltype in toplot1.index:
                # scatter them, color with red
                pos_celltype_df = pd.DataFrame({'x': test_results1.loc[pos_celltype, cre], 
                                                'y': test_results2.loc[pos_celltype, cre]},
                                                index=[pos_celltype])
                sns.scatterplot(pos_celltype_df, x='x', y='y', color='red', ax=ax, label='pos control')
        if positive_control_info2 is not None:
            pos_celltype = positive_control_info1.loc[cre, 'best_subclass']
            if pos_celltype in toplot1.index:
                # scatter them, color with red
                pos_celltype_df = pd.DataFrame({'x': test_results1.loc[pos_celltype, cre], 
                                                'y': test_results2.loc[pos_celltype, cre]},
                                                index=[pos_celltype])
                sns.scatterplot(x=test_results1.loc[pos_celltype, cre], y=test_results2.loc[pos_celltype, cre], color='red', ax=ax, label='pos control')
        # check number of CREs that are both above and below 2 std from the mean
        cre_big1 = toplot1[(toplot1 >= mean1 + 2*std1)]
        cre_big2 = toplot2[(toplot2 >= mean2 + 2*std2)]
        cre_small1 = toplot1[(toplot1 <= mean1 - 2*std1)]
        cre_small2 = toplot2[(toplot2 <= mean2 - 2*std2)]
        cre_big = toplot1[(toplot1 >= mean1 + 2*std1) & (toplot2 >= mean2 + 2*std2)]
        ax.text(mean1 + 2*std1, mean2 + 2*std2, f'{len(cre_big)} CellTypes', fontsize=8, ha='left') if len(cre_big) > 0 else None
        ax_up.text(mean1 + 2*std1, ax_up.get_ylim()[1] * 0.75, f'{len(cre_big1)} CellTypes', fontsize=8, ha='center') if len(cre_big1) > 0 else None
        ax_up.text(mean1 - 2*std1, ax_up.get_ylim()[1] * 0.75, f'{len(cre_small1)} CellTypes', fontsize=8, ha='center') if len(cre_small1) > 0 else None
        ax_right.text(ax_right.get_xlim()[1] * 0.75, mean2 + 2*std2, f'{len(cre_big2)} CellTypes', fontsize=8, va='center', rotation=270) if len(cre_big2) > 0 else None
        ax_right.text(ax_right.get_xlim()[1] * 0.75, mean2 - 2*std2, f'{len(cre_small2)} CellTypes', fontsize=8, va='center', rotation=270) if len(cre_small2) > 0 else None
        ax_up.set_title(cre)
        # add spearman and pearson correlation
        pearson = pearsonr(x=toplot1.astype(float), y=toplot2.astype(float))
        spearman = spearmanr(a=toplot1.astype(float), b=toplot2.astype(float))
        ax.text(0.5, 0.9, 
                f'Pearson: {pearson[0]:.2f} ({pearson[1]:.2e})\nSpearman: {spearman[0]:.2f} ({spearman[1]:.2e})',
                fontsize=8, ha='center', va='center', transform=ax.transAxes)
        ax_right.set_ylabel('')
        ax_up.set_xlabel('')
        ax_right.set_yticks([])
        ax_up.set_xticks([])
        ax_x_lab = x_lab
        ax_y_lab = y_lab
        if lib_size_counts1 is not None:
            ax_x_lab += f' (libsize={lib_size_counts1.loc[cre, 'counts']})'
        if lib_size_counts2 is not None:
            ax_y_lab += f' (libsize={lib_size_counts2.loc[cre, 'counts']})'
        if log1:
            ax_x_lab += ' (log10)'
        if log2:
            ax_y_lab += ' (log10)'
        ax.set_xlabel(ax_x_lab)
        ax.set_ylabel(ax_y_lab)
        leg = ax.get_legend()
        if leg is not None:
            leg.remove()
    fig.tight_layout()
    plt.close(fig)
    return fig

def plot_celltype_activity_distribution_compare(obj1: STARRFISH, obj2: STARRFISH, 
                                                cell_types_to_use, cres_to_use, 
                                                test_method, test_configs, 
                                                contour=True, hist=True, log=False, filter_zero=True, 
                                                show_mean_std=True, show_positive_control=True, ncol=8):
    normalize_by_lib_size = True
    if test_method == 'fold_change':
        if test_configs['normalize_by_negative_control'] or test_configs["filter_zero_counts"]:
            normalize_by_lib_size = False
    test_results1 = fetch_data(obj1, test_method, test_configs, normalize_by_lib_size=normalize_by_lib_size)
    test_results2 = fetch_data(obj2, test_method, test_configs, normalize_by_lib_size=normalize_by_lib_size)
    # get cell type counts
    cell_type_counts1 = obj1.get_celltypes().value_counts()
    cell_type_counts2 = obj2.get_celltypes().value_counts()
    if cell_types_to_use is not None:
        cell_types_to_use = cell_types_to_use[cell_types_to_use.isin(test_results1.index)]
        cell_types_to_use = cell_types_to_use[cell_types_to_use.isin(test_results2.index)]
        test_results1 = test_results1.loc[cell_types_to_use]
        test_results2 = test_results2.loc[cell_types_to_use]
    else:
        cell_types_to_use = cell_type_counts1[cell_type_counts1 > 50].index.intersection(
            cell_type_counts2[cell_type_counts2 > 50].index)
        test_results1 = test_results1.loc[cell_types_to_use]
        test_results2 = test_results2.loc[cell_types_to_use]
    # get negative control results
    negative_control_cres = obj1.get_creinfo()
    negative_control_cres = negative_control_cres[negative_control_cres['labeling_type'] == 'negative control'].index
    negative_control_results1 = test_results1[negative_control_cres]
    negative_control_results2 = test_results2[negative_control_cres]
    # align the columns of the two dataframes
    if cres_to_use is not None:
        cres_to_use = pd.Series(cres_to_use)
        cres_to_use = cres_to_use[cres_to_use.isin(test_results1.columns)]
        cres_to_use = cres_to_use[cres_to_use.isin(test_results2.columns)]
    else:
        cres_to_use = test_results1.columns.intersection(test_results2.columns)
    test_results1 = test_results1[cres_to_use]
    test_results2 = test_results2[cres_to_use]
    # log transform the data
    if log:
        test_results1 = np.log10(test_results1.astype(float) + 1)
        test_results2 = np.log10(test_results2.astype(float) + 1)
        negative_control_results1 = np.log10(negative_control_results1.astype(float) + 1)
        negative_control_results2 = np.log10(negative_control_results2.astype(float) + 1)
    # plot distribution of cell type activity
    nrow = int(np.ceil(len(cell_types_to_use) / ncol))
    # change ncol to len(cell_types_to_use) if nrow == 1
    if nrow == 1:
        ncol = len(cell_types_to_use)
    fig = scatter_plot_with_margin_density_by_celltype(
        test_results1, test_results2, cell_types_to_use,
        ncol=ncol, nrow=nrow, 
        cell_type_counts1=cell_type_counts1, cell_type_counts2=cell_type_counts2, 
        log=log, filter_zero=filter_zero, hist=hist, contour=contour, 
        negative_control_results1=negative_control_results1, negative_control_results2=negative_control_results2,
        positive_control_info1=obj1.get_creinfo(),
        show_mean_std=show_mean_std, show_positive_control=show_positive_control)
    plt.close(fig)
    return fig

def plot_celltype_activity_atac_distribution_compare(obj: STARRFISH, cell_types_to_use, cres_to_use, test_method, test_configs, 
                                                     contour=True, hist=True, log=False, filter_zero=True, ncol=8):
    normalize_by_lib_size = True
    if test_method == 'fold_change':
        if test_configs['normalize_by_negative_control'] or test_configs["filter_zero_counts"]:
            normalize_by_lib_size = False
    test_results1 = fetch_data(obj, test_method, test_configs, normalize_by_lib_size=normalize_by_lib_size)
    test_results2 = obj.atac_cpm
    # get cell type counts
    cell_type_counts = obj.get_celltypes().value_counts()
    if cell_types_to_use is not None:
        cell_types_to_use = cell_types_to_use[cell_types_to_use.isin(test_results1.index)]
        cell_types_to_use = cell_types_to_use[cell_types_to_use.isin(test_results2.index)]
    else:
        cell_types_to_use = cell_type_counts[cell_type_counts > 50].index
        cell_types_to_use = cell_types_to_use[cell_types_to_use.isin(test_results2.index)]
    test_results1 = test_results1.loc[cell_types_to_use]
    test_results2 = test_results2.loc[cell_types_to_use]
    # get negative control results
    negative_control_cres = obj.get_creinfo()
    negative_control_cres = negative_control_cres[negative_control_cres['labeling_type'] == 'negative control'].index
    negative_control_cres = negative_control_cres.intersection(test_results1.columns)
    negative_control_results1 = test_results1[negative_control_cres]
    if cres_to_use is not None:
        cres_to_use = pd.Series(cres_to_use)
        cres_to_use = cres_to_use[cres_to_use.isin(test_results1.columns)]
        cres_to_use = cres_to_use[cres_to_use.isin(test_results2.columns)]
    else:
        cres_to_use = test_results1.columns.intersection(test_results2.columns)
    test_results1 = test_results1[cres_to_use]
    test_results2 = test_results2[cres_to_use]
    # log transform the data
    if log:
        test_results1 = np.log10(test_results1.astype(float) + 1)
        test_results2 = np.log10(test_results2.astype(float) + 1)
        negative_control_results1 = np.log10(negative_control_results1.astype(float) + 1)
    # plot distribution of cell type activity
    nrow = int(np.ceil(len(cell_types_to_use) / ncol))
    # change ncol to len(cell_types_to_use) if nrow == 1
    if nrow == 1:
        ncol = len(cell_types_to_use)
    fig = scatter_plot_with_margin_density_by_celltype(
        test_results1, test_results2, cell_types_to_use,
        ncol=ncol, nrow=nrow, x_lab='CRE activity', y_lab='ATAC cpm',
        cell_type_counts1=cell_type_counts, cell_type_counts2=None, log=log,
        filter_zero=filter_zero, hist=hist, contour=contour, 
        negative_control_results1=negative_control_results1,
        positive_control_info1=obj.get_creinfo())
    plt.close(fig)
    # plot another box plot, showing binary version of the data
    fig2 = box_plot_by_celltype(
        test_results1, test_results2, cell_types_to_use,
        ncol=ncol, nrow=nrow, z_threshold=2, x_lab='CRE activity', y_lab='ATAC cpm',
        cell_type_counts1=cell_type_counts, cell_type_counts2=None, log=log,
        filter_zero=filter_zero, negative_control_results1=negative_control_results1,
        filter_by_negative_control_z=None, filter_by_test_results2=None)
    return fig, fig2

def plot_cre_activity_distribution_compare(obj1: STARRFISH, obj2: STARRFISH, cell_types_to_use, cres_to_use, test_method, test_configs, 
                                           contour=True, hist=True, log=False, filter_zero=True, ncol=8):
    test_results1 = fetch_data(obj1, test_method, test_configs)
    test_results2 = fetch_data(obj2, test_method, test_configs)
    # get cell type counts
    cell_type_counts1 = obj1.get_celltypes().value_counts()
    cell_type_counts2 = obj2.get_celltypes().value_counts()
    if cell_types_to_use is not None:
        cell_types_to_use = cell_types_to_use[cell_types_to_use.isin(test_results1.index)]
        cell_types_to_use = cell_types_to_use[cell_types_to_use.isin(test_results2.index)]
        test_results1 = test_results1.loc[cell_types_to_use]
        test_results2 = test_results2.loc[cell_types_to_use]
    else:
        cell_types_to_use = cell_type_counts1[cell_type_counts1 > 50].index.intersection(
            cell_type_counts2[cell_type_counts2 > 50].index)
        test_results1 = test_results1.loc[cell_types_to_use]
        test_results2 = test_results2.loc[cell_types_to_use]
    if cres_to_use is not None:
        cres_to_use = pd.Series(cres_to_use)
        cres_to_use = cres_to_use[cres_to_use.isin(test_results1.columns)]
        cres_to_use = cres_to_use[cres_to_use.isin(test_results2.columns)]
    else:
        cres_to_use = test_results1.columns.intersection(test_results2.columns)
        test_results1 = test_results1[cres_to_use]
        test_results2 = test_results2[cres_to_use]
    # align the columns of the two dataframes
    test_results1 = test_results1[cres_to_use]
    test_results2 = test_results2[cres_to_use]
    # log transform the data
    if log:
        test_results1 = np.log10(test_results1.astype(float) + 1)
        test_results2 = np.log10(test_results2.astype(float) + 1)
    # plot distribution of cell type activity
    nrow = int(np.ceil(test_results1.shape[1] / ncol))
    # change ncol to len(cell_types_to_use) if nrow == 1
    if nrow == 1:
        ncol = test_results1.shape[1]
    fig = scatter_plot_with_margin_density_by_cre(
        test_results1, test_results2, cres_to_use, 
        ncol=ncol, nrow=nrow, log1=log, log2=log,
        filter_zero=filter_zero, hist=hist, contour=contour)
    plt.close(fig)
    return fig

def plot_cre_activity_atac_distribution_compare(obj: STARRFISH, cell_types_to_use, cres_to_use, test_method, test_configs, 
                                                     contour=True, hist=True, log1=False, log2=False, filter_zero=True, ncol=8):
    test_results1 = fetch_data(obj, test_method, test_configs)
    test_results2 = obj.atac_cpm
    # get cell type counts
    cell_type_counts = obj.get_celltypes().value_counts()
    if cell_types_to_use is not None:
        cell_types_to_use = pd.Series(cell_types_to_use)
        cell_types_to_use = cell_types_to_use[cell_types_to_use.isin(test_results1.index)]
        cell_types_to_use = cell_types_to_use[cell_types_to_use.isin(test_results2.index)]
        test_results1 = test_results1.loc[cell_types_to_use]
        test_results2 = test_results2.loc[cell_types_to_use]
    else:
        cell_types_to_use = cell_type_counts[cell_type_counts > 50].index
        cell_types_to_use = cell_types_to_use[cell_types_to_use.isin(test_results2.index)]
        test_results1 = test_results1.loc[cell_types_to_use]
        test_results2 = test_results2.loc[cell_types_to_use]
    if cres_to_use is not None:
        cres_to_use = pd.Series(cres_to_use)
        cres_to_use = cres_to_use[cres_to_use.isin(test_results1.columns)]
        cres_to_use = cres_to_use[cres_to_use.isin(test_results2.columns)]
    else:
        cres_to_use = test_results1.columns.intersection(test_results2.columns)
        test_results1 = test_results1[cres_to_use]
        test_results2 = test_results2[cres_to_use]
    # align the columns of the two dataframes
    test_results1 = test_results1[cres_to_use]
    test_results2 = test_results2[cres_to_use]
    # log transform the data
    if log1:
        test_results1 = np.log10(test_results1.astype(float) + 1)
    if log2:
        test_results2 = np.log10(test_results2.astype(float) + 1)
    # plot distribution of cell type activity
    nrow = int(np.ceil(len(cres_to_use) / ncol))
    # change ncol to len(cell_types_to_use) if nrow == 1
    if nrow == 1:
        ncol = len(cres_to_use)
    fig = scatter_plot_with_margin_density_by_cre(
        test_results1, test_results2, cres_to_use, 
        ncol=ncol, nrow=nrow, x_lab='CRE activity', y_lab='ATAC cpm', log1=log1, log2=log2,
        filter_zero=filter_zero, hist=hist, contour=contour, 
        lib_size_counts1=obj.lib_size,
        positive_control_info1=obj.get_creinfo())
    plt.close(fig)
    return fig

def plot_all_and_save(obj1, obj2, cell_types_to_use=None, cres_to_use=None, 
                      test_method='fold_change', test_configs=None,
                      log_activity=False, log_atac=False, filter_zero=True):
    os.makedirs(f'results/{test_method}', exist_ok=True)
    fig, fig2, cre_corr1, cre_corr2, cre_corr = plot_atac_cre_corr_compare(
        obj1, obj2, cell_types_to_use=cell_types_to_use, log_activity=log_activity, log_atac=log_atac,
        test_method=test_method, test_configs=test_configs)
    plt.close(fig)
    plt.close(fig2)
    fig.savefig(f'results/{test_method}/atac_cre.pdf')
    fig2.savefig(f'results/{test_method}/experiments_cre.pdf')
    # get both significant cres
    expr1_significant_cres = cre_corr1[cre_corr1['pearson_p'] <= 0.05].index
    expr2_significant_cres = cre_corr2[cre_corr2['pearson_p'] <= 0.05].index
    both_significant_cres = expr1_significant_cres.intersection(expr2_significant_cres)
    fig, fig2, celltype_corr1, celltype_corr2, celltype_corr = plot_atac_celltype_corr_compare(
        obj1, obj2, cell_types_to_use=cell_types_to_use, log_activity=log_activity, log_atac=log_atac,
        test_method=test_method, test_configs=test_configs)
    fig.savefig(f'results/{test_method}/atac_celltype.pdf')
    fig2.savefig(f'results/{test_method}/experiments_celltype.pdf')
    # get both significant cell types
    # expr1_significant_cell_types = celltype_corr1[celltype_corr1['pearson_p'] <= 0.05].index
    # expr2_significant_cell_types = celltype_corr2[celltype_corr2['pearson_p'] <= 0.05].index
    # both_significant_cell_types = expr1_significant_cell_types.intersection(expr2_significant_cell_types)
    # visualize the distributions of activity and atac
    fig3 = plot_celltype_activity_distribution_compare(
        obj1, obj2, cell_types_to_use=cell_types_to_use,
        test_method=test_method, test_configs=test_configs, log=log_activity, filter_zero=filter_zero)
    fig3.savefig(f'results/{test_method}/experiments_celltype_distribution.pdf')
    fig4, fig42 = plot_celltype_activity_atac_distribution_compare(
        obj1, cell_types_to_use=cell_types_to_use, cres_to_use=cres_to_use,
        test_method=test_method, test_configs=test_configs, log=log_activity, filter_zero=filter_zero)
    fig4.savefig(f'results/{test_method}/expr1_celltype_distribution.pdf')
    fig42.savefig(f'results/{test_method}/expr1_celltype_distribution_box.pdf')
    fig5, fig52 = plot_celltype_activity_atac_distribution_compare(
        obj2, cell_types_to_use=cell_types_to_use, cres_to_use=cres_to_use,
        test_method=test_method, test_configs=test_configs, log=log_activity, filter_zero=filter_zero)
    fig5.savefig(f'results/{test_method}/expr2_celltype_distribution.pdf')
    fig52.savefig(f'results/{test_method}/expr2_celltype_distribution_box.pdf')
    fig3 = plot_cre_activity_distribution_compare(
        obj1, obj2, cres_to_use=cres_to_use,
        cell_types_to_use=cell_types_to_use,
        test_method=test_method, test_configs=test_configs, log=log_activity, filter_zero=filter_zero)
    fig3.savefig(f'results/{test_method}/experiments_cre_distribution.pdf')
    fig4 = plot_cre_activity_atac_distribution_compare(
        obj1, cell_types_to_use=cell_types_to_use, cres_to_use=cres_to_use,
        test_method=test_method, test_configs=test_configs, log=log_activity, filter_zero=filter_zero)
    fig4.savefig(f'results/{test_method}/expr1_cre_distribution.pdf')
    fig5 = plot_cre_activity_atac_distribution_compare(
        obj2, cell_types_to_use=cell_types_to_use, cres_to_use=cres_to_use,
        test_method=test_method, test_configs=test_configs, log=log_activity, filter_zero=filter_zero)
    fig5.savefig(f'results/{test_method}/expr2_cre_distribution.pdf')
    return expr2_significant_cres

# check the negative control counts for those cell types
def negative_control_regression_plot(obj, cell_types_to_check):
    counts2 = obj.get_cre_expression().groupby(obj.get_celltypes()).sum()
    negative_control_counts2 = counts2[obj.get_negative_control_cres()]
    negative_control_counts2['sum'] = negative_control_counts2.sum(axis=1)
    negative_control_counts2['zero'] = 0
    # in each cell type, check the linear relationship between negative control counts and library size
    ncols=8
    neg_controls_to_check = obj.get_negative_control_cres()
    other_to_check = obj.get_creinfo().index
    # remove CRE217
    other_to_check = other_to_check[other_to_check != 'CRE217']
    neg_controls_to_check = neg_controls_to_check[neg_controls_to_check != 'CRE334']
    nrows = int(np.ceil(len(cell_types_to_check) / ncols))
    fig, ax = plt.subplots(ncols=ncols, nrows=nrows, figsize=(4*ncols, 4*nrows))
    lib_size = obj.lib_size['counts']
    ng_libsize = lib_size.loc[neg_controls_to_check]
    # ng_libsize.loc['sum'] = ng_libsize.sum()
    # ng_libsize.loc['zero'] = 0
    neg_controls_to_check = neg_controls_to_check.tolist()
    # neg_controls_to_check.append('sum')
    # neg_controls_to_check.append('zero')
    sig = 0
    for i, cell_type in enumerate(cell_types_to_check):
        ax_ = ax[i//ncols, i%ncols]
        neg_sum = negative_control_counts2.loc[cell_type, 'sum'].copy()
        # pos_controls_to_check = starrfish2_filtered.get_positive_control_cres(cell_type)
        pos_controls_to_check = obj.get_atac_z_cres(cell_type, 2)
        # remove CRE217
        if pos_controls_to_check is not None:
            pos_controls_to_check = pos_controls_to_check[pos_controls_to_check != 'CRE217']
        cell_type_ng_counts = np.log1p(negative_control_counts2.loc[cell_type, neg_controls_to_check])
        cell_type_counts = np.log1p(counts2.loc[cell_type, other_to_check])
        # Fit model WITH INTERCEPT
        X = sm.add_constant(ng_libsize)  # Add intercept term
        model = sm.OLS(cell_type_ng_counts, X).fit()
        # Generate predictions (include intercept)
        x_vals = np.linspace(np.log1p(lib_size).min(), np.log1p(lib_size).max(), 100)
        X_pred = sm.add_constant(x_vals)
        predictions = model.get_prediction(X_pred)
        predicted_means = predictions.predicted_mean
        conf_int = predictions.conf_int(alpha=0.05)
        # add a linear regression line
        # sns.scatterplot(x=np.log1p(ng_libsize), y=cell_type_ng_counts, ax=ax_, color='blue')
        sns.scatterplot(x=ng_libsize, y=cell_type_ng_counts, ax=ax_, color='blue')
        # plot all other CREs
        sns.scatterplot(x=np.log1p(lib_size.loc[other_to_check]), y=cell_type_counts, ax=ax_, color='gray', alpha=0.5)
        # plot positive controls
        if pos_controls_to_check is not None:
            sns.scatterplot(x=np.log1p(lib_size.loc[pos_controls_to_check]), y=cell_type_counts.loc[pos_controls_to_check], ax=ax_, color='red', alpha=0.5)
        # plot regression line
        ax_.plot(x_vals, predicted_means, color='blue', linewidth=2)
        # Plot confidence interval
        ax_.fill_between(x_vals, conf_int[:, 0], conf_int[:, 1],  color='blue', alpha=0.1)
        # pearson correlation and p-value
        pearson = pearsonr(ng_libsize, cell_type_ng_counts)
        ax_.annotate(f'pearson: {pearson[0]:.2f}', xy=(0.05, 0.95), xycoords='axes fraction', fontsize=12, ha='left', va='top')
        # spearman correlation
        spearman = spearmanr(ng_libsize, cell_type_ng_counts)
        ax_.annotate(f'spearman: {spearman[0]:.2f}', xy=(0.05, 0.85), xycoords='axes fraction', fontsize=12, ha='left', va='top')
        ax_.set_title(cell_type)
        ax_.set_xlabel('library size (log)')
        ax_.set_ylabel('total counts (log)')
    fig.tight_layout()
    plt.close(fig)
    return fig

def cre_corr_dotplot(obj, cres_to_use, cell_types_to_use, test_method, test_configs, 
                scale_by_cre = True, z_score_by_cre = True, figsize=(20, 12)):
    test_result = fetch_data(obj, test_method, test_configs)
    test_result = np.log1p(test_result[cres_to_use])
    atac_cpm = np.log1p(obj.atac_cpm[cres_to_use])
    if cell_types_to_use is None:
        cell_types_to_use = test_result.index
    cell_types_to_use = cell_types_to_use[cell_types_to_use.isin(test_result.index)]
    cell_types_to_use = cell_types_to_use[cell_types_to_use.isin(atac_cpm.index)]
    # order cell types to use by subcluster number
    cluster_annotation_term = pd.read_csv('Data/abc_atlas/cluster_annotation_term.csv', index_col=0)
    cluster_annotation_term['subclass'] = cluster_annotation_term['subclass'].str.replace('/', '-')
    if obj.celltype_tag == 'obs:subclass' and not hasattr(obj, 'celltype_tag_orig'):
        cell_types_to_use_cluster_number = cluster_annotation_term['subclass_number'].groupby(cluster_annotation_term['subclass']).first().loc[cell_types_to_use].values
    elif obj.celltype_tag == 'obs:class':
        cell_types_to_use_cluster_number = cluster_annotation_term['class_number'].groupby(cluster_annotation_term['class']).first().loc[cell_types_to_use].values
    elif hasattr(obj, 'celltype_tag_orig'):
        # rename subclass to allen institute AAV screen paper
        subclass_rename = pd.read_excel('Data/abc_atlas/allen_institute_subclass_rename.xlsx')
        subclass_rename['subclass_simple_label'] = subclass_rename['subclass_simple_label'].str.replace('/', '-')
        subclass_rename['subclass_label'] = subclass_rename['subclass_label'].str.replace('/', '-')
        # filter to non-NaN
        subclass_rename = subclass_rename[subclass_rename['subclass_simple_label'].notna()]
        cell_types_to_use_cluster_number = subclass_rename['subclass_simple_label_number'].groupby(subclass_rename['subclass_simple_label']).first().loc[cell_types_to_use].values
    # reorder cell types to use by subcluster number
    cell_types_to_use = pd.Index(cell_types_to_use[np.argsort(cell_types_to_use_cluster_number)])
    # reorder the cres by the best recall cell type
    cres_best_recall_celltype = atac_cpm.loc[cell_types_to_use].idxmax(axis=0)
    # cres_best_recall_celltype = obj.get_creinfo()['best_subclass'].loc[significant_cres]
    cres_best_recall_celltype_idx = cell_types_to_use.get_indexer(cres_best_recall_celltype)
    cres_to_use = cres_to_use[np.argsort(cres_best_recall_celltype_idx)]
    # reorder test result and atac cpm
    test_result = test_result[cres_to_use]
    atac_cpm = atac_cpm[cres_to_use]
    test_result = test_result.loc[cell_types_to_use]
    atac_cpm = atac_cpm.loc[cell_types_to_use]
    # if scale_by_cre, we scale the test result by the max of each cre
    size_name = 'ATAC cpm (log)'
    hue_name = 'activity (log)'
    if scale_by_cre:
        test_result = test_result / test_result.max(axis=0)
        atac_cpm = atac_cpm / atac_cpm.max(axis=0)
        size_name += ' (scaled)'
        hue_name += ' (scaled)'
    if z_score_by_cre:
        # scale test result to z-score along the cres
        test_result = test_result.transpose()
        test_result = (test_result - test_result.mean(axis=0)) / test_result.std(axis=0)
        test_result = test_result.transpose()
        # scale atac_cpm to z-score along the cres
        atac_cpm = atac_cpm.transpose()
        atac_cpm = (atac_cpm - atac_cpm.mean(axis=0)) / atac_cpm.std(axis=0)
        atac_cpm = atac_cpm.transpose()
        hue_name += ' (z-score)'
    # find positive controls
    positive_control_info = obj.get_creinfo()
    positive_control_df = pd.DataFrame(index=test_result.index, columns=test_result.columns)
    print(positive_control_df.shape)
    for cre in test_result.columns:
        if positive_control_info.loc[cre, 'best_subclass'] in test_result.index:
            positive_control_df.loc[positive_control_info.loc[cre, 'best_subclass'], cre] = True
    positive_control_df = positive_control_df.fillna(False)
    # make to plot dataframe, flatten recall and atac_cpm
    toplot = pd.DataFrame({'activity': test_result.values.flatten(),
                           'atac_cpm': atac_cpm.loc[cell_types_to_use].values.flatten(),
                           'cell_types': cell_types_to_use.values.repeat(len(cres_to_use)),
                           'cres': np.tile(cres_to_use, len(cell_types_to_use)),
                           'positive_control': positive_control_df.values.flatten()})
    # rename columns
    toplot.rename(columns={'activity': hue_name,
                           'atac_cpm': size_name}, inplace=True)
    # plot dot plot, no edge color
    fig, ax = plt.subplots(figsize=figsize)
    sns.scatterplot(data=toplot, x='cell_types', y='cres', size=size_name, hue=hue_name, edgecolor='none',
                    sizes=(5, 500), alpha=0.8)
    # scatter positive controls
    sns.scatterplot(data=toplot[toplot['positive_control'] == True], 
                    x='cell_types', y='cres', s=500, alpha=0.8, marker='s', facecolor='none', edgecolor='red')
    plt.xticks(rotation=90)
    plt.xlabel('Cell Types')
    plt.ylabel('CREs')
    # legend position to the bottom and horizontal
    ncol = 7  # Number of columns you want
    handles, labels = ax.get_legend_handles_labels()
    handles = np.array(handles).reshape(-1, ncol).T.flatten()
    labels = np.array(labels).reshape(-1, ncol).T.flatten()
    ax.legend(handles, labels, loc='lower center', ncol=ncol,
              bbox_to_anchor=(0.5, -0.25), fontsize=10, labelspacing=1.5)
    fig.tight_layout()
    plt.close(fig)
    return fig

def celltype_corr_dotplot(obj, cres_to_use, cell_types_to_use, test_method, test_configs, 
                     filter_celltype_activate=None, filter_celltype_recall=None,
                     fig_size=(20, 12)):
    if cres_to_use is None:
        cres_to_use = obj.get_creinfo().index
    # if all cres_to_use are not in atac columns, then just plot the activity
    if len(cres_to_use[cres_to_use.isin(obj.atac_cpm.columns)]) == 0:
        atac_cpm = obj.atac_cpm.transpose().copy()
        atac_cpm = atac_cpm.reindex(cres_to_use, fill_value=1)
        atac_cpm = atac_cpm.transpose()
        no_atac = True
    else:
        no_atac = False
        atac_cpm = np.log1p(obj.atac_cpm[cres_to_use])
        cres_to_use = cres_to_use[cres_to_use.isin(obj.atac_cpm.columns)]
    if test_method in ['fisher_exact', 'fisher_exact_cre']:
        test_result = getattr(obj, f'{test_method}_test')(**test_configs)
        test_result_recall_n = test_result['recall_n'][cres_to_use]
        test_result_activated_n = test_result['recall_n'] * test_result['recall']
        test_result = test_result['recall'][cres_to_use]
    else:
        test_result = fetch_data(obj, test_method, test_configs)
        test_result = test_result[cres_to_use]
    test_result = np.log1p(test_result)
    if cell_types_to_use is None:
        cell_types_to_use = test_result.index
    cell_types_to_use = cell_types_to_use[cell_types_to_use.isin(test_result.index)]
    cell_types_to_use = cell_types_to_use[cell_types_to_use.isin(atac_cpm.index)]
    # if filter_celltype, we filter out the cell types that average recall is less than 0.1
    if filter_celltype_activate is not None:
        # filter out the cell types with test_result_n < filter_celltype_activate
        cell_types_to_use_total_activate = test_result_activated_n.loc[cell_types_to_use].sum(axis=1)
        cell_types_to_use = cell_types_to_use[cell_types_to_use_total_activate > filter_celltype_activate]
    if filter_celltype_recall is not None and test_method in ['fisher_exact', 'fisher_exact_cre']:
        # filter out the cell types with sum recall < filter_celltype_recall
        cell_types_to_use_max_recall = test_result.loc[cell_types_to_use].max(axis=1)
        cell_types_to_use = cell_types_to_use[cell_types_to_use_max_recall > filter_celltype_recall]
    # order cell types to use by subcluster number
    cluster_annotation_term = pd.read_csv('Data/abc_atlas/cluster_annotation_term.csv', index_col=0)
    cluster_annotation_term['subclass'] = cluster_annotation_term['subclass'].str.replace('/', '-')
    if obj.celltype_tag == 'obs:subclass' and not hasattr(obj, 'celltype_tag_orig'):
        cell_types_to_use_cluster_number = cluster_annotation_term['subclass_number'].groupby(cluster_annotation_term['subclass']).first().loc[cell_types_to_use].values
    elif obj.celltype_tag == 'obs:class':
        cell_types_to_use_cluster_number = cluster_annotation_term['class_number'].groupby(cluster_annotation_term['class']).first().loc[cell_types_to_use].values
    elif hasattr(obj, 'celltype_tag_orig'):
        # rename subclass to allen institute AAV screen paper
        subclass_rename = pd.read_excel('Data/abc_atlas/allen_institute_subclass_rename.xlsx')
        subclass_rename['subclass_simple_label'] = subclass_rename['subclass_simple_label'].str.replace('/', '-')
        subclass_rename['subclass_label'] = subclass_rename['subclass_label'].str.replace('/', '-')
        # filter to non-NaN
        subclass_rename = subclass_rename[subclass_rename['subclass_simple_label'].notna()]
        cell_types_to_use_cluster_number = subclass_rename['subclass_simple_label_number'].groupby(subclass_rename['subclass_simple_label']).first().loc[cell_types_to_use].values
    # reorder cell types to use by subcluster number
    cell_types_to_use = pd.Index(cell_types_to_use[np.argsort(cell_types_to_use_cluster_number)])
    # reorder the cres by the best recall cell type
    if not no_atac:
        cres_best_recall_celltype = atac_cpm.loc[cell_types_to_use].idxmax(axis=0)
        # cres_best_recall_celltype = obj.get_creinfo()['best_subclass'].loc[significant_cres]
        cres_best_recall_celltype_idx = cell_types_to_use.get_indexer(cres_best_recall_celltype)
        cres_to_use = cres_to_use[np.argsort(cres_best_recall_celltype_idx)]
    # reorder test result and atac cpm
    test_result = test_result[cres_to_use]
    atac_cpm = atac_cpm[cres_to_use]
    # change significant cres to add the number of cells activated
    if test_method in ['fisher_exact', 'fisher_exact_cre']:
        cres_to_use = [f'{cre} ({test_result_recall_n[cre].iloc[0]} cells)' for cre in cres_to_use]
    # scale activity to z-score along the cell types
    test_result = test_result.transpose()
    test_result = (test_result - test_result.mean(axis=0)) / test_result.std(axis=0)
    test_result = test_result.transpose()
    # scale atac_cpm to z-score along the cell types
    if not no_atac:
        atac_cpm = atac_cpm.transpose()
        atac_cpm = (atac_cpm - atac_cpm.mean(axis=0)) / atac_cpm.std(axis=0)
        atac_cpm = atac_cpm.transpose()
    # after dropping of cell types, re-calculate the specificity
    # test_result = test_result.loc[cell_types_to_use]
    # test_result = test_result / test_result.sum(axis=0)
    # make to plot dataframe, flatten recall and atac_cpm
    toplot = pd.DataFrame({'activity (z-score)': test_result.loc[cell_types_to_use].values.flatten(),
                           'atac_cpm (log)': atac_cpm.loc[cell_types_to_use].values.flatten(),
                           'cell_types': cell_types_to_use.values.repeat(len(cres_to_use)),
                           'cres': np.tile(cres_to_use, len(cell_types_to_use))})
    # plot dot plot
    fig, ax = plt.subplots(figsize=fig_size)
    sns.scatterplot(data=toplot, x='cell_types', y='cres', size='atac_cpm (log)', hue='activity (z-score)', 
                    sizes=(5, 500), alpha=0.8)
    plt.xticks(rotation=90)
    plt.xlabel('CREs')
    plt.ylabel('Cell Types')
    # legend position to the right
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10, labelspacing=1.5)
    plt.close(fig)
    return fig

def draw_custom_dendrogram(cre_order_token, ordered_cres, ax, reorder_penalty=0.05):
    # Step 1: Get unique tokens and map to CREs
    unique_tokens, inverse_indices = np.unique(cre_order_token, axis=0, return_inverse=True)
    token_groups = {i: np.where(inverse_indices == i)[0] for i in range(len(unique_tokens))}

    n = len(ordered_cres)
    dist_matrix = np.ones((n, n))
    np.fill_diagonal(dist_matrix, 0)

    # Step 2: Set small intra-group distances
    for group in token_groups.values():
        for i in group:
            for j in group:
                if i != j:
                    dist_matrix[i, j] = 0.1

    # Step 3: Add token dissimilarity + position-based penalty
    for i in range(n):
        for j in range(i + 1, n):
            if dist_matrix[i, j] > 0.1:  # Skip already close intra-group CREs
                # Token-based distance
                token_dist = np.linalg.norm(cre_order_token[i] - cre_order_token[j])
                # Position-based penalty
                order_dist = reorder_penalty * abs(i - j)
                total_dist = token_dist + order_dist
                dist_matrix[i, j] = dist_matrix[j, i] = total_dist

    # Step 4: Cluster and draw
    D = squareform(dist_matrix)
    Z = linkage(D, method='average')
    Z_opt = optimal_leaf_ordering(Z, D)
    dendrogram(Z_opt, labels=ordered_cres, orientation='left', ax=ax, color_threshold=0.5 * np.max(Z_opt[:, 2]))
    ax.set_yticklabels([])
    ax.tick_params(left=False)

def cre_pval_dotplot(q_value, activity, cres_to_use, cell_types_to_use, positive_control_info, figsize=(20, 12)):
    if cres_to_use is None:
        cres_to_use = q_value.columns
    if cell_types_to_use is None:
        cell_types_to_use = q_value.index
    cell_types_to_use = cell_types_to_use[cell_types_to_use.isin(q_value.index)]
    # order cell types to use by subcluster number
    cluster_annotation_term = pd.read_csv('Data/abc_atlas/cluster_annotation_term.csv', index_col=0)
    cluster_annotation_term['subclass'] = cluster_annotation_term['subclass'].str.replace('/', '-')
    cell_types_to_use_cluster_number = cluster_annotation_term['subclass_number'].groupby(cluster_annotation_term['subclass']).first().loc[cell_types_to_use].values
    # reorder cell types to use by subcluster number
    cell_types_to_use = pd.Index(cell_types_to_use[np.argsort(cell_types_to_use_cluster_number)])
    # reorder test result and atac cpm
    q_value = q_value.loc[cell_types_to_use, cres_to_use]
    activity = activity.loc[cell_types_to_use, cres_to_use]
    # find positive controls
    if positive_control_info is not None:
        target_df = pd.DataFrame(index=q_value.index, columns=q_value.columns)
        for cre in q_value.columns:
            best_subclass = positive_control_info.loc[cre, 'best_subclass']
            if best_subclass == 'Negative Control':
                # if the best subclass is negative control, we skip this cre
                target_df.loc[:, cre] = 'Negative Control'
                continue
            elif best_subclass == 'CRE':
                # if the best subclass is CREs, we skip this cre
                target_df.loc[:, cre] = 'CRE'
                continue
            # significant cres
            cell_types = q_value[cre].index[q_value[cre] <= 0.01]
            # on target cell types
            if best_subclass in cell_types:
                target_df.loc[best_subclass, cre] = 'on-target'
                cell_types = cell_types[cell_types != best_subclass]
            else:
                if best_subclass in q_value.index:
                    target_df.loc[best_subclass, cre] = 'miss'
            # other cell types are off-target
            cell_types = cell_types[cell_types.isin(q_value.index)]
            if len(cell_types) > 0:
                target_df.loc[cell_types, cre] = 'off-target'
        # divide the cres into 4 categories: only on-target, on-target + off-target, only off-target, no target
        def assign_cre_type(x):
            if 'Negative Control' in x.values:
                return 'Negative Controls'
            elif 'CRE' in x.values:
                return 'CREs'
            elif 'on-target' in x.values:
                if 'off-target' in x.values:
                    return 'Mix-target'
                else:
                    return 'On-target'
            elif 'off-target' in x.values:
                return 'Off-target'
            else:
                return 'No target'
        cre_type = target_df.apply(assign_cre_type, axis=0)
    else:
        target_df = pd.DataFrame(index=q_value.index, columns=q_value.columns)
        target_df[:] = 'CREs'
        cre_type = np.repeat('CREs', len(cres_to_use))
    # Data transformations
    q_value = q_value.clip(lower=1/5000).astype(float)  # Clip to avoid log10(0)
    activity = activity.clip(lower=1e-2).astype(float)  # Clip to avoid log10(0)
    activity = np.log10(activity).T  # Transpose for CRE clustering
    activity = activity.sub(activity.mean(axis=1), axis=0).div(activity.std(axis=1), axis=0)  # Z-score per CRE
    q_value = -np.log10(q_value).T
    # if scale_by_cre, we scale the test result by the max of each cre
    hue_name = 'log(activity) (z-score)'
    size_name = '-log10(q value)'
    # make to plot dataframe, flatten recall and atac_cpm
    toplot = pd.DataFrame({size_name: q_value.T.values.flatten(),
                           hue_name: activity.T.values.flatten(),
                           'cre_type': np.tile(cre_type, len(cell_types_to_use))})
    hue_min, hue_max = toplot[hue_name].min(), toplot[hue_name].max()
    size_min, size_max = toplot[size_name].min(), toplot[size_name].max()
    # plot dot plot, no edge color, use 4 sub plots
    # Set the height ratios for the subplots
    cre_categories = np.array(['On-target', 'Mix-target', 'Off-target', 'No target', 'CREs', 'Negative Controls'])
    height_ratios = np.array([sum(cre_type == category) for category in cre_categories])
    # remove the categories with 0 counts
    cre_categories = cre_categories[height_ratios != 0]
    height_ratios = height_ratios[height_ratios != 0]
    # Create 4 subplots aligned vertically, assign different heights
    fig, axes = plt.subplots(len(height_ratios), 2, figsize=figsize, sharex=False, sharey=False,
                             gridspec_kw={'height_ratios': height_ratios, 'width_ratios': [0.2, 1]})
    # Plot each CRE type in a subplot
    for i, category in enumerate(cre_categories):
        if len(cre_categories) == 1:
            ax = axes[1]
            ax_dend = axes[0]
        else:
            ax = axes[i, 1]
            # Plot dendrogram
            ax_dend = axes[i, 0]
        category_cres = cres_to_use[cre_type == category]
        # Cluster CREs within category
        activity_subset = activity.loc[category_cres]  # Use pre-transposed data
        q_value_subset = q_value.loc[category_cres]
        if category != 'No target':
            # set the orders based on q-value
            # create a order token for each CRE
            max_sig_n = (q_value_subset >= -np.log10(0.01)).sum(axis=1).max()
            cre_order_token = []
            for j, cre in enumerate(category_cres):
                # get significant cell types
                significant_cell_types = q_value_subset.loc[cre][q_value_subset.loc[cre] >= -np.log10(0.01)].index
                # get the index of the cell types in cell_types_to_use
                significant_cell_types_idx = cell_types_to_use.get_indexer(significant_cell_types) + 1
                significant_q_value = q_value_subset.loc[cre][significant_cell_types].values
                # fill to max_sig_n
                significant_cell_types_idx = np.pad(significant_cell_types_idx, (0, max_sig_n - len(significant_cell_types_idx)), constant_values=0)
                significant_q_value = np.pad(significant_q_value, (0, max_sig_n - len(significant_q_value)), constant_values=0)
                # append the q-value
                cre_order_token.append(np.concatenate((significant_cell_types_idx, -significant_q_value)))
            # sort the cres by the order token
            cre_order_token = np.array(cre_order_token)
            # sort cres by the order token
            if cre_order_token.shape[1] > 0:
                ordered_cres = category_cres[np.lexsort(cre_order_token.T[::-1,:])]
                cre_order_token = cre_order_token[np.lexsort(cre_order_token.T[::-1,:])]
            else:
                ordered_cres = category_cres
                cre_order_token = np.zeros((len(category_cres), 0))
            # Sort data by clustered order
            activity_subset = activity_subset.loc[ordered_cres].copy()
            q_value_subset = q_value.loc[ordered_cres].copy()
            target_df_subset = target_df.loc[:, ordered_cres].copy()
        else:
            ordered_cres = category_cres
            target_df_subset = target_df.loc[:, ordered_cres].copy()
            cre_order_token = np.zeros((len(category_cres), 0))
            max_sig_n = 0
        subset = pd.DataFrame({size_name: q_value_subset.T.values.flatten(),
                               hue_name: activity_subset.T.values.flatten(),
                               'cell_types': list(cell_types_to_use.values.repeat(len(ordered_cres))),
                               'cres': np.tile(ordered_cres, len(cell_types_to_use)),
                               'positive_control': target_df_subset.values.flatten()})
        subset['cres'] = pd.Categorical(subset['cres'], categories=ordered_cres, ordered=True)
        # dot plots
        sns.scatterplot(data=subset, x='cell_types', y='cres', size=size_name, hue=hue_name, edgecolor='none',
                        hue_norm=(hue_min, hue_max), size_norm=(size_min, size_max),
                        sizes=(3, 250), alpha=0.8, palette='coolwarm', ax=ax)
        # dendrogram
        draw_custom_dendrogram(cre_order_token[:, :max_sig_n], ordered_cres, ax_dend)
        # ax.invert_yaxis()
        ax_dend.axis('off')
        # Add markers for positive controls
        if positive_control_info is not None:
            markers = {
                'on-target': ('red', 's'), 
                'off-target': ('blue', 's'), 
                'miss': ('grey', 's')
            }
            for control, (color, marker) in markers.items():
                data = subset[subset['positive_control'] == control]
                if not data.empty:
                    sns.scatterplot(
                        data=data, x='cell_types', y='cres', 
                        s=250, alpha=0.8, marker=marker,
                        facecolor='none', edgecolor=color, legend=False, ax=ax
                    )
        # ax.margins(y=0.5/figsize[1]/height_ratios[i] * max(np.array(height_ratios)))
        ax.margins(y=0.6 / height_ratios[i])
        ax.set_xlim(-0.5, len(cell_types_to_use)-0.5) 
        # remove the ticks, x label
        if i != len(cre_categories) - 1:
            ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
            ax.set_xlabel('')
        else:
            # rotate the x labels
            ax.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=True)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
            ax.set_xlabel('Cell Types')
        if category == 'Off-target' or category == 'CREs':
            # Capture legend handles/labels from the FIRST subplot before removal
            legend = ax.get_legend()
            legend_handles = legend.legend_handles
            legend_labels = [t.get_text() for t in legend.get_texts()]
        # Remove subplot legend
        ax.get_legend().remove()
        if category == 'CREs':
            # remove y axis label
            ax.set_ylabel('')
            # set y ticks font size to 8
            ax.tick_params(axis='y', which='both', labelsize=8)
        elif category == 'Negative Controls':
            ax.tick_params(axis='y', which='both', labelsize=8)
        else:
            ax.set_ylabel(category)
    # put legends to right of the plot
    fig.legend(legend_handles, legend_labels, loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10, labelspacing=1.5)
    fig.tight_layout()
    plt.close(fig)
    return fig
# %% consistency between two experiments
fold_change_test_config = {
        "cell_types_to_use": cell_types_to_use_nc.tolist(),
        "normalize_by_cell_rna": False, "normalize_by_cell_volume": False, 
        "normalize_by_celltype_rna": False, "normalize_by_celltype_volume": False,
        "normalize_by_negative_control": True, 
        "normalize_by_infected_cell": False, "normalize_by_libsize": False,
        "log_transform": False, "rank_transform": None,
        "filter_zero_counts": False,
        "bootstrap_number": 10000, "n_jobs": 296}
fig = plot_celltype_activity_distribution_compare(
    starrfish1_filtered, starrfish2_filtered, cell_types_to_use=cell_types_to_use_nc, cres_to_use=cres_to_use_libsize_high,
    test_method='fold_change', test_configs=fold_change_test_config, log=True, filter_zero=False,
    show_mean_std=False, show_positive_control=False)
fig.savefig(f'results/fold_change/expr1_expr2_celltype_distribution.pdf')
# consistency between two experiments, stats test
res1 = starrfish1.fold_change_test(**fold_change_test_config)
res2 = starrfish2.fold_change_test(**fold_change_test_config)
res1_q = pd.DataFrame(multitest.multipletests(res1['pvalue_activity'].astype(float).values.flatten(), method='fdr_bh')[1].reshape(res1['pvalue_activity'].shape), 
                      index=res1['pvalue_activity'].index, columns=res1['pvalue_activity'].columns)
res2_q = pd.DataFrame(multitest.multipletests(res2['pvalue_activity'].astype(float).values.flatten(), method='fdr_bh')[1].reshape(res2['pvalue_activity'].shape), 
                      index=res2['pvalue_activity'].index, columns=res2['pvalue_activity'].columns)
# %%fold_change test, CRE-wise
fold_change_test_config = {"cell_types_to_use": None,
                           "normalize_by_cell_rna": False,
                           "normalize_by_cell_volume": False,
                           "normalize_by_celltype_rna": False,
                           "normalize_by_celltype_volume": False,
                           "normalize_by_negative_control": True, # normalize by negative control
                           "normalize_by_infected_cell": False,
                           "normalize_by_libsize": False,
                           "log_transform": False,
                           "filter_zero_counts": False,
                           "bootstrap_number": None}
activity_df = fetch_data(starrfish2_filtered, 'fold_change', fold_change_test_config, 
                         normalize_by_lib_size=True)
# normalize activity_df by library size
cre_corr, celltype_corr = starrfish2_filtered.corr_atac_cpm(
    cell_types_to_use=cell_types_to_use_nc_2, cres_to_use=None, 
    acvitity_df=activity_df, 
    filter_by_atac_z_threshold=None, filter_by_atac_raw_threshold=None,
    filter_by_negative_control_z_threshold=None,
    log_activity=True,
    log_atac=True)
significant_cres = cre_corr[(cre_corr['pearson_p'] <= 0.05) & (cre_corr['pearson'] > 0)].index
significant_celltypes = celltype_corr[(celltype_corr['pearson_p'] <= 0.05) & (celltype_corr['pearson'] > 0)].index
print(len(significant_cres), len(significant_celltypes))
fig = cre_corr_dotplot(starrfish2_filtered, significant_cres, cell_types_to_use_nc_2,
                  test_method='fold_change', test_configs=fold_change_test_config,
                  scale_by_cre=True, z_score_by_cre=False, figsize=(12, 16))
fig.savefig(f'results/fold_change/expr2_cre_dotplot_vertical.pdf')
fig = cre_corr_dotplot(starrfish2_filtered, significant_cres, cell_types_to_use_nc_2,
                       test_method='fold_change', test_configs=fold_change_test_config,
                       scale_by_cre=True, z_score_by_cre=False, figsize=(16, 12))
fig.savefig(f'results/fold_change/expr2_cre_dotplot_horizontal.pdf')
# %% plot cumulative correlation versus CREs, we need to see that but not necessarily in the manuscript
starrfish2_filtered.load_cpm('Data/ATAC_cpm_peakBysubclass.csv', attr_to_add='atac_cpm')
starrfish2_filtered.load_cpm('Data/H3K4me1_cpm_peakBysubclass.csv', attr_to_add='h3k4me1_cpm')
starrfish2_filtered.load_cpm('Data/H3K9me3_cpm_peakBysubclass.csv', attr_to_add='h3k9me3_cpm')
starrfish2_filtered.load_cpm('Data/H3K27ac_cpm_peakBysubclass.csv', attr_to_add='h3k27ac_cpm')
starrfish2_filtered.load_cpm('Data/H3K27me3_cpm_peakBysubclass.csv', attr_to_add='h3k27me3_cpm')
corr_cutoffs = np.linspace(-1, 1, 200)
prob = {'atac_cpm': [], 'h3k4me1_cpm': [], 'h3k9me3_cpm': [], 'h3k27ac_cpm': [], 'h3k27me3_cpm': []}
for mod in ['atac_cpm', 'h3k4me1_cpm', 'h3k9me3_cpm', 'h3k27ac_cpm', 'h3k27me3_cpm']:
    cre_corr, celltype_corr = starrfish2_filtered.corr_atac_cpm(
        cell_types_to_use=cell_types_to_use_nc_2, cres_to_use=None, 
        acvitity_df=activity_df, 
        filter_by_atac_z_threshold=None, filter_by_atac_raw_threshold=None,
        filter_by_negative_control_z_threshold=None,
        log_activity=True, log_atac=True, attr_to_use=mod)
    for corr_cutoff in corr_cutoffs:
        if mod in ['h3k9me3_cpm', 'h3k27me3_cpm']:
            prop = (cre_corr['pearson'] <= -corr_cutoff).sum() / len(cre_corr)
        else:
            prop = (cre_corr['pearson'] >= corr_cutoff).sum() / len(cre_corr)
        prob[mod].append(prop)
# %%
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(corr_cutoffs, prob['atac_cpm'], label='ATAC', color='blue')
ax.plot(corr_cutoffs, prob['h3k4me1_cpm'], label='H3K4me1', color='green')
ax.plot(corr_cutoffs, prob['h3k9me3_cpm'], label='H3K9me3', color='red')
ax.plot(corr_cutoffs, prob['h3k27ac_cpm'], label='H3K27ac', color='orange')
ax.plot(corr_cutoffs, prob['h3k27me3_cpm'], label='H3K27me3', color='purple')
ax.set_xlabel('Pearson correlation with epigenomic markers')
ax.set_ylabel('Proportion correlation ≥ cutoff')
ax.set_xlim(0, 1)
ax.set_ylim(0, 0.6)
fig.tight_layout()
fig.savefig(f'results/fold_change/expr2_cre_cumulative_prob.pdf')
# %%
# get the p-value only in those cell types
fold_change_test_config = {"cell_types_to_use": cell_types_to_use_nc_2.tolist(),
                           "normalize_by_cell_rna": False,
                           "normalize_by_cell_volume": False,
                           "normalize_by_celltype_rna": False,
                           "normalize_by_celltype_volume": False,
                           "normalize_by_negative_control": True,
                           "normalize_by_infected_cell": False,
                           "normalize_by_libsize": False,
                           "log_transform": False,
                           "rank_transform": None,
                           "filter_zero_counts": False,
                           "bootstrap_number": 10000,
                           'fill_nan': True,
                           'n_jobs': 256, 
                           'load_stored': True,}
res2 = starrfish2_filtered.fold_change_test(**fold_change_test_config)
# for each CRE, do q-value correction
res2_q = res2['pvalue_activity'].copy()
# get the max cell for each CRE and cell type
max_cell_activity = starrfish2_filtered.get_cre_expression().groupby(starrfish2_filtered.get_celltypes()).max()
sum_cell_activity = starrfish2_filtered.get_cre_expression().groupby(starrfish2_filtered.get_celltypes()).sum()
max_cell_contribution = (max_cell_activity / sum_cell_activity).loc[cell_types_to_use_nc_2]
# for any max_cell_contribution > 0.5, we set the pvalue to 1
res2_q[max_cell_contribution > 0.5] = 1
# q-value correction
res2_q = pd.DataFrame(multitest.multipletests(res2_q.values.flatten(), method='fdr_bh')[1].reshape(res2_q.shape),
                      index=res2_q.index, columns=res2_q.columns)
neg_q = res2_q[negative_control_cres]
neg_q.loc[(neg_q <= 0.01).any(axis=1), (neg_q <= 0.01).any(axis=0)]
# %%
target_df = pd.DataFrame(index=res2_q.columns, columns=['on-target', 'off-target', 'best_subclass'])
for cre in res2_q.columns:
    # get on-target cell types
    target_celltypes = starrfish2_filtered.get_creinfo().loc[cre, 'best_subclass']
    if isinstance(target_celltypes, str):
        target_celltypes = [target_celltypes]
    target_df.loc[cre, 'on-target'] = res2_q.index[res2_q[cre] <= 0.01].isin(target_celltypes).sum()
    target_df.loc[cre, 'off-target'] = len(res2_q.index[res2_q[cre] <= 0.01]) - target_df.loc[cre, 'on-target']
    target_df.loc[cre, 'best_subclass'] = target_celltypes
print(target_df['on-target'].sum(), (target_df['off-target'] > 0).sum(), ((target_df['off-target']==0) & (target_df['on-target'] > 0)).sum())
# volcano plot to visualize the on-target q-value
for cre in res2['cre_info'].index:
    # get the best subclass
    best_subclass = res2['cre_info'].loc[cre, 'best_subclass']
    # get the pvalue and qvalue for the best subclass
    if best_subclass in res2['pvalue_activity'].index:
        res2['cre_info'].loc[cre, 'pvalue_activity'] = res2['pvalue_activity'].loc[best_subclass, cre]
        res2['cre_info'].loc[cre, 'qvalue_activity'] = res2['qvalue_activity'].loc[best_subclass, cre]
toplot = res2['cre_info'].loc[target_cres, ['foldchange', 'qvalue_activity']].copy()
fig, ax = plt.subplots(figsize=(6, 6))
# clip min foldchange to 1e-5
# clip min qvalue to 1e-5
toplot['foldchange'] = toplot['foldchange'].clip(lower=1e-2)
toplot['qvalue_activity'] = toplot['qvalue_activity'].clip(lower=1/5000)
toplot['significant'] = (toplot['qvalue_activity'] <= 0.01) & (toplot['foldchange'] > 1)
# log transform
toplot['foldchange'] = np.log10(toplot['foldchange'])
toplot['qvalue_activity'] = -np.log10(toplot['qvalue_activity'])
# plot the volcano plot
sns.scatterplot(data=toplot, x='foldchange', y='qvalue_activity', hue='significant', palette=['gray', 'red'], alpha=0.8, legend=False)
# plot the line foldchange = 1, qvalue = 0.05
ax.axhline(y=-np.log10(0.05), linestyle='--', color='gray')
ax.axvline(x=0, linestyle='--', color='gray')
ax.set_xlabel('log10(foldchange)')
ax.set_ylabel('-log10(qvalue)')
plt.close(fig)
fig.savefig(f'results/fold_change/expr2_cre_volcano.pdf')
#%% plot a dot plot of all cres
have_target_cres = target_df.index[(target_df['on-target']!=0) | (target_df['off-target']!=0)].intersection(non_negative_control_cres)
cre_info = starrfish2_filtered.get_creinfo().copy()
cre_info['best_subclass'] = 'CRE'
cre_info.loc[negative_control_cres, 'best_subclass'] = 'Negative Control'
fig = cre_pval_dotplot(res2_q, res2['celltype_activity'], 
                       have_target_cres, cell_types_to_use_nc_2,
                       positive_control_info=None, figsize=(8, 20))
fig.savefig(f'results/fold_change/expr2_qvalue_dotplot_all_cres.pdf', bbox_inches='tight')
# plot negative control cres + have target cres
fig = cre_pval_dotplot(res2_q, res2['celltype_activity'], 
                       negative_control_cres.union(have_target_cres), cell_types_to_use_nc_2,
                       positive_control_info=cre_info, figsize=(8, 20))
fig.savefig(f'results/fold_change/expr2_qvalue_dotplot_negative_control_cres.pdf', bbox_inches='tight')
# for each CRE, select the best subclass based on best atac_cpm
for cre in cre_info.index:
    if cre_info.loc[cre, 'best_subclass'] != 'Negative Control':
        cre_atac = starrfish2_filtered.atac_cpm.loc[cell_types_to_use_nc_2.intersection(starrfish2_filtered.atac_cpm.index), cre]
        best_subclass = cre_atac.idxmax()
        cre_info.loc[cre, 'best_subclass'] = best_subclass
fig = cre_pval_dotplot(res2_q, res2['celltype_activity'], 
                       negative_control_cres.union(have_target_cres), cell_types_to_use_nc_2,
                       positive_control_info=cre_info, figsize=(8, 20))
fig.savefig(f'results/fold_change/expr2_qvalue_dotplot_best_atac.pdf', bbox_inches='tight')
# %% visualization
# visulization of a specific CRE by atac signals
for cre in significant_cres:
    # pick the top 5 best ATAC
    atac_cpm_rank = starrfish2_filtered.atac_cpm.loc[cell_types_to_use_nc_2.intersection(starrfish2_filtered.atac_cpm.index), cre].rank(ascending=False)
    # order by rank
    atac_cpm_rank = atac_cpm_rank.sort_values(ascending=True)
    cell_types_to_visualize = atac_cpm_rank[atac_cpm_rank <= 5].index
    fig = starrfish2_filtered.plot_gene(
        cre, average_by_celltype=False,
        norm_by_negative_control_cell_type_sum=False,
        norm_by_negative_control_cell_type_mean=True,
        norm_by_negative_control_single_cell=False,
        cell_types_to_visualize=cell_types_to_visualize, scale_size_by='counts',
        log=False, transpose=-1, flipx=-1, sz_max=50,
        cell_types_to_use=cell_types_to_use_nc_2)
    fig.savefig(f'results/fold_change/cres/top5_atac/expr2_{cre}.pdf')
    fig.savefig(f'results/fold_change/cres/top5_atac/expr2_{cre}.png', dpi=500)
# visualize by on target cell types
for cre in target_df.index[(target_df['on-target'] != 0) | (target_df['off-target'] != 0)]:
    # rank by q-value
    cre_q_values = res2['qvalue_activity'].loc[cell_types_to_use_nc_2, cre]
    cre_q_values = cre_q_values[cre_q_values <= 0.05] 
    # order by rank
    cre_q_values = cre_q_values.sort_values(ascending=True)
    cell_types_to_visualize = cre_q_values.index
    fig = starrfish2_filtered.plot_gene(
        cre, average_by_celltype=False, # if true, all cells from same cell type will have same value
        norm_by_negative_control_cell_type_sum=False, # normalize raw counts by the sum of negative control in the cell type
        norm_by_negative_control_cell_type_mean=True, # normalize raw counts by the mean of negative control in the cell type
        norm_by_negative_control_single_cell=False, # normalize raw counts by the negative control in each single cell
        cell_types_to_visualize=cell_types_to_visualize, # only visualize some cell types
        scale_size_by='counts', # scale size by "counts": normalized counts; or "celltype_number": number of cells in the cell type
        log=False, transpose=-1, flipx=-1, sz_max=50,
        cell_types_to_use=cell_types_to_use_nc_2)
    fig.savefig(f'results/fold_change/cres/q_value/expr2_{cre}.pdf')
    fig.savefig(f'results/fold_change/cres/q_value/expr2_{cre}.png', dpi=500)
# %% visualization of cell types
fig=starrfish2_filtered.plot_cluster(cell_types_to_use_nc_2, plot_legend=True, transpose=-1, flipx=-1, 
                                     sbig=20, figsize=(24, 12),)
fig.savefig(f'results/fold_change/expr2_celltypes_selected.pdf')
fig.savefig(f'results/fold_change/expr2_celltypes_selected.png', dpi=500)
fig=starrfish2.plot_cluster(starrfish2.get_celltypes().unique(), plot_legend=False, transpose=-1, flipx=-1, 
                            sbig=20, figsize=(24, 12),)
fig.savefig(f'results/fold_change/expr2_celltypes.pdf')
fig.savefig(f'results/fold_change/expr2_celltypes.png', dpi=500)
# %% plot the umap of cell types
fig = starrfish2.plot_umap(starrfish2.get_celltypes().unique(), plot_legend=False, size=5, figsize=(6, 6),)
fig.savefig(f'results/fold_change/expr2_celltypes_umap.pdf')
fig.savefig(f'results/fold_change/expr2_celltypes_umap.png', dpi=500)
fig = starrfish2_filtered.plot_umap(cell_types_to_use_nc_2, plot_legend=True, size=5, figsize=(6, 6),)
fig.savefig(f'results/fold_change/expr2_celltypes_selected_umap.pdf')
fig.savefig(f'results/fold_change/expr2_celltypes_selected_umap.png', dpi=500)
# %% plot the distribution of activity and atac
target_cres = starrfish2_filtered.get_creinfo().index[starrfish2_filtered.get_creinfo()['best_subclass'].isin(cell_types_to_use_nc_2)]
print(target_cres[target_cres.isin(significant_cres)])
print(len(target_cres), sum(significant_cres.isin(target_cres)), len(significant_cres))
fig5 = plot_cre_activity_atac_distribution_compare(
        starrfish2_filtered, cell_types_to_use=cell_types_to_use_nc_2, cres_to_use=target_cres[~target_cres.isin(significant_cres)],
        test_method='fold_change', test_configs=fold_change_test_config, log2=True, filter_zero=False)
fig5.savefig(f'results/fold_change/expr2_cre_distribution_bad_CRE.pdf')
fig5 = plot_cre_activity_atac_distribution_compare(
        starrfish2_filtered, cell_types_to_use=cell_types_to_use_nc_2, cres_to_use=significant_cres,
        test_method='fold_change', test_configs=fold_change_test_config, log2=True, filter_zero=False)
fig5.savefig(f'results/fold_change/expr2_cre_distribution_good_CRE.pdf')
# %% split the CREs by on-target and off-target rates
# select the best cell type for each CRE, check if it is on-target or off-target
target_cre_df = activity_df.loc[cell_types_to_use_nc_2, target_cres]
precision = []
recall = []
# for each CRE, select top rank cell type
for z in np.arange(0, 3, 0.1):
    on_target = 0
    off_target = 0
    for cre in target_cre_df.columns:
        z_score = target_cre_df[cre] - target_cre_df[cre].mean()
        z_score /= target_cre_df[cre].std()
        top_rank_celltype = z_score[z_score > z].index
        target_celltype = starrfish2_filtered.get_creinfo().loc[cre, 'best_subclass']
        # on-target and off-target rates
        on_target += target_celltype in top_rank_celltype
        off_target += len(top_rank_celltype)
    if off_target == 0:
        precision.append(0)
        recall.append(0)
    else:
        precision.append(on_target / off_target)
        recall.append(on_target / len(target_cre_df.columns))
# Create figure and first axis
fig, ax1 = plt.subplots(figsize=(10, 6))

# Plot Precision on left y-axis
x_values = np.arange(0, 3, 0.1)
sns.lineplot(x=x_values, y=precision, ax=ax1, color='blue', label='Precision')
ax1.set_xlabel('Z-score', fontsize=12)
ax1.set_ylabel('Precision', color='blue', fontsize=12)
ax1.tick_params(axis='y', labelcolor='blue')

# Create twin axis for Recall on the right
ax2 = ax1.twinx()
sns.lineplot(x=x_values, y=recall, ax=ax2, color='red', label='Recall')
ax2.set_ylabel('Recall', color='red', fontsize=12)
ax2.tick_params(axis='y', labelcolor='red')

# Add legend (optional)
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')
# remove ax2 legend
ax2.legend_.remove()

plt.title('Precision and Recall vs. Z-score', fontsize=14)
plt.show()
# %%
# simple regression of motif scores to activity, didn't work
motif_scores = pd.read_csv('results/CRE_motif.csv')
motif_scores['enh'] = motif_scores['Chromosome'] + ':' + motif_scores['Start'].astype(str) + '-' + motif_scores['End'].astype(str)
motif_scores.index = starrfish2_filtered.get_creinfo().index[starrfish2_filtered.get_creinfo()['labeling_type'] != 'negative control']
motif_scores['lib_size'] = starrfish2_filtered.lib_size.loc[motif_scores.index]
# %% genomespy to visualize
plot = starrfish2_filtered.plot_atac_genomespy(cell_types_to_use_nc_2, cre='CRE004')
plot.show(filename='genomespy.html')
plot.close()
# %%
starrfish2_filtered.plot_pygenometracks(cell_types_to_use_nc_2, 'CRE004', 'CRE004.pdf', 
                                        nbins=500, padding=20000, min=None, max=2)
# %%
ethan_anno = pd.read_csv('Data/annotation/my_cre_annot_final.tsv', sep='\t', index_col=0)
ethan_anno
# %%
# use homer to find motifs
human_mouse_map = pd.read_csv('Data/human_mouse_ortholog.tsv', sep='\t')
non_negative_control_cres = starrfish2_filtered.get_creinfo().index[starrfish2_filtered.get_creinfo()['labeling_type'] != 'negative control']
genes_of_interest = []
for cell_type in cell_types_to_use_nc_2:
    cres = res2['qvalue_activity'].loc[cell_type].index[res2['qvalue_activity'].loc[cell_type] <= 0.05]
    cres = cres[cres.isin(non_negative_control_cres)]
    if len(cres) < 5:
        continue
    bg_cres = non_negative_control_cres[~non_negative_control_cres.isin(cres)]
    homer_genes = starrfish2_filtered.motif_enrichment_homer(
        cres_to_use=cres, background_cres=bg_cres, 
        outputdir=f'results/homer_motif/{cell_type.replace(" ", "_")}/',
        overwrite=False,)
    if homer_genes is None:
        continue
    # filter to q-value < 0.05
    homer_genes = homer_genes[homer_genes['q-value (Benjamini)'] < 0.05]['gene'].str.split(';|:|,|-').explode()
    # get mouse genes
    homer_genes_mouse = []
    for gene in homer_genes:
        if gene in human_mouse_map['Gene name'].values:
            mouse_genes = human_mouse_map.loc[human_mouse_map['Gene name'] == gene, 'Mouse gene stable ID'].values
            homer_genes_mouse.extend(mouse_genes)
        elif gene in human_mouse_map['Mouse gene name'].values:
            mouse_genes = human_mouse_map.loc[human_mouse_map['Mouse gene name'] == gene, 'Mouse gene stable ID'].values
            homer_genes_mouse.extend(mouse_genes)
    homer_genes_mouse = pd.Series(homer_genes_mouse).dropna().unique()
    np.save(f'results/homer_motif/{cell_type.replace(" ", "_")}.homer_genes_mouse.npy', homer_genes_mouse)
    genes_of_interest.extend(homer_genes_mouse)
genes_of_interest = pd.Series(genes_of_interest).dropna().unique()
np.save('results/homer_motif/genes_of_interest.npy', genes_of_interest)

# %%
# preprocess the chromatine state
chromstate = pd.read_csv('Data/allCRE.amb.PairedTag.annot.tsv', sep='\t')
chromstate_a = chromstate[chromstate['chromHMMState'] == 'Chr-A']
chromstate_o = chromstate[chromstate['chromHMMState'] == 'Chr-O']
# %%
import pyranges as pr
# Prepare CRE info
creinfo = starrfish2_filtered.get_creinfo()
creinfo = creinfo[creinfo['labeling_type'] != 'negative control']
creinfo = creinfo[['Chrom', 'Start', 'End']].copy()
creinfo.columns = ['Chromosome', 'Start', 'End']
creinfo['Start'] = creinfo['Start'].astype(int)
creinfo['End'] = creinfo['End'].astype(int)
creinfo['CRE_ID'] = creinfo.index  # keep track of original rows
cre_gr = pr.PyRanges(creinfo)

# Output containers
celltypes = chromstate['subclass'].unique()
cre_chrom_a = pd.DataFrame(index=creinfo['CRE_ID'], columns=celltypes)
cre_chrom_o = pd.DataFrame(index=creinfo['CRE_ID'], columns=celltypes)

# Loop over cell types
for celltype in celltypes:
    for chrom_label, chrom_df, target_df in zip(
        ['a', 'o'], [chromstate_a, chromstate_o], [cre_chrom_a, cre_chrom_o]
    ):
        subset = chrom_df[chrom_df['subclass'] == celltype].copy()
        subset = subset[['chrom', 'startFrom', 'endTo']].copy()
        subset.columns = ['Chromosome', 'Start', 'End']
        subset['Start'] = subset['Start'].astype(int)
        subset['End'] = subset['End'].astype(int)
        subset_gr = pr.PyRanges(subset)

        # Perform intersection
        intersection = cre_gr.join(subset_gr)

        # Compute overlap length
        overlap_df = intersection.df
        overlap_df['overlap'] = (
            overlap_df[['Start_b', 'Start']].max(axis=1) -
            overlap_df[['End_b', 'End']].min(axis=1)
        ) * -1  # negative because min(start) - max(end) < 0

        overlap_df = overlap_df[overlap_df['overlap'] > 0]

        # Normalize by CRE length
        overlap_df['cre_len'] = overlap_df['End'] - overlap_df['Start']
        overlap_df['overlap_ratio'] = overlap_df['overlap'] / overlap_df['cre_len']

        # Aggregate by CRE_ID: keep max overlap per CRE
        best_overlap = overlap_df.groupby('CRE_ID')['overlap_ratio'].max()

        # Fill results
        target_df[celltype] = best_overlap.reindex(creinfo['CRE_ID']).fillna(0).values
# %%
