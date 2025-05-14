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
# %% preprocess and load data
adata1 = sc.read_h5ad(f'{PWD}/Data/scdata_12_11NoT7_BRBB500gn_withCRE_final.h5ad')
adata2 = sc.read_h5ad(f'{PWD}/Data/scdata_03_14_BRBB500gn_withCRE_final.h5ad')
# operate fov, it is the index names
adata1.obs['fov'] = adata1.obs.index.str.split('--').str[0]
adata2.obs['fov'] = adata2.obs.index.str.split('--').str[0]
# change adata2 obs subclass_name to subclass
adata1.obs['subclass'] = adata1.obs['subclass_name'].str.replace('^[0-9]+ ', '', regex=True)
adata2.obs['subclass'] = adata2.obs['subclass_name'].str.replace('^[0-9]+ ', '', regex=True)
adata1.uns['CRE_info']['best_subclass'] = adata1.uns['CRE_info']['best_subclass'].str.replace('_', ' ')
adata2.uns['CRE_info']['best_subclass'] = adata2.uns['CRE_info']['best_subclass'].str.replace('_', ' ')
adata1.uns['CRE_info'].index = ['CRE' + str(i+1).zfill(3) for i in range(len(adata1.uns['CRE_info']))]
adata2.uns['CRE_info'].index = ['CRE' + str(i+1).zfill(3) for i in range(len(adata2.uns['CRE_info']))]
adata1.obsm['CRE'] = adata1.obsm['CRE'][adata1.uns['CRE_info'].index]
adata2.obsm['CRE'] = adata2.obsm['CRE'][adata2.uns['CRE_info'].index]
# %% load data and form STARRFISH object
if os.path.exists('results/starrfish1.pkl'):
    starrfish1 = STARRFISH.load('results/starrfish1.pkl')
else:
    starrfish1 = STARRFISH(adata1)
if os.path.exists('results/starrfish2.pkl'):
    starrfish2 = STARRFISH.load('results/starrfish2.pkl')
else:
    starrfish2 = STARRFISH(adata2)
if os.path.exists('results/starrfish1_filtered.pkl'):
    starrfish1_filtered = STARRFISH.load('results/starrfish1_filtered.pkl')
else:
    starrfish1_filtered = STARRFISH(adata1[(adata1.obsm['CRE'] > 0).sum(axis=1) > 0])
if os.path.exists('results/starrfish2_filtered.pkl'):
    starrfish2_filtered = STARRFISH.load('results/starrfish2_filtered.pkl')
else:
    starrfish2_filtered = STARRFISH(adata2[(adata2.obsm['CRE'] > 0).sum(axis=1) > 0])
# %% reload starrfish object, if update utils.py
starrfish1_filtered = reload(starrfish1_filtered)
starrfish2_filtered = reload(starrfish2_filtered)
# %% drop existing test results, if any, specified by to_drop
to_drop = '' # drop nothing
starrfish1_filtered = drop_test(starrfish1_filtered, to_drop)
starrfish2_filtered = drop_test(starrfish2_filtered, to_drop)
# %% fold change test
fold_change_test_config = {"cell_types_to_use": None,
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
                           "bootstrap_number": None}
res1 = starrfish1_filtered.fold_change_test(**fold_change_test_config)
res2 = starrfish2_filtered.fold_change_test(**fold_change_test_config)
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
# cell types to use are cell types that have at least 50 cells
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
                                            log=True, filter_zero=True, hist=True, contour=True,
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
        ncol=ncol, nrow=nrow, log=log,
        filter_zero=filter_zero, hist=hist, contour=contour)
    plt.close(fig)
    return fig

def plot_cre_activity_atac_distribution_compare(obj: STARRFISH, cell_types_to_use, cres_to_use, test_method, test_configs, 
                                                     contour=True, hist=True, log=False, filter_zero=True, ncol=8):
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
    if log:
        test_results1 = np.log10(test_results1.astype(float) + 1)
        test_results2 = np.log10(test_results2.astype(float) + 1)
    # plot distribution of cell type activity
    nrow = int(np.ceil(len(cres_to_use) / ncol))
    # change ncol to len(cell_types_to_use) if nrow == 1
    if nrow == 1:
        ncol = len(cres_to_use)
    fig = scatter_plot_with_margin_density_by_cre(
        test_results1, test_results2, cres_to_use, 
        ncol=ncol, nrow=nrow, x_lab='CRE activity', y_lab='ATAC cpm', log=log,
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

def cre_dotplot(obj, cres_to_use, cell_types_to_use, test_method, test_configs, 
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
    # plot dot plot
    fig, ax = plt.subplots(figsize=figsize)
    sns.scatterplot(data=toplot, x='cell_types', y='cres', size=size_name, hue=hue_name, 
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

def celltype_dotplot(obj, cres_to_use, cell_types_to_use, test_method, test_configs, 
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
# %%
# define CREs to use
lib_size = starrfish2_filtered.lib_size['counts']
# fold to average lib_size
lib_size_fold = lib_size / lib_size.mean()
# remove CREs with less than 5 fold enrichment
cres_to_use_libsize_high = lib_size_fold[lib_size_fold > 1/5].index
# remove CRE217
cres_to_use_libsize_high = cres_to_use_libsize_high[cres_to_use_libsize_high != 'CRE217']
len(cres_to_use_libsize_high), lib_size.loc[cres_to_use_libsize_high].min()
# %%
# define cell types to use for filtered data
cell_types_counts1 = starrfish1_filtered.get_celltypes().value_counts()
cell_types_counts2 = starrfish2_filtered.get_celltypes().value_counts()
cell_types_to_use_1 = cell_types_counts1[cell_types_counts1 > 50].index
cell_types_to_use_2 = cell_types_counts2[cell_types_counts2 > 50].index
cell_types_to_use = cell_types_to_use_1.intersection(cell_types_to_use_2)
# check the negative control counts for those cell types
negative_control_counts1 = starrfish1_filtered.get_cre_expression()[starrfish1_filtered.get_negative_control_cres()].sum(axis=1).groupby(starrfish1_filtered.get_celltypes()).sum()
negative_control_counts2 = starrfish2_filtered.get_cre_expression()[starrfish2_filtered.get_negative_control_cres()].sum(axis=1).groupby(starrfish2_filtered.get_celltypes()).sum()
# define the cell types by the negative control counts > 50
cell_types_to_use_nc_1 = negative_control_counts1[negative_control_counts1 > 40].index
cell_types_to_use_nc_2 = negative_control_counts2[negative_control_counts2 > 40].index
cell_types_to_use_nc = cell_types_to_use_nc_1.intersection(cell_types_to_use_nc_2)
len(cell_types_to_use), len(cell_types_to_use_nc), len(cell_types_to_use_nc_2)
# %% consistency between two experiments
fig = plot_celltype_activity_distribution_compare(
    starrfish1_filtered, starrfish2_filtered, cell_types_to_use=cell_types_to_use_nc, cres_to_use=cres_to_use_libsize_high,
    test_method='fold_change', test_configs=fold_change_test_config, log=True, filter_zero=False,
    show_mean_std=False, show_positive_control=False)
fig.savefig(f'results/fold_change/expr1_expr2_celltype_distribution.pdf')
# %% fold_change test, CRE-wise, just add up counts
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
acvitity_df = fetch_data(starrfish2, 'fold_change', fold_change_test_config, 
                         normalize_by_lib_size=True)
# normalize activity_df by library size
cre_corr, celltype_corr = starrfish2.corr_atac_cpm(
    cell_types_to_use=cell_types_to_use_nc_2, cres_to_use=None, 
    acvitity_df=acvitity_df, 
    filter_by_atac_z_threshold=None, filter_by_atac_raw_threshold=None,
    filter_by_negative_control_z_threshold=None,
    log_activity=True,
    log_atac=True)
significant_cres = cre_corr[(cre_corr['pearson_p'] <= 0.05) & (cre_corr['pearson'] > 0)].index
significant_celltypes = celltype_corr[(celltype_corr['pearson_p'] <= 0.05) & (celltype_corr['pearson'] > 0)].index
print(len(significant_cres), len(significant_celltypes))
fig = cre_dotplot(starrfish2, significant_cres, cell_types_to_use_nc_2,
                  test_method='fold_change', test_configs=fold_change_test_config,
                  scale_by_cre=True, z_score_by_cre=False, figsize=(20, 12))
fig.savefig(f'results/fold_change/expr2_cre_dotplot.pdf')
# %% visualization
# visulization of a specific CRE
for cre, cell_types_to_visualize  in zip(['CRE004', 'CRE173', 'CRE377', 'CRE177'],
                                         [None, None, ['PB Evx2 Glut'], ['STR D1 Gaba']]):
    fig = starrfish2_filtered.plot_gene(
        cre, average_by_celltype=False,
        norm_by_negative_control_cell_type_sum=True,
        norm_by_negative_control_cell_type_mean=False,
        norm_by_negative_control_single_cell=False,
        cell_types_to_visualize=cell_types_to_visualize,
        log=False, transpose=-1, flipx=-1,
        cell_types_to_use=cell_types_to_use_nc_2)
    fig.savefig(f'results/fold_change/expr2_{cre}.pdf')
# visualization of a specific CRE, zoom in
for cre, figsize, cell_types_to_visualize  in zip(['CRE004', 'CRE173', 'CRE377', 'CRE177'], 
                                                 [(9, 6), (9, 6), (9, 4), (9, 6)],
                                                 [None, None, ['PB Evx2 Glut'], ['STR D1 Gaba']]):
    print(cre, cell_types_to_visualize, figsize)
    fig = starrfish2_filtered.plot_gene(
        cre, average_by_celltype=False,
        norm_by_negative_control_cell_type_sum=True,
        norm_by_negative_control_cell_type_mean=False,
        norm_by_negative_control_single_cell=False,
        cell_types_to_visualize=cell_types_to_visualize,
        log=False, transpose=-1, flipx=-1, show_celltypes=False,
        select_region_by_best_celltype=True, figsize=figsize, 
        cell_types_to_use=cell_types_to_use_nc_2)
    fig.savefig(f'results/fold_change/expr2_{cre}_zoomin.pdf')
# visualization of cell types
fig=starrfish2_filtered.plot_cluster(['TH Prkcd Grin2c Glut', # CRE004
                                  'NDB-SI-MA-STRv Lhx8 Gaba', # CRE173
                                  'PB Evx2 Glut'], # CRE377,
                                 plot_legend=True, transpose=-1, flipx=-1,
                                 sbig=20, figsize=(24, 12),
                                 )
fig.savefig(f'results/fold_change/expr2_celltypes.pdf')
# visualization of a specific cell type, zoom in
for cell_types_to_visualize, figsize, i  in zip(['TH Prkcd Grin2c Glut', 'NDB-SI-MA-STRv Lhx8 Gaba', 'PB Evx2 Glut', 'STR D1 Gaba'],
                                                [(6, 6), (6, 6), (6, 4), (9, 6)], range(4)):
    print(cell_types_to_visualize, figsize)
    Xcells = starrfish2_filtered.adata.obsm['X_spatial'][:, ::-1] * [-1, 1]
    celltype_idx = starrfish2_filtered.adata.obs['subclass'].isin([cell_types_to_visualize])
    x_min = Xcells[celltype_idx, 0].min()
    x_max = Xcells[celltype_idx, 0].max()
    y_min = Xcells[celltype_idx, 1].min()
    y_max = Xcells[celltype_idx, 1].max()
    fig = starrfish2_filtered.plot_cluster(
        [cell_types_to_visualize], plot_legend=False, transpose=-1, flipx=-1,
        x_region=[x_min, x_max], y_region=[y_min, y_max],
        cmap=starrfish2_filtered.adata.uns['cmap'][i:],
        sbig=20, figsize=figsize,)
    fig.savefig(f'results/fold_change/expr2_{cell_types_to_visualize}_zoomin.pdf')
# %% plot the distribution of activity and atac
target_cres = starrfish2.get_creinfo().index[starrfish2.get_creinfo()['best_subclass'].isin(cell_types_to_use_nc_2)]
print(target_cres[target_cres.isin(significant_cres)])
print(len(target_cres), sum(significant_cres.isin(target_cres)), len(significant_cres))
target_cres = target_cres[~target_cres.isin(significant_cres)]
fig5 = plot_cre_activity_atac_distribution_compare(
        starrfish2, cell_types_to_use=cell_types_to_use_nc_2, cres_to_use=target_cres,
        test_method='fold_change', test_configs=fold_change_test_config, log=True, filter_zero=False)
fig5.savefig(f'results/fold_change/expr2_cre_distribution_bad_CRE.pdf')
fig5 = plot_cre_activity_atac_distribution_compare(
        starrfish2, cell_types_to_use=cell_types_to_use_nc_2, cres_to_use=significant_cres,
        test_method='fold_change', test_configs=fold_change_test_config, log=True, filter_zero=False)
fig5.savefig(f'results/fold_change/expr2_cre_distribution_good_CRE.pdf')