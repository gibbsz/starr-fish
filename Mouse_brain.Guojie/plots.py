
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
PWD = os.path.dirname(os.path.abspath(__file__))
sys.path.append(f'{PWD}/')
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
        lib_size = obj.lib_size.values.reshape(1, -1)
        to_norm = 1 / lib_size * lib_size.mean()
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
    cre_corr['lib_size'] = obj1.lib_size['counts'].loc[cre_corr.index]
    cre_corr['lib_size'] = obj1.lib_size['counts'].loc[cre_corr.index]
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
                                    log_activity=True, log_atac=True):
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
    fig1, ax1 = plt.subplots(ncols=2, nrows=1, figsize=(12, 5))
    subplot_cre_corr(ax1[0], celltype_corr, 'corr', title='Correlation between experiments')
    subplot_cre_corr(ax1[1], celltype_corr, 'pval', title='Correlation P-value between experiments')
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
    return fig, fig1, celltype_corr1, celltype_corr2, celltype_corr

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
    fig = plt.figure(figsize=(ncol * 5, nrow * 5.05))
    gs = GridSpec(
        3*nrow, 3*ncol,  # 3*rows, 3*columns for layout control
        figure=fig,
        width_ratios=[1, 0.2, 0.25] * ncol,  # Adjust column widths, 0.05 is spacing
        height_ratios=[0.2, 1, 0.25] * nrow,  # Adjust row heights, 0.05 is spacing
        hspace=0.05,
        wspace=0.1
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
        if pearson[1] <= 0.05:
            color = 'red'
        else:
            color = 'black'
        ax.text(0.5, 0.9, 
                f'Pearson: {pearson[0]:.2f} ({pearson[1]:.2e})\nSpearman: {spearman[0]:.2f} ({spearman[1]:.2e})',
                color=color, fontsize=8, ha='center', va='center', transform=ax.transAxes)
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
                                                test_method, test_configs, normalize_by_lib_size=True,
                                                contour=True, hist=True, log=False, filter_zero=True, 
                                                show_mean_std=True, show_positive_control=True, ncol=8):
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

def plot_cre_activity_atac_distribution_compare(obj: STARRFISH, cell_types_to_use, cres_to_use, mod, test_method, test_configs, 
                                                     contour=True, hist=True, log1=False, log2=False, filter_zero=True, ncol=8):
    test_results1 = fetch_data(obj, test_method, test_configs)
    test_results2 = getattr(obj, mod)
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
        ncol=ncol, nrow=nrow, x_lab='CRE activity', y_lab=mod, log1=log1, log2=log2,
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
        obj1, cell_types_to_use=cell_types_to_use, cres_to_use=cres_to_use, mod='atac_cpm',
        test_method=test_method, test_configs=test_configs, log=log_activity, filter_zero=filter_zero)
    fig4.savefig(f'results/{test_method}/expr1_cre_distribution.pdf')
    fig5 = plot_cre_activity_atac_distribution_compare(
        obj2, cell_types_to_use=cell_types_to_use, cres_to_use=cres_to_use, mod='atac_cpm',
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
    lib_size = obj.lib_size['counts'].copy()
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
        cell_type_ng_counts = negative_control_counts2.loc[cell_type, neg_controls_to_check]
        cell_type_counts = counts2.loc[cell_type, other_to_check]
        # Fit model WITH INTERCEPT
        X = sm.add_constant(ng_libsize)  # Add intercept term
        model = sm.OLS(cell_type_ng_counts, X).fit()
        # Generate predictions (include intercept)
        x_vals = np.linspace(ng_libsize.min(), ng_libsize.max(), 100)
        X_pred = sm.add_constant(x_vals)
        predictions = model.get_prediction(X_pred)
        predicted_means = predictions.predicted_mean
        conf_int = predictions.conf_int(alpha=0.05)
        # add a linear regression line
        # sns.scatterplot(x=np.log1p(ng_libsize), y=cell_type_ng_counts, ax=ax_, color='blue')
        sns.scatterplot(x=ng_libsize, y=cell_type_ng_counts, ax=ax_, color='blue')
        # plot all other CREs
        # sns.scatterplot(x=lib_size.loc[other_to_check], y=cell_type_counts, ax=ax_, color='gray', alpha=0.5)
        # plot positive controls
        if pos_controls_to_check is not None:
            sns.scatterplot(x=lib_size.loc[pos_controls_to_check], y=cell_type_counts.loc[pos_controls_to_check], ax=ax_, color='red', alpha=0.5)
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

def cre_corr_dotplot(obj, cres_to_use, cell_types_to_use, mods, test_method, test_configs, qval_df=None,
                     log=True, scale_by_cre = True, z_score_by_cre = True, sz_max=300, figsize=(20, 12)):
    test_result = fetch_data(obj, test_method, test_configs)
    # pull cpm info for each modality
    cpm_df_list = []
    for mod in mods:
        cpm = getattr(obj, mod).copy()
        cpm_df_list.append(cpm)
    test_result = test_result[cres_to_use]
    cpm_df_list = [cpm[cres_to_use] for cpm in cpm_df_list]
    if log:
        test_result = np.log1p(test_result)
        cpm_df_list = [np.log1p(cpm) for cpm in cpm_df_list]
    if cell_types_to_use is None:
        cell_types_to_use = test_result.index
    cell_types_to_use = cell_types_to_use[cell_types_to_use.isin(test_result.index)]
    for cpm in cpm_df_list:
        cell_types_to_use = cell_types_to_use[cell_types_to_use.isin(cpm.index)]
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
    # reorder the cres by the best recall cell type of the first modality
    cres_best_recall_celltype = test_result.loc[cell_types_to_use].idxmax(axis=0)
    # cres_best_recall_celltype = obj.get_creinfo()['best_subclass'].loc[significant_cres]
    cres_best_recall_celltype_idx = cell_types_to_use.get_indexer(cres_best_recall_celltype)
    cres_to_use = cres_to_use[np.argsort(cres_best_recall_celltype_idx)]
    # reorder test result and atac cpm
    test_result = test_result.loc[cell_types_to_use, cres_to_use]
    cpm_df_list = [cpm.loc[cell_types_to_use, cres_to_use] for cpm in cpm_df_list]
    # if scale_by_cre, we scale the test result by the max of each cre
    hue_names = [mod.replace('_', ' ') for mod in mods]
    size_name = 'activity'
    if log:
        hue_names = [f'{name} (log)' for name in hue_names]
        size_name += ' (log)'
    if z_score_by_cre:
        # scale test result to z-score along the cres
        test_result = (test_result - test_result.mean(axis=0)) / test_result.std(axis=0)
        # scale atac_cpm to z-score along the cres
        cpm_df_list = [(cpm - cpm.mean(axis=0)) / cpm.std(axis=0) for cpm in cpm_df_list]
        hue_names = [f'{name} (z-score)' for name in hue_names]
        size_name += ' (z-score)'
    if scale_by_cre:
        test_result = (test_result - test_result.min(axis=0)) / (test_result.max(axis=0) - test_result.min(axis=0))
        cpm_df_list = [(cpm - cpm.min(axis=0)) / (cpm.max(axis=0) - cpm.min(axis=0)) for cpm in cpm_df_list]
        hue_names = [f'{name} (scaled)' for name in hue_names]
        size_name += ' (scaled)'
    # find positive controls
    positive_control_df = pd.DataFrame(index=test_result.index, columns=test_result.columns)
    print(positive_control_df.shape)
    for cre in test_result.columns:
        # get the best atac cpm cell type for this cre
        best_cell_types = qval_df[cre].index[qval_df[cre] <= 0.05] if qval_df is not None else None
        if best_cell_types is not None and len(best_cell_types) > 0:
            best_cell_types = best_cell_types[best_cell_types.isin(positive_control_df.index)]
            positive_control_df.loc[best_cell_types, cre] = True
    positive_control_df = positive_control_df.fillna(False)
    # make to plot dataframe, flatten recall and atac_cpm
    toplot = pd.DataFrame({'activity': test_result.values.flatten(),
                           'cell_types': cell_types_to_use.values.repeat(len(cres_to_use)),
                           'cres': np.tile(cres_to_use, len(cell_types_to_use)),
                           'positive_control': positive_control_df.values.flatten()})
    toplot['cell_types'] = toplot['cell_types'].astype(str)
    # rename columns
    toplot.rename(columns={'activity': size_name}, inplace=True)
    for mod, cpm, name in zip(mods, cpm_df_list, hue_names):
        toplot[mod] = cpm.values.flatten()
        toplot.rename(columns={mod: name}, inplace=True)
    # plot dot plot, no edge color
    fig, ax = plt.subplots(figsize=figsize)
    # Fixed color map for modalities
    mod_color_dict = {
        'atac_cpm': '#4682B4',       # steelblue
        'h3k4me1_cpm': '#66A753',    # dark olive green
        'h3k9me3_cpm': '#D95C44',    # coral red
        'h3k27ac_cpm': '#F57C00',    # amber orange
        'h3k27me3_cpm': '#7B68EE'    # medium slate blue
    }
    # Only use selected modality colors
    color_rgb_list = [to_rgb(mod_color_dict[mod]) for mod in mods]

    # Map cell types and CREs to positions
    xticks = toplot['cres'].unique()
    yticks = toplot['cell_types'].unique()
    celltype_to_x = {name: i for i, name in enumerate(xticks)}
    cre_to_y = {name: i for i, name in enumerate(yticks)}
    # Scale dot sizes
    min_size, max_size = 5, sz_max
    # Function to blend with white
    def blend_with_white(base_rgb, strength):
        return tuple((1 - strength) * 1.0 + strength * c for c in base_rgb)
    # Draw pie markers
    for i, row in toplot.iterrows():
        x = celltype_to_x[row['cres']]
        y = cre_to_y[row['cell_types']]
        r = (row[size_name] * (max_size - min_size) + min_size) / 200.0  # radius for the pie chart
        for j in range(len(mods)):
            theta1 = j * 360 / len(mods)
            theta2 = (j + 1) * 360 / len(mods)
            norm_val = row[hue_names[j]]
            blended_color = blend_with_white(color_rgb_list[j], norm_val)
            wedge = mpatches.Wedge(
                center=(x, y), r=r, theta1=theta1, theta2=theta2,
                facecolor=blended_color, edgecolor='grey', linewidth=0.5)
            ax.add_patch(wedge)
        # Outline the whole circle
        circle = mpatches.Circle((x, y), r, fill=False, edgecolor='black', linewidth=0.5)
        ax.add_patch(circle)
    # overlay red boxes for positive controls
    for i, row in toplot[toplot['positive_control']].iterrows():
        x = celltype_to_x[row['cres']]
        y = cre_to_y[row['cell_types']]
        r = (row[size_name] * (max_size - min_size) + min_size) / 200.0 # same radius used for the pie
        # draw a square of side length 2r, centered at (x,y)
        rect = mpatches.Rectangle(
            (x - r, y - r), 2 * r, 2 * r, fill=False, edgecolor='red',
            linewidth=1.2, transform=ax.transData
        )
        ax.add_patch(rect)
    # ── Axis formatting ─────────────────────────────────────
    ax.set_xticks(range(len(xticks)))
    ax.set_xticklabels(xticks, rotation=45, ha='right')
    ax.set_yticks(range(len(yticks)))
    ax.set_yticklabels(yticks)
    ax.set_xlabel('CREs')
    ax.set_ylabel('Cell Types')
    # set x and y axis margin, with respect to the axes limits
    ax.margins(x=0.01, y=0.01)
    # ── Size legend ─────────────────────────────────────────
    from matplotlib.lines import Line2D
    size_proxy = Line2D([0], [0], marker='o', color='w',
                        markerfacecolor='gray', markersize=sz_max/20)
    swatches = [
        Line2D([0],[0], marker='o', color='w', markersize=8,
            markerfacecolor=mod_color_dict[m])
        for m in mods
    ]
    labels = [m.replace('_',' ') for m in mods]
    swatch_leg = ax.legend(
        swatches, labels,
        loc='upper left',
        bbox_to_anchor=(1.05, 0.90),     # just outside the axes, near top
        title='Modality (color)',
        fontsize=8,
        borderaxespad=0.0
    )
    # add it as an artist so it *sticks*
    ax.add_artist(swatch_leg)
    # — 2) now draw the *size* legend —
    size_leg = ax.legend(
        [size_proxy], [size_name],
        loc='upper left',
        bbox_to_anchor=(1.05, 0.75),     # stack it just below the swatches
        title=size_name,
        fontsize=8,
        borderaxespad=0.0
    )
    plt.tight_layout()
    plt.close(fig)
    return fig

def cre_corr_heatmap(obj, cres_to_use_mods, cell_types_to_use, mods, test_method, test_configs, qval_df,
                     log=True, scale_by_cre = True, z_score_by_cre = True, figsize=(0.2, 10), flip=False):
    test_result_orig = fetch_data(obj, test_method, test_configs)
    cres_to_use = pd.Series(list(set.union(*[set(cres) for cres in cres_to_use_mods])))
    # pull cpm info for each modality
    cpm_df_list = []
    if mods is None:
        mods = ['Activity']
        cpm = test_result_orig.copy()
        cpm_df_list.append(cpm)
    else:
        for mod in mods:
            cpm = getattr(obj, mod).copy()
            cpm_df_list.append(cpm)
    test_result_orig = test_result_orig[cres_to_use]
    cpm_df_list = [cpm[cres_to_use] for cpm in cpm_df_list]
    if log:
        test_result_orig = np.log1p(test_result_orig)
        cpm_df_list = [np.log1p(cpm) for cpm in cpm_df_list]
    if cell_types_to_use is None:
        cell_types_to_use = test_result_orig.index
    cell_types_to_use = cell_types_to_use[cell_types_to_use.isin(test_result_orig.index)]
    for cpm in cpm_df_list:
        cell_types_to_use = cell_types_to_use[cell_types_to_use.isin(cpm.index)]
    # first order the cres_to_use by the variance of the best modality to second best
    best_second_ratio = []
    for cre in cres_to_use:
        best_second_ratio.append(
            test_result_orig.loc[cell_types_to_use, cre].max() / test_result_orig.loc[cell_types_to_use, cre].nlargest(2).iloc[-1]
        )
    # sort cres_to_use by the best_second_ratio
    cres_to_use = cres_to_use[np.argsort(best_second_ratio)][::-1]
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
    # reorder the cres by the best recall cell type of the first modality
    cres_best_recall_celltype = test_result_orig.loc[cell_types_to_use, cres_to_use].idxmax(axis=0)
    # cres_best_recall_celltype = obj.get_creinfo()['best_subclass'].loc[significant_cres]
    cres_best_recall_celltype_idx = cell_types_to_use.get_indexer(cres_best_recall_celltype)
    cres_to_use = cres_to_use[np.argsort(cres_best_recall_celltype_idx)]
    # reorder test result and atac cpm
    test_result_orig = test_result_orig.loc[cell_types_to_use, cres_to_use]
    cpm_df_list = [cpm.loc[cell_types_to_use, cres_to_use] for cpm in cpm_df_list]
    # if scale_by_cre, we scale the test result by the max of each cre
    # make figure
    if flip:
        fig = plt.figure(figsize=(figsize[0]*np.sum([len(cres) for cres in cres_to_use_mods]), figsize[1]*(2)))
        gs = fig.add_gridspec(2, len(mods), hspace=0.1, wspace=0.2, 
                            width_ratios=[len(cres) for cres in cres_to_use_mods])
    else:
        fig = plt.figure(figsize=(figsize[0]*(2), figsize[1]*np.sum([len(cres) for cres in cres_to_use_mods])))
        gs = fig.add_gridspec(len(mods), 2, hspace=0.2, wspace=0, 
                            height_ratios=[len(cres) for cres in cres_to_use_mods])
    for i, cres, mod, cpm in zip(range(len(mods)), cres_to_use_mods, mods, cpm_df_list):
        # reorder the cres by the best recall cell type of the first modality
        cres_best_recall_celltype = cpm.loc[cell_types_to_use, cres].idxmax(axis=0)
        # cres_best_recall_celltype = obj.get_creinfo()['best_subclass'].loc[significant_cres]
        cres_best_recall_celltype_idx = cell_types_to_use.get_indexer(cres_best_recall_celltype)
        cres = cres[np.argsort(cres_best_recall_celltype_idx)]
        test_result = test_result_orig[cres]
        cpm = cpm[cres]
        hue_name = mod.replace('_', ' ')
        size_name = 'activity'
        if log:
            hue_name += ' (log)'
            size_name += ' (log)'
        if z_score_by_cre:
            # scale test result to z-score along the cres
            test_result = (test_result - test_result.mean(axis=0)) / test_result.std(axis=0)
            # scale atac_cpm to z-score along the cres
            cpm = (cpm - cpm.mean(axis=0)) / cpm.std(axis=0)
            hue_name += ' (z-score)'
            size_name += ' (z-score)'
        if scale_by_cre:
            test_result = (test_result - test_result.min(axis=0)) / (test_result.max(axis=0) - test_result.min(axis=0))
            cpm = (cpm - cpm.min(axis=0)) / (cpm.max(axis=0) - cpm.min(axis=0))
            hue_name += ' (scaled)'
            size_name += ' (scaled)'
        # find positive controls
        positive_control_df = pd.DataFrame(index=test_result.index, columns=test_result.columns)
        print(positive_control_df.shape)
        for cre in test_result.columns:
            # get the best atac cpm cell type for this cre
            best_cell_types = qval_df[cre].index[qval_df[cre] <= 0.05] if qval_df is not None else None
            if best_cell_types is not None and len(best_cell_types) > 0:
                best_cell_types = best_cell_types[best_cell_types.isin(positive_control_df.index)]
                positive_control_df.loc[best_cell_types, cre] = True
        positive_control_df = positive_control_df.fillna(False)
        # Top row: Heatmaps for selected CREs across all metrics
        # rename mod
        mod_rename = {'atac_cpm': 'ATAC', 'h3k4me1_cpm': 'H3K4me1', 'h3k27ac_cpm': 'H3K27ac', 
                      'h3k9me3_cpm': 'H3K9me3', 'h3k27me3_cpm': 'H3K27me3', 'Activity': 'Activity'}
        matrices_selected = {'Activity': test_result, mod_rename[mod]: cpm}
        # Fixed color map for modalities
        mod_color_dict = {
            'Activity':'#D95C44',   # coral red
            'ATAC':   '#4682B4',   # steel blue
            'H3K4me1':'#66A753',   # dark olive green
            'H3K27ac':'#F57C00',   # amber orange
            'H3K9me3':'#7B68EE',  # medium slate blue
            'H3K27me3':'#984EA3'   # deep purple
        }
        for j, (title, data) in enumerate(matrices_selected.items()):
            if flip:
                ax = fig.add_subplot(gs[j, i])
            else:
                ax = fig.add_subplot(gs[i, j])
            # scale data in each row
            data = (data - data.min(axis=0)) / (data.max(axis=0) - data.min(axis=0))
            base = mod_color_dict[title]
            cmap = LinearSegmentedColormap.from_list(f"{title}_cmap", ['white', base])
            if flip:
                im = ax.imshow(data.values, aspect='auto', cmap=cmap, 
                               vmin=0, vmax=1, interpolation='nearest')
            else:
                im = ax.imshow(data.transpose().values, aspect='auto', cmap=cmap, 
                            vmin=0, vmax=1, interpolation='nearest')
            ax.set_title(f'{title}', fontsize=14, fontweight='bold')
            if flip:
                if i == 0:
                    ax.set_ylabel('Cell Types', fontsize=12)
                    ax.set_yticks(range(len(data.index)))
                    ax.set_yticklabels(data.index, fontsize=8)
                else:
                    ax.set_yticks([])
                if j == 1:
                    ax.set_xlabel('CREs', fontsize=12)
                    ax.set_xticks(range(len(data.columns)))
                    ax.set_xticklabels(data.columns, rotation=45, fontsize=10, ha='right')
                else:
                    ax.set_xticks([])
            else:
                if j == 0:
                    ax.set_ylabel('CREs', fontsize=12)
                    ax.set_yticks(range(len(data.columns)))
                    ax.set_yticklabels(data.columns, fontsize=8)
                else:
                    ax.set_yticks([])
                if i == len(mods) - 1:
                    ax.set_xlabel('Cell Types', fontsize=12)
                    ax.set_xticks(range(len(data.index)))
                    ax.set_xticklabels(data.index, rotation=45, fontsize=10, ha='right')
                else:
                    ax.set_xticks([])
            # show positive control as * if it is significant
            if title == 'Activity':
                for row_idx, cell in enumerate(positive_control_df.index):
                    hits = np.where(positive_control_df.loc[cell].values)[0]
                    for col_idx in hits:
                        # place a star at (x=col_idx, y=row_idx)
                        if flip:
                            ax.text(col_idx, row_idx, '*', ha='center', va='center', color='black', fontsize=8)
                        else:
                            ax.text(row_idx, col_idx, '*', ha='center', va='center', color='black', fontsize=8)
            plt.colorbar(im, ax=ax, shrink=0.6)
    fig.tight_layout()
    # set figure margins
    if flip:
        fig.subplots_adjust(left=0.2, right=0.95, top=0.95, bottom=0.05)
    else:
        fig.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.3)
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

def draw_custom_dendrogram(cre_order_token, ordered_cres, ax, reorder_penalty=0.05, flip_axis=False):
    # fill cre_order_token NA with zeros
    cre_order_token = np.nan_to_num(cre_order_token, nan=0)
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
    # fill dist_matrix with a large number if it encountered Inf distance
    dist_matrix[np.isinf(dist_matrix)] = 1e6
    D = squareform(dist_matrix)
    Z = linkage(D, method='average')
    Z_opt = optimal_leaf_ordering(Z, D)
    if flip_axis:
        orientation = 'bottom'
    else:
        orientation = 'left'
    dendrogram(Z_opt, labels=ordered_cres, orientation=orientation, ax=ax, color_threshold=0.3 * np.max(Z_opt[:, 2]))
    ax.set_yticklabels([])
    ax.tick_params(left=False)
    if flip_axis:
        # flip x axis
        ax.invert_xaxis()

def cre_pval_dotplot(q_value, activity, cres_to_use, cell_types_to_use, positive_control_info, 
                     cre_categories=np.array(['On-target', 'Mix-target', 'Off-target', 'No target', 'CREs', 'Negative Controls']),
                     reorder_cres=True, significant_cutoff=0.05, figsize=(20, 12), z_norm=True, flip_axis=False):
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
            # if best_subclass is seperated by ";", we split
            if pd.notna(best_subclass) and ';' in best_subclass:
                best_subclass = best_subclass.split(';')
            else:
                best_subclass = [best_subclass]
            # significant cres
            cell_types = q_value[cre].index[q_value[cre] <= significant_cutoff]
            for best_sub in best_subclass:
                if pd.isna(best_sub):
                    # do nothing
                    print(f'Warning: {cre} has no best subclass')
                elif best_sub == 'Negative Control':
                    # if the best subclass is negative control, we skip this cre
                    target_df.loc[:, cre] = 'Negative Control'
                    continue
                elif best_sub == 'CRE':
                    # if the best subclass is CREs, we skip this cre
                    target_df.loc[:, cre] = 'CRE'
                    continue
                # on target cell types
                if best_sub in cell_types:
                    target_df.loc[best_sub, cre] = 'on-target'
                    cell_types = cell_types[cell_types != best_sub]
                else:
                    if best_sub in q_value.index:
                        target_df.loc[best_sub, cre] = 'miss'
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
    if z_norm:
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
    height_ratios = np.array([sum(cre_type == category) for category in cre_categories])
    # remove the categories with 0 counts
    cre_categories = cre_categories[height_ratios != 0]
    height_ratios = height_ratios[height_ratios != 0]
    # Create 4 subplots aligned vertically, assign different heights
    if flip_axis:
        fig, axes = plt.subplots(2, len(height_ratios), figsize=figsize, sharex=False, sharey=False,
                                 gridspec_kw={'height_ratios': [1, 0.2], 'width_ratios': height_ratios[::-1]})
    else:    
        fig, axes = plt.subplots(len(height_ratios), 2, figsize=figsize, sharex=False, sharey=False,
                                 gridspec_kw={'height_ratios': height_ratios, 'width_ratios': [0.2, 1]})
    # Plot each CRE type in a subplot
    final_order = []
    for i, category in enumerate(cre_categories):
        if len(cre_categories) == 1:
            ax = axes[1]
            ax_dend = axes[0]
        else:
            if flip_axis:
                ax = axes[0, -i-1]
                ax_dend = axes[1, -i-1]
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
            max_sig_n = (q_value_subset >= -np.log10(significant_cutoff)).sum(axis=1).max()
            cre_order_token = []
            for j, cre in enumerate(category_cres):
                # get significant cell types
                significant_cell_types = q_value_subset.loc[cre][q_value_subset.loc[cre] >= -np.log10(significant_cutoff)].index
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
            if cre_order_token.shape[1] > 0 and reorder_cres:
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
        final_order.extend(ordered_cres)
        subset['cres'] = pd.Categorical(subset['cres'], categories=ordered_cres, ordered=True)
        # dot plots
        if flip_axis:
            x = 'cres'
            y = 'cell_types'
        else:
            x = 'cell_types'
            y = 'cres'
        sns.scatterplot(data=subset, x=x, y=y, size=size_name, hue=hue_name, edgecolor='none',
                        hue_norm=(hue_min, hue_max), size_norm=(size_min, size_max),
                        sizes=(3, 250), alpha=0.8, palette='coolwarm', ax=ax)
        # dendrogram
        draw_custom_dendrogram(cre_order_token[:, :max_sig_n], ordered_cres, ax_dend, flip_axis=flip_axis)
        # ax.invert_yaxis()
        ax_dend.axis('off')
        # Add markers for positive controls
        if positive_control_info is not None and category != 'CREs' and category != 'Negative Controls':
            markers = {
                'on-target': ('red', 's'), 
                'off-target': ('blue', 's'), 
                'miss': ('grey', 's')
            }
            for control, (color, marker) in markers.items():
                data = subset[subset['positive_control'] == control]
                if not data.empty:
                    sns.scatterplot(
                        data=data, x=x, y=y, 
                        s=250, alpha=0.8, marker=marker,
                        facecolor='none', edgecolor=color, legend=False, ax=ax
                    )
        # ax.margins(y=0.5/figsize[1]/height_ratios[i] * max(np.array(height_ratios)))
        if flip_axis:
            ax.margins(x=0.6 / height_ratios[i])
            ax.set_ylim(-0.5, len(cell_types_to_use)-0.5)
        else:
            ax.margins(y=0.6 / height_ratios[i])
            ax.set_xlim(-0.5, len(cell_types_to_use)-0.5) 
        # remove the ticks, x label
        if i != len(cre_categories) - 1:
            if flip_axis:
                # remove y ticks and y label
                ax.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
                ax.set_ylabel('')
                # rotate the x labels
                ax.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=True)
                ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
            else:
                ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
                ax.set_xlabel('')
        else:
            if flip_axis:
                # rotate the y labels
                ax.tick_params(axis='y', which='both', left=True, right=False, labelleft=True)
                ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
                ax.set_ylabel('Cell Types')
                # rotate the x labels
                ax.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=True)
                ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
            else:
                # rotate the x labels
                ax.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=True)
                ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
                ax.set_xlabel('Cell Types')
        if category in ['On-target', 'Mix-target', 'Off-target', 'CREs']:
            # Capture legend handles/labels from the FIRST subplot before removal
            legend = ax.get_legend()
            legend_handles = legend.legend_handles
            legend_labels = [t.get_text() for t in legend.get_texts()]
        # Remove subplot legend
        ax.get_legend().remove()
        if category == 'CREs':
            if flip_axis:
                # remove x axis label
                ax.set_xlabel('')
            else:
                # remove y axis label
                ax.set_ylabel('')
        else:
            if flip_axis:
                ax.set_xlabel(category)
            else:
                ax.set_ylabel(category)
        if flip_axis:
            ax.tick_params(axis='x', which='both', labelsize=8)
        else:
            ax.tick_params(axis='y', which='both', labelsize=8)
    # put legends to right of the plot
    fig.legend(legend_handles, legend_labels, loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10, labelspacing=1.5)
    fig.tight_layout()
    plt.close(fig)
    return fig, final_order

def cre_proportion_dotplot(proportion, activity, cres_to_use, cell_types_to_use, 
                           significant_cutoff=0.05, figsize=(20, 12), flip_axis=False):
    if cres_to_use is None:
        cres_to_use = proportion.columns.intersection(activity.columns)
    if cell_types_to_use is None:
        cell_types_to_use = proportion.index
    cell_types_to_use = cell_types_to_use[cell_types_to_use.isin(proportion.index)]
    # order cell types to use by subcluster number
    cluster_annotation_term = pd.read_csv('Data/abc_atlas/cluster_annotation_term.csv', index_col=0)
    cluster_annotation_term['subclass'] = cluster_annotation_term['subclass'].str.replace('/', '-')
    cell_types_to_use_cluster_number = cluster_annotation_term['subclass_number'].groupby(cluster_annotation_term['subclass']).first().loc[cell_types_to_use].values
    # reorder cell types to use by subcluster number
    cell_types_to_use = pd.Index(cell_types_to_use[np.argsort(cell_types_to_use_cluster_number)])
    # reorder test result and atac cpm
    proportion = proportion.loc[cell_types_to_use, cres_to_use]
    activity = activity.loc[cell_types_to_use, cres_to_use]
    # Data transformations
    # proportion = proportion.clip(lower=1/10).astype(float)  # Clip to avoid log10(0)
    activity = activity.clip(lower=1e-2).astype(float)  # Clip to avoid log10(0)
    activity = np.log10(activity).T  # Transpose for CRE clustering
    activity = activity.sub(activity.mean(axis=1), axis=0).div(activity.std(axis=1), axis=0)  # Z-score per CRE
    proportion = proportion.T
    # scale 
    proportion = proportion.sub(proportion.min(axis=1), axis=0).div((proportion.max(axis=1) - proportion.min(axis=1)), axis=0)  # Scale proportion to [0, 1]
    # proportion = proportion.sub(proportion.mean(axis=1), axis=0).div(proportion.std(axis=1), axis=0)  # Z-score per CRE
    # if scale_by_cre, we scale the test result by the max of each cre
    hue_name = 'log(activity) (z-score)'
    size_name = 'log(proportion) (z-score)'
    # make to plot dataframe, flatten recall and atac_cpm
    toplot = pd.DataFrame({hue_name: activity.T.values.flatten(),
                           size_name: proportion.T.values.flatten()})
    hue_min, hue_max = toplot[hue_name].min(), toplot[hue_name].max()
    size_min, size_max = toplot[size_name].min(), toplot[size_name].max()
    # Cluster CREs within category
    activity_subset = activity.loc[cres_to_use]  # Use pre-transposed data
    proportion_subset = proportion.loc[cres_to_use]
    # set the orders based on q-value
    # create a order token for each CRE
    cre_order_token = []
    for j, cre in enumerate(cres_to_use):
        # get significant cell types
        proportion_best = proportion_subset.loc[cre].idxmax(axis=0)
        activity_best = activity_subset.loc[cre].idxmax(axis=0)
        # get the index of the cell types in cell_types_to_use
        proportion_best_idx = cell_types_to_use.get_indexer([proportion_best]) + 1
        activity_best_idx = cell_types_to_use.get_indexer([activity_best]) + 1
        cre_order_token.append(np.concatenate((activity_best_idx, proportion_best_idx)))
    # sort the cres by the order token
    cre_order_token = np.array(cre_order_token)
    # sort cres by the order token
    if cre_order_token.shape[1] > 0:
        ordered_cres = cres_to_use[np.lexsort(cre_order_token.T[::-1,:])]
        cre_order_token = cre_order_token[np.lexsort(cre_order_token.T[::-1,:])]
    else:
        ordered_cres = cres_to_use
        cre_order_token = np.zeros((len(cres_to_use), 0))
    # Sort data by clustered order
    activity_subset = activity_subset.loc[ordered_cres].copy()
    proportion_subset = proportion.loc[ordered_cres].copy()
    subset = pd.DataFrame({size_name: proportion_subset.T.values.flatten(),
                           hue_name: activity_subset.T.values.flatten(),
                           'cell_types': list(cell_types_to_use.values.repeat(len(ordered_cres))),
                           'cres': np.tile(ordered_cres, len(cell_types_to_use))})
    subset['cres'] = pd.Categorical(subset['cres'], categories=ordered_cres, ordered=True)
    # dot plots
    if flip_axis:
        x = 'cres'
        y = 'cell_types'
    else:
        x = 'cell_types'
        y = 'cres'
    fig, ax = plt.subplots(figsize=figsize)
    sns.scatterplot(data=subset, x=x, y=y, size=hue_name, hue=size_name, edgecolor='none',
                    hue_norm=(hue_min, hue_max), size_norm=(size_min, size_max),
                    sizes=(1, 100), alpha=0.8, palette='coolwarm', ax=ax)
    if flip_axis:
        # rotate the y labels
        ax.tick_params(axis='y', which='both', left=True, right=False, labelleft=True)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
        ax.set_ylabel('Cell Types')
        # rotate the x labels
        ax.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=True)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    else:
        # rotate the x labels
        ax.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=True)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
        ax.set_xlabel('Cell Types')
    legend = ax.get_legend()
    legend_handles = legend.legend_handles
    legend_labels = [t.get_text() for t in legend.get_texts()]
    # remove legend
    ax.get_legend().remove()
    # put legends to right of the plot
    fig.legend(legend_handles, legend_labels, loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10, labelspacing=1.5)
    fig.tight_layout()
    plt.close(fig)
    return fig

def celltype_pval_dotplot(q_value, activity, cres_to_use, cell_types_to_use, positive_control_info, 
                          cre_categories=np.array(['On-target', 'Mix-target', 'Off-target', 'No target', 'CREs', 'Negative Controls']),
                          reorder_cres=True, significant_cutoff=0.05, figsize=(20, 12), activity_log=False, z_norm=True, flip_axis=False):
    if cres_to_use is None:
        cres_to_use = q_value.columns
    cres_to_use = cres_to_use[cres_to_use.isin(q_value.columns)]
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
            # if best_subclass is seperated by ";", we split
            if pd.notna(best_subclass) and ';' in best_subclass:
                best_subclass = best_subclass.split(';')
            else:
                best_subclass = [best_subclass]
            # significant cres
            cell_types = q_value[cre].index[q_value[cre] <= significant_cutoff]
            for best_sub in best_subclass:
                if pd.isna(best_sub):
                    # do nothing
                    print(f'Warning: {cre} has no best subclass')
                elif best_sub == 'Negative Control':
                    # if the best subclass is negative control, we skip this cre
                    target_df.loc[:, cre] = 'Negative Control'
                    continue
                elif best_sub == 'CRE':
                    # if the best subclass is CREs, we skip this cre
                    target_df.loc[:, cre] = 'CRE'
                    continue
                # on target cell types
                if best_sub in cell_types:
                    target_df.loc[best_sub, cre] = 'on-target'
                    cell_types = cell_types[cell_types != best_sub]
                else:
                    if best_sub in q_value.index:
                        target_df.loc[best_sub, cre] = 'miss'
            # other cell types are off-target
            cell_types = cell_types[cell_types.isin(q_value.index)]
            if len(cell_types) > 0 and best_subclass != ['CRE'] and best_subclass != ['Negative Control']:
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
    # filter any non-significant values to nan
    activity = activity.where(q_value <= significant_cutoff, other=np.nan)
    q_value = q_value.clip(lower=1/5000).astype(float)  # Clip to avoid log10(0)
    q_value = q_value.where(q_value <= significant_cutoff, other=np.nan)
    if activity_log:
        activity = activity.clip(lower=1e-2).astype(float)  # Clip to avoid log10(0)
        activity = np.log10(activity)  # Transpose for CRE clustering
    if z_norm:
        activity = activity.T
        activity = activity.sub(np.nanmean(activity, axis=1), axis=0).div(np.nanstd(activity, axis=1), axis=0)  # Z-score per CRE
    else:
        activity = activity.T
    q_value = -np.log10(q_value).T
    # if scale_by_cre, we scale the test result by the max of each cre
    hue_name = 'activity'
    size_name = '-log10(q value)'
    if activity_log:
        hue_name = 'log10(activity)'
    if z_norm:
        hue_name += ' z-score'
    # make to plot dataframe, flatten recall and atac_cpm
    toplot = pd.DataFrame({size_name: q_value.T.values.flatten(),
                            hue_name: activity.T.values.flatten(),
                            'cre_type': np.tile(cre_type, len(cell_types_to_use))})
    hue_min, hue_max = toplot[hue_name].min(), toplot[hue_name].max()
    size_min, size_max = toplot[size_name].min(), toplot[size_name].max()
    # plot dot plot, no edge color, use 4 sub plots
    # Set the height ratios for the subplots
    height_ratios = np.array([sum(cre_type == category) for category in cre_categories])
    # remove the categories with 0 counts
    cre_categories = cre_categories[height_ratios != 0]
    height_ratios = height_ratios[height_ratios != 0]
    # Create 4 subplots aligned vertically, assign different heights
    if flip_axis:
        fig, axes = plt.subplots(2, len(height_ratios), figsize=figsize, sharex=False, sharey=False,
                                    gridspec_kw={'height_ratios': [1, 0.2], 'width_ratios': height_ratios[::-1]})
    else:    
        fig, axes = plt.subplots(len(height_ratios), 2, figsize=figsize, sharex=False, sharey=False,
                                gridspec_kw={'height_ratios': height_ratios, 'width_ratios': [0.2, 1]})
    # Plot each CRE type in a subplot
    final_order = []
    for i, category in enumerate(cre_categories):
        if len(cre_categories) == 1:
            ax = axes[1]
            ax_dend = axes[0]
        else:
            if flip_axis:
                ax = axes[0, -i-1]
                ax_dend = axes[1, -i-1]
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
            max_sig_n = (q_value_subset >= -np.log10(significant_cutoff)).sum(axis=1).max()
            cre_order_token = []
            for j, cre in enumerate(category_cres):
                # get significant cell types
                significant_cell_types = q_value_subset.loc[cre][q_value_subset.loc[cre] >= -np.log10(significant_cutoff)].index
                # get the index of the cell types in cell_types_to_use
                significant_cell_types_idx = cell_types_to_use.get_indexer(significant_cell_types) + 1
                significant_activities = activity_subset.loc[cre][significant_cell_types].values
                significant_activities_mean = significant_activities.mean()
                # sort the significant_cell_types_idx by significant_activities
                significant_cell_types_idx = significant_cell_types_idx[np.argsort(significant_activities)[::-1]]
                significant_activities = np.sort(significant_activities)[::-1]
                # fill to max_sig_n
                significant_cell_types_idx = np.pad(significant_cell_types_idx, (0, max_sig_n - len(significant_cell_types_idx)), constant_values=0)
                significant_activities = np.pad(significant_activities, (0, max_sig_n - len(significant_activities)), constant_values=0)
                # append the q-value
                cre_order_token.append(
                    np.concatenate((significant_activities_mean.reshape(-1), 
                                    np.concatenate((significant_cell_types_idx.reshape(-1, 1), 
                                                    significant_activities.reshape(-1, 1)), axis=1).reshape(-1))))
            # sort the cres by the order token
            cre_order_token = np.array(cre_order_token)
            # sort cres by the order token
            if cre_order_token.shape[1] > 0 and reorder_cres:
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
        # if all nan in q_value_df, we need something in activity_df
        if q_value_subset.isna().all(axis=0).any():
            activity_subset.iloc[:, q_value_subset.isna().all(axis=0)] = 0
            q_value_subset.iloc[:, q_value_subset.isna().all(axis=0)] = 0
        subset = pd.DataFrame({size_name: q_value_subset.T.values.flatten(),
                               hue_name: activity_subset.T.values.flatten(),
                               'cell_types': list(cell_types_to_use.values.repeat(len(ordered_cres))),
                               'cres': np.tile(ordered_cres, len(cell_types_to_use)),
                               'positive_control': target_df_subset.values.flatten()})
        final_order.extend(ordered_cres)
        subset['cres'] = pd.Categorical(subset['cres'], categories=ordered_cres, ordered=True)
        # dot plots
        if flip_axis:
            x = 'cres'
            y = 'cell_types'
        else:
            x = 'cell_types'
            y = 'cres'
        sns.scatterplot(data=subset, x=x, y=y, size=size_name, hue=hue_name, edgecolor='white',
                        hue_norm=(hue_min, hue_max), size_norm=(size_min, size_max),
                        sizes=(3, 250), alpha=0.8, palette='coolwarm', ax=ax)
        # dendrogram
        draw_custom_dendrogram(cre_order_token[:, :max_sig_n], ordered_cres, ax_dend, flip_axis=flip_axis)
        # ax.invert_yaxis()
        ax_dend.axis('off')
        # Add markers for positive controls
        if positive_control_info is not None and category != 'CREs':
            markers = {
                'on-target': ('red', 's'), 
                # 'off-target': ('blue', 's'), 
                'miss': ('grey', 's')
            }
            for control, (color, marker) in markers.items():
                data = subset[subset['positive_control'] == control]
                if not data.empty:
                    sns.scatterplot(
                        data=data, x=x, y=y, 
                        s=250, alpha=0.8, marker=marker,
                        facecolor='none', edgecolor=color, legend=False, ax=ax
                    )
        # ax.margins(y=0.5/figsize[1]/height_ratios[i] * max(np.array(height_ratios)))
        if flip_axis:
            ax.margins(x=0.6 / height_ratios[i])
            ax.set_ylim(-0.5, len(cell_types_to_use)-0.5)
        else:
            ax.margins(y=0.6 / height_ratios[i])
            ax.set_xlim(-0.5, len(cell_types_to_use)-0.5) 
        # remove the ticks, x label
        if i != len(cre_categories) - 1:
            if flip_axis:
                # remove y ticks and y label
                ax.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
                ax.set_ylabel('')
                # rotate the x labels
                ax.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=True)
                ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
            else:
                ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
                ax.set_xlabel('')
        else:
            if flip_axis:
                # rotate the y labels
                ax.tick_params(axis='y', which='both', left=True, right=False, labelleft=True)
                ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
                ax.set_ylabel('Cell Types')
                # rotate the x labels
                ax.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=True)
                ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
            else:
                # rotate the x labels
                ax.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=True)
                ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
                ax.set_xlabel('Cell Types')
        if category in ['On-target', 'Mix-target', 'Off-target', 'CREs']:
            # Capture legend handles/labels from the FIRST subplot before removal
            legend = ax.get_legend()
            legend_handles = legend.legend_handles
            legend_labels = [t.get_text() for t in legend.get_texts()]
        # Remove subplot legend
        ax.get_legend().remove()
        if category == 'CREs':
            if flip_axis:
                # remove x axis label
                ax.set_xlabel('')
            else:
                # remove y axis label
                ax.set_ylabel('')
        else:
            if flip_axis:
                ax.set_xlabel(category)
            else:
                ax.set_ylabel(category)
        if flip_axis:
            ax.tick_params(axis='x', which='both', labelsize=8)
        else:
            ax.tick_params(axis='y', which='both', labelsize=8)
    # put legends to right of the plot
    fig.legend(legend_handles, legend_labels, loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10, labelspacing=1.5)
    fig.tight_layout()
    plt.close(fig)
    return fig, final_order

def plot_q_value_celltype_reproducibility(res_q1, res_q2, res_q, cell_counts1, cell_counts2, cell_counts, q_cutoff = 0.05):
    common_celltypes = res_q1.index.intersection(res_q2.index)
    res_compare = pd.DataFrame(index=common_celltypes, columns=['Sec1', 'Sec2', 'All', 
                                                                'Common', 'Percentage',
                                                                'Common_sec1', 'Percentage_sec1',
                                                                'Common_sec2', 'Percentage_sec2'])
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
    res_compare['Cell_counts1'] = cell_counts1.loc[common_celltypes].values
    res_compare['Cell_counts2'] = cell_counts2.loc[common_celltypes].values
    res_compare['Cell_counts'] = cell_counts.loc[common_celltypes].values
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

def plot_q_value_cre_reproducibility(res_q1, res_q2, res_q, q_cutoff=0.05, plot=True):
    common_cres = res_q1.columns.intersection(res_q2.columns)
    res_compare = pd.DataFrame(index=common_cres, columns=['Sec1', 'Sec2', 'All', 
                                                           'Common', 'Percentage',
                                                           'Common_sec1', 'Percentage_sec1',
                                                           'Common_sec2', 'Percentage_sec2'])
    for cre in common_cres:
        res_compare.loc[cre, 'Sec1'] = (res_q1[cre] <= q_cutoff).sum()
        res_compare.loc[cre, 'Sec2'] = (res_q2[cre] <= q_cutoff).sum()
        res_compare.loc[cre, 'All'] = (res_q[cre] <= q_cutoff).sum()
        res_compare.loc[cre, 'Common'] = res_q1[cre].index[res_q1[cre] <= q_cutoff].intersection(res_q2[cre].index[res_q2[cre] <= q_cutoff]).shape[0]
        res_compare.loc[cre, 'Percentage'] = res_compare.loc[cre, 'Common'] / np.minimum(res_compare.loc[cre, 'Sec1'], res_compare.loc[cre, 'Sec2'])
        res_compare.loc[cre, 'Common_sec1'] = res_q1[cre].index[res_q1[cre] <= q_cutoff].intersection(res_q[cre].index[res_q[cre] <= q_cutoff]).shape[0]
        res_compare.loc[cre, 'Percentage_sec1'] = res_compare.loc[cre, 'Common_sec1'] / np.minimum(res_compare.loc[cre, 'Sec1'], res_compare.loc[cre, 'All'])
        res_compare.loc[cre, 'Common_sec2'] = res_q2[cre].index[res_q2[cre] <= q_cutoff].intersection(res_q[cre].index[res_q[cre] <= q_cutoff]).shape[0]
        res_compare.loc[cre, 'Percentage_sec2'] = res_compare.loc[cre, 'Common_sec2'] / np.minimum(res_compare.loc[cre, 'Sec2'], res_compare.loc[cre, 'All'])
    # plot the percentage vs number of cells
    res_compare['lib_size'] = starrfish3.lib_size.loc[common_cres, 'counts'].values
    if not plot:
        return res_compare
    fig, ax = plt.subplots(ncols=4, figsize=(16, 4))
    # Create separate plots for NaN and non-NaN values
    mask_valid = ~res_compare['Percentage_sec1'].isna()
    mask_nan = res_compare['Percentage_sec1'].isna()
    # Plot valid values with coolwarm palette
    if mask_valid.any():
        sns.scatterplot(x=res_compare.loc[mask_valid, 'Sec1'], 
                            y=res_compare.loc[mask_valid, 'All'],
                            hue=res_compare.loc[mask_valid, 'Percentage_sec1'], 
                            palette='coolwarm', ax=ax[0], alpha=0.5)
    # Plot NaN values in grey
    if mask_nan.any():
        ax[0].scatter(res_compare.loc[mask_nan, 'Sec1'], 
                res_compare.loc[mask_nan, 'All'], edgecolors='none', s=20,  # Adjust size as needed
                color='grey', alpha=0.5, label='NA')
    ax[0].set_xlabel('Significant cell types in Sec1')
    ax[0].set_ylabel('Significant cell types in All')

    mask_valid = ~res_compare['Percentage_sec2'].isna()
    mask_nan = res_compare['Percentage_sec2'].isna()
    # Plot valid values with coolwarm palette
    if mask_valid.any():
        sns.scatterplot(x=res_compare.loc[mask_valid, 'Sec2'], 
                            y=res_compare.loc[mask_valid, 'All'],
                            hue=res_compare.loc[mask_valid, 'Percentage_sec2'], 
                            palette='coolwarm', ax=ax[1], alpha=0.5)
    # Plot NaN values in grey
    if mask_nan.any():
        ax[1].scatter(res_compare.loc[mask_nan, 'Sec2'], 
                res_compare.loc[mask_nan, 'All'], edgecolors='none', s=20,  # Adjust size as needed
                color='grey', alpha=0.5, label='NA')
    ax[1].set_xlabel('Significant cell types in Sec2')
    ax[1].set_ylabel('Significant cell types in All')

    mask_valid = ~res_compare['Percentage'].isna()
    mask_nan = res_compare['Percentage'].isna()
    # Plot valid values with coolwarm palette
    if mask_valid.any():
        sns.scatterplot(x=res_compare.loc[mask_valid, 'Sec1'], 
                        y=res_compare.loc[mask_valid, 'Sec2'],
                        hue=res_compare.loc[mask_valid, 'Percentage'], 
                        palette='coolwarm', ax=ax[2], alpha=0.5)
        sns.scatterplot(x=res_compare.loc[mask_valid, 'lib_size'], 
                        y=res_compare.loc[mask_valid, 'All'],
                        hue=res_compare.loc[mask_valid, 'Percentage'], 
                        palette='coolwarm', ax=ax[3], alpha=0.5)
    # Plot NaN values in grey
    if mask_nan.any():
        ax[2].scatter(res_compare.loc[mask_nan, 'Sec1'], 
                    res_compare.loc[mask_nan, 'Sec2'], edgecolors='none', s=20,  # Adjust size as needed
                    color='grey', alpha=0.5, label='NA')
        ax[3].scatter(res_compare.loc[mask_nan, 'lib_size'], 
                      res_compare.loc[mask_nan, 'All'], edgecolors='none', s=20,  # Adjust size as needed
                      color='grey', alpha=0.5, label='NA')
    ax[2].set_xlabel('Significant cell types in Sec1')
    ax[2].set_ylabel('Significant cell types in Sec2')
    ax[3].set_xscale('log')
    ax[3].set_xlabel('log library size')
    ax[3].set_ylabel('Significant cell types in All')

    # plot violin plot
    fig, ax = plt.subplots(ncols=3, figsize=(12, 4), gridspec_kw={'wspace': 0.4})
    sns.violinplot(y=res_compare['Percentage_sec1'], ax=ax[0])
    sns.violinplot(y=res_compare['Percentage_sec2'], ax=ax[1])
    sns.violinplot(y=res_compare['Percentage'], ax=ax[2])
    return res_compare

def get_pr_df(qvalue_df, starrfish_obj, cell_types_to_use,
              metric = ['atac_cpm', 'h3k4me1_cpm', 'h3k9me3_cpm', 'h3k27ac_cpm', 'h3k27me3_cpm'],
              z_cutoffs=np.arange(0, 5, 0.1), q_threshold=0.05):
    res_df = pd.DataFrame()
    # filter cell_types_to_use based on mod
    for mod in metric:
        mod_cpm = getattr(starrfish_obj, mod).copy()
        cell_types_to_use = cell_types_to_use[cell_types_to_use.isin(mod_cpm.index)]
    # for each CRE, select top rank cell type
    for z in z_cutoffs:
        for mod in metric:
            if mod.endswith('_cpm'):
                mod_cpm = getattr(starrfish_obj, mod).copy()
                qvalue_df = qvalue_df[qvalue_df.columns.intersection(mod_cpm.columns)]
                mod_cpm = mod_cpm.loc[qvalue_df.index.intersection(mod_cpm.index), qvalue_df.columns]
                # log transform
                mod_cpm = np.log1p(mod_cpm.astype(float))
                mod_cpm_z = mod_cpm.sub(mod_cpm.mean(axis=0), axis=1).div(mod_cpm.std(axis=0), axis=1)  # Z-score per CRE
            else:
                mod_cpm_z = getattr(starrfish_obj, mod).copy()
                qvalue_df = qvalue_df[qvalue_df.columns.intersection(mod_cpm_z.columns)]
                mod_cpm_z = mod_cpm_z.loc[qvalue_df.index.intersection(mod_cpm_z.index), qvalue_df.columns]
            for cell_type in cell_types_to_use:
                target_cres = qvalue_df.loc[cell_type].index[qvalue_df.loc[cell_type] <= q_threshold]
                z_score = mod_cpm_z.loc[cell_type]
                if mod in ['h3k9me3_cpm', 'h3k27me3_cpm']:
                    z_score = -z_score
                pred_cres = z_score.index[z_score >= z]
                # on-target and off-target rates
                correct = target_cres.isin(pred_cres).sum()
                all_pred = len(pred_cres)
                res_df = pd.concat((res_df,
                pd.DataFrame({
                    'cell_type': cell_type,
                    'mod': mod.replace('_cpm', ''),
                    'z_cutoff': z,
                    'precision': correct / all_pred if all_pred > 0 else 0,
                    'recall': f'{correct}/{all_pred}' if all_pred > 0 else '0/0',
                    'all_pred': all_pred,
                    'correct': correct,
                    'target': len(target_cres),
                }, index=[0])), ignore_index=True)
    # drop NaN values
    res_df = res_df.dropna(subset=['precision', 'recall'])
    # order by allen institute's nominature
    cluster_annotation_term = pd.read_csv('Data/abc_atlas/cluster_annotation_term.csv', index_col=0)
    cluster_annotation_term['subclass'] = cluster_annotation_term['subclass'].str.replace('/', '-')
    res_df['cell_type_rank'] = cluster_annotation_term['subclass_number'].groupby(cluster_annotation_term['subclass']).first().loc[res_df['cell_type']].values
    # reorder by cell type rank
    res_df = res_df.sort_values(by=['cell_type_rank']).reset_index(drop=True)
    return res_df

def plot_bar(df_bar, legend_loc=None, figsize=(3, 1.5), flip_axis=False, fontsize=3):
    fig, ax = plt.subplots(figsize=figsize)
    df_bar['cell_type'] = df_bar.apply(lambda x: f"{x['cell_type']} ({x['target']})", axis=1)
    # Define the categorical orderings (as used by seaborn)
    cell_type_order = df_bar['cell_type'].unique().tolist()  # or specify manually
    mod_order = df_bar['mod'].unique().tolist()  # or pass hue_order=... to sns.barplot
    df_bar_sorted = (df_bar.copy().astype({'cell_type': pd.CategoricalDtype(categories=cell_type_order, ordered=True),
                                           'mod': pd.CategoricalDtype(categories=mod_order, ordered=True)})
                     .sort_values(['mod', 'cell_type'])
                     .reset_index(drop=True))
    # Plot
    palette = {'atac': '#A6CEE3', 'h3k4me1': '#B2DF8A', 'h3k9me3': '#FB8072',
               'h3k27ac': '#FDB462', 'h3k27me3': '#CAB2D6',
               'chromatin_o': 'blue', 'chromatin_a': 'red', 'snapatac2_de_fc': 'yellow'}
    if flip_axis:
        sns.barplot(data=df_bar_sorted, y='cell_type', x='precision', hue='mod',
                    palette=palette, order=cell_type_order, hue_order=mod_order, ax=ax)
    else:
        sns.barplot(data=df_bar_sorted, x='cell_type', y='precision', hue='mod',
                    palette=palette, order=cell_type_order, hue_order=mod_order, ax=ax)
    # Annotate using df_bar_sorted
    for patch, (_, row) in zip(ax.patches, df_bar_sorted.iterrows()):
        recall = row['recall']
        if flip_axis:
            precision = patch.get_width()
            y = patch.get_y() + patch.get_height() / 2
            ax.text(precision, y, str(recall), va='center', ha='center', fontsize=fontsize)
        else:
            precision = patch.get_height()
            x = patch.get_x() + patch.get_width() / 2
            ax.text(x, precision, str(recall), va='center', ha='center', fontsize=fontsize)
    # set limits and labels based on orientation
    if flip_axis:
        ax.set_xlim(0, df_bar_sorted['precision'].max() + 0.02)
        ax.tick_params(axis='x', labelsize=fontsize)
        ax.set_xlabel('Precision', fontsize=6)
        ax.set_ylabel('')
    else:
        ax.set_ylim(0, df_bar_sorted['precision'].max() + 0.02)
        ax.tick_params(axis='y', labelsize=fontsize)
        ax.set_ylabel('Precision', fontsize=6)
        ax.set_xlabel('')
    # move legend a little bit down
    if legend_loc is not None:
        ax.legend(bbox_to_anchor=legend_loc, loc='upper right', borderaxespad=0.)
    return fig, ax