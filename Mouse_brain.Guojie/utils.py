import statsmodels.formula.api as smf
import numpy as np
import pandas as pd
import warnings
import time
import multiprocessing
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)
from pydeseq2.dds import DeseqDataSet
from pydeseq2.default_inference import DefaultInference
from pydeseq2.ds import DeseqStats
from scipy import stats, optimize
import statsmodels.api as sm
from statsmodels.stats import multitest
from typing import Union, List, Literal
import scanpy as sc
import pickle
import torch
import matplotlib.pyplot as plt
# add current path to sys.path
import sys
import os
import scvi
PWD = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PWD)
from starr_fish_vae import STARRFISHVI
from cmdstanpy import CmdStanModel, cmdstan_path, set_cmdstan_path
from scipy.stats import linregress
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
import logging
from typing import Literal
set_cmdstan_path('/share/vault/Users/gz2294/miniconda3/envs/scvi/bin/cmdstan/')
cmdstanpy_logger = logging.getLogger("cmdstanpy")
cmdstanpy_logger.disabled = True

def fit_glm(formula, y, x, volm, fov, only_keep_positive_per_cre=True):
    try:
        # remove zeros in x
        if only_keep_positive_per_cre:
            to_keep = y > 0
        else:
            to_keep = np.ones(len(x), dtype=bool)
        y = y[to_keep]
        x = x[to_keep]
        volm = volm[to_keep]
        fov = fov[to_keep]
        # if data points too few, return NaN
        if len(y) < 10:
            return pd.DataFrame({'coef': [np.nan], 'P>|t|': [np.nan]}, index=['x'])
        fit_data=pd.DataFrame({'y': y, 'x': x, 'volm': volm, 'fov': fov})
        glm_results = smf.ols(formula, data=fit_data).fit()
        glm_summary = pd.read_html(glm_results.summary().tables[1].as_html(), header=0, index_col=0)[0]
        glm_summary = pd.DataFrame(glm_summary.loc['x']).T
    except Exception as e:
        return pd.DataFrame({'coef': [np.nan], 'P>|t|': [np.nan]}, index=['x'])
    return glm_summary


def glm(adata, cell_types_to_use=None, CREs=None, norm_by_volm=False, volm_covariate=False, fov_covariate=False, 
        filter_infected_cells=True, only_keep_positive_per_cre=True, multiprocess_threads=256, verbose=False):
    # result is a matrix of cell_types x CREs
    if CREs is None:
        CREs = adata.uns['CRE_info'].index
    # if filter infected cells is True, only keep the cells with CREs > 0
    if filter_infected_cells:
        infected = ((adata.obsm['CRE'] > 0).sum(axis=1) > 0)
        adata = adata[infected].copy()
    # get unique cell types
    if cell_types_to_use is None:
        cell_types_to_use = adata.obs['subclass'].value_counts()
        # filter out the cell types with less than 20 cells
        cell_types_to_use = cell_types_to_use[cell_types_to_use >= 20].index.tolist()
    adata.obsm['RNA'] = adata.obsm['X_raw'].copy()
    if norm_by_volm:
        adata.obsm['RNA'] = adata.obsm['RNA'] / adata.obs['volm'].values[:, np.newaxis]
    coef = pd.DataFrame(index=cell_types_to_use, columns=CREs)
    pvalue = pd.DataFrame(index=cell_types_to_use, columns=CREs)
    formula = 'y ~ x'
    if volm_covariate:
        formula += ' + volm'
    if fov_covariate:
        formula += ' + C(fov)'
    if multiprocess_threads is not None:
        pool = multiprocessing.Pool(processes=min(multiprocess_threads, int(multiprocessing.cpu_count()*0.8)))
    for k, cell_type in enumerate(cell_types_to_use):
        start = time.time()
        # get the data for the cell type
        adata_cell_type = adata[adata.obs['subclass'] == cell_type].copy()
        # get the data for the CREs
        adata_cre = adata_cell_type.obsm['CRE'][CREs]
        # get the data for the RNA
        adata_rna = adata_cell_type.obsm['RNA'].sum(axis=1)
        # fit the model, use multiprocessing to speed up the process
        if multiprocess_threads is not None:
            results = pool.starmap(fit_glm, [(formula, adata_cre[cre], adata_rna, 
                                              adata_cell_type.obs['volm'].values, 
                                              adata_cell_type.obs['fov'].values,
                                              only_keep_positive_per_cre) for cre in CREs])
            if verbose:
                print('Finished fitting for cell type:', cell_type, ' (', k+1, '/', len(cell_types_to_use), ')',  
                      'Time taken:', time.time() - start, 'seconds')
            # get the results
            for i, glm_summary in enumerate(results):
                coef.loc[cell_type, CREs[i]] = glm_summary['coef'].loc['x']
                pvalue.loc[cell_type, CREs[i]] = glm_summary['P>|t|'].loc['x']
        else:
            for cre in CREs:
                glm_summary = fit_glm(formula, adata_cre[cre], adata_rna, 
                                      adata_cell_type.obs['volm'].values, 
                                      adata_cell_type.obs['fov'].values,
                                      only_keep_positive_per_cre)
                coef.loc[cell_type, cre] = glm_summary['coef'].values[0]
                pvalue.loc[cell_type, cre] = glm_summary['P>|t|'].values[0]
            # print the time taken
            if verbose:
                print('Finished fitting for cell type:', cell_type, ' (', k+1, '/', len(cell_types_to_use), ')',  
                      'Time taken:', time.time() - start, 'seconds')
    # return is coef and pvalue
    if multiprocess_threads is not None:
        pool.close()
    result = {'coef': coef, 'pvalue': pvalue}
    return result


def create_pseudo_bulk(i, adata_cell_type: pd.DataFrame, adata_non_cell_type: pd.DataFrame, percentage_bootstrap):
    # set seed as i
    np.random.seed(i)
    pseudo_bulk_celltype = pd.DataFrame(index=[f'celltype_{i}'], columns=adata_cell_type.columns)
    pseudo_bulk_non_celltype = pd.DataFrame(index=[f'non_celltype_{i}'], columns=adata_non_cell_type.columns)
    # randomly select percentage_bootstrap of cells from cell_type
    adata_cell_type_sample = adata_cell_type.sample(frac=percentage_bootstrap, replace=False)
    adata_non_cell_type_sample = adata_non_cell_type.sample(frac=percentage_bootstrap, replace=False)
    # sum the cells
    pseudo_bulk_celltype.iloc[0] = adata_cell_type_sample.sum(axis=0)
    pseudo_bulk_non_celltype.iloc[0] = adata_non_cell_type_sample.sum(axis=0)
    # pseudo_bulk = pd.concat([pseudo_bulk_celltype, pseudo_bulk_non_celltype])
    return pseudo_bulk_celltype, pseudo_bulk_non_celltype


def cre_deseq2(adata, cell_type, pseudo_bulk_number=1000, replace=True, percentage_bootstrap=0.5, multi_processes=128):
    # do pydeseq2 differential expression
    # randomly select percentage_bootstrap of cells from cell_type
    adata_cell_type = adata[adata.obs['subclass'] == cell_type].obsm['CRE'].copy()
    adata_non_cell_type = adata[adata.obs['subclass'] != cell_type].obsm['CRE'].copy()
    if replace:
        if multi_processes is not None:
            with multiprocessing.Pool(processes=min(multi_processes, int(multiprocessing.cpu_count()*0.8))) as pool:
                pseudo_bulks = pool.starmap(create_pseudo_bulk, [(i, adata_cell_type, adata_non_cell_type, percentage_bootstrap) for i in range(pseudo_bulk_number)])
        else:
            pseudo_bulks = [create_pseudo_bulk(i, adata_cell_type, adata_non_cell_type, percentage_bootstrap) for i in range(pseudo_bulk_number)]
        # combine the data
        pseudo_bulk_celltype = pd.concat([pd[0] for pd in pseudo_bulks])
        pseudo_bulk_non_celltype = pd.concat([pd[1] for pd in pseudo_bulks])
        pseudo_bulk = pd.concat([pseudo_bulk_celltype, pseudo_bulk_non_celltype])
    else:
        # no replace, percentage_bootstrap should be 1/pseudo_bulk_number
        # get a random sample of pseudo_bulk_number classes
        cell_type_assignment = np.random.choice(pseudo_bulk_number, size=adata_cell_type.shape[0], replace=True)
        non_cell_type_assignment = np.random.choice(pseudo_bulk_number, size=adata_non_cell_type.shape[0], replace=True)
        # create psudo_bulk
        pseudo_bulk_celltype = adata_cell_type.groupby(cell_type_assignment).sum()
        pseudo_bulk_non_celltype = adata_non_cell_type.groupby(non_cell_type_assignment).sum()
        # combine the data
        pseudo_bulk = pd.concat([pseudo_bulk_celltype, pseudo_bulk_non_celltype])
    condition_celltype = pd.DataFrame(index=[f'celltype_{i}' for i in range(pseudo_bulk_number)], columns=['condition'])
    condition_non_celltype = pd.DataFrame(index=[f'non_celltype_{i}' for i in range(pseudo_bulk_number)], columns=['condition'])
    condition_celltype['condition'] = 'B'
    condition_non_celltype['condition'] = 'A'
    # combine the condition
    condition = pd.concat([condition_celltype, condition_non_celltype])
    dds = DeseqDataSet(
        counts=pseudo_bulk,
        metadata=condition,
        design="~condition",  # compare samples based on the "condition" B vs A
        refit_cooks=True,
        inference=DefaultInference(n_cpus=256),
    )
    dds.deseq2()
    ds = DeseqStats(
        dds,
        contrast=np.array([0, 1]),
        alpha=0.05,
        cooks_filter=True,
        independent_filter=True,
    )
    ds.run_wald_test()
    ds.summary()
    return ds


def plot_gene_scdata(scdata2, gene='SOX9', use='X', nmax=None, sz_min=5, sz_max=30,
                     transpose=1, flipx=1, flipy=1, tag='X_spatial'):
    Xcells = scdata2.obsm[tag][:, ::transpose] * [flipx, flipy]
    # get best cell type
    if use == 'CRE':
        best_celltype = scdata2.uns['CRE_info'].loc[gene, 'best_subclass']
    # Get expression data
    if use == 'X':
        gene_idx = list(scdata2.var.index).index(gene)
        cts = scdata2.X[:, gene_idx].copy()
    else:
        cts = scdata2.obsm[use][gene].copy()
    
    # Prepare plot parameters
    cts = np.nan_to_num(cts)
    nmax = np.nanmax(cts) if nmax is None else nmax
    ncts = np.clip(cts/nmax, 0, 1)
    size = sz_min + ncts * (sz_max - sz_min)
    cmap = plt.cm.coolwarm(ncts)
    
    # Create single figure and axes
    if use == 'CRE':
        fig, ax = plt.subplots(1, 2, figsize=(30, 10), facecolor='k')
        plot_cluster_scdata(scdata2, clusters=[best_celltype], use='subclass', 
                            transpose=transpose, flipx=flipx, flipy=flipy, 
                            sbig=sz_max, small=1, ax=ax[1], plot_legend=False)
        ax[0].set_title(f'{gene} ({best_celltype}) - N max {nmax}', color='white')
        ax[0].set_facecolor('black')
        
        # Plot data
        XC = -Xcells[:, ::-1]
        cell_with_genes = np.where(cts > 0)[0]
        # first plot cells without genes, then plot cells with genes
        ax[0].scatter(XC[:, 0], XC[:, 1], c='grey', s=sz_min, marker='.')
        # ax[0].scatter(XC[~cell_with_genes, 0], XC[~cell_with_genes, 1], c=cmap[~cell_with_genes], s=size[~cell_with_genes])
        ax[0].scatter(XC[cell_with_genes, 0], XC[cell_with_genes, 1], c=cmap[cell_with_genes], s=sz_max)
        # ax[0].scatter(XC[:, 0], XC[:, 1], c=cmap, s=size)
        
        # Format axes
        ax[0].grid(False)
        ax[0].set_xticks([])
        ax[0].set_yticks([])
        ax[0].set_aspect('equal')
    else:
        fig, ax = plt.subplots(figsize=(10, 10), facecolor='black')
        ax.set_title(f'{gene} - N max {nmax}', color='white')
        ax.set_facecolor('black')
        
        # Plot data
        XC = -Xcells[:, ::-1]
        cell_with_genes = np.where(cts > 0)[0]
        # first plot cells without genes, then plot cells with genes
        ax.scatter(XC[:, 0], XC[:, 1], c='grey', s=sz_min, marker='.')
        # ax.scatter(XC[~cell_with_genes, 0], XC[~cell_with_genes, 1], c=cmap[~cell_with_genes], s=size[~cell_with_genes])
        ax.scatter(XC[cell_with_genes, 0], XC[cell_with_genes, 1], c=cmap[cell_with_genes], s=sz_max)
        
        # Format axes
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('equal')
    
    fig.tight_layout()
    return None


def plot_cluster_scdata(scdata, clusters=['Endo NN'], use='subclass', 
                        transpose=1, flipx=1, flipy=1, sbig=30, small=5, 
                        x_region=None, y_region=None, cmap=None,
                        ax=None, plot_legend = False, tag='X_spatial',
                        figsize=(20,10)):
    Xcells = scdata.obsm[tag][:, ::transpose] * [flipx, flipy]
    if cmap is None:
        cmap = scdata.uns['cmap']
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, facecolor="black")
        toreturn = True
    else:
        fig = ax.figure
        toreturn = False
    x = Xcells[:, 0]
    y = Xcells[:, 1]
    x_ = x.copy()
    y_ = y.copy()
    if x_region is not None:
        select_region = (x_ > x_region[0]) & (x_ < x_region[1])
        x_ = x_[select_region]
        y_ = y_[select_region]
    if y_region is not None:
        select_region = (y_ > y_region[0]) & (y_ < y_region[1])
        x_ = x_[select_region]
        y_ = y_[select_region]
    plt.scatter(x_, y_, c='gray', s=small, marker='.', rasterized=True)
    for i, cluster in enumerate(clusters):
        cluster_ = str(cluster)
        inds = scdata.obs[use] == cluster_
        x_ = x[inds]
        y_ = y[inds]
        col = cmap[i % len(cmap)]
        if x_region is not None:
            select_region = (x_ > x_region[0]) & (x_ < x_region[1])
            x_ = x_[select_region]
            y_ = y_[select_region]
        if y_region is not None:
            select_region = (y_ > y_region[0]) & (y_ < y_region[1])
            x_ = x_[select_region]
            y_ = y_[select_region]
        ax.scatter(x_, y_, c=col, s=sbig, marker='.',label = cluster_, rasterized=True)
    
    # if cluster len is 1, then plot title
    ax.set_title(f"Cell types", color='white', fontsize=20)
    if plot_legend:
        # if cluster len larger than 5, plot it outside
        if len(clusters) > 5:
            ax.legend(fontsize=20, loc='upper left', bbox_to_anchor=(1.05, 1))
        else:
            ax.legend(fontsize=20, loc='lower right')
    # Format axes
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal')
    ax.set_facecolor('black')
    if toreturn:
        fig.tight_layout()
        plt.close(fig)
        return fig


def calculate_fold_change(cre_celltypes_expression: pd.DataFrame, cell_types_to_use: pd.Series, CRE_info: pd.DataFrame,
                          rna_celltypes_expression: pd.DataFrame, volm: pd.Series,
                          normalize_by_celltype_rna=False, normalize_by_celltype_volume=False,
                          normalize_by_negative_control=False, lib_size=None,
                          normalize_by_infected_cell=False, normalize_by_libsize=False, filter_zero_counts=False,
                          rank_transform=None):
    foldchange = pd.DataFrame(index=pd.unique(cell_types_to_use), columns=cre_celltypes_expression.columns)
    if filter_zero_counts:
        # get the number of infected cells for each CRE in each cell type
        celltype_activity_matrix = cre_celltypes_expression.groupby(cell_types_to_use).sum()
        non_zero_cells = (cre_celltypes_expression > 0).groupby(cell_types_to_use).sum()
        # before division, fill zeros with 1
        non_zero_cells[non_zero_cells == 0] = 1
        celltype_activity_matrix = celltype_activity_matrix / non_zero_cells
    else:
        # just average
        celltype_activity_matrix = cre_celltypes_expression.groupby(cell_types_to_use).mean()
    celltype_rna_matrix = rna_celltypes_expression.mean(axis=1).groupby(cell_types_to_use).mean()
    celltype_volm_matrix = volm.groupby(cell_types_to_use).mean()
    if normalize_by_celltype_rna:
        # get cell type RNA
        celltype_activity_matrix = celltype_activity_matrix / celltype_rna_matrix.values.reshape(-1, 1)
    if normalize_by_celltype_volume:
        # get cell type volume
        celltype_activity_matrix = celltype_activity_matrix / celltype_volm_matrix.values.reshape(-1, 1)
    if normalize_by_libsize and not normalize_by_negative_control:
        # normalize by lib size
        celltype_activity_matrix = celltype_activity_matrix / lib_size.values.reshape(1, -1)
    if normalize_by_negative_control:
        # get the negative control
        negative_control = CRE_info[CRE_info['labeling_type'] == 'negative control'].index
        negative_control_mean = celltype_activity_matrix.loc[:, negative_control].mean(axis=1)
        # get the negative control lib size
        negative_control_lib_size = lib_size.loc[negative_control].mean(axis=0)
        if normalize_by_libsize:
            negative_control_mean = negative_control_mean / negative_control_lib_size
            celltype_activity_matrix = celltype_activity_matrix / lib_size.values.reshape(1, -1)
        # normalize by negative control
        celltype_activity_matrix = celltype_activity_matrix / negative_control_mean.values.reshape(-1, 1)
    if normalize_by_infected_cell:
        # get infect rates per cell type
        infected = ((cre_celltypes_expression >= 1).sum(axis=1) > 0)
        infect_rate_celltype = infected.groupby(cell_types_to_use).mean()
        # normalize by the infected cell type
        celltype_activity_matrix = celltype_activity_matrix / infect_rate_celltype.values.reshape(-1, 1)
    if rank_transform is not None:
        # rank transform the data
        if rank_transform == 'cre':
            # shuffle the data
            celltype_activity_matrix = celltype_activity_matrix.sample(frac=1, axis=1, random_state=0)
            celltype_activity_matrix = celltype_activity_matrix.rank(axis=1, method='first')
        elif rank_transform == 'celltype':
            celltype_activity_matrix = celltype_activity_matrix.sample(frac=1, axis=0, random_state=0)
            celltype_activity_matrix = celltype_activity_matrix.rank(axis=0, method='first')
        return celltype_activity_matrix, celltype_activity_matrix
    for celltype in pd.unique(cell_types_to_use):
        # get the data for the cell type
        celltype_activity = celltype_activity_matrix.loc[celltype]
        # get non_celltype activity
        non_celltype_activity = cre_celltypes_expression[cell_types_to_use != celltype].mean(axis=0)
        non_celltype_rna = rna_celltypes_expression.mean(axis=1)[cell_types_to_use != celltype].mean(axis=0)
        non_celltype_volm = volm[cell_types_to_use != celltype].mean()
        if normalize_by_celltype_rna:
            non_celltype_activity = non_celltype_activity / non_celltype_rna
        if normalize_by_celltype_volume:
            non_celltype_activity = non_celltype_activity / non_celltype_volm
        if normalize_by_libsize and not normalize_by_negative_control:
            non_celltype_activity = non_celltype_activity / lib_size
        if normalize_by_negative_control:
            non_celltype_negative_control = non_celltype_activity[negative_control].sum()
            if normalize_by_libsize:
                non_celltype_activity = non_celltype_activity / lib_size
                non_celltype_negative_control = non_celltype_negative_control / negative_control_lib_size
            non_celltype_activity = non_celltype_activity / non_celltype_negative_control
        if normalize_by_infected_cell:
            non_celltype_infect_rate = infected[cell_types_to_use != celltype].mean(axis=0)
            non_celltype_activity = non_celltype_activity / non_celltype_infect_rate
        foldchange.loc[celltype] = celltype_activity / non_celltype_activity
    return foldchange, celltype_activity_matrix


def col_corr(df1: pd.DataFrame, df2: pd.DataFrame, bin_threshold1=None, bin_threshold2=None):
    # do col wise correlation
    col_result = pd.DataFrame(index=df1.columns, 
                              columns=['pearson', 'spearman', 'fisher', 
                                       'pearson_p', 'spearman_p', 'fisher_p',
                                       'pearson_q', 'spearman_q', 'fisher_q'])
    for i, cre in enumerate(df1.columns):
        # calculate the correlation
        try:
            tokeep = np.isfinite(df1[cre]) & np.isfinite(df2[cre])
            x1 = df1[cre][tokeep].astype(float)
            x2 = df2[cre][tokeep].astype(float)
            pearson = stats.pearsonr(x1, x2)
            spearman = stats.spearmanr(x1, x2)
            # fisher test
            if bin_threshold1 is not None:
                bin1 = df1[cre] > bin_threshold1
            else:
                bin1 = df1[cre] > df1[cre].mean()
            if bin_threshold2 is not None:
                bin2 = df2[cre] > bin_threshold2
            else:
                bin2 = df2[cre] > df2[cre].mean()
            fisher = stats.fisher_exact(pd.crosstab(bin1, bin2))
            # add the result to the dataframe
            col_result.loc[cre, 'pearson'] = pearson[0]
            col_result.loc[cre, 'spearman'] = spearman[0]
            col_result.loc[cre, 'fisher'] = fisher[0]
            col_result.loc[cre, 'pearson_p'] = pearson[1]
            col_result.loc[cre, 'spearman_p'] = spearman[1]
            col_result.loc[cre, 'fisher_p'] = fisher[1]
        except:
            print('Error in calculating correlation for CRE: ', cre)
            col_result.loc[cre, 'pearson'] = 0
            col_result.loc[cre, 'spearman'] = 0
            col_result.loc[cre, 'fisher'] = 0
            col_result.loc[cre, 'pearson_p'] = 1
            col_result.loc[cre, 'spearman_p'] = 1
            col_result.loc[cre, 'fisher_p'] = 1
    col_result['pearson_q'] = multitest.multipletests(col_result['pearson_p'], method='fdr_bh')[1]
    col_result['spearman_q'] = multitest.multipletests(col_result['spearman_p'], method='fdr_bh')[1]
    col_result['fisher_q'] = multitest.multipletests(col_result['fisher_p'], method='fdr_bh')[1]
    return col_result


def row_corr(df1: pd.DataFrame, df2: pd.DataFrame, bin_threshold1=None, bin_threshold2=None):
    # do row wise correlation
    row_result = pd.DataFrame(index=df1.index,
                              columns=['pearson', 'spearman', 'pearson_p', 'spearman_p', 'pearson_q', 'spearman_q'])
    for i, celltype in enumerate(df1.index):
        # calculate the correlation
        try:
            tokeep = np.isfinite(df1.loc[celltype]) & np.isfinite(df2.loc[celltype])
            x1 = df1.loc[celltype][tokeep].astype(float)
            x2 = df2.loc[celltype][tokeep].astype(float)
            pearson = stats.pearsonr(x1, x2)
            spearman = stats.spearmanr(x1, x2)
            # fisher test
            if bin_threshold1 is not None:
                bin1 = df1.loc[celltype] > bin_threshold1
            else:
                bin1 = df1.loc[celltype] > df1.loc[celltype].mean()
            if bin_threshold2 is not None:
                bin2 = df2.loc[celltype] > bin_threshold2
            else:
                bin2 = df2.loc[celltype] > df2.loc[celltype].mean()
            fisher = stats.fisher_exact(pd.crosstab(bin1, bin2))
            # add the result to the dataframe
            row_result.loc[celltype, 'pearson'] = pearson[0]
            row_result.loc[celltype, 'spearman'] = spearman[0]
            row_result.loc[celltype, 'fisher'] = fisher[0]
            row_result.loc[celltype, 'pearson_p'] = pearson[1]
            row_result.loc[celltype, 'spearman_p'] = spearman[1]
            row_result.loc[celltype, 'fisher_p'] = fisher[1]
        except:
            print('Error in calculating correlation for celltype: ', celltype)
            row_result.loc[celltype, 'pearson'] = 0
            row_result.loc[celltype, 'spearman'] = 0
            row_result.loc[celltype, 'fisher'] = 0
            row_result.loc[celltype, 'pearson_p'] = 1
            row_result.loc[celltype, 'spearman_p'] = 1
            row_result.loc[celltype, 'fisher_p'] = 1
    row_result['pearson_q'] = multitest.multipletests(row_result['pearson_p'], method='fdr_bh')[1]
    row_result['spearman_q'] = multitest.multipletests(row_result['spearman_p'], method='fdr_bh')[1]
    row_result['fisher_q'] = multitest.multipletests(row_result['fisher_p'], method='fdr_bh')[1]
    return row_result


def cross_talk_fisher_test(celltype_activated: np.ndarray):
    pval = np.ndarray((1, celltype_activated.shape[1], celltype_activated.shape[1]))
    for j in range(celltype_activated.shape[1]):
        for k in range(j, celltype_activated.shape[1]):
            # do fisher exact test
            cre1_activated = celltype_activated[:, j]
            cre2_activated = celltype_activated[:, k]
            # create contingency table and do fisher exact test
            _, p = stats.fisher_exact(pd.crosstab(cre1_activated, cre2_activated))
            pval[0, j, k] = p
            pval[0, k, j] = p
    return pval


def cross_talk_corr_test(celltype_expression: np.ndarray, method='pearson'):
    pval = np.ndarray((1, celltype_expression.shape[1], celltype_expression.shape[1]))
    corr = np.ndarray((1, celltype_expression.shape[1], celltype_expression.shape[1]))
    for j in range(celltype_expression.shape[1]):
        for k in range(j, celltype_expression.shape[1]):
            # do fisher exact test
            cre1_activated = celltype_expression[:, j]
            cre2_activated = celltype_expression[:, k]
            # create contingency table and do fisher exact test
            if method == 'spearman':
                cor, p = stats.spearmanr(pd.to_numeric(cre1_activated), pd.to_numeric(cre2_activated))
            elif method == 'pearson':
                cor, p = stats.pearsonr(pd.to_numeric(cre1_activated), pd.to_numeric(cre2_activated))
            pval[0, j, k] = p
            pval[0, k, j] = p
            corr[0, j, k] = cor
            corr[0, k, j] = cor
    return pval, corr


def fit_sklearn_gauss_mixture(x, x_orig):
    estimator = GaussianMixture(n_components=2, covariance_type="full", max_iter=20, random_state=0)
    # remove zeros and NaNs
    x1 = x[x != 0 & ~np.isnan(x)]
    x1_orig = x_orig[x != 0 & ~np.isnan(x)]
    if len(x1_orig) <= 50:
        fit_res = {
            'means': np.array([[np.nan], [np.nan]]), # dim = (2, 1)
            'covariances': np.array([[np.nan], [np.nan]]), # dim = (2, 1, 1)
            'weights': np.array([[np.nan], [np.nan]]), # dim = (2,)
            'y_pred': np.zeros(x.shape[0]),
            'x': x1
        }
        return fit_res
    estimator.means_init = np.array(
        [[np.log1p(0)], [np.log1p(2)]]
    )
    # Train the other parameters using the EM algorithm.
    estimator.fit(x1.reshape(-1, 1))
    fit_res = {
        'means': estimator.means_, # dim = (2, 1)
        'covariances': estimator.covariances_, # dim = (2, 1, 1)
        'weights': estimator.weights_, # dim = (2,)
        'y_pred': estimator.predict(x1.reshape(-1, 1)),
        'x': x1
    }
    return fit_res


def fit_stan(x, model, chains=4):
    # filter out zeros
    x1 = x[x != 0]
    y1u = np.unique(x1)
    data = {'N1': x1.shape[0], 'N1u': y1u.shape[0], 
            'y1': x1, 'y1u': y1u}
    # fit the model
    fit = model.sample(data=data, chains=chains, parallel_chains=chains, iter_sampling=int(4000/chains), iter_warmup=int(4000/chains), show_progress=False, show_console=False)
    # return the results
    return fit.summary().copy()


class STARRFISH:
    def __init__(self, adata: Union[sc.AnnData, str], 
                 cre_tag = 'obsm:CRE', celltype_tag='obs:subclass', spatial_tag='obsm:X_spatial', creinfo_tag='uns:CRE_info',
                 atac_cpm: Union[pd.DataFrame, str] = 'Data/ATAC/cpm_peakBysubclass.csv',
                 atac_counts: Union[pd.DataFrame, str] = 'Data/ATAC/count_peakBysubclass.csv',
                 lib_size: Union[pd.DataFrame, str] = 'Data/SFv8_400CRE_nanopore_counts.csv'):
        if isinstance(adata, str):
            self.adata_path = adata
            self.load_adata(adata)
        else:
            self.adata_path = None
            self.adata: sc.AnnData = adata.copy()
        self.cre_tag = cre_tag
        self.celltype_tag = celltype_tag
        self.spatial_tag = spatial_tag
        self.cre_info_tag = creinfo_tag
        # change the "/" to "-" in obs:subclass
        self.adata.obs['subclass'] = self.adata.obs['subclass'].str.replace('/', '-')
        # if celltype tag is "obs:class", we transform atac_cpm to class level
        if celltype_tag == 'obs:subclass':
            if isinstance(atac_cpm, str):
                atac_cpm = pd.read_csv(atac_cpm, index_col=0)
                atac_cpm.columns = atac_cpm.columns.str.replace('\\.', '-')
                atac_cpm.columns = atac_cpm.columns.str.replace('_', ' ')
            if isinstance(atac_counts, str):
                atac_counts = pd.read_csv(atac_counts, index_col=0)
                atac_counts.columns = atac_counts.columns.str.replace('\\.', '-')
                atac_counts.columns = atac_counts.columns.str.replace('_', ' ')
        elif celltype_tag == 'obs:class':
            allen_cell_type_nomination = pd.read_excel('Data/abc_atlas/allen_institute_nominature.xlsx', sheet_name='subclass_annotation')
            allen_cell_type_nomination['subclass_label'] = allen_cell_type_nomination['subclass_label'].str.replace('/', '-')
            atac_counts = pd.read_csv('Data/ATAC/count_peakBysubclass.csv', index_col=0)
            atac_counts.columns = atac_counts.columns.str.replace('\\.', '-')
            atac_counts.columns = atac_counts.columns.str.replace('_', ' ')
            # transpose and group by class
            atac_counts = atac_counts.transpose()
            atac_counts_class = allen_cell_type_nomination['class_label'].groupby(allen_cell_type_nomination['subclass_label']).first().loc[atac_counts.index]
            atac_counts = atac_counts.groupby(atac_counts_class).sum()
            # norm to cpm
            atac_cpm = atac_counts.div(atac_counts.sum(axis=1), axis=0) * 1e7
            # transpose back
            atac_cpm = atac_cpm.transpose()
            atac_counts = atac_counts.transpose()
            # assign the obs['class'] to adata
            self.adata.obs['class'] = allen_cell_type_nomination['class_label'].groupby(allen_cell_type_nomination['subclass_label']).first().loc[self.adata.obs['subclass']].values
            # to be safe, change subclass to class as well
            self.adata.obs['subclass_orig'] = self.adata.obs['subclass'].copy()
            self.adata.obs['subclass'] = self.adata.obs['class']
            # change the cre info
            best_class = self.get_creinfo()['best_subclass'].copy()
            best_class[best_class != ''] = allen_cell_type_nomination['class_label'].groupby(allen_cell_type_nomination['subclass_label']).first().loc[best_class[best_class != '']].values
            self.adata.uns['CRE_info']['best_class'] = best_class
            # just in case, change the subclass to class
            self.adata.uns['CRE_info']['best_subclass_orig'] = self.adata.uns['CRE_info']['best_subclass'].copy()
            self.adata.uns['CRE_info']['best_subclass'] = best_class
        elif celltype_tag == 'obs:subclass_rename':
            # rename subclass to allen institute AAV screen paper
            subclass_rename = pd.read_excel('Data/abc_atlas/allen_institute_subclass_rename.xlsx')
            subclass_rename['subclass_simple_label'] = subclass_rename['subclass_simple_label'].str.replace('/', '-')
            subclass_rename['subclass_label'] = subclass_rename['subclass_label'].str.replace('/', '-')
            # filter to non-NaN
            subclass_rename = subclass_rename[subclass_rename['subclass_simple_label'].notna()]
            # filter adata to subclass_rename
            self.adata = self.adata[self.adata.obs['subclass'].isin(subclass_rename['subclass_label'])].copy()
            # rename subclass
            self.adata.obs['subclass'] = subclass_rename['subclass_simple_label'].groupby(subclass_rename['subclass_label']).first().loc[self.adata.obs['subclass']].values
            # rename atac_cpm
            atac_counts = pd.read_csv('Data/ATAC/count_peakBysubclass.csv', index_col=0)
            atac_counts.columns = atac_counts.columns.str.replace('\\.', '-')
            atac_counts.columns = atac_counts.columns.str.replace('_', ' ')
            # transpose and group by class
            atac_counts = atac_counts.transpose()
            atac_counts = atac_counts.loc[atac_counts.index.isin(subclass_rename['subclass_label'])]
            atac_counts_class = subclass_rename['subclass_simple_label'].groupby(subclass_rename['subclass_label']).first().loc[atac_counts.index]
            atac_counts = atac_counts.groupby(atac_counts_class).sum()
            # norm to cpm
            atac_cpm = atac_counts.div(atac_counts.sum(axis=1), axis=0) * 1e7
            # transpose back
            atac_cpm = atac_cpm.transpose()
            atac_counts = atac_counts.transpose()
            # change celltype_tag to subclass
            self.celltype_tag_orig = 'obs:subclass_rename'
            self.celltype_tag = 'obs:subclass'
        if atac_cpm is not None:
            # only keep the cres that are in cre_info
            cre_info = self.get_creinfo().copy()
            cre_info = cre_info[cre_info['enh'].isin(atac_cpm.index)]
            atac_cpm = atac_cpm.loc[cre_info['enh']]
            atac_cpm.index = cre_info.index
            # transpose the atac_cpm
            self.atac_cpm = atac_cpm.transpose()
        if atac_counts is not None:
            # only keep the cres that are in cre_info
            cre_info = self.get_creinfo().copy()
            cre_info = cre_info[cre_info['enh'].isin(atac_counts.index)]
            atac_counts = atac_counts.loc[cre_info['enh']]
            atac_counts.index = cre_info.index
            # transpose the atac_counts
            self.atac_counts = atac_counts.transpose()
        if isinstance(lib_size, str):
            lib_size = pd.read_csv(lib_size, index_col=0)
        if lib_size is not None:
            # match the index with CRE_info
            lib_size = lib_size.loc[lib_size.index.isin(self.get_creinfo().index)]
            # reindex the lib_size to match the CRE_info
            lib_size = lib_size.reindex(self.get_creinfo().index, fill_value=0)
            self.lib_size = lib_size
            
    def save(self, path, overwrite_adata=False):
        # save self
        if self.adata_path is None:
            self.adata_path = f'{path}_adata.h5ad'
        if self.adata is not None:
            if not os.path.exists(self.adata_path) or overwrite_adata:
                self.adata.write(self.adata_path)
        # drop self.adata and save other attributes
        adata = self.adata.copy()
        self.adata = None
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        # put adata back
        self.adata = adata
            
    @staticmethod
    def load(path, adata: Union[sc.AnnData, str]=None) -> 'STARRFISH':
        # load self
        with open(path, 'rb') as f:
            starrfish = pickle.load(f)
        # load adata
        if isinstance(adata, str):
            starrfish.adata_path = adata
        elif isinstance(adata, sc.AnnData):
            starrfish.adata_path = None
            starrfish.adata = adata
        if starrfish.adata_path is not None:
            starrfish.load_adata(starrfish.adata_path)
        return starrfish
    
    def load_adata(self, adata_path):
        # load the adata
        adata = sc.read(adata_path)
        self.adata: sc.AnnData = adata
        
    def get_tag(self, tag) -> Union[pd.DataFrame, pd.Series]:
        # get the CREs
        tag_attr = tag.split(':')[0]
        tag_col = tag.split(':')[1]
        return self.adata.__getattribute__(tag_attr)[tag_col]
    
    def get_cre_expression(self) -> pd.DataFrame:
        return self.get_tag(self.cre_tag)

    def get_rna_expression(self) -> pd.DataFrame:
        # get the RNA expression
        return self.get_tag('obsm:X_raw')

    def get_cre_expression_normalized(self, cell_types_to_use=None, normalize_by_cell_rna=True, normalize_by_volume=True, log_transform=False) -> pd.DataFrame:
        if cell_types_to_use is not None:
            # get the cell types
            cre_celltypes_expression, rna_celltypes_expression, cell_types_to_use = self.get_cre_rna_celltypes(cell_types_to_use)
        else:
            cre_celltypes_expression = self.get_cre_expression().copy()
            rna_celltypes_expression = self.get_rna_expression().copy()
            cell_types_to_use = self.get_celltypes()
        if normalize_by_cell_rna and normalize_by_volume:
            volm = self.get_tag('obs:volm').copy()
            volm = volm.loc[cell_types_to_use.index]
            rna_per_volume = rna_celltypes_expression / volm.values.reshape(-1, 1)
            cre_celltypes_expression = cre_celltypes_expression / rna_per_volume.mean(axis=1).reshape(-1, 1)
        elif normalize_by_cell_rna and not normalize_by_volume:
            cre_celltypes_expression = cre_celltypes_expression / rna_celltypes_expression.mean(axis=1).reshape(-1, 1)
        elif normalize_by_volume and not normalize_by_cell_rna:
            volume = self.get_tag('obs:volm').copy()
            volume = volume.loc[cell_types_to_use.index]
            cre_celltypes_expression = cre_celltypes_expression / volume.values.reshape(-1, 1)
        if log_transform:
            cre_celltypes_expression = np.log1p(cre_celltypes_expression)
        return cre_celltypes_expression, cell_types_to_use

    def get_celltypes(self) -> pd.Series:
        # get the cell types
        return self.get_tag(self.celltype_tag)
    
    def get_cre_celltypes(self, celltypes) -> tuple[pd.DataFrame, pd.Series]:
        # get cre for the cell types
        cres = self.get_cre_expression().copy()
        celltypes_orig = self.get_celltypes().copy()
        # get the cre for the cell types
        cre_celltypes = cres[celltypes_orig.isin(celltypes)]
        celltypes = celltypes_orig[celltypes_orig.isin(celltypes)]
        return cre_celltypes, celltypes

    def get_cre_rna_celltypes(self, celltypes) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
        # get cre for the cell types
        cres = self.get_cre_expression().copy()
        rna = self.get_rna_expression().copy()
        celltypes_orig = self.get_celltypes().copy()
        # get the cre for the cell types
        cre_celltypes = cres[celltypes_orig.isin(celltypes)]
        rna_celltypes = rna[celltypes_orig.isin(celltypes)]
        celltypes = celltypes_orig[celltypes_orig.isin(celltypes)]
        return cre_celltypes, rna_celltypes, celltypes
    
    def get_creinfo(self) -> pd.DataFrame:
        # get the CRE info
        return self.get_tag(self.cre_info_tag)
    
    def get_negative_control_cres(self) -> pd.Series:
        # get the negative control cres
        cres = self.get_creinfo().copy()
        cres = cres[cres['labeling_type'] == 'negative control']
        return cres.index
    
    def get_positive_control_cres(self, cell_type) -> pd.Series:
        # get the positive control cres
        cres = self.get_creinfo().copy()
        cres = cres[cres['best_subclass'] == cell_type]
        return cres.index
    
    def get_atac_z_cres(self, cell_type, z=2) -> pd.Series:
        # get the positive control cres
        atac_cpm_z = self.atac_cpm.copy()
        atac_cpm_z = np.log1p(atac_cpm_z)
        if cell_type not in atac_cpm_z.index:
            return None
        atac_cpm_z = atac_cpm_z.loc[cell_type]
        atac_cpm_z = (atac_cpm_z - atac_cpm_z.mean()) / atac_cpm_z.std()
        cres = atac_cpm_z[atac_cpm_z > z]
        return cres.index
    
    def plot_gene(self, gene='CRE129', use='CRE', 
                  average_by_celltype=False, 
                  norm_by_negative_control_cell_type_mean=False, norm_by_negative_control_cell_type_sum=True, 
                  norm_by_negative_control_single_cell=False, log=True,
                  cell_types_to_use=None, cell_types_to_visualize=None, 
                  nmin=None, nmax=None, sz_min=5, sz_max=30, scale_size_by: Literal['counts', 'celltype_number']='counts',  
                  cmap_name='Reds',
                  x_region=None, y_region=None, select_region_by_best_celltype=False, show_celltypes=True,
                  transpose=1, flipx=1, flipy=1, smooth_k=None, figsize=(30, 10)):
        tag = self.spatial_tag.split(':')[1]
        Xcells = self.adata.obsm[tag][:, ::transpose] * [flipx, flipy]
        # get best cell type
        if use == 'CRE':
            if cell_types_to_visualize is None:
                best_celltype = [self.adata.uns['CRE_info'].loc[gene, 'best_subclass']]
            else:
                best_celltype = list(cell_types_to_visualize)
        # Get expression data
        if use == 'X':
            gene_idx = list(self.adata.var.index).index(gene)
            cts = self.adata.X[:, gene_idx].copy()
        else:
            cts = self.adata.obsm[use][gene].copy()
        # if average_by_celltype, then average by cell type
        if average_by_celltype:
            # get the cell types
            cell_type_cts = cts.groupby(self.get_celltypes()).mean()
            cts = cell_type_cts.loc[self.get_celltypes()].copy()
            # rename the index
            cts.index = self.get_celltypes().index
        # if norm_by_negative_control, then normalize by negative control
        if norm_by_negative_control_cell_type_mean:
            negative_control_counts = self.get_cre_expression()[self.get_negative_control_cres()].sum(axis=1).groupby(self.get_celltypes()).mean()
            norm_factor = negative_control_counts.loc[self.get_celltypes()]
            cts = cts / norm_factor.values
        if norm_by_negative_control_cell_type_sum:
            negative_control_counts = self.get_cre_expression()[self.get_negative_control_cres()].sum(axis=1).groupby(self.get_celltypes()).sum()
            norm_factor = negative_control_counts.loc[self.get_celltypes()]
            cts = cts / norm_factor.values
        if norm_by_negative_control_single_cell:
            negative_control_counts = self.get_cre_expression()[self.get_negative_control_cres()].sum(axis=1)
            cts = cts / negative_control_counts.values
        if log:
            cts = np.log1p(cts)
        if cell_types_to_use is not None:
            # only cts for the cell types to use
            cts[~self.get_celltypes().isin(cell_types_to_use)] = np.nan
        # Prepare plot parameters
        cts = np.nan_to_num(cts, nan=0, posinf=0, neginf=0)
        # if smoothing_k is not None, then smooth the data spatially by k-nearest neighbors
        if smooth_k is not None:
            # get the k nearest neighbors
            knn = NearestNeighbors(n_neighbors=smooth_k, algorithm='ball_tree').fit(Xcells)
            _, indices = knn.kneighbors(Xcells)
            # smooth the data
            cts = np.nanmean(cts[indices], axis=1)
        # filter out nmin
        if nmin is not None:
            cts[cts < nmin] = 0
        nmax = np.nanmax(cts) if nmax is None else nmax
        ncts = np.clip(cts/nmax, 0, 1)
        if scale_size_by == 'counts':
            size = sz_min + ncts * (sz_max - sz_min)
        elif scale_size_by == 'celltype_number':
            # get the number of cell types
            celltype_number = self.get_celltypes().value_counts().loc[self.get_celltypes()].values
            # if not in cell_types_to_use, then set to 0
            celltype_number[~self.get_celltypes().isin(cell_types_to_use)] = 0
            # normalize the celltype_number
            celltype_number = celltype_number.max() - celltype_number + 1
            celltype_number = np.clip(celltype_number / celltype_number.max(), 0, 1)
            size = sz_min + celltype_number * (sz_max - sz_min)
        cmap = plt.get_cmap(cmap_name)(ncts)
        if select_region_by_best_celltype:
            # select region by best_celltype
            xmin = Xcells[:, 0].min()
            xmax = Xcells[:, 0].max()
            ymin = Xcells[:, 1].min()
            ymax = Xcells[:, 1].max()
            for celltype in best_celltype:
                # get the cell type
                celltype_idx = self.get_celltypes() == celltype
                # get the coordinates of the cells
                x_min = Xcells[celltype_idx, 0].min()
                x_max = Xcells[celltype_idx, 0].max()
                y_min = Xcells[celltype_idx, 1].min()
                y_max = Xcells[celltype_idx, 1].max()
                # select the region
                if x_min > xmin:
                    xmin = x_min
                if x_max < xmax:
                    xmax = x_max
                if y_min > ymin:
                    ymin = y_min
                if y_max < ymax:
                    ymax = y_max
            x_region = (xmin, xmax)
            y_region = (ymin, ymax)
        if x_region is not None:
            select_region = (Xcells[:, 0] > x_region[0]) & (Xcells[:, 0] < x_region[1])
            Xcells = Xcells[select_region]
            cts = cts[select_region]
            cmap = cmap[select_region]
            size = size[select_region]
        if y_region is not None:
            select_region = (Xcells[:, 1] > y_region[0]) & (Xcells[:, 1] < y_region[1])
            Xcells = Xcells[select_region]
            cts = cts[select_region]
            cmap = cmap[select_region]
            size = size[select_region]
        # Create single figure and axes
        if use == 'CRE':
            if show_celltypes:
                fig = plt.figure(figsize=figsize, facecolor='k')
                gs = fig.add_gridspec(1, 3, width_ratios=[0.49, 0.02, 0.49], wspace=0.05)
                ax_main = fig.add_subplot(gs[0])
                ax_cbar = fig.add_subplot(gs[1])
                ax_ctypes = fig.add_subplot(gs[2])
                plot_cluster_scdata(self.adata, clusters=best_celltype, use='subclass', 
                                    transpose=transpose, flipx=flipx, flipy=flipy, 
                                    x_region=x_region, y_region=y_region,
                                    sbig=np.minimum(sz_max, 30), small=sz_min, ax=ax_ctypes, plot_legend=True)
            else:
                gs = fig.add_gridspec(1, 3, width_ratios=[0.95, 0.05], wspace=0.05)
                ax_main = fig.add_subplot(gs[0])
                ax_cbar = fig.add_subplot(gs[1])
            ax_main.set_title(f'{gene}', color='white', fontsize=20)
            ax_main.set_facecolor('black')
            
            # Plot data
            cell_with_genes = np.where(cts > 0)[0]
            # first plot cells without genes, then plot cells with genes
            ax_main.scatter(Xcells[:, 0], Xcells[:, 1], c='grey', s=sz_min, marker='.', rasterized=True)
            ax_main.scatter(Xcells[cell_with_genes, 0], Xcells[cell_with_genes, 1], c=cmap[cell_with_genes], sizes=size[cell_with_genes], rasterized=True)
            
            # Format axes
            ax_main.grid(False)
            ax_main.set_xticks([])
            ax_main.set_yticks([])
            ax_main.set_aspect('equal')
            
            # Set colorbar axis background
            ax_cbar.set_facecolor('black')

            # Create colorbar in the dedicated axis
            sm = plt.cm.ScalarMappable(cmap=plt.get_cmap(cmap_name), norm=plt.Normalize(vmin=0, vmax=nmax))
            cbar = plt.colorbar(
                sm,
                cax=ax_cbar,  # Use dedicated axis
                orientation='vertical',
                shrink=0.1          # Scale height of colorbar (0.8 = 80% of plot height)
            )

            # Format colorbar
            cbar.set_label('Normalized Counts', color='white', fontsize=16)
            cbar.ax.yaxis.set_tick_params(color='white')
            cbar.ax.tick_params(labelcolor='white', labelsize=10)
            cbar.ax.set_yticks(np.linspace(0, nmax, 5))
            cbar.ax.set_yticklabels([f'{i:.2f}' for i in np.linspace(0, nmax, 5)], color='white')

            # Remove axis spines from colorbar
            ax_cbar.spines['top'].set_visible(False)
            ax_cbar.spines['right'].set_visible(False)
            ax_cbar.spines['bottom'].set_visible(False)
            ax_cbar.spines['left'].set_visible(False)
        else:
            fig, ax = plt.subplots(figsize=(10, 10), facecolor='black')
            ax.set_title(f'{gene} - N max {nmax}', color='white')
            ax.set_facecolor('black')
            
            # Plot data
            XC = -Xcells[:, ::-1]
            cell_with_genes = np.where(cts > 0)[0]
            # first plot cells without genes, then plot cells with genes
            ax.scatter(XC[:, 0], XC[:, 1], c='grey', s=sz_min, marker='.', rasterized=True)
            # ax.scatter(XC[~cell_with_genes, 0], XC[~cell_with_genes, 1], c=cmap[~cell_with_genes], s=size[~cell_with_genes])
            ax.scatter(XC[cell_with_genes, 0], XC[cell_with_genes, 1], c=cmap[cell_with_genes], s=sz_max, rasterized=True)
            
            # Format axes
            ax.grid(False)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect('equal')
        
        fig.tight_layout()
        plt.close(fig)
        return fig

    def plot_activate_cells(self, cre='CRE001', activate_threshold=2, atac_z_score_threshold=2, 
                            remove_celltypes='Non neuron',
                            sz_min=5, sz_max=30,  transpose=1, flipx=1, flipy=1):
        tag = self.spatial_tag.split(':')[1]
        Xcells = self.adata.obsm[tag][:, ::transpose] * [flipx, flipy]
        cre_expression = self.get_cre_expression().copy()
        cell_types_to_use = self.get_celltypes().copy()
        activate_threshold = self.estimate_activate_threshold_array(activate_threshold, cre_expression, cell_types_to_use)
        activated_cells = cre_expression[cre] > activate_threshold
        # remove Non neuron cell types
        activated_cells = activated_cells & (~cell_types_to_use.str.contains('NN'))
        # get best cell type, by atac z-score ≥ 2
        atac_z_score = np.log1p(self.atac_cpm[cre].copy().astype(float))
        atac_z_score = (atac_z_score - atac_z_score.mean()) / atac_z_score.std()
        target_celltypes = atac_z_score[atac_z_score >= atac_z_score_threshold].index
        # Prepare plot parameters
        ncts = cre_expression[cre]
        ncts[activated_cells] = 1
        # size = sz_min + ncts * (sz_max - sz_min)
        cmap = plt.cm.coolwarm(ncts)
        # Create single figure and axes
        fig, ax = plt.subplots(1, 2, figsize=(30, 10), facecolor='k')
        plot_cluster_scdata(self.adata, clusters=target_celltypes.unique().tolist(), use='subclass', 
                            transpose=transpose, flipx=flipx, flipy=flipy, 
                            sbig=sz_max, small=1, ax=ax[1], plot_legend=False)
        ax[0].set_title(f'{cre}', color='white', fontsize=20)
        ax[0].set_facecolor('black')
        # Plot data
        XC = -Xcells[:, ::-1]
        cell_with_genes = activated_cells
        # first plot cells without genes, then plot cells with genes
        ax[0].scatter(XC[:, 0], XC[:, 1], c='grey', s=sz_min, marker='.')
        ax[0].scatter(XC[cell_with_genes, 0], XC[cell_with_genes, 1], c=cmap[cell_with_genes], s=sz_max)
        # ax[0].scatter(XC[:, 0], XC[:, 1], c=cmap, s=size)
        # Format axes
        ax[0].grid(False)
        ax[0].set_xticks([])
        ax[0].set_yticks([])
        ax[0].set_aspect('equal')
        fig.tight_layout()
        return None
         
    def plot_cluster(self, clusters=['Endo NN'], use='subclass',
                     transpose=1, flipx=1, flipy=1, sbig=30, small=5, cmap=None,
                     x_region=None, y_region=None, plot_legend = False, figsize=(20, 10)):
        tag = self.spatial_tag.split(':')[1]
        return plot_cluster_scdata(self.adata, clusters=clusters, use=use, transpose=transpose, 
                                   flipx=flipx, flipy=flipy, sbig=sbig, small=small, tag=tag, cmap=cmap,
                                   x_region=x_region, y_region=y_region, plot_legend = plot_legend, figsize=figsize)
    
    def plot_umap(self, clusters=['Endo NN'], use='subclass', cmap=None, size=1,
                  ax=None, plot_legend = False, tag='X_umap', figsize=(20,10)):
        Xcells = self.adata.obsm[tag]
        if cmap is None:
            cmap = self.adata.uns['cmap']
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize, facecolor="white")
            toreturn = True
        else:
            fig = ax.figure
            toreturn = False
        x = Xcells[:, 0]
        y = Xcells[:, 1]
        x_ = x.copy()
        y_ = y.copy()
        plt.scatter(x_, y_, c='gray', s=size, marker='.', rasterized=True)
        for i, cluster in enumerate(clusters):
            cluster_ = str(cluster)
            inds = self.adata.obs[use] == cluster_
            x_ = x[inds]
            y_ = y[inds]
            col = cmap[i % len(cmap)]
            ax.scatter(x_, y_, c=col, s=size, marker='.',label = cluster_, rasterized=True)
        
        # if cluster len is 1, then plot title
        ax.set_title(f"Cell types", color='black', fontsize=20)
        if plot_legend:
            # if cluster len larger than 5, plot it outside
            if len(clusters) > 5:
                ax.legend(fontsize=5, loc='upper left', bbox_to_anchor=(1.05, 1))
            else:
                ax.legend(fontsize=5, loc='lower right')
        # Format axes
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('equal')
        ax.set_facecolor('white')
        if toreturn:
            fig.tight_layout()
            plt.close(fig)
            return fig
    
    def cre_deseq2(self, cell_type, pseudo_bulk_number=1000, replace=True, 
                   percentage_bootstrap=0.5, multi_processes=128) -> DeseqStats:
        config = {'cell_type': cell_type, 
                  'pseudo_bulk_number': pseudo_bulk_number,
                  'replace': replace}
        if replace:
            config['percentage_bootstrap'] = percentage_bootstrap
        # check if the results already exist
        if hasattr(self, 'cre_deseq2_results') and hasattr(self, 'cre_deseq2_configs'):
            for stored_config, cre_deseq2_result in zip(self.cre_deseq2_configs, self.cre_deseq2_results):
                if stored_config == config:
                    # if the results already exist, return the results
                    print('Results already exist, return stored results')
                    return cre_deseq2_result.copy()
        # do pydeseq2 differential expression
        result = cre_deseq2(self.adata, cell_type, pseudo_bulk_number, replace,
                             percentage_bootstrap, multi_processes)
        # save results to attribute, save config to attribute
        if not hasattr(self, 'cre_deseq2_results') or not hasattr(self, 'cre_deseq2_configs'):
            self.cre_deseq2_results = []
            self.cre_deseq2_configs = []
        self.cre_deseq2_results.append(result)
        self.cre_deseq2_configs.append(config)
        return result
        
    def fisher_exact_test(self, cell_types_to_use: List=None, activate_threshold=2, infect_threshold=1) -> dict:
        config = {
            'cell_types_to_use': cell_types_to_use,
            'infect_threshold': infect_threshold,
            'activate_threshold': activate_threshold
        }
        # check if the results already exist
        if hasattr(self, 'fisher_exact_test_results') and hasattr(self, 'fisher_exact_test_configs'):
            for stored_config, fisher_exact_test_result in zip(self.fisher_exact_test_configs, self.fisher_exact_test_results):
                if stored_config == config:
                    # if the results already exist, return the results
                    print('Results already exist, return stored results')
                    return fisher_exact_test_result.copy()
        if cell_types_to_use is not None:
            # get the cell types
            cre_celltypes_expression, cell_types_to_use = self.get_cre_celltypes(cell_types_to_use)
        else:
            cre_celltypes_expression = self.get_cre_expression().copy()
            cell_types_to_use = self.get_celltypes()
        infected = ((cre_celltypes_expression >= infect_threshold).sum(axis=1) > 0)
        cre_celltypes_expression = cre_celltypes_expression[infected]
        cell_types_to_use = cell_types_to_use[infected]
        activated = cre_celltypes_expression >= activate_threshold
        p_value, q_value, precision, recall, foldchange, precision_n, recall_n = (
            pd.DataFrame(index=cell_types_to_use.unique(), columns=cre_celltypes_expression.columns) for _ in range(7)
        )
        for cell_type in cell_types_to_use.unique():
            celltype_activated = activated.loc[cell_types_to_use == cell_type]
            noncelltype_activated = activated.loc[cell_types_to_use != cell_type]
            # create contingency table
            # -------                               | other cells                   | celltype
            # overall infected but not activated    | FF = ~noncelltype_activated   | FT = ~celltype_activated
            # activated                             | TF = noncelltype_activated    | TT = celltype_activated
            FF = (~noncelltype_activated).sum(axis=0)
            FT = (~celltype_activated).sum(axis=0)
            TF = (noncelltype_activated).sum(axis=0)
            TT = (celltype_activated).sum(axis=0)
            for cre in cre_celltypes_expression.columns:
                # do fisher exact test
                oddsratio, p = stats.fisher_exact([[int(FF[cre]), int(FT[cre])], [int(TF[cre]), int(TT[cre])]])
                p_value.loc[cell_type, cre] = p
                foldchange.loc[cell_type, cre] = oddsratio
            precision.loc[cell_type] = TT / (TT + TF)
            recall.loc[cell_type] = TT / (TT + FT)
            precision_n.loc[cell_type] = TT + TF
            recall_n.loc[cell_type] = TT + FT
            q_value.loc[cell_type] = multitest.multipletests(p_value.loc[cell_type], method='fdr_bh')[1]
        # for CRE in CRE_info, assign the p, q to best_subclass
        cre_info = self.get_creinfo().copy()
        for cre in cre_info.index:
            # get the best subclass
            best_subclass = cre_info.loc[cre, 'best_subclass']
            # get the p, q values for the best subclass
            if best_subclass in p_value.index:
                cre_info.loc[cre, 'p_value'] = p_value.loc[best_subclass, cre]
                cre_info.loc[cre, 'q_value'] = q_value.loc[best_subclass, cre]
                cre_info.loc[cre, 'precision'] = precision.loc[best_subclass, cre]
                cre_info.loc[cre, 'recall'] = recall.loc[best_subclass, cre]
                cre_info.loc[cre, 'precision_n'] = precision_n.loc[best_subclass, cre]
                cre_info.loc[cre, 'recall_n'] = recall_n.loc[best_subclass, cre]
                cre_info.loc[cre, 'foldchange'] = foldchange.loc[best_subclass, cre]
            # calculate the entropy for each cre
            cre_info.loc[cre, 'entropy'] = stats.entropy(precision[cre].astype(float))
        # save results to attribute
        fisher_exact_test_result = {
            'cre_info': cre_info,
            'p_value': p_value,
            'q_value': q_value,
            'precision': precision,
            'recall': recall,
            'precision_n': precision_n,
            'recall_n': recall_n,
            'foldchange': foldchange
        }
        if not hasattr(self, 'fisher_exact_test_results') or not hasattr(self, 'fisher_exact_test_configs'):
            self.fisher_exact_test_results = []
            self.fisher_exact_test_configs = []
        self.fisher_exact_test_results.append(fisher_exact_test_result)
        self.fisher_exact_test_configs.append(config)
        return fisher_exact_test_result
    
    def estimate_activate_threshold_array(self, activate_threshold, cre_celltypes_expression, cell_types_to_use) -> np.ndarray:
        # if activate_threshold is "celltype", then set cell type specific threshold
        if activate_threshold == 'celltype_mean_2std':
            celltype_activate_threshold = pd.DataFrame(index=cell_types_to_use.unique(), columns=['threshold'])
            # get the cell type cre_expression flattened
            for celltype in cell_types_to_use.unique():
                # get the cre_expression for the cell type
                celltype_cre_expr_flattened = cre_celltypes_expression[cell_types_to_use == celltype].values.flatten()
                # filter out zero, calculate mean and std
                cell_type_cre_expr_flattened = np.log10(celltype_cre_expr_flattened[celltype_cre_expr_flattened > 0])
                mean = cell_type_cre_expr_flattened.mean()
                std = cell_type_cre_expr_flattened.std()
                # set the threshold
                celltype_activate_threshold.loc[celltype, 'threshold'] = np.power(10, mean + 2*std)
            activate_threshold_array = celltype_activate_threshold['threshold'].loc[cell_types_to_use].values
        elif activate_threshold == 'celltype_top100':
            celltype_activate_threshold = pd.DataFrame(index=cell_types_to_use.unique(), columns=['threshold'])
            # get the cell type cre_expression flattened
            for celltype in cell_types_to_use.unique():
                # get the cre_expression for the cell type
                celltype_cre_expr_flattened = cre_celltypes_expression[cell_types_to_use == celltype].values.flatten()
                # set the threshold as the top 100 cre value, if less than 2, set to 2
                celltype_cre_expr_flattened = np.sort(celltype_cre_expr_flattened)[::-1]
                if len(celltype_cre_expr_flattened) < 100:
                    thres = 1
                else:
                    thres = celltype_cre_expr_flattened[100]
                celltype_activate_threshold.loc[celltype, 'threshold'] = np.maximum(thres, 1)
            activate_threshold_array = celltype_activate_threshold['threshold'].loc[cell_types_to_use].values
        elif activate_threshold == 'celltype_poisson_point_estimate':
            celltype_activate_threshold = pd.DataFrame(index=cell_types_to_use.unique(), columns=['threshold'])
            # get the cell type cre_expression flattened
            for celltype in cell_types_to_use.unique():
                # get the cre_expression for the cell type
                celltype_cre_expr_flattened = cre_celltypes_expression[cell_types_to_use == celltype].values.flatten()
                # calculate poisson lambda
                one_count_proportion = (celltype_cre_expr_flattened == 1).sum() / len(celltype_cre_expr_flattened)
                two_count_proportion = (celltype_cre_expr_flattened == 2).sum() / len(celltype_cre_expr_flattened)
                poisson_lambda = two_count_proportion / one_count_proportion * 2
                # set the threshold
                celltype_activate_threshold.loc[celltype, 'threshold'] = stats.poisson.ppf(0.999, mu=poisson_lambda)
            activate_threshold_array = celltype_activate_threshold['threshold'].loc[cell_types_to_use].values
        elif activate_threshold == 'celltype_poisson_fit':
            celltype_activate_threshold = pd.DataFrame(index=cell_types_to_use.unique(), columns=['threshold'])
            def negative_log_likelihood(params, data):
                pi, lambda1, lambda2 = params
                pi = np.clip(pi, 0, 1)  # Ensure π ∈ [0,1]
                lambda1 = max(lambda1, 1e-6)  # Avoid invalid λ ≤ 0
                lambda2 = max(lambda2, 1e-6)
                log_likelihood = np.log(pi * stats.poisson.pmf(data, lambda1) + (1 - pi) * stats.poisson.pmf(data, lambda2))
                return -np.sum(log_likelihood)
            def fit(data):
                # Initial guess and bounds
                initial_params = [0.5, np.mean(data)/5, np.mean(data)*5]
                bounds = [(0, 1), (1e-6, None), (1e-6, None)]
                # Optimize
                result = optimize.minimize(
                    negative_log_likelihood, 
                    initial_params, 
                    args=(data,), 
                    bounds=bounds, 
                    method='L-BFGS-B'
                )
                pi, lambda1, lambda2 = result.x
                return pi, lambda1, lambda2
            # get the cell type cre_expression flattened
            for celltype in cell_types_to_use.unique():
                # get the cre_expression for the cell type
                celltype_cre_expr_flattened = cre_celltypes_expression[cell_types_to_use == celltype].values.flatten()
                # calculate poisson lambda
                _, _, lambda2 = fit(celltype_cre_expr_flattened)
                # set the threshold
                celltype_activate_threshold.loc[celltype, 'threshold'] = lambda2
            activate_threshold_array = celltype_activate_threshold['threshold'].loc[cell_types_to_use].values
        elif isinstance(activate_threshold, (int, float)):
            # is numeric, just use the threshold
            activate_threshold_array = np.full(len(cell_types_to_use), activate_threshold)
        return activate_threshold_array
    
    def fisher_exact_cre_test(self, cell_types_to_use: List=None, activate_threshold=2, infect_threshold=1) -> dict:
        config = {
            'cell_types_to_use': cell_types_to_use,
            'infect_threshold': infect_threshold,
            'activate_threshold': activate_threshold
        }
        # check if the results already exist
        if hasattr(self, 'fisher_exact_cre_test_results') and hasattr(self, 'fisher_exact_cre_test_configs'):
            for stored_config, fisher_exact_cre_test_result in zip(self.fisher_exact_cre_test_configs, self.fisher_exact_cre_test_results):
                if stored_config == config:
                    # if the results already exist, return the results
                    print('Results already exist, return stored results')
                    return fisher_exact_cre_test_result.copy()
        if cell_types_to_use is not None:
            # get the cell types
            cre_celltypes_expression, cell_types_to_use = self.get_cre_celltypes(cell_types_to_use)
        else:
            cre_celltypes_expression = self.get_cre_expression().copy()
            cell_types_to_use = self.get_celltypes()
        p_value, q_value, precision, recall, foldchange, precision_n, recall_n = (
            pd.DataFrame(index=cell_types_to_use.unique(), columns=cre_celltypes_expression.columns) for _ in range(7)
        )
        # if activate_threshold is "celltype", then set cell type specific threshold
        activate_threshold_array = self.estimate_activate_threshold_array(activate_threshold, cre_celltypes_expression, cell_types_to_use)
        for cre in cre_celltypes_expression.columns:
            cre_infected = cre_celltypes_expression[cre] >= infect_threshold
            cre_infected_expression = cre_celltypes_expression[cre][cre_infected]
            cell_types_infected = cell_types_to_use[cre_infected]
            cre_activated = cre_infected_expression >= activate_threshold_array[cre_infected]
            # check the number of cells in each cell type that are infected and activated
            n_celltype_infected = cell_types_infected.value_counts().reindex(cell_types_to_use.unique(), fill_value=0)
            n_celltype_activated = cell_types_infected[cre_activated].value_counts().reindex(cell_types_to_use.unique(), fill_value=0)
            # create contingency table
            # -------       | infected but not activated                                    | activated
            # other cells   | FF = sum(n_celltype_infected) - sum(n_celltype_activated) - TF| FT = sum(n_celltype_activated) - n_celltype_activated
            # celltype      | TF = n_celltype_infected - n_celltype_activated               | TT = n_celltype_activated
            TT = n_celltype_activated
            FT = n_celltype_activated.sum() - n_celltype_activated
            TF = n_celltype_infected - n_celltype_activated
            FF = n_celltype_infected.sum() - n_celltype_activated.sum() - TF
            # do fisher exact test
            for cell_type in cell_types_to_use.unique():
                # do fisher exact test
                oddsratio, p = stats.fisher_exact([[int(FF.loc[cell_type]), int(FT.loc[cell_type])], [int(TF.loc[cell_type]), int(TT.loc[cell_type])]])
                p_value.loc[cell_type, cre] = p
                foldchange.loc[cell_type, cre] = oddsratio
            precision[cre] = TT / (TT + TF)
            recall[cre] = TT / (TT + FT)
            precision_n[cre] = TT + TF
            recall_n[cre] = TT + FT
            q_value[cre] = multitest.multipletests(p_value[cre], method='fdr_bh')[1]
        # make NaN values to 0 for precision and recall
        precision = precision.fillna(0)
        recall = recall.fillna(0)
        # for CRE in CRE_info, assign the p, q to best_subclass
        cre_info = self.get_creinfo().copy()
        for cre in cre_info.index:
            # get the best subclass
            best_subclass = cre_info.loc[cre, 'best_subclass']
            # get the p, q values for the best subclass
            if best_subclass in p_value.index:
                cre_info.loc[cre, 'p_value'] = p_value.loc[best_subclass, cre]
                cre_info.loc[cre, 'q_value'] = q_value.loc[best_subclass, cre]
                cre_info.loc[cre, 'precision'] = precision.loc[best_subclass, cre]
                cre_info.loc[cre, 'recall'] = recall.loc[best_subclass, cre]
                cre_info.loc[cre, 'foldchange'] = foldchange.loc[best_subclass, cre]
            # get the entropy for each cre
            cre_info.loc[cre, 'entropy'] = stats.entropy(recall[cre].astype(float))
        # save results to attribute
        fisher_exact_cre_test_result = {
            'cre_info': cre_info,
            'p_value': p_value,
            'q_value': q_value,
            'precision': precision,
            'recall': recall,
            'precision_n': precision_n,
            'recall_n': recall_n,
            'foldchange': foldchange
        }
        if not hasattr(self, 'fisher_exact_cre_test_results') or not hasattr(self, 'fisher_exact_cre_test_configs'):
            self.fisher_exact_cre_test_results = []
            self.fisher_exact_cre_test_configs = []
        self.fisher_exact_cre_test_results.append(fisher_exact_cre_test_result)
        self.fisher_exact_cre_test_configs.append(config)
        return fisher_exact_cre_test_result
    
    def atac_ontarget_cre_test(self, cell_types_to_use: List=None, activate_threshold=2, infect_threshold=1, atac_z_score_threshold=2) -> dict:
        config = {
            'cell_types_to_use': cell_types_to_use,
            'infect_threshold': infect_threshold,
            'activate_threshold': activate_threshold
        }
        # check if the results already exist
        if hasattr(self, 'atac_ontarget_cre_test_results') and hasattr(self, 'atac_ontarget_cre_test_configs'):
            for stored_config, atac_ontarget_cre_test_result in zip(self.atac_ontarget_cre_test_configs, self.atac_ontarget_cre_test_results):
                if stored_config == config:
                    # if the results already exist, return the results
                    print('Results already exist, return stored results')
                    return atac_ontarget_cre_test_result.copy()
        if cell_types_to_use is not None:
            # get the cell types
            cre_celltypes_expression, cell_types_to_use = self.get_cre_celltypes(cell_types_to_use)
        else:
            cre_celltypes_expression = self.get_cre_expression().copy()
            cell_types_to_use = self.get_celltypes()
        # only keep the cell types that are in the atac_cpm
        in_atac_cpm = cell_types_to_use.isin(self.atac_cpm.index)
        cre_celltypes_expression = cre_celltypes_expression[in_atac_cpm]
        cell_types_to_use = cell_types_to_use[in_atac_cpm]
        atac_cpm = self.atac_cpm.copy().loc[cell_types_to_use.unique()]
        # activate threshold array
        activate_threshold_array = self.estimate_activate_threshold_array(activate_threshold, cre_celltypes_expression, cell_types_to_use)
        # for each CRE in CRE_info, get the Target Cell Type
        cre_info = self.get_creinfo().copy()
        for cre in cre_info.index:
            if cre in atac_cpm.columns:
                # get the best subclass
                atac_z_score = np.log1p(atac_cpm[cre].copy().astype(float))
                atac_z_score = (atac_z_score - atac_z_score.mean()) / atac_z_score.std()
                target_celltypes = atac_z_score[atac_z_score >= atac_z_score_threshold].index
                # define infected cells for this CRE
                infected = cre_celltypes_expression[cre] >= infect_threshold
                infected_celltypes = cell_types_to_use[infected]
                # define activated cells for this CRE
                activated = cre_celltypes_expression[cre] >= activate_threshold_array
                activated_celltypes = cell_types_to_use[activated]
                # build contingency table
                # -------       | infected but not activated                                    | activated
                # Off target    | FF = sum(n_celltype_infected) - sum(n_celltype_activated) - TF| FT = sum(n_celltype_activated) - n_celltype_activated
                # On target     | TF = n_celltype_infected - n_celltype_activated               | TT = n_celltype_activated
                TT = activated_celltypes.isin(target_celltypes).sum()
                FT = activated_celltypes.shape[0] - TT
                TF = infected_celltypes.isin(target_celltypes).sum() - TT
                FF = infected_celltypes.shape[0] - FT - TF - TT
                # do fisher exact test
                oddsratio, p = stats.fisher_exact([[int(FF), int(FT)], [int(TF), int(TT)]])
                # assign to the CRE_info
                cre_info.loc[cre, 'p_value'] = p
                cre_info.loc[cre, 'foldchange'] = oddsratio
                cre_info.loc[cre, 'precision'] = TT / (TT + TF)
                cre_info.loc[cre, 'recall'] = TT / (TT + FT)
                cre_info.loc[cre, 'precision_n'] = TT + TF
                cre_info.loc[cre, 'recall_n'] = TT + FT
                # assign the target_celltypes
                cre_info.loc[cre, 'target_celltypes'] = ';'.join(target_celltypes)
                cre_info.loc[cre, 'target_celltype_number'] = len(target_celltypes)
        # fill NaN values
        cre_info['p_value'] = cre_info['p_value'].fillna(1)
        cre_info['foldchange'] = cre_info['foldchange'].fillna(0)
        cre_info['precision'] = cre_info['precision'].fillna(0)
        cre_info['recall'] = cre_info['recall'].fillna(0)
        cre_info['precision_n'] = cre_info['precision_n'].fillna(0)
        cre_info['recall_n'] = cre_info['recall_n'].fillna(0)
        cre_info['target_celltype_number'] = cre_info['target_celltype_number'].fillna(0)
        cre_info['target_celltypes'] = cre_info['target_celltypes'].fillna('')
        # do q value correction
        cre_info['q_value'] = multitest.multipletests(cre_info['p_value'], method='fdr_bh')[1]
        if not hasattr(self, 'atac_ontarget_cre_test_results') or not hasattr(self, 'atac_ontarget_cre_test_configs'):
            self.atac_ontarget_cre_test_results = []
            self.atac_ontarget_cre_test_configs = []
        self.atac_ontarget_cre_test_results.append(cre_info)
        self.atac_ontarget_cre_test_configs.append(config)
        return cre_info
    
    def fold_change_test(self, cell_types_to_use: List=None, 
                         normalize_by_cell_rna=False, normalize_by_cell_volume=False,
                         normalize_by_celltype_rna=False, normalize_by_celltype_volume=False,
                         normalize_by_negative_control=False, normalize_by_infected_cell=False,
                         normalize_by_libsize=False,
                         filter_zero_counts=False, log_transform=False,
                         rank_transform=None,
                         bootstrap_number=None, n_jobs=256) -> dict:
        config = {
            'cell_types_to_use': cell_types_to_use,
            'normalize_by_cell_rna': normalize_by_cell_rna,
            'normalize_by_cell_volume': normalize_by_cell_volume,
            'normalize_by_celltype_rna': normalize_by_celltype_rna,
            'normalize_by_celltype_volume': normalize_by_celltype_volume,
            'normalize_by_negative_control': normalize_by_negative_control,
            'normalize_by_infected_cell': normalize_by_infected_cell,
            'normalize_by_libsize': normalize_by_libsize,
            "filter_zero_counts": filter_zero_counts,
            'log_transform': log_transform,
            'rank_transform': rank_transform,
            'bootstrap_number': bootstrap_number}
        # check if the results already exist
        partial_loaded = False
        fold_change_test_result = None
        if hasattr(self, 'fold_change_test_results') and hasattr(self, 'fold_change_test_configs'):
            for stored_config, stored_result in zip(self.fold_change_test_configs, self.fold_change_test_results):
                # only partially check the config, everything the same except bootstrap_number
                if all(stored_config[k] == config[k] for k in config if k != 'bootstrap_number'):
                    # if the results already exist, return the results
                    fold_change_test_result = stored_result.copy()
                    if stored_config['bootstrap_number'] == config['bootstrap_number'] or config['bootstrap_number'] is None:
                        print('Results already exist, return stored results')
                        return fold_change_test_result
        if fold_change_test_result is not None:
            partial_loaded = True
        if cell_types_to_use is not None:
            # get the cell types
            cre_celltypes_expression, rna_celltypes_expression, cell_types_to_use = self.get_cre_rna_celltypes(cell_types_to_use)
        else:
            cre_celltypes_expression = self.get_cre_expression().copy()
            rna_celltypes_expression = self.get_rna_expression().copy()
            cell_types_to_use = self.get_celltypes()
        volm = self.get_tag('obs:volm').copy()
        volm = volm.loc[cell_types_to_use.index]
        # transform rna_celltypes_expression to dataframe
        rna_celltypes_expression = pd.DataFrame(rna_celltypes_expression, index=cell_types_to_use.index)
        if normalize_by_cell_rna and normalize_by_cell_volume:
            rna_per_volume = rna_celltypes_expression / volm.values.reshape(-1, 1)
            cre_celltypes_expression = cre_celltypes_expression / rna_per_volume.mean(axis=1).values.reshape(-1, 1)
        elif normalize_by_cell_rna and not normalize_by_cell_volume:
            cre_celltypes_expression = cre_celltypes_expression / rna_celltypes_expression.mean(axis=1).values.reshape(-1, 1)
        elif normalize_by_cell_volume and not normalize_by_cell_rna:
            cre_celltypes_expression = cre_celltypes_expression / volm.values.reshape(-1, 1)
        if log_transform:
            cre_celltypes_expression = np.log1p(cre_celltypes_expression)
        cre_info = self.get_creinfo().copy()
        if not partial_loaded:
            foldchange, celltype_activity = calculate_fold_change(
                cre_celltypes_expression, cell_types_to_use.to_numpy(), cre_info, 
                rna_celltypes_expression, volm,
                normalize_by_celltype_rna, normalize_by_celltype_volume,
                normalize_by_negative_control, np.log1p(self.lib_size['counts'] + 1),
                normalize_by_infected_cell, normalize_by_libsize, filter_zero_counts,
                rank_transform,
            )
            # get the foldchange for the CREs
            for cre in cre_info.index:
                # get the best subclass
                best_subclass = cre_info.loc[cre, 'best_subclass']
                # get the foldchange for the best subclass
                if best_subclass in foldchange.index:
                    cre_info.loc[cre, 'foldchange'] = foldchange.loc[best_subclass, cre]
                    cre_info.loc[cre, 'celltype_activity'] = celltype_activity.loc[best_subclass, cre]
            fold_change_test_result = {
                'cre_info': cre_info,
                'foldchange': foldchange,
                'celltype_activity': celltype_activity,
            }
        # do bootstrap, if bootstrap_number is not None
        if bootstrap_number is not None:
            foldchange = fold_change_test_result['foldchange']
            celltype_activity = fold_change_test_result['celltype_activity']
            with multiprocessing.Pool(processes=min(n_jobs, int(multiprocessing.cpu_count()*0.8))) as pool:
                bootstrap_results = pool.starmap(
                    calculate_fold_change, 
                    [(cre_celltypes_expression, cell_types_to_use.sample(frac=1, replace=False, random_state=i).to_numpy(), cre_info, 
                      rna_celltypes_expression, volm,
                      normalize_by_celltype_rna, normalize_by_celltype_volume,
                      normalize_by_negative_control, np.log1p(self.lib_size['counts'] + 1),
                      normalize_by_infected_cell, normalize_by_libsize, filter_zero_counts, rank_transform) 
                     for i in range(bootstrap_number)]
                )
            foldchange_array = np.ndarray((bootstrap_number, foldchange.shape[0], foldchange.shape[1]))
            activity_array = np.ndarray((bootstrap_number, celltype_activity.shape[0], celltype_activity.shape[1]))
            for i, (fc, act) in enumerate(bootstrap_results):
                foldchange_array[i] = fc
                activity_array[i] = act
            # calculate p-value for foldchange and celltype_activity
            pvalue = pd.DataFrame(index=foldchange.index, columns=foldchange.columns)
            pvalue_activity = pd.DataFrame(index=celltype_activity.index, columns=celltype_activity.columns)
            # if we use normalize_by_negative_control, fill the nan with 0
            if normalize_by_negative_control:
                foldchange_array[np.isnan(foldchange_array)] = 0
                activity_array[np.isnan(activity_array)] = 0
            for i in range(foldchange.shape[0]):
                for j in range(foldchange.shape[1]):
                    pvalue.iloc[i, j] = np.sum(foldchange_array[:, i, j] >= foldchange.iloc[i, j]) / bootstrap_number
                    pvalue_activity.iloc[i, j] = np.sum(activity_array[:, i, j] >= celltype_activity.iloc[i, j]) / bootstrap_number
            qvalue = pvalue.copy()
            qvalue_activity = pvalue_activity.copy()
            for i in range(pvalue.shape[1]):
                qvalue.iloc[:, i] = multitest.multipletests(pvalue.iloc[:, i], method='fdr_bh')[1]
                qvalue_activity.iloc[:, i] = multitest.multipletests(pvalue_activity.iloc[:, i], method='fdr_bh')[1]
            # add pvalue and qvalue to cre_info
            for cre in fold_change_test_result['cre_info'].index:
                # get the best subclass
                best_subclass = fold_change_test_result['cre_info'].loc[cre, 'best_subclass']
                # get the pvalue and qvalue for the best subclass
                if best_subclass in pvalue.index:
                    fold_change_test_result['cre_info'].loc[cre, 'pvalue'] = pvalue.loc[best_subclass, cre]
                    fold_change_test_result['cre_info'].loc[cre, 'qvalue'] = qvalue.loc[best_subclass, cre]
                    fold_change_test_result['cre_info'].loc[cre, 'pvalue_activity'] = pvalue_activity.loc[best_subclass, cre]
                    fold_change_test_result['cre_info'].loc[cre, 'qvalue_activity'] = qvalue_activity.loc[best_subclass, cre]
            fold_change_test_result['pvalue'] = pvalue
            fold_change_test_result['qvalue'] = qvalue
            fold_change_test_result['foldchange_array'] = foldchange_array
            fold_change_test_result['activity_array'] = activity_array
            fold_change_test_result['pvalue_activity'] = pvalue_activity
            fold_change_test_result['qvalue_activity'] = qvalue_activity
        # save results to attribute
        if not hasattr(self, 'fold_change_test_results') or not hasattr(self, 'fold_change_test_configs'):
            self.fold_change_test_results = []
            self.fold_change_test_configs = []
        self.fold_change_test_results.append(fold_change_test_result)
        self.fold_change_test_configs.append(config)
        return fold_change_test_result
    
    def neg_control_regression_test(self, cell_types_to_use: List=None, 
                                    negative_control=None,
                                    normalize_by_cell_rna=False, normalize_by_cell_volume=False,
                                    normalize_by_celltype_rna=False, normalize_by_celltype_volume=False,
                                    log_transform=False) -> dict:
        config = {
            'cell_types_to_use': cell_types_to_use,
            'negative_control': negative_control,
            'normalize_by_cell_rna': normalize_by_cell_rna,
            'normalize_by_cell_volume': normalize_by_cell_volume,
            'normalize_by_celltype_rna': normalize_by_celltype_rna,
            'normalize_by_celltype_volume': normalize_by_celltype_volume,
            'log_transform': log_transform
        }
        # check if the results already exist
        neg_control_regression_test_result = None
        if hasattr(self, 'neg_control_regression_test_results') and hasattr(self, 'neg_control_regression_test_configs'):
            for stored_config, stored_result in zip(self.neg_control_regression_test_configs, self.neg_control_regression_test_results):
                # only partially check the config, everything the same except bootstrap_number
                if all(stored_config[k] == config[k] for k in config):
                    # if the results already exist, return the results
                    neg_control_regression_test_result = stored_result.copy()
                    return neg_control_regression_test_result
        if cell_types_to_use is not None:
            # get the cell types
            cre_celltypes_expression, rna_celltypes_expression, cell_types_to_use = self.get_cre_rna_celltypes(cell_types_to_use)
        else:
            cre_celltypes_expression = self.get_cre_expression().copy()
            rna_celltypes_expression = self.get_rna_expression().copy()
            cell_types_to_use = self.get_celltypes()
        volm = self.get_tag('obs:volm').copy()
        volm = volm.loc[cell_types_to_use.index]
        # transform rna_celltypes_expression to dataframe
        rna_celltypes_expression = pd.DataFrame(rna_celltypes_expression, index=cell_types_to_use.index)
        if normalize_by_cell_rna and normalize_by_cell_volume:
            rna_per_volume = rna_celltypes_expression / volm.values.reshape(-1, 1)
            cre_celltypes_expression = cre_celltypes_expression / rna_per_volume.mean(axis=1).values.reshape(-1, 1)
        elif normalize_by_cell_rna and not normalize_by_cell_volume:
            cre_celltypes_expression = cre_celltypes_expression / rna_celltypes_expression.mean(axis=1).values.reshape(-1, 1)
        elif normalize_by_cell_volume and not normalize_by_cell_rna:
            cre_celltypes_expression = cre_celltypes_expression / volm.values.reshape(-1, 1)
        # get negative controls
        if negative_control is None:
            negative_control = self.get_negative_control_cres().tolist()
        else:
            negative_control = pd.Series(negative_control).tolist()
        # aggregate to bulk
        cre_celltypes_bulk_expression = cre_celltypes_expression.groupby(cell_types_to_use).sum()
        rna_celltypes_bulk_expression = rna_celltypes_expression.groupby(cell_types_to_use).sum()
        celltypes_volm_bulk = volm.groupby(cell_types_to_use).sum()
        if normalize_by_celltype_rna and normalize_by_celltype_volume:
            rna_per_volume = rna_celltypes_bulk_expression / celltypes_volm_bulk.values.reshape(-1, 1)
            cre_celltypes_bulk_expression = cre_celltypes_bulk_expression / rna_per_volume.mean(axis=1).values.reshape(-1, 1)
        elif normalize_by_celltype_rna and not normalize_by_celltype_volume:
            cre_celltypes_bulk_expression = cre_celltypes_bulk_expression / rna_celltypes_bulk_expression.mean(axis=1).values.reshape(-1, 1)
        elif normalize_by_celltype_volume and not normalize_by_celltype_rna:
            cre_celltypes_bulk_expression = cre_celltypes_bulk_expression / celltypes_volm_bulk.values.reshape(-1, 1)
        # prepare lib size
        lib_size = self.lib_size['counts'].copy()
        # if 'sum' in negative_control, then prepare a lib size for sum negative control
        if 'sum' in negative_control:
            tmp = [cre for cre in negative_control if cre != 'sum']
            lib_size.loc['sum'] = lib_size.loc[tmp].sum()
            cre_celltypes_bulk_expression['sum'] = cre_celltypes_bulk_expression[tmp].sum(axis=1)
        # apply log transform
        if log_transform:
            cre_celltypes_bulk_expression = np.log1p(cre_celltypes_bulk_expression)
        # do for each cell type
        model_stats = pd.DataFrame(index=cre_celltypes_bulk_expression.index, columns=['slope', 'intercept', 'r_squared'])
        fold_change = pd.DataFrame(index=cre_celltypes_bulk_expression.index, columns=cre_celltypes_bulk_expression.columns)
        p_values = pd.DataFrame(index=cre_celltypes_bulk_expression.index, columns=cre_celltypes_bulk_expression.columns)
        q_values = pd.DataFrame(index=cre_celltypes_bulk_expression.index, columns=cre_celltypes_bulk_expression.columns)
        for cell_type in cre_celltypes_bulk_expression.index:
            # get the cre expression for the cell type
            cre_data = cre_celltypes_bulk_expression.loc[cell_type]
            # get the negative control expression for the cell type
            neg_control_data = cre_celltypes_bulk_expression.loc[cell_type, negative_control]
            # fit a linear regression model
            X_train = lib_size.loc[negative_control].values.reshape(-1, 1)
            y_train = neg_control_data.values
            # if y_train all equal, skip
            if np.all(y_train == y_train[0]):
                continue
            model = sm.OLS(y_train, X_train).fit()
            # predict for all cres
            X_test = lib_size.values.reshape(-1, 1)
            y_pred = model.get_prediction(X_test)
            summary = y_pred.summary_frame()
            predicted = summary['mean'].values
            std_dev = summary['mean_se'].values
            # calculate the fold change
            fc = cre_data.values / predicted
            # calculate the p-value
            t_stats = (cre_data.values - predicted) / std_dev
            dof = model.df_resid  # Degrees of freedom
            p = 2 * (1 - stats.t.cdf(np.abs(t_stats), dof))
            # save the results
            model_stats.loc[cell_type, 'slope'] = model.params[0]
            model_stats.loc[cell_type, 'r_squared'] = model.rsquared
            fold_change.loc[cell_type] = fc
            p[np.isnan(p)] = 1
            p_values.loc[cell_type] = p
            q_values.loc[cell_type] = multitest.multipletests(p, method='fdr_bh')[1]
        neg_control_regression_test_result = {
            'model_stats': model_stats,
            'fold_change': fold_change,
            'p_values': p_values,
            'q_values': q_values
        }
        # save results to attribute
        if not hasattr(self, 'neg_control_regression_test_results') or not hasattr(self, 'neg_control_regression_test_configs'):
            self.neg_control_regression_test_results = []
            self.neg_control_regression_test_configs = []
        self.neg_control_regression_test_results.append(neg_control_regression_test_result)
        self.neg_control_regression_test_configs.append(config)
        return neg_control_regression_test_result

    def mixture_model_test(self, cell_types_to_use: List=None,
                           normalize_by_cell_rna=False, normalize_by_cell_volume=False, log_transform=False,
                           model='STARR_FISH_MIXTURE_NB.stan'):
        config = {
            'cell_types_to_use': cell_types_to_use,
            'normalize_by_cell_rna': normalize_by_cell_rna,
            'normalize_by_cell_volume': normalize_by_cell_volume,
            'log_transform': log_transform,
            'model': model
        }
        # check if the results already exist
        if hasattr(self, 'mixture_model_test_results') and hasattr(self, 'mixture_model_test_configs'):
            for stored_config, mixture_model_test_result in zip(self.mixture_model_test_configs, self.mixture_model_test_results):
                if stored_config == config:
                    # if the results already exist, return the results
                    print('Results already exist, return stored results')
                    return mixture_model_test_result.copy()
        if cell_types_to_use is not None:
            # get the cell types
            cre_celltypes_expression, rna_celltypes_expression, cell_types_to_use = self.get_cre_rna_celltypes(cell_types_to_use)
        else:
            cre_celltypes_expression = self.get_cre_expression().copy()
            rna_celltypes_expression = self.get_rna_expression().copy()
            cell_types_to_use = self.get_celltypes()
        volm = self.get_tag('obs:volm').copy()
        volm = volm.loc[cell_types_to_use.index]
        # transform rna_celltypes_expression to dataframe
        rna_celltypes_expression = pd.DataFrame(rna_celltypes_expression, index=cell_types_to_use.index)
        cre_celltypes_expression_orig = cre_celltypes_expression.copy()
        if normalize_by_cell_rna and normalize_by_cell_volume:
            rna_per_volume = rna_celltypes_expression / volm.values.reshape(-1, 1)
            cre_celltypes_expression = cre_celltypes_expression / rna_per_volume.mean(axis=1).values.reshape(-1, 1)
        elif normalize_by_cell_rna and not normalize_by_cell_volume:
            cre_celltypes_expression = cre_celltypes_expression / rna_celltypes_expression.mean(axis=1).values.reshape(-1, 1)
        elif normalize_by_cell_volume and not normalize_by_cell_rna:
            cre_celltypes_expression = cre_celltypes_expression / volm.values.reshape(-1, 1)
        if log_transform:
            cre_celltypes_expression = np.log1p(cre_celltypes_expression)
        # check model to use
        activity_df = pd.DataFrame(index=cell_types_to_use.unique(), columns=cre_celltypes_expression.columns)
        if model == f'STARR_FISH_MIXTURE_NB.stan':
            # needs to filter out zeros
            # assert no normalization and log transform, because the model is not designed for normalized data
            assert not normalize_by_cell_rna and not normalize_by_cell_volume and not log_transform, \
                'The model is not designed for normalized data, please use raw data'
            # load model
            model = CmdStanModel(stan_file=f'{PWD}/stan_model/{model}')
            # set up result dict
            fit_results = {}
            # do multiprocessing
            with multiprocessing.Pool(processes=int(multiprocessing.cpu_count()*0.8)) as pool:
                    # fit for each cell type
                for cell_type in cell_types_to_use.unique():
                    start = time.time()
                    print("Begin fitting for cell type:", cell_type)
                    # get the cre expression for the cell type
                    cre_data = cre_celltypes_expression[cell_types_to_use == cell_type]
                    # first apply to all CREs
                    x = cre_data.values.astype(int).flatten()
                    # save the results
                    fit_results[f'{cell_type}_ALL'] = fit_stan(x, model)
                    # fit for each CRE
                    tmp_results = pool.starmap(fit_stan, [(cre_data[i].values.astype(int).flatten(), model, 1) for i in cre_data.columns])
                    for i, cre in enumerate(cre_data.columns):
                        # save the results
                        fit_results[f'{cell_type}_{cre}'] = tmp_results[i]
                        # get the activity for the cell type and cre
                        activity_df.loc[cell_type, cre] = fit_results[f'{cell_type}_{cre}'].loc['gamma_mean', 'Mean']
                    print("Finished in ", time.time() - start, "seconds")
        elif model == 'sklearn_gaussian_mixture':
            # use sklearn gaussian mixture model
            fit_results = {}
            # fit for each CRE, do parallel
            for cell_type in cell_types_to_use.unique():
                start = time.time()
                print("Begin fitting for cell type:", cell_type)
                # get the cre expression for the cell type
                cre_data = cre_celltypes_expression[cell_types_to_use == cell_type]
                cre_data_orig = cre_celltypes_expression_orig[cell_types_to_use == cell_type]
                # first apply to all CREs
                x = cre_data.values.flatten()
                x_orig = cre_data_orig.values.flatten()
                # append results
                fit_results[f'{cell_type}_ALL'] = fit_sklearn_gauss_mixture(x, x_orig)
                # fit for each CRE
                # tmp_results = pool.starmap(fit_sklearn_gauss_mixture, [(cre_data[i].values.flatten(), cre_data_orig[i].values.flatten()) for i in cre_data.columns])
                for i, cre in enumerate(cre_data.columns):
                    # append results
                    fit_results[f'{cell_type}_{cre}'] = fit_sklearn_gauss_mixture(cre_data[cre].values.flatten(), cre_data_orig[cre].values.flatten())
                    # get the activity for the cell type and cre
                    activity_df.loc[cell_type, cre] = fit_results[f'{cell_type}_{cre}']['means'][1][0]
                print("Finished in ", time.time() - start, "seconds")
        else:
            raise NotImplementedError(f'Model {model} is not implemented')
        # save the results to the attribute
        res = {
            'fit_results': fit_results,
            'activity_df': activity_df
        }
        if not hasattr(self, 'mixture_model_test_results') or not hasattr(self, 'mixture_model_test_configs'):
            self.mixture_model_test_results = []
            self.mixture_model_test_configs = []
        self.mixture_model_test_results.append(res)
        self.mixture_model_test_configs.append(config)
        return res
    
    def glm_test(self, cell_types_to_use: List=None, norm_by_volm=False, volm_covariate=False, fov_covariate=False, 
                 filter_infected_cells=True, only_keep_positive_per_cre=True, multiprocess_threads=256) -> dict:
        config = {
            'norm_by_volm': norm_by_volm,
            'volm_covariate': volm_covariate,
            'fov_covariate': fov_covariate,
            'filter_infected_cells': filter_infected_cells,
            'only_keep_positive_per_cre': only_keep_positive_per_cre
        }
        # if the results already exist, return the results
        if hasattr(self, 'glm_results') and hasattr(self, 'glm_configs'):
            for stored_config, glm_result in zip(self.glm_configs, self.glm_results):
                if stored_config == config:
                    # if the results already exist, return the results
                    print('Results already exist, return stored results')
                    return glm_result.copy()
        result = glm(self.adata, cell_types_to_use=cell_types_to_use, norm_by_volm=norm_by_volm, 
                     volm_covariate=volm_covariate, fov_covariate=fov_covariate, 
                     filter_infected_cells=filter_infected_cells, 
                     only_keep_positive_per_cre=only_keep_positive_per_cre,
                     multiprocess_threads=multiprocess_threads)
        # add results to attribute
        if not hasattr(self, 'glm_test_results') or not hasattr(self, 'glm_test_configs'):
            self.glm_test_results = []
            self.glm_test_configs = []
        self.glm_test_results.append(result)
        self.glm_test_configs.append(config)
        return result

    def pseudo_bulk_glm_test(self, cell_types_to_use: List=None, norm_by_volm=False, volm_covariate=False,
                             filter_infected_cells=True, only_keep_positive_per_cre=False, 
                             pseudo_bulk_size=50, pseudo_bulk_percentage=None, 
                             pseudo_bulk_number=1000, replace=True, multiprocess_threads=256) -> dict:
        # check if the results already exist
        config = {'cell_types_to_use': cell_types_to_use, 
                  'norm_by_volm': norm_by_volm, 
                  'volm_covariate': volm_covariate,
                  'filter_infected_cells': filter_infected_cells,
                  'only_keep_positive_per_cre': only_keep_positive_per_cre,
                  'pseudo_bulk_size': pseudo_bulk_size,
                  'pseudo_bulk_percentage': pseudo_bulk_percentage,
                  'pseudo_bulk_number': pseudo_bulk_number,
                  'replace': replace}
        if hasattr(self, 'pseudo_bulk_glm_test_results') and hasattr(self, 'pseudo_bulk_glm_test_configs'):
            for stored_config, pseudo_bulk_glm_test_result in zip(self.pseudo_bulk_glm_test_configs, self.pseudo_bulk_glm_test_results):
                if stored_config == config:
                    # if the results already exist, return the results
                    print('Results already exist, return stored results')
                    return pseudo_bulk_glm_test_result.copy()
        # for each cell type to use, create a pseudo bulk
        celltypes = self.get_celltypes()
        cre_expression = self.get_cre_expression()
        volumes = self.get_tag('obs:volm')
        if filter_infected_cells:
            # get the infected cells
            infected_cells = ((cre_expression >= 1).sum(axis=1) > 0)
            celltypes = celltypes[infected_cells]
            cre_expression = cre_expression[infected_cells]
            volumes = volumes[infected_cells]
        if cell_types_to_use is None:
            cell_types_to_use = celltypes.unique()
        # get the cell type cell counts
        cell_counts = celltypes.value_counts().loc[cell_types_to_use]
        # filter out cell types with insufficient cell counts if not replace
        if not replace:
            if pseudo_bulk_size is not None:
                cell_counts = cell_counts[cell_counts >= pseudo_bulk_size]
            else:
                cell_counts = cell_counts[cell_counts >= 50 / pseudo_bulk_percentage]
        # redefine the cell types to use
        cell_types_to_use = cell_counts.index.tolist()
        # generate pseudo bulk for each cell type
        pseudo_bulk = pd.DataFrame()
        pseudo_bulk_obs = pd.DataFrame()
        for cell_type in cell_types_to_use:
            # get the cre expression for the cell type cells
            cre_expression_cell_type = cre_expression[celltypes == cell_type]
            volume_cell_type = volumes[celltypes == cell_type]
            if pseudo_bulk_size is None:
                assert pseudo_bulk_percentage is not None, 'pseudo_bulk_size or pseudo_bulk_percentage must be set'
                sample_size = int(cell_counts.loc[cell_type] * pseudo_bulk_percentage)
            else:
                sample_size = pseudo_bulk_size
            # get the pseudo bulk for the cell type
            bootstrap_indices = np.concat([np.random.default_rng(seed=i).choice(cre_expression_cell_type.shape[0], size=(1, sample_size), replace=replace) 
                                           for i in range(pseudo_bulk_number)], axis=0)
            samples = cre_expression_cell_type.values[bootstrap_indices].sum(axis=1)
            samples = pd.DataFrame(samples, index=[cell_type + ':sample_' + str(i) for i in range(pseudo_bulk_number)], columns=cre_expression_cell_type.columns)
            # add the pseudo bulk to the pseudo bulk dataframe
            pseudo_bulk = pd.concat([pseudo_bulk, samples])
            # add volume to the observation
            sample_volumes = volume_cell_type.values[bootstrap_indices].sum(axis=1)
            sample_obs = pd.DataFrame(sample_volumes, index=[cell_type + ':sample_' + str(i) for i in range(pseudo_bulk_number)], columns=['volm'])
            sample_obs['subclass'] = cell_type
            sample_obs['fov'] = cell_type
            # add the pseudo bulk obs to the pseudo bulk obs dataframe
            pseudo_bulk_obs = pd.concat([pseudo_bulk_obs, sample_obs])
        # create a new AnnData object for the pseudo bulk
        pseudo_bulk_adata = sc.AnnData(pseudo_bulk, obs=pseudo_bulk_obs)
        pseudo_bulk_adata.obsm['X_raw'] = pseudo_bulk
        pseudo_bulk_adata.obsm['CRE'] = pseudo_bulk
        # perform glm test on the pseudo bulk
        result = glm(pseudo_bulk_adata, cell_types_to_use=cell_types_to_use, CREs=pseudo_bulk.columns,
                     norm_by_volm=norm_by_volm, volm_covariate=volm_covariate, 
                     fov_covariate=False, filter_infected_cells=False, 
                     only_keep_positive_per_cre=only_keep_positive_per_cre,
                     multiprocess_threads=multiprocess_threads)
        pseudo_bulk_glm_test_result = {'pseudo_bulk_adata': pseudo_bulk_adata,
                                       'result': result}
        # add results to attribute
        if not hasattr(self, 'pseudo_bulk_glm_test_results') or not hasattr(self, 'pseudo_bulk_glm_test_configs'):
            self.pseudo_bulk_glm_test_results = []
            self.pseudo_bulk_glm_test_configs = []
        self.pseudo_bulk_glm_test_results.append(pseudo_bulk_glm_test_result)
        self.pseudo_bulk_glm_test_configs.append(config)
        return pseudo_bulk_glm_test_result
    
    def scvi(self, use_model: Literal['STARRFISHVI', 'SCVI'] = 'STARRFISHVI', model_args: dict = None, train_args: dict = None) -> dict:
        # use scvi to denoise the data
        if use_model == 'STARRFISHVI':
            SCVIMODEL = STARRFISHVI
        elif use_model == 'SCVI':
            SCVIMODEL = scvi.model.SCVI
        # infer infection rate prior
        non_infected_cells = ((self.adata.obsm['CRE'] > 0).sum(axis=1) == 0).sum()
        infection_rate = -np.log(non_infected_cells / self.adata.shape[0]).item()
        # first set the default arguments
        if model_args is None:
            # set default model args
            model_args = {'n_latent': 10, 
                          'n_hidden': 128}
            if use_model == 'STARRFISHVI':
                model_args.update({
                    'gene_likelihood': "nb",
                    'infection_rate_inference': 'encoder',
                    'infection_rate_generative': 'sample',
                    'accessibility_generative': 'split',
                    'infection_rate_type': 'gene',
                    'kl_infection_rate_type': "gene-cosine",
                    'infection_rate_library_size': self.lib_size.values,
                    'infection_rate_prior': infection_rate
                })
            elif use_model == 'SCVI':
                model_args.update({'n_layers': 2})
        if train_args is None:
            # set default train args
            train_args = {
                'max_epochs': 500,
                'batch_size': 1280,
                'accelerator': 'auto'
            }
            # if gpu is available, use gpu
            if torch.cuda.is_available():
                train_args['accelerator'] = 'gpu'
                train_args['devices'] = 1
        # create a global config to save
        config = {'use_model': use_model,
                  'model_args': model_args.copy(),
                  'train_args': train_args.copy()}
        # drop config['train_args']['accelerator'] from config only
        if 'accelerator' in config['train_args']:
            config['train_args'].pop('accelerator')
        if 'devices' in config['train_args']:
            config['train_args'].pop('devices')
        if 'infection_rate_library_size' in config['model_args']:
            config['model_args'].pop('infection_rate_library_size')
        # check if the results already exist
        if hasattr(self, 'scvi_results') and hasattr(self, 'scvi_configs'):
            for stored_config, scvi_result in zip(self.scvi_configs, self.scvi_results):
                if stored_config == config:
                    # if the results already exist, return the results
                    print('Results already exist, return stored results')
                    return scvi_result.copy()
        # prepare adata_mvi for scvi
        gene_info = self.adata.var.copy()
        gene_info['modality'] = 'Gene Expression'
        CRE_info = self.adata.uns['CRE_info'].copy()
        CRE_info['modality'] = 'Peaks'
        paired_info = pd.concat([gene_info, CRE_info], axis=0)
        paired_info['ID'] = paired_info.index
        adata_paired = sc.AnnData(
            X=np.concatenate([self.adata.obsm['X_raw'], self.adata.obsm['CRE']], axis=1),
            obs=self.adata.obs.copy(),
            var=paired_info
        )
        adata_mvi = scvi.data.organize_multiome_anndatas(adata_paired)
        adata_mvi = adata_mvi[:, adata_mvi.var["modality"].argsort()].copy()
        # set up model
        SCVIMODEL.setup_anndata(adata_mvi, batch_key="modality")
        if use_model == 'STARRFISHVI':
            model_args.update({
                'n_genes': (adata_mvi.var["modality"] == "Gene Expression").sum(),
                'n_regions': (adata_mvi.var["modality"] == "Peaks").sum(),
            })
            # rematch the libsize index
            model_args['infection_rate_library_size'] = self.lib_size.reindex(adata_mvi.var[adata_mvi.var['modality'] == 'Peaks'].index, fill_value=0).values
        model = SCVIMODEL(
            adata_mvi,
            **model_args
        )
        model.train(**train_args)
        if use_model == 'STARRFISHVI':
            access = model.get_accessibility_estimates()
            infect_rate = model.get_infection_rate_estimate()
            lib_size_estimate = model.get_library_size_factors()['accessibility']
            adata_mvi.obsm['activity'] = access
            adata_mvi.obsm['infect_rate'] = infect_rate
            adata_mvi.obsm['lib_size'] = lib_size_estimate
            adata_mvi.uns['infection_rate_gene'] = torch.nn.functional.softplus(model.module.infection_rate_gene).detach().cpu().numpy()
        elif use_model == 'SCVI':
            access = model.get_normalized_expression()
            adata_mvi.obsm['activity'] = access
        # save to attribute
        if not hasattr(self, 'scvi_results') or not hasattr(self, 'scvi_configs'):
            self.scvi_results = []
            self.scvi_configs = []
        self.scvi_results.append(adata_mvi)
        self.scvi_configs.append(config)
        return adata_mvi

    def corr_atac_cpm(self, cell_types_to_use: Union[List, pd.Series]=None, cres_to_use: Union[List, pd.Series]=None,
                      acvitity_df: pd.DataFrame = None, log_atac=False, log_activity=False, 
                      filter_by_atac_z_threshold=None,
                      filter_by_atac_raw_threshold=None,
                      filter_by_negative_control_z_threshold=None) -> tuple[pd.DataFrame, pd.DataFrame]:
        # filter atac_cpm and activity_df by cell_types_to_use
        if cell_types_to_use is not None:
            # first transform the cell_types_to_use as np.array
            cell_types_to_use = pd.Series(cell_types_to_use)
            cell_types_to_use = cell_types_to_use[cell_types_to_use.isin(self.atac_cpm.index)]
            cell_types_to_use = cell_types_to_use[cell_types_to_use.isin(acvitity_df.index)]
        else:
            cell_types_to_use = self.atac_cpm.index
            cell_types_to_use = cell_types_to_use[cell_types_to_use.isin(acvitity_df.index)]
        atac_cpm = self.atac_cpm.loc[cell_types_to_use]
        activity_df = acvitity_df.loc[cell_types_to_use]
        # match the index
        if cres_to_use is not None:
            cres_to_use = pd.Series(cres_to_use)
            cres_to_use = cres_to_use[cres_to_use.isin(atac_cpm.columns)]
            cres_to_use = cres_to_use[cres_to_use.isin(activity_df.columns)]
        else:
            cres_to_use = atac_cpm.columns.intersection(activity_df.columns)
        atac_cpm = atac_cpm[cres_to_use]
        activity_df = activity_df[cres_to_use]
        # do log transform
        if log_atac:
            atac_cpm = np.log(atac_cpm.astype(float) + 1)
        if log_activity:
            activity_df = np.log10(activity_df.astype(float) + 1)
        # for each cell type, calculate the mean and std of negative control CREs
        if filter_by_negative_control_z_threshold is not None:
            negative_control_cres = self.get_negative_control_cres()
            # get the 6 lowest negative control activities
            negative_control_cres = negative_control_cres[negative_control_cres.isin(activity_df.columns)]
            negative_control_cres_mean = activity_df[negative_control_cres].mean(axis=1)
            negative_control_cres_std = activity_df[negative_control_cres].std(axis=1)
            # for each cell type, filter the activity by negative control mean + 3 std
            negative_control_cres_threshold = negative_control_cres_mean + filter_by_negative_control_z_threshold * negative_control_cres_std
            for cell_type in activity_df.index:
                # filter the activity by negative control mean + 3 std
                activity_df.loc[cell_type][activity_df.loc[cell_type] < negative_control_cres_threshold[cell_type]] = np.nan
        if filter_by_atac_z_threshold is not None:
            atac_cpm_mean = atac_cpm.mean(axis=1)
            atac_cpm_std = atac_cpm.std(axis=1)
            # for each cell type, filter the activity by atac mean + 3 std
            atac_cpm_threshold = atac_cpm_mean + filter_by_atac_z_threshold * atac_cpm_std
            for cell_type in atac_cpm.index:
                # filter the activity by atac mean + 3 std
                atac_cpm.loc[cell_type][atac_cpm.loc[cell_type] < atac_cpm_threshold[cell_type]] = np.nan
        if filter_by_atac_raw_threshold is not None:
            for cell_type in atac_cpm.index:
                # filter the activity by atac mean + 3 std
                atac_count = self.atac_counts.loc[cell_types_to_use]
                atac_count = atac_count[cres_to_use]
                atac_cpm.loc[cell_type][atac_count.loc[cell_type] < filter_by_atac_raw_threshold] = np.nan
        # calculate the correlation for each cre
        # first do col wise correlation
        col_result = col_corr(atac_cpm, activity_df)
        # do row wise correlation
        row_result = row_corr(atac_cpm, activity_df)
        return col_result, row_result
    
    def cross_talk_test(self, cell_types_to_use: List=None, normalize_by_cell_rna=False, normalize_by_volume=False,
                        method: Literal['pearson', 'spearman', 'fisher_exact'] = 'fisher_exact',
                        n_jobs=256) -> dict:
        config = {
            'cell_types_to_use': cell_types_to_use,
            'normalize_by_cell_rna': normalize_by_cell_rna,
            'normalize_by_volume': normalize_by_volume,
            'method': method
        }
        # check if the results already exist
        if hasattr(self, 'cross_talk_test_results') and hasattr(self, 'cross_talk_test_configs'):
            for stored_config, cross_talk_test_result in zip(self.cross_talk_test_configs, self.cross_talk_test_results):
                if stored_config == config:
                    # if the results already exist, return the results
                    print('Results already exist, return stored results')
                    return cross_talk_test_result.copy()
        # for each cell type, calculate the correlation between CREs
        # first only keep cell types in cell_types_to_use
        if cell_types_to_use is not None and cell_types_to_use != ['ALL']:
            # get the cell types
            cre_celltypes_expression, cell_types_to_use = self.get_cre_celltypes(cell_types_to_use)
        else:
            cre_celltypes_expression = self.get_cre_expression().copy()
            if cell_types_to_use != ['ALL']:
                cell_types_to_use = self.get_celltypes()
        if method == 'fisher_exact':
            infect_threshold = 1
            activate_threshold = 2
            # infected = ((cre_celltypes_expression >= infect_threshold).sum(axis=1) > 0)
            # cre_celltypes_expression = cre_celltypes_expression[infected]
            # cell_types_to_use = cell_types_to_use[infected]
            activated = cre_celltypes_expression >= activate_threshold
            if cell_types_to_use == ['ALL']:
                p_value = None
            else:
                # p value is (cell types to use) x (cre1, cre2)
                with multiprocessing.Pool(processes=min(n_jobs, int(multiprocessing.cpu_count()*0.8))) as pool:
                    p_value = pool.starmap(
                        cross_talk_fisher_test,
                        [(activated.loc[cell_types_to_use == cell_type].to_numpy(),) for cell_type in cell_types_to_use.unique()]
                    ) 
                p_value = np.concat(p_value, axis=0)
            p_value_all = cross_talk_fisher_test(activated.to_numpy())
            result = {'by_cell_type': p_value, 'all': p_value_all}
        elif method == 'pearson' or method == 'spearman':
            if normalize_by_cell_rna and normalize_by_volume:
                rna_per_volume = self.get_rna_expression() / self.get_tag('obs:volm').values.reshape(-1, 1)
                cre_celltypes_expression = cre_celltypes_expression / rna_per_volume.mean(axis=1).reshape(-1, 1)
            elif normalize_by_cell_rna and not normalize_by_volume:
                cre_celltypes_expression = cre_celltypes_expression / self.get_rna_expression().mean(axis=1).reshape(-1, 1)
            elif normalize_by_volume and not normalize_by_cell_rna:
                volume = self.get_tag('obs:volm').copy()
                volume = volume.loc[cell_types_to_use.index]
                cre_celltypes_expression = cre_celltypes_expression / volume.values.reshape(-1, 1)
            if cell_types_to_use == ['ALL']:
                p_value = None
                corr = None
            else:
                with multiprocessing.Pool(processes=min(n_jobs, int(multiprocessing.cpu_count()*0.8))) as pool:
                    test_result = pool.starmap(
                        cross_talk_corr_test,
                        [(cre_celltypes_expression.loc[cell_types_to_use == cell_type].to_numpy(), method, ) for cell_type in cell_types_to_use.unique()]
                    )
                p_value = np.concat([r[0] for r in test_result], axis=0)
                corr = np.concat([r[1] for r in test_result], axis=0)
            p_value_all = cross_talk_corr_test(cre_celltypes_expression.to_numpy(), method)
            result = {'by_cell_type': {'p_value': p_value, 'corr': corr},
                      'all': {'p_value': p_value_all[0], 'corr': p_value_all[1]}}
        # save results to attribute
        if not hasattr(self, 'cross_talk_test_results') or not hasattr(self, 'cross_talk_test_configs'):
            self.cross_talk_test_results = []
            self.cross_talk_test_configs = []
        self.cross_talk_test_results.append(result)
        self.cross_talk_test_configs.append(config)
        return result
    
    @staticmethod
    def corr_starrfish(activity_df1: pd.DataFrame, activity_df2: pd.DataFrame,
                       cell_types_to_use: Union[List, pd.Series]=None, 
                       log_activity=False) -> tuple[pd.DataFrame, pd.DataFrame]:
        # filter atac_cpm and activity_df by cell_types_to_use
        if cell_types_to_use is not None:
            # first transform the cell_types_to_use as np.array
            cell_types_to_use = pd.Series(cell_types_to_use)
            cell_types_to_use = cell_types_to_use[cell_types_to_use.isin(activity_df1.index)]
            cell_types_to_use = cell_types_to_use[cell_types_to_use.isin(activity_df2.index)]
        else:
            cell_types_to_use = activity_df1.index.intersection(activity_df2.index)
        activity_df1 = activity_df1.loc[cell_types_to_use]
        activity_df2 = activity_df2.loc[cell_types_to_use]
        cre_to_use = activity_df1.columns.intersection(activity_df2.columns)
        activity_df1 = activity_df1[cre_to_use]
        activity_df2 = activity_df2[cre_to_use]
        if log_activity:
            activity_df1 = np.log1p(activity_df1.astype('float64'))
            activity_df2 = np.log1p(activity_df2.astype('float64'))
        # calculate the correlation for each cre
        # first do col wise correlation
        col_result = col_corr(activity_df1, activity_df2)
        # do row wise correlation
        row_result = row_corr(activity_df1, activity_df2)
        return col_result, row_result
        