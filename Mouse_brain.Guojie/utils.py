import statsmodels.formula.api as smf
import numpy as np
import pandas as pd
import warnings
import time
import multiprocessing
from joblib import Parallel, delayed
from scipy.optimize import minimize
from scipy.special import logsumexp
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=SyntaxWarning, module="docrep")
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
import re
PWD = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PWD)
from starr_fish_vae import STARRFISHVI
from tracksClass import PlotTracks
from cmdstanpy import CmdStanModel
from scipy.stats import linregress
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import logging
from typing import Literal
from genomespy import igv
from pygenometracks.utilities import get_region
from get_preprocess_utils import get_motif, query_motif
cmdstanpy_logger = logging.getLogger("cmdstanpy")
cmdstanpy_logger.disabled = True

def fit_glm(formula, y, x, volm, fov, rna, positive_x_or_y=True, only_keep_positive_x=False, only_keep_positive_y=False):
    try:
        # remove zeros in y
        if positive_x_or_y:
            to_keep = (y > 0) | (x > 0)
        else:
            to_keep = np.ones(len(x), dtype=bool)
        if only_keep_positive_x:
            to_keep &= x > 0
        if only_keep_positive_y:
            to_keep &= y > 0
        y = y[to_keep]
        x = x[to_keep]
        volm = volm[to_keep]
        fov = fov[to_keep]
        rna = rna[to_keep] if rna is not None else None
        # if data points too few, return NaN
        if len(y) < 3:
            return {'coef': np.nan, 'pvalue': np.nan}
        fit_data=pd.DataFrame({'y': y, 'x': x, 'volm': volm, 'fov': fov, 'RNA': rna})
        glm_results = smf.ols(formula, data=fit_data).fit()
        # Direct access to coefficients and p-values instead of HTML parsing
        coef = glm_results.params.get('x', np.nan)
        pvalue = glm_results.pvalues.get('x', np.nan)
        return {'coef': coef, 'pvalue': pvalue}
    except Exception as e:
        return {'coef': np.nan, 'pvalue': np.nan}


def glm(adata, variate='RNA', cell_types_to_use=None, CREs=None, norm_by_volm=False, 
        volm_covariate=False, fov_covariate=False, rna_covariate=False,
        filter_infected_cells=True, positive_x_or_y=True, only_keep_positive_x=True, only_keep_positive_y=True, 
        multiprocess_threads=256, verbose=False):
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
    if variate == 'RNA' or rna_covariate:
        # if variate is RNA, then use the RNA data
        if 'X_raw' not in adata.obsm.keys():
            raise ValueError("X_raw not found in adata.obsm. Please run adata.raw = adata before running this function.")
        if 'RNA' not in adata.obsm.keys():
            adata.obsm['RNA'] = adata.obsm['X_raw'].copy()
    if variate == 'RNA':
        variate = adata.obsm['RNA'].copy().sum(axis=1)
        # repeat # of CREs times
        variate = np.repeat(variate[:, np.newaxis], len(CREs), axis=1)
        # make it to a DataFrame
        variate = pd.DataFrame(variate, index=adata.obs_names, columns=CREs)
    elif variate == 'T7':
        variate = adata.obsm['T7CRE'].copy()
    else:
        raise ValueError("variate must be 'RNA' or 'T7'.")
    if norm_by_volm:
        variate = variate / adata.obs['volm'].values[:, np.newaxis]
    
    # Prepare formula
    formula = 'y ~ x'
    if volm_covariate:
        formula += ' + volm'
    if fov_covariate:
        formula += ' + C(fov)'
    if rna_covariate:
        formula += ' + rna'
    
    # Prepare arguments for all GLM fits using pre-allocated arrays (eliminates slow append operations)
    total_combinations = len(cell_types_to_use) * len(CREs)
    
    # Pre-allocate arrays to avoid slow append operations
    glm_args = [None] * total_combinations
    cell_type_indices = [0] * total_combinations
    cre_indices = [0] * total_combinations
    
    # Pre-compute all cell masks to avoid repeated boolean indexing
    cell_masks = [adata.obs['subclass'] == cell_type for cell_type in cell_types_to_use]
    
    # Pre-compute data arrays that will be reused
    obs_names_array = adata.obs_names.values
    volm_values = adata.obs['volm'].values
    fov_values = adata.obs['fov'].values
    cre_data = adata.obsm['CRE']
    rna_data = adata.obsm['RNA'] if rna_covariate else None
    
    idx = 0
    for k, cell_mask in enumerate(cell_masks):
        # Pre-slice all arrays for this cell type once
        cell_obs_names = obs_names_array[cell_mask]
        cell_volm = volm_values[cell_mask]
        cell_fov = fov_values[cell_mask]
        cell_rna_sum = rna_data[cell_mask].sum(axis=1) if rna_covariate else None
        
        # Extract all CRE data for this cell type at once (vectorized)
        cell_cre_matrix = cre_data.loc[cell_mask, CREs]
        cell_variate_matrix = variate.loc[cell_obs_names, CREs]
        
        # Use list comprehension for remaining loop
        args_batch = [(
            formula,
            cell_cre_matrix.iloc[:, j].values,
            cell_variate_matrix.iloc[:, j].values,
            cell_volm,
            cell_fov,
            cell_rna_sum,
            positive_x_or_y, only_keep_positive_x, only_keep_positive_y
        ) for j in range(len(CREs))]
        
        # Batch assignment
        glm_args[idx:idx+len(CREs)] = args_batch
        cell_type_indices[idx:idx+len(CREs)] = [k] * len(CREs)
        cre_indices[idx:idx+len(CREs)] = list(range(len(CREs)))
        idx += len(CREs)
    
    # Run all GLM fits in parallel
    if multiprocess_threads is not None and multiprocess_threads > 1:
        results = Parallel(n_jobs=min(multiprocess_threads, int(multiprocessing.cpu_count()*0.8)))(
            delayed(fit_glm)(*args) for args in glm_args)
    else:
        results = [fit_glm(*args) for args in glm_args]
    
    # Populate results matrix
    coef = pd.DataFrame(index=cell_types_to_use, columns=CREs, dtype=float)
    pvalue = pd.DataFrame(index=cell_types_to_use, columns=CREs, dtype=float)
    
    for i, result in enumerate(results):
        cell_type_idx = cell_type_indices[i]
        cre_idx = cre_indices[i]
        cell_type = cell_types_to_use[cell_type_idx]
        cre = CREs[cre_idx]
        coef.loc[cell_type, cre] = result['coef']
        pvalue.loc[cell_type, cre] = result['pvalue']
    
    if verbose:
        print(f'Finished fitting GLM for {len(cell_types_to_use)} cell types and {len(CREs)} CREs')
    
    return {'coef': coef, 'pvalue': pvalue}


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
                        ax=None, plot_legend = False, show_title=False, tag='X_spatial', 
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
    plt.scatter(x_, y_, c='gray', s=small, alpha=0.7, marker='.', rasterized=True, edgecolors='none')
    for i, cluster in enumerate(clusters):
        cluster_ = str(cluster)
        inds = scdata.obs[use] == cluster_
        x_ = x[inds]
        y_ = y[inds]
        if isinstance(cmap, dict):
            if cluster in cmap.keys():
                col = cmap[cluster]
            else:
                col = list(cmap.values())[-i % len(cmap)-1]
        else:
            col = cmap[i % len(cmap)]
        if x_region is not None:
            select_region = (x_ > x_region[0]) & (x_ < x_region[1])
            x_ = x_[select_region]
            y_ = y_[select_region]
        if y_region is not None:
            select_region = (y_ > y_region[0]) & (y_ < y_region[1])
            x_ = x_[select_region]
            y_ = y_[select_region]
        ax.scatter(x_, y_, c=col, s=sbig, marker='.',label = cluster_, rasterized=True, edgecolors='none')
    
    # if cluster len is 1, then plot title
    if show_title:
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


def _calculate_fold_change_with_bootstrap(cre_cells_expression, cell_types_order, CRE_info, rna_cells_expression, volm, t7_cells_expression, calc_kwargs, bootstrap_args):
    i, cell_types_to_use, bootstrap_to_fixed_sample_size = bootstrap_args
    if bootstrap_to_fixed_sample_size is not None:
        # if bootstrap_to_fixed_sample is -1, then only select the corresponding cell type of cells
        if bootstrap_to_fixed_sample_size == -1:
            cells_bootstrap = pd.concat(
                [cell_types_to_use[cell_types_to_use == celltype].sample(
                    sum(cell_types_to_use == celltype), replace=True, random_state=i
                ) for celltype in cell_types_to_use.unique()]
            )
            # randomly assign the cell idxs
            cells_idx = pd.concat(
                [cell_types_to_use[cell_types_to_use != celltype].sample(
                    sum(cell_types_to_use == celltype), replace=True, random_state=i
                ) for celltype in cell_types_to_use.unique()]
            )
        else:
            # for each cell_types_to_use, sample the same number of cells
            cells_bootstrap = pd.concat(
                [cell_types_to_use[cell_types_to_use == celltype].sample(
                    bootstrap_to_fixed_sample_size, replace=True, random_state=i
                ) for celltype in cell_types_to_use.unique()]
            )
            # randomly assign the cell idxs
            cells_idx = pd.concat(
                [cell_types_to_use[cell_types_to_use != celltype].sample(
                    bootstrap_to_fixed_sample_size, replace=True, random_state=i
                ) for celltype in cell_types_to_use.unique()]
            )
        cells_bootstrap.index = cells_idx.index
    else:
        cells_bootstrap = cell_types_to_use.sample(frac=1, replace=False, random_state=i)
        cells_bootstrap.index = cell_types_to_use.index
    return calculate_fold_change(cre_cells_expression, cells_bootstrap, cell_types_order, CRE_info, 
                                 rna_cells_expression, volm, t7_cells_expression, **calc_kwargs)


def _calculate_average_with_bootstrap(cre_cells_expression, cell_types_order, CRE_info, rna_cells_expression, volm, t7_cells_expression, calc_kwargs, bootstrap_args):
    i, cell_types_to_use, bootstrap_to_fixed_sample_size, bootstrap_to_fixed_pct = bootstrap_args
    if bootstrap_to_fixed_sample_size is not None:
        # if bootstrap_to_fixed_sample is -1, then only select the corresponding cell type of cells
        if bootstrap_to_fixed_sample_size == -1:
            cells_bootstrap = pd.concat(
                [cell_types_to_use[cell_types_to_use == celltype].sample(
                    sum(cell_types_to_use == celltype), replace=True, random_state=i
                ) for celltype in cell_types_to_use.unique()]
            )
        else:
            # for each cell_types_to_use, sample the same number of cells
            cells_bootstrap = pd.concat(
                [cell_types_to_use[cell_types_to_use == celltype].sample(
                    bootstrap_to_fixed_sample_size, replace=True, random_state=i
                ) for celltype in cell_types_to_use.unique()]
            )
    else:
        cells_bootstrap = cell_types_to_use.sample(frac=bootstrap_to_fixed_pct, replace=True, random_state=i)
    return calculate_fold_change(cre_cells_expression, cells_bootstrap, cell_types_order, CRE_info, 
                                 rna_cells_expression, volm, t7_cells_expression, **calc_kwargs)


def calculate_fold_change(cre_cells_expression: pd.DataFrame, cell_types_to_use: pd.Series, cell_types_order: pd.Series,
                          CRE_info: pd.DataFrame, rna_cells_expression: pd.DataFrame, volm: pd.Series, t7_cells_expression: pd.DataFrame =None,
                          normalize_by_celltype_rna=False, normalize_by_celltype_volume=False,
                          normalize_by_negative_control=False, normalize_by_total_cre=False, normalize_by_infected_cell=False,
                          normalize_by_celltype_t7=False, normalize_by_libsize=False, lib_size=None,
                          filter_zero_counts=False, rank_transform=None, calculate_fdc=False):
    foldchange = pd.DataFrame(index=cell_types_order, columns=cre_cells_expression.columns)
    if filter_zero_counts:
        # get the number of infected cells for each CRE in each cell type
        celltype_activity_matrix = cre_cells_expression.loc[cell_types_to_use.index].groupby(cell_types_to_use.to_numpy()).agg(np.nansum)
        celltype_t7_matrix = t7_cells_expression.loc[cell_types_to_use.index].groupby(cell_types_to_use.to_numpy()).agg(np.nansum) if t7_cells_expression is not None else None
        non_zero_cells = (cre_cells_expression > 0).loc[cell_types_to_use.index].groupby(cell_types_to_use.to_numpy()).agg(np.nansum)
        # before division, fill zeros with 1
        non_zero_cells[non_zero_cells == 0] = 1
        celltype_activity_matrix = celltype_activity_matrix / non_zero_cells
        if t7_cells_expression is not None:
            celltype_t7_matrix = celltype_t7_matrix / non_zero_cells
    else:
        # just average
        celltype_activity_matrix = cre_cells_expression.loc[cell_types_to_use.index].groupby(cell_types_to_use.to_numpy()).agg(np.nanmean)
        celltype_t7_matrix = t7_cells_expression.loc[cell_types_to_use.index].groupby(cell_types_to_use.to_numpy()).agg(np.nanmean) if t7_cells_expression is not None else None
    # store raw celltype_activity_matrix
    celltype_activity_matrix_raw = {'CRE': celltype_activity_matrix.copy(),
                                    'T7': celltype_t7_matrix.copy() if t7_cells_expression is not None else None}
    # get proportion of cells express the CRE in each cell type
    celltype_proportion_matrix = (cre_cells_expression > 0).loc[cell_types_to_use.index].groupby(cell_types_to_use.to_numpy()).agg(np.nanmean)
    if normalize_by_celltype_rna:
        # get cell type RNA
        celltype_rna_matrix = rna_cells_expression.mean(axis=1).loc[cell_types_to_use.index].groupby(cell_types_to_use.to_numpy()).mean()
        celltype_activity_matrix = celltype_activity_matrix / celltype_rna_matrix.values.reshape(-1, 1)
    if normalize_by_celltype_volume:
        # get cell type volume
        celltype_volm_matrix = volm.loc[cell_types_to_use.index].groupby(cell_types_to_use.to_numpy()).mean()
        celltype_activity_matrix = celltype_activity_matrix / celltype_volm_matrix.values.reshape(-1, 1)
    if normalize_by_libsize:
        # normalize by lib size
        celltype_activity_matrix = celltype_activity_matrix / lib_size.values.reshape(1, -1)
    if normalize_by_celltype_t7:
        assert t7_cells_expression is not None, "t7_cells_expression must be provided if normalize_by_celltype_t7 is True"
        # divided by t7 cells expression
        celltype_activity_matrix = celltype_activity_matrix / celltype_t7_matrix
    if normalize_by_negative_control:
        # get the negative control
        negative_control = CRE_info[CRE_info['labeling_type'] == 'negative control'].index
        # if we already divided by t7 cells expression, we do geometric mean
        if normalize_by_celltype_t7:
            from scipy.stats import gmean
            negative_control_mean = celltype_activity_matrix.loc[:, negative_control].apply(lambda x: gmean(x[~np.isnan(x)]), axis=1)
        else:
            negative_control_mean = celltype_activity_matrix.loc[:, negative_control].apply(lambda x: np.nanmean(x), axis=1)
        negative_control_proportion_mean = celltype_proportion_matrix.loc[:, negative_control].mean(axis=1)
        # normalize by negative control
        celltype_activity_matrix = celltype_activity_matrix / negative_control_mean.values.reshape(-1, 1)
        celltype_proportion_matrix = celltype_proportion_matrix / negative_control_proportion_mean.values.reshape(-1, 1)
    if normalize_by_total_cre:
        # get total cre counts
        total_cre = celltype_activity_matrix.sum(axis=1)
        celltype_activity_matrix = celltype_activity_matrix / total_cre.values.reshape(-1, 1)
    if normalize_by_infected_cell:
        # get infect rates per cell type
        infected = ((cre_cells_expression >= 1).sum(axis=1) > 0)
        infect_rate_celltype = infected.loc[cell_types_to_use.index].groupby(cell_types_to_use.to_numpy()).mean()
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
    if calculate_fdc:
        for celltype in np.unique(cell_types_to_use):
            # get the data for the cell type
            celltype_activity = celltype_activity_matrix.loc[celltype]
            # get non_celltype activity
            non_celltype_activity = np.nanmean(cre_cells_expression.loc[cell_types_to_use.index][cell_types_to_use != celltype].mean(axis=0))
            non_celltype_t7 = np.nanmean(t7_cells_expression.loc[cell_types_to_use.index][cell_types_to_use != celltype].mean(axis=0)) if t7_cells_expression is not None else None
            if normalize_by_celltype_rna:
                non_celltype_rna = np.nanmean(rna_cells_expression.mean(axis=1).loc[cell_types_to_use.index][cell_types_to_use != celltype].mean(axis=0))
                non_celltype_activity = non_celltype_activity / non_celltype_rna
            if normalize_by_celltype_volume:
                non_celltype_volm = volm.loc[cell_types_to_use.index][cell_types_to_use != celltype].mean()
                non_celltype_activity = non_celltype_activity / non_celltype_volm
            if normalize_by_libsize:
                non_celltype_activity = non_celltype_activity / lib_size
            if normalize_by_celltype_t7:
                assert non_celltype_t7 is not None, "t7_cells_expression must be provided if normalize_by_celltype_t7 is True"
                non_celltype_activity = non_celltype_activity / non_celltype_t7
            if normalize_by_negative_control:
                non_celltype_negative_control = non_celltype_activity[negative_control].sum()
                non_celltype_activity = non_celltype_activity / non_celltype_negative_control
            if normalize_by_infected_cell:
                non_celltype_infect_rate = infected.loc[cell_types_to_use.index][cell_types_to_use != celltype].mean(axis=0)
                non_celltype_activity = non_celltype_activity / non_celltype_infect_rate
            foldchange.loc[celltype] = celltype_activity / non_celltype_activity
    # remap by cell_types_order
    foldchange = foldchange.reindex(cell_types_order)
    celltype_activity_matrix = celltype_activity_matrix.reindex(cell_types_order)
    celltype_activity_matrix_raw['CRE'] = celltype_activity_matrix_raw['CRE'].reindex(cell_types_order)
    celltype_activity_matrix_raw['T7'] = celltype_activity_matrix_raw['T7'].reindex(cell_types_order) if t7_cells_expression is not None else None
    return foldchange, celltype_activity_matrix, celltype_proportion_matrix, celltype_activity_matrix_raw


def col_corr(df1: pd.DataFrame, df2: pd.DataFrame, bin_threshold1=None, bin_threshold2=None):
    # do col wise correlation
    col_result = pd.DataFrame(index=df1.columns, 
                              columns=['pearson', 'spearman', 'fisher', 
                                       'pearson_p', 'spearman_p', 'fisher_p', 'effect_n', 
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
            col_result.loc[cre, 'effect_n'] = tokeep.sum()
        except:
            print('Error in calculating correlation for CRE: ', cre)
    col_result['pearson_q'] = multitest.multipletests(col_result['pearson_p'], method='fdr_bh')[1]
    col_result['spearman_q'] = multitest.multipletests(col_result['spearman_p'], method='fdr_bh')[1]
    col_result['fisher_q'] = multitest.multipletests(col_result['fisher_p'], method='fdr_bh')[1]
    return col_result


def row_corr(df1: pd.DataFrame, df2: pd.DataFrame, bin_threshold1=None, bin_threshold2=None):
    # do row wise correlation
    row_result = pd.DataFrame(index=df1.index,
                              columns=['pearson', 'spearman', 
                                       'pearson_p', 'spearman_p', 'fisher_p', 'effect_n',
                                       'pearson_q', 'spearman_q', 'fisher_q'])
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
            row_result.loc[celltype, 'effect_n'] = tokeep.sum()
        except:
            print('Error in calculating correlation for celltype: ', celltype)
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


def motif_enrichment(ranked_scores: pd.DataFrame):
    es = ranked_scores.expanding().mean()
    # scale by the sum of ranked scores
    es = es / ranked_scores.mean()
    # integrate the enrichment score
    enrichment_score = es.mean()
    return enrichment_score

# Worker function for multiprocessing optimization - defined at module level for Jupyter compatibility
def _optimize_celltype_cre_worker(args):
    """Worker function for parallel optimization of celltype-cre combinations."""
    celltype, cre, obs_cre, obs_t7, initial_guess = args
    
    # Define the simple optimization function (same as nested function)
    def poisson_negative_binomial(x, obs_cre, obs_t7):
        # parameters
        # x[0]: infection poisson lambda
        # x[1]: T7 detection 0 log probability
        # x[2]: CRE detection 0 log probability
        # x[3], x[4]: CRE activity mean, dispersion
        # iterate infection i from 0 to 65
        ll = np.zeros((obs_cre.shape[0], 66))
        for i in range(66):
            # poisson process
            ll[:, i] += stats.poisson.logpmf(i, mu=x[0])
            if i == 0:
                # T7 detection 0 probability
                ll[:, i] += np.log(1 - obs_t7)
                # CRE detection 0 probability
                ll[:, i] += np.log(obs_cre == 0)
            else:
                # T7 detection probability
                ll[:, i] += (1 - obs_t7) * i * x[1] + obs_t7 * np.log(1 - np.exp(i * x[1]))
                # negative binomial
                ll[:, i] += stats.nbinom.logpmf(obs_cre, n=i * x[4], p=x[4] / (x[4] + x[3] * np.exp(x[2])))
        ll_summed = logsumexp(ll, axis=1)
        return -ll_summed.sum()
    
    # Run optimization
    estimates = minimize(poisson_negative_binomial, initial_guess, args=(obs_cre, obs_t7),
                        bounds=((1e-8, None), (None, -1e-10), (None, -1e-10), (1e-8, None), (1e-8, None)), method='L-BFGS-B')
    return celltype, cre, estimates.x[0], estimates.x[3], estimates.x[4]


class STARRFISH:
    def __init__(self, adata: Union[sc.AnnData, str], 
                 cre_tag = 'obsm:CRE', t7_tag = 'obsm:T7CRE', celltype_tag='obs:subclass', spatial_tag='obsm:X_spatial', creinfo_tag='uns:CRE_info',
                 atac_cpm: Union[pd.DataFrame, str] = 'Data/ATAC/cpm_peakBysubclass.csv',
                 atac_counts: Union[pd.DataFrame, str] = 'Data/ATAC/count_peakBysubclass.csv',
                 lib_size: Union[pd.DataFrame, str] = 'Data/SFv8_400CRE_nanopore_counts.csv',
                 log_lib_size: bool = True):
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
        self.t7_tag = t7_tag
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
            self.lib_size_raw = lib_size.copy()
            if log_lib_size:
                lib_size = np.log1p(lib_size)
            else:
                # assign 0.5 to the zeros
                lib_size[lib_size == 0] = 0.5
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
    
    def load_cpm(self, cpm_path: str, attr_to_add: str = 'atac_cpm'):
        cpm = pd.read_csv(cpm_path, index_col=0)
        cpm.columns = cpm.columns.str.replace('\\.', '-')
        cpm.columns = cpm.columns.str.replace('_', ' ')
        # only keep the cres that are in cre_info
        cre_info = self.get_creinfo().copy()
        # check cpm index format
        if cpm.index.str.startswith('chr').any():    
            cre_info = cre_info[cre_info['enh'].isin(cpm.index)]
            cpm = cpm.loc[cre_info['enh']]
            cpm.index = cre_info.index
        # transpose the cpm
        self.__setattr__(attr_to_add, cpm.transpose())
    
    def load_libsize(self, lib_size_path: str, log_transform: bool = True):
        lib_size = pd.read_csv(lib_size_path, index_col=0)
        # only keep the cres that are in cre_info
        cre_info = self.get_creinfo().copy()
        lib_size = lib_size.loc[lib_size.index.isin(cre_info.index)]
        # reindex the lib_size to match the CRE_info
        lib_size = lib_size.reindex(cre_info.index, fill_value=0)
        self.lib_size_raw = lib_size.copy()
        if log_transform:
            lib_size = np.log1p(lib_size)
        else:
            # assign 0.5 to the zeros
            lib_size[lib_size == 0] = 0.5
        self.lib_size = lib_size
    
    def get_tag(self, tag) -> Union[pd.DataFrame, pd.Series]:
        # get the CREs
        tag_attr = tag.split(':')[0]
        tag_col = tag.split(':')[1]
        if tag_col not in self.adata.__getattribute__(tag_attr).keys():
            return None
        return self.adata.__getattribute__(tag_attr)[tag_col]
    
    def get_cre_expression(self) -> pd.DataFrame:
        return self.get_tag(self.cre_tag)

    def get_t7_expression(self) -> pd.DataFrame:
        if not hasattr(self, 't7_tag') or self.t7_tag is None:
            return None
        return self.get_tag(self.t7_tag)

    def get_rna_expression(self) -> pd.DataFrame:
        # get the RNA expression
        return self.get_tag('obsm:X_raw')

    def get_k_nearest_neighbors(self, cell_id, k=10, spatial_tag='obsm:X_spatial') -> pd.DataFrame:
        # get the k nearest neighbors based on coordinates
        spatial_coords = self.get_tag(spatial_tag)
        cell_mask = self.adata.obs_names == cell_id
        if not cell_mask.any():
            raise ValueError(f"Cell ID {cell_id} not found.")
        # Get target cell coordinates
        cell_coordinates = spatial_coords[cell_mask][0]
        # Vectorized distance calculation to all cells
        distances = np.linalg.norm(spatial_coords - cell_coordinates, axis=1)
        # Exclude the target cell itself
        other_cell_mask = ~cell_mask
        other_distances = distances[other_cell_mask]
        other_cell_names = self.adata.obs_names[other_cell_mask]
        # Get k nearest neighbor indices
        k_nearest_idx = np.argpartition(other_distances, min(k-1, len(other_distances)-1))[:k]
        # Create result DataFrame with cell indices and distances
        result_df = pd.DataFrame({
            'distance': other_distances[k_nearest_idx],
            'X': spatial_coords[other_cell_mask][k_nearest_idx, 0],
            'Y': spatial_coords[other_cell_mask][k_nearest_idx, 1]
        }, index=other_cell_names[k_nearest_idx])
        return result_df.sort_values('distance')

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
    
    def get_positive_control_cres(self, cell_type, use='define') -> pd.Series:
        # get the positive control cres
        if use == 'define':
            cres = self.get_creinfo().copy()
            cres = cres[cres['best_subclass'] == cell_type]
            return cres.index
        elif use == 'atac-peak':
            cre_atac_peaks = pd.read_csv('Data/cre_atac_peaks.csv', index_col=0)
            if cell_type not in cre_atac_peaks.index:
                return None
            cre_atac_peaks = cre_atac_peaks.loc[cell_type]
            cres = cre_atac_peaks[cre_atac_peaks > 0.5].index
            return cres
        elif use == 'chromatin-a':
            chromatin_a = pd.read_csv('Data/cre_chromatin_state_a.csv', index_col=0)
            # get the chromatin-a cres for the cell type
            if cell_type not in chromatin_a.index:
                return None
            chromatin_a = chromatin_a.loc[cell_type]
            cres = chromatin_a[chromatin_a > 0.5].index
            return cres
        elif use == 'chromatin-o':
            chromatin_o = pd.read_csv('Data/cre_chromatin_state_o.csv', index_col=0)
            chromatin_a = pd.read_csv('Data/cre_chromatin_state_a.csv', index_col=0)
            # get the chromatin-o cres for the cell type
            if cell_type not in chromatin_o.index or cell_type not in chromatin_a.index:
                return None
            chromatin_o = chromatin_o.loc[cell_type]
            chromatin_a = chromatin_a.loc[cell_type]
            cres = chromatin_o[chromatin_o > 0.5].index.union(chromatin_a[chromatin_a > 0.5].index)
            return cres
    
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
                  norm_by_negative_control_cell_type_mean=False, norm_by_negative_control_cell_type_sum=False, norm_by_negative_control_single_cell=False,
                  norm_by_t7_cell_type_mean=True, norm_by_t7_cell_type_sum=False, norm_by_t7_single_cell=False,
                  log=True,
                  cell_types_to_use=None, cell_types_to_visualize=None, 
                  nmin=None, nmax=None, sz_background=3, sz_min=5, sz_max=30, 
                  scale_size_by: Literal['counts', 'celltype_number']='counts',  
                  cmap_name='Reds',
                  x_region=None, y_region=None, select_region_by_best_celltype=False, 
                  show_celltypes=True, show_scalebar=True, show_title=True,
                  transpose=1, flipx=1, flipy=1, smooth_k=None, figsize=(30, 10)):
        tag = self.spatial_tag.split(':')[1]
        Xcells = self.adata.obsm[tag][:, ::transpose] * [flipx, flipy]
        # get best cell type
        if use == 'CRE' or use == 'T7CRE':
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
        if norm_by_t7_cell_type_mean and self.get_t7_expression() is not None:
            t7_counts = self.get_t7_expression()[gene].groupby(self.get_celltypes()).mean()
            norm_factor = t7_counts.loc[self.get_celltypes()]
            cts = cts / norm_factor.values
        if norm_by_t7_cell_type_sum and self.get_t7_expression() is not None:
            t7_counts = self.get_t7_expression()[gene].groupby(self.get_celltypes()).sum()
            norm_factor = t7_counts.loc[self.get_celltypes()]
            cts = cts / norm_factor.values
        if norm_by_t7_single_cell and self.get_t7_expression() is not None:
            t7_counts = self.get_t7_expression()[gene]
            cts = cts / t7_counts.values
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
        if y_region is not None:
            select_region = (Xcells[:, 1] > y_region[0]) & (Xcells[:, 1] < y_region[1])
            Xcells = Xcells[select_region]
            cts = cts[select_region]
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
        # Create single figure and axes
        if use == 'CRE' or use == 'T7CRE':
            if show_celltypes:
                fig = plt.figure(figsize=figsize, facecolor='k')
                gs = fig.add_gridspec(1, 3, width_ratios=[0.49, 0.02, 0.49], wspace=0.05)
                ax_main = fig.add_subplot(gs[0])
                ax_cbar = inset_axes(
                                ax_main,
                                width="15%",  # Width of inset
                                height="40%",  # Height of inset
                                loc='upper right',  # Position inside ax_main
                                bbox_to_anchor=(0, 0, 0.9, 1.5),
                                bbox_transform=ax_main.transAxes,
                                borderpad=1
                            )
                ax_ctypes = fig.add_subplot(gs[2])
                plot_cluster_scdata(self.adata, clusters=best_celltype, use='subclass', 
                                    transpose=transpose, flipx=flipx, flipy=flipy, 
                                    x_region=x_region, y_region=y_region,
                                    sbig=20, small=3, ax=ax_ctypes, plot_legend=show_title, show_title=show_title)
            else:
                fig = plt.figure(figsize=figsize, facecolor='k')
                gs = fig.add_gridspec(1, 2, width_ratios=[0.95, 0.05], wspace=0.05)
                ax_main = fig.add_subplot(gs[0])
                ax_cbar = fig.add_subplot(gs[1])
            if show_title:
                ax_main.set_title(f'{gene}', color='white', fontsize=20)
            ax_main.set_facecolor('black')
            # Plot data
            cell_with_genes = np.where(cts > 0)[0]
            # first plot cells without genes, then plot cells with genes
            ax_main.scatter(Xcells[:, 0], Xcells[:, 1], c='grey', s=sz_background, marker='.', alpha=0.7, rasterized=True, edgecolors='none')
            # scale the alpha values for the points based on counts
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_alpha = scaler.fit_transform(ncts.reshape(-1,1)).flatten()
            # np clip the scaled_alpha to [0, 1]
            scaled_alpha = np.clip(scaled_alpha, 0, 1)
            # plot the CRE counts
            # ax_main.scatter(Xcells[cell_with_genes, 0], Xcells[cell_with_genes, 1], c='#00FF00', sizes=size[cell_with_genes], alpha=scaled_alpha[cell_with_genes], rasterized=True)
            ax_main.scatter(Xcells[:, 0], Xcells[:, 1], c='#00FF00', sizes=size, alpha=scaled_alpha, rasterized=True, edgecolors='none')

            # Format axes
            ax_main.grid(False)
            ax_main.set_xticks([])
            ax_main.set_yticks([])
            ax_main.set_aspect('equal')
            # Format colorbar
            ax_cbar.set_facecolor('black')
            if show_scalebar:
                ax_cbar.axis('off')

                # Define scale bar points
                legend_vals = np.linspace(0, 1.0, 7)  # Normalized from 0 to 1
                legend_cts = legend_vals * nmax

                # Reuse the size and alpha scaling from main plot
                legend_scaled_sizes = legend_vals ** 3  # Same emphasis
                legend_sizes = sz_min + (sz_max - sz_min) * legend_scaled_sizes
                legend_alphas = legend_vals  # Directly scale alpha from value

                # Plot circles in ax_cbar
                dot_spacing = 0.15  # smaller = tighter packing
                for i, (val, sz, alpha) in enumerate(zip(legend_cts, legend_sizes, legend_alphas)):
                    x = i * dot_spacing
                    ax_cbar.scatter(x, 0.25, s=sz, alpha=alpha, color='#00FF00', edgecolors='none')

                # Add only min and max labels
                ax_cbar.text(-0.6, 0.25, f'{legend_cts[0]:.2f}', va='center', ha='center', color='white', fontsize=12)
                ax_cbar.text(1.4, 0.25, f'{legend_cts[-1]:.2f}', va='center', ha='center', color='white', fontsize=12)

                # Set limits and aesthetics
                ax_cbar.set_xlim(0, 2)
                ax_cbar.set_xlim(-0.5, (len(legend_cts)-1) * dot_spacing + 0.5)
                ax_cbar.set_ylim(0, 1.5)  # Enough vertical space for dots + labels
                # ax_cbar.set_ylim(-0.5, len(legend_cts) - 0.5)
                ax_cbar.text(0.5 * (len(legend_cts)-1) * dot_spacing, 0.4, 'Normalized Counts',
                ha='center', va='top', color='white', fontsize=12)

                # Format colorbar
                # cbar.set_label('Normalized Counts', color='white', fontsize=16)
                # cbar.ax.yaxis.set_tick_params(color='white')
                # cbar.ax.tick_params(labelcolor='white', labelsize=10)
                # cbar.ax.set_yticks(np.linspace(0, nmax, 5))
                # cbar.ax.set_yticklabels([f'{i:.2f}' for i in np.linspace(0, nmax, 5)], color='white')

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
            if isinstance(cmap, list) or isinstance(cmap, np.ndarray):
                col = cmap[i % len(cmap)]
            else:
                if cluster_ in cmap.keys():
                    col = cmap[cluster_]
                else:
                    col = list(cmap.values())[-i % len(cmap)-1]
            ax.scatter(x_, y_, c=col, s=size, marker='.',label = cluster_, rasterized=True)
        
        # if cluster len is 1, then plot title
        ax.set_title(f"Cell types", color='white', fontsize=20)
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
        ax.set_facecolor('black')
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
                         normalize_by_cell_rna=False, normalize_by_cell_volume=False, normalize_by_cell_t7=False, filter_by_cell_t7=None,
                         normalize_by_celltype_rna=False, normalize_by_celltype_volume=False,
                         normalize_by_negative_control=False, normalize_by_infected_cell=False,
                         normalize_by_celltype_t7=False, normalize_by_total_cre=False, normalize_by_libsize=False,
                         filter_zero_counts=False, log_transform=False, bulk_log_transform=False, rank_transform=None,
                         bootstrap_number=None, bootstrap_to_fixed_sample_size=None, apply_bootstrap_in_observation=False,
                         calculate_fdc=False, fill_nan=True, n_jobs=256, load_stored=True, dry_run=False) -> dict:
        config = {
            'cell_types_to_use': cell_types_to_use,
            'normalize_by_cell_rna': normalize_by_cell_rna,
            'normalize_by_cell_volume': normalize_by_cell_volume,
            'normalize_by_cell_t7': normalize_by_cell_t7,
            'filter_by_cell_t7': filter_by_cell_t7,
            'normalize_by_celltype_rna': normalize_by_celltype_rna,
            'normalize_by_celltype_volume': normalize_by_celltype_volume,
            'normalize_by_negative_control': normalize_by_negative_control,
            'normalize_by_total_cre': normalize_by_total_cre,
            'normalize_by_infected_cell': normalize_by_infected_cell,
            'normalize_by_celltype_t7': normalize_by_celltype_t7,
            'normalize_by_libsize': normalize_by_libsize,
            "filter_zero_counts": filter_zero_counts,
            'log_transform': log_transform,
            'rank_transform': rank_transform,
            'bootstrap_number': bootstrap_number,
            'bootstrap_to_fixed_sample_size': bootstrap_to_fixed_sample_size,
            'apply_bootstrap_in_observation': apply_bootstrap_in_observation,
            'calculate_fdc': calculate_fdc,
            'fill_nan': fill_nan,}
        default_config = {
            'cell_types_to_use': None,
            'normalize_by_cell_rna': False, 'normalize_by_cell_volume': False, 'normalize_by_cell_t7': False, 'filter_by_cell_t7': None,
            'normalize_by_celltype_rna': False, 'normalize_by_celltype_volume': False, 'normalize_by_celltype_t7': False,
            'normalize_by_negative_control': False, 'normalize_by_infected_cell': False,
            'normalize_by_total_cre': False, 'normalize_by_libsize': False,
            'filter_zero_counts': False, 'log_transform': False, 'rank_transform': None,
            'bootstrap_number': None, 
            'bootstrap_to_fixed_sample_size': None, 'apply_bootstrap_in_observation': False,
            'calculate_fdc': False, 'fill_nan': True,
        }
        # check if the results already exist
        partial_loaded = False
        fold_change_test_result = None
        if hasattr(self, 'fold_change_test_results') and hasattr(self, 'fold_change_test_configs') and load_stored:
            for stored_config, stored_result in zip(self.fold_change_test_configs, self.fold_change_test_results):
                # only partially check the config, everything the same except bootstrap_number
                def check_config_matched(config, stored_config):
                    keys_matched = []
                    for k in config:
                        if k == 'bootstrap_number':
                            continue
                        if k in stored_config:
                            if config[k] == stored_config[k]:
                                keys_matched.append(True)
                            else:
                                keys_matched.append(False)
                        else:
                            if config[k] == default_config[k]:
                                keys_matched.append(True)
                            else:
                                keys_matched.append(False)
                    return all(keys_matched)
                if check_config_matched(config, stored_config):
                    # if the results already exist, return the results
                    fold_change_test_result = stored_result.copy()
                    if stored_config['bootstrap_number'] == config['bootstrap_number'] or config['bootstrap_number'] is None:
                        print('Results already exist, return stored results')
                        return fold_change_test_result
        if dry_run:
            if fold_change_test_result is None:
                print('Dry run, no results loaded or calculated.')
            return fold_change_test_result
        if fold_change_test_result is not None:
            partial_loaded = True
        if cell_types_to_use is not None:
            # get the cell types
            cre_cells_expression, rna_cells_expression, cell_types_to_use = self.get_cre_rna_celltypes(cell_types_to_use)
        else:
            cre_cells_expression = self.get_cre_expression().copy()
            rna_cells_expression = self.get_rna_expression().copy()
            cell_types_to_use = self.get_celltypes()
        volm = self.get_tag('obs:volm').copy()
        volm = volm.loc[cell_types_to_use.index]
        # transform rna_celltypes_expression to dataframe
        rna_cells_expression = pd.DataFrame(rna_cells_expression, index=cell_types_to_use.index)
        t7_cells_expression = self.get_t7_expression()
        if t7_cells_expression is not None:
            t7_cells_expression = t7_cells_expression.loc[cre_cells_expression.index]
        if normalize_by_cell_rna and normalize_by_cell_volume:
            rna_per_volume = rna_cells_expression / volm.values.reshape(-1, 1)
            cre_cells_expression = cre_cells_expression / rna_per_volume.mean(axis=1).values.reshape(-1, 1)
        elif normalize_by_cell_rna and not normalize_by_cell_volume:
            cre_cells_expression = cre_cells_expression / rna_cells_expression.mean(axis=1).values.reshape(-1, 1)
        elif normalize_by_cell_volume and not normalize_by_cell_rna:
            cre_cells_expression = cre_cells_expression / volm.values.reshape(-1, 1)
        if filter_by_cell_t7 is not None:
            # filter out the cells that don't have t7 expression above the filter_by_cell_t7 threshold
            if t7_cells_expression is not None:
                cre_cells_expression[t7_cells_expression < filter_by_cell_t7] = np.nan
                t7_cells_expression[t7_cells_expression < filter_by_cell_t7] = np.nan
        if normalize_by_cell_t7:
            assert t7_cells_expression is not None, "t7_cells_expression is required when normalize_by_cell_t7 is True"
            cre_cells_expression = cre_cells_expression / t7_cells_expression
            # if we encounter NaN or Inf, we will fill them with 0
            cre_cells_expression = cre_cells_expression.fillna(0).replace([np.inf, -np.inf], 0)
        if not normalize_by_cell_t7 and not normalize_by_celltype_t7:
            t7_cells_expression = None
        if log_transform:
            cre_cells_expression = np.log1p(cre_cells_expression)
        cre_info = self.get_creinfo().copy()
        calculate_fold_change_args = {
            'cre_cells_expression': cre_cells_expression,
            'cell_types_to_use': cell_types_to_use,
            'cell_types_order': np.unique(cell_types_to_use),
            'CRE_info': cre_info, 'rna_cells_expression': rna_cells_expression, 'volm': volm,
            'normalize_by_celltype_rna': normalize_by_celltype_rna, 'normalize_by_celltype_volume': normalize_by_celltype_volume,
            'normalize_by_negative_control': normalize_by_negative_control, 'normalize_by_total_cre': normalize_by_total_cre, 'normalize_by_infected_cell': normalize_by_infected_cell, 
            'normalize_by_celltype_t7': normalize_by_celltype_t7, 't7_cells_expression': t7_cells_expression,
            'normalize_by_libsize': normalize_by_libsize, 'lib_size': self.lib_size['counts'], 
            'filter_zero_counts': filter_zero_counts, 'rank_transform': rank_transform, 'calculate_fdc': calculate_fdc
        }
        if not partial_loaded:
            if bootstrap_to_fixed_sample_size is not None and apply_bootstrap_in_observation:
                # for each cell_types_to_use, sample the same number of cells
                cells_bootstrap = pd.concat([cell_types_to_use[cell_types_to_use == celltype].sample(bootstrap_to_fixed_sample_size, replace=True, random_state=2**32-1) for celltype in cell_types_to_use.unique()])
                calculate_fold_change_args['cell_types_to_use'] = cells_bootstrap
            foldchange, celltype_activity, celltype_proportion, _ = calculate_fold_change(**calculate_fold_change_args)
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
                'celltype_proportion': celltype_proportion
            }
        # do bootstrap, if bootstrap_number is not None
        if bootstrap_number is not None:
            foldchange = fold_change_test_result['foldchange']
            celltype_activity = fold_change_test_result['celltype_activity']
            if n_jobs is None:
                n_jobs = int(multiprocessing.cpu_count()*0.8)
            bootstrap_prep_args = [(i, cell_types_to_use, bootstrap_to_fixed_sample_size) for i in range(bootstrap_number)]
            # Prepare kwargs for calculate_fold_change
            calc_kwargs = {
                'normalize_by_celltype_rna': normalize_by_celltype_rna,
                'normalize_by_celltype_volume': normalize_by_celltype_volume,
                'normalize_by_negative_control': normalize_by_negative_control,
                'lib_size': self.lib_size['counts'],
                'normalize_by_total_cre': normalize_by_total_cre,
                'normalize_by_infected_cell': normalize_by_infected_cell,
                'normalize_by_libsize': normalize_by_libsize,
                'normalize_by_celltype_t7': normalize_by_celltype_t7,
                'filter_zero_counts': filter_zero_counts,
                'rank_transform': rank_transform,
                'calculate_fdc': calculate_fdc,
            }
            print('Finished preparing bootstrap args, start calculating bootstrap')
            bootstrap_results = Parallel(n_jobs=n_jobs, backend='loky', batch_size=1)(
                delayed(_calculate_fold_change_with_bootstrap)(
                    cre_cells_expression, np.unique(cell_types_to_use), cre_info, 
                    rna_cells_expression, volm, t7_cells_expression, calc_kwargs, args
                ) for args in bootstrap_prep_args
            )
            foldchange_array = np.ndarray((bootstrap_number, foldchange.shape[0], foldchange.shape[1]))
            activity_array = np.ndarray((bootstrap_number, celltype_activity.shape[0], celltype_activity.shape[1]))
            proportion_array = np.ndarray((bootstrap_number, celltype_proportion.shape[0], celltype_proportion.shape[1]))
            for i, (fc, act, porp, _) in enumerate(bootstrap_results):
                foldchange_array[i] = fc
                activity_array[i] = act
                proportion_array[i] = porp
            # calculate p-value for foldchange and celltype_activity
            pvalue = pd.DataFrame(index=foldchange.index, columns=foldchange.columns)
            pvalue_activity = pd.DataFrame(index=celltype_activity.index, columns=celltype_activity.columns)
            pvalue_proportion = pd.DataFrame(index=celltype_proportion.index, columns=celltype_proportion.columns)
            # if we use normalize_by_negative_control, fill the nan with 0
            if fill_nan:
                foldchange_array[np.isnan(foldchange_array)] = 0
                activity_array[np.isnan(activity_array)] = 0
                proportion_array[np.isnan(proportion_array)] = 0
            for i in range(foldchange.shape[0]):
                for j in range(foldchange.shape[1]):
                    # get nan values
                    nan_values = np.sum(np.isnan(foldchange_array[:, i, j] >= foldchange.iloc[i, j]))
                    # if nan_values larger than half of bootstrap_number, set pvalue to 1
                    pvalue.iloc[i, j] = np.nanmean(foldchange_array[:, i, j] >= foldchange.iloc[i, j]) \
                        if nan_values < bootstrap_number / 2 else 1.0
                    # get nan values for activity
                    nan_values_activity = np.sum(np.isnan(activity_array[:, i, j] >= celltype_activity.iloc[i, j]))
                    pvalue_activity.iloc[i, j] = np.nanmean(activity_array[:, i, j] >= celltype_activity.iloc[i, j]) \
                        if nan_values_activity < bootstrap_number / 2 else 1.0
                    nan_values_proportion = np.sum(np.isnan(proportion_array[:, i, j] >= celltype_proportion.iloc[i, j]))
                    pvalue_proportion.iloc[i, j] = np.nanmean(proportion_array[:, i, j] >= celltype_proportion.iloc[i, j]) \
                        if nan_values_proportion < bootstrap_number / 2 else 1.0
            # do multiple testing correction for all pvalues
            qvalue = pd.DataFrame(multitest.multipletests(pvalue.astype(float).values.flatten(), method='fdr_bh')[1].reshape(pvalue.shape),
                                 index=pvalue.index, columns=pvalue.columns)
            qvalue_activity = pd.DataFrame(multitest.multipletests(pvalue_activity.astype(float).values.flatten(), method='fdr_bh')[1].reshape(pvalue_activity.shape),
                                           index=pvalue_activity.index, columns=pvalue_activity.columns)
            qvalue_proportion = pd.DataFrame(multitest.multipletests(pvalue_proportion.astype(float).values.flatten(), method='fdr_bh')[1].reshape(pvalue_proportion.shape),
                                             index=pvalue_proportion.index, columns=pvalue_proportion.columns)
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
                    fold_change_test_result['cre_info'].loc[cre, 'pvalue_proportion'] = pvalue_proportion.loc[best_subclass, cre]
                    fold_change_test_result['cre_info'].loc[cre, 'qvalue_proportion'] = qvalue_proportion.loc[best_subclass, cre]
            fold_change_test_result['foldchange_array'] = foldchange_array
            fold_change_test_result['pvalue'] = pvalue
            fold_change_test_result['qvalue'] = qvalue
            fold_change_test_result['activity_array'] = activity_array
            fold_change_test_result['pvalue_activity'] = pvalue_activity
            fold_change_test_result['qvalue_activity'] = qvalue_activity
            fold_change_test_result['proportion_array'] = proportion_array
            fold_change_test_result['pvalue_proportion'] = pvalue_proportion
            fold_change_test_result['qvalue_proportion'] = qvalue_proportion
        # save results to attribute
        if not hasattr(self, 'fold_change_test_results') or not hasattr(self, 'fold_change_test_configs'):
            self.fold_change_test_results = []
            self.fold_change_test_configs = []
        self.fold_change_test_results.append(fold_change_test_result)
        self.fold_change_test_configs.append(config)
        return fold_change_test_result

    def average_bootstrap_test(self, cell_types_to_use, normalize_by_cell_rna=False, normalize_by_cell_volume=False, normalize_by_cell_t7=False,
                               normalize_by_celltype_rna=False, normalize_by_celltype_volume=False,
                               normalize_by_negative_control=False, normalize_by_celltype_t7=False, filter_by_cell_t7=None,
                               normalize_by_libsize=False, log_transform=False, 
                               bootstrap_number=None, bootstrap_to_fixed_sample_size=None, bootstrap_to_fixed_pct=None,
                               fill_nan=True, n_jobs=256, load_stored=True, dry_run=False) -> dict:
        # This function is similar to fold_change_test but only do bootstrap within each cell type.
        config = {
            'cell_types_to_use': cell_types_to_use,
            'normalize_by_cell_rna': normalize_by_cell_rna,
            'normalize_by_cell_volume': normalize_by_cell_volume,
            'normalize_by_cell_t7': normalize_by_cell_t7,
            'normalize_by_celltype_rna': normalize_by_celltype_rna,
            'normalize_by_celltype_volume': normalize_by_celltype_volume,
            'normalize_by_negative_control': normalize_by_negative_control,
            'normalize_by_celltype_t7': normalize_by_celltype_t7,
            'filter_by_cell_t7': filter_by_cell_t7,
            'normalize_by_libsize': normalize_by_libsize,
            'log_transform': log_transform,
            'bootstrap_number': bootstrap_number,
            'bootstrap_to_fixed_sample_size': bootstrap_to_fixed_sample_size,
            'bootstrap_to_fixed_pct': bootstrap_to_fixed_pct,
            'fill_nan': fill_nan,
        }
        # check if the results already exist
        if load_stored and hasattr(self, 'average_bootstrap_test_results') and hasattr(self, 'average_bootstrap_test_configs'):
            for stored_config, stored_result in zip(self.average_bootstrap_test_configs, self.average_bootstrap_test_results):
                if all(stored_config[k] == config[k] for k in config if k in stored_config):
                    # if the results already exist, return the results
                    print('Results already exist, return stored results')
                    return stored_result.copy()
        if cell_types_to_use is not None:
            # get the cell types
            cre_cells_expression, rna_cells_expression, cell_types_to_use = self.get_cre_rna_celltypes(cell_types_to_use)
        else:
            cre_cells_expression = self.get_cre_expression().copy()
            rna_cells_expression = self.get_rna_expression().copy()
            cell_types_to_use = self.get_celltypes()
        volm = self.get_tag('obs:volm').copy()
        volm = volm.loc[cell_types_to_use.index]
        # transform rna_celltypes_expression to dataframe
        rna_cells_expression = pd.DataFrame(rna_cells_expression, index=cell_types_to_use.index)
        t7_cells_expression = self.get_t7_expression()
        if t7_cells_expression is not None:
            t7_cells_expression = t7_cells_expression.loc[cre_cells_expression.index]
        if normalize_by_cell_rna and normalize_by_cell_volume:
            rna_per_volume = rna_cells_expression / volm.values.reshape(-1, 1)
            cre_cells_expression = cre_cells_expression / rna_per_volume.mean(axis=1).values.reshape(-1, 1)
        elif normalize_by_cell_rna and not normalize_by_cell_volume:
            cre_cells_expression = cre_cells_expression / rna_cells_expression.mean(axis=1).values.reshape(-1, 1)
        elif normalize_by_cell_volume and not normalize_by_cell_rna:
            cre_cells_expression = cre_cells_expression / volm.values.reshape(-1, 1)
        if normalize_by_cell_t7:
            assert t7_cells_expression is not None, "t7_cells_expression is required when normalize_by_cell_t7 is True"
            cre_cells_expression = cre_cells_expression / t7_cells_expression
            # fill inf values with nan
            cre_cells_expression[np.isinf(cre_cells_expression)] = np.nan
            # if normalize_by_cell_t7 is a numeric value, fill nan for cells with t7 smaller than that value
            if isinstance(normalize_by_cell_t7, (int, float)):
                cre_cells_expression[t7_cells_expression < normalize_by_cell_t7] = np.nan
        if filter_by_cell_t7 is not None:
            # filter the cells with t7 expression smaller than filter_by_cell_t7
            if t7_cells_expression is not None:
                cre_cells_expression[t7_cells_expression < filter_by_cell_t7] = np.nan
                t7_cells_expression[t7_cells_expression < filter_by_cell_t7] = np.nan
        if not normalize_by_cell_t7 and not normalize_by_celltype_t7:
            t7_cells_expression = None
        if log_transform:
            cre_cells_expression = np.log(cre_cells_expression)
        cre_info = self.get_creinfo().copy()
        # Prepare kwargs for calculate_fold_change
        calc_kwargs = {
            'normalize_by_celltype_rna': normalize_by_celltype_rna,
            'normalize_by_celltype_volume': normalize_by_celltype_volume,
            'normalize_by_negative_control': normalize_by_negative_control,
            'lib_size': self.lib_size['counts'],
            'normalize_by_libsize': normalize_by_libsize,
            'normalize_by_celltype_t7': normalize_by_celltype_t7,
            'calculate_fdc': False,
        }
        if bootstrap_number is not None:
            if n_jobs is None:
                n_jobs = int(multiprocessing.cpu_count()*0.8)
            bootstrap_prep_args = [(i, cell_types_to_use, bootstrap_to_fixed_sample_size, bootstrap_to_fixed_pct) for i in range(bootstrap_number)]
            print('Finished preparing bootstrap args, start calculating bootstrap')
            bootstrap_results = Parallel(n_jobs=n_jobs, backend='loky', batch_size=1)(
                delayed(_calculate_average_with_bootstrap)(
                    cre_cells_expression, np.unique(cell_types_to_use), cre_info, 
                    rna_cells_expression, volm, t7_cells_expression, calc_kwargs, args
                ) for args in bootstrap_prep_args
            )
            # store into an array
            celltype_activity_array = np.ndarray((bootstrap_number, len(np.unique(cell_types_to_use)), cre_cells_expression.shape[1]))
            celltype_CRE_raw = np.ndarray((bootstrap_number, len(np.unique(cell_types_to_use)), cre_cells_expression.shape[1]))
            celltype_T7_raw = np.ndarray((bootstrap_number, len(np.unique(cell_types_to_use)), cre_cells_expression.shape[1]))
            for i, (_, act, _, raw) in enumerate(bootstrap_results):
                celltype_activity_array[i] = act
                celltype_CRE_raw[i] = raw['CRE']
                celltype_T7_raw[i] = raw['T7']
            # replace inf with NaN
            tmp = celltype_activity_array.copy()
            tmp[np.isinf(tmp)] = np.nan
            cre_celltype_activity = np.nanmean(tmp, axis=0)
            # transform to DataFrame
            cre_celltype_activity = pd.DataFrame(cre_celltype_activity, index=np.unique(cell_types_to_use), 
                                                 columns=cre_cells_expression.columns)
        else:
            # just calculate the fold change without bootstrap
            res = calculate_fold_change(cre_cells_expression, cell_types_to_use, np.unique(cell_types_to_use), cre_info, 
                                        rna_cells_expression, volm, t7_cells_expression, **calc_kwargs)
            celltype_activity_array = None
            celltype_CRE_raw = None
            celltype_T7_raw = None
            cre_celltype_activity = res[1]
        # store the results
        if not hasattr(self, 'average_bootstrap_test_results') or not hasattr(self, 'average_bootstrap_test_configs'):
            self.average_bootstrap_test_results = []
            self.average_bootstrap_test_configs = []
        res = {'celltype_activity_array': celltype_activity_array,
               'celltype_CRE_raw': celltype_CRE_raw,
               'celltype_T7_raw': celltype_T7_raw,
               'celltype_activity': cre_celltype_activity}
        self.average_bootstrap_test_results.append(res)
        self.average_bootstrap_test_configs.append(config)
        return res

    def average_bootstrap_test_q(self, res, threshold: Literal['0', 'total', 'total_dist', 'neg_control', 'neg_control_dist'] = '0', 
                                 norm='T7', tail='right', to_filter=None):
        # fold change to T7 array
        res_array = res['celltype_activity_array'].copy()
        # log then average
        res_array = np.log(res_array)
        # assign inf to NaN
        res_array[np.isinf(res_array)] = np.nan
        # if we have to_filter, then fill it with np.nan
        if to_filter is not None:
            for cell_type in to_filter.index:
                res_array[:, res['celltype_activity'].index == cell_type, to_filter.loc[cell_type]] = np.nan
        neg_control_array = res_array[:, :, res['celltype_activity'].columns.isin(self.get_negative_control_cres())]
        neg_control_array = np.nanmean(neg_control_array, axis=2)
        # turn to DataFrame
        res_df = pd.DataFrame(np.nanmean(res_array, axis=0), index=res['celltype_activity'].index, columns=res['celltype_activity'].columns)
        # for each cell type, calculate the fold change of total CRE / total T7
        res_p1 = pd.DataFrame(index=res['celltype_activity'].index, columns=res['celltype_activity'].columns)
        res_p2 = pd.DataFrame(index=res['celltype_activity'].index, columns=res['celltype_activity'].columns)
        for cell_type in res['celltype_activity'].index:
            cre_ct = self.get_cre_expression().loc[self.get_celltypes() == cell_type].copy()
            t7_ct = self.get_t7_expression().loc[self.get_celltypes() == cell_type].copy()
            total_cre = cre_ct.sum(axis=0).sum()
            total_t7 = t7_ct.sum(axis=0).sum() if t7_ct is not None else 0
            total_libsize = self.lib_size['counts'].sum() if self.lib_size is not None else 0
            total_neg_control_cre = cre_ct[self.get_negative_control_cres()].sum(axis=0).sum()
            total_neg_control_t7 = t7_ct[self.get_negative_control_cres()].sum(axis=0).sum() if t7_ct is not None else 0
            total_neg_control_libsize = self.lib_size['counts'][self.get_negative_control_cres()].sum() if self.lib_size is not None else 0
            # calculate the fold change
            if threshold == '0':
                fdc_u = 0
                fdc_l = fdc_u
            elif threshold == 'total':
                if norm == 'T7':
                    fdc_u = np.log(total_cre / total_t7)
                elif norm == 'libsize':
                    fdc_u = np.log(total_cre / total_libsize)
                fdc_l = fdc_u
            elif threshold == 'total_dist':
                # use the distribution of total CREs to set the threshold
                fdc = np.nanmean(res_array[:, res['celltype_activity'].index == cell_type])
                fdc_std = np.nanstd(res_array[:, res['celltype_activity'].index == cell_type])
                fdc_u = fdc + 2 * fdc_std
                fdc_l = fdc - 2 * fdc_std
            elif threshold == 'neg_control':
                if norm == 'T7':
                    fdc_u = np.log(total_neg_control_cre / total_neg_control_t7)
                elif norm == 'libsize':
                    fdc_u = np.log(total_neg_control_cre / total_neg_control_libsize)
                fdc_l = fdc_u
            elif threshold == 'neg_control_dist':
                # use the distribution of negative control CREs to set the threshold
                fdc = np.nanmean(neg_control_array[:, res['celltype_activity'].index == cell_type])
                fdc_std = np.nanstd(neg_control_array[:, res['celltype_activity'].index == cell_type])
                fdc_u = fdc + 2 * fdc_std
                fdc_l = fdc - 2 * fdc_std
            if np.isnan(fdc_u) or np.isinf(fdc_u):
                res_p1.loc[cell_type] = np.nan
            else:
                res_p1.loc[cell_type] = np.nanmean(res_array[:, res['celltype_activity'].index == cell_type, :] < fdc_u, axis=0)
            if np.isnan(fdc_l) or np.isinf(fdc_l):
                res_p2.loc[cell_type] = np.nan
            else:
                res_p2.loc[cell_type] = np.nanmean(res_array[:, res['celltype_activity'].index == cell_type, :] > fdc_l, axis=0)
            # if too many nan, the test is failed
            res_nansum = (~np.isnan(res_array[:, res['celltype_activity'].index == cell_type, :])).sum(axis=0)
            res_p1.loc[cell_type][res_nansum[0] < res_array.shape[0] * 0.01] = np.nan
            res_p2.loc[cell_type][res_nansum[0] < res_array.shape[0] * 0.01] = np.nan
            res_df.loc[cell_type][res_nansum[0] < res_array.shape[0] * 0.01] = np.nan
        # do q-value correction
        # For res_p1
        p_values_flat1 = res_p1.values.flatten()
        valid_mask1 = ~pd.isna(p_values_flat1)
        q_values_flat1 = np.full_like(p_values_flat1, np.nan, dtype=float)
        if valid_mask1.any():
            q_values_flat1[valid_mask1] = multitest.multipletests(p_values_flat1[valid_mask1], method='fdr_bh')[1]
        res2_q1 = pd.DataFrame(q_values_flat1.reshape(res_p1.shape), index=res_p1.index, columns=res_p1.columns)
        
        # For res_p2
        p_values_flat2 = res_p2.values.flatten()
        valid_mask2 = ~pd.isna(p_values_flat2)
        q_values_flat2 = np.full_like(p_values_flat2, np.nan, dtype=float)
        if valid_mask2.any():
            q_values_flat2[valid_mask2] = multitest.multipletests(p_values_flat2[valid_mask2], method='fdr_bh')[1]
        res2_q2 = pd.DataFrame(q_values_flat2.reshape(res_p2.shape), index=res_p2.index, columns=res_p2.columns)
        res2_q = pd.DataFrame(np.minimum(res2_q1.values, res2_q2.values), index=res_p1.index, columns=res_p1.columns)
        if tail == 'right':
            return res2_q1, res_df
        elif tail == 'left':
            return res2_q2, res_df
        elif tail == 'both':
            return res2_q, res_df

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
            # check if we log transform the lib or not
            if self.lib_size_raw is not None:
                lib_size.loc['sum'] = np.log1p(self.lib_size_raw['counts'].loc[tmp].sum())
            else:
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

    def glm_test(self, variate='RNA', cell_types_to_use: List=None, norm_by_volm=False, volm_covariate=False, fov_covariate=False, rna_covariate=False,
                 filter_infected_cells=True, positive_x_or_y = True, only_keep_positive_x=False, only_keep_positive_y = False, multiprocess_threads=256) -> dict:
        config = {
            'variate': variate,
            'norm_by_volm': norm_by_volm,
            'volm_covariate': volm_covariate,
            'fov_covariate': fov_covariate,
            'rna_covariate': rna_covariate,
            'filter_infected_cells': filter_infected_cells,
            'positive_x_or_y': positive_x_or_y,
            'only_keep_positive_x': only_keep_positive_x,
            'only_keep_positive_y': only_keep_positive_y,
        }
        # if the results already exist, return the results
        if hasattr(self, 'glm_test_results') and hasattr(self, 'glm_test_configs'):
            for stored_config, glm_result in zip(self.glm_test_configs, self.glm_test_results):
                if stored_config == config:
                    # if the results already exist, return the results
                    print('Results already exist, return stored results')
                    return glm_result.copy()
        result = glm(self.adata, variate=variate, cell_types_to_use=cell_types_to_use, norm_by_volm=norm_by_volm, 
                     volm_covariate=volm_covariate, fov_covariate=fov_covariate, rna_covariate=rna_covariate,
                     filter_infected_cells=filter_infected_cells, 
                     positive_x_or_y=positive_x_or_y,
                     only_keep_positive_x=only_keep_positive_x,
                     only_keep_positive_y=only_keep_positive_y,
                     multiprocess_threads=multiprocess_threads)
        # add results to attribute
        if not hasattr(self, 'glm_test_results') or not hasattr(self, 'glm_test_configs'):
            self.glm_test_results = []
            self.glm_test_configs = []
        self.glm_test_results.append(result)
        self.glm_test_configs.append(config)
        return result

    def pseudo_bulk_glm_test(self, variate='RNA', cell_types_to_use: List=None, norm_by_volm=False, volm_covariate=False, rna_covariate=False,
                             filter_infected_cells=True, positive_x_or_y=True, only_keep_positive_x=False, only_keep_positive_y=False, 
                             pseudo_bulk_size=50, pseudo_bulk_percentage=None, pseudo_bulk_number=1000, replace=True, 
                             multiprocess_threads=256) -> dict:
        # check if the results already exist
        config = {'cell_types_to_use': cell_types_to_use, 
                  'norm_by_volm': norm_by_volm, 
                  'volm_covariate': volm_covariate,
                  'filter_infected_cells': filter_infected_cells,
                  'positive_x_or_y': positive_x_or_y,
                  'only_keep_positive_x': only_keep_positive_x,
                  'only_keep_positive_y': only_keep_positive_y,
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
        t7_expression = self.get_t7_expression()
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
        # if we have t7 expression, we will also create a pseudo bulk for t7
        pseudo_bulk_t7 = pd.DataFrame()
        for cell_type in cell_types_to_use:
            # get the cre expression for the cell type cells
            cre_expression_cell_type = cre_expression[celltypes == cell_type]
            t7_expression_cell_type = t7_expression[celltypes == cell_type] if t7_expression is not None else None
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
            if t7_expression_cell_type is not None:
                # get the t7 expression for the cell type
                t7_samples = t7_expression_cell_type.values[bootstrap_indices].sum(axis=1)
                t7_samples = pd.DataFrame(t7_samples, index=[cell_type + ':sample_' + str(i) for i in range(pseudo_bulk_number)], columns=t7_expression_cell_type.columns)
                # add the pseudo bulk to the pseudo bulk t7 dataframe
                pseudo_bulk_t7 = pd.concat([pseudo_bulk_t7, t7_samples])
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
        if t7_expression is not None:
            pseudo_bulk_adata.obsm['T7CRE'] = pseudo_bulk_t7
        # perform glm test on the pseudo bulk
        result = glm(pseudo_bulk_adata, variate=variate, cell_types_to_use=cell_types_to_use, CREs=pseudo_bulk.columns,
                     norm_by_volm=norm_by_volm, volm_covariate=volm_covariate, rna_covariate=rna_covariate,
                     fov_covariate=False, filter_infected_cells=False, 
                     positive_x_or_y=positive_x_or_y,
                     only_keep_positive_x=only_keep_positive_x,
                     only_keep_positive_y=only_keep_positive_y,
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
        if 'T7CRE' in self.adata.obsm.keys():
            non_infected_cells = ((self.adata.obsm['T7CRE'] > 0).sum(axis=1) == 0).sum()
        else:
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
        # RNA info
        gene_info = self.adata.var.copy()
        gene_info['modality'] = 'Gene Expression'
        # CRE info, stored as ATAC peaks
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
        # T7 info, stored as Protein
        if 'T7CRE' in self.adata.obsm.keys():
            t7cre = self.adata.obsm['T7CRE'].copy()
            # set index of T7CRE
            t7cre.index = pd.Series(t7cre.index) + '_paired'
            adata_mvi.obsm['T7CRE'] = t7cre
            adata_mvi.uns['T7CRE_info'] = self.adata.uns['CRE_info'].copy()
        # set up model
        SCVIMODEL.setup_anndata(adata_mvi, batch_key="modality", 
                                protein_expression_obsm_key='T7CRE' if 'T7CRE' in adata_mvi.obsm.keys() else None,
                                protein_names_uns_key='T7CRE_info' if 'T7CRE' in adata_mvi.obsm.keys() else None,)
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
            adata_mvi.obsm['CRE'] = access
            adata_mvi.obsm['infect_rate'] = infect_rate
            adata_mvi.obsm['lib_size'] = lib_size_estimate
            if model_args['infection_rate_type'] == 'gene':
                adata_mvi.uns['infection_rate_gene'] = torch.nn.functional.softplus(model.module.infection_rate_gene).detach().cpu().numpy()
        elif use_model == 'SCVI':
            access = model.get_normalized_expression()
            adata_mvi.obsm['X_scvi'] = access[adata_mvi.var.index[adata_mvi.var['modality'] == 'Gene Expression']]
            adata_mvi.obsm['CRE'] = access[adata_mvi.var.index[adata_mvi.var['modality'] == 'Peaks']]
        # add uns to adata_mvi
        adata_mvi.uns['model'] = use_model
        adata_mvi.uns['model_args'] = model_args
        for k, v in self.adata.uns.items():
            if k not in adata_mvi.uns.keys():
                adata_mvi.uns[k] = v.copy()
        for k, v in self.adata.obsm.items():
            if k not in adata_mvi.obsm.keys() and k not in ['X_raw', 'CRE']:
                adata_mvi.obsm[k] = v.copy()
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
                      filter_by_negative_control_z_threshold=None,
                      attr_to_use='atac_cpm') -> tuple[pd.DataFrame, pd.DataFrame]:
        # filter atac_cpm and activity_df by cell_types_to_use
        if cell_types_to_use is not None:
            # first transform the cell_types_to_use as np.array
            cell_types_to_use = pd.Series(cell_types_to_use)
            cell_types_to_use = cell_types_to_use[cell_types_to_use.isin(getattr(self, attr_to_use).index)]
            cell_types_to_use = cell_types_to_use[cell_types_to_use.isin(acvitity_df.index)]
        else:
            cell_types_to_use = getattr(self, attr_to_use).index
            cell_types_to_use = cell_types_to_use[cell_types_to_use.isin(acvitity_df.index)]
        atac_cpm = getattr(self, attr_to_use).loc[cell_types_to_use]
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
    
    def plot_genomespy(self, cell_types_to_use, cre, padding=5000):
        bw_list = pd.read_csv('Data/ATAC/bw_list.csv', index_col=0)
        # set index as cell types
        bw_list = bw_list.set_index('celltype')
        tracks = {}
        for cell_type in cell_types_to_use:
            if cell_type in bw_list.index:
                bw_file = bw_list.loc[cell_type]['path']
                tracks[cell_type] = {
                    "path": f'Data/ATAC/wmb_bigwig/subclass_macs2/{bw_file}',
                    "height": 40,
                    "type": "bigwig"
                }
        # get cre chrom start and end
        cre_info = self.get_creinfo()
        start = cre_info.loc[cre]['Start']
        end = cre_info.loc[cre]['End']
        chrom = cre_info.loc[cre]['Chrom']
        plot = igv(tracks, region={"chrom": chrom, "start": int(start)-padding, "end": int(end)+padding}, server_port=18089)
        return plot
    
    def plot_pygenometracks(self, cell_types_to_use, cre, mod, outFileName, region=None, show_gene=True,
                            activity_df=None, padding=2500, nbins=700, max=None, min=None, width=80, height=80):
        # order the cell_types_to_use by name order
        cluster_annotation_term = pd.read_csv('Data/abc_atlas/cluster_annotation_term.csv', index_col=0)
        cluster_annotation_term['subclass'] = cluster_annotation_term['subclass'].str.replace('/', '-')
        cell_types_to_use_cluster_number = cluster_annotation_term['subclass_number'].groupby(cluster_annotation_term['subclass']).first().loc[cell_types_to_use].values
        # reorder cell types to use by subcluster number
        cell_types_to_use = pd.Index(cell_types_to_use[np.argsort(cell_types_to_use_cluster_number)])
        track_file='tmp.ini'
        available_tracks = PlotTracks.get_available_tracks()
        bw_list = pd.read_csv(f'Data/{mod}_meta.csv', index_col=0)
        # set index as cell types
        bw_list = bw_list.set_index('celltype')
        out = open(track_file, 'w')
        out.write(f"""
[x-axis]
# optional
fontsize = {0.4 * width}
# default is bottom meaning below the axis line
where = top

[spacer]
# height of space in cm (optional)
height = {height / 160}
""")
        if show_gene:
            out.write(
f"""
[gtf]
file = /share/vault/Users/gz2294/Data/gencode.v48.annotation.gtf.gz
height = 10
title = gene
merge_transcripts = true
prefered_name = gene_name
fontsize = {0.3 * width}
file_type = bed""")
        cell_types_in_plot = []
        for i, cell_type in enumerate(cell_types_to_use):
            track_added = False
            if cell_type in bw_list.index:
                bw_file = bw_list.loc[cell_type]['path']
                file_h = open(f'{bw_file}', 'r')
                cell_types_in_plot.append(cell_type)
            else:
                continue
            for track_type, track_class in available_tracks.items():
                for ending in track_class.SUPPORTED_ENDINGS:
                    if file_h.name.endswith(ending):
                        default_values = track_class.OPTIONS_TXT
                        default_values = default_values.replace("title =", f"title = {cell_type}")
                        # replace color to user selected color
                        default_values = default_values.replace("color = #666666", f"color = {self.adata.uns['cmap'][i % len(self.adata.uns['cmap'])]}")
                        # replace number_of_bins to user selected number of bins
                        default_values = default_values.replace("number_of_bins = 700", f"number_of_bins = {nbins}")
                        # if max or min is not None, replace them
                        if min is not None:
                            default_values = default_values.replace("min_value = 0", f"min_value = {min}")
                        if max is not None:
                            default_values = default_values.replace("#max_value = auto", f"max_value = {max}")
                        out.write(f"\n[{cell_type}]\nfile = {file_h.name}\n{default_values}")
                        track_added = True
            if track_added is False:
                sys.stdout.write(f"WARNING: file format not recognized for: {file_h.name}\n")
        # close the file
        out.close()
        # Identify the regions to plot from regions: the format is chr:start-end
        if cre is not None:
            region_chrom = self.get_creinfo().loc[cre, 'Chrom']
            region_start = self.get_creinfo().loc[cre, 'Start']
            region_end = self.get_creinfo().loc[cre, 'End']
            regions = f'{region_chrom}:{int(region_start)-padding}-{int(region_end)+padding}'
        else:
            assert region is not None, "Please provide a region to plot."
            region_chrom, region_start, region_end = region
            regions = f'{region_chrom}:{int(region_start)-padding}-{int(region_end)+padding}'
        regions = [get_region(regions)]

        if len(regions) == 0:
            raise ValueError("There is no valid regions to plot.")
        
        tracks = open(track_file, 'r')
        dpi = 500
        trackLabelFraction = 0.1
        trackLabelHAlign = 'left'
        plotWidth = width * 0.5 # width of the plot
        decreasingXAxis = False
        title = None
        fontSize=0.3 * width
        # Create all the tracks
        trp = PlotTracks(tracks.name, fig_width=width, fig_height=height,
                         fontsize=fontSize, dpi=dpi,
                         track_label_width=trackLabelFraction,
                         plot_regions=regions, plot_width=plotWidth)

        # Create dir if dir does not exists:
        # Modified from https://stackoverflow.com/questions/12517451/automatically-creating-directories-with-file-output
        os.makedirs(os.path.dirname(os.path.abspath(outFileName)), exist_ok=True)

        # Plot them
        # if activity_df is not None, we add highlight_region_height
        if activity_df is not None:
            # get the activity for the cre
            activity = activity_df.loc[cell_types_in_plot, cre].values
            # get the min and max of the activity
            activity_min = np.nanmin(activity)
            activity_max = np.nanmax(activity)
            # normalize the activity to 0-1
            activity_norm = (activity - activity_min) / (activity_max - activity_min)
            # set highlight_region_height to 0.5 * height
            highlight_region_height = activity_norm
            # turn into a dictionary
            highlight_region_height = {cell_type: activity for cell_type, activity in zip(cell_types_in_plot, highlight_region_height)}
        else:
            highlight_region_height = None
        current_fig = trp.plot(outFileName, *regions[0], title=title,
                               highlight_region=(int(region_start), int(region_end)),
                               highlight_region_height=highlight_region_height,
                               h_align_titles=trackLabelHAlign, remove_y_axis=max is not None,
                               decreasing_x_axis=decreasingXAxis)
        plt.close(current_fig)
        # remove the track file
        os.remove(track_file)
        trp.close_files()
    
    def get_celltypes_peaks_close_to_cre(self, cell_types_to_use, cre, range=10000) -> pd.DataFrame:
        cre_chrom = self.get_creinfo().loc[cre]['Chrom']
        cre_start = int(self.get_creinfo().loc[cre]['Start'])
        cre_end = int(self.get_creinfo().loc[cre]['End'])
        # get the cell type files
        peak_list = pd.read_csv('Data/ATAC/peak_list.csv', index_col=0)
        # set index as cell types
        peak_list = peak_list.set_index('celltype')
        result = pd.DataFrame()
        for cell_type in cell_types_to_use:
            if cell_type in peak_list.index:
                peak_file = peak_list.loc[cell_type]['path']
                peak_file = pd.read_csv(f'Data/ATAC/subclass2CRE/{peak_file}', header=None)
                peak_file['Chromosome'] = peak_file[0].str.split(':').str[0]
                peak_file['Start'] = peak_file[0].str.split(':').str[1].str.split('-').str[0].astype(int)
                peak_file['End'] = peak_file[0].str.split(':').str[1].str.split('-').str[1].astype(int)
                # set peak_file column 0 to another name
                peak_file = peak_file.rename(columns={0: 'Peak'})
                # reorder columns
                peak_file = peak_file[['Chromosome', 'Start', 'End', 'Peak']]
                # filter the cell type peaks by range
                peak_file = peak_file[(peak_file['Chromosome'] == cre_chrom) & 
                                      (peak_file['Start'] >= cre_start - range) & 
                                      (peak_file['End'] <= cre_end + range)]
                peak_file['celltype'] = cell_type
                # append to result
                result = pd.concat([result, peak_file])
                # reset the index by dropping the index
                if not result.empty:
                    result = result.reset_index(drop=True)
        # write to bed file, and query motifs
        peak_bed = 'tmp.bed'
        # write ['Chromosome', 'Start', 'End'] to bed file
        result[['Chromosome', 'Start', 'End']].to_csv(peak_bed, sep='\t', header=False, index=False)
        # query the motifs
        peaks_motif = query_motif(peak_bed, f"{PWD}/Data/annotation/mm10.archetype_motifs.v1.0.bed.gz")
        get_motif_output = get_motif(peak_bed, peaks_motif, assembly='mm10')
        # Read the peak motif bed file
        peak_motif = pd.read_csv(
            get_motif_output,
            sep="\t",
            header=None,
            names=["Chromosome", "Start", "End", "Motif_cluster", "Score"],
        )
        # Pivot the data
        peak_motif_pivoted = peak_motif.pivot_table(
            index=["Chromosome", "Start", "End"],
            columns="Motif_cluster",
            values="Score",
            fill_value=0,
        )
        peak_motif_pivoted.reset_index(inplace=True)
        # Create the 'Name' column
        peak_motif_pivoted["Name"] = peak_motif_pivoted.apply(
            lambda x: f'{x["Chromosome"]}:{x["Start"]}-{x["End"]}', axis=1
        )
        peak_motif_pivoted = peak_motif_pivoted.drop(columns=["Chromosome", "Start", "End"])
        # Read the original peak bed file
        original_peaks = pd.read_csv(
            peak_bed, sep="\t", header=None, names=["Chromosome", "Start", "End", "Score"]
        )
        # exclude chrM and chrY
        original_peaks = original_peaks[~original_peaks.Chromosome.isin(["chrM", "chrY"])]
        original_peaks["Name"] = original_peaks.apply(
            lambda x: f'{x["Chromosome"]}:{x["Start"]}-{x["End"]}', axis=1
        )
        # Merge the pivoted data with the original peaks
        merged_data = pd.merge(original_peaks, peak_motif_pivoted, on="Name", how="left")
        # Fill NaN values with 0 for motif columns
        motif_columns = [
            col
            for col in merged_data.columns
            if col not in ["Chromosome", "Start", "End", "Score", "Name"]
        ]
        merged_data[motif_columns] = merged_data[motif_columns].fillna(0)
        # combine merged_data with result
        result = pd.concat([result, merged_data.iloc[:, 5:]], axis=1)
        # remove get_motif_output, peaks_motif, peak_bed
        os.remove(get_motif_output)
        os.remove(peak_bed)
        os.remove(peaks_motif)
        return result
    
    def motif_enrichment_test(self, cell_types_to_use, cres_to_use, activity_df, bootstrap_number=1000) -> dict:
        motif_scores = pd.read_csv('results/CRE_motif.csv', index_col=0)
        # only keep 4:
        motif_scores = motif_scores[motif_scores.columns[5:]]
        if cell_types_to_use is None:
            cell_types_to_use = activity_df.index
        if cres_to_use is None:
            cres_to_use = activity_df.columns.intersection(motif_scores.index)
        # filter activity_df by cell_types_to_use
        activity_df = activity_df.loc[cell_types_to_use, cres_to_use]
        # for each cell type, calculate the motif_enrichment
        motif_enrichment_p = pd.DataFrame(index=cell_types_to_use, columns=motif_scores.columns)
        motif_enrichment_q = pd.DataFrame(index=cell_types_to_use, columns=motif_scores.columns)
        motif_enrichment_es = pd.DataFrame(index=cell_types_to_use, columns=motif_scores.columns)
        motif_enrichment_bg = np.ndarray((1000, motif_enrichment_p.shape[0], motif_enrichment_p.shape[1]))
        with multiprocessing.Pool(processes=min(256, int(multiprocessing.cpu_count()))) as pool:
            for i, cell_type in enumerate(cell_types_to_use):
                print(f'Processing cell type {cell_type} ({i+1}/{len(cell_types_to_use)})')
                # get the activity of the cell type
                activity = activity_df.loc[cell_type]
                # get the motif scores of the cell type, rank by activity
                cre_sorted = activity.sort_values(ascending=False).index
                motif_score_cell_type = motif_scores.loc[cre_sorted].copy()
                # calculate the enrichment
                enrichment_score = motif_enrichment(motif_score_cell_type)
                # shuffle and calculate the background, do parallel
                bgs = pool.starmap(
                    motif_enrichment,
                    [(motif_score_cell_type.sample(frac=1, replace=False, random_state=i),) for i in range(bootstrap_number)], 
                )
                motif_enrichment_es.loc[cell_type] = enrichment_score
                for j, bg in enumerate(bgs):
                    motif_enrichment_bg[j, i, :] = bg
                # calculate the p-value
                p_value = np.zeros(motif_enrichment_es.shape[1])
                for j, score in enumerate(enrichment_score):
                    p_value[j] = np.sum(motif_enrichment_bg[:, i, j] >= score, axis=0) / bootstrap_number
                motif_enrichment_p.loc[cell_type] = p_value
                motif_enrichment_q.loc[cell_type] = multitest.multipletests(p_value, method='fdr_bh')[1]
        # return the results
        motif_enrichment_results = {
            'p_value': motif_enrichment_p,
            'q_value': motif_enrichment_q,
            'enrichment_score': motif_enrichment_es,
            'background': motif_enrichment_bg
        }
        return motif_enrichment_results
    
    def motif_enrichment_homer(self, cres_to_use, background_cres=None, outputdir=None, overwrite=False):
        if outputdir is None:
            outputdir = 'results/motif_enrichment_homer'
        # check if the outputdir exists, if not, run it
        if not os.path.exists(f'{outputdir}/homerResults.html') or overwrite:
            cre_info = self.get_creinfo()
            cre_info = cre_info[cre_info['labeling_type'] != 'negative control']
            cre_info['name'] = cre_info.index
            cre_info = cre_info[['name', 'Chrom', 'Start', 'End']]
            # get cres_to_use info
            cre_pos, cre_neg = cre_info.loc[cres_to_use].copy(), cre_info.loc[cres_to_use].copy()
            # duplicate and use both '+' and '-' strand
            cre_pos['Strand'] = '+'
            cre_pos['name'] = cre_pos['name'] + '_plus'
            cre_neg['Strand'] = '-'
            cre_neg['name'] = cre_neg['name'] + '_minus'
            cre = pd.concat((cre_pos, cre_neg), ignore_index=True)
            # if background_cres is not None, get background cres info
            if background_cres is not None:
                background_pos, background_neg = cre_info.loc[background_cres].copy(), cre_info.loc[background_cres].copy()
                background_pos['Strand'] = '+'
                background_pos['name'] = background_pos['name'] + '_plus'
                background_neg['Strand'] = '-'
                background_neg['name'] = background_neg['name'] + '_minus'
                background = pd.concat((background_pos, background_neg), ignore_index=True)
            # make output directory
            os.makedirs(outputdir, exist_ok=True)
            cre.to_csv(f'{outputdir}/cre.bed', sep='\t', index=False, header=False)
            if background_cres is not None:
                background.to_csv(f'{outputdir}/background.bed', sep='\t', index=False, header=False)
            # run homer findMotifsGenome.pl
            if background_cres is not None:
                cmd = f"export PERL5LIB=/share/vault/Users/gz2294/Homer/bin:$PERL5LIB; \
                        export PATH=/share/vault/Users/gz2294/Homer/bin:$PATH; \
                        findMotifsGenome.pl {outputdir}/cre.bed mm10 '{outputdir}' -bg {outputdir}/background.bed -p 128"
            else:
                cmd = f"export PERL5LIB=/share/vault/Users/gz2294/Homer/bin:$PERL5LIB; \
                        export PATH=/share/vault/Users/gz2294/Homer/bin:$PATH; \
                        findMotifsGenome.pl {outputdir}/cre.bed mm10 '{outputdir} -p 128"
            print(f'Running command: {cmd}')
            result = os.system(cmd)
            if result == 0:
                print(f'Motif enrichment completed, results saved to {outputdir}')
            else:
                print('Motif enrichment failed, please check the command and the input files.')
                return None
        # read the homer results
        def modify_motif_name(motifnm):
            motifnm = re.sub(r"\(.*", "", motifnm)
            motifnm = re.sub(r"COUP-TFII", "NR2F2", motifnm)
            motifnm = re.sub(r"-distal", "", motifnm)
            motifnm = re.sub(r"\+.*", "", motifnm)
            motifnm = re.sub(r"-AP1", "", motifnm)
            motifnm = re.sub(r"NF-E2", "Nfe2", motifnm)
            motifnm = re.sub(r"-halfsite", "", motifnm)
            motifnm = re.sub(r"n-Myc", "Mycn", motifnm)
            motifnm = re.sub(r"c-Myc", "Myc", motifnm)
            motifnm = re.sub(r"Nkx2\.1", "Nkx2-1", motifnm)
            motifnm = re.sub(r"Nkx2\.2", "Nkx2-2", motifnm)
            motifnm = re.sub(r"Nkx2\.5", "Nkx2-5", motifnm)
            motifnm = re.sub(r"Nkx3\.1", "Nkx3-1", motifnm)
            motifnm = re.sub(r"Nkx6\.1", "Nkx6-1", motifnm)
            motifnm = re.sub(r"\+il21", "", motifnm)
            motifnm = re.sub(r"HIF-1b", "Arnt", motifnm)
            motifnm = re.sub(r"HIF-1a", "Hif1a", motifnm)
            motifnm = re.sub(r"\+1bp", "", motifnm)
            motifnm = re.sub(r"OCT:OCT-short", "OCT", motifnm)
            motifnm = re.sub(r"OCT:OCT", "OCT", motifnm)
            motifnm = re.sub(r"RBPJ:Ebox", "Rbpj", motifnm)
            motifnm = re.sub(r"PU\.1", "Spi1", motifnm)
            motifnm = re.sub(r"ZNF143\|STAF", "Zfp143", motifnm)
            motifnm = re.sub(r"NFkB2-p52", "NFkb2", motifnm)
            motifnm = re.sub(r"NFkB-p65", "Rela", motifnm)
            motifnm = re.sub(r"AP-2alpha", "Tfap2a", motifnm)
            return motifnm
        if os.path.exists(f'{outputdir}/knownResults.txt'):
            homer_res = pd.read_csv(f'{outputdir}/knownResults.txt', sep='\t')
            # modify the homer names
            homer_res['gene'] = homer_res['Motif Name'].apply(modify_motif_name)
            return homer_res
        else:
            print('Homer results not found, please check the output directory.')
            return None
    
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
    
    def negbiom_cmdstanpy(self, cell_types_to_use, cres_to_use, 
                         stan_model_bg='NegBinom.stan', stan_model_main='NegBinom2.stan',
                         chains=1, iter_warmup=1000, iter_sampling=2000, 
                         n_jobs=None):
        """
        Negative binomial Bayesian analysis using CmdStanPy (Python port of R cmdstanr workflow)
        
        Parameters:
        -----------
        rna_file : str, path to RNA transcript counts CSV
        df_file : str, path to negative control transcript counts CSV  
        lib_file : str, path to library size CSV
        df_all_file : str, path to all element transcript counts CSV
        lib_all_file : str, path to all element library sizes CSV
        atac_file : str, path to ATAC data CSV
        stan_model_bg : str, path to background Stan model
        stan_model_main : str, path to main Stan model
        chains : int, number of MCMC chains
        iter_warmup : int, warmup iterations
        iter_sampling : int, sampling iterations
        n_jobs : int, number of parallel jobs (None = all cores)
        output_file : str, output pickle file path
        """
        if n_jobs is None:
            n_jobs = os.cpu_count() * 0.8  # Use 80% of available cores
        config = {
            'cell_types_to_use': cell_types_to_use,
            'cres_to_use': cres_to_use,
            'stan_model_bg': stan_model_bg,
            'stan_model_main': stan_model_main,
            'chains': chains,
            'iter_warmup': iter_warmup,
            'iter_sampling': iter_sampling,
        }
        # Check if results already exist
        if hasattr(self, 'negbiom_results') and hasattr(self, 'negbiom_configs'):
            for stored_config, negbiom_result in zip(self.negbiom_configs, self.negbiom_results):
                if stored_config == config:
                    # If results already exist, return them
                    print('Results already exist, returning stored results')
                    return negbiom_result.copy()
        # Load data
        if cell_types_to_use is not None:
            cre_cells_expression, rna_cells_expression, cell_types_to_use = self.get_cre_rna_celltypes(cell_types_to_use)
        else:
            cre_cells_expression = self.get_cre_expression().copy()
            rna_cells_expression = self.get_rna_expression().copy()
            cell_types_to_use = self.get_celltypes()
        if cres_to_use is not None:
            cre_cells_expression = cre_cells_expression[cres_to_use]
        fdc_df = pd.DataFrame(index=cell_types_to_use.unique(), columns=cre_cells_expression.columns)
        ess_df = pd.DataFrame(index=cell_types_to_use.unique(), columns=cre_cells_expression.columns)
        bkg_df = pd.DataFrame(index=cell_types_to_use.unique(), columns=['mean_x_mean', 'beta_x_mean', 'mean_x_std', 'beta_x_std'])
        # First run all background models in parallel
        def run_background_model(cell_type):
            """Run background model for a single cell type"""
            df_all = cre_cells_expression[cell_types_to_use == cell_type].copy()
            rna = rna_cells_expression[cell_types_to_use == cell_type].copy().sum(axis=1)
            # fit negative control elements first
            df = df_all[self.get_negative_control_cres()].copy()
            # Add row-wise sum as additional column
            df['row_sum'] = df.sum(axis=1)
            N = len(df)  # number of cells
            E = len(df.columns)  # number of negative control elements (including row_sum)
            # Prepare library sizes with sum of all negative controls
            lib_sizes = self.lib_size.loc[self.get_negative_control_cres(), 'counts'].astype(float)
            lib_sum = lib_sizes.sum()
            lib_sizes_extended = np.append(lib_sizes.values, lib_sum)
            # Prepare data for background model
            stan_data_bg = {
                'N': N,
                'E': E,
                'x': df.values.astype(int),
                'xx': rna.astype(float),
                'L': lib_sizes_extended
            }
            # Compile model once
            model_bg = CmdStanModel(stan_file=stan_model_bg)
            # Iterative fitting until ESS > 1000
            max_iterations = 2
            iteration = 0
            ess_threshold = 500
            while iteration < max_iterations:
                iteration += 1
                # Run background model
                fit_bg = model_bg.sample(
                    data=stan_data_bg,
                    chains=chains,
                    iter_warmup=iter_warmup,
                    iter_sampling=iter_sampling,
                    show_progress=False
                )
                # Extract background parameters
                bg_summary = fit_bg.summary()
                mean_x_mean = bg_summary.loc['mean_x', 'Mean']
                beta_x_mean = bg_summary.loc['beta_x', 'Mean']
                mean_x_std = bg_summary.loc['mean_x', 'StdDev']
                beta_x_std = bg_summary.loc['beta_x', 'StdDev']
                # Check ESS for all parameters
                if 'N_Eff' not in bg_summary.columns:
                    print(f"Cell type {cell_type}: N_Eff not available, continuing sampling...")
                    continue
                ess_mean_x = bg_summary.loc['mean_x', 'N_Eff']
                ess_beta_x = bg_summary.loc['beta_x', 'N_Eff']
                min_ess = min(ess_mean_x, ess_beta_x)
                if min_ess >= ess_threshold:
                    print(f"Cell type {cell_type}: Converged after {iteration} iterations (min ESS: {min_ess:.0f})")
                    break
                else:
                    print(f"Cell type {cell_type}: Iteration {iteration}, min ESS: {min_ess:.0f} < {ess_threshold}, refitting...")
            if 'N_Eff' in bg_summary.columns and min_ess < ess_threshold:
                print(f"Cell type {cell_type}: Warning - Did not converge after {max_iterations} iterations (final min ESS: {min_ess:.0f})")
            elif 'N_Eff' not in bg_summary.columns:
                print(f"Cell type {cell_type}: Warning - N_Eff not available, completed {max_iterations} iterations")
            return cell_type, {
                'mean_x_mean': mean_x_mean,
                'beta_x_mean': beta_x_mean,
                'mean_x_std': mean_x_std,
                'beta_x_std': beta_x_std,
                'background': bg_summary,
                'final_ess': min_ess if 'N_Eff' in bg_summary.columns else None,
                'iterations': iteration
            }
        print(f"Running background models for {len(cell_types_to_use.unique())} cell types in parallel...")
        background_results = Parallel(n_jobs=min(n_jobs, len(cell_types_to_use.unique())))(
            delayed(run_background_model)(cell_type) 
            for cell_type in cell_types_to_use.unique()
        )
        # Store background results
        background_params = {}
        for cell_type, bg_result in background_results:
            background_params[cell_type] = bg_result
            bkg_df.loc[cell_type] = {
                'mean_x_mean': bg_result['mean_x_mean'],
                'beta_x_mean': bg_result['beta_x_mean'],
                'mean_x_std': bg_result['mean_x_std'],
                'beta_x_std': bg_result['beta_x_std']
            }
        # Now run element models for each cell type
        for cell_type in cell_types_to_use.unique():
            df_all = cre_cells_expression[cell_types_to_use == cell_type].copy()
            rna = rna_cells_expression[cell_types_to_use == cell_type].copy().sum(axis=1)
            # Get background parameters for this cell type
            bg_params = background_params[cell_type]
            results = {'background': bg_params['background']}
            # Load all elements data
            N_all = len(df_all)
            def run_single_element(col_name):
                """Run Stan model for a single element with ESS checking"""
                stan_data = {
                    'N': N_all,
                    'E': 1,
                    'x': df_all[col_name].values.astype(int).reshape(-1, 1),
                    'xx': rna.astype(float),
                    'L': np.array([self.lib_size.loc[col_name, 'counts']], dtype=float),
                    'mean_x_mean': bg_params['mean_x_mean'],
                    'beta_x_mean': bg_params['beta_x_mean'],
                    'mean_x_std': bg_params['mean_x_std'],
                    'beta_x_std': bg_params['beta_x_std']
                }
                # Compile model once
                model = CmdStanModel(stan_file=stan_model_main)
                # Iterative fitting until ESS > 1000
                max_iterations = 5
                iteration = 0
                ess_threshold = 500
                while iteration < max_iterations:
                    iteration += 1
                    # Run model
                    fit = model.sample(
                        data=stan_data,
                        chains=chains,
                        iter_warmup=iter_warmup,
                        iter_sampling=iter_sampling,
                        show_progress=False
                    )
                    # Check ESS for fold_x parameter
                    summary = fit.summary()
                    try:
                        fold_x_ess = summary.loc['fold_x', 'N_Eff']
                        if fold_x_ess >= ess_threshold:
                            return col_name, summary
                        else:
                            if iteration < max_iterations:
                                continue  # Try again
                            else:
                                # Return result even if ESS is low after max iterations
                                return col_name, summary
                    except KeyError:
                        # fold_x parameter not found, return summary anyway
                        return col_name, summary
                return col_name, summary
            # Run models in parallel
            print(f"Running {len(df_all.columns)} element models in parallel...")
            parallel_results = Parallel(n_jobs=n_jobs)(
                delayed(run_single_element)(col_name) 
                for col_name in df_all.columns
            )
            # Collect results
            for col_name, summary in parallel_results:
                results[col_name] = summary
            # Quality check (similar to R version)
            for col_name in df_all.columns:
                if col_name in results:
                    summary = results[col_name]
                    try:
                        fold_x_row = summary.loc['fold_x']
                        ess_val = fold_x_row['N_Eff']
                        ess_df.loc[cell_type, col_name] = ess_val
                        fdc_df.loc[cell_type, col_name] = fold_x_row['Mean']
                    except KeyError:
                        # fold_x parameter not found in summary
                        pass
            # Background parameters already stored above
        res = {
            'ess': ess_df,
            'fdc': fdc_df,
            'background': bkg_df,
            'background_results': background_results,
            'results': results
        }
        # save results to attribute
        if not hasattr(self, 'negbiom_results') or not hasattr(self, 'negbiom_configs'):
            self.negbiom_results = []
            self.negbiom_configs = []
        self.negbiom_results.append(res)
        self.negbiom_configs.append(config)
        return res
    
    def poisson_neg_binom_mle_all(self, cell_types_to_use=None, cres_to_use=None):
        # Implement the Poisson negative binomial MLE estimation here
        # Load data
        if cell_types_to_use is not None:
            cre_cells_expression, _, cell_types_to_use = self.get_cre_rna_celltypes(cell_types_to_use)
        else:
            cre_cells_expression = self.get_cre_expression().copy()
            cell_types_to_use = self.get_celltypes()
        if cres_to_use is not None:
            cre_cells_expression = cre_cells_expression[cres_to_use]
        else:
            cres_to_use = cre_cells_expression.columns
        t7_cells_expression = self.get_t7_expression()
        assert t7_cells_expression is not None, "T7 expression data is not available."
        t7_cells_expression = t7_cells_expression.loc[cre_cells_expression.index, cre_cells_expression.columns]
        # categorize cell_types_to_use
        unique_cell_types, cell_types_to_use_idx = np.unique(cell_types_to_use, return_inverse=True)
        # define the optimize function
        def poisson_negative_binomial(x, obs_cre, obs_t7, n_cres, n_celltypes, cell_types_to_use_idx):
            # parameters
            # x[0:n_cres*n_celltypes]: infection poisson lambda
            # x[n_cres*n_celltypes: 2*n_cres*n_celltypes], x[2*n_cres*n_celltypes: 3*n_cres*n_celltypes]: CRE activity mean, dispersion
            # x[3*n_cres*n_celltypes]: T7 detection 0 log probability
            # x[3*n_cres*n_celltypes+1]: CRE detection 0 log probability
            
            # Convert to numpy arrays if needed
            obs_cre = obs_cre.values if hasattr(obs_cre, 'values') else obs_cre
            obs_t7 = obs_t7.values if hasattr(obs_t7, 'values') else obs_t7
            
            # Transform x using BLAS-optimized reshape operations
            poisson_lambda = x[0:n_cres*n_celltypes].reshape((n_cres, n_celltypes))
            mu = x[n_cres*n_celltypes:2*n_cres*n_celltypes].reshape((n_cres, n_celltypes))
            disp = x[2*n_cres*n_celltypes:3*n_cres*n_celltypes].reshape((n_cres, n_celltypes))
            t7_drop = x[3*n_cres*n_celltypes]
            cre_detect = x[3*n_cres*n_celltypes + 1]
            
            # Pre-compute constants
            i_values = np.arange(66, dtype=np.float64)  # Use float64 for BLAS optimization
            exp_cre_detect = np.exp(cre_detect)
            
            # Use advanced indexing with BLAS-optimized operations
            # Get parameters for all CREs and cells using matrix operations
            lambda_selected = poisson_lambda[:, cell_types_to_use_idx]  # [n_cres, n_cells]
            mu_selected = mu[:, cell_types_to_use_idx]  # [n_cres, n_cells] 
            disp_selected = disp[:, cell_types_to_use_idx]  # [n_cres, n_cells]
            
            # Reshape for broadcasting - use C-contiguous arrays for BLAS
            lambda_all = np.ascontiguousarray(lambda_selected[:, :, None])  # [n_cres, n_cells, 1]
            mu_all = np.ascontiguousarray(mu_selected[:, :, None])  # [n_cres, n_cells, 1]
            disp_all = np.ascontiguousarray(disp_selected[:, :, None])  # [n_cres, n_cells, 1]
            i_expanded = np.ascontiguousarray(i_values[None, None, :])  # [1, 1, 66]
            
            # Fast manual Poisson log-probability calculation
            # logpmf(k, mu) = k*log(mu) - mu - log(k!)
            # Pre-compute log factorials for i=0 to 65
            log_factorials = np.cumsum(np.log(np.maximum(np.arange(1, 67), 1)))
            log_factorials = np.concatenate([[0], log_factorials[:-1]])  # log(0!) = 0
            log_fact_expanded = log_factorials[None, None, :]  # [1, 1, 66]
            
            # Vectorized Poisson log probabilities - MUCH faster than scipy
            ll = i_expanded * np.log(np.maximum(lambda_all, 1e-10)) - lambda_all - log_fact_expanded
            
            # Pre-compute observation arrays
            obs_t7_expanded = np.ascontiguousarray(obs_t7[:, :, None])  # [n_cres, n_cells, 1]
            obs_cre_expanded = np.ascontiguousarray(obs_cre[:, :, None])  # [n_cres, n_cells, 1]
            
            # Pre-compute masks for vectorized operations
            i_is_zero = (i_expanded == 0)
            i_nonzero = (i_expanded > 0)
            
            # Vectorized probability calculations using BLAS operations
            # Use np.where for conditional operations that leverage BLAS
            one_minus_obs_t7 = 1 - obs_t7_expanded
            
            # T7 terms using vectorized operations
            t7_prob_zero = np.where(i_is_zero, np.log(one_minus_obs_t7), 0)
            cre_prob_zero = np.where(i_is_zero, np.log(obs_cre_expanded == 0), 0)
            
            # T7 terms for i>0 using BLAS-optimized matrix operations
            t7_term1 = np.where(i_nonzero, one_minus_obs_t7 * i_expanded * t7_drop, 0)
            exp_term = np.exp(i_expanded * t7_drop)
            t7_term2 = np.where(i_nonzero, obs_t7_expanded * np.log(1 - exp_term), 0)
            
            # Optimized negative binomial calculation using BLAS operations
            # Pre-compute parameters using matrix operations
            mu_exp_cre = mu_all * exp_cre_detect  # BLAS-optimized multiplication
            n_param = i_expanded * disp_all  # Element-wise but vectorized
            p_param = disp_all / (disp_all + mu_exp_cre)  # BLAS-optimized division
            
            # Fast manual negative binomial log-probability calculation
            # nbinom.logpmf(k, n, p) = log(Γ(k+n)) - log(Γ(n)) - log(k!) + n*log(p) + k*log(1-p)
            # For integer n: log(Γ(k+n)) - log(Γ(n)) = log((k+n-1)!) - log((n-1)!) = sum(log(n+i)) for i=0 to k-1
            
            # Pre-compute for vectorized operation
            k_vals = obs_cre_expanded  # [n_cres, n_cells, 1]
            n_vals = n_param  # [n_cres, n_cells, 66]
            p_vals = p_param  # [n_cres, n_cells, 66]
            
            # Compute log(k!) using pre-computed factorials
            k_int = np.clip(k_vals.astype(int), 0, 65)  # Clip to valid range
            log_k_fact = log_factorials[k_int.flatten()].reshape(k_vals.shape)
            
            # For negative binomial with integer n, use gamma function identity
            # Use scipy.special.gammaln for numerical stability
            from scipy.special import gammaln
            log_gamma_k_plus_n = gammaln(k_vals + n_vals)
            log_gamma_n = gammaln(n_vals)
            
            # Vectorized negative binomial log probability
            nbinom_ll_full = (log_gamma_k_plus_n - log_gamma_n - log_k_fact + 
                             n_vals * np.log(np.maximum(p_vals, 1e-10)) + 
                             k_vals * np.log(np.maximum(1 - p_vals, 1e-10)))
            
            # Apply mask for i>0 cases only
            nbinom_ll = np.where(i_nonzero, nbinom_ll_full, 0)
            
            # Combine all terms using BLAS-optimized array operations
            ll = ll + t7_prob_zero + cre_prob_zero + t7_term1 + t7_term2 + nbinom_ll
            
            # Use scipy.special.logsumexp for numerical stability and BLAS optimization
            ll_summed = logsumexp(ll, axis=2)  # [n_cres, n_cells]
            
            # Final sum using BLAS-optimized operation
            return -np.sum(ll_summed)
        # make initial guess
        cre_celltype_mean = cre_cells_expression.groupby(cell_types_to_use).mean().loc[unique_cell_types, cres_to_use].T
        cre_celltype_var = cre_cells_expression.groupby(cell_types_to_use).var().loc[unique_cell_types, cres_to_use].T
        cre_celltype_disp = cre_celltype_mean**2 / (cre_celltype_var - cre_celltype_mean)
        infection_rate = -np.log((t7_cells_expression == 0).groupby(cell_types_to_use).mean().loc[unique_cell_types, cres_to_use].T)
        initial_guess = np.concatenate([infection_rate.values.flatten(), cre_celltype_mean.values.flatten(), cre_celltype_disp.values.flatten(), 
                                        [np.log(0.05)], [np.log(0.05)]])
        estimates = minimize(poisson_negative_binomial, initial_guess, args=(cre_cells_expression.values.T, t7_cells_expression.values.T, len(cres_to_use), len(unique_cell_types), cell_types_to_use_idx),
                             bounds=[(1e-8, None)]*len(unique_cell_types)*len(cres_to_use)*3 + [(None, -1e-10), (None, -1e-10)], method='L-BFGS-B')
        return estimates
        
    def poisson_neg_binom_mle_separate(self, cell_types_to_use=None, cres_to_use=None):
        # Implement the Poisson negative binomial MLE estimation here
        # Load data
        if cell_types_to_use is not None:
            cre_cells_expression, _, cell_types_to_use = self.get_cre_rna_celltypes(cell_types_to_use)
        else:
            cre_cells_expression = self.get_cre_expression().copy()
            cell_types_to_use = self.get_celltypes()
        if cres_to_use is not None:
            cre_cells_expression = cre_cells_expression[cres_to_use]
        else:
            cres_to_use = cre_cells_expression.columns
        t7_cells_expression = self.get_t7_expression()
        assert t7_cells_expression is not None, "T7 expression data is not available."
        t7_cells_expression = t7_cells_expression.loc[cre_cells_expression.index, cre_cells_expression.columns]
        # categorize cell_types_to_use
        unique_cell_types = np.unique(cell_types_to_use)
        # make initial guess
        cre_celltype_mean = cre_cells_expression.groupby(cell_types_to_use).mean().loc[unique_cell_types, cres_to_use]
        cre_celltype_var = cre_cells_expression.groupby(cell_types_to_use).var().loc[unique_cell_types, cres_to_use]
        cre_celltype_disp = cre_celltype_mean**2 / (cre_celltype_var - cre_celltype_mean)
        infection_rate = -np.log((t7_cells_expression == 0).groupby(cell_types_to_use).mean().loc[unique_cell_types, cres_to_use])
        # Prepare tasks for multiprocessing
        tasks = []
        for celltype in unique_cell_types:
            for cre in cres_to_use:
                obs_cre = cre_cells_expression[cell_types_to_use == celltype][cre].values
                obs_t7 = t7_cells_expression[cell_types_to_use == celltype][cre].values
                initial_guess = [infection_rate.loc[celltype, cre], np.log(0.05), np.log(0.05), cre_celltype_mean.loc[celltype, cre], cre_celltype_disp.loc[celltype, cre]]
                tasks.append((celltype, cre, obs_cre, obs_t7, initial_guess))
        
        # Run optimization tasks in parallel using joblib Parallel and delayed
        n_jobs = min(multiprocessing.cpu_count(), len(tasks))  # Use all available CPUs but not more than tasks
        
        print(f"Running {len(tasks)} optimization tasks across {n_jobs} jobs...")
        results = Parallel(n_jobs=n_jobs)(
            delayed(_optimize_celltype_cre_worker)(task) for task in tasks)
        
        # Store results in dataframes
        infection_rate_df = pd.DataFrame(index=unique_cell_types, columns=cres_to_use)
        mu_df = pd.DataFrame(index=unique_cell_types, columns=cres_to_use)
        disp_df = pd.DataFrame(index=unique_cell_types, columns=cres_to_use)
        
        for celltype, cre, infection_rate_val, mu_val, disp_val in results:
            infection_rate_df.loc[celltype, cre] = infection_rate_val
            mu_df.loc[celltype, cre] = mu_val
            disp_df.loc[celltype, cre] = disp_val
            
        return infection_rate_df, mu_df, disp_df