import logging
import multiprocessing
import os
import pickle
import re
import sys
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import scvi
import statsmodels.api as sm
import statsmodels.formula.api as smf
import torch
import torch.nn as nn
import torch.optim as optim
from cmdstanpy import CmdStanModel
from genomespy import igv
from joblib import Parallel, delayed
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from pydeseq2.dds import DeseqDataSet
from pydeseq2.default_inference import DefaultInference
from pydeseq2.ds import DeseqStats
from pygenometracks.utilities import get_region
from scipy import stats, optimize
from scipy.optimize import minimize
from scipy.special import logsumexp
from scipy.stats import linregress
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler
from statsmodels.stats import multitest
from torch.utils.tensorboard import SummaryWriter
from typing import Union, List, Literal

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=SyntaxWarning, module="docrep")

# Add current path to sys.path
PWD = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PWD)

from starr_fish_vae import STARRFISHVI
from tracksClass import PlotTracks
from get_preprocess_utils import get_motif, query_motif

cmdstanpy_logger = logging.getLogger("cmdstanpy")
cmdstanpy_logger.disabled = True


# Configuration for data file paths
class DataPaths:
    """Centralized configuration for data file paths."""
    CRE_ATAC_PEAKS = 'Data/cre_atac_peaks.csv'
    CRE_H3K27AC_PEAKS = 'Data/cre_h3k27ac_peaks.csv'
    CRE_H3K4ME1_PEAKS = 'Data/cre_h3k4me1_peaks.csv'
    CRE_CHROMATIN_STATE_A = 'Data/cre_chromatin_state_a.csv'
    CRE_CHROMATIN_STATE_O = 'Data/cre_chromatin_state_o.csv'


# Helper function to reduce code duplication in get_positive_control_* methods
def _load_csv_and_filter(file_path, key, axis='row', threshold=0.5):
    """
    Load CSV file and filter by threshold.

    Parameters:
    -----------
    file_path : str
        Path to CSV file
    key : str
        Row or column key to filter
    axis : str
        'row' to get columns where row[key] > threshold
        'col' to get rows where column[key] > threshold
    threshold : float
        Threshold value for filtering

    Returns:
    --------
    pd.Index or None
        Filtered index or None if key not found
    """
    data = pd.read_csv(file_path, index_col=0)

    if axis == 'row':
        if key not in data.index:
            return None
        row_data = data.loc[key]
        return row_data[row_data > threshold].index
    else:  # axis == 'col'
        if key not in data.columns:
            return None
        col_data = data[key]
        return col_data[col_data > threshold].index


def fit_glm(formula, y, x, volm, fov, rna, size, positive_x_or_y=True, only_keep_positive_x=False, only_keep_positive_y=False, transform_x_y=None, intercept_x=None, intercept_y=None):
    try:
        # remove zeros in y
        if positive_x_or_y:
            to_keep = (y > 0) | (x > 0)
        else:
            to_keep = np.ones(len(x), dtype=bool)
        if only_keep_positive_x or transform_x_y == 'log':
            to_keep &= x > 0
        if only_keep_positive_y or transform_x_y == 'log':
            to_keep &= y > 0
        y = y[to_keep]
        x = x[to_keep]
        if transform_x_y == 'log':
            x = np.log(x)
            y = np.log(y)
        elif transform_x_y == 'log1p':
            x = np.log1p(x)
            y = np.log1p(y)
        if intercept_x is not None:
            x -= intercept_x
        if intercept_y is not None:
            y -= intercept_y
        volm = volm[to_keep]
        fov = fov[to_keep]
        rna = rna[to_keep] if rna is not None else None
        size = size[to_keep] if size is not None else None
        # if data points too few, return NaN
        if len(y) < 3:
            return {'coef': np.nan, 'pvalue': np.nan, 'intercept': np.nan}
        fit_data=pd.DataFrame({'y': y, 'x': x, 'volm': volm, 'fov': fov, 'RNA': rna, 'size': size})
        glm_results = smf.ols(formula, data=fit_data).fit()
        # Direct access to coefficients and p-values instead of HTML parsing
        coef = glm_results.params.get('x', np.nan)
        intercept = glm_results.params.get('Intercept', np.nan)
        pvalue = glm_results.pvalues.get('x', np.nan)
        return {'coef': coef, 'intercept': intercept, 'pvalue': pvalue}
    except Exception as e:
        return {'coef': np.nan, 'intercept': np.nan, 'pvalue': np.nan}


def glm(adata, variate='T7', cell_types_to_use=None, CREs=None, norm_by_volm=False, 
        volm_covariate=False, fov_covariate=False, rna_covariate=False, size_covariate=False,
        filter_infected_cells=False, positive_x_or_y=False, only_keep_positive_x=False, only_keep_positive_y=False, 
        transform_x_y=None, fix_intercept=None, 
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
    if size_covariate:
        # pseudo bulk size covariate
        formula += ' + size'
    formula_orig = formula
    
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
    size_data = adata.obs['size'].values if size_covariate else None

    idx = 0
    for k, cell_mask in enumerate(cell_masks):
        # Pre-slice all arrays for this cell type once
        cell_obs_names = obs_names_array[cell_mask]
        cell_volm = volm_values[cell_mask]
        cell_fov = fov_values[cell_mask]
        cell_rna_sum = rna_data[cell_mask].sum(axis=1) if rna_covariate else None
        cell_size = size_data[cell_mask] if size_covariate else None

        # Extract all CRE data for this cell type at once (vectorized)
        cell_cre_matrix = cre_data.loc[cell_mask, CREs]
        cell_variate_matrix = variate.loc[cell_obs_names, CREs]
        
        # if we want to fix x or y, we need to get the intercept first
        if fix_intercept is not None:
            if fix_intercept.startswith('negative_control'):
                negative_control_cres = adata.uns['CRE_info']['labeling_type'] == 'negative control'
                args_batch = [(
                    formula_orig,
                    cell_cre_matrix.iloc[:, negative_control_cres.values].values.flatten(),
                    cell_variate_matrix.iloc[:, negative_control_cres.values].values.flatten(),
                    np.tile(cell_volm, negative_control_cres.sum()),
                    np.tile(cell_fov, negative_control_cres.sum()),
                    np.tile(cell_rna_sum, negative_control_cres.sum()) if rna_covariate else None,
                    np.tile(cell_size, negative_control_cres.sum()) if size_covariate else None,
                    positive_x_or_y, only_keep_positive_x, only_keep_positive_y, transform_x_y,
                )]
            elif fix_intercept.startswith('total'):
                args_batch = [(
                    formula_orig,
                    cell_cre_matrix.values.flatten(),
                    cell_variate_matrix.values.flatten(),
                    np.tile(cell_volm, cell_cre_matrix.shape[1]),
                    np.tile(cell_fov, cell_cre_matrix.shape[1]),
                    np.tile(cell_rna_sum, cell_cre_matrix.shape[1]) if rna_covariate else None,
                    np.tile(cell_size, cell_cre_matrix.shape[1]) if size_covariate else None,
                    positive_x_or_y, only_keep_positive_x, only_keep_positive_y, transform_x_y,
                )]
            else:
                raise ValueError("fix_intercept must start with 'negative_control' or 'total'.")
            # get intercept by fitting on all data
            intercept_res = fit_glm(*args_batch[0])
            if fix_intercept.endswith('x'):
                intercept_x = -intercept_res['intercept'] / intercept_res['coef']
                intercept_y = None
            elif fix_intercept.endswith('y'):
                intercept_x = None
                intercept_y = intercept_res['intercept']
            else:
                raise ValueError("fix_intercept must end with 'x' or 'y'.")
            formula = formula_orig + ' - 1'  # remove intercept from formula
        else:
            intercept_x = None
            intercept_y = None
        # Use list comprehension for remaining loop
        args_batch = [(
            formula,
            cell_cre_matrix.iloc[:, j].values,
            cell_variate_matrix.iloc[:, j].values,
            cell_volm,
            cell_fov,
            cell_rna_sum,
            cell_size,
            positive_x_or_y, only_keep_positive_x, only_keep_positive_y, 
            transform_x_y, intercept_x, intercept_y
        ) for j in range(len(CREs))]
        
        # Batch assignment
        glm_args[idx:idx+len(CREs)] = args_batch
        cell_type_indices[idx:idx+len(CREs)] = [k] * len(CREs)
        cre_indices[idx:idx+len(CREs)] = list(range(len(CREs)))
        idx += len(CREs)
    
    # Run all GLM fits in parallel
    if multiprocess_threads is not None and multiprocess_threads > 1:
        results = Parallel(n_jobs=min(multiprocess_threads, int(multiprocessing.cpu_count()*0.8)), verbose=10)(
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
    cluster_color_map = {}
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
        cluster_color_map[cluster_] = col
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
        return fig, cluster_color_map
    else:
        return cluster_color_map


def _check_cached_result(obj, result_attr_name, config_attr_name, config):
    """
    Check if cached results exist for given configuration.

    Parameters:
    -----------
    obj : object
        Object to check for cached results (typically self)
    result_attr_name : str
        Name of attribute storing results list
    config_attr_name : str
        Name of attribute storing configs list
    config : dict
        Configuration to match against

    Returns:
    --------
    result or None
        Cached result if found, None otherwise
    """
    if hasattr(obj, result_attr_name) and hasattr(obj, config_attr_name):
        results = getattr(obj, result_attr_name)
        configs = getattr(obj, config_attr_name)
        for stored_config, stored_result in zip(configs, results):
            if stored_config == config:
                print('Results already exist, return stored results')
                return stored_result.copy()
    return None


def _store_result(obj, result_attr_name, config_attr_name, result, config):
    """
    Store result and config in object attributes.

    Parameters:
    -----------
    obj : object
        Object to store results in (typically self)
    result_attr_name : str
        Name of attribute to store results list
    config_attr_name : str
        Name of attribute to store configs list
    result : any
        Result to store
    config : dict
        Configuration to store
    """
    if not hasattr(obj, result_attr_name) or not hasattr(obj, config_attr_name):
        setattr(obj, result_attr_name, [])
        setattr(obj, config_attr_name, [])
    getattr(obj, result_attr_name).append(result)
    getattr(obj, config_attr_name).append(config)


def _assign_cre_info_from_best_subclass(cre_info, result_dfs, metrics):
    """
    Assign metrics to cre_info based on best_subclass.

    Parameters:
    -----------
    cre_info : pd.DataFrame
        CRE information dataframe with 'best_subclass' column
    result_dfs : dict
        Dictionary mapping metric names to DataFrames (indexed by cell_type, columned by CRE)
    metrics : list
        List of metric names to assign

    Returns:
    --------
    pd.DataFrame
        Updated cre_info
    """
    for cre in cre_info.index:
        best_subclass = cre_info.loc[cre, 'best_subclass']
        for metric in metrics:
            if metric in result_dfs and best_subclass in result_dfs[metric].index:
                cre_info.loc[cre, metric] = result_dfs[metric].loc[best_subclass, cre]
    return cre_info


def _bootstrap_sample_cells(cell_types_to_use, random_state, bootstrap_to_fixed_sample_size=None,
                           bootstrap_to_fixed_pct=None, permute_labels=False):
    """
    Bootstrap sample cells with various sampling strategies.

    Parameters:
    -----------
    cell_types_to_use : pd.Series
        Series mapping cell indices to cell types
    random_state : int
        Random seed for reproducibility
    bootstrap_to_fixed_sample_size : int or None
        If -1: sample all cells from each type with replacement
        If > 0: sample fixed number of cells from each type with replacement
        If None: use bootstrap_to_fixed_pct instead
    bootstrap_to_fixed_pct : float or None
        Fraction of cells to sample (used when bootstrap_to_fixed_sample_size is None)
    permute_labels : bool
        If True, randomly reassign cell type labels (for fold change test)

    Returns:
    --------
    pd.Series
        Bootstrapped cell type assignments
    """
    if bootstrap_to_fixed_sample_size is not None:
        if bootstrap_to_fixed_sample_size == -1:
            # Sample all cells from each type with replacement
            sample_sizes = [sum(cell_types_to_use == celltype) for celltype in cell_types_to_use.unique()]
        else:
            # Sample fixed number from each type
            sample_sizes = [bootstrap_to_fixed_sample_size] * len(cell_types_to_use.unique())

        cells_bootstrap = pd.concat([
            cell_types_to_use[cell_types_to_use == celltype].sample(
                n=size, replace=True, random_state=random_state
            ) for celltype, size in zip(cell_types_to_use.unique(), sample_sizes)
        ])

        if permute_labels:
            # Randomly reassign cell indices (for permutation test)
            cells_idx = pd.concat([
                cell_types_to_use[cell_types_to_use != celltype].sample(
                    n=size, replace=True, random_state=random_state
                ) for celltype, size in zip(cell_types_to_use.unique(), sample_sizes)
            ])
            cells_bootstrap.index = cells_idx.index
    else:
        # Sample by fraction
        frac = 1.0 if bootstrap_to_fixed_pct is None else bootstrap_to_fixed_pct
        cells_bootstrap = cell_types_to_use.sample(frac=frac, replace=True, random_state=random_state)

        if permute_labels:
            # Permute all labels
            cells_bootstrap.index = cell_types_to_use.index

    return cells_bootstrap


def _calculate_fold_change_with_bootstrap(cre_cells_expression, cell_types_order, CRE_info,
                                          rna_cells_expression, volm, t7_cells_expression,
                                          calc_kwargs, bootstrap_args):
    """Calculate fold change with bootstrap sampling (permutation test)."""
    i, cell_types_to_use, bootstrap_to_fixed_sample_size = bootstrap_args
    cells_bootstrap = _bootstrap_sample_cells(
        cell_types_to_use, random_state=i,
        bootstrap_to_fixed_sample_size=bootstrap_to_fixed_sample_size,
        permute_labels=True
    )
    return calculate_fold_change(cre_cells_expression, cells_bootstrap, cell_types_order, CRE_info,
                                 rna_cells_expression, volm, t7_cells_expression, **calc_kwargs)


def _calculate_average_with_bootstrap(cre_cells_expression, cell_types_order, CRE_info,
                                      rna_cells_expression, volm, t7_cells_expression,
                                      calc_kwargs, bootstrap_args):
    """Calculate average with bootstrap sampling (within cell type)."""
    i, cell_types_to_use, bootstrap_to_fixed_sample_size, bootstrap_to_fixed_pct = bootstrap_args
    cells_bootstrap = _bootstrap_sample_cells(
        cell_types_to_use, random_state=i,
        bootstrap_to_fixed_sample_size=bootstrap_to_fixed_sample_size,
        bootstrap_to_fixed_pct=bootstrap_to_fixed_pct,
        permute_labels=False
    )
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
            # sum all negative control CRE reads and divide by sum of all t7 reads
            negative_control_mean = celltype_activity_matrix_raw['CRE'].loc[:, negative_control].sum(axis=1) / celltype_activity_matrix_raw['T7'].loc[:, negative_control].sum(axis=1)
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
    # Collect results in dictionaries for efficient batch assignment
    results_dict = {col: [] for col in col_result.columns}

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
            # Collect results
            results_dict['pearson'].append(pearson[0])
            results_dict['spearman'].append(spearman[0])
            results_dict['fisher'].append(fisher[0])
            results_dict['pearson_p'].append(pearson[1])
            results_dict['spearman_p'].append(spearman[1])
            results_dict['fisher_p'].append(fisher[1])
            results_dict['effect_n'].append(tokeep.sum())
        except:
            print('Error in calculating correlation for CRE: ', cre)
            # Append NaN for failed calculations
            for col in ['pearson', 'spearman', 'fisher', 'pearson_p', 'spearman_p', 'fisher_p', 'effect_n']:
                results_dict[col].append(np.nan)

    # Assign all results at once
    for col in ['pearson', 'spearman', 'fisher', 'pearson_p', 'spearman_p', 'fisher_p', 'effect_n']:
        col_result[col] = results_dict[col]
    col_result['pearson_q'] = multitest.multipletests(col_result['pearson_p'], method='fdr_bh')[1]
    col_result['spearman_q'] = multitest.multipletests(col_result['spearman_p'], method='fdr_bh')[1]
    col_result['fisher_q'] = multitest.multipletests(col_result['fisher_p'], method='fdr_bh')[1]
    return col_result


def row_corr(df1: pd.DataFrame, df2: pd.DataFrame, bin_threshold1=None, bin_threshold2=None):
    # do row wise correlation
    row_result = pd.DataFrame(index=df1.index,
                              columns=['pearson', 'spearman', 'fisher',
                                       'pearson_p', 'spearman_p', 'fisher_p', 'effect_n',
                                       'pearson_q', 'spearman_q', 'fisher_q'])
    # Collect results in dictionaries for efficient batch assignment
    results_dict = {col: [] for col in row_result.columns}

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
            # Collect results
            results_dict['pearson'].append(pearson[0])
            results_dict['spearman'].append(spearman[0])
            results_dict['fisher'].append(fisher[0])
            results_dict['pearson_p'].append(pearson[1])
            results_dict['spearman_p'].append(spearman[1])
            results_dict['fisher_p'].append(fisher[1])
            results_dict['effect_n'].append(tokeep.sum())
        except:
            print('Error in calculating correlation for celltype: ', celltype)
            # Append NaN for failed calculations
            for col in ['pearson', 'spearman', 'fisher', 'pearson_p', 'spearman_p', 'fisher_p', 'effect_n']:
                results_dict[col].append(np.nan)

    # Assign all results at once
    for col in ['pearson', 'spearman', 'fisher', 'pearson_p', 'spearman_p', 'fisher_p', 'effect_n']:
        row_result[col] = results_dict[col]
    row_result['pearson_q'] = multitest.multipletests(row_result['pearson_p'], method='fdr_bh')[1]
    row_result['spearman_q'] = multitest.multipletests(row_result['spearman_p'], method='fdr_bh')[1]
    row_result['fisher_q'] = multitest.multipletests(row_result['fisher_p'], method='fdr_bh')[1]
    return row_result


def cross_talk_fisher_test(celltype_activated: np.ndarray):
    # Initialize with NaN to ensure all values are properly set
    pval = np.full((1, celltype_activated.shape[1], celltype_activated.shape[1]), np.nan)
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
    # Initialize with NaN to ensure all values are properly set
    pval = np.full((1, celltype_expression.shape[1], celltype_expression.shape[1]), np.nan)
    corr = np.full((1, celltype_expression.shape[1], celltype_expression.shape[1]), np.nan)
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


class T7CRE_Joint_DistributionEM:
    # jointly model T7 and CRE distributions via the number of infected virus K
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu', use_x0=True):
        self.device = device
        self.use_x0 = use_x0
    
    def expectation_step(self, x0, x1, x2, obs_t7, obs_cre, cell_type_indexes, dim=20):
        """
        E-step: Calculate posterior distribution of k given current parameters
        x0: shape of (n_celltypes, n_cres)
        x1: shape of (2): r and q for t7
        x2: shape of (n_celltypes, n_cres, 2) cre activities
        obs_t7: shape of (n_obs, n_cres)
        obs_cre: shape of (n_obs, n_cres)
        """
        obs_t7 = obs_t7.to(self.device).long()
        obs_cre = obs_cre.to(self.device).long()
        cell_type_indexes = cell_type_indexes.to(self.device).long()
        x0 = x0.to(self.device).float()
        x1 = x1.to(self.device).float()
        x2 = x2.to(self.device).float()

        k_values = torch.arange(dim, device=self.device, dtype=torch.float32)  # (dim,)
        k_values_pos = torch.arange(1, dim, device=self.device, dtype=torch.float32)  # (dim-1,)
        
        # Get lambda values for each observation based on cell type
        k_lambda = torch.nn.functional.softplus(x0[cell_type_indexes, :])  # (n_obs, n_cres)

        # Expand dimensions for broadcasting
        obs_t7_expanded = obs_t7.unsqueeze(-1)  # (n_obs, n_cres, 1)
        obs_cre_expanded = obs_cre.unsqueeze(-1)  # (n_obs, n_cres, 1)
        k_lambda_expanded = k_lambda.unsqueeze(-1)  # (n_obs, n_cres, 1)
        k_values_expanded = k_values.unsqueeze(0).unsqueeze(0)  # (1, 1, dim)
        
        # k ~ Poisson(k_lambda): log P(k|k_lambda)
        ll_k = torch.distributions.Poisson(k_lambda_expanded).log_prob(k_values_expanded)  # (n_obs, n_cres, dim)
        
        # Vectorized Poisson log PMF for obs_t7 ~ Poisson(exp(x1) * k)
        total_counts_t7 = torch.nn.functional.softplus(x1[0])
        mu_t7 = total_counts_t7 * k_values_pos  # (dim-1,)
        mu_t7_expanded = mu_t7.unsqueeze(0).unsqueeze(0)  # (1, 1, dim-1)
            
        # T7 log PMF computation
        ll_t7 = torch.distributions.negative_binomial.NegativeBinomial(
            total_count=mu_t7_expanded, logits=x1[1]
        ).log_prob(obs_t7_expanded)  # (n_obs, n_cres, dim-1)
        # handle n=0 case, if obs == 0, prob is 0, otherwise, -inf
        ll_t7 = torch.concat((torch.where(obs_t7_expanded == 0, 0, -torch.inf), ll_t7), dim=-1)

        # Vectorized obs_cre ~ NegBinom(exp(x2[cell_type_indexes, 0]) * k, torch.sigmoid(x2[cell_type_indexes, 1]))
        total_counts_cre = torch.nn.functional.softplus(x2[cell_type_indexes, :, 0]).unsqueeze(-1)  # (n_obs, n_cres, 1)
        logits_cre = x2[cell_type_indexes, :, 1].unsqueeze(-1)  # (n_obs, n_cres, 1)
        mu_cre = total_counts_cre * k_values_pos.unsqueeze(0).unsqueeze(0)  # (n_obs, n_cres, dim-1)

        # CRE log PMF computation
        ll_cre = torch.distributions.negative_binomial.NegativeBinomial(
            total_count=mu_cre, logits=logits_cre
        ).log_prob(obs_cre_expanded)  # (n_obs, n_cres, dim-1)
        # now handle n = 0 case, if obs == 0, prob is 0, otherwise, -inf
        ll_cre = torch.concat((torch.where(obs_cre_expanded == 0, 0, -torch.inf), ll_cre), dim=-1)

        # Combined likelihood
        likelihood_mat = ll_k + ll_t7 + ll_cre
        
        # Posterior distribution using logsumexp for numerical stability
        likelihood_sum = torch.logsumexp(likelihood_mat, dim=-1, keepdim=True)
        posterior_k = likelihood_mat - likelihood_sum
        
        # Calculate total log-likelihood for this step
        total_log_likelihood = likelihood_sum.sum()
        
        return posterior_k, total_log_likelihood
    
    def maximization_step_x0(self, x0, cell_type_indexes, posterior_k):
        """M-step: Optimize x0 parameters"""
        x0_orig = x0.clone().detach()
        x0 = nn.Parameter(x0.clone().detach().requires_grad_(True))
        optimizer = optim.LBFGS([x0])  # Reduced lr and max_iter
        
        def closure():
            optimizer.zero_grad()
            
            dim = posterior_k.shape[-1]
            k_values = torch.arange(dim, device=self.device, dtype=torch.float32)
            
            # Get lambda values for each observation based on cell type
            k_lambda = torch.nn.functional.softplus(x0[cell_type_indexes, :])  # (n_obs, n_cres)
            k_lambda_expanded = k_lambda.unsqueeze(-1)  # (n_obs, n_cres, 1)
            k_values_expanded = k_values.unsqueeze(0).unsqueeze(0)  # (1, 1, dim)
            
            # k ~ Poisson(k_lambda): log P(k|k_lambda)
            ll_k = torch.distributions.Poisson(k_lambda_expanded).log_prob(k_values_expanded)

            # Compute expected log likelihood
            ll = posterior_k + ll_k
            ll_sum = torch.logsumexp(ll, dim=-1)
            loss = -ll_sum.sum()
            
            # Add L2 regularization to prevent extreme values
            loss += 0.001 * (x0 ** 2).sum()
            
            loss.backward()
            return loss
        
        optimizer.step(closure)
        
        return x0.detach(), x0.detach() - x0_orig
    
    def maximization_step_x1(self, x1, obs_t7, posterior_k):
        """M-step: Optimize x1 parameter"""
        x1_orig = x1.clone().detach()
        x1 = nn.Parameter(x1.clone().detach().requires_grad_(True))
        optimizer = optim.LBFGS([x1])  # Reduced lr and max_iter

        def closure():
            optimizer.zero_grad()
            
            obs_t7_expanded = obs_t7.unsqueeze(-1)  # (n_obs, 1)
            dim = posterior_k.shape[-1]
            k_values = torch.arange(1, dim, device=self.device, dtype=torch.float32) # (dim-1)
            
            # Vectorized Poisson log PMF for obs_t7 ~ Poisson(exp(x1) * k)
            total_counts_t7 = torch.nn.functional.softplus(x1[0])
            mu_t7 = total_counts_t7 * k_values  # (dim-1,)
            mu_t7_expanded = mu_t7.unsqueeze(0).unsqueeze(0)  # (1, 1, dim-1)
            
            # T7 log PMF computation
            ll_t7 = torch.distributions.negative_binomial.NegativeBinomial(
                total_count=mu_t7_expanded, logits=x1[1]
            ).log_prob(obs_t7_expanded)  # (n_obs, n_cres, dim-1)
            
            # now handle n = 0 case, if obs == 0, prob is 0, otherwise, -inf
            ll_t7 = torch.concat((torch.where(obs_t7_expanded == 0, 0, -torch.inf), ll_t7), dim=-1)

            ll = posterior_k + ll_t7
            ll_sum = torch.logsumexp(ll, dim=-1)
            loss = -ll_sum.sum()
            
            # Add L2 regularization to prevent extreme values
            loss += 0.001 * (x1 ** 2).sum()
            
            loss.backward()
            return loss
        
        optimizer.step(closure)
        
        return x1.detach(), x1.detach() - x1_orig

    def maximization_step_x2(self, x2, obs_cre, posterior_k, cell_type_indexes):
        """M-step: Optimize x2 parameter"""
        x2_orig = x2.clone().detach()
        x2 = nn.Parameter(x2.clone().detach().requires_grad_(True))
        optimizer = optim.LBFGS([x2])  # Reduced lr and max_iter

        def closure():
            optimizer.zero_grad()
            
            obs_cre_expanded = obs_cre.unsqueeze(-1)  # (n_obs, 1)
            dim = posterior_k.shape[-1]
            k_values = torch.arange(1, dim, device=self.device, dtype=torch.float32) # (dim-1)

            # Vectorized obs_cre ~ NegBinom(exp(x2[cell_type_indexes, 0]) * k, torch.sigmoid(x2[cell_type_indexes, 1]))
            total_counts_cre = torch.nn.functional.softplus(x2[cell_type_indexes, :, 0]).unsqueeze(-1)  # (n_obs, n_cres, 1)
            logits_cre = x2[cell_type_indexes, :, 1].unsqueeze(-1)  # (n_obs, n_cres, 1)
            mu_cre = total_counts_cre * k_values.unsqueeze(0).unsqueeze(0)  # (n_obs, n_cres, dim-1)

            # CRE log PMF computation
            ll_cre = torch.distributions.negative_binomial.NegativeBinomial(
                total_count=mu_cre, logits=logits_cre
            ).log_prob(obs_cre_expanded)  # (n_obs, n_cres, dim-1)

            # now handle n = 0 case, if obs == 0, prob is 0, otherwise, -inf
            ll_cre = torch.concat((torch.where(obs_cre_expanded == 0, 0, -torch.inf), ll_cre), dim=-1)

            ll = posterior_k + ll_cre
            ll_sum = torch.logsumexp(ll, dim=-1)
            loss = -ll_sum.sum()
            
            # Add L2 regularization to prevent extreme values
            loss += 0.001 * (x2 ** 2).sum()
            
            loss.backward()
            return loss
        
        optimizer.step(closure)
        
        return x2.detach(), x2.detach() - x2_orig

    def fit(self, cell_types, obs_t7, obs_cre, dim=20, max_iter=50, x0_prior=None, log_dir=None, 
            x0_checkpoint_path=None, x1_checkpoint_path=None, x2_checkpoint_path=None):
        """Main EM algorithm"""
        # Initialize TensorBoard writer
        if log_dir is None:
            log_dir = f"runs/em_algorithm_{int(time.time())}"
        writer = SummaryWriter(log_dir)
        print(f"TensorBoard logging to: {log_dir}")
        # Convert inputs to numpy arrays first
        # Handle pandas objects (Series, Categorical, etc.)
        if hasattr(cell_types, 'values'):
            cell_types = cell_types.values
        if hasattr(obs_t7, 'values'):
            obs_t7 = obs_t7.values
        if hasattr(obs_cre, 'values'):
            obs_cre = obs_cre.values

        # Ensure we have numpy arrays
        if not isinstance(cell_types, np.ndarray):
            cell_types = np.array(cell_types)
        if not isinstance(obs_t7, np.ndarray):
            obs_t7 = np.array(obs_t7)
        if not isinstance(obs_cre, np.ndarray):
            obs_cre = np.array(obs_cre)
            
        # Ensure obs_t7 is float type (while it's still numpy)
        obs_t7 = obs_t7.astype(np.long)
        obs_cre = obs_cre.astype(np.long)
        
        # Convert categorical cell types to integer indices
        unique_celltypes, cell_type_indexes = np.unique(cell_types, return_inverse=True)
        
        # Convert to torch tensors (no more .astype calls after this point)
        cell_type_indexes = torch.from_numpy(cell_type_indexes).to(self.device).long()
        obs_t7 = torch.from_numpy(obs_t7).to(self.device).long()
        obs_cre = torch.from_numpy(obs_cre).to(self.device).long()

        # Initialize parameters from checkpoints if provided, otherwise use default initialization
        x0_loaded, x1_loaded, x2_loaded = self.load_checkpoint(x0_checkpoint_path, x1_checkpoint_path, x2_checkpoint_path)
        
        # Initialize x0
        if x0_loaded is not None:
            x0 = x0_loaded
            print("Using x0 from checkpoint")
        elif x0_prior is None:
            x0 = torch.zeros((len(unique_celltypes), obs_t7.shape[1]), device=self.device, dtype=torch.float32)
        elif x0_prior == 'zero_percentage':
            # check the number of non-zero elements in each cell type
            zero_counts = (pd.DataFrame(obs_t7.cpu().numpy()) == 0).groupby(cell_type_indexes.cpu().numpy()).mean()
            # set x0 to the mean of the non-zero elements
            x0 = np.clip(-np.log(zero_counts), min=1e-9)
            # transform to torch.softplus logits
            x0 = np.log(np.exp(x0) - 1)
            x0 = torch.tensor(x0.values, device=self.device, dtype=torch.float32)
        
        # Initialize x1
        if x1_loaded is not None:
            x1 = x1_loaded
            print("Using x1 from checkpoint")
        else:
            x1 = torch.tensor([-1.0, 0], device=self.device, dtype=torch.float32)  # Start with more conservative value
        
        # Initialize x2
        if x2_loaded is not None:
            x2 = x2_loaded
            print("Using x2 from checkpoint")
        else:
            x2 = torch.tensor(np.zeros((len(unique_celltypes), obs_t7.shape[1], 2)), device=self.device, dtype=torch.float32)  # Start with more conservative value
        
        print(f"Running EM algorithm on {self.device}")
        print(f"Initial x0: {x0.cpu().numpy().mean()}")
        print(f"Initial x1: {x1.cpu().numpy()}")
        print(f"Initial x2: {x2.cpu().numpy().mean()}")
        
        # Log initial parameters
        writer.add_scalar('Parameters/x0_mean', x0.cpu().numpy().mean(), 0)
        writer.add_scalar('Parameters/x0_std', x0.cpu().numpy().std(), 0)
        writer.add_scalar('Parameters/x1_0', x1.cpu().numpy()[0], 0)
        writer.add_scalar('Parameters/x1_1', x1.cpu().numpy()[1], 0)
        writer.add_scalar('Parameters/x2_mean', x2.cpu().numpy().mean(), 0)
        writer.add_scalar('Parameters/x2_std', x2.cpu().numpy().std(), 0)
        
        # Initialize for likelihood improvement tracking
        prev_log_likelihood = None

        for i in range(max_iter):
            # Check for NaN/inf values before each step
            if torch.isnan(x0).any() or torch.isinf(x0).any() or torch.isnan(x1).any() or torch.isinf(x1).any() or torch.isnan(x2).any() or torch.isinf(x2).any():
                print(f"Warning: NaN/inf detected in parameters at step {i}")
                break
            
            # E-step
            start = time.time()
            posterior_k, log_likelihood = self.expectation_step(x0, x1, x2, obs_t7, obs_cre, cell_type_indexes, dim=dim)
            mid1 = time.time()
            print(f"Step {i}, E-step time: {mid1 - start:.4f}s, Log-likelihood: {log_likelihood.item():.6f}")
            
            # Log likelihood to TensorBoard
            writer.add_scalar('Likelihood/log_likelihood', log_likelihood.item(), i+1)
            
            # Log likelihood improvement
            if prev_log_likelihood is not None:
                likelihood_improvement = log_likelihood.item() - prev_log_likelihood
                writer.add_scalar('Likelihood/likelihood_improvement', likelihood_improvement, i+1)
                print(f"Step {i}, Likelihood improvement: {likelihood_improvement:.6f}")
            prev_log_likelihood = log_likelihood.item()
            # Check for NaN in posterior
            if torch.isnan(posterior_k).any():
                print(f"Warning: NaN in posterior at step {i}")
                break
            
            if self.use_x0:
                # M-step for x0
                x0, x0_diff = self.maximization_step_x0(x0, cell_type_indexes, posterior_k)
                mid2 = time.time()
                print(f"Step {i}, M-step x0 time: {mid2 - mid1:.4f}s")
                # Check for NaN in posterior
                if torch.isnan(x0).any():
                    print(f"Warning: NaN in posterior at step {i}")
                    break
            else:
                mid2 = time.time()
                x0_diff = x0.clone()
            
            # M-step for x1
            x1, x1_diff = self.maximization_step_x1(x1, obs_t7, posterior_k)
            end = time.time()
            print(f"Step {i}, M-step x1 time: {end - mid2:.4f}s")
            # Check for NaN in posterior
            if torch.isnan(x1).any():
                print(f"Warning: NaN in posterior at step {i}")
                break
            
            # M-step for x2
            x2, x2_diff = self.maximization_step_x2(x2, obs_cre, posterior_k, cell_type_indexes)
            end = time.time()
            print(f"Step {i}, M-step x2 time: {end - mid2:.4f}s")
            # Check for NaN in posterior
            if torch.isnan(x2).any():
                print(f"Warning: NaN in posterior at step {i}")
                break
            
            print(f"Step {i}, x0: {x0.cpu().numpy().mean()}, x1: {x1.cpu().numpy()}, x2: {x2.cpu().numpy().mean()}")
            
            # Log parameters to TensorBoard
            writer.add_scalar('Parameters/x0_mean', x0.cpu().numpy().mean(), i+1)
            writer.add_scalar('Parameters/x0_std', x0.cpu().numpy().std(), i+1)
            writer.add_scalar('Parameters/x1_0', x1.cpu().numpy()[0], i+1)
            writer.add_scalar('Parameters/x1_1', x1.cpu().numpy()[1], i+1)
            writer.add_scalar('Parameters/x2_mean', x2.cpu().numpy().mean(), i+1)
            writer.add_scalar('Parameters/x2_std', x2.cpu().numpy().std(), i+1)
            
            # Log parameter changes
            writer.add_scalar('Parameter_Changes/x0_diff_mean', torch.abs(x0_diff).mean().cpu().numpy(), i+1)
            writer.add_scalar('Parameter_Changes/x1_diff_mean', torch.abs(x1_diff).mean().cpu().numpy(), i+1)
            writer.add_scalar('Parameter_Changes/x2_diff_mean', torch.abs(x2_diff).mean().cpu().numpy(), i+1)
            
            # Log relative changes for convergence monitoring
            x0_rel_change = torch.abs(x0_diff) / (torch.abs(x0) + 1e-8)
            x1_rel_change = torch.abs(x1_diff) / (torch.abs(x1) + 1e-8)
            x2_rel_change = torch.abs(x2_diff) / (torch.abs(x2) + 1e-8)
            
            writer.add_scalar('Convergence/x0_rel_change_max', x0_rel_change.max().cpu().numpy(), i+1)
            writer.add_scalar('Convergence/x1_rel_change_max', x1_rel_change.max().cpu().numpy(), i+1)
            writer.add_scalar('Convergence/x2_rel_change_max', x2_rel_change.max().cpu().numpy(), i+1)

            # check convergence
            if torch.all(torch.abs(x0_diff) / torch.abs(x0) < 1e-5) and torch.all(torch.abs(x1_diff) / torch.abs(x1) < 1e-5) and torch.all(torch.abs(x2_diff) / torch.abs(x2) < 1e-5):
                print(f"Converged at step {i}")
                writer.add_scalar('Training/converged_at_step', i, 0)
                break
        
        # Log final training metrics
        writer.add_scalar('Training/total_iterations', i+1, 0)
        writer.add_scalar('Training/max_iterations', max_iter, 0)
        
        
        # Close the writer
        writer.close()
        
        return x0.cpu().numpy(), x1.cpu().numpy(), x2.cpu().numpy()
    
    def load_checkpoint(self, x0_path=None, x1_path=None, x2_path=None):
        """Load model parameters from separate .npy files"""
        x0, x1, x2 = None, None, None
        
        if x0_path is not None:
            x0 = torch.from_numpy(np.load(x0_path)).to(self.device).float()
            print(f"Loaded x0 from {x0_path}")
        
        if x1_path is not None:
            x1 = torch.from_numpy(np.load(x1_path)).to(self.device).float()
            print(f"Loaded x1 from {x1_path}")
        
        if x2_path is not None:
            x2 = torch.from_numpy(np.load(x2_path)).to(self.device).float()
            print(f"Loaded x2 from {x2_path}")
        
        return x0, x1, x2


class T7CRE_Split_DistributionEM:
    # separately model T7 and CRE distribution, first estimate the infection rate
    # lambda in each cell type and each CRE, then apply the infection rate to correct 
    # the activity
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu', use_x0=True, dim=20):
        self.device = device
        self.use_x0 = use_x0
        self.dim = dim
    
    def expectation_t7_step(self, x0, x1, obs_t7, cell_type_indexes):
        """
        E-step: Calculate posterior distribution of k given current parameters
        x0: infection rate shape of (n_celltypes, n_cres)
        x1: shape of (2): r and q for t7
        obs_t7: shape of (n_obs, n_cres)
        """
        obs_t7 = obs_t7.to(self.device).long()
        cell_type_indexes = cell_type_indexes.to(self.device).long()
        x0 = x0.to(self.device).float()
        x1 = x1.to(self.device).float()

        k_values = torch.arange(self.dim, device=self.device, dtype=torch.float32)  # (dim,)
        k_values_pos = torch.arange(1, self.dim, device=self.device, dtype=torch.float32)  # (dim-1,)
        
        # Get lambda values for each observation based on cell type
        k_lambda = torch.nn.functional.softplus(x0[cell_type_indexes, :])  # (n_obs, n_cres)

        # Expand dimensions for broadcasting
        obs_t7_expanded = obs_t7.unsqueeze(-1)  # (n_obs, n_cres, 1)
        k_lambda_expanded = k_lambda.unsqueeze(-1)  # (n_obs, n_cres, 1)
        k_values_expanded = k_values.unsqueeze(0).unsqueeze(0)  # (1, 1, dim)
        
        # k ~ Poisson(k_lambda): log P(k|k_lambda)
        ll_k = torch.distributions.Poisson(k_lambda_expanded).log_prob(k_values_expanded)  # (n_obs, n_cres, dim)
        
        # Vectorized Poisson log PMF for obs_t7 ~ Poisson(exp(x1) * k)
        total_counts_t7 = torch.nn.functional.softplus(x1[0])
        mu_t7 = total_counts_t7 * k_values_pos  # (dim-1,)
        mu_t7_expanded = mu_t7.unsqueeze(0).unsqueeze(0)  # (1, 1, dim-1)
            
        # T7 log PMF computation
        ll_t7 = torch.distributions.negative_binomial.NegativeBinomial(
            total_count=mu_t7_expanded, logits=x1[1]
        ).log_prob(obs_t7_expanded)  # (n_obs, n_cres, dim-1)
        # handle n=0 case, if obs == 0, prob is 0, otherwise, -inf
        ll_t7 = torch.concat((torch.where(obs_t7_expanded == 0, 0, -torch.inf), ll_t7), dim=-1)

        # Combined likelihood
        likelihood_mat = ll_k + ll_t7
        
        # Posterior distribution using logsumexp for numerical stability
        likelihood_sum = torch.logsumexp(likelihood_mat, dim=-1, keepdim=True)
        posterior_k = likelihood_mat - likelihood_sum
        
        # Calculate total log-likelihood for this step
        total_log_likelihood = likelihood_sum.sum()
        
        return posterior_k, total_log_likelihood
    
    def maximization_step_x0(self, x0, cell_type_indexes, posterior_k):
        """M-step: Optimize x0 parameters"""
        x0_orig = x0.clone().detach()
        x0 = nn.Parameter(x0.clone().detach().requires_grad_(True))
        optimizer = optim.LBFGS([x0])  # Reduced lr and max_iter
        
        def closure():
            optimizer.zero_grad()
            
            dim = posterior_k.shape[-1]
            k_values = torch.arange(dim, device=self.device, dtype=torch.float32)
            
            # Get lambda values for each observation based on cell type
            k_lambda = torch.nn.functional.softplus(x0[cell_type_indexes, :])  # (n_obs, n_cres)
            k_lambda_expanded = k_lambda.unsqueeze(-1)  # (n_obs, n_cres, 1)
            k_values_expanded = k_values.unsqueeze(0).unsqueeze(0)  # (1, 1, dim)
            
            # k ~ Poisson(k_lambda): log P(k|k_lambda)
            ll_k = torch.distributions.Poisson(k_lambda_expanded).log_prob(k_values_expanded)

            # Compute expected log likelihood
            ll = posterior_k + ll_k
            ll_sum = torch.logsumexp(ll, dim=-1)
            loss = -ll_sum.sum()
            
            # Add L2 regularization to prevent extreme values
            loss += 0.001 * (x0 ** 2).sum()
            
            loss.backward()
            return loss
        
        optimizer.step(closure)
        
        return x0.detach(), x0.detach() - x0_orig
    
    def maximization_step_x1(self, x1, obs_t7, posterior_k):
        """M-step: Optimize x1 parameter"""
        x1_orig = x1.clone().detach()
        x1 = nn.Parameter(x1.clone().detach().requires_grad_(True))
        optimizer = optim.LBFGS([x1])  # Reduced lr and max_iter

        def closure():
            optimizer.zero_grad()
            
            obs_t7_expanded = obs_t7.unsqueeze(-1)  # (n_obs, 1)
            dim = posterior_k.shape[-1]
            k_values = torch.arange(1, dim, device=self.device, dtype=torch.float32) # (dim-1)
            
            # Vectorized Poisson log PMF for obs_t7 ~ Poisson(exp(x1) * k)
            total_counts_t7 = torch.nn.functional.softplus(x1[0])
            mu_t7 = total_counts_t7 * k_values  # (dim-1,)
            mu_t7_expanded = mu_t7.unsqueeze(0).unsqueeze(0)  # (1, 1, dim-1)
            
            # T7 log PMF computation
            ll_t7 = torch.distributions.negative_binomial.NegativeBinomial(
                total_count=mu_t7_expanded, logits=x1[1]
            ).log_prob(obs_t7_expanded)  # (n_obs, n_cres, dim-1)
            
            # now handle n = 0 case, if obs == 0, prob is 0, otherwise, -inf
            ll_t7 = torch.concat((torch.where(obs_t7_expanded == 0, 0, -torch.inf), ll_t7), dim=-1)

            ll = posterior_k + ll_t7
            ll_sum = torch.logsumexp(ll, dim=-1)
            loss = -ll_sum.sum()
            
            # Add L2 regularization to prevent extreme values
            loss += 0.001 * (x1 ** 2).sum()
            
            loss.backward()
            return loss
        
        optimizer.step(closure)
        
        return x1.detach(), x1.detach() - x1_orig

    def expectation_cre_step(self, x0, x2, obs_cre, cell_type_indexes):
        """
        E-step: Calculate posterior distribution of k given current parameters
        """
        obs_cre = obs_cre.to(self.device).long()
        cell_type_indexes = cell_type_indexes.to(self.device).long()
        x0 = x0.to(self.device).float()
        x2 = x2.to(self.device).float()

        k_values = torch.arange(self.dim, device=self.device, dtype=torch.float32)  # (dim,)
        k_values_pos = torch.arange(1, self.dim, device=self.device, dtype=torch.float32)  # (dim-1,)
        
        # Get lambda values for each observation based on cell type
        k_lambda = torch.nn.functional.softplus(x0[cell_type_indexes, :])  # (n_obs, n_cres)

        # Expand dimensions for broadcasting
        obs_cre_expanded = obs_cre.unsqueeze(-1)  # (n_obs, n_cres, 1)
        k_lambda_expanded = k_lambda.unsqueeze(-1)  # (n_obs, n_cres, 1)
        k_values_expanded = k_values.unsqueeze(0).unsqueeze(0)  # (1, 1, dim)
        
        # k ~ Poisson(k_lambda): log P(k|k_lambda)
        ll_k = torch.distributions.Poisson(k_lambda_expanded).log_prob(k_values_expanded)  # (n_obs, n_cres, dim)
        
        # Vectorized obs_cre ~ NegBinom(exp(x2[cell_type_indexes, 0]) * k, torch.sigmoid(x2[cell_type_indexes, 1]))
        total_counts_cre = torch.nn.functional.softplus(x2[cell_type_indexes, :, 0]).unsqueeze(-1)  # (n_obs, n_cres, 1)
        logits_cre = x2[cell_type_indexes, :, 1].unsqueeze(-1)  # (n_obs, n_cres, 1)
        mu_cre = total_counts_cre * k_values_pos.unsqueeze(0).unsqueeze(0)  # (n_obs, n_cres, dim-1)

        # CRE log PMF computation
        ll_cre = torch.distributions.negative_binomial.NegativeBinomial(
            total_count=mu_cre, logits=logits_cre
        ).log_prob(obs_cre_expanded)  # (n_obs, n_cres, dim-1)
        # now handle n = 0 case, if obs == 0, prob is 0, otherwise, -inf
        ll_cre = torch.concat((torch.where(obs_cre_expanded == 0, 0, -torch.inf), ll_cre), dim=-1)

        # Combined likelihood
        likelihood_mat = ll_k + ll_cre
        
        # Posterior distribution using logsumexp for numerical stability
        likelihood_sum = torch.logsumexp(likelihood_mat, dim=-1, keepdim=True)
        posterior_k = likelihood_mat - likelihood_sum
        
        # Calculate total log-likelihood for this step
        total_log_likelihood = likelihood_sum.sum()
        
        return posterior_k, total_log_likelihood

    def maximization_step_x2(self, x2, obs_cre, posterior_k, cell_type_indexes):
        """M-step: Optimize x2 parameter"""
        x2_orig = x2.clone().detach()
        x2 = nn.Parameter(x2.clone().detach().requires_grad_(True))
        optimizer = optim.LBFGS([x2])  # Reduced lr and max_iter

        def closure():
            optimizer.zero_grad()
            
            obs_cre_expanded = obs_cre.unsqueeze(-1)  # (n_obs, 1)
            dim = posterior_k.shape[-1]
            k_values = torch.arange(1, dim, device=self.device, dtype=torch.float32) # (dim-1)

            # Vectorized obs_cre ~ NegBinom(exp(x2[cell_type_indexes, 0]) * k, torch.sigmoid(x2[cell_type_indexes, 1]))
            total_counts_cre = torch.nn.functional.softplus(x2[cell_type_indexes, :, 0]).unsqueeze(-1)  # (n_obs, n_cres, 1)
            logits_cre = x2[cell_type_indexes, :, 1].unsqueeze(-1)  # (n_obs, n_cres, 1)
            mu_cre = total_counts_cre * k_values.unsqueeze(0).unsqueeze(0)  # (n_obs, n_cres, dim-1)

            # CRE log PMF computation
            ll_cre = torch.distributions.negative_binomial.NegativeBinomial(
                total_count=mu_cre, logits=logits_cre
            ).log_prob(obs_cre_expanded)  # (n_obs, n_cres, dim-1)

            # now handle n = 0 case, if obs == 0, prob is 0, otherwise, -inf
            ll_cre = torch.concat((torch.where(obs_cre_expanded == 0, 0, -torch.inf), ll_cre), dim=-1)

            ll = posterior_k + ll_cre
            ll_sum = torch.logsumexp(ll, dim=-1)
            loss = -ll_sum.sum()
            
            # Add L2 regularization to prevent extreme values
            loss += 0.001 * (x2 ** 2).sum()
            
            loss.backward()
            return loss
        
        optimizer.step(closure)
        
        return x2.detach(), x2.detach() - x2_orig

    def fit(self, cell_types, obs_t7, obs_cre, max_iter=50, x0_prior=None, log_dir=None,
            x0_checkpoint_path=None, x1_checkpoint_path=None, x2_checkpoint_path=None):
        """Main EM algorithm"""
        # Initialize TensorBoard writer
        if log_dir is None:
            log_dir = f"runs/em_split_algorithm_{int(time.time())}"
        writer = SummaryWriter(log_dir)
        print(f"TensorBoard logging to: {log_dir}")
        # Convert inputs to numpy arrays first
        # Handle pandas objects (Series, Categorical, etc.)
        if hasattr(cell_types, 'values'):
            cell_types = cell_types.values
        if hasattr(obs_t7, 'values'):
            obs_t7 = obs_t7.values
        if hasattr(obs_cre, 'values'):
            obs_cre = obs_cre.values

        # Ensure we have numpy arrays
        if not isinstance(cell_types, np.ndarray):
            cell_types = np.array(cell_types)
        if not isinstance(obs_t7, np.ndarray):
            obs_t7 = np.array(obs_t7)
        if not isinstance(obs_cre, np.ndarray):
            obs_cre = np.array(obs_cre)
            
        # Ensure obs_t7 is float type (while it's still numpy)
        obs_t7 = obs_t7.astype(np.long)
        obs_cre = obs_cre.astype(np.long)
        
        # Convert categorical cell types to integer indices
        unique_celltypes, cell_type_indexes = np.unique(cell_types, return_inverse=True)
        
        # Convert to torch tensors (no more .astype calls after this point)
        cell_type_indexes = torch.from_numpy(cell_type_indexes).to(self.device).long()
        obs_t7 = torch.from_numpy(obs_t7).to(self.device).long()
        obs_cre = torch.from_numpy(obs_cre).to(self.device).long()

        # Initialize parameters from checkpoints if provided, otherwise use default initialization
        x0_loaded, x1_loaded, x2_loaded = self.load_checkpoint(x0_checkpoint_path, x1_checkpoint_path, x2_checkpoint_path)
        
        # Initialize x0
        if x0_loaded is not None:
            x0 = x0_loaded
            print("Using x0 from checkpoint")
        elif x0_prior is None:
            x0 = torch.zeros((len(unique_celltypes), obs_t7.shape[1]), device=self.device, dtype=torch.float32)
        elif x0_prior == 'zero_percentage':
            # check the number of non-zero elements in each cell type
            zero_counts = (pd.DataFrame(obs_t7.cpu().numpy()) == 0).groupby(cell_type_indexes.cpu().numpy()).mean()
            # set x0 to the mean of the non-zero elements
            x0 = np.clip(-np.log(zero_counts), min=1e-9)
            # transform to torch.softplus logits
            x0 = np.log(np.exp(x0) - 1)
            x0 = torch.tensor(x0.values, device=self.device, dtype=torch.float32)
        
        # Initialize x1
        if x1_loaded is not None:
            x1 = x1_loaded
            print("Using x1 from checkpoint")
        else:
            x1 = torch.tensor([-1.0, 0], device=self.device, dtype=torch.float32)  # Start with more conservative value
        
        # Initialize x2
        if x2_loaded is not None:
            x2 = x2_loaded
            print("Using x2 from checkpoint")
        else:
            x2 = torch.tensor(np.zeros((len(unique_celltypes), obs_t7.shape[1], 2)), device=self.device, dtype=torch.float32)  # Start with more conservative value
        
        print(f"Running EM algorithm on {self.device}")
        print(f"Initial x0: {x0.cpu().numpy().mean()}")
        print(f"Initial x1: {x1.cpu().numpy()}")
        print(f"Initial x2: {x2.cpu().numpy().mean()}")
        
        # Log initial parameters
        writer.add_scalar('Parameters/x0_mean', x0.cpu().numpy().mean(), 0)
        writer.add_scalar('Parameters/x0_std', x0.cpu().numpy().std(), 0)
        writer.add_scalar('Parameters/x1_0', x1.cpu().numpy()[0], 0)
        writer.add_scalar('Parameters/x1_1', x1.cpu().numpy()[1], 0)
        writer.add_scalar('Parameters/x2_mean', x2.cpu().numpy().mean(), 0)
        writer.add_scalar('Parameters/x2_std', x2.cpu().numpy().std(), 0)
        
        # Initialize for likelihood improvement tracking
        prev_log_likelihood_t7 = None
        prev_log_likelihood_cre = None

        for i in range(max_iter):
            # Check for NaN/inf values before each step
            if torch.isnan(x0).any() or torch.isinf(x0).any() or torch.isnan(x1).any() or torch.isinf(x1).any() or torch.isnan(x2).any() or torch.isinf(x2).any():
                print(f"Warning: NaN/inf detected in parameters at step {i}")
                break
            
            # E-step
            start = time.time()
            posterior_k, log_likelihood_t7 = self.expectation_t7_step(x0, x1, obs_t7, cell_type_indexes)
            mid1 = time.time()
            print(f"Step {i}, T7 E-step time: {mid1 - start:.4f}s, Log-likelihood: {log_likelihood_t7.item():.6f}")
            
            # Log T7 likelihood to TensorBoard
            writer.add_scalar('Likelihood/log_likelihood_t7', log_likelihood_t7.item(), i+1)
            
            # Log T7 likelihood improvement
            if prev_log_likelihood_t7 is not None:
                likelihood_improvement_t7 = log_likelihood_t7.item() - prev_log_likelihood_t7
                writer.add_scalar('Likelihood/likelihood_improvement_t7', likelihood_improvement_t7, i+1)
                print(f"Step {i}, T7 Likelihood improvement: {likelihood_improvement_t7:.6f}")
            prev_log_likelihood_t7 = log_likelihood_t7.item()
            # Check for NaN in posterior
            if torch.isnan(posterior_k).any():
                print(f"Warning: NaN in posterior at step {i}")
                break
            
            if self.use_x0:
                # M-step for x0
                x0, x0_diff = self.maximization_step_x0(x0, cell_type_indexes, posterior_k)
                mid2 = time.time()
                print(f"Step {i}, M-step x0 time: {mid2 - mid1:.4f}s")
                # Check for NaN in posterior
                if torch.isnan(x0).any():
                    print(f"Warning: NaN in posterior at step {i}")
                    break
            else:
                mid2 = time.time()
                x0_diff = x0.clone()
            
            # M-step for x1
            x1, x1_diff = self.maximization_step_x1(x1, obs_t7, posterior_k)
            end = time.time()
            print(f"T7 Step {i}, M-step x1 time: {end - mid2:.4f}s")
            # Check for NaN in posterior
            if torch.isnan(x1).any():
                print(f"Warning: NaN in posterior at T7 step {i}")
                break

            print(f"T7 Step {i}, x0: {x0.cpu().numpy().mean()}, x1: {x1.cpu().numpy()}")

            # check convergence
            if torch.all(torch.abs(x0_diff) / torch.abs(x0) < 1e-5) and torch.all(torch.abs(x1_diff) / torch.abs(x1) < 1e-5):
                print(f"Converged for T7 at step {i}")
                break
            
        # next, fix x0 and update x2
        for i in range(max_iter):
            # Check for NaN/inf values before each step
            if torch.isnan(x0).any() or torch.isinf(x0).any() or torch.isnan(x1).any() or torch.isinf(x1).any() or torch.isnan(x2).any() or torch.isinf(x2).any():
                print(f"Warning: NaN/inf detected in parameters at step {i}")
                break
            
            # E-step
            start = time.time()
            posterior_k, log_likelihood_cre = self.expectation_cre_step(x0, x2, obs_cre, cell_type_indexes)
            mid1 = time.time()
            print(f"Step {i}, CRE E-step time: {mid1 - start:.4f}s, Log-likelihood: {log_likelihood_cre.item():.6f}")
            
            # Log CRE likelihood to TensorBoard
            writer.add_scalar('Likelihood/log_likelihood_cre', log_likelihood_cre.item(), i+1)
            
            # Log CRE likelihood improvement
            if prev_log_likelihood_cre is not None:
                likelihood_improvement_cre = log_likelihood_cre.item() - prev_log_likelihood_cre
                writer.add_scalar('Likelihood/likelihood_improvement_cre', likelihood_improvement_cre, i+1)
                print(f"Step {i}, CRE Likelihood improvement: {likelihood_improvement_cre:.6f}")
            prev_log_likelihood_cre = log_likelihood_cre.item()
            # Check for NaN in posterior
            if torch.isnan(posterior_k).any():
                print(f"Warning: NaN in posterior at step {i}")
                break
            
            # M-step for x2
            x2, x2_diff = self.maximization_step_x2(x2, obs_cre, posterior_k, cell_type_indexes)
            end = time.time()
            print(f"CRE Step {i}, M-step x2 time: {end - mid2:.4f}s")
            # Check for NaN in posterior
            if torch.isnan(x2).any():
                print(f"Warning: NaN in posterior at step {i}")
                break
            
            print(f"CRE Step {i}, x0: {x0.cpu().numpy().mean()}, x1: {x1.cpu().numpy()}, x2: {x2.cpu().numpy().mean()}")

            # check convergence
            if torch.all(torch.abs(x2_diff) / torch.abs(x2) < 1e-5):
                print(f"Converged for CRE at step {i}")
                break
            
        return x0.cpu().numpy(), x1.cpu().numpy(), x2.cpu().numpy()
    
    def load_checkpoint(self, x0_path=None, x1_path=None, x2_path=None):
        """Load model parameters from separate .npy files"""
        x0, x1, x2 = None, None, None
        
        if x0_path is not None:
            x0 = torch.from_numpy(np.load(x0_path)).to(self.device).float()
            print(f"Loaded x0 from {x0_path}")
        
        if x1_path is not None:
            x1 = torch.from_numpy(np.load(x1_path)).to(self.device).float()
            print(f"Loaded x1 from {x1_path}")
        
        if x2_path is not None:
            x2 = torch.from_numpy(np.load(x2_path)).to(self.device).float()
            print(f"Loaded x2 from {x2_path}")
        
        return x0, x1, x2


class STARRFISH:
    def __init__(self, adata: Union[sc.AnnData, str],
                 cre_tag = 'obsm:CRE', t7_tag = 'obsm:T7CRE', celltype_tag='obs:subclass', spatial_tag='obsm:X_spatial', creinfo_tag='uns:CRE_info',
                 atac_cpm: Union[pd.DataFrame, str] = 'Data/ATAC/cpm_peakBysubclass.csv',
                 atac_counts: Union[pd.DataFrame, str] = 'Data/ATAC/count_peakBysubclass.csv',
                 lib_size: Union[pd.DataFrame, str] = 'Data/SFv8_400CRE_nanopore_counts.csv',
                 log_lib_size: bool = True,
                 blacklist_cre: List[str] = []):
        """
        Initialize STARRFISH object for analyzing spatial transcriptomics data with CRE activity.

        Parameters
        ----------
        adata : sc.AnnData or str
            AnnData object or path to .h5ad file containing spatial transcriptomics data
        cre_tag : str, optional
            Tag to access CRE expression data in adata (default: 'obsm:CRE')
        t7_tag : str, optional
            Tag to access T7-CRE expression data (default: 'obsm:T7CRE')
        celltype_tag : str, optional
            Tag to access cell type annotations (default: 'obs:subclass')
        spatial_tag : str, optional
            Tag to access spatial coordinates (default: 'obsm:X_spatial')
        creinfo_tag : str, optional
            Tag to access CRE metadata (default: 'uns:CRE_info')
        atac_cpm : pd.DataFrame or str, optional
            DataFrame or path to ATAC-seq CPM data by cell type
        atac_counts : pd.DataFrame or str, optional
            DataFrame or path to ATAC-seq count data by cell type
        lib_size : pd.DataFrame or str, optional
            DataFrame or path to library size data for normalization
        log_lib_size : bool, optional
            Whether to log-transform library size (default: True)
        blacklist_cre : list of str, optional
            List of CRE IDs to exclude from analysis

        Notes
        -----
        Loads spatial transcriptomics data, processes cell type annotations, loads ATAC-seq data,
        and prepares library size normalization factors. Handles different cell type granularities
        (subclass, class, region) and aggregates ATAC data accordingly.
        """
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
            best_class[best_class.isin(allen_cell_type_nomination['subclass_label'])] = allen_cell_type_nomination['class_label'].groupby(allen_cell_type_nomination['subclass_label']).first().loc[best_class[best_class.isin(allen_cell_type_nomination['subclass_label'])]].values
            self.adata.uns['CRE_info']['best_class'] = best_class
            # just in case, change the subclass to class
            self.adata.uns['CRE_info']['best_subclass_orig'] = self.adata.uns['CRE_info']['best_subclass'].copy()
            self.adata.uns['CRE_info']['best_subclass'] = best_class
        elif celltype_tag == 'obs:region':
            allen_cell_type_nomination = pd.read_excel('Data/abc_atlas/allen_institute_nominature.xlsx', sheet_name='subclass_annotation')
            allen_cell_type_nomination['subclass_label'] = allen_cell_type_nomination['subclass_label'].str.replace('/', '-')
            atac_counts = pd.read_csv('Data/ATAC/count_peakBysubclass.csv', index_col=0)
            atac_counts.columns = atac_counts.columns.str.replace('\\.', '-')
            atac_counts.columns = atac_counts.columns.str.replace('_', ' ')
            # transpose and group by class
            atac_counts = atac_counts.transpose()
            atac_counts_class = allen_cell_type_nomination['neighborhood'].groupby(allen_cell_type_nomination['subclass_label']).first().loc[atac_counts.index]
            atac_counts = atac_counts.groupby(atac_counts_class).sum()
            # norm to cpm
            atac_cpm = atac_counts.div(atac_counts.sum(axis=1), axis=0) * 1e7
            # transpose back
            atac_cpm = atac_cpm.transpose()
            atac_counts = atac_counts.transpose()
            # assign the obs['class'] to adata
            self.adata.obs['region'] = allen_cell_type_nomination['neighborhood'].groupby(allen_cell_type_nomination['subclass_label']).first().loc[self.adata.obs['subclass']].values
            # to be safe, change subclass to class as well
            self.adata.obs['subclass_orig'] = self.adata.obs['subclass'].copy()
            self.adata.obs['subclass'] = self.adata.obs['region']
            # change the cre info
            best_class = self.get_creinfo()['best_subclass'].copy()
            best_class[best_class.isin(allen_cell_type_nomination['subclass_label'])] = allen_cell_type_nomination['neighborhood'].groupby(allen_cell_type_nomination['subclass_label']).first().loc[best_class[best_class.isin(allen_cell_type_nomination['subclass_label'])]].values
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
        self.blacklist_cre = blacklist_cre
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
        """
        Save STARRFISH object to file.

        Parameters
        ----------
        path : str
            Path to save the STARRFISH object (pickle file)
        overwrite_adata : bool, optional
            Whether to overwrite existing AnnData file (default: False)

        Notes
        -----
        Saves the STARRFISH object to a pickle file while optionally saving the AnnData
        object separately to an .h5ad file. The AnnData is temporarily removed before
        pickling to reduce file size.
        """
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
        """
        Load STARRFISH object from file.

        Parameters
        ----------
        path : str
            Path to the saved STARRFISH pickle file
        adata : sc.AnnData or str, optional
            AnnData object or path to .h5ad file to load

        Returns
        -------
        STARRFISH
            Loaded STARRFISH object with data

        Notes
        -----
        Loads a previously saved STARRFISH object from pickle file and optionally
        loads the associated AnnData object.
        """
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
        """
        Load AnnData object from file.

        Parameters
        ----------
        adata_path : str
            Path to the .h5ad file

        Notes
        -----
        Loads an AnnData object from disk and stores it in the STARRFISH object.
        """
        # load the adata
        adata = sc.read(adata_path)
        self.adata: sc.AnnData = adata
    
    def load_cpm(self, cpm_path: str, attr_to_add: str = 'atac_cpm'):
        """
        Load CPM (counts per million) data from CSV file.

        Parameters
        ----------
        cpm_path : str
            Path to CSV file containing CPM data
        attr_to_add : str, optional
            Attribute name to store the loaded CPM data (default: 'atac_cpm')

        Notes
        -----
        Loads and processes CPM data, matches it to CRE information, and stores
        as a transposed DataFrame where rows are cell types and columns are CREs.
        """
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
        """
        Load library size data for CRE normalization.

        Parameters
        ----------
        lib_size_path : str
            Path to CSV file containing library size data
        log_transform : bool, optional
            Whether to apply log1p transformation to library sizes (default: True)

        Notes
        -----
        Loads library size data, matches to CRE information, and optionally log-transforms.
        Stores both raw (`lib_size_raw`) and processed (`lib_size`) versions.
        """
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
        """
        Retrieve data from AnnData object using tag notation.

        Parameters
        ----------
        tag : str
            Tag in format 'attribute:key' (e.g., 'obs:subclass', 'obsm:CRE')

        Returns
        -------
        pd.DataFrame or pd.Series or None
            Data from the specified tag location, or None if tag doesn't exist

        Notes
        -----
        Parses tag string to access nested attributes in AnnData object.
        Tag format is 'attribute:key' where attribute is 'obs', 'obsm', 'uns', etc.
        """
        # get the CREs
        tag_attr = tag.split(':')[0]
        tag_col = tag.split(':')[1]
        if tag_col not in self.adata.__getattribute__(tag_attr).keys():
            return None
        return self.adata.__getattribute__(tag_attr)[tag_col]
    
    def get_cre_expression(self) -> pd.DataFrame:
        """
        Get CRE expression data for all cells.

        Returns
        -------
        pd.DataFrame
            DataFrame with cells as rows and CREs as columns containing expression values

        Notes
        -----
        Retrieves CRE expression data from the location specified by self.cre_tag.
        """
        return self.get_tag(self.cre_tag)

    def get_t7_expression(self) -> pd.DataFrame:
        """
        Get T7-CRE expression data for all cells.

        Returns
        -------
        pd.DataFrame or None
            DataFrame with cells as rows and CREs as columns, or None if T7 data unavailable

        Notes
        -----
        T7-CRE is used to measure transfection/infection efficiency. Returns None if
        t7_tag is not set or T7 data was not included in the dataset.
        """
        if not hasattr(self, 't7_tag') or self.t7_tag is None:
            return None
        return self.get_tag(self.t7_tag)

    def get_rna_expression(self) -> pd.DataFrame:
        """
        Get RNA expression data for all cells.

        Returns
        -------
        pd.DataFrame
            DataFrame with cells as rows and genes as columns containing RNA counts

        Notes
        -----
        Retrieves raw RNA expression data from 'obsm:X_raw' in the AnnData object.
        """
        # get the RNA expression
        return self.get_tag('obsm:X_raw')

    def get_k_nearest_neighbors(self, cell_id, k=10, spatial_tag='obsm:X_spatial') -> pd.DataFrame:
        """
        Find k nearest neighbor cells based on spatial coordinates.

        Parameters
        ----------
        cell_id : str
            Cell identifier to find neighbors for
        k : int, optional
            Number of nearest neighbors to find (default: 10)
        spatial_tag : str, optional
            Tag specifying spatial coordinate location (default: 'obsm:X_spatial')

        Returns
        -------
        pd.DataFrame
            DataFrame with neighbor cell IDs as index and columns for distance, X, and Y coordinates,
            sorted by distance

        Notes
        -----
        Uses Euclidean distance to find spatial neighbors. The query cell itself is excluded
        from the results.
        """
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
        """
        Get normalized CRE expression data.

        Parameters
        ----------
        cell_types_to_use : list, optional
            List of cell types to include (default: all cell types)
        normalize_by_cell_rna : bool, optional
            Normalize by RNA content per cell (default: True)
        normalize_by_volume : bool, optional
            Normalize by cell volume (default: True)
        log_transform : bool, optional
            Apply log1p transformation (default: False)

        Returns
        -------
        tuple of (pd.DataFrame, pd.Series)
            Normalized CRE expression matrix and cell type labels for included cells

        Notes
        -----
        Performs cell-level normalization to account for differences in RNA content and/or
        cell volume before analyzing CRE activity.
        """
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

    def get_celltypes(self, celltype_tag=None) -> pd.Series:
        """
        Get cell type annotations for all cells.

        Parameters
        ----------
        celltype_tag : str, optional
            Tag specifying cell type location (default: uses self.celltype_tag)

        Returns
        -------
        pd.Series
            Series with cell IDs as index and cell type labels as values

        Notes
        -----
        Retrieves cell type annotations from the AnnData object. By default uses
        the tag specified during initialization.
        """
        # get the cell types
        if celltype_tag is None:
            celltype_tag = self.celltype_tag
        return self.get_tag(celltype_tag)
    
    def get_cre_celltypes(self, celltypes, celltype_tag=None) -> tuple[pd.DataFrame, pd.Series]:
        """
        Get CRE expression data for specific cell types.

        Parameters
        ----------
        celltypes : list
            List of cell type labels to filter for
        celltype_tag : str, optional
            Tag specifying cell type location (default: uses self.celltype_tag)

        Returns
        -------
        tuple of (pd.DataFrame, pd.Series)
            Filtered CRE expression matrix and cell type labels for matching cells

        Notes
        -----
        Filters cells to include only those matching the specified cell types.
        """
        # get cre for the cell types
        cres = self.get_cre_expression().copy()
        celltypes_orig = self.get_celltypes(celltype_tag=celltype_tag).copy()
        # get the cre for the cell types
        cre_celltypes = cres[celltypes_orig.isin(celltypes)]
        celltypes = celltypes_orig[celltypes_orig.isin(celltypes)]
        return cre_celltypes, celltypes

    def get_cre_rna_celltypes(self, celltypes) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
        """
        Get both CRE and RNA expression data for specific cell types.

        Parameters
        ----------
        celltypes : list
            List of cell type labels to filter for

        Returns
        -------
        tuple of (pd.DataFrame, pd.DataFrame, pd.Series)
            Filtered CRE expression matrix, RNA expression matrix, and cell type labels
            for matching cells

        Notes
        -----
        Filters cells to include only those matching the specified cell types and returns
        both CRE and RNA data for downstream normalization or analysis.
        """
        # get cre for the cell types
        cres = self.get_cre_expression().copy()
        rna = self.get_rna_expression().copy()
        celltypes_orig = self.get_celltypes().copy()
        # get the cre for the cell types
        cre_celltypes = cres[celltypes_orig.isin(celltypes)]
        rna_celltypes = rna[celltypes_orig.isin(celltypes)]
        celltypes = celltypes_orig[celltypes_orig.isin(celltypes)]
        return cre_celltypes, rna_celltypes, celltypes

    def _preprocess_expressions(self, cell_types_to_use=None, normalize_by_cell_rna=False,
                                normalize_by_cell_volume=False, normalize_by_cell_t7=False,
                                filter_by_cell_t7=None, binarize_t7=False, log_transform=False,
                                log_func='log1p'):
        """
        Common preprocessing logic for CRE, RNA, T7 expressions.

        This method extracts the common preprocessing steps used in both
        fold_change_test and average_bootstrap_test methods.

        Parameters:
        -----------
        log_func : str
            Which log function to use: 'log1p' (default) or 'log'

        Returns:
        --------
        tuple: (cre_cells_expression, rna_cells_expression, cell_types_to_use, volm, t7_cells_expression)
        """
        # Get data
        if cell_types_to_use is not None:
            cre_cells_expression, rna_cells_expression, cell_types_to_use = self.get_cre_rna_celltypes(cell_types_to_use)
        else:
            cre_cells_expression = self.get_cre_expression().copy()
            rna_cells_expression = self.get_rna_expression().copy()
            cell_types_to_use = self.get_celltypes()

        volm = self.get_tag('obs:volm').copy().loc[cell_types_to_use.index]
        rna_cells_expression = pd.DataFrame(rna_cells_expression, index=cell_types_to_use.index)

        # Get T7 expression
        t7_cells_expression = self.get_t7_expression()
        if t7_cells_expression is not None:
            t7_cells_expression = t7_cells_expression.loc[cre_cells_expression.index]
            if binarize_t7:
                t7_cells_expression = (t7_cells_expression > 0).astype(float)

        # Apply cell-level normalizations
        if normalize_by_cell_rna and normalize_by_cell_volume:
            rna_per_volume = rna_cells_expression / volm.values.reshape(-1, 1)
            cre_cells_expression = cre_cells_expression / rna_per_volume.mean(axis=1).values.reshape(-1, 1)
        elif normalize_by_cell_rna:
            cre_cells_expression = cre_cells_expression / rna_cells_expression.mean(axis=1).values.reshape(-1, 1)
        elif normalize_by_cell_volume:
            cre_cells_expression = cre_cells_expression / volm.values.reshape(-1, 1)

        # Apply T7 normalization
        if normalize_by_cell_t7:
            assert t7_cells_expression is not None, "t7_cells_expression required when normalize_by_cell_t7=True"
            cre_cells_expression = cre_cells_expression / t7_cells_expression
            # Handle inf/nan values
            cre_cells_expression[np.isinf(cre_cells_expression)] = np.nan
            # If normalize_by_cell_t7 is a threshold value, filter cells below it
            if isinstance(normalize_by_cell_t7, (int, float)):
                cre_cells_expression[t7_cells_expression < normalize_by_cell_t7] = np.nan
            else:
                # For fold_change_test: fillna(0) behavior
                cre_cells_expression = cre_cells_expression.fillna(0)

        # Apply T7 filtering (separate from normalization)
        if filter_by_cell_t7 is not None and t7_cells_expression is not None:
            cre_cells_expression[t7_cells_expression < filter_by_cell_t7] = np.nan
            t7_cells_expression[t7_cells_expression < filter_by_cell_t7] = np.nan

        # Apply log transform
        if log_transform:
            if log_func == 'log':
                cre_cells_expression = np.log(cre_cells_expression)
            else:  # log1p
                cre_cells_expression = np.log1p(cre_cells_expression)

        return cre_cells_expression, rna_cells_expression, cell_types_to_use, volm, t7_cells_expression

    def get_creinfo(self) -> pd.DataFrame:
        """
        Get metadata information for all CREs.

        Returns
        -------
        pd.DataFrame
            DataFrame with CRE IDs as index and metadata columns (e.g., genomic coordinates,
            best cell type, labeling type)

        Notes
        -----
        Retrieves CRE metadata from the location specified by self.cre_info_tag.
        Contains information such as enhancer coordinates, target genes, and cell type specificity.
        """
        # get the CRE info
        return self.get_tag(self.cre_info_tag)
    
    def get_negative_control_cres(self) -> pd.Series:
        """
        Get negative control CREs.

        Returns
        -------
        pd.Index
            Index of CRE IDs labeled as negative controls

        Notes
        -----
        Negative controls are CREs that should not show cell type-specific activity
        and are used as background references in statistical tests.
        """
        # get the negative control cres
        cres = self.get_creinfo().copy()
        cres = cres[cres['labeling_type'] == 'negative control']
        return cres.index
    
    def get_positive_control_cres(self, cell_type, use='define') -> pd.Series:
        """
        Get positive control CREs for a given cell type.

        Parameters
        ----------
        cell_type : str
            Cell type label to find positive control CREs for
        use : str, optional
            Method to define positive controls: 'define' (CRE metadata), 'atac-peak',
            'h3k27ac-peak', 'h3k4me1-peak', 'chromatin-a', or 'chromatin-o' (default: 'define')

        Returns
        -------
        pd.Index or None
            Index of CRE IDs expected to be active in the given cell type, or None if unavailable

        Notes
        -----
        Positive controls are CREs expected to show activity in the specified cell type based
        on various epigenomic or experimental evidence sources.
        """
        if use == 'define':
            cres = self.get_creinfo().copy()
            cres = cres[cres['best_subclass'] == cell_type]
            return cres.index
        elif use == 'atac-peak':
            return _load_csv_and_filter(DataPaths.CRE_ATAC_PEAKS, cell_type, axis='row')
        elif use == 'h3k27ac-peak':
            return _load_csv_and_filter(DataPaths.CRE_H3K27AC_PEAKS, cell_type, axis='row')
        elif use == 'h3k4me1-peak':
            return _load_csv_and_filter(DataPaths.CRE_H3K4ME1_PEAKS, cell_type, axis='row')
        elif use == 'chromatin-a':
            return _load_csv_and_filter(DataPaths.CRE_CHROMATIN_STATE_A, cell_type, axis='row')
        elif use == 'chromatin-o':
            cres_o = _load_csv_and_filter(DataPaths.CRE_CHROMATIN_STATE_O, cell_type, axis='row')
            cres_a = _load_csv_and_filter(DataPaths.CRE_CHROMATIN_STATE_A, cell_type, axis='row')
            if cres_o is None or cres_a is None:
                return None
            return cres_o.union(cres_a)
    
    def get_positive_control_celltypes(self, cre, use='define') -> pd.Series:
        """
        Get positive control cell types for a given CRE.

        Parameters
        ----------
        cre : str
            CRE ID to find positive control cell types for
        use : str, optional
            Method to define positive controls: 'define' (CRE metadata), 'atac-peak',
            'h3k27ac-peak', 'h3k4me1-peak', 'chromatin-a', or 'chromatin-o' (default: 'define')

        Returns
        -------
        pd.Series or pd.Index or None
            Cell type labels where the CRE is expected to be active, or None if unavailable

        Notes
        -----
        Returns cell types where the given CRE should show activity based on various
        epigenomic or experimental evidence sources.
        """
        if use == 'define':
            cell_types = self.get_creinfo().copy()
            cell_types = cell_types['best_subclass'].loc[cre]
            return pd.Series(cell_types)
        elif use == 'atac-peak':
            return _load_csv_and_filter(DataPaths.CRE_ATAC_PEAKS, cre, axis='col')
        elif use == 'h3k27ac-peak':
            return _load_csv_and_filter(DataPaths.CRE_H3K27AC_PEAKS, cre, axis='col')
        elif use == 'h3k4me1-peak':
            return _load_csv_and_filter(DataPaths.CRE_H3K4ME1_PEAKS, cre, axis='col')
        elif use == 'chromatin-a':
            return _load_csv_and_filter(DataPaths.CRE_CHROMATIN_STATE_A, cre, axis='col')
        elif use == 'chromatin-o':
            celltypes_o = _load_csv_and_filter(DataPaths.CRE_CHROMATIN_STATE_O, cre, axis='col')
            celltypes_a = _load_csv_and_filter(DataPaths.CRE_CHROMATIN_STATE_A, cre, axis='col')
            if celltypes_o is None or celltypes_a is None:
                return None
            return celltypes_o.union(celltypes_a)
    
    def get_atac_z_cres(self, cell_type, z=2) -> pd.Series:
        """
        Get CREs with ATAC-seq signal above z-score threshold for a cell type.

        Parameters
        ----------
        cell_type : str
            Cell type label to query ATAC-seq data for
        z : float, optional
            Z-score threshold for ATAC signal (default: 2)

        Returns
        -------
        pd.Index or None
            Index of CRE IDs with ATAC-seq z-score > threshold, or None if cell type not found

        Notes
        -----
        Identifies CREs with strong chromatin accessibility in the specified cell type
        by computing z-scores of log-transformed ATAC CPM values.
        """
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
                  binarize_t7=False, norm_by_t7_cell_type_mean=True, norm_by_t7_cell_type_sum=False, norm_by_t7_single_cell=False,
                  log=True, calibrate=None, aggregate_background_celltypes=False,
                  cell_types_to_use=None, cell_types_to_visualize=None, cell_types_tag=None,
                  nmin=None, nmax=None, sz_background=3, sz_min=5, sz_max=30, 
                  scale_size_by: Literal['counts', 'celltype_number']='counts',  
                  cmap_name='Reds', use_celltype_cmap=False, celltype_cmap=None,
                  x_region=None, y_region=None, select_region_by_best_celltype=False, 
                  show_celltypes=True, show_scalebar=True, show_title=True,
                  transpose=1, flipx=1, flipy=1, smooth_k=None, figsize=(30, 10)):
        tag = self.spatial_tag.split(':')[1]
        Xcells = self.adata.obsm[tag][:, ::transpose] * [flipx, flipy]
        # Track cell types for coloring later
        celltypes = self.get_celltypes(cell_types_tag).values.copy()
        # get best cell type
        if use == 'CRE' or use == 'T7CRE':
            if cell_types_to_visualize is None:
                assert aggregate_background_celltypes == False, "If cell_types_to_visualize is None, aggregate_background_celltypes must be False"
                best_celltype = [self.adata.uns['CRE_info'].loc[gene, 'best_subclass']]
            else:
                best_celltype = list(cell_types_to_visualize)
        else:
            if cell_types_to_visualize is not None:
                best_celltype = list(cell_types_to_visualize)
            else:
                best_celltype = []
        # Get expression data
        if use == 'X':
            gene_idx = list(self.adata.var.index).index(gene)
            cts = self.adata.X[:, gene_idx].copy()
        else:
            cts = self.adata.obsm[use][gene].copy()
        # if average_by_celltype, then average by cell type
        if average_by_celltype:
            # get the cell types
            cell_type_cts = cts.groupby(self.get_celltypes(cell_types_tag)).mean()
            # only assign to non-zero cts
            zero_cts = cts == 0
            cts = cell_type_cts.loc[self.get_celltypes(cell_types_tag)].copy()
            # rename the index
            cts.index = self.get_celltypes(cell_types_tag).index
            # set the zero counts to 0
            cts[zero_cts] = 0
        negative_control_cres = self.get_negative_control_cres()
        # remove black list cres
        negative_control_cres = [cre for cre in negative_control_cres if cre not in self.blacklist_cre]   
        # get t7 expression
        t7_expression = self.get_t7_expression()
        if binarize_t7 and t7_expression is not None:
            t7_expression = (t7_expression > 0).astype(float)
        # get the background cell types
        background_celltypes = self.get_celltypes(cell_types_tag)[~self.get_celltypes(cell_types_tag).isin(best_celltype)]
        if cell_types_to_use is not None:
            background_celltypes = background_celltypes[background_celltypes.isin(cell_types_to_use)]
        # if norm_by_negative_control, then normalize by negative control
        if norm_by_negative_control_cell_type_mean:
            negative_control_counts = self.get_cre_expression()[negative_control_cres].sum(axis=1).groupby(self.get_celltypes(cell_types_tag)).mean()
            if aggregate_background_celltypes:
                # average the negative control counts for background cell types
                background_negative_control_counts = self.get_cre_expression().loc[background_celltypes.index][negative_control_cres].sum(axis=1).mean()
                # assign the background negative control counts to all background cell types
                negative_control_counts.loc[background_celltypes.unique()] = background_negative_control_counts
            if norm_by_t7_cell_type_mean or norm_by_t7_cell_type_sum or norm_by_t7_single_cell:
                negative_control_t7 = t7_expression[negative_control_cres].sum(axis=1).groupby(self.get_celltypes(cell_types_tag)).mean()
                if aggregate_background_celltypes:
                    background_negative_control_t7 = t7_expression.loc[background_celltypes.index][negative_control_cres].sum(axis=1).mean()
                    # assign the background negative control t7 to all background cell types
                    negative_control_t7.loc[background_celltypes.unique()] = background_negative_control_t7
                # fill a 0.5 value to avoid inf results
                negative_control_t7[negative_control_t7 == 0] = 0.5
                negative_control_counts = negative_control_counts / negative_control_t7
            norm_factor = negative_control_counts.loc[self.get_celltypes(cell_types_tag)]
            cts = cts / norm_factor.values
        if norm_by_negative_control_cell_type_sum:
            negative_control_counts = self.get_cre_expression()[negative_control_cres].sum(axis=1).groupby(self.get_celltypes(cell_types_tag)).sum()
            if aggregate_background_celltypes:
                # get the background cell types
                background_celltypes = self.get_celltypes(cell_types_tag)[~self.get_celltypes(cell_types_tag).isin(best_celltype)]
                if cell_types_to_use is not None:
                    background_celltypes = background_celltypes[background_celltypes.isin(cell_types_to_use)]
                # average the negative control counts for background cell types
                background_negative_control_counts = self.get_cre_expression().loc[background_celltypes.index][negative_control_cres].sum(axis=1).sum()
                # assign the background negative control counts to all background cell types
                negative_control_counts.loc[background_celltypes.unique()] = background_negative_control_counts
            if norm_by_t7_cell_type_sum or norm_by_t7_cell_type_mean or norm_by_t7_single_cell:
                negative_control_t7 = t7_expression[negative_control_cres].sum(axis=1).groupby(self.get_celltypes(cell_types_tag)).sum()
                if aggregate_background_celltypes:
                    background_negative_control_t7 = t7_expression.loc[background_celltypes.index][negative_control_cres].sum(axis=1).sum()
                    # assign the background negative control t7 to all background cell types
                    negative_control_t7.loc[background_celltypes.unique()] = background_negative_control_t7
                # fill a 0.5 value to avoid inf results
                negative_control_t7[negative_control_t7 == 0] = 0.5
                negative_control_counts = negative_control_counts / negative_control_t7
            norm_factor = negative_control_counts.loc[self.get_celltypes(cell_types_tag)]
            cts = cts / norm_factor.values
        if norm_by_negative_control_single_cell:
            negative_control_counts = self.get_cre_expression()[negative_control_cres].sum(axis=1)
            cts = cts / negative_control_counts.values
        if norm_by_t7_cell_type_mean and t7_expression is not None:
            t7_counts = t7_expression[gene].groupby(self.get_celltypes(cell_types_tag)).mean()
            norm_factor = t7_counts.loc[self.get_celltypes(cell_types_tag)]
            norm_factor.index = self.get_celltypes(cell_types_tag).index
            if aggregate_background_celltypes:
                background_t7_counts = t7_expression.loc[background_celltypes.index][gene].mean()
                # assign the background t7 counts to all background cell types
                norm_factor.loc[background_celltypes.index] = background_t7_counts
            cts = cts / norm_factor.values
        if norm_by_t7_cell_type_sum and t7_expression is not None:
            t7_counts = t7_expression[gene].groupby(self.get_celltypes(cell_types_tag)).sum()
            norm_factor = t7_counts.loc[self.get_celltypes(cell_types_tag)]
            norm_factor.index = self.get_celltypes(cell_types_tag).index
            if aggregate_background_celltypes:
                background_t7_counts = t7_expression.loc[background_celltypes.index][gene].sum()
                # assign the background t7 counts to all background cell types
                norm_factor.loc[background_celltypes.index] = background_t7_counts
            cts = cts / norm_factor.values
        if norm_by_t7_single_cell and t7_expression is not None:
            t7_counts = t7_expression[gene]
            cts = cts / t7_counts.values
        if log:
            cts = np.log1p(cts)
        # do calibration if specified
        if calibrate is not None:
            cts = cts - calibrate
        if cell_types_to_use is not None:
            # only cts for the cell types to use
            cts[~self.get_celltypes(cell_types_tag).isin(cell_types_to_use)] = np.nan
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
                celltype_idx = self.get_celltypes(cell_types_tag) == celltype
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
            celltypes = celltypes[select_region]
        if y_region is not None:
            select_region = (Xcells[:, 1] > y_region[0]) & (Xcells[:, 1] < y_region[1])
            Xcells = Xcells[select_region]
            cts = cts[select_region]
            celltypes = celltypes[select_region]
        # filter out nmin
        cts_background = cts[self.get_celltypes(cell_types_tag).isin(background_celltypes)]
        cts_foreground = cts[self.get_celltypes(cell_types_tag).isin(best_celltype)]
        if nmin is not None:
            cts[cts < nmin] = nmin
        else:
            nmin=cts_background[cts_background > 0].min() if np.any(cts_background > 0) else 0
        if nmax is not None:
            cts[cts > nmax] = nmax
        else:
            if cts_foreground.size == 0:
                nmax = cts.max()
            else:
                nmax=cts_foreground.max()
        ncts = np.clip((cts-nmin)/(nmax-nmin), 0, 1)
        if scale_size_by == 'counts':
            size = sz_min + ncts * (sz_max - sz_min)
        elif scale_size_by == 'celltype_number':
            # get the number of cell types
            celltype_number = self.get_celltypes(cell_types_tag).value_counts().loc[self.get_celltypes(cell_types_tag)].values
            # if not in cell_types_to_use, then set to 0
            celltype_number[~self.get_celltypes(cell_types_tag).isin(cell_types_to_use)] = 0
            # normalize the celltype_number
            celltype_number = celltype_number.max() - celltype_number + 1
            celltype_number = np.clip(celltype_number / celltype_number.max(), 0, 1)
            size = sz_min + celltype_number * (sz_max - sz_min)

        # Create custom colormap from white to light red for better contrast
        from matplotlib.colors import LinearSegmentedColormap
        custom_cmap = LinearSegmentedColormap.from_list('white_to_red', ['#FFFFFF', '#FF6B6B'])
        cmap = custom_cmap(ncts)

        # Create single figure and axes
        if use == 'CRE' or use == 'T7CRE' or use == 'T7Sum':
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
                cluster_color_map = plot_cluster_scdata(
                    self.adata, clusters=best_celltype, use=cell_types_tag.split(':')[1] if cell_types_tag is not None else 'subclass',
                    transpose=transpose, flipx=flipx, flipy=flipy,
                    x_region=x_region, y_region=y_region, cmap=celltype_cmap,
                    sbig=20, small=3, ax=ax_ctypes, plot_legend=show_title, show_title=show_title)
            else:
                fig = plt.figure(figsize=figsize, facecolor='k')
                gs = fig.add_gridspec(1, 2, width_ratios=[0.95, 0.05], wspace=0.05)
                ax_main = fig.add_subplot(gs[0])
                ax_cbar = fig.add_subplot(gs[1])
                # Create cluster_color_map for all cell types in cell_types_to_visualize or best_celltype
                cluster_color_map = {}
                if celltype_cmap is None:
                    celltype_cmap = self.adata.uns['cmap']
                clusters_to_map = best_celltype if cell_types_to_visualize is None else list(cell_types_to_visualize)
                for i, cluster in enumerate(clusters_to_map):
                    if isinstance(celltype_cmap, dict):
                        if cluster in celltype_cmap.keys():
                            cluster_color_map[cluster] = celltype_cmap[cluster]
                        else:
                            cluster_color_map[cluster] = list(celltype_cmap.values())[i % len(celltype_cmap)]
                    else:
                        cluster_color_map[cluster] = celltype_cmap[i % len(celltype_cmap)]
            if show_title:
                ax_main.set_title(f'{gene}', color='white', fontsize=20)
            ax_main.set_facecolor('black')
            # Plot data
            if calibrate is not None:
                cell_with_genes = np.where(cts > -calibrate)[0]
            else:
                cell_with_genes = np.where(cts > 0)[0]
            # first plot cells without genes, then plot cells with genes
            ax_main.scatter(Xcells[:, 0], Xcells[:, 1], c='grey', s=sz_background, marker='.', alpha=0.7, rasterized=True, edgecolors='none')
            # scale the alpha values for the points based on counts
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_alpha = scaler.fit_transform(ncts.reshape(-1,1)).flatten()
            # np clip the scaled_alpha to [0, 1]
            scaled_alpha = np.clip(scaled_alpha, 0, 1)

            # Map cell types to colors using the cluster_color_map
            cell_colors = []
            for ct in celltypes:
                if ct in cluster_color_map:
                    cell_colors.append(cluster_color_map[ct])
                else:
                    # Use a harmless color as default if cell type not in cluster_color_map
                    cell_colors.append('#fabed4')
            cell_colors = np.array(cell_colors)

            # plot the CRE counts
            # ax_main.scatter(Xcells[cell_with_genes, 0], Xcells[cell_with_genes, 1], c='#00FF00', sizes=size[cell_with_genes], alpha=scaled_alpha[cell_with_genes], rasterized=True)
            if use_celltype_cmap:
                ax_main.scatter(Xcells[:, 0], Xcells[:, 1], c=cell_colors, sizes=size, alpha=scaled_alpha, rasterized=True, edgecolors='none')
            else:
                # use green
                ax_main.scatter(Xcells[cell_with_genes, 0], Xcells[cell_with_genes, 1], c='#00FF00', s=size[cell_with_genes], alpha=scaled_alpha[cell_with_genes], rasterized=True, edgecolors='none')
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

                # Get colors from the same colormap as main plot
                legend_colors = custom_cmap(legend_vals)

                # Plot circles in ax_cbar
                dot_spacing = 0.08  # smaller = tighter packing
                for i, (val, sz, alpha, color) in enumerate(zip(legend_cts, legend_sizes, legend_alphas, legend_colors)):
                    x = i * dot_spacing
                    ax_cbar.scatter(x, 0.25, s=sz, alpha=alpha, color=color, edgecolors='none')

                # Add only min and max labels
                ax_cbar.text(-0.3, 0.25, f'{legend_cts[0]:.2f}', va='center', ha='center', color='white', fontsize=8)
                ax_cbar.text((len(legend_cts)-1) * dot_spacing + 0.3, 0.25, f'{legend_cts[-1]:.2f}', va='center', ha='center', color='white', fontsize=8)

                # Set limits and aesthetics
                ax_cbar.set_xlim(0, 2)
                ax_cbar.set_xlim(-0.3, (len(legend_cts)-1) * dot_spacing + 0.3)
                ax_cbar.set_ylim(0, 1.5)  # Enough vertical space for dots + labels
                # ax_cbar.set_ylim(-0.5, len(legend_cts) - 0.5)
                ax_cbar.text(0.5 * (len(legend_cts)-1) * dot_spacing, 0.4, 'Normalized Counts',
                ha='center', va='top', color='white', fontsize=9)

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
            # Create figure with colorbar axis
            fig = plt.figure(figsize=(10, 10), facecolor='black')
            gs = fig.add_gridspec(2, 1, height_ratios=[0.95, 0.05], hspace=0.05)
            ax = fig.add_subplot(gs[0])
            ax_cbar = fig.add_subplot(gs[1])

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

                # Get colors from the same colormap as main plot
                legend_colors = custom_cmap(legend_vals)

                # Plot circles in ax_cbar
                dot_spacing = 0.015  # smaller = tighter packing
                for i, (val, sz, alpha, color) in enumerate(zip(legend_cts, legend_sizes, legend_alphas, legend_colors)):
                    x = i * dot_spacing
                    ax_cbar.scatter(x, 0.75, s=sz*20, alpha=alpha, color=color, edgecolors='none')

                # Add only min and max labels
                ax_cbar.text(-0.05, 0.75, f'{legend_cts[0]:.2f}', va='center', ha='center', color='white', fontsize=8)
                ax_cbar.text((len(legend_cts)-1) * dot_spacing + 0.05, 0.75, f'{legend_cts[-1]:.2f}', va='center', ha='center', color='white', fontsize=8)

                # Set limits and aesthetics
                ax_cbar.set_xlim(0, 2)
                ax_cbar.set_xlim(-0.3, (len(legend_cts)-1) * dot_spacing + 0.3)
                ax_cbar.set_ylim(0, 1.5)  # Enough vertical space for dots + labels
                # ax_cbar.set_ylim(-0.5, len(legend_cts) - 0.5)
                ax_cbar.text(0.5 * (len(legend_cts)-1) * dot_spacing, 0.1, 'Normalized Counts',
                ha='center', va='top', color='white', fontsize=9)

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
        """
        Perform Fisher's exact test for CRE activation enrichment in cell types.

        Parameters
        ----------
        cell_types_to_use : list, optional
            List of cell types to include in analysis (default: all cell types)
        activate_threshold : float, optional
            Expression threshold to consider a CRE as activated (default: 2)
        infect_threshold : float, optional
            T7 expression threshold to consider a cell as infected (default: 1)

        Returns
        -------
        dict
            Dictionary containing:
            - 'activity': DataFrame with cell type x CRE enrichment p-values
            - 'config': Configuration parameters used
            - Additional statistics from the test

        Notes
        -----
        Tests whether each CRE shows significant enrichment of activation in each cell type
        compared to background using Fisher's exact test. Results are cached for reuse.
        """
        config = {
            'cell_types_to_use': cell_types_to_use,
            'infect_threshold': infect_threshold,
            'activate_threshold': activate_threshold
        }

        # Check for cached results
        cached_result = _check_cached_result(self, 'fisher_exact_test_results', 'fisher_exact_test_configs', config)
        if cached_result is not None:
            return cached_result

        # Load data
        if cell_types_to_use is not None:
            cre_celltypes_expression, cell_types_to_use = self.get_cre_celltypes(cell_types_to_use)
        else:
            cre_celltypes_expression = self.get_cre_expression().copy()
            cell_types_to_use = self.get_celltypes()

        # Filter infected cells
        infected = ((cre_celltypes_expression >= infect_threshold).sum(axis=1) > 0)
        cre_celltypes_expression = cre_celltypes_expression[infected]
        cell_types_to_use = cell_types_to_use[infected]
        activated = cre_celltypes_expression >= activate_threshold

        # Initialize result DataFrames
        p_value, q_value, precision, recall, foldchange, precision_n, recall_n = (
            pd.DataFrame(index=cell_types_to_use.unique(), columns=cre_celltypes_expression.columns) for _ in range(7)
        )

        # Run Fisher's exact test for each cell type
        for cell_type in cell_types_to_use.unique():
            celltype_activated = activated.loc[cell_types_to_use == cell_type]
            noncelltype_activated = activated.loc[cell_types_to_use != cell_type]

            # Create contingency table components
            FF = (~noncelltype_activated).sum(axis=0)
            FT = (~celltype_activated).sum(axis=0)
            TF = (noncelltype_activated).sum(axis=0)
            TT = (celltype_activated).sum(axis=0)

            # Run Fisher's exact test for each CRE
            for cre in cre_celltypes_expression.columns:
                oddsratio, p = stats.fisher_exact([[int(FF[cre]), int(FT[cre])], [int(TF[cre]), int(TT[cre])]])
                p_value.loc[cell_type, cre] = p
                foldchange.loc[cell_type, cre] = oddsratio

            # Calculate precision, recall
            precision.loc[cell_type] = TT / (TT + TF)
            recall.loc[cell_type] = TT / (TT + FT)
            precision_n.loc[cell_type] = TT + TF
            recall_n.loc[cell_type] = TT + FT
            q_value.loc[cell_type] = multitest.multipletests(p_value.loc[cell_type], method='fdr_bh')[1]

        # Assign metrics to cre_info based on best_subclass
        cre_info = self.get_creinfo().copy()
        result_dfs = {
            'p_value': p_value, 'q_value': q_value, 'precision': precision,
            'recall': recall, 'precision_n': precision_n, 'recall_n': recall_n,
            'foldchange': foldchange
        }
        cre_info = _assign_cre_info_from_best_subclass(cre_info, result_dfs,
                                                       ['p_value', 'q_value', 'precision', 'recall',
                                                        'precision_n', 'recall_n', 'foldchange'])

        # Calculate entropy
        for cre in cre_info.index:
            cre_info.loc[cre, 'entropy'] = stats.entropy(precision[cre].astype(float))

        # Store results
        fisher_exact_test_result = {
            'cre_info': cre_info, 'p_value': p_value, 'q_value': q_value,
            'precision': precision, 'recall': recall, 'precision_n': precision_n,
            'recall_n': recall_n, 'foldchange': foldchange
        }
        _store_result(self, 'fisher_exact_test_results', 'fisher_exact_test_configs',
                     fisher_exact_test_result, config)

        return fisher_exact_test_result
    
    def estimate_activate_threshold_array(self, activate_threshold, cre_celltypes_expression, cell_types_to_use) -> np.ndarray:
        # Pre-compute unique cell types to avoid redundant calls
        unique_cell_types = cell_types_to_use.unique()

        # if activate_threshold is "celltype", then set cell type specific threshold
        if activate_threshold == 'celltype_mean_2std':
            # Use dictionary for efficient collection of thresholds
            threshold_dict = {}
            for celltype in unique_cell_types:
                # get the cre_expression for the cell type
                celltype_cre_expr_flattened = cre_celltypes_expression[cell_types_to_use == celltype].values.flatten()
                # filter out zero, calculate mean and std
                cell_type_cre_expr_flattened = np.log10(celltype_cre_expr_flattened[celltype_cre_expr_flattened > 0])
                mean = cell_type_cre_expr_flattened.mean()
                std = cell_type_cre_expr_flattened.std()
                # set the threshold
                threshold_dict[celltype] = np.power(10, mean + 2*std)
            # Map to original cell types order
            activate_threshold_array = np.array([threshold_dict[ct] for ct in cell_types_to_use])

        elif activate_threshold == 'celltype_top100':
            # Use dictionary for efficient collection of thresholds
            threshold_dict = {}
            for celltype in unique_cell_types:
                # get the cre_expression for the cell type
                celltype_cre_expr_flattened = cre_celltypes_expression[cell_types_to_use == celltype].values.flatten()
                # set the threshold as the top 100 cre value, if less than 2, set to 2
                celltype_cre_expr_flattened = np.sort(celltype_cre_expr_flattened)[::-1]
                if len(celltype_cre_expr_flattened) < 100:
                    thres = 1
                else:
                    thres = celltype_cre_expr_flattened[100]
                threshold_dict[celltype] = np.maximum(thres, 1)
            # Map to original cell types order
            activate_threshold_array = np.array([threshold_dict[ct] for ct in cell_types_to_use])

        elif activate_threshold == 'celltype_poisson_point_estimate':
            # Use dictionary for efficient collection of thresholds
            threshold_dict = {}
            for celltype in unique_cell_types:
                # get the cre_expression for the cell type
                celltype_cre_expr_flattened = cre_celltypes_expression[cell_types_to_use == celltype].values.flatten()
                # calculate poisson lambda
                one_count_proportion = (celltype_cre_expr_flattened == 1).sum() / len(celltype_cre_expr_flattened)
                two_count_proportion = (celltype_cre_expr_flattened == 2).sum() / len(celltype_cre_expr_flattened)
                poisson_lambda = two_count_proportion / one_count_proportion * 2
                # set the threshold
                threshold_dict[celltype] = stats.poisson.ppf(0.999, mu=poisson_lambda)
            # Map to original cell types order
            activate_threshold_array = np.array([threshold_dict[ct] for ct in cell_types_to_use])

        elif activate_threshold == 'celltype_poisson_fit':
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

            # Use dictionary for efficient collection of thresholds
            threshold_dict = {}
            for celltype in unique_cell_types:
                # get the cre_expression for the cell type
                celltype_cre_expr_flattened = cre_celltypes_expression[cell_types_to_use == celltype].values.flatten()
                # calculate poisson lambda
                _, _, lambda2 = fit(celltype_cre_expr_flattened)
                # set the threshold
                threshold_dict[celltype] = lambda2
            # Map to original cell types order
            activate_threshold_array = np.array([threshold_dict[ct] for ct in cell_types_to_use])
        elif isinstance(activate_threshold, (int, float)):
            # is numeric, just use the threshold
            activate_threshold_array = np.full(len(cell_types_to_use), activate_threshold)
        return activate_threshold_array
    
    def fisher_exact_cre_test(self, cell_types_to_use: List=None, activate_threshold=2, infect_threshold=1) -> dict:
        """Fisher's exact test for CRE activation enrichment (per-CRE analysis)."""
        config = {
            'cell_types_to_use': cell_types_to_use,
            'infect_threshold': infect_threshold,
            'activate_threshold': activate_threshold
        }

        # Check for cached results
        cached_result = _check_cached_result(self, 'fisher_exact_cre_test_results', 'fisher_exact_cre_test_configs', config)
        if cached_result is not None:
            return cached_result

        # Load data
        if cell_types_to_use is not None:
            cre_celltypes_expression, cell_types_to_use = self.get_cre_celltypes(cell_types_to_use)
        else:
            cre_celltypes_expression = self.get_cre_expression().copy()
            cell_types_to_use = self.get_celltypes()

        # Pre-compute unique cell types to avoid redundant calls
        unique_cell_types = cell_types_to_use.unique()

        # Initialize result DataFrames
        p_value, q_value, precision, recall, foldchange, precision_n, recall_n = (
            pd.DataFrame(index=unique_cell_types, columns=cre_celltypes_expression.columns) for _ in range(7)
        )

        # Get activation threshold array (may be cell-type specific)
        activate_threshold_array = self.estimate_activate_threshold_array(activate_threshold, cre_celltypes_expression, cell_types_to_use)

        # Run Fisher's exact test for each CRE
        for cre in cre_celltypes_expression.columns:
            cre_infected = cre_celltypes_expression[cre] >= infect_threshold
            cre_infected_expression = cre_celltypes_expression[cre][cre_infected]
            cell_types_infected = cell_types_to_use[cre_infected]
            cre_activated = cre_infected_expression >= activate_threshold_array[cre_infected]

            # Count infected and activated cells per cell type (use pre-computed unique_cell_types)
            n_celltype_infected = cell_types_infected.value_counts().reindex(unique_cell_types, fill_value=0)
            n_celltype_activated = cell_types_infected[cre_activated].value_counts().reindex(unique_cell_types, fill_value=0)

            # Create contingency table components
            TT = n_celltype_activated
            FT = n_celltype_activated.sum() - n_celltype_activated
            TF = n_celltype_infected - n_celltype_activated
            FF = n_celltype_infected.sum() - n_celltype_activated.sum() - TF

            # Run Fisher's exact test for each cell type (use pre-computed unique_cell_types)
            for cell_type in unique_cell_types:
                oddsratio, p = stats.fisher_exact([[int(FF.loc[cell_type]), int(FT.loc[cell_type])],
                                                   [int(TF.loc[cell_type]), int(TT.loc[cell_type])]])
                p_value.loc[cell_type, cre] = p
                foldchange.loc[cell_type, cre] = oddsratio

            # Calculate precision, recall
            precision[cre] = TT / (TT + TF)
            recall[cre] = TT / (TT + FT)
            precision_n[cre] = TT + TF
            recall_n[cre] = TT + FT
            q_value[cre] = multitest.multipletests(p_value[cre], method='fdr_bh')[1]

        # Fill NaN values
        precision = precision.fillna(0)
        recall = recall.fillna(0)

        # Assign metrics to cre_info based on best_subclass
        cre_info = self.get_creinfo().copy()
        result_dfs = {
            'p_value': p_value, 'q_value': q_value, 'precision': precision,
            'recall': recall, 'foldchange': foldchange
        }
        cre_info = _assign_cre_info_from_best_subclass(cre_info, result_dfs,
                                                       ['p_value', 'q_value', 'precision', 'recall', 'foldchange'])

        # Calculate entropy
        for cre in cre_info.index:
            cre_info.loc[cre, 'entropy'] = stats.entropy(recall[cre].astype(float))

        # Store results
        fisher_exact_cre_test_result = {
            'cre_info': cre_info, 'p_value': p_value, 'q_value': q_value,
            'precision': precision, 'recall': recall, 'precision_n': precision_n,
            'recall_n': recall_n, 'foldchange': foldchange
        }
        _store_result(self, 'fisher_exact_cre_test_results', 'fisher_exact_cre_test_configs',
                     fisher_exact_cre_test_result, config)

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
                         filter_zero_counts=False, log_transform=False, binarize_t7=False, bulk_log_transform=False, rank_transform=None,
                         bootstrap_number=None, bootstrap_to_fixed_sample_size=None, apply_bootstrap_in_observation=False,
                         calculate_fdc=False, fill_nan=True, n_jobs=256, load_stored=True, dry_run=False) -> dict:
        """
        Compute fold-change enrichment of CRE activity across cell types with extensive normalization options.

        Parameters
        ----------
        cell_types_to_use : list, optional
            Cell types to include in analysis
        normalize_by_cell_rna : bool, optional
            Normalize by RNA content per cell (default: False)
        normalize_by_cell_volume : bool, optional
            Normalize by cell volume (default: False)
        normalize_by_cell_t7 : bool or float, optional
            Normalize by T7 expression per cell (default: False)
        filter_by_cell_t7 : float, optional
            Filter cells with T7 expression below threshold
        normalize_by_celltype_rna : bool, optional
            Normalize by median RNA per cell type (default: False)
        normalize_by_celltype_volume : bool, optional
            Normalize by median volume per cell type (default: False)
        normalize_by_negative_control : bool, optional
            Normalize by negative control CRE expression (default: False)
        normalize_by_infected_cell : bool, optional
            Normalize by infected cell fraction (default: False)
        normalize_by_celltype_t7 : bool, optional
            Normalize by median T7 per cell type (default: False)
        normalize_by_total_cre : bool, optional
            Normalize by total CRE expression (default: False)
        normalize_by_libsize : bool, optional
            Normalize by library size (default: False)
        filter_zero_counts : bool, optional
            Filter out zero counts (default: False)
        log_transform : bool, optional
            Apply log transformation at cell level (default: False)
        binarize_t7 : bool, optional
            Convert T7 to binary infected/uninfected (default: False)
        bulk_log_transform : bool, optional
            Apply log after aggregation (default: False)
        rank_transform : str, optional
            Apply rank transformation ('celltype' or 'cre')
        bootstrap_number : int, optional
            Number of bootstrap iterations
        bootstrap_to_fixed_sample_size : int, optional
            Resample to fixed sample size
        apply_bootstrap_in_observation : bool, optional
            Bootstrap at observation vs aggregated level (default: False)
        calculate_fdc : bool, optional
            Calculate fold discovery curve (default: False)
        fill_nan : bool, optional
            Fill NaN values with 0 (default: True)
        n_jobs : int, optional
            Number of parallel jobs (default: 256)
        load_stored : bool, optional
            Load cached results if available (default: True)
        dry_run : bool, optional
            Return config without running (default: False)

        Returns
        -------
        dict
            Dictionary containing:
            - 'activity': DataFrame with cell type x CRE fold-change values
            - 'config': Configuration parameters
            - 'bootstrap_std': Standard deviations from bootstrap (if requested)
            - Additional statistics

        Notes
        -----
        Computes fold-change of each CRE in each cell type relative to other cell types,
        with extensive normalization and bootstrap options for robustness. Results are cached.
        """
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
            'binarize_t7': binarize_t7,
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
            'filter_zero_counts': False, 'log_transform': False, 'binarize_t7': False, 'rank_transform': None,
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

        # Preprocess expressions using common helper
        cre_cells_expression, rna_cells_expression, cell_types_to_use, volm, t7_cells_expression = \
            self._preprocess_expressions(
                cell_types_to_use=cell_types_to_use,
                normalize_by_cell_rna=normalize_by_cell_rna,
                normalize_by_cell_volume=normalize_by_cell_volume,
                normalize_by_cell_t7=normalize_by_cell_t7,
                filter_by_cell_t7=filter_by_cell_t7,
                binarize_t7=binarize_t7,
                log_transform=log_transform
            )

        # Set t7_cells_expression to None if not needed for celltype-level normalization
        if not normalize_by_cell_t7 and not normalize_by_celltype_t7:
            t7_cells_expression = None

        cre_info = self.get_creinfo().copy()
        calculate_fold_change_args = {
            'cre_cells_expression': cre_cells_expression,
            'cell_types_to_use': cell_types_to_use,
            'cell_types_order': np.unique(cell_types_to_use),
            'CRE_info': cre_info[~cre_info.index.isin(self.blacklist_cre)], 'rna_cells_expression': rna_cells_expression, 'volm': volm,
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
            bootstrap_results = Parallel(n_jobs=n_jobs, backend='loky', batch_size=1, verbose=10)(
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
        """
        Compute average CRE activity per cell type with bootstrap confidence intervals.

        Parameters
        ----------
        cell_types_to_use : list
            Cell types to include in analysis
        normalize_by_cell_rna : bool, optional
            Normalize by RNA content per cell (default: False)
        normalize_by_cell_volume : bool, optional
            Normalize by cell volume (default: False)
        normalize_by_cell_t7 : bool or float, optional
            Normalize by T7 expression per cell (default: False)
        normalize_by_celltype_rna : bool, optional
            Normalize by median RNA per cell type (default: False)
        normalize_by_celltype_volume : bool, optional
            Normalize by median volume per cell type (default: False)
        normalize_by_negative_control : bool, optional
            Normalize by negative control CRE expression (default: False)
        normalize_by_celltype_t7 : bool, optional
            Normalize by median T7 per cell type (default: False)
        filter_by_cell_t7 : float, optional
            Filter cells with T7 expression below threshold
        normalize_by_libsize : bool, optional
            Normalize by library size (default: False)
        log_transform : bool, optional
            Apply log transformation (default: False)
        bootstrap_number : int, optional
            Number of bootstrap iterations
        bootstrap_to_fixed_sample_size : int, optional
            Resample to fixed sample size
        bootstrap_to_fixed_pct : float, optional
            Resample to fixed percentage of cells
        fill_nan : bool, optional
            Fill NaN values with 0 (default: True)
        n_jobs : int, optional
            Number of parallel jobs (default: 256)
        load_stored : bool, optional
            Load cached results if available (default: True)
        dry_run : bool, optional
            Return config without running (default: False)

        Returns
        -------
        dict
            Dictionary containing:
            - 'activity': DataFrame with cell type x CRE average expression
            - 'bootstrap_std': Bootstrap standard deviations
            - 'config': Configuration parameters

        Notes
        -----
        Similar to fold_change_test but computes average expression within each cell type
        rather than fold-changes. Bootstrap resampling is performed within each cell type
        to estimate uncertainty. Results are cached.
        """
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

        # Preprocess expressions using common helper
        cre_cells_expression, rna_cells_expression, cell_types_to_use, volm, t7_cells_expression = \
            self._preprocess_expressions(
                cell_types_to_use=cell_types_to_use,
                normalize_by_cell_rna=normalize_by_cell_rna,
                normalize_by_cell_volume=normalize_by_cell_volume,
                normalize_by_cell_t7=normalize_by_cell_t7,
                filter_by_cell_t7=filter_by_cell_t7,
                binarize_t7=False,
                log_transform=log_transform,
                log_func='log'  # average_bootstrap_test uses np.log instead of np.log1p
            )

        # Set t7_cells_expression to None if not needed for celltype-level normalization
        if not normalize_by_cell_t7 and not normalize_by_celltype_t7:
            t7_cells_expression = None

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
            bootstrap_results = Parallel(n_jobs=n_jobs, backend='loky', batch_size=1, verbose=10)(
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
                                 calibrate='CRE/T7', norm='T7', tail='right', to_filter=None):
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
        if calibrate is not None:
            # calibrate the fdc by total T7 or libsize
            if calibrate == 'T7-CRE':
                t7_all = np.log(self.get_t7_expression().sum().values).reshape(1, 1, -1)
                res_array = res_array + t7_all
            elif calibrate == 'T7-all':
                t7_all1 = np.log(self.get_t7_expression().sum().values).reshape(1, 1, -1)
                t7_all2 = np.log(self.get_t7_expression().sum(axis=1).groupby(self.get_celltypes()).sum().values).reshape(1, -1, 1)
                res_array = res_array + t7_all1 + t7_all2
            elif calibrate == 'CRE/T7':
                cre_t7_all = np.log(self.get_cre_expression().sum().values).reshape(1, 1, -1) - np.log(self.get_t7_expression().sum().values).reshape(1, 1, -1)
                res_array = res_array - cre_t7_all
            elif calibrate == 'libsize':
                lib_all = np.log(self.lib_size['counts'].values).reshape(1, 1, -1)
                res_array = res_array + lib_all
            elif calibrate == 'self-CRE':
                # self calibrate based on average of activity across all cell types, all bootstraps
                cre_mean = np.nanmean(res_array, axis=(0, 1)).reshape(1, 1, -1)
                res_array = res_array - cre_mean
            elif calibrate == 'self-CRE-avg':
                # self calibrate based on average of activity across all cell types, all bootstraps
                cre_mean = np.nanmean(np.nanmean(res_array, axis=0), axis=0).reshape(1, 1, -1)
                res_array = res_array - cre_mean
            elif calibrate == 'self-CellType':
                # self calibrate based on average of activity across all cell types, all bootstraps
                cre_mean = np.nanmean(res_array, axis=(0, 2)).reshape(1, -1, 1)
                res_array = res_array - cre_mean
            elif calibrate == 'self-CellType-avg':
                # self calibrate based on average of activity across all cell types, all bootstraps
                cre_mean = np.nanmean(np.nanmean(res_array, axis=0), axis=1).reshape(1, -1, 1)
                res_array = res_array - cre_mean
            elif calibrate == 'self-all':
                # self calibrate based on average of activity across all cell types, all bootstraps
                cre_mean1 = np.nanmean(res_array, axis=(0, 2)).reshape(1, -1, 1)
                cre_mean2 = np.nanmean(res_array, axis=(0, 1)).reshape(1, 1, -1)
                res_array = res_array - cre_mean1 - cre_mean2
            elif calibrate == 'self-all-avg':
                # self calibrate based on average of activity across all cell types, all bootstraps
                cre_mean1 = np.nanmean(np.nanmean(res_array, axis=0), axis=1).reshape(1, -1, 1)
                cre_mean2 = np.nanmean(np.nanmean(res_array, axis=0), axis=0).reshape(1, 1, -1)
                res_array = res_array - cre_mean1 - cre_mean2
        neg_control_array = res_array[:, :, res['celltype_activity'].columns.isin(self.get_negative_control_cres())]
        neg_control_array = np.nanmean(neg_control_array, axis=2)
        # turn to DataFrame
        res_df = pd.DataFrame(np.nanmean(res_array, axis=0), index=res['celltype_activity'].index, columns=res['celltype_activity'].columns)
        res_df_fdc = res_df.copy()
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
            elif threshold == 'neg_control_mean':
                # use the distribution of negative control CREs to set the threshold
                fdc = np.nanmean(neg_control_array[:, res['celltype_activity'].index == cell_type])
                fdc_l = fdc_u = fdc
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
            # store the fdc calibrated res_df
            res_df_fdc.loc[cell_type] = res_df.loc[cell_type] - fdc
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
            return res2_q1, res_df, res_df_fdc
        elif tail == 'left':
            return res2_q2, res_df, res_df_fdc
        elif tail == 'both':
            return res2_q, res_df, res_df_fdc
        elif tail == 'all':
            return res2_q, res2_q1, res2_q2, res_df, res_df_fdc

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

    def bayesian_activity_test(self, level: Literal['class', 'subclass'] = 'class',
                               channel: Literal['t7', 'joint'] = 'joint',
                               method: Literal['svi', 'nuts'] = 'svi',
                               infection_model: Literal['copy_number', 'binary'] = 'copy_number',
                               subclass_tag: str = 'obs:subclass', class_tag: str = 'obs:class',
                               kmax: int = None, init: Literal['moments'] = 'moments',
                               num_steps: int = 20000, lr: float = 5e-3, guide: str = 'AutoNormal',
                               num_warmup: int = 1000, num_samples: int = 1000, num_chains: int = 2,
                               num_posterior: int = 1000, seed: int = 0, load_stored: bool = True) -> dict:
        """Fit a Bayesian hierarchical infection model (see ``bayesian_hierarchical``).

        Stages, per the design: ``(level='class', channel='t7')`` calibrates infection from
        T7 alone; ``(level='class', channel='joint')`` adds CRE activity at class granularity;
        ``(level='subclass', channel='joint')`` is the full subclass-nested-in-class model.

        Parameters
        ----------
        level : 'class' | 'subclass'
            Cell-type granularity of ``rho`` / ``gamma``. 'subclass' nests within 'class'.
        channel : 't7' | 'joint'
            't7' = T7-only infection calibration; 'joint' = T7 + CRE.
        method : 'svi' | 'nuts'
            SVI (scales to full data) or NUTS (use at class level / calibration only).
        infection_model : 'copy_number' | 'binary'
            'copy_number' marginalizes a latent Poisson virus-copy count. 'binary'
            marginalizes a shared infected/not-infected gate followed by NB channels.
        kmax : int, optional
            Latent copy-number truncation. If None, chosen adaptively from the data and
            validated against the posterior Poisson tail. Ignored by the binary model.
        ... : inference hyperparameters passed through to ``fit_svi`` / ``fit_nuts``.

        Returns
        -------
        dict with keys ``summary`` (rho/gamma/delta DataFrames with evidence + CI widths),
        ``evidence`` (pre-fit audit), ``ppc`` (posterior-predictive checks), ``diagnostics``,
        ``scalar_samples`` (global parameter draws), ``kmax``, ``group_names``, ``cre_names``,
        and ``config``.
        """
        import baystarrfish as bh

        if infection_model not in bh.MODEL_FAMILIES:
            raise ValueError(f"unsupported infection_model={infection_model}; "
                             f"available: {sorted(bh.MODEL_FAMILIES)}")
        if (level, channel) not in bh.MODEL_FAMILIES[infection_model]:
            raise ValueError(f"unsupported (level, channel)=({level}, {channel}); "
                             f"available: {sorted(bh.MODEL_FAMILIES[infection_model].keys())}")

        config = dict(level=level, channel=channel, method=method, infection_model=infection_model,
                      subclass_tag=subclass_tag,
                      class_tag=class_tag, kmax=kmax, init=init, num_steps=num_steps, lr=lr,
                      guide=guide, num_warmup=num_warmup, num_samples=num_samples,
                      num_chains=num_chains, seed=seed,
                      blacklist_cre=list(self.blacklist_cre))
        if load_stored:
            cached = _check_cached_result(self, 'bayesian_activity_test_results',
                                          'bayesian_activity_test_configs', config)
            if cached is not None:
                return cached

        # --- assemble arrays and delegate to the array-level core ---
        if not hasattr(self, 'lib_size') or self.lib_size is None:
            raise ValueError("self.lib_size is required (call load_libsize first)")
        cre_names = [cre for cre in list(self.lib_size.index) if cre not in set(self.blacklist_cre)]
        cre_info = self.get_creinfo().reindex(cre_names)
        negative_control_mask = cre_info['labeling_type'].astype(str).eq('negative control').to_numpy()
        config['negative_control_cre'] = np.asarray(cre_names)[negative_control_mask].tolist()
        t7_df = self.get_t7_expression()
        cre_df = self.get_cre_expression()
        if t7_df is None:
            raise ValueError("T7 expression unavailable (t7_tag not set)")
        t7 = t7_df.reindex(columns=cre_names).to_numpy()
        cre = cre_df.reindex(columns=cre_names).to_numpy()
        lib_size_log = self.lib_size['counts'].reindex(cre_names).to_numpy().astype(np.float64)
        sub = self.get_celltypes(subclass_tag).astype(str).to_numpy()
        cls = self.get_celltypes(class_tag).astype(str).to_numpy()

        res = bh.run_model(t7, cre, sub, cls, lib_size_log, cre_names,
                           level=level, channel=channel, method=method, kmax=kmax,
                           num_steps=num_steps, lr=lr, guide=guide, num_warmup=num_warmup,
                           num_samples=num_samples, num_chains=num_chains,
                           num_posterior=num_posterior, seed=seed,
                           negative_control_mask=negative_control_mask,
                           infection_model=infection_model)
        res['config'] = config
        res['config']['blacklist_cre'] = list(self.blacklist_cre)
        _store_result(self, 'bayesian_activity_test_results', 'bayesian_activity_test_configs',
                      res, config)
        return res

    def glm_test(self, variate='T7', cell_types_to_use: List=None, norm_by_volm=False, volm_covariate=False, fov_covariate=False, rna_covariate=False, size_covariate=False,
                 filter_infected_cells=False, positive_x_or_y=False, only_keep_positive_x=False, only_keep_positive_y=False, transform_x_y=None, fix_intercept=None, multiprocess_threads=256) -> dict:
        # Initialize cache attributes if needed
        if not hasattr(self, 'glm_test_results'):
            self.glm_test_results = []
            self.glm_test_configs = []

        # Create config for caching
        config = {
            'variate': variate,
            'norm_by_volm': norm_by_volm,
            'volm_covariate': volm_covariate,
            'fov_covariate': fov_covariate,
            'rna_covariate': rna_covariate,
            'size_covariate': size_covariate,
            'filter_infected_cells': filter_infected_cells,
            'positive_x_or_y': positive_x_or_y,
            'only_keep_positive_x': only_keep_positive_x,
            'only_keep_positive_y': only_keep_positive_y,
            'transform_x_y': transform_x_y,
            'fix_intercept': fix_intercept
        }

        # Check cache for existing results
        for stored_config, glm_result in zip(self.glm_test_configs, self.glm_test_results):
            if stored_config == config:
                print('Results already exist, return stored results')
                return glm_result.copy()

        # Compute new result
        result = glm(self.adata, variate=variate, cell_types_to_use=cell_types_to_use, norm_by_volm=norm_by_volm,
                     volm_covariate=volm_covariate, fov_covariate=fov_covariate, rna_covariate=rna_covariate, size_covariate=size_covariate,
                     filter_infected_cells=filter_infected_cells,
                     positive_x_or_y=positive_x_or_y,
                     only_keep_positive_x=only_keep_positive_x,
                     only_keep_positive_y=only_keep_positive_y,
                     transform_x_y=transform_x_y,
                     fix_intercept=fix_intercept,
                     multiprocess_threads=multiprocess_threads)

        # Cache result
        self.glm_test_results.append(result)
        self.glm_test_configs.append(config)
        return result

    def pseudo_bulk_glm_test(self, variate='T7', cell_types_to_use: List=None, norm_by_volm=False, volm_covariate=False, rna_covariate=False, size_covariate=False,
                             filter_infected_cells=False, positive_x_or_y=False, only_keep_positive_x=False, only_keep_positive_y=False, transform_x_y=None, fix_intercept=None,
                             pseudo_bulk_size=[50], pseudo_bulk_percentage=None, pseudo_bulk_number=[1000], replace=True,
                             multiprocess_threads=256) -> dict:
        # Initialize cache attributes if needed
        if not hasattr(self, 'pseudo_bulk_glm_test_results'):
            self.pseudo_bulk_glm_test_results = []
            self.pseudo_bulk_glm_test_configs = []

        # Create config for caching
        config = {'cell_types_to_use': cell_types_to_use,
                  'variate': variate,
                  'norm_by_volm': norm_by_volm,
                  'volm_covariate': volm_covariate,
                  'rna_covariate': rna_covariate,
                  'size_covariate': size_covariate,
                  'filter_infected_cells': filter_infected_cells,
                  'positive_x_or_y': positive_x_or_y,
                  'only_keep_positive_x': only_keep_positive_x,
                  'only_keep_positive_y': only_keep_positive_y,
                  'transform_x_y': transform_x_y,
                  'fix_intercept': fix_intercept,
                  'pseudo_bulk_size': pseudo_bulk_size,
                  'pseudo_bulk_percentage': pseudo_bulk_percentage,
                  'pseudo_bulk_number': pseudo_bulk_number,
                  'replace': replace}

        pseudo_bulk_keys = ['cell_types_to_use', 'filter_infected_cells', 'replace',
                            'pseudo_bulk_size', 'pseudo_bulk_percentage', 'pseudo_bulk_number']
        partial_load = False

        # Check cache for existing results
        for stored_config, pseudo_bulk_glm_test_result in zip(self.pseudo_bulk_glm_test_configs, self.pseudo_bulk_glm_test_results):
            # Check for exact match
            if stored_config == config:
                print('Results already exist, return stored results')
                return pseudo_bulk_glm_test_result.copy()
            # Check for partial match (can reuse pseudo bulk adata)
            if all(stored_config.get(key) == config.get(key) for key in pseudo_bulk_keys):
                print('Partially load pseudo bulk adata')
                pseudo_bulk_adata = pseudo_bulk_glm_test_result['pseudo_bulk_adata'].copy()
                partial_load = True

        if not partial_load:
            # for each cell type to use, create a pseudo bulk
            celltypes = self.get_celltypes()
            cre_expression = self.get_cre_expression()
            t7_expression = self.get_t7_expression()
            volumes = self.get_tag('obs:volm')
            if filter_infected_cells:
                # get the infected cells
                infected_cells = ((cre_expression >= 0).sum(axis=1) > 0)
                celltypes = celltypes[infected_cells]
                cre_expression = cre_expression[infected_cells]
                t7_expression = t7_expression[infected_cells]
                volumes = volumes[infected_cells]
            if cell_types_to_use is None:
                cell_types_to_use = celltypes.unique()
            else:
                celltypes = celltypes[celltypes.isin(cell_types_to_use)]
                cre_expression = cre_expression[celltypes.index]
                t7_expression = t7_expression[celltypes.index]
                volumes = volumes[celltypes.index]
            # get the cell type cell counts
            cell_counts = celltypes.value_counts().loc[cell_types_to_use]
            celltypes = pd.DataFrame(celltypes)
            # filter out cell types with insufficient cell counts if not replace
            if not replace:
                if pseudo_bulk_size is not None:
                    cell_counts = cell_counts[cell_counts >= pseudo_bulk_size]
                else:
                    cell_counts = cell_counts[cell_counts >= 50 / pseudo_bulk_percentage]
            # redefine the cell types to use
            cell_types_to_use = cell_counts.index.tolist()
            # generate pseudo bulk for each cell type
            def sample_aggregate(df3, df_list, n_samples=50, percentage=None, random_state=42, replace=replace):
                """
                Ultra memory-efficient parallel sampling and aggregation using swifter
                """
                def sample_and_sum_group(group):
                    # Sample indexes for this group
                    if percentage is not None:
                        # Sample by percentage
                        sampled_idx = group.sample(frac=percentage, replace=replace, random_state=random_state).index
                    else:
                        # Sample by exact number
                        if replace:
                            sampled_idx = group.sample(n=n_samples, replace=replace, random_state=random_state).index
                        else:
                            sampled_idx = group.sample(n=min(n_samples, len(group)), replace=replace, random_state=random_state).index
                    # Return sums for all target DataFrames
                    return [df.loc[sampled_idx].sum() for df in df_list]
                # Apply in parallel using swifter
                grouped_results = df3.groupby('subclass').apply(sample_and_sum_group)
                # Reorganize results into separate DataFrames
                results = []
                for i in range(len(df_list)):
                    df_result = pd.DataFrame(
                        [result[i] for result in grouped_results.values], 
                        index=grouped_results.index
                    )
                    results.append(df_result)
                return results
            # Prepare lists to collect dataframes (avoid repeated concatenations)
            pseudo_bulk_list = []
            pseudo_bulk_t7_list = []
            pseudo_bulk_obs_list = []

            if pseudo_bulk_size is None:
                assert pseudo_bulk_percentage is not None, 'pseudo_bulk_size or pseudo_bulk_percentage must be set'
                pseudo_bulk_size = [None] * len(pseudo_bulk_percentage)
            if pseudo_bulk_percentage is None:
                assert pseudo_bulk_size is not None, 'pseudo_bulk_size or pseudo_bulk_percentage must be set'
                pseudo_bulk_percentage = [None] * len(pseudo_bulk_size)

            for i, (size, percentage, number) in enumerate(zip(pseudo_bulk_size, pseudo_bulk_percentage, pseudo_bulk_number)):
                for s in range(number):
                    cre_expression_pseudo_bulk, t7_expression_pseudo_bulk, volumes_pseudo_bulk = sample_aggregate(
                        celltypes, [cre_expression, t7_expression, volumes],
                        n_samples=size, percentage=percentage, random_state=s
                    )
                    # change the index to reflect the celltype, i and s
                    cell_types = cre_expression_pseudo_bulk.index.copy()
                    cre_expression_pseudo_bulk.index = cell_types.astype(str) + f'_pseudo_bulk_{i}_{s}'
                    t7_expression_pseudo_bulk.index = cell_types.astype(str) + f'_pseudo_bulk_{i}_{s}'
                    volumes_pseudo_bulk.index = cell_types.astype(str) + f'_pseudo_bulk_{i}_{s}'

                    # Collect dataframes
                    pseudo_bulk_list.append(cre_expression_pseudo_bulk)
                    pseudo_bulk_t7_list.append(t7_expression_pseudo_bulk)

                    sample_obs = pd.DataFrame(volumes_pseudo_bulk, columns=['volm'])
                    sample_obs['subclass'] = cell_types.astype(str)
                    sample_obs['fov'] = cell_types.astype(str)
                    sample_obs['size'] = size
                    sample_obs['percentage'] = percentage
                    sample_obs['seed'] = s
                    pseudo_bulk_obs_list.append(sample_obs)

            # Concatenate all at once (much more efficient)
            pseudo_bulk = pd.concat(pseudo_bulk_list, axis=0)
            pseudo_bulk_t7 = pd.concat(pseudo_bulk_t7_list, axis=0)
            pseudo_bulk_obs = pd.concat(pseudo_bulk_obs_list, axis=0)
            # create a new AnnData object for the pseudo bulk
            pseudo_bulk_adata = sc.AnnData(pseudo_bulk, obs=pseudo_bulk_obs)
            pseudo_bulk_adata.obsm['X_raw'] = pseudo_bulk
            pseudo_bulk_adata.obsm['CRE'] = pseudo_bulk
            if t7_expression is not None:
                pseudo_bulk_adata.obsm['T7CRE'] = pseudo_bulk_t7
        # Perform glm test on the pseudo bulk
        result = glm(pseudo_bulk_adata, variate=variate, cell_types_to_use=cell_types_to_use, CREs=pseudo_bulk_adata.var.index,
                     norm_by_volm=norm_by_volm, volm_covariate=volm_covariate, rna_covariate=rna_covariate, size_covariate=size_covariate,
                     fov_covariate=False, filter_infected_cells=False,
                     positive_x_or_y=positive_x_or_y,
                     only_keep_positive_x=only_keep_positive_x,
                     only_keep_positive_y=only_keep_positive_y,
                     transform_x_y=transform_x_y,
                     fix_intercept=fix_intercept,
                     multiprocess_threads=multiprocess_threads)

        pseudo_bulk_glm_test_result = {'pseudo_bulk_adata': pseudo_bulk_adata,
                                       'result': result}

        # Cache result
        self.pseudo_bulk_glm_test_results.append(pseudo_bulk_glm_test_result)
        self.pseudo_bulk_glm_test_configs.append(config)
        return pseudo_bulk_glm_test_result
    
    def pseudo_bulk_t7_sum_test(self, cell_types_to_use: List=None, t7_pseudo_bulk_size=100,
                                pseudo_bulk_number=1000, infected_cells_threshold=None, replace=True, multiprocess_threads=256) -> dict:
        # Initialize cache attributes if needed
        if not hasattr(self, 'pseudo_bulk_t7_sum_test_results'):
            self.pseudo_bulk_t7_sum_test_results = []
            self.pseudo_bulk_t7_sum_test_configs = []

        # Create config for caching
        config = {'cell_types_to_use': cell_types_to_use,
                  't7_pseudo_bulk_size': t7_pseudo_bulk_size,
                  'pseudo_bulk_number': pseudo_bulk_number,
                  'infected_cells_threshold': infected_cells_threshold,
                  'replace': replace}

        # Check cache for existing results
        for stored_config, pseudo_bulk_t7_sum_test_result in zip(self.pseudo_bulk_t7_sum_test_configs, self.pseudo_bulk_t7_sum_test_results):
            if stored_config == config:
                print('Results already exist, return stored results')
                return pseudo_bulk_t7_sum_test_result.copy()
        # for each cell type to use, create a pseudo bulk
        celltypes = self.get_celltypes()
        cre_expression = self.get_cre_expression()
        t7_expression = self.get_t7_expression()
        volumes = self.get_tag('obs:volm')
        if cell_types_to_use is None:
            cell_types_to_use = celltypes.unique()
        else:
            celltypes = celltypes[celltypes.isin(cell_types_to_use)]
            cre_expression = cre_expression[celltypes.index]
            t7_expression = t7_expression[celltypes.index]
            volumes = volumes[celltypes.index]
        # get the cell type cell counts
        cell_counts = celltypes.value_counts().loc[cell_types_to_use]
        # redefine the cell types to use
        cell_types_to_use = cell_counts.index.tolist()
        # generate pseudo bulk for each cell type
        def sample_to_fixed_t7(t7_array, cre_array):
            # Estimate number of samples needed based on mean
            mean_t7 = np.mean(t7_array)
            if mean_t7 > 0:
                estimated_samples = max(10, int(t7_pseudo_bulk_size / mean_t7 * 1.2))  # 20% buffer
            else:
                estimated_samples = len(t7_array)
            # Sample indices with replacement
            indices = np.random.choice(len(t7_array), size=estimated_samples, replace=True)
            sampled_t7 = t7_array[indices]
            sampled_cre = cre_array[indices]
            # Check if we have enough samples, if not continue sampling
            while np.sum(sampled_t7) < t7_pseudo_bulk_size:
                # Sample additional batch
                additional_indices = np.random.choice(len(t7_array), size=estimated_samples, replace=True)
                sampled_t7 = np.concatenate([sampled_t7, t7_array[additional_indices]])
                sampled_cre = np.concatenate([sampled_cre, cre_array[additional_indices]])
            # Find cutoff point where cumsum reaches threshold
            cumsum = np.cumsum(sampled_t7)
            cutoff_idx = np.searchsorted(cumsum, t7_pseudo_bulk_size, side='right') + 1
            return np.sum(sampled_t7[:cutoff_idx]), np.sum(sampled_cre[:cutoff_idx]), cutoff_idx
        def sample_single_combination(args):
            pseudo_bulk_idx, cell_type_idx, cre_idx = args
            # Get data for this cell type and CRE
            if isinstance(celltypes, pd.DataFrame):
                cell_mask = (celltypes.iloc[:, 0] == cell_types_to_use[cell_type_idx]).values
            else:
                cell_mask = (celltypes == cell_types_to_use[cell_type_idx]).values
            t7_data = t7_expression.iloc[cell_mask, cre_idx].values
            cre_data = cre_expression.iloc[cell_mask, cre_idx].values
            # Apply sampling function
            sampled_t7, sampled_cre, n_samples = sample_to_fixed_t7(t7_data, cre_data)
            return pseudo_bulk_idx, cell_type_idx, cre_idx, sampled_t7, sampled_cre, n_samples
        # Prepare all combinations
        n_cres = cre_expression.shape[1]
        n_cell_types = len(cell_types_to_use)

        # Pre-filter cell type-CRE combinations if threshold is set
        if infected_cells_threshold is not None:
            valid_combinations = set()
            for cell_type_idx in range(n_cell_types):
                if isinstance(celltypes, pd.DataFrame):
                    cell_mask = (celltypes.iloc[:, 0] == cell_types_to_use[cell_type_idx]).values
                else:
                    cell_mask = (celltypes == cell_types_to_use[cell_type_idx]).values
                for cre_idx in range(n_cres):
                    cre_data = cre_expression.iloc[cell_mask, cre_idx].values
                    if (cre_data > 0).sum() >= infected_cells_threshold:
                        valid_combinations.add((cell_type_idx, cre_idx))

            # Build combinations using pre-filtered set
            combinations = [
                (pseudo_bulk_idx, cell_type_idx, cre_idx)
                for pseudo_bulk_idx in range(pseudo_bulk_number)
                for cell_type_idx, cre_idx in valid_combinations
            ]
        else:
            # Build all combinations directly
            combinations = [
                (pseudo_bulk_idx, cell_type_idx, cre_idx)
                for pseudo_bulk_idx in range(pseudo_bulk_number)
                for cell_type_idx in range(n_cell_types)
                for cre_idx in range(n_cres)
            ]
        # Run in parallel
        results = Parallel(n_jobs=min(multiprocess_threads, int(multiprocessing.cpu_count() * 0.8)), verbose=10)(
            delayed(sample_single_combination)(combo) for combo in combinations
        )
        # Initialize arrays with NaN (more efficient than zeros + fill)
        array_shape = (pseudo_bulk_number, n_cell_types, n_cres)
        pseudo_bulk_t7_array = np.full(array_shape, np.nan)
        pseudo_bulk_cre_array = np.full(array_shape, np.nan)
        pseudo_bulk_n_array = np.full(array_shape, np.nan)

        # Fill arrays with results
        for pseudo_bulk_idx, cell_type_idx, cre_idx, t7_sum, cre_sum, n_samples in results:
            pseudo_bulk_t7_array[pseudo_bulk_idx, cell_type_idx, cre_idx] = t7_sum
            pseudo_bulk_cre_array[pseudo_bulk_idx, cell_type_idx, cre_idx] = cre_sum
            pseudo_bulk_n_array[pseudo_bulk_idx, cell_type_idx, cre_idx] = n_samples

        pseudo_bulk_t7_sum_test_result = {
            'pseudo_bulk_t7': pseudo_bulk_t7_array,
            'pseudo_bulk_cre': pseudo_bulk_cre_array,
            'pseudo_bulk_n': pseudo_bulk_n_array,
        }

        # Cache result
        self.pseudo_bulk_t7_sum_test_results.append(pseudo_bulk_t7_sum_test_result)
        self.pseudo_bulk_t7_sum_test_configs.append(config)
        return pseudo_bulk_t7_sum_test_result

    def scvi(self, use_model: Literal['STARRFISHVI', 'SCVI'] = 'STARRFISHVI', model_args: dict = None, train_args: dict = None) -> dict:
        """
        Run single-cell variational inference (scVI) to model CRE activity.

        Parameters
        ----------
        use_model : {'STARRFISHVI', 'SCVI'}, optional
            Model to use: 'STARRFISHVI' for STARR-FISH-specific model or 'SCVI' for standard
            scVI (default: 'STARRFISHVI')
        model_args : dict, optional
            Arguments to pass to model initialization
        train_args : dict, optional
            Arguments to pass to model training

        Returns
        -------
        dict
            Dictionary containing:
            - 'model': Trained scVI model
            - 'latent': Latent representations
            - 'config': Configuration parameters

        Notes
        -----
        Uses variational inference to learn low-dimensional representations of CRE activity
        while accounting for technical variation. STARRFISHVI is optimized for STARR-FISH data.
        """
        # Initialize cache attributes if needed
        if not hasattr(self, 'scvi_results'):
            self.scvi_results = []
            self.scvi_configs = []

        # Select model
        if use_model == 'STARRFISHVI':
            SCVIMODEL = STARRFISHVI
        elif use_model == 'SCVI':
            SCVIMODEL = scvi.model.SCVI

        # Infer infection rate prior
        if 'T7CRE' in self.adata.obsm.keys():
            non_infected_cells = ((self.adata.obsm['T7CRE'] > 0).sum(axis=1) == 0).sum()
        else:
            non_infected_cells = ((self.adata.obsm['CRE'] > 0).sum(axis=1) == 0).sum()
        infection_rate = -np.log(non_infected_cells / self.adata.shape[0]).item()

        # Set default model arguments
        if model_args is None:
            model_args = {'n_latent': 10, 'n_hidden': 128}
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

        # Set default training arguments
        if train_args is None:
            train_args = {
                'max_epochs': 500,
                'batch_size': 1280,
                'accelerator': 'gpu' if torch.cuda.is_available() else 'auto'
            }
            if torch.cuda.is_available():
                train_args['devices'] = 1

        # Create config for caching (exclude non-deterministic parameters)
        config = {
            'use_model': use_model,
            'model_args': {k: v for k, v in model_args.items() if k != 'infection_rate_library_size'},
            'train_args': {k: v for k, v in train_args.items() if k not in ['accelerator', 'devices']}
        }

        # Check cache for existing results
        for stored_config, scvi_result in zip(self.scvi_configs, self.scvi_results):
            if stored_config == config:
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
        # Cache result
        self.scvi_results.append(adata_mvi)
        self.scvi_configs.append(config)
        return adata_mvi

    def corr_atac_cpm(self, cell_types_to_use: Union[List, pd.Series]=None, cres_to_use: Union[List, pd.Series]=None,
                      acvitity_df: pd.DataFrame = None, log_atac=False, log_activity=False,
                      filter_by_atac_z_threshold=None,
                      filter_by_atac_raw_threshold=None,
                      filter_by_negative_control_z_threshold=None,
                      attr_to_use='atac_cpm') -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Compute correlation between STARR-FISH activity and ATAC-seq chromatin accessibility.

        Parameters
        ----------
        cell_types_to_use : list or pd.Series, optional
            Cell types to include in correlation
        cres_to_use : list or pd.Series, optional
            CREs to include in correlation
        acvitity_df : pd.DataFrame, optional
            Custom activity matrix (default: uses stored results)
        log_atac : bool, optional
            Log-transform ATAC-seq values (default: False)
        log_activity : bool, optional
            Log-transform activity values (default: False)
        filter_by_atac_z_threshold : float, optional
            Filter CREs by ATAC-seq z-score threshold
        filter_by_atac_raw_threshold : float, optional
            Filter CREs by raw ATAC-seq threshold
        filter_by_negative_control_z_threshold : float, optional
            Filter by negative control z-score threshold
        attr_to_use : str, optional
            Attribute name for ATAC data (default: 'atac_cpm')

        Returns
        -------
        tuple of (pd.DataFrame, pd.DataFrame)
            Correlation matrix and p-value matrix (cell types x CREs)

        Notes
        -----
        Computes Pearson correlation between STARR-FISH measured enhancer activity and
        ATAC-seq chromatin accessibility across cell types to validate CRE activity predictions.
        """
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
        # Initialize cache attributes if needed
        if not hasattr(self, 'cross_talk_test_results'):
            self.cross_talk_test_results = []
            self.cross_talk_test_configs = []

        # Create config for caching
        config = {
            'cell_types_to_use': cell_types_to_use,
            'normalize_by_cell_rna': normalize_by_cell_rna,
            'normalize_by_volume': normalize_by_volume,
            'method': method
        }

        # Check cache for existing results
        for stored_config, cross_talk_test_result in zip(self.cross_talk_test_configs, self.cross_talk_test_results):
            if stored_config == config:
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
        # Cache result
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
        """
        Compute correlation between two STARR-FISH activity matrices.

        Parameters
        ----------
        activity_df1 : pd.DataFrame
            First activity matrix (cell types x CREs)
        activity_df2 : pd.DataFrame
            Second activity matrix (cell types x CREs) to correlate with first
        cell_types_to_use : list or pd.Series, optional
            Cell types to include in correlation
        log_activity : bool, optional
            Log-transform activity values before correlation (default: False)

        Returns
        -------
        tuple of (pd.DataFrame, pd.DataFrame)
            Correlation matrix and p-value matrix (CREs x CREs)

        Notes
        -----
        Static method to compare CRE activity patterns between different STARR-FISH
        experiments or analysis methods. Useful for validating reproducibility or
        comparing different normalization strategies.
        """
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
        # Initialize cache attributes if needed
        if not hasattr(self, 'negbiom_results'):
            self.negbiom_results = []
            self.negbiom_configs = []

        if n_jobs is None:
            n_jobs = os.cpu_count() * 0.8  # Use 80% of available cores

        # Create config for caching
        config = {
            'cell_types_to_use': cell_types_to_use,
            'cres_to_use': cres_to_use,
            'stan_model_bg': stan_model_bg,
            'stan_model_main': stan_model_main,
            'chains': chains,
            'iter_warmup': iter_warmup,
            'iter_sampling': iter_sampling,
        }

        # Check cache for existing results
        for stored_config, negbiom_result in zip(self.negbiom_configs, self.negbiom_results):
            if stored_config == config:
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
        background_results = Parallel(n_jobs=min(n_jobs, len(cell_types_to_use.unique())), verbose=10)(
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
            parallel_results = Parallel(n_jobs=n_jobs, verbose=10)(
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
        # Cache result
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
        results = Parallel(n_jobs=n_jobs, verbose=10)(
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
