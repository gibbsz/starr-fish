# implement of starrfish vae
# %%
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
load_full_stats = True
if not load:
    adata1 = preprocess(f'{PWD}/Data/scdata_12_11NoT7_BRBB500gn_withCRE_final.h5ad')
    adata2 = preprocess(f'{PWD}/Data/scdata_03_14_BRBB500gn_withCRE_final.h5ad')
adata_cpm = 'Data/ATAC_cpm_peakBysubclass.csv'
if load:
    if load_full_stats:
        starrfish1 = STARRFISH.load('results/full_stats/starrfish1.pkl')
        starrfish2 = STARRFISH.load('results/full_stats/starrfish2.pkl')
        starrfish1_filtered = STARRFISH.load('results/full_stats/starrfish1_filtered.pkl')
        starrfish2_filtered = STARRFISH.load('results/full_stats/starrfish2_filtered.pkl')
    else:
        starrfish1 = STARRFISH.load('results/simple/starrfish1.pkl')
        starrfish2 = STARRFISH.load('results/simple/starrfish2.pkl')
        starrfish1_filtered = STARRFISH.load('results/simple/starrfish1_filtered.pkl')
        starrfish2_filtered = STARRFISH.load('results/simple/starrfish2_filtered.pkl')
else:
    starrfish1 = STARRFISH(adata1, atac_cpm=adata_cpm)
    starrfish2 = STARRFISH(adata2, atac_cpm=adata_cpm)
    starrfish1_filtered = STARRFISH(adata1[(adata1.obsm['CRE'] > 0).sum(axis=1) > 0], atac_cpm=adata_cpm)
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
cres_to_use = lib_size_fold.index
cres_to_use = cres_to_use[cres_to_use != 'CRE217']  # remove CRE217
cres_to_use_libsize_high = lib_size_fold[(lib_size_fold > 1/10)].index
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
cell_types_to_use_1 = cell_types_counts1[cell_types_counts1 > 50].index
cell_types_to_use_2 = cell_types_counts2[cell_types_counts2 > 50].index
cell_types_to_use = cell_types_to_use_1.intersection(cell_types_to_use_2)
# check the negative control counts for those cell types
negative_control_counts1 = starrfish1_filtered.get_cre_expression()[negative_control_cres].groupby(starrfish1_filtered.get_celltypes()).sum()
negative_control_counts2 = starrfish2_filtered.get_cre_expression()[negative_control_cres].groupby(starrfish2_filtered.get_celltypes()).sum()
negative_control_sum_counts1 = starrfish1_filtered.get_cre_expression()[starrfish1_filtered.get_negative_control_cres()].sum(axis=1).groupby(starrfish1_filtered.get_celltypes()).sum()
negative_control_sum_counts2 = starrfish2_filtered.get_cre_expression()[starrfish2_filtered.get_negative_control_cres()].sum(axis=1).groupby(starrfish2_filtered.get_celltypes()).sum()
common_cell_types_sum_20_nc = negative_control_sum_counts1[negative_control_sum_counts1 > 20].index.intersection(negative_control_sum_counts2[negative_control_sum_counts2 > 20].index)
# define the cell types by the negative control counts > 50
cell_types_to_use_nc_1 = negative_control_sum_counts1[negative_control_sum_counts1 > 10].index
cell_types_to_use_nc_2 = negative_control_sum_counts2[negative_control_sum_counts2 > 10].index
cell_types_to_use_nc = cell_types_to_use_nc_1.intersection(cell_types_to_use_nc_2)
target_cres = starrfish2_filtered.get_creinfo().index[starrfish2_filtered.get_creinfo()['best_subclass'].isin(cell_types_to_use_nc_2)]
len(cell_types_to_use), len(cell_types_to_use_nc), len(cell_types_to_use_nc_2), len(target_cres)
# %%
negbiom_test_config = {"cell_types_to_use": None,
                       'cres_to_use': None}
starrfish2_filtered.negbiom_cmdstanpy(**negbiom_test_config)
starrfish2_filtered = reload(starrfish2_filtered)
starrfish2_filtered.save('results/full_stats/starrfish2_filtered_tmp.pkl')
# %%
starrfish1.negbiom_cmdstanpy(**negbiom_test_config)
starrfish1 = reload(starrfish1)
starrfish1.save('results/full_stats/starrfish1_tmp.pkl')
# %%
starrfish1_filtered.negbiom_cmdstanpy(**negbiom_test_config)
starrfish1_filtered = reload(starrfish1_filtered)
starrfish1_filtered.save('results/full_stats/starrfish1_filtered_tmp.pkl')
# %%
starrfish2.negbiom_cmdstanpy(**negbiom_test_config)
starrfish2 = reload(starrfish2)
starrfish2.save('results/full_stats/starrfish2_tmp.pkl')