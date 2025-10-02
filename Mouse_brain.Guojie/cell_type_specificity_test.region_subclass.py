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
    # add region label
    cluster_annotation_term = pd.read_excel('Data/abc_atlas/allen_institute_nominature.xlsx')
    adata.obs['region'] = cluster_annotation_term['neighborhood'].groupby(cluster_annotation_term['subclass_label']).first().reindex(adata.obs['subclass']).values
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
adata3 = preprocess(f'{PWD}/Data/scdata_5_28_2025_BRBB500gn_final_CRE_T7CRE.h5ad')
# split the adata3 into two parts based on two sections
adata3.obs['section'] = (adata3.obsm['X_spatial'][:, 0] >= -1900).astype(int)
adata3_sec1 = adata3[adata3.obs['section'] == 0].copy()
adata3_sec2 = adata3[adata3.obs['section'] == 1].copy()
# make two STARRFISH objects
starrfish3_sec1 = STARRFISH(adata3_sec1)
starrfish3_sec2 = STARRFISH(adata3_sec2)
starrfish3 = STARRFISH(adata3)


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


# %% define regions for the test
regions = starrfish3.adata.obs['region'].str.split(';').explode().unique().tolist()
for region in regions:
    print(f'Processing region: {region}')
    cell_types_to_use = starrfish3.get_celltypes()[starrfish3.adata.obs['region'].str.contains(region)].unique().tolist()
    fig = starrfish3.plot_cluster(cell_types_to_use)
    fig.savefig(f'results/expr3/region_subclass/region_{region}.pdf')
    # %% run the fold change test for T7
    fold_change_test_config = {"cell_types_to_use": cell_types_to_use,
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
                            "n_jobs": 56,
                            'load_stored': True,}
    res1 = starrfish3_sec1.fold_change_test(**fold_change_test_config)
    starrfish3_sec1.save('results/starrfish3_sec1.region_subclass.pkl')
    res2 = starrfish3_sec2.fold_change_test(**fold_change_test_config)
    starrfish3_sec2.save('results/starrfish3_sec2.region_subclass.pkl')
    res = starrfish3.fold_change_test(**fold_change_test_config)
    starrfish3.save('results/starrfish3.region_subclass.pkl')