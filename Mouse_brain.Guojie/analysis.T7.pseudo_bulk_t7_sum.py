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
# run the pseudo bulk bootstrap test for T7
pseudo_bulk_t7_sum_test_config = {
    'cell_types_to_use': None,
    't7_pseudo_bulk_size': 2000,
    'pseudo_bulk_number': 1000,
    'replace': True,
    'multiprocess_threads': 144,
    'infected_cells_threshold': 5
}
# %%
cre_blacklist = ['CRE061', 'CRE143', 'CRE001']
mismatching_cres = pd.read_csv('Data/AAV_ONT_Barcode_Counts_vs_Mismatch_Percentage.csv', index_col=0)
cre_blacklist = np.unique(cre_blacklist + mismatching_cres.index[mismatching_cres['MismatchPercent'] > 20].tolist()).tolist()
# %%
import pickle
save = True
infected_cells_threshold = 5
starrfish3_sec1 = STARRFISH.load('results/starrfish3_sec1.pkl')
res1 = starrfish3_sec1.pseudo_bulk_t7_sum_test(**pseudo_bulk_t7_sum_test_config)
if save:
    with open('results/starrfish3_sec1_pseudo_bulk_t7_sum_test.pkl', 'wb') as f:
        pickle.dump(res1, f)
    starrfish3_sec1.save('results/starrfish3_sec1.pkl')
del starrfish3_sec1

starrfish3_sec2 = STARRFISH.load('results/starrfish3_sec2.pkl')
res2 = starrfish3_sec2.pseudo_bulk_t7_sum_test(**pseudo_bulk_t7_sum_test_config)
if save:
    with open('results/starrfish3_sec2_pseudo_bulk_t7_sum_test.pkl', 'wb') as f:
        pickle.dump(res2, f)
    starrfish3_sec2.save('results/starrfish3_sec2.pkl')
del starrfish3_sec2

starrfish3 = STARRFISH.load('results/starrfish3.pkl')
res = starrfish3.pseudo_bulk_t7_sum_test(**pseudo_bulk_t7_sum_test_config)
if save:
    with open('results/starrfish3_pseudo_bulk_t7_sum_test.pkl', 'wb') as f:
        pickle.dump(res, f)
    starrfish3.save('results/starrfish3.pkl')
del starrfish3

# %%
with open('results/starrfish3_sec1_pseudo_bulk_t7_sum_test.pkl', 'rb') as f:
    res1 = pickle.load(f)
with open('results/starrfish3_sec2_pseudo_bulk_t7_sum_test.pkl', 'rb') as f:
    res2 = pickle.load(f)
with open('results/starrfish3_pseudo_bulk_t7_sum_test.pkl', 'rb') as f:
    res = pickle.load(f)
