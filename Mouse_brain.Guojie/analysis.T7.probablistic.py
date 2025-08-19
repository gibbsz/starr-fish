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
starrfish2 = STARRFISH.load('results/starrfish2.pkl')
starrfish3 = STARRFISH.load('results/starrfish3.pkl')
# get subclass name and subclass transform
subclass_annotation = pd.read_excel(f'Data/abc_atlas/allen_institute_nominature.xlsx')
subclass_annotation['subclass'] = subclass_annotation['subclass_id_label'].str.replace('^[0-9]+ ', '', regex=True)
subclass_annotation['subclass'] = subclass_annotation['subclass'].str.replace('/', '-', regex=True)
subclass_to_subclass_name = subclass_annotation['subclass_id_label'].groupby(subclass_annotation['subclass']).first().to_dict()
subclass_name_to_subclass = subclass_annotation['subclass'].groupby(subclass_annotation['subclass_id_label']).first().to_dict()
cre_blacklist = ['CRE061', 'CRE143', 'CRE001']
cre_whitelist = starrfish3_sec1.get_creinfo().index[~starrfish3_sec1.get_creinfo().index.isin(cre_blacklist)]
# %% define cell types to use for filtered data
negative_control_cres = starrfish3_sec1.get_negative_control_cres()
cell_types_counts1 = starrfish3_sec1.get_celltypes().value_counts()
cell_types_counts2 = starrfish3_sec2.get_celltypes().value_counts()
cell_types_to_use_1 = cell_types_counts1[cell_types_counts1 > 500].index
cell_types_to_use_2 = cell_types_counts2[cell_types_counts2 > 500].index
cell_types_to_use = cell_types_to_use_1.intersection(cell_types_to_use_2)
# check the negative control counts for those cell types
negative_control_counts1 = starrfish3_sec1.get_cre_expression()[negative_control_cres].groupby(starrfish3_sec1.get_celltypes()).sum()
negative_control_counts2 = starrfish3_sec2.get_cre_expression()[negative_control_cres].groupby(starrfish3_sec2.get_celltypes()).sum()
negative_control_sum_counts1 = starrfish3_sec1.get_cre_expression()[starrfish3_sec1.get_negative_control_cres()].sum(axis=1).groupby(starrfish3_sec1.get_celltypes()).sum()
negative_control_sum_counts2 = starrfish3_sec2.get_cre_expression()[starrfish3_sec2.get_negative_control_cres()].sum(axis=1).groupby(starrfish3_sec2.get_celltypes()).sum()
common_cell_types_sum_20_nc = negative_control_sum_counts1[negative_control_sum_counts1 > 20].index.intersection(negative_control_sum_counts2[negative_control_sum_counts2 > 20].index)
# define the cell types by the negative control counts > 50
cell_types_to_use_nc_1 = negative_control_sum_counts1[negative_control_sum_counts1 > 10].index
cell_types_to_use_nc_2 = negative_control_sum_counts2[negative_control_sum_counts2 > 10].index
cell_types_to_use_nc = cell_types_to_use_nc_1.intersection(cell_types_to_use_nc_2)
target_cres = starrfish3_sec1.get_creinfo().index[starrfish3_sec1.get_creinfo()['best_subclass'].isin(cell_types_to_use_nc_2)]
len(cell_types_to_use), len(cell_types_to_use_nc), len(cell_types_to_use_nc_2), len(target_cres)



# %% use scipy.optimize to estimate the negative binomial distribution parameters
from scipy.optimize import minimize
from scipy.special import logsumexp
def negative_binomial(x, obs):
    # parameters, x[0] is mean, x[1] is dispersion
    ll = np.sum(stats.nbinom.logpmf(obs, n=x[1]/1000, p=x[1]/(x[0]+x[1])))
    return -ll
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
            # CRE detection probability
            ll[:, i] += np.log(obs_cre == 0)
        else:
            # T7 detection 0 probability
            ll[:, i] += (1 - obs_t7) * i * x[1] + obs_t7 * np.log(1 - np.exp(i * x[1]))
            # CRE detection probability
            ll[:, i] += stats.nbinom.logpmf(obs_cre, n=i*x[4], p=x[4]/(x[4]+x[3]*np.exp(x[2])))
    ll_sum = logsumexp(ll, axis=1)
    return -ll_sum.sum()
# use CRE129 in Endo NN
obs_cre = starrfish3_sec1.get_cre_expression()['CRE129'][starrfish3_sec1.get_celltypes() == 'Endo NN'].values
obs_t7 = (starrfish3_sec1.get_t7_expression()['CRE129'][starrfish3_sec1.get_celltypes() == 'Endo NN'].values > 0).astype(int)
# make init guess
init_guess_nb = [np.mean(obs_cre)*1000, np.mean(obs_cre)**2 / (np.var(obs_cre) - np.mean(obs_cre))]
init_guess_pois_nb = [-np.log((obs_t7 == 0).mean()), np.log(0.05), np.log(0.05), np.mean(obs_cre)*1000, np.mean(obs_cre)**2 / (np.var(obs_cre) - np.mean(obs_cre))]
# add bounds for estimation
estimates_nb = minimize(negative_binomial, init_guess_nb, args=(obs_cre,), bounds=((1e-8, None), (1e-8, None)))
estimates_pois_nb = minimize(poisson_negative_binomial, init_guess_pois_nb, args=(obs_cre, obs_t7), 
                             bounds=((1e-8, None), (None, -1e-10), (None, -1e-10), (1e-8, None), (1e-8, None)), method='L-BFGS-B')





# %%
# use CRE129 in Endo NN
obs_cre = starrfish3_sec2.get_cre_expression()['CRE129'][starrfish3_sec2.get_celltypes() == 'Endo NN'].values
obs_t7 = (starrfish3_sec2.get_t7_expression()['CRE129'][starrfish3_sec2.get_celltypes() == 'Endo NN'].values > 0).astype(int)
# make init guess
init_guess_nb = [np.mean(obs_cre)*1000, np.mean(obs_cre)**2 / (np.var(obs_cre) - np.mean(obs_cre))]
init_guess_pois_nb = [-np.log((obs_t7 == 0).mean()), np.log(0.05), np.log(0.05), np.mean(obs_cre)*1000, np.mean(obs_cre)**2 / (np.var(obs_cre) - np.mean(obs_cre))]
# add bounds for estimation
estimates_nb = minimize(negative_binomial, init_guess_nb, args=(obs_cre,), bounds=((1e-8, None), (1e-8, None)), method='L-BFGS-B')
estimates_pois_nb = minimize(poisson_negative_binomial, init_guess_pois_nb, args=(obs_cre, obs_t7), 
                             bounds=((1e-8, None), (None, -1e-10), (None, -1e-10), (1e-8, None), (1e-8, None)), method='L-BFGS-B')


# %%
infection_rate_df, mu_df, disp_df = starrfish3.poisson_neg_binom_mle_separate(None, None)
# %%
