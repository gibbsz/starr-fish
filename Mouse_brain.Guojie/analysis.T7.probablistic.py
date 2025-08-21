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
from scipy.optimize import minimize
from scipy.special import logsumexp
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
import numpy as np
import seaborn as sns
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


# %% check reproducibility
# read in files
sec1_disp = pd.read_csv(f'{PWD}/results/expr3/poisson_neg_binom_mle_separate.disp.sec1.csv', index_col=0)
sec1_mu = pd.read_csv(f'{PWD}/results/expr3/poisson_neg_binom_mle_separate.mu.sec1.csv', index_col=0)
sec1_infection = pd.read_csv(f'{PWD}/results/expr3/poisson_neg_binom_mle_separate.infection.rate.sec1.csv', index_col=0)
sec2_disp = pd.read_csv(f'{PWD}/results/expr3/poisson_neg_binom_mle_separate.disp.sec2.csv', index_col=0)
sec2_mu = pd.read_csv(f'{PWD}/results/expr3/poisson_neg_binom_mle_separate.mu.sec2.csv', index_col=0)
sec2_infection = pd.read_csv(f'{PWD}/results/expr3/poisson_neg_binom_mle_separate.infection.rate.sec2.csv', index_col=0)

# %% do pearson correlation
cre_corr, celltype_corr = starrfish3.corr_starrfish(sec1_mu, sec2_mu)



# %%
# we take a step back, only focus on the T7 transcripts
t7_counts = starrfish3.get_t7_expression()
# it should be a mixed poisson process, where each enzyme hit will trigger a transcript, so we can model it as such
# %%
def t7_distribution(x, obs_t7):
    # fit a mixed poisson distribution to the data
    # parameters x[0], lambda, infection rate
    # x[1], t7 lambda, T7 enzyme efficiency times detection efficiency, log transformed
    ll = np.zeros((obs_t7.shape[0], 100))
    for i in range(100):
        ll[:, i] += stats.poisson.logpmf(i, mu=x[0])
        ll[:, i] += stats.poisson.logpmf(obs_t7, mu=np.exp(x[1])*i)
    ll_sum = logsumexp(ll, axis=1)
    return -ll_sum.sum()
cre = 'CRE129'
estimates = minimize(t7_distribution, [1, -0.1], args=(t7_counts[cre].values,), 
                     bounds=((1e-10, None), (None, -1e-10)), method='L-BFGS-B')
    # add bounds for estimation
# %%
import time
# use EM algorighm to jointly estimate all parameters 
def t7_distribution_expectation(x, obs_t7, n_celltypes, cell_type_indexes, dim=1000):
    # E-step: given estimated x and obs_t7, calculate posterior distribution of k
    # k ~ Poisson(x[0])
    obs_t7 = np.asarray(obs_t7)  # Convert pandas Series to numpy array
    k_lambda = x[cell_type_indexes]  # shape: (n_obs,)
    
    # Vectorized computation using broadcasting
    k_values = np.arange(dim)  # shape: (dim,)
    obs_t7_expanded = obs_t7[:, np.newaxis]  # shape: (n_obs, 1)
    k_lambda_expanded = k_lambda[:, np.newaxis]  # shape: (n_obs, 1)
    # Fast Poisson log PMF implementation
    # Precompute log factorials for k_values
    log_factorial_k = np.concatenate([[0], np.cumsum(np.log(np.arange(1, dim)))])
    # k ~ Poisson(k_lambda): log P(k|k_lambda)
    ll_k = (k_values * np.log(k_lambda_expanded) - k_lambda_expanded - log_factorial_k)  # shape: (n_obs, dim)
    # obs_t7 ~ Poisson(exp(x[n_celltypes]) * k): log P(obs_t7|k)
    lambda_t7 = np.exp(x[n_celltypes])
    mu_t7 = lambda_t7 * k_values  # shape: (dim,)
    # Compute log factorials for obs_t7 values
    max_obs = int(obs_t7.max()) if len(obs_t7) > 0 else 0
    log_factorial_obs = np.concatenate([[0], np.cumsum(np.log(np.arange(1, max_obs + 1)))])
    obs_log_factorial = log_factorial_obs[obs_t7.astype(int)][:, np.newaxis]  # shape: (n_obs, 1)
    # T7 log PMF computation
    ll_t7 = (obs_t7_expanded * np.log(np.maximum(mu_t7, 1e-10)) - 
             mu_t7 - obs_log_factorial)  # shape: (n_obs, dim)
    # Combined likelihood
    likelihood_mat = ll_k + ll_t7
    # get posterior distribution
    likelihood_sum = logsumexp(likelihood_mat, axis=1)
    posterior_k = likelihood_mat - likelihood_sum[:, np.newaxis]
    return posterior_k

def t7_distribution_x1_maximization(x1, obs_t7, posterior_k):
    # M-step: given posterior of k, maximize x1
    obs_t7 = np.asarray(obs_t7)  # Convert pandas Series to numpy array
    dim = posterior_k.shape[1]
    k_values = np.arange(dim)  # shape: (dim,)
    obs_t7_expanded = obs_t7[:, np.newaxis]  # shape: (n_obs, 1)
    # Vectorized Poisson log PMF for obs_t7 ~ Poisson(exp(x1) * k)
    lambda_t7 = np.exp(x1)
    mu_t7 = lambda_t7 * k_values  # shape: (dim,)
    # Compute log factorials for obs_t7 values
    max_obs = int(obs_t7.max()) if len(obs_t7) > 0 else 0
    log_factorial_obs = np.concatenate([[0], np.cumsum(np.log(np.arange(1, max_obs + 1)))])
    obs_log_factorial = log_factorial_obs[obs_t7.astype(int)][:, np.newaxis]  # shape: (n_obs, 1)
    # T7 log PMF computation
    ll_t7 = (obs_t7_expanded * np.log(np.maximum(mu_t7, 1e-10)) - 
             mu_t7 - obs_log_factorial)  # shape: (n_obs, dim)
    ll = posterior_k + ll_t7
    ll_sum = logsumexp(ll, axis=1)
    return -ll_sum.sum()

def t7_distribution_x0_maximization(x0, cell_type_indexes, posterior_k):
    # M-step: given posterior of k, maximize x0
    dim = posterior_k.shape[1]
    k_values = np.arange(dim)  # shape: (dim,)
    # Get lambda values for each observation based on cell type
    k_lambda = x0[cell_type_indexes]  # shape: (n_obs,)
    k_lambda_expanded = k_lambda[:, np.newaxis]  # shape: (n_obs, 1)
    # Fast Poisson log PMF implementation
    # Precompute log factorials for k_values
    log_factorial_k = np.concatenate([[0], np.cumsum(np.log(np.arange(1, dim)))])
    # k ~ Poisson(k_lambda): log P(k|k_lambda)
    ll_k = (k_values * np.log(k_lambda_expanded) - k_lambda_expanded - log_factorial_k)  # shape: (n_obs, dim)
    ll = posterior_k + ll_k
    ll_sum = logsumexp(ll, axis=1)
    return -ll_sum.sum()

def t7_distribution_em(cell_types, obs_t7, dim=1000):
    unique_celltypes, cell_type_indexes = np.unique(cell_types, return_inverse=True)
    n_celltypes = unique_celltypes.size
    # initialize x array
    x0 = -np.log((obs_t7 == 0).groupby(cell_types).mean().loc[unique_celltypes])
    x1 = -1e-5
    for i in range(10):
        x = np.concatenate([x0, [x1]])
        # E-step
        start = time.time()
        posterior_k = t7_distribution_expectation(x, obs_t7, n_celltypes, cell_type_indexes, dim=dim)
        mid1 = time.time()
        print("Step", i, "E-step time:", mid1 - start)
        # M-step
        x0_opt = minimize(t7_distribution_x0_maximization, x0, args=(cell_type_indexes, posterior_k),
                          bounds=[(1e-10, None)]*n_celltypes, method='L-BFGS-B')
        mid2 = time.time()
        print("Step", i, "M-step x0 time:", mid2 - mid1)
        x1_opt = minimize(t7_distribution_x1_maximization, x1, args=(obs_t7, posterior_k),
                          bounds=[(None, -1e-10)], method='L-BFGS-B')
        print("Step", i, "M-step x1 time:", time.time() - mid2)
        x0 = x0_opt.x
        x1 = x1_opt.x
        print("Step", i, "x0:", x0, "x1:", x1)
    return x0, x1
# %% for cre in t7_counts.columns:
cre = 'CRE129'
estimates = t7_distribution_em(starrfish3.get_celltypes().values, t7_counts[cre])





# %% too slow, use torch and gpu to accelerate
