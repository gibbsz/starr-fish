# %% Fixed version with numerical stability
import torch
import torch.nn as nn
import torch.optim as optim
import time
import numpy as np
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
starrfish3_sec1 = STARRFISH.load('results/starrfish3_sec1.pkl')
starrfish3_sec2 = STARRFISH.load('results/starrfish3_sec2.pkl')
# %%
class T7CRE_DistributionEM:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu', use_x0=True):
        self.device = device
        self.use_x0 = use_x0
    
    def expectation_step(self, x0, x1, x2, obs_t7, obs_cre, cell_type_indexes, dim=1000):
        """E-step: Calculate posterior distribution of k given current parameters"""
        obs_t7 = obs_t7.to(self.device).float()
        obs_cre = obs_cre.to(self.device).float()
        cell_type_indexes = cell_type_indexes.to(self.device).long()
        x0 = x0.to(self.device).float()
        x1 = x1.to(self.device).float()
        x2 = x2.to(self.device).float()

        k_values = torch.arange(dim, device=self.device, dtype=torch.float32)  # (dim,)
        
        # Get lambda values for each observation based on cell type
        k_lambda = torch.exp(torch.clamp(x0[cell_type_indexes], max=10))  # Clamp to prevent overflow
        
        # Expand dimensions for broadcasting
        obs_t7_expanded = obs_t7.unsqueeze(1)  # (n_obs, 1)
        obs_cre_expanded = obs_cre.unsqueeze(1)  # (n_obs, 1)
        k_lambda_expanded = k_lambda.unsqueeze(1)  # (n_obs, 1)
        k_values_expanded = k_values.unsqueeze(0)  # (1, dim)
        
        if self.use_x0:
            # k ~ Poisson(k_lambda): log P(k|k_lambda)
            ll_k = torch.distributions.Poisson(k_lambda_expanded).log_prob(k_values_expanded)  # (n_obs, dim)
        else:
            ll_k = 0
            
        # Vectorized obs_t7 ~ NegBinom(exp(x1[0]) * k, torch.sigmoid(x1[1]))
        total_counts_t7 = torch.exp(torch.clamp(x1[0], min=-10, max=10)) # Apply constraint
        logits_t7 = torch.clamp(x1[1], min=-10, max=10)
        mu_t7 = total_counts_t7 * k_values  # (dim,)
        mu_t7_expanded = mu_t7.unsqueeze(0)  # (1, dim)
        
        # T7 log PMF computation
        ll_t7 = torch.distributions.negative_binomial.NegativeBinomial(
            total_count=mu_t7_expanded, logits=logits_t7
        ).log_prob(obs_t7_expanded)  # (n_obs, dim)
        
        # Vectorized obs_cre ~ NegBinom(exp(x2[cell_type_indexes, 0]) * k, torch.sigmoid(x2[cell_type_indexes, 1]))
        total_counts_cre = torch.exp(torch.clamp(x2[cell_type_indexes, 0], min=-10, max=10))  # (n_obs,)
        logits_cre = torch.clamp(x2[cell_type_indexes, 1], min=-10, max=10).unsqueeze(1) # (n_obs,)
        mu_cre = total_counts_cre.unsqueeze(1) * k_values.unsqueeze(0)  # (n_obs, dim)

        # CRE log PMF computation
        ll_cre = torch.distributions.negative_binomial.NegativeBinomial(
            total_count=mu_cre, logits=logits_cre
        ).log_prob(obs_cre_expanded)  # (n_obs, dim)

        # Combined likelihood
        likelihood_mat = ll_k + ll_t7 + ll_cre
        
        # Posterior distribution using logsumexp for numerical stability
        likelihood_sum = torch.logsumexp(likelihood_mat, dim=1, keepdim=True)
        posterior_k = likelihood_mat - likelihood_sum
        
        return posterior_k
    
    def maximization_step_x0(self, x0, cell_type_indexes, posterior_k):
        """M-step: Optimize x0 parameters"""
        x0_orig = x0.clone().detach()
        x0 = nn.Parameter(x0.clone().detach().requires_grad_(True))
        optimizer = optim.LBFGS([x0])  # Reduced lr and max_iter
        
        def closure():
            optimizer.zero_grad()
            
            dim = posterior_k.shape[1]
            k_values = torch.arange(dim, device=self.device, dtype=torch.float32)
            
            # Apply constraint within closure and clamp to prevent overflow
            x0_clamped = torch.clamp(x0, min=-10, max=10)
            
            # Get lambda values for each observation based on cell type
            k_lambda = torch.exp(x0_clamped[cell_type_indexes])  # (n_obs,)
            k_lambda_expanded = k_lambda.unsqueeze(1)  # (n_obs, 1)
            k_values_expanded = k_values.unsqueeze(0)  # (1, dim)
            
            # k ~ Poisson(k_lambda): log P(k|k_lambda)
            ll_k = torch.distributions.Poisson(k_lambda_expanded).log_prob(k_values_expanded)

            # Compute expected log likelihood
            ll = posterior_k + ll_k
            ll_sum = torch.logsumexp(ll, dim=1)
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
            
            obs_t7_expanded = obs_t7.unsqueeze(1)  # (n_obs, 1)
            dim = posterior_k.shape[1]
            k_values = torch.arange(dim, device=self.device, dtype=torch.float32)
            
            # Vectorized obs_t7 ~ NegBinom(exp(x1[0]) * k, torch.sigmoid(x1[1]))
            total_counts_t7 = torch.exp(torch.clamp(x1[0], min=-10, max=10)) # Apply constraint
            logits_t7 = torch.clamp(x1[1], min=-10, max=10)
            mu_t7 = total_counts_t7 * (k_values + 1e-8)  # (dim,)
            mu_t7_expanded = mu_t7.unsqueeze(0)  # (1, dim)
            
            # T7 log PMF computation
            ll_t7 = torch.distributions.negative_binomial.NegativeBinomial(
                total_count=mu_t7_expanded, logits=logits_t7
            ).log_prob(obs_t7_expanded)  # (n_obs, dim)
            
            ll = posterior_k + ll_t7
            ll_sum = torch.logsumexp(ll, dim=1)
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
            
            obs_cre_expanded = obs_cre.unsqueeze(1)  # (n_obs, 1)
            dim = posterior_k.shape[1]
            k_values = torch.arange(dim, device=self.device, dtype=torch.float32)
            
            # Vectorized obs_cre ~ NegBinom(exp(x2[cell_type_indexes, 0]) * k, torch.sigmoid(x2[cell_type_indexes, 1]))
            total_counts_cre = torch.exp(torch.clamp(x2[cell_type_indexes, 0], min=-10, max=10))  # (n_obs,)
            logits_cre = torch.clamp(x2[cell_type_indexes, 1], min=-10, max=10).unsqueeze(1)  # (n_obs, 1)
            mu_cre = total_counts_cre.unsqueeze(1) * (k_values + 1e-8).unsqueeze(0)  # (n_obs, dim)

            # CRE log PMF computation
            ll_cre = torch.distributions.negative_binomial.NegativeBinomial(
                total_count=mu_cre, logits=logits_cre
            ).log_prob(obs_cre_expanded)  # (n_obs, dim)
            
            ll = posterior_k + ll_cre
            ll_sum = torch.logsumexp(ll, dim=1)
            loss = -ll_sum.sum()
            
            # Add L2 regularization to prevent extreme values
            loss += 0.001 * (x2 ** 2).sum()
            
            loss.backward()
            return loss
        
        optimizer.step(closure)
        
        return x2.detach(), x2.detach() - x2_orig

    def fit(self, cell_types, obs_t7, obs_cre, dim=1000, max_iter=50):
        """Main EM algorithm"""
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
        obs_t7 = obs_t7.astype(np.float32)
        obs_cre = obs_cre.astype(np.float32)
        
        # Convert categorical cell types to integer indices
        unique_celltypes, cell_type_indexes = np.unique(cell_types, return_inverse=True)
        
        # Convert to torch tensors (no more .astype calls after this point)
        cell_type_indexes = torch.from_numpy(cell_type_indexes).to(self.device).long()
        obs_t7 = torch.from_numpy(obs_t7).to(self.device)
        obs_cre = torch.from_numpy(obs_cre).to(self.device)

        # Initialize parameters more conservatively
        # Group by cell types to compute initial x0
        x0_init = []
        # Convert tensors back to numpy for initialization calculations
        cell_type_indexes_np = cell_type_indexes.cpu().numpy()
        obs_t7_np = obs_t7.cpu().numpy()

        for i, ct in enumerate(unique_celltypes):
            mask = (cell_type_indexes_np == i)
            # Use actual zero fraction instead of hardcoded 0.99
            zero_frac = max((obs_t7_np[mask] == 0).mean(), 0.01)  # Ensure it's not too extreme
            zero_frac = min(zero_frac, 0.99)  # Cap at 0.99
            x0_init.append(np.log(-np.log(zero_frac)))

        x0 = torch.tensor(x0_init, device=self.device, dtype=torch.float32)
        # Clamp initial x0 to reasonable range
        x0 = torch.clamp(x0, min=-5, max=5)
        
        x1 = torch.tensor([-1.0, 0], device=self.device, dtype=torch.float32)  # Start with more conservative value
        x2 = torch.tensor(np.zeros((len(unique_celltypes), 2)), device=self.device, dtype=torch.float32)  # Start with more conservative value
        
        print(f"Running EM algorithm on {self.device}")
        print(f"Initial x0: {x0.cpu().numpy().mean()}")
        print(f"Initial x1: {x1.cpu().numpy()}")
        print(f"Initial x2: {x2.cpu().numpy().mean()}")

        for i in range(max_iter):
            # Check for NaN/inf values before each step
            if torch.isnan(x0).any() or torch.isinf(x0).any() or torch.isnan(x1).any() or torch.isinf(x1).any() or torch.isnan(x2).any() or torch.isinf(x2).any():
                print(f"Warning: NaN/inf detected in parameters at step {i}")
                break
            
            # E-step
            start = time.time()
            posterior_k = self.expectation_step(x0, x1, x2, obs_t7, obs_cre, cell_type_indexes, dim=dim)
            mid1 = time.time()
            print(f"Step {i}, E-step time: {mid1 - start:.4f}s")
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

            # check convergence
            if torch.all(torch.abs(x0_diff) / torch.abs(x0) < 1e-5) and torch.all(torch.abs(x1_diff) / torch.abs(x1) < 1e-5) and torch.all(torch.abs(x2_diff) / torch.abs(x2) < 1e-5):
                print(f"Converged at step {i}")
                break
            
        return x0.cpu().numpy(), x1.cpu().numpy(), x2.cpu().numpy()

# %% for cre in t7_counts.columns:
common_celltypes = starrfish3_sec2.get_celltypes().value_counts().index.intersection(starrfish3_sec1.get_celltypes().value_counts().index)
t7_counts_sec1 = starrfish3_sec1.get_t7_expression()[starrfish3_sec1.get_celltypes().isin(common_celltypes)]
t7_counts_sec2 = starrfish3_sec2.get_t7_expression()[starrfish3_sec2.get_celltypes().isin(common_celltypes)]

cre_counts_sec1 = starrfish3_sec1.get_cre_expression().loc[t7_counts_sec1.index]
cre_counts_sec2 = starrfish3_sec2.get_cre_expression().loc[t7_counts_sec2.index]

celltypes_sec1 = starrfish3_sec1.get_celltypes().loc[t7_counts_sec1.index]
celltypes_sec2 = starrfish3_sec2.get_celltypes().loc[t7_counts_sec2.index]

x0_df_sec1 = pd.DataFrame(index=common_celltypes, columns=t7_counts_sec1.columns)
x0_df_sec2 = pd.DataFrame(index=common_celltypes, columns=t7_counts_sec2.columns)
x1_df_sec1 = pd.DataFrame(index=t7_counts_sec1.columns, columns=['total_counts', 'logits'])
x1_df_sec2 = pd.DataFrame(index=t7_counts_sec2.columns, columns=['total_counts', 'logits'])
x2_r_df_sec1 = pd.DataFrame(index=common_celltypes, columns=t7_counts_sec1.columns)
x2_r_df_sec2 = pd.DataFrame(index=common_celltypes, columns=t7_counts_sec2.columns)
x2_p_df_sec1 = pd.DataFrame(index=common_celltypes, columns=t7_counts_sec1.columns)
x2_p_df_sec2 = pd.DataFrame(index=common_celltypes, columns=t7_counts_sec2.columns)
# %%
em_model = T7CRE_DistributionEM(device='cuda', use_x0=False)
for cre in t7_counts_sec1.columns:
    x0_sec1, x1_sec1, x2_sec1 = em_model.fit(celltypes_sec1.values, t7_counts_sec1[cre], cre_counts_sec1[cre], dim=50)
    x0_sec2, x1_sec2, x2_sec2 = em_model.fit(celltypes_sec2.values, t7_counts_sec2[cre], cre_counts_sec2[cre], dim=50)
    x0_df_sec1[cre] = x0_sec1
    x0_df_sec2[cre] = x0_sec2
    x1_df_sec1.loc[cre] = x1_sec1
    x1_df_sec2.loc[cre] = x1_sec2
    x2_r_df_sec1[cre] = x2_sec1[:, 0]
    x2_p_df_sec1[cre] = x2_sec1[:, 1]
    x2_r_df_sec2[cre] = x2_sec2[:, 0]
    x2_p_df_sec2[cre] = x2_sec2[:, 1]
# %%
x0_df_sec1.to_csv(f'{PWD}/results/expr3/t7cre_mle_separate_nox0.x0.sec1.csv')
x0_df_sec2.to_csv(f'{PWD}/results/expr3/t7cre_mle_separate_nox0.x0.sec2.csv')
x1_df_sec1.to_csv(f'{PWD}/results/expr3/t7cre_mle_separate_nox0.x1.sec1.csv')
x1_df_sec2.to_csv(f'{PWD}/results/expr3/t7cre_mle_separate_nox0.x1.sec2.csv')
x2_r_df_sec1.to_csv(f'{PWD}/results/expr3/t7cre_mle_separate_nox0.x2.r.sec1.csv')
x2_r_df_sec2.to_csv(f'{PWD}/results/expr3/t7cre_mle_separate_nox0.x2.r.sec2.csv')
x2_p_df_sec1.to_csv(f'{PWD}/results/expr3/t7cre_mle_separate_nox0.x2.p.sec1.csv')
x2_p_df_sec2.to_csv(f'{PWD}/results/expr3/t7cre_mle_separate_nox0.x2.p.sec2.csv')