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
pseudo_bulk_glm_test_config = {
    'cell_types_to_use': None,
    'variate': 'T7',
    'norm_by_volm': False,
    'volm_covariate': False,  # normalize by T7, filter cells with T7 < 4
    'rna_covariate': False,
    'filter_infected_cells': False,
    'positive_x_or_y': False,  # normalize by T7
    'only_keep_positive_x': False,
    'only_keep_positive_y': False,  # normalize by negative control
    'log_x_y': True,
    'pseudo_bulk_size': [100, 200, 400, 600, 800, 1000, 1200, 1400, 1600, 2000, 2400, 2800, 3200, 3600, 4000],
    'pseudo_bulk_percentage': None,
    'pseudo_bulk_number': [100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100],
    'replace': True,
    'multiprocess_threads': 96,
}
# %%
cre_blacklist = ['CRE061', 'CRE143', 'CRE001']
mismatching_cres = pd.read_csv('Data/AAV_ONT_Barcode_Counts_vs_Mismatch_Percentage.csv', index_col=0)
cre_blacklist = np.unique(cre_blacklist + mismatching_cres.index[mismatching_cres['MismatchPercent'] > 20].tolist()).tolist()
# %%
starrfish3_sec1 = STARRFISH.load('results/starrfish3_sec1.pkl')
cell_counts1 = starrfish3_sec1.get_celltypes().value_counts()
res1 = starrfish3_sec1.pseudo_bulk_glm_test(**pseudo_bulk_glm_test_config)
res1_summary = res1['result'].copy()
res1['pseudo_bulk_adata'].uns['CRE_info'] = starrfish3_sec1.adata.uns['CRE_info'].copy()
res1['pseudo_bulk_adata'].obs.index.name = None
res1 = STARRFISH(res1['pseudo_bulk_adata'])
res1.adata.obs['percentage'] = res1.adata.obs['percentage'].astype(float)
res1.save('results/starrfish3_sec1_pseudo_bulk.pkl', overwrite_adata=True)
starrfish3_sec1.save('results/starrfish3_sec1.pkl')
del starrfish3_sec1

starrfish3_sec2 = STARRFISH.load('results/starrfish3_sec2.pkl')
cell_counts2 = starrfish3_sec2.get_celltypes().value_counts()
res2 = starrfish3_sec2.pseudo_bulk_glm_test(**pseudo_bulk_glm_test_config)
res2_summary = res2['result'].copy()
res2['pseudo_bulk_adata'].uns['CRE_info'] = starrfish3_sec2.adata.uns['CRE_info'].copy()
res2['pseudo_bulk_adata'].obs.index.name = None
res2 = STARRFISH(res2['pseudo_bulk_adata'])
res2.adata.obs['percentage'] = res2.adata.obs['percentage'].astype(float)
res2.save('results/starrfish3_sec2_pseudo_bulk.pkl', overwrite_adata=True)
starrfish3_sec2.save('results/starrfish3_sec2.pkl')
del starrfish3_sec2

starrfish3 = STARRFISH.load('results/starrfish3.pkl')
cell_counts = starrfish3.get_celltypes().value_counts()
res = starrfish3.pseudo_bulk_glm_test(**pseudo_bulk_glm_test_config)
res_summary = res['result'].copy()
res['pseudo_bulk_adata'].uns['CRE_info'] = starrfish3.adata.uns['CRE_info'].copy()
res['pseudo_bulk_adata'].obs.index.name = None
res = STARRFISH(res['pseudo_bulk_adata'])
res.adata.obs['percentage'] = res.adata.obs['percentage'].astype(float)
res.save('results/starrfish3_pseudo_bulk.pkl', overwrite_adata=True)
starrfish3.save('results/starrfish3.pkl')
del starrfish3
# %%
res = STARRFISH.load('results/starrfish3_pseudo_bulk.pkl')
res1 = STARRFISH.load('results/starrfish3_sec1_pseudo_bulk.pkl')
res2 = STARRFISH.load('results/starrfish3_sec2_pseudo_bulk.pkl')
# %% if start over
glm_test_config = {
    'cell_types_to_use': None,
    'variate': 'T7',
    'multiprocess_threads': 96,
}
glm_res1 = res1.glm_test(**glm_test_config)
glm_res2 = res2.glm_test(**glm_test_config)
glm_res = res.glm_test(**glm_test_config)
# %% check the results
cre_corr, celltype_corr = res.corr_starrfish(res1_summary['coef'], res2_summary['coef'])
# %% plot cell type corr
cre_corr['libsize'] = res.lib_size['counts'].loc[cre_corr.index]
celltype_corr['celltype_sec1'] = cell_counts1.loc[celltype_corr.index].values
celltype_corr['celltype_sec2'] = cell_counts2.loc[celltype_corr.index].values
celltype_corr['celltype_n'] = np.minimum(celltype_corr['celltype_sec1'], celltype_corr['celltype_sec2'])
# %% plot
fig, ax = plt.subplots(figsize=(4, 4))
sns.scatterplot(data=celltype_corr, x='celltype_n', y='pearson', ax=ax)
ax.set_xscale('log')
# %% visualize best cell type
fig, ax = plt.subplots(figsize=(4, 4))
sns.scatterplot(x=res1_summary['coef'].loc['Oligo NN'], y=res2_summary['coef'].loc['Oligo NN'], color='blue', ax=ax)
# plot negative control
sns.scatterplot(x=res1_summary['coef'].loc['Oligo NN', res1.get_negative_control_cres()], 
                y=res2_summary['coef'].loc['Oligo NN', res1.get_negative_control_cres()], color='orange', ax=ax)
sns.scatterplot(x=res1_summary['coef'].loc['Oligo NN', res1.get_positive_control_cres('Oligo NN', use='atac-peak')], 
                y=res2_summary['coef'].loc['Oligo NN', res1.get_positive_control_cres('Oligo NN', use='atac-peak')], color='red', ax=ax)
sns.scatterplot(x=res1_summary['coef'].loc['Oligo NN', cre_blacklist], 
                y=res2_summary['coef'].loc['Oligo NN', cre_blacklist], color='green', ax=ax)
# %% check CRE363, strongest CRE
fig, ax = plt.subplots(figsize=(4, 4))
ct = 'Oligo NN'
cre = 'CRE363'
sns.scatterplot(x=(res1.get_t7_expression()[(res1.get_celltypes() == ct)][cre]),
                y=(res1.get_cre_expression()[(res1.get_celltypes() == ct)][cre]), alpha=0.5,
                hue=res1.adata.obs[(res1.get_celltypes() == ct)]['size'], palette='coolwarm',
                ax=ax)
ax.set_xlabel('T7 Pseudo bulk Expression')
ax.set_ylabel('CRE Pseudo bulk Expression')
# %% check if covariate of size helps
thres = 0
import statsmodels.formula.api as smf
coef1 = []
pvalue1 = []
for cre in res1.get_cre_expression().columns:
    fit_data=pd.DataFrame({'y': np.log1p(res1.get_cre_expression()[(res1.get_celltypes() == ct) & (res1.adata.obs['size'] >= thres)][cre]), 
                           'x': np.log1p(res1.get_t7_expression()[(res1.get_celltypes() == ct) & (res1.adata.obs['size'] >= thres)][cre]), 
                           'volm': None, 'fov': None, 'RNA': None, 
                           'size': np.log(res1.adata.obs[(res1.get_celltypes() == ct) & (res1.adata.obs['size'] >= thres)]['size'])})
    glm_results = smf.ols('y ~ x + size', data=fit_data).fit()
    coef1.append(glm_results.params.get('x', np.nan))
    pvalue1.append(glm_results.pvalues.get('x', np.nan))
# %%
coef2 = []
pvalue2 = []
for cre in res1.get_cre_expression().columns:
    fit_data=pd.DataFrame({'y': np.log1p(res2.get_cre_expression()[(res2.get_celltypes() == ct) & (res2.adata.obs['size'] >= thres)][cre]), 
                           'x': np.log1p(res2.get_t7_expression()[(res2.get_celltypes() == ct) & (res2.adata.obs['size'] >= thres)][cre]), 
                           'volm': None, 'fov': None, 'RNA': None, 
                           'size': np.log(res2.adata.obs[(res2.get_celltypes() == ct) & (res2.adata.obs['size'] >= thres)]['size'])})
    glm_results = smf.ols('y ~ x + size', data=fit_data).fit()
    coef2.append(glm_results.params.get('x', np.nan))
    pvalue2.append(glm_results.pvalues.get('x', np.nan))
# %%
res = pd.DataFrame({'coef1': coef1, 'pvalue1': pvalue1, 'coef2': coef2, 'pvalue2': pvalue2}, index=res1.get_cre_expression().columns)
fig, ax = plt.subplots(figsize=(4, 4))
sns.scatterplot(x=res['coef1'], y=res['coef2'], color='blue', ax=ax)
# plot negative control
sns.scatterplot(x=res['coef1'].loc[res1.get_negative_control_cres()], 
                y=res['coef2'].loc[res1.get_negative_control_cres()], color='orange', ax=ax)
sns.scatterplot(x=res['coef1'].loc[res1.get_positive_control_cres('Oligo NN', use='atac-peak')], 
                y=res['coef2'].loc[res1.get_positive_control_cres('Oligo NN', use='atac-peak')], color='red', ax=ax)
sns.scatterplot(x=res['coef1'].loc[cre_blacklist], 
                y=res['coef2'].loc[cre_blacklist], color='green', ax=ax)
# %%
res['libsize'] = res1.lib_size['counts'].loc[res.index]
fig, ax = plt.subplots(figsize=(4, 4))
sns.scatterplot(data = res, x='coef1', y='coef2', hue='libsize', palette = 'coolwarm', ax=ax)
# %%
sns.scatterplot(x=res['libsize'].loc[res1.get_negative_control_cres()], 
                y=res['coef1'].loc[res1.get_negative_control_cres()], color='orange', ax=ax)
sns.scatterplot(x=res['libsize'].loc[res1.get_positive_control_cres('Oligo NN', use='atac-peak')], 
                y=res['coef1'].loc[res1.get_positive_control_cres('Oligo NN', use='atac-peak')], color='red', ax=ax)
sns.scatterplot(x=res['libsize'].loc[cre_blacklist], 
                y=res['coef1'].loc[cre_blacklist], color='green', ax=ax)
# %%
