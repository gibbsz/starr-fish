# %%
import scvi
import numpy as np
import warnings
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
import scanpy as sc
from scipy.stats import pearsonr, spearmanr, ttest_ind
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
import pyBigWig
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
    adata = sc.read_h5ad(adata_path)
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
    # rename enh
    adata.uns['CRE_info']['enh'] = adata.uns['CRE_info']['Chrom'] + ':' + adata.uns['CRE_info']['Start'].astype(str) + '-' + adata.uns['CRE_info']['End'].astype(str)
    # rename best_subclass
    adata.uns['CRE_info']['best_subclass'] = adata.uns['CRE_info']['best_subclass'].str.replace('_', ' ')
    adata.uns['CRE_info'].index = ['CRE' + str(i+1).zfill(3) for i in range(len(adata.uns['CRE_info']))]
    adata.obsm['CRE'] = adata.obsm['CRE'][adata.uns['CRE_info'].index]
    return adata
# %% preprocess and load data
adata2 = preprocess(f'{PWD}/Data/scdata_03_14_BRBB500gn_withCRE_final.h5ad')
cres = adata2.uns['CRE_info']['enh'][adata2.uns['CRE_info']['Start'] != '']
# %%
# load bigwig files
bigwig_path = f'/share/vault/Users/gz2294/STARR_FISH/Mouse_brain/Data/ATAC/wmb_bigwig/subclass_macs2/'
# list all bigwig files
bigwig_files = os.listdir(bigwig_path)
celltypes = []
for f in bigwig_files:
    if f.endswith('.bw'):
        celltype = f[4:-20]
        celltype = celltype.replace('.', '-')
        celltype = celltype.replace('_', ' ')
        celltypes.append(celltype)
atac_df = pd.DataFrame(index=cres, columns=celltypes)
for f in bigwig_files:
    if f.endswith('.bw'):
        celltype = f[4:-20]
        celltype = celltype.replace('.', '-')
        celltype = celltype.replace('_', ' ')
        bw = pyBigWig.open(bigwig_path + f)
        for i in cres:
            atac_df.loc[i, celltype] = bw.stats(adata2.uns['CRE_info']['Chrom'].loc[i], adata2.uns['CRE_info']['Start'].loc[i], adata2.uns['CRE_info']['End'].loc[i], type='sum')[0]
        bw.close()
atac_df.to_csv(f'{PWD}/Data/ATAC/rpkm_peakBysubclass.csv')
# %%
# read original atac_df
atac_cpm = pd.read_csv(f'{PWD}/Data/ATAC/cpm_peakBysubclass.csv', index_col=0)
atac_cpm.columns = atac_cpm.columns.str.replace('\\.', '-')
atac_cpm.columns = atac_cpm.columns.str.replace('_', ' ')
# %%
atac_df = atac_df.loc[atac_df.index.isin(atac_cpm.index)]
atac_cpm = atac_cpm.loc[atac_df.index]
# %%
fig, ax = plt.subplots(ncols=10, nrows=atac_df.shape[0]//10 + 1, figsize=(40, 4*(atac_df.shape[0]//10 + 1)))
for i, cre in enumerate(atac_df.index):
    atac_df.loc[cre] = atac_df.loc[cre].astype(float)
    atac_cpm.loc[cre] = atac_cpm.loc[cre].astype(float)
    sns.scatterplot(x=atac_df.loc[cre], y=atac_cpm.loc[cre], alpha=0.5, ax=ax[i//10, i%10])
    ax[i//10, i%10].set_xlabel('RPKM')
    ax[i//10, i%10].set_ylabel('CPM')
    ax[i//10, i%10].set_title(cre)
# %%
fig, ax = plt.subplots(ncols=10, nrows=atac_df.shape[1]//10 + 1, figsize=(40, 4*(atac_df.shape[0]//10 + 1)))
for i, celltype in enumerate(atac_df.columns):
    atac_df[celltype] = atac_df[celltype].astype(float)
    atac_cpm[celltype] = atac_cpm[celltype].astype(float)
    sns.scatterplot(x=atac_df[celltype], y=atac_cpm[celltype], alpha=0.5, ax=ax[i//10, i%10])
    ax[i//10, i%10].set_xlabel('RPKM')
    ax[i//10, i%10].set_ylabel('CPM')
    ax[i//10, i%10].set_title(celltype)
# %%
bw.header()
# %%
for f in bigwig_files:
    if f.endswith('.bw'):
        celltype = f[4:-20]
        celltype = celltype.replace('.', '-')
        celltype = celltype.replace('_', ' ')
        bw = pyBigWig.open(bigwig_path + f)
        print(bw.header())
# %%
fig, ax = plt.subplots(ncols=10, nrows=len(cell_types_to_use_nc_2)//10 + 1,
                       figsize=(40, 4*(len(cell_types_to_use_nc_2)//10 + 1)))
for i, celltype in enumerate(cell_types_to_use_nc_2):
    sns.scatterplot(x=np.log1p(neg_lib), y=celltype_neg.loc[celltype], alpha=0.5, ax=ax[i//10, i%10])
    # linear regression
    sns.regplot(x=np.log1p(neg_lib), y=celltype_neg.loc[celltype], ax=ax[i//10, i%10], scatter=False, color='red')
    # calculate pearson and spearman correlation
    pearson_corr = pearsonr(np.log1p(neg_lib), celltype_neg.loc[celltype])[0]
    spearman_corr = spearmanr(np.log1p(neg_lib), celltype_neg.loc[celltype])[0]
    # calculate p-value
    p_value = ttest_ind(np.log1p(neg_lib), celltype_neg.loc[celltype])[1]
    # add text
    ax[i//10, i%10].text(0.1, 0.9, f'Pearson: {pearson_corr:.2f}\nSpearman: {spearman_corr:.2f}\np-value: {p_value:.2e}',
                         transform=ax[i//10, i%10].transAxes, fontsize=12,
                         verticalalignment='top', bbox=dict(boxstyle='round,pad=0.3', edgecolor='black', facecolor='white'))
# %%
