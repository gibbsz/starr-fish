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
adata2 = preprocess(f'/share/vault/Users/gz2294/starr-fish/Mouse_brain.Guojie/Data/scdata_03_14_BRBB500gn_withCRE_final.h5ad')
cres = adata2.uns['CRE_info'].index[adata2.uns['CRE_info']['Start'] != '']
# %%
# load bigwig files
modality = ['H3K27ac', 'H3K9me3', 'H3K4me1', 'H3K27me3']
bigwig_path = f'{PWD}/DNAbw/'
# list all bigwig files
bigwig_files = os.listdir(bigwig_path)
for mod in modality:
    celltypes = []
    if mod == 'H3K9me3':
        pattern = f'{mod}.e100.bs100.sm1000.bw'
    else:
        pattern = f'{mod}.e100.bs100.sm300.bw'
    for f in bigwig_files:
        if f.endswith(pattern):
            celltype = f[4:-20]
            # split by '.' and take the first part
            celltype = celltype.split('.')[0]
            celltype = celltype.replace('.', '-')
            celltype = celltype.replace('_', ' ')
            celltypes.append(celltype)
    histone_df = pd.DataFrame(index=cres, columns=celltypes)
    for f in bigwig_files:
        if f.endswith(pattern):
            print(f)
            celltype = f[4:-20]
            # split by '.' and take the first part
            celltype = celltype.split('.')[0]
            celltype = celltype.replace('.', '-')
            celltype = celltype.replace('_', ' ')
            bw = pyBigWig.open(bigwig_path + f)
            for i in cres:
                histone_df.loc[i, celltype] = bw.stats(adata2.uns['CRE_info']['Chrom'].loc[i], int(adata2.uns['CRE_info']['Start'].loc[i]), int(adata2.uns['CRE_info']['End'].loc[i]), type='sum')[0]
            bw.close()
    histone_df.to_csv(f'{PWD}/{mod}_rpkm_peakBysubclass.csv')
# %%