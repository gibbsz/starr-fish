# %%
import pandas as pd
import pyranges as pr
import re
import scanpy as sc
import os
# %%
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
adata = preprocess(f'/share/vault/Users/gz2294/starr-fish/STARRFISH_in_vivo/Data/scdata_03_14_BRBB500gn_withCRE_final.h5ad')
# %%
# Prepare CRE info
creinfo = adata.uns['CRE_info'].copy()
creinfo['cre'] = creinfo.index
# %%
celltypes = []
# list all de files
de_files = os.listdir('/share/vault/Users/gz2294/starr-fish/STARRFISH_in_vivo/Data/snapatac2_de/')
for f in de_files:
    if f.endswith('.csv'):
        # get the cell type name from cluster_annotation_term
        celltype = f.replace('.csv', '').replace('_', ' ')
        celltypes.append(celltype)
de_pval_df = pd.DataFrame(index=creinfo.index, columns=celltypes)
de_fc_df = pd.DataFrame(index=creinfo.index, columns=celltypes)
for f in de_files:
    if f.endswith('.csv'):
        celltype = f.replace('.csv', '').replace('_', ' ')
        try:
            de = pd.read_csv('/share/vault/Users/gz2294/starr-fish/STARRFISH_in_vivo/Data/snapatac2_de/' + f, sep='\t')
            # rename de index
            de.index = creinfo['cre'].groupby(creinfo['enh']).first().loc[de['feature name']].values
            # assign pval and fc to the de_pval_df and de_fc_df
            de_pval_df.loc[de.index, celltype] = de['p-value']
            de_fc_df.loc[de.index, celltype] = de['log2(fold_change)']
        except:
            print(f'Error processing {f}, skipping...')
            continue
# %%
# fill na with 0 for foldchange, 1 for p-value
de_fc_df = de_fc_df.fillna(0)
de_pval_df = de_pval_df.fillna(1)
# %%
de_fc_df.T.to_csv('/share/vault/Users/gz2294/starr-fish/STARRFISH_in_vivo/Data/snapatac2_de_fc.csv')
de_pval_df.T.to_csv('/share/vault/Users/gz2294/starr-fish/STARRFISH_in_vivo/Data/snapatac2_de_pval.csv')
# %%
