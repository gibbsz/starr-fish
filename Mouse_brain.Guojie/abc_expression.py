# %%
import pandas as pd
from pathlib import Path
import numpy as np
import anndata
import time
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns
from abc_atlas_access.abc_atlas_cache.abc_project_cache import AbcProjectCache, LocalCache
PWD = '/gpfs/commons/groups/ren_lab/guojiezhong/starr-fish/Mouse_brain.Guojie'
# %%
download_base = Path('Mouse_brain.Guojie/Data/abc_atlas')
abc_cache = AbcProjectCache.from_cache_dir(download_base)

abc_cache.current_manifest
# %%
cell = abc_cache.get_metadata_dataframe(
    directory='WMB-10X',
    file_name='cell_metadata',
    dtype={'cell_label': str}
)
cell.set_index('cell_label', inplace=True)
print("Number of cells = ", len(cell))
cell.head(5)
# %%
cluster_details = abc_cache.get_metadata_dataframe(
    directory='WMB-taxonomy',
    file_name='cluster_to_cluster_annotation_membership_pivoted',
    keep_default_na=False
)
cluster_details.set_index('cluster_alias', inplace=True)
cluster_details.head(5)
# %%
cell_extended = cell.join(cluster_details, on='cluster_alias')
cell_extended.head(5)
# %%
abc_cache._local = True
for file in abc_cache.list_data_files('WMB-10Xv2'):
    try:
        file = abc_cache.get_data_path(directory='WMB-10Xv2', file_name=file)
        print(file)
    except Exception as e:
        print(f"Error processing {file}: {e}")
        # needs to be redownloaded
        abc_cache._local = False
        file = abc_cache.get_data_path(directory='WMB-10Xv2', file_name=file, force_download=True)
        print(f"Redownloaded {file}")
        abc_cache._local = True
# %%
for file in abc_cache.list_data_files('WMB-10Xv3'):
    try:
        file = abc_cache.get_data_path(directory='WMB-10Xv3', file_name=file)
        print(file)
    except Exception as e:
        print(f"Error processing {file}: {e}")
        # needs to be redownloaded
        abc_cache._local = False
        file = abc_cache.get_data_path(directory='WMB-10Xv3', file_name=file, force_download=True)
        print(f"Redownloaded {file}")
        abc_cache._local = True
# %% get gene list first
adata3 = sc.read_h5ad(f'{PWD}/Data/scdata_5_28_2025_BRBB500gn_final_CRE_T7CRE.h5ad')
adata3.X = adata3.obsm['X_raw'].copy()
sc.pp.normalize_total(adata3, target_sum=1e6)
sc.pp.log1p(adata3)
gene_list = adata3.var_names.tolist()
# %% load each 10xv2-raw and 10xv3-raw data and aggregate
wmb_10xv2_adata = None
for file in abc_cache.list_data_files('WMB-10Xv2'):
    print(file)
    if 'log2' in file:
        continue
    data_path = abc_cache.get_data_path(directory='WMB-10Xv2', file_name=file)
    adata = sc.read_h5ad(data_path)
    adata.var_names_make_unique()
    # filter gene list
    adata = adata[:, adata.var['gene_symbol'].isin(gene_list)]
    adata.obs['batch'] = file
    if wmb_10xv2_adata is None:
        wmb_10xv2_adata = adata
    else:
        wmb_10xv2_adata = wmb_10xv2_adata.concatenate(adata, index_unique=None)
    del adata
wmb_10xv3_adata = None
for file in abc_cache.list_data_files('WMB-10Xv3'):
    print(file)
    if 'log2' in file:
        continue
    data_path = abc_cache.get_data_path(directory='WMB-10Xv3', file_name=file)
    adata = sc.read_h5ad(data_path)
    adata.var_names_make_unique()
    # filter gene list
    adata = adata[:, adata.var['gene_symbol'].isin(gene_list)]
    adata.obs['batch'] = file
    if wmb_10xv3_adata is None:
        wmb_10xv3_adata = adata
    else:
        wmb_10xv3_adata = wmb_10xv3_adata.concatenate(adata, index_unique=None)
    del adata
# %% add labels to wmb_10xv2_adata and wmb_10xv3_adata, join on cell_label, ignore cell_barcode and library_label
columns_to_drop = ['cell_barcode', 'library_label']
cell_extended_filtered = cell_extended.drop(columns=columns_to_drop, errors='ignore')
wmb_10xv2_adata.obs = wmb_10xv2_adata.obs.join(
    cell_extended_filtered,
    on=['cell_label'], how='left'
)
wmb_10xv3_adata.obs = wmb_10xv3_adata.obs.join(
    cell_extended_filtered,
    on=['cell_label'], how='left'
)
# %% generate pseudo bulk
def generate_pseudobulk(adata, groupby):
    pseudobulk_dict = {}
    groups = adata.obs[groupby].unique()
    # remove nan groups
    groups = groups[~pd.isna(groups)]
    for group in groups:
        print(f"Processing group: {group}")
        group_adata = adata[adata.obs[groupby] == group]
        pseudobulk_counts = group_adata.X.sum(axis=0)
        pseudobulk_dict[group] = np.asarray(pseudobulk_counts).flatten()
    pseudobulk_df = pd.DataFrame.from_dict(
        pseudobulk_dict,
        orient='index',
        columns=adata.var_names
    )
    return pseudobulk_df
# %%
wmb = wmb_10xv2_adata.concatenate(wmb_10xv3_adata, index_unique=None)
# save
wmb.write_h5ad(f'{PWD}/Data/abc_atlas/WMB_10xv2v3_combined.h5ad')
wmb_10xv2_adata.write_h5ad(f'{PWD}/Data/abc_atlas/WMB_10xv2.h5ad')
wmb_10xv3_adata.write_h5ad(f'{PWD}/Data/abc_atlas/WMB_10xv3.h5ad')
# %% load back
wmb = sc.read_h5ad(f'{PWD}/Data/abc_atlas/WMB_10xv2v3_combined.h5ad')
wmb_10xv2_adata = sc.read_h5ad(f'{PWD}/Data/abc_atlas/WMB_10xv2.h5ad')
wmb_10xv3_adata = sc.read_h5ad(f'{PWD}/Data/abc_atlas/WMB_10xv3.h5ad')
# %%
wmb_pseudobulk = generate_pseudobulk(wmb, groupby='subclass')
wmb_pseudobulk.columns = wmb.var['gene_symbol']
wmb_10xv2_pseudobulk = generate_pseudobulk(wmb_10xv2_adata, groupby='subclass')
wmb_10xv2_pseudobulk.columns = wmb_10xv2_adata.var['gene_symbol']
wmb_10xv3_pseudobulk = generate_pseudobulk(wmb_10xv3_adata, groupby='subclass')
wmb_10xv3_pseudobulk.columns = wmb_10xv3_adata.var['gene_symbol']
adata3.X = adata3.obsm['X_raw']
starrfish_pseudobulk = generate_pseudobulk(adata3, groupby='subclass_name')
starrfish_pseudobulk = starrfish_pseudobulk[wmb_10xv2_pseudobulk.columns]
# %% log TPM transform
def log_tpm_transform(df):
    tpm = df.div(df.sum(axis=1), axis=0) * 1e6
    log_tpm = np.log1p(tpm)
    return log_tpm
wmb_pseudobulk_logtpm = log_tpm_transform(wmb_pseudobulk)
starrfish_pseudobulk_logtpm = log_tpm_transform(starrfish_pseudobulk)
# %% 
# raw correlation of wmb_pseudobulk vs starrfish_pseudobulk, row by row
# drop the cell types with < 100 cells in starrfish_pseudobulk_logtpm
celltype_n = adata3.obs['subclass_name'].value_counts()
correlation_matrix = pd.DataFrame(index=celltype_n.index[celltype_n >= 100],
                                  columns=celltype_n.index[celltype_n >= 100])
for subclass in correlation_matrix.index:
    for starrfish_subclass in correlation_matrix.columns:
        corr = np.corrcoef(
            wmb_pseudobulk_logtpm.loc[subclass],
            starrfish_pseudobulk_logtpm.loc[starrfish_subclass]
        )[0, 1]
        correlation_matrix.loc[subclass, starrfish_subclass] = corr
# reorder rows and columns
correlation_matrix = correlation_matrix.reindex(
    index=sorted(correlation_matrix.index),
    columns=sorted(correlation_matrix.columns)
)
# %% plot heatmap
fig, ax = plt.subplots(figsize=(25, 20))
sns.heatmap(correlation_matrix.astype(float), annot=False, cmap='coolwarm', ax=ax)
ax.set_title('Correlation between WMB Pseudobulk and STARRFISH Pseudobulk')
ax.set_xlabel('STARRFISH Subclass')
ax.set_ylabel('WMB Subclass')
fig.tight_layout()
fig.savefig(f'{PWD}/results/expr3/abc_atlas/WMB_vs_STARRFISH_pseudobulk_correlation_heatmap.pdf')
# %% use label transfer matrix
# first do pca and neighbors on wmb
sc.pp.normalize_total(wmb, target_sum=1e6)
sc.pp.log1p(wmb)
sc.tl.pca(wmb)
sc.pp.neighbors(wmb)
# %% perform ingestion
# rename the adata3 var names to gene ID
wmb.var['gene_ID'] = wmb.var_names.copy()
# add the umap coords to wmb.obsm
wmb_umap = np.array((wmb.obs['x'], wmb.obs['y'])).T
wmb.obsm['X_umap'] = wmb_umap
# add fake umap params
wmb.uns['umap'] = {}
wmb.uns['umap']['params'] = {'a': 1.0, 'b': 1.0}
# map gene symbols to gene IDs
adata3.var_names = wmb.var['gene_ID'].groupby(wmb.var['gene_symbol']).first().reindex(adata3.var_names).values
# reorder adata3 var to match wmb var
adata3 = adata3[:, wmb.var['gene_ID'].values]
sc.tl.ingest(adata3, wmb, embedding_method='pca')
# %% use knn to predict labels
from sklearn.neighbors import KNeighborsClassifier

# Get the PCA representation used for neighbors in wmb
wmb_rep = wmb.obsm['X_pca']

# Get query representation (already computed by ingest)
query_rep = adata3.obsm['X_pca']

# Train k-NN classifier on reference labels
k = wmb.uns['neighbors']['params']['n_neighbors']
knn = KNeighborsClassifier(n_neighbors=k)
# drop wmb obs with nan
knn.fit(wmb_rep[~wmb.obs['subclass'].isna()], wmb.obs.loc[~wmb.obs['subclass'].isna(), 'subclass'])

# Predict labels for query data
adata3.obs['subclass'] = knn.predict(query_rep)
adata3.obs['subclass'] = adata3.obs['subclass'].astype('category')
# %% draw label consensus heatmap
# create confusion matrix
confusion_matrix = pd.crosstab(
    adata3.obs['subclass_name'],
    adata3.obs['subclass'],
    normalize='index'
)
# reorder rows and columns
confusion_matrix = confusion_matrix.reindex(
    index=sorted(confusion_matrix.index),
    columns=sorted(confusion_matrix.columns)
)
# filter out cells with < 100 original cells
celltype_n = adata3.obs['subclass_name'].value_counts()
confusion_matrix = confusion_matrix.loc[celltype_n.index[celltype_n >= 100],
                                        celltype_n.index[celltype_n >= 100]]
# plot heatmap
fig, ax = plt.subplots(figsize=(25, 20))
sns.heatmap(confusion_matrix, annot=False, cmap='Reds', ax=ax)
ax.set_title('Label Transfer Confusion Matrix: STARRFISH vs WMB')
ax.set_xlabel('Predicted WMB Subclass')
ax.set_ylabel('Original STARRFISH Subclass')
fig.tight_layout()
fig.savefig(f'{PWD}/results/expr3/abc_atlas/WMB_vs_STARRFISH_label_transfer_confusion_heatmap.pdf')
# %% save adata3 with new labels
adata3.write_h5ad(f'{PWD}/Data/scdata_5_28_2025_BRBB500gn_final_CRE_T7CRE_integrate_wmb.h5ad')

# %% plot confusion matrix diag vs celltype_n
confusion_diag = confusion_matrix.values.diagonal()
fig, ax = plt.subplots(figsize=(6, 6))
sns.scatterplot(x=celltype_n[celltype_n >= 100], y=confusion_diag, ax=ax)
ax.set_title('Label Transfer Accuracy vs Original Cell Type Size')
ax.set_xlabel('Original Cell Type Size (Number of Cells)')
ax.set_ylabel('Label Transfer Accuracy (Diagonal of Confusion Matrix)')
ax.set_xscale('log')
fig.tight_layout()
fig.savefig(f'{PWD}/results/expr3/abc_atlas/WMB_vs_STARRFISH_label_transfer_accuracy_vs_size.pdf')

# %%
