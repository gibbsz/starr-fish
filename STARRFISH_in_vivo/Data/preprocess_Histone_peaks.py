# %%
import pandas as pd
import pyranges as pr
import re
import scanpy as sc
import os
# %%
# preprocess the chromatine state
file_list = os.listdir('Data/Histone_peaks/')
cluster_numbers = []
cell_types = []
modalities = []
for f in file_list:
    if f.endswith('.peak') and "H3K4me1" in f:
        ff = f.split('.')[0].replace('_', ' ')
        modalities.append(ff.split('-')[1])  # first part is modality
        ff = ff.split('-')[0]  # remove modality part
        cell_types.append(re.sub('^[0-9]+ ', '', ff))  # remove number prefix
        cluster_numbers.append(int(ff.split(' ')[0]))  # first part is cluster number
# %%
subclass_names = pd.read_csv('Data/abc_atlas/cluster_annotation_term.csv', index_col=0)
subclass_names['subclass'] = subclass_names['subclass'].str.replace('/', ' ')
subclass_names['subclass'] = subclass_names['subclass'].str.replace('-', ' ')
# check if all cell types are in subclass_names
sum(pd.Series(cell_types).isin(subclass_names['subclass'].unique()))  
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
adata = preprocess(f'./Data/scdata_03_14_BRBB500gn_withCRE_final.h5ad')
# %%
# Prepare CRE info
creinfo = adata.uns['CRE_info'].copy()
creinfo = creinfo[creinfo['labeling_type'] != 'negative control']
creinfo = creinfo[['Chrom', 'Start', 'End']].copy()
creinfo.columns = ['Chromosome', 'Start', 'End']
creinfo['Start'] = creinfo['Start'].astype(int)
creinfo['End'] = creinfo['End'].astype(int)
creinfo['CRE_ID'] = creinfo.index  # keep track of original rows
cre_gr = pr.PyRanges(creinfo)

# Output containers
target_df = pd.DataFrame(index=creinfo['CRE_ID'], columns=pd.Series(cell_types))
# %%
# Loop over cell types
for file, celltype, cluster_number, modality in zip(file_list, cell_types, cluster_numbers, modalities):
    peaks = pd.read_csv(
        f'Data/Histone_peaks/{file}', sep='\t| '
    )
    peaks = peaks[['chrom', 'startFrom', 'endTo']]  # only keep necessary columns
    peaks.columns = ['Chromosome', 'Start', 'End']
    peaks['Start'] = peaks['Start'].astype(int)
    peaks['End'] = peaks['End'].astype(int)
    subset_gr = pr.PyRanges(peaks)

    # Perform intersection
    intersection = cre_gr.join(subset_gr)
    if intersection.empty:
        continue
    # Compute overlap length
    overlap_df = intersection.df
    overlap_df['overlap'] = (
        overlap_df[['Start_b', 'Start']].max(axis=1) -
        overlap_df[['End_b', 'End']].min(axis=1)
    ) * -1  # negative because min(start) - max(end) < 0

    overlap_df = overlap_df[overlap_df['overlap'] > 0]

    # Normalize by CRE length
    overlap_df['cre_len'] = overlap_df['End'] - overlap_df['Start']
    overlap_df['overlap_ratio'] = overlap_df['overlap'] / overlap_df['cre_len']

    # Aggregate by CRE_ID: keep max overlap per CRE
    best_overlap = overlap_df.groupby('CRE_ID')['overlap_ratio'].max()

    # Fill results
    target_df[celltype] = best_overlap.reindex(creinfo['CRE_ID']).fillna(0).values
# %%
target_df = target_df.fillna(0)
target_df.T.to_csv('Data/cre_h3k4me1_peaks.csv')
# %%
