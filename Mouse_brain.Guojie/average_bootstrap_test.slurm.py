# %%
import warnings
warnings.filterwarnings("ignore", category=SyntaxWarning, module="docrep")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import sys
import os
import re
import pandas as pd
import scanpy as sc
import scvi
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# add current path to sys.path
try:
    PWD = os.path.dirname(os.path.abspath(__file__))
except NameError:
    PWD = '/gpfs/commons/groups/ren_lab/guojiezhong/starr-fish/Mouse_brain.Guojie'
sys.path.append(f'{PWD}/')
os.chdir(PWD)
from utils import STARRFISH

scvi.settings.seed = 0
print("Last run with scvi-tools version:", scvi.__version__)
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
# get subclass name and subclass transform
subclass_annotation = pd.read_excel(f'Data/abc_atlas/allen_institute_nominature.xlsx')
subclass_annotation['subclass'] = subclass_annotation['subclass_id_label'].str.replace('^[0-9]+ ', '', regex=True)
subclass_annotation['subclass'] = subclass_annotation['subclass'].str.replace('/', '-', regex=True)
subclass_to_subclass_name = subclass_annotation['subclass_id_label'].groupby(subclass_annotation['subclass']).first().to_dict()
subclass_name_to_subclass = subclass_annotation['subclass'].groupby(subclass_annotation['subclass_id_label']).first().to_dict()
cre_blacklist = ['CRE061', 'CRE143', 'CRE001']
cre_whitelist = starrfish3_sec1.get_creinfo().index[~starrfish3_sec1.get_creinfo().index.isin(cre_blacklist)]
# read in the mismatching barcode CREs
mismatching_cres = pd.read_csv('Data/AAV_ONT_Barcode_Counts_vs_Mismatch_Percentage.csv', index_col=0)
cre_whitelist = cre_whitelist[~cre_whitelist.isin(mismatching_cres.index[mismatching_cres['MismatchPercent'] > 20])]
cre_blacklist = np.unique(cre_blacklist + mismatching_cres.index[mismatching_cres['MismatchPercent'] > 20].tolist()).tolist()
# %%
average_bootstrap_test_config = {
    'cell_types_to_use': None,
    'normalize_by_cell_rna': False,
    'normalize_by_cell_volume': False,
    'normalize_by_cell_t7': False,  # normalize by T7, filter cells with T7 < 4
    'normalize_by_celltype_rna': False,
    'normalize_by_celltype_volume': False,
    'normalize_by_celltype_t7': True,  # normalize by T7
    'filter_by_cell_t7': None,
    'normalize_by_negative_control': False,  # normalize by negative control
    'normalize_by_libsize': False,
    'log_transform': False,
    'bootstrap_number': 10000,
    'bootstrap_to_fixed_pct': 1,
    'bootstrap_to_fixed_sample_size': None,
    'load_stored': True,
    'n_jobs': 62,
}
infected_cells_threshold = 5
threshold = 'neg_control_mean'
starrfish3_sec1 = STARRFISH.load('results/starrfish3_sec1.bak.pkl')
res1 = starrfish3_sec1.average_bootstrap_test(**average_bootstrap_test_config)
del starrfish3_sec1
starrfish3_sec1 = STARRFISH.load('results/starrfish3_sec1.pkl')
to_filter_sec1 = (starrfish3_sec1.get_cre_expression() > 0).groupby(starrfish3_sec1.get_celltypes()).sum() < infected_cells_threshold
to_filter_sec1[cre_blacklist] = True
res_q1, res_df1, _ = starrfish3_sec1.average_bootstrap_test_q(res1, threshold=threshold, norm='T7', tail='both', to_filter=to_filter_sec1, calibrate=None)

starrfish3_sec2 = STARRFISH.load('results/starrfish3_sec2.bak.pkl')
res2 = starrfish3_sec2.average_bootstrap_test(**average_bootstrap_test_config)
del starrfish3_sec2
starrfish3_sec2 = STARRFISH.load('results/starrfish3_sec2.pkl')
to_filter_sec2 = (starrfish3_sec2.get_cre_expression() > 0).groupby(starrfish3_sec2.get_celltypes()).sum() < infected_cells_threshold
to_filter_sec2[cre_blacklist] = True
res_q2, res_df2, _ = starrfish3_sec2.average_bootstrap_test_q(res2, threshold=threshold, norm='T7', tail='both', to_filter=to_filter_sec2, calibrate=None)

starrfish3 = STARRFISH.load('results/starrfish3.bak.pkl')
res = starrfish3.average_bootstrap_test(**average_bootstrap_test_config)
del starrfish3
starrfish3 = STARRFISH.load('results/starrfish3.pkl')
to_filter = (starrfish3.get_cre_expression() > 0).groupby(starrfish3.get_celltypes()).sum() < infected_cells_threshold
to_filter[cre_blacklist] = True
res_q, res_df, _ = starrfish3.average_bootstrap_test_q(res, threshold=threshold, norm='T7', tail='both', to_filter=to_filter, calibrate=None)


# %%
# regress on the total T7 counts ~ log fold change of each CRE/Cell Type pair
# fist visualize the result
toplot = pd.DataFrame({'value': res_df.values.flatten(), 
                       'T7 total': np.tile(starrfish3.get_t7_expression().sum().values, res_df.shape[0])})
fig, ax = plt.subplots(figsize=(4, 4))
sns.scatterplot(data=toplot, x='T7 total', y='value', alpha=0.1, ax=ax)
toplot_avg = res_df.apply(np.nanmean, axis=0)
sns.scatterplot(x=starrfish3.get_t7_expression().sum().values, y=toplot_avg.values, color='red', ax=ax, alpha=0.5)
ax.set_xscale('log')
ax.set_xlabel('Total T7 counts (per CRE)')
ax.set_ylabel('Activity per CRE/Cell Type pair')
# %% check cell type bias
toplot = pd.DataFrame({'value': res_df.values.flatten(), 
                       'T7 total': np.repeat(starrfish3.get_t7_expression().sum(axis=1).groupby(starrfish3.get_celltypes()).sum().values, res_df.shape[1])})
fig, ax = plt.subplots(figsize=(4, 4))
sns.scatterplot(data=toplot, x='T7 total', y='value', alpha=0.1, ax=ax)
toplot_avg = res_df.apply(np.nanmean, axis=1)
sns.scatterplot(x=starrfish3.get_t7_expression().sum(axis=1).groupby(starrfish3.get_celltypes()).sum().values, y=toplot_avg.values, color='red', ax=ax, alpha=0.5)
ax.set_xscale('log')
ax.set_xlabel('Total T7 counts (per Cell Type)')
ax.set_ylabel('Activity per CRE/Cell Type pair')



# %%
# heatmap after correction
from plots import plot_grouped_clustermap
cre_info = starrfish3.get_creinfo().copy()
cre_info['best_subclass'] = 'CRE'
cre_info.loc[starrfish3.get_negative_control_cres(), 'best_subclass'] = 'Negative Control'

_, final_order = plot_grouped_clustermap(res_df.loc[pd.isna(res_df).any(axis=1), pd.isna(res_df).any(axis=0)], cre_info, 'All', figsize=(15, 8))
# %% dot plot after correction
threshold = 'neg_control_mean'
to_filter = (starrfish3.get_cre_expression() > 0).groupby(starrfish3.get_celltypes()).sum() < 0
to_filter[cre_blacklist] = True
res_q, res_q_right, res_q_left, res_df, res_df_fdc = starrfish3.average_bootstrap_test_q(res, threshold=threshold, norm='T7', tail='all', 
                                                                                         to_filter=to_filter, calibrate='self-CRE')
# %%
to_filter_sec1 = (starrfish3_sec1.get_cre_expression() > 0).groupby(starrfish3_sec1.get_celltypes()).sum() < 0
to_filter_sec1[cre_blacklist] = True
res_q1, res_q1_right, res_q1_left, res_df1, res_df1_fdc = starrfish3.average_bootstrap_test_q(res1, threshold=threshold, norm='T7', tail='all',
                                                                                              to_filter=to_filter_sec1, calibrate='self-CRE')
# %%
to_filter_sec2 = (starrfish3_sec2.get_cre_expression() > 0).groupby(starrfish3_sec2.get_celltypes()).sum() < 0
to_filter_sec2[cre_blacklist] = True
res_q2, res_q2_right, res_q2_left, res_df2, res_df2_fdc = starrfish3.average_bootstrap_test_q(res2, threshold=threshold, norm='T7', tail='all',
                                                                                              to_filter=to_filter_sec2, calibrate='self-CRE')

# %% reproducibility
cre_corr, celltype_corr = starrfish3.corr_starrfish(res_df1, res_df2)
cre_corr['libsize'] = starrfish3.lib_size['counts'].loc[cre_corr.index]
celltype_corr['celltype_sec1'] = starrfish3_sec1.get_celltypes().value_counts().reindex(celltype_corr.index).fillna(0).values
celltype_corr['celltype_sec2'] = starrfish3_sec2.get_celltypes().value_counts().reindex(celltype_corr.index).fillna(0).values
celltype_corr['celltype_full'] = starrfish3.get_celltypes().value_counts().reindex(celltype_corr.index).fillna(0).values
celltype_corr['celltype_n'] = np.minimum(celltype_corr['celltype_sec1'], celltype_corr['celltype_sec2'])
# %% plot
n_cre_threshold = 20
n_celltype_threshold = 5
fig, ax = plt.subplots(figsize=(4, 4))
sns.scatterplot(data=celltype_corr[(celltype_corr['pearson_p'] > 0.05) & (celltype_corr['effect_n'] >= n_cre_threshold)], x='celltype_n', y='pearson', color='blue', ax=ax)
sns.scatterplot(data=celltype_corr[(celltype_corr['pearson_p'] <= 0.05) & (celltype_corr['effect_n'] >= n_cre_threshold)], x='celltype_n', y='pearson', color='red', ax=ax)
ax.set_xscale('log')
fig.savefig('results/expr3/reproducibility_by_celltype_pearson_sec1_sec2.pdf')
fig, ax = plt.subplots(figsize=(4, 4))
sns.scatterplot(data=cre_corr[(cre_corr['pearson_p'] > 0.05) & (cre_corr['effect_n'] >= n_celltype_threshold)], x='libsize', y='pearson', color='blue', ax=ax)
sns.scatterplot(data=cre_corr[(cre_corr['pearson_p'] <= 0.05) & (cre_corr['effect_n'] >= n_celltype_threshold)], x='libsize', y='pearson', color='red', ax=ax)
reproducible_celltypes = celltype_corr.index[(celltype_corr['pearson_p'] <= 0.05) & (celltype_corr['effect_n'] >= n_cre_threshold)]
ax.set_xscale('log')
# %% plot pearson R in violin
fig, ax = plt.subplots(figsize=(2, 4))
sns.violinplot(data=celltype_corr[(celltype_corr['effect_n'] >= n_cre_threshold) & (celltype_corr['celltype_n'] >= 1000)], y='pearson', ax=ax)
reproducible_celltypes = celltype_corr.index[(celltype_corr['effect_n'] >= n_cre_threshold) & (celltype_corr['celltype_n'] >= 1000)]
fig.savefig('results/expr3/reproducibility_by_celltype_pearson_violin_sec1_sec2.pdf')

# %% plot by CRE
n_cre_threshold = 10
n_celltype_threshold = 10
fig, ax = plt.subplots(figsize=(4, 4))
sns.scatterplot(data=cre_corr[(cre_corr['pearson_p'] > 0.05) & (cre_corr['effect_n'] >= n_celltype_threshold)], x='libsize', y='pearson', color='blue', ax=ax)
sns.scatterplot(data=cre_corr[(cre_corr['pearson_p'] <= 0.05) & (cre_corr['effect_n'] >= n_celltype_threshold)], x='libsize', y='pearson', color='red', ax=ax)
ax.set_xscale('log')
fig.savefig('results/expr3/reproducibility_by_cre_pearson_sec1_sec2.pdf')
# %% plot pearson R in violin
fig, ax = plt.subplots(figsize=(2, 4))
sns.violinplot(data=cre_corr[(cre_corr['effect_n'] >= n_celltype_threshold)], y='pearson', ax=ax)
fig.savefig('results/expr3/reproducibility_by_cre_pearson_violin_sec1_sec2.pdf')



# %% check overlap of significant CREs between sections
res_q2_overlap = res_q2_right[(~res_q2_right.isna()) & (~res_q1_right.isna()) & (~res_q_right.isna())].copy()
res_q1_overlap = res_q1_right[(~res_q2_right.isna()) & (~res_q1_right.isna()) & (~res_q_right.isna())].copy()
res_q_overlap = res_q_right[(~res_q2_right.isna()) & (~res_q1_right.isna()) & (~res_q_right.isna())].copy()
overlap_df = pd.DataFrame(index=res_q2_overlap.index.intersection(res_q1_overlap.index),
                          columns=['sec1', 'sec2', 'overlap'])
overlap_df['sec1'] = (res_q1_overlap.loc[overlap_df.index] <= 0.05).sum(axis=1)
overlap_df['sec2'] = (res_q2_overlap.loc[overlap_df.index] <= 0.05).sum(axis=1)
overlap_df['all'] = (res_q_overlap.loc[overlap_df.index] <= 0.05).sum(axis=1)
overlap_df['overlap_sec1_sec2'] = ((res_q1_overlap.loc[overlap_df.index] <= 0.05) & (res_q2_overlap.loc[overlap_df.index] <= 0.05)).sum(axis=1)
overlap_df['overlap_sec1_all'] = ((res_q1_overlap.loc[overlap_df.index] <= 0.05) & (res_q_overlap.loc[overlap_df.index] <= 0.05)).sum(axis=1) 
overlap_df['overlap_sec2_all'] = ((res_q2_overlap.loc[overlap_df.index] <= 0.05) & (res_q_overlap.loc[overlap_df.index] <= 0.05)).sum(axis=1)
overlap_df['percentage_sec1_sec2'] = overlap_df['overlap_sec1_sec2'] / np.minimum(overlap_df['sec1'], overlap_df['sec2'])
overlap_df['percentage_sec1_all'] = overlap_df['overlap_sec1_all'] / np.minimum(overlap_df['sec1'], overlap_df['all'])
overlap_df['percentage_sec2_all'] = overlap_df['overlap_sec2_all'] / np.minimum(overlap_df['sec2'], overlap_df['all'])
overlap_df = overlap_df.sort_values('percentage_sec1_sec2', ascending=False)
overlap_df['celltype_n_sec1'] = starrfish3_sec1.get_celltypes().value_counts().reindex(overlap_df.index).fillna(0).astype(int).values
overlap_df['celltype_n_sec2'] = starrfish3_sec2.get_celltypes().value_counts().reindex(overlap_df.index).fillna(0).astype(int).values
overlap_df['celltype_n_all'] = starrfish3.get_celltypes().value_counts().reindex(overlap_df.index).fillna(0).astype(int).values
overlap_df['celltype_n'] = np.minimum(overlap_df['celltype_n_sec1'], overlap_df['celltype_n_sec2'], overlap_df['celltype_n_all'])
# %%
def plot_reproducibility(overlap_df, celltypes_to_use, percentage_col, bar1_col, bar2_col, bar1_label, bar2_label):
    # get cell type orders
    # order by allen institute's nominature
    cluster_annotation_term = pd.read_csv('Data/abc_atlas/cluster_annotation_term.csv', index_col=0)
    cluster_annotation_term['subclass'] = cluster_annotation_term['subclass'].str.replace('/', '-')
    overlap_df['cell_type_rank'] = cluster_annotation_term['subclass_number'].groupby(cluster_annotation_term['subclass']).first().loc[overlap_df.index].values
    overlap_df = overlap_df.sort_values(['cell_type_rank'], ascending=[True])
    # Filter for cell types with celltype_n >= 1000
    overlap_df_filtered = overlap_df[overlap_df['celltype_n'] >= 1000]
    overlap_df_filtered = overlap_df_filtered.loc[celltypes_to_use.intersection(overlap_df_filtered.index)]
    overlap_df_filtered = overlap_df_filtered.sort_values(['cell_type_rank'], ascending=[True])
    # Create x positions
    x = np.arange(len(overlap_df_filtered))
    cell_types = overlap_df_filtered.index

    fig, ax1 = plt.subplots(figsize=(14, 6))

    # Left y-axis for percentage
    sns.barplot(x=cell_types, y=overlap_df_filtered[percentage_col], ax=ax1, alpha=0.8)
    ax1.set_xlabel('Cell Types')
    ax1.set_ylabel('Reproducibility (percentage)', color='black')
    ax1.tick_params(axis='y', labelcolor='black')
    plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')

    # Right y-axis for bar values
    ax2 = ax1.twinx()
    x_pos = np.arange(len(overlap_df_filtered))

    # Set colors based on column names
    color1 = 'orange' if bar1_col == 'sec1' else 'green' if bar1_col == 'sec2' else 'pink'
    color2 = 'orange' if bar2_col == 'sec1' else 'green' if bar2_col == 'sec2' else 'pink'

    ax2.bar(x_pos - 0.2, overlap_df_filtered[bar1_col], 0.4, label=bar1_label, alpha=0.8, color=color1)
    ax2.bar(x_pos + 0.2, overlap_df_filtered[bar2_col], 0.4, label=bar2_label, alpha=0.8, color=color2)
    ax2.set_ylabel('# significant CREs', color='black')
    ax2.tick_params(axis='y', labelcolor='black')
    ax2.legend(loc='upper right')

    plt.tight_layout()
    return fig

# Plot sec1-sec2
fig = plot_reproducibility(overlap_df, reproducible_celltypes, 'percentage_sec1_sec2', 'sec1', 'sec2', 'sec1', 'sec2')
fig.savefig('results/expr3/reproducibility_by_celltype_sec1_sec2.pdf')
# Plot sec1-all
fig = plot_reproducibility(overlap_df, reproducible_celltypes, 'percentage_sec1_all', 'sec1', 'all', 'sec1', 'all')
fig.savefig('results/expr3/reproducibility_by_celltype_sec1_all.pdf')

# Plot sec2-all
fig = plot_reproducibility(overlap_df, reproducible_celltypes, 'percentage_sec2_all', 'sec2', 'all', 'sec2', 'all')
fig.savefig('results/expr3/reproducibility_by_celltype_sec2_all.pdf')

# %% calculate overlap statistics for each CRE
def calculate_overlap_cre_df(q_values_df1, q_values_df2, comparison_name1='group1', comparison_name2='group2', cre_blacklist=None):
    """
    Calculate overlap statistics between two q-value dataframes.

    Parameters:
    q_values_df1: DataFrame with q-values for first comparison
    q_values_df2: DataFrame with q-values for second comparison
    comparison_name1: Name for first comparison (used in column names)
    comparison_name2: Name for second comparison (used in column names)
    cre_blacklist: List of CREs to mark as blacklisted

    Returns:
    DataFrame with overlap statistics and reproducibility categories
    """
    if cre_blacklist is None:
        cre_blacklist = []

    overlap_cre_df = pd.DataFrame(index=q_values_df1.columns,
                                 columns=['no_na', comparison_name1, comparison_name2, 'overlap', 'percentage'])

    for cre in q_values_df1.columns:
        if comparison_name2 == comparison_name1:
            overlap_cre_df.loc[cre, 'no_na'] = sum(~pd.isna(q_values_df1[cre]))
        else:
            overlap_cre_df.loc[cre, 'no_na'] = sum(~pd.isna(q_values_df1[cre]) & ~pd.isna(q_values_df2[cre]))

        overlap_cre_df.loc[cre, comparison_name1] = sum(q_values_df1[cre] <= 0.05)
        overlap_cre_df.loc[cre, comparison_name2] = sum(q_values_df2[cre] <= 0.05)
        overlap_cre_df.loc[cre, 'overlap'] = sum((q_values_df1[cre] <= 0.05) & (q_values_df2[cre] <= 0.05))

        if overlap_cre_df.loc[cre, comparison_name1] == 0 and overlap_cre_df.loc[cre, comparison_name2] == 0:
            if overlap_cre_df.loc[cre, 'no_na'] > 0:
                overlap_cre_df.loc[cre, 'percentage'] = 1
            else:
                overlap_cre_df.loc[cre, 'percentage'] = -1
        else:
            overlap_cre_df.loc[cre, 'percentage'] = overlap_cre_df.loc[cre, 'overlap'] / np.maximum(
                overlap_cre_df.loc[cre, comparison_name1], overlap_cre_df.loc[cre, comparison_name2])

    overlap_cre_df['reproducibility'] = 'Non-reproducible'
    overlap_cre_df.loc[overlap_cre_df['percentage'] == 1, 'reproducibility'] = 'All Reproducible'
    overlap_cre_df.loc[overlap_cre_df['percentage'] == -1, 'reproducibility'] = 'All NA'
    overlap_cre_df.loc[(overlap_cre_df['percentage'] > 0) & (overlap_cre_df['percentage'] < 1), 'reproducibility'] = 'Partially Reproducible'
    overlap_cre_df.loc[cre_blacklist, 'reproducibility'] = 'Blacklisted'

    return overlap_cre_df


# %% get some numbers of reproducibility in each CRE
res_q1_overlap = res_q1_overlap.loc[overlap_df.index]
res_q2_overlap = res_q2_overlap.loc[overlap_df.index]
res_q_overlap = res_q_overlap.loc[overlap_df.index]
overlap_cre_df = calculate_overlap_cre_df(res_q1_overlap, res_q2_overlap, 'sec1', 'sec2', cre_blacklist)
sum(overlap_cre_df['percentage'] > 0), sum(overlap_cre_df['percentage'] == -1), sum(overlap_cre_df['percentage'] == 1)


# %% do this for sec1 to all and sec2 to all, separately
overlap_cre_df_sec1_all = calculate_overlap_cre_df(res_q1_overlap, res_q_overlap, 'sec1', 'all', cre_blacklist)
sum(overlap_cre_df_sec1_all['percentage'] > 0), sum(overlap_cre_df_sec1_all['percentage'] == -1), sum(overlap_cre_df_sec1_all['percentage'] == 1)

# %% sec2
overlap_cre_df_sec2_all = calculate_overlap_cre_df(res_q2_overlap, res_q_overlap, 'sec2', 'all', cre_blacklist)
sum(overlap_cre_df_sec2_all['percentage'] > 0), sum(overlap_cre_df_sec2_all['percentage'] == -1), sum(overlap_cre_df_sec2_all['percentage'] == 1)


# %% plot the percentage
toplot = pd.concat((overlap_cre_df['reproducibility'].value_counts().rename('sec1_sec2'),
                    overlap_cre_df_sec1_all['reproducibility'].value_counts().rename('sec1_all'),
                    overlap_cre_df_sec2_all['reproducibility'].value_counts().rename('sec2_all')), axis=1)
# merge All NA into All Reproducible
toplot.loc['All Reproducible'] = toplot.loc['All Reproducible'].fillna(0) + toplot.loc['All NA'].fillna(0)
toplot = toplot.drop(index='All NA', errors='ignore')
# Reshape data for plotting
toplot_reset = toplot.reset_index()
index_col_name = toplot_reset.columns[0]  # Get the actual name of the index column
toplot_melted = toplot_reset.melt(id_vars=index_col_name, var_name='comparison', value_name='count')
toplot_melted = toplot_melted.rename(columns={index_col_name: 'reproducibility'})

fig, ax = plt.subplots(figsize=(10, 4))
sns.barplot(data=toplot_melted[toplot_melted['reproducibility'] != 'Non-reproducible'], x='comparison', y='count', hue='reproducibility', ax=ax)
ax.set_title('CRE Reproducibility Across Comparisons')
ax.set_xlabel('Comparison')
ax.set_ylabel('Count')
ax.legend(title='Reproducibility Category', bbox_to_anchor=(1.05, 1), loc='upper left')
fig.tight_layout()
fig.savefig('results/expr3/reproducibility_comparison_barplot.pdf')
fig.show()


# %%
from plots import celltype_pval_dotplot
# %%
cell_types_to_use = res_q1_right.index.intersection(res_q2_right.index)
cres_to_use = res_q_right.columns[np.nanmin(res_q_right.loc[cell_types_to_use], axis=0) < 0.05].union(starrfish3.get_negative_control_cres())
cres_to_use = cres_to_use[~cres_to_use.isin(cre_blacklist)]
fig, final_order = celltype_pval_dotplot(res_q_right, res_df_fdc, cres_to_use, cell_types_to_use,
                                         positive_control_info=cre_info, significant_cutoff=0.05, z_norm=False,
                                         figsize=(50, 30))
fig.savefig('results/expr3/celltype_pval_dotplot_complete.pdf')
# %%
cell_types_to_use = starrfish3.get_celltypes().value_counts().index[starrfish3.get_celltypes().value_counts()>=1000]
cres_to_use = res_q_right.columns[np.nanmin(res_q_right.loc[cell_types_to_use], axis=0) < 0.05].union(starrfish3.get_negative_control_cres())
cres_to_use = cres_to_use[~cres_to_use.isin(cre_blacklist)]
fig, final_order = celltype_pval_dotplot(res_q_right, res_df_fdc, cres_to_use, cell_types_to_use,
                                         positive_control_info=cre_info, significant_cutoff=0.05, z_norm=False,
                                         figsize=(20, 30))
fig.savefig('results/expr3/celltype_pval_dotplot_all.pdf')
# %%
fig, final_order = celltype_pval_dotplot(res_q1_right, res_df1_fdc, pd.Index(final_order), cell_types_to_use, reorder_cres=False,
                                         positive_control_info=cre_info, significant_cutoff=0.05, z_norm=False,
                                         figsize=(20, 30))
fig.savefig('results/expr3/celltype_pval_dotplot_sec1.pdf')
# %%
fig, final_order = celltype_pval_dotplot(res_q2_right, res_df2_fdc, pd.Index(final_order), cell_types_to_use, reorder_cres=False,
                                         positive_control_info=cre_info, significant_cutoff=0.05, z_norm=False,
                                         figsize=(20, 30))
fig.savefig('results/expr3/celltype_pval_dotplot_sec2.pdf')



# %% get calibration matrix
res_array = res['celltype_activity_array'].copy()
res_array = np.log1p(res_array)
# assign inf to NaN
res_array[np.isinf(res_array)] = np.nan
# if we have to_filter, then fill it with np.nan
for cell_type in to_filter.index:
    res_array[:, res['celltype_activity'].index == cell_type, to_filter.loc[cell_type]] = np.nan
# self calibrate based on average of activity across all cell types, all bootstraps
cre_mean = np.nanmean(res_array, axis=(0, 1))
cre_celltype_mean = np.nanmean(res_array, axis=0)
# %% plot some examples
for celltype in reproducible_celltypes:
    # rank by activity
    cre_activity = res_df_fdc.loc[celltype]
    # find significant right tail and left tails
    cre_q_values_right = res_q_right.loc[celltype]
    cre_q_values_left = res_q_left.loc[celltype]
    cre_q_values_right = cre_q_values_right[cre_q_values_right <= 0.05]
    cre_q_values_left = cre_q_values_left[cre_q_values_left <= 0.05]
    # order by rank
    cre_right = cre_activity.loc[cre_q_values_right.index].sort_values(ascending=False).index
    cre_left = cre_activity.loc[cre_q_values_left.index].sort_values(ascending=True).index
    # get nmax
    nmax = starrfish3.get_cre_expression().loc[starrfish3.get_celltypes() == celltype, cre_right[0]].max()
    t7_nmax = starrfish3.get_t7_expression().loc[starrfish3.get_celltypes() == celltype, cre_right[0]].mean()
    nmax = np.log1p(nmax / t7_nmax) - cre_mean[res_df_fdc.columns==cre_right[0]][0]
    # round up nmax to nearest integer
    nmax = int(np.ceil(nmax))
    # only plot top 1 of each tail
    if len(cre_right) > 0:
        cre = cre_right[0]
        print(f'Plotting {cre} in {celltype} (right tail)')
        cell_types_to_visualize = [celltype]
        fig = starrfish3.plot_gene(
            cre, average_by_celltype=False, # if true, all cells from same cell type will have same value
            cell_types_to_visualize=cell_types_to_visualize, # only visualize some cell types
            scale_size_by='counts', # scale size by "counts": normalized counts; or "celltype_number": number of cells in the cell type
            log=True, calibrate=cre_mean[res_df_fdc.columns==cre][0], nmax=nmax,
            transpose=-1, flipx=-1, sz_max=50,
            cell_types_to_use=cell_types_to_visualize)
        fig.savefig(f'results/expr3/celltype_significant_cres/{celltype}_{cre}_right_tail.pdf')
    if len(cre_left) > 0:
        cre = cre_left[0]
        print(f'Plotting {cre} in {celltype} (left tail)')
        cell_types_to_visualize = [celltype]
        fig = starrfish3.plot_gene(
            cre, average_by_celltype=False, # if true, all cells from same cell type will have same value
            cell_types_to_visualize=cell_types_to_visualize, # only visualize some cell types
            scale_size_by='counts', # scale size by "counts": normalized counts; or "celltype_number": number of cells in the cell type
            log=True, calibrate=cre_mean[res_df_fdc.columns==cre][0], nmax=nmax,
            transpose=-1, flipx=-1, sz_max=50,
            cell_types_to_use=cell_types_to_visualize)
        fig.savefig(f'results/expr3/celltype_significant_cres/{celltype}_{cre}_left_tail.pdf')





# %% show consistency with ATAC
def get_precision_df(res_q_df, starrfish, use='atac-peak'):
    precision_df = pd.DataFrame(index=res_q_df.index, columns=['TP', 'Total', 'ATAC', 'precision', 'recall'])
    for celltype in precision_df.index:
        atac_peaks = starrfish.get_positive_control_cres(cell_type=celltype, use=use)
        sig_cres = res_q_df.columns[res_q_df.loc[celltype] <= 0.05]
        nan_cres = res_q_df.columns[pd.isna(res_q_df.loc[celltype])]
        atac_peaks = atac_peaks[~atac_peaks.isin(nan_cres)] if atac_peaks is not None else None
        if atac_peaks is not None and len(sig_cres) > 0:
            precision_df.loc[celltype, 'TP'] = sig_cres.isin(atac_peaks).sum()
        precision_df.loc[celltype, 'Total'] = sum(~pd.isna(res_q_df.loc[celltype]))
        precision_df.loc[celltype, use] = len(atac_peaks) if atac_peaks is not None else 0
        if precision_df.loc[celltype, 'Total'] != 0:
            precision_df.loc[celltype, 'recall'] = precision_df.loc[celltype, 'TP'] / precision_df.loc[celltype, 'Total']
        if precision_df.loc[celltype, use] != 0:
            precision_df.loc[celltype, 'precision'] = precision_df.loc[celltype, 'TP'] / precision_df.loc[celltype, use]
    precision_df = precision_df.sort_values('precision', ascending=False)
    return precision_df

# Plot ATAC precision using the same strategy as plot_reproducibility
def plot_atac_precision(atac_precision_df, celltypes_to_use, use='atac-peak'):
    # get cell type orders
    # order by allen institute's nominature
    cluster_annotation_term = pd.read_csv('Data/abc_atlas/cluster_annotation_term.csv', index_col=0)
    cluster_annotation_term['subclass'] = cluster_annotation_term['subclass'].str.replace('/', '-')
    atac_precision_df['cell_type_rank'] = cluster_annotation_term['subclass_number'].groupby(cluster_annotation_term['subclass']).first().loc[atac_precision_df.index].values
    atac_precision_df = atac_precision_df.sort_values(['cell_type_rank'], ascending=[True])
    # Filter for cell types with celltype_n >= 1000
    atac_precision_df_filtered = atac_precision_df.loc[celltypes_to_use.intersection(atac_precision_df.index)]
    atac_precision_df_filtered = atac_precision_df_filtered.sort_values(['cell_type_rank'], ascending=[True])
    # Create x positions
    x = np.arange(len(atac_precision_df_filtered))
    cell_types = atac_precision_df_filtered.index

    fig, ax1 = plt.subplots(figsize=(14, 6))

    # Left y-axis for precision
    sns.barplot(x=cell_types, y=atac_precision_df_filtered['precision'], ax=ax1, alpha=0.8)
    ax1.set_xlabel('Cell Types')
    ax1.set_ylabel('ATAC Precision', color='black')
    ax1.tick_params(axis='y', labelcolor='black')
    plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')

    # Add horizontal dashed line for overall precision
    overall_precision = atac_precision_df_filtered['TP'].sum() / atac_precision_df_filtered[use].sum()
    ax1.axhline(y=overall_precision, color='black', linestyle='--', alpha=0.7, label=f'Overall precision: {overall_precision:.3f}')
    ax1.legend(loc='upper left')

    # Right y-axis for bar values
    ax2 = ax1.twinx()
    x_pos = np.arange(len(atac_precision_df_filtered))

    ax2.bar(x_pos - 0.2, atac_precision_df_filtered['TP'], 0.4, label=f'Significant CREs in {use}', alpha=0.8, color='orange')
    ax2.bar(x_pos + 0.2, atac_precision_df_filtered[use], 0.4, label=f'Total {use}', alpha=0.8, color='green')
    ax2.set_ylabel('Count', color='black')
    ax2.tick_params(axis='y', labelcolor='black')
    ax2.legend(loc='upper right')

    plt.tight_layout()
    return fig

# Plot ATAC precision
overall_tp = {}
overall_total = {}
for use in ['atac-peak', 'h3k27ac-peak', 'h3k4me1-peak', 'chromatin-a']:
    precision_df = get_precision_df(res_q_right, starrfish3, use=use)
    fig = plot_atac_precision(precision_df, cell_types_to_use, use=use)
    fig.savefig(f'results/expr3/{use.replace("-", "_")}_precision_by_celltype.pdf')
    overall_tp[use] = precision_df.loc[cell_types_to_use, 'TP'].sum()
    overall_total[use] = precision_df.loc[cell_types_to_use, use].sum()
# %% plot bar of overall precision
cre_precision_data = pd.DataFrame({
    'Assay': list(overall_tp.keys()),
    'TP': list(overall_tp.values()),
    'Total': list(overall_total.values()),
    'Precision': [tp / total if total > 0 else 0 for tp, total in zip(overall_tp.values(), overall_total.values())]
})
fig, ax = plt.subplots(figsize=(6, 4))
sns.barplot(data=cre_precision_data, x='Assay', y='Precision', ax=ax, alpha=0.8)
ax.set_xlabel('Assay')
ax.set_ylabel('Overall Precision')
ax.set_ylim(0, 0.3)
for i, row in cre_precision_data.iterrows():
    ax.text(i, row['Precision'] + 0.02, f"{row['TP']}/{row['Total']}", ha='center', va='bottom')
fig.savefig('results/expr3/overall_precision_barplot.pdf')
fig.show()

# %% make reproducible q value df
celltypes_overlap = res_q1_right.index.intersection(res_q2_right.index)
res_q_reproducible = (
    (res_q1_right.loc[celltypes_overlap] <= 0.05) & (res_q2_right.loc[celltypes_overlap] <= 0.05)
)
res_q_right_reproducible = res_q_right.loc[celltypes_overlap].copy()
res_q_right_reproducible[~res_q_reproducible] = np.nan
res_df_fdc_reproducible = res_df_fdc.loc[celltypes_overlap].copy()
res_df_fdc_reproducible[res_q_right_reproducible.isna()] = np.nan
# do for sec1
res_q1_right_reproducible = res_q1_right.loc[celltypes_overlap].copy()
res_q1_right_reproducible[~res_q_reproducible] = np.nan
res_df1_fdc_reproducible = res_df1_fdc.loc[celltypes_overlap].copy()
res_df1_fdc_reproducible[res_q1_right_reproducible.isna()] = np.nan
# do for sec2
res_q2_right_reproducible = res_q2_right.loc[celltypes_overlap].copy()
res_q2_right_reproducible[~res_q_reproducible] = np.nan
res_df2_fdc_reproducible = res_df2_fdc.loc[celltypes_overlap].copy()
res_df2_fdc_reproducible[res_q2_right_reproducible.isna()] = np.nan
# %% plot dotplot for reproducible CRE-Celltype pairs
from plots import celltype_pval_dotplot
cell_types_to_use = res_q_right_reproducible.index[np.nanmin(res_q_right_reproducible, axis=1) < 0.05]
cres_to_use = res_q_right_reproducible.columns[np.nanmin(res_q_right_reproducible.loc[cell_types_to_use], axis=0) < 0.05].union(starrfish3.get_negative_control_cres())
cres_to_use = cres_to_use[~cres_to_use.isin(cre_blacklist)]
fig, final_order = celltype_pval_dotplot(res_q_right_reproducible, res_df_fdc_reproducible, cres_to_use, cell_types_to_use,
                                         positive_control_info=cre_info, significant_cutoff=0.05, z_norm=False,
                                         figsize=(12, 20))
fig.savefig('results/expr3/celltype_pval_dotplot_reproducible_CRE_CellType_pair.pdf')
# %%
fig, final_order = celltype_pval_dotplot(res_q1_right_reproducible, res_df1_fdc_reproducible, pd.Index(final_order), cell_types_to_use, reorder_cres=False,
                                         positive_control_info=cre_info, significant_cutoff=0.05, z_norm=False,
                                         figsize=(12, 20))
fig.savefig('results/expr3/celltype_pval_dotplot_sec1_reproducible_CRE_CellType_pair.pdf')
# %%
fig, final_order = celltype_pval_dotplot(res_q2_right_reproducible, res_df2_fdc_reproducible, pd.Index(final_order), cell_types_to_use, reorder_cres=False,
                                         positive_control_info=cre_info, significant_cutoff=0.05, z_norm=False,
                                         figsize=(12, 20))
fig.savefig('results/expr3/celltype_pval_dotplot_sec2_reproducible_CRE_CellType_pair.pdf')


# %% number of on-target CREs
def get_cre_precision_df(res_q_df, starrfish, celltypes_to_use=None, use='atac-peak'):
    precision_df = pd.DataFrame(index=res_q_df.columns, columns=['TP', 'Total', 'precision', 'recall'])
    for cre in precision_df.index:
        atac_peaks = starrfish.get_positive_control_celltypes(cre=cre, use=use)
        sig_celltypes = res_q_df.index[res_q_df[cre] <= 0.05]
        nan_celltypes = res_q_df.index[pd.isna(res_q_df[cre])]
        if celltypes_to_use is not None:
            sig_celltypes = sig_celltypes[sig_celltypes.isin(celltypes_to_use)]
            nan_celltypes = nan_celltypes[nan_celltypes.isin(celltypes_to_use)]
        atac_peaks = atac_peaks[~atac_peaks.isin(nan_celltypes)] if atac_peaks is not None else None
        if atac_peaks is not None and len(sig_celltypes) > 0:
            precision_df.loc[cre, 'TP'] = sig_celltypes.isin(atac_peaks).sum()
        precision_df.loc[cre, 'Total'] = sum(~pd.isna(res_q_df[cre]))
        precision_df.loc[cre, use] = len(atac_peaks) if atac_peaks is not None else 0
        if precision_df.loc[cre, 'Total'] != 0:
            precision_df.loc[cre, 'recall'] = precision_df.loc[cre, 'TP'] / precision_df.loc[cre, 'Total']
        if precision_df.loc[cre, use] != 0:
            precision_df.loc[cre, 'precision'] = precision_df.loc[cre, 'TP'] / precision_df.loc[cre, use]
        # add the atac-peaks cell types
        precision_df.loc[cre, 'atac_peaks_celltypes'] = ', '.join(atac_peaks) if atac_peaks is not None else ''
    precision_df = precision_df.sort_values('precision', ascending=False)
    return precision_df
reproducible_cres = overlap_cre_df.index[overlap_cre_df['reproducibility'].isin(['All Reproducible', 'Partially Reproducible'])]
cre_precision_data_all = []
for use in ['atac-peak', 'h3k27ac-peak', 'h3k4me1-peak', 'chromatin-a']:
    for y in ['precision', 'percentage']:
        cre_precision_df_repro = get_cre_precision_df(res_q_right[reproducible_cres].copy(), starrfish3, use=use)
        repro_percentage = sum(cre_precision_df_repro['TP'] > 0) / sum(~cre_precision_df_repro['TP'].isna())
        repro_precision = sum(cre_precision_df_repro['TP'] > 0) / sum(cre_precision_df_repro[use] > 0)
        cre_precision_df_repro.to_csv(f'results/expr3/cre_{y}_reproducible_{use.replace("-", "_")}_df.csv')
        
        cre_precision_df_all = get_cre_precision_df(res_q_right.copy(), starrfish3, use=use)
        all_percentage = sum(cre_precision_df_all['TP'] > 0) / sum(~cre_precision_df_all['TP'].isna())
        all_precision = sum(cre_precision_df_all['TP'] > 0) / sum(cre_precision_df_all[use] > 0)
        cre_precision_df_all.to_csv(f'results/expr3/cre_{y}_all_{use.replace("-", "_")}_df.csv')

        # Plot CRE precision as two separate side-by-side bar plots
        repro_tp_count = sum(cre_precision_df_repro['TP'] > 0)
        repro_total_count = sum(~cre_precision_df_repro['TP'].isna())
        all_tp_count = sum(cre_precision_df_all['TP'] > 0)
        all_total_count = sum(~cre_precision_df_all['TP'].isna())
        print(f'{use} Reproducible CREs percentage: {repro_tp_count} out of {repro_total_count}')
        print(f'{use} Reproducible CREs precision: {repro_tp_count} out of {sum(cre_precision_df_repro[use] > 0)}')
        print(f'{use} All CREs percentage: {all_tp_count} out of {all_total_count}')
        print(f'{use} All CREs precision: {all_tp_count} out of {sum(cre_precision_df_all[use] > 0)}')

        cre_precision_data = pd.DataFrame({
            'CRE_type': ['Sec1-Sec2\nReproducible CREs', 'All CREs'],
            'percentage': [repro_percentage, all_percentage],
            'precision': [repro_precision, all_precision],
            'TP_count': [repro_tp_count, all_tp_count],
            'Total_count': [sum(cre_precision_df_repro[use] > 0), sum(cre_precision_df_all[use] > 0)],
            'use': [use, use],
            'y': [y, y]
        })
        cre_precision_data_all.append(cre_precision_data)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 4))

        # Left plot: Percentage
        sns.barplot(data=cre_precision_data, x='CRE_type', y=y, ax=ax1, alpha=0.8)
        ax1.set_xlabel('CRE Categories')
        ax1.set_ylabel('Percentage')
        ax1.set_ylim(0, 1)
        if y == 'precision':
            ax1.set_title('Precision of CREs on-target')
            # Add precision counts as text
            ax1.text(0, cre_precision_data.loc[0, y] + 0.02,
                    f"{repro_tp_count}/{sum(cre_precision_df_repro[use] > 0)}",
                    ha='center', va='bottom', fontsize=9)
            ax1.text(1, cre_precision_data.loc[1, y] + 0.02,
                    f"{all_tp_count}/{sum(cre_precision_df_all[use] > 0)}",
                    ha='center', va='bottom', fontsize=9)
        else:
            ax1.set_title('Percentage of CREs on-target')
            # Add percentage counts as text
            ax1.text(0, cre_precision_data.loc[0, y] + 0.02,
                    f"{repro_tp_count}/{repro_total_count}",
                    ha='center', va='bottom', fontsize=9)
            ax1.text(1, cre_precision_data.loc[1, y] + 0.02,
                    f"{all_tp_count}/{all_total_count}",
                    ha='center', va='bottom', fontsize=9)

        # Right plot: Count
        sns.barplot(data=cre_precision_data, x='CRE_type', y='TP_count', ax=ax2, alpha=0.8, color='orange')
        ax2.set_xlabel('CRE Categories')
        ax2.set_ylabel('Number')
        ax2.set_title('Count of CREs on-target')
        # Add count values as text on bars
        for i, row in cre_precision_data.iterrows():
            ax2.text(i, row['TP_count'] + max(cre_precision_data['TP_count']) * 0.01,
                    f"{int(row['TP_count'])}",
                    ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        fig.savefig(f'results/expr3/cre_{y}_barplot_{use.replace("-", "_")}.pdf')
        plt.show()

# %%
cre_precision_data = pd.concat(cre_precision_data_all, axis=0, ignore_index=True)
cre_precision_data = cre_precision_data[cre_precision_data['use'].isin(['atac-peak', 'chromatin-a'])]
cre_precision_data = cre_precision_data.drop_duplicates()
cre_precision_data = cre_precision_data[cre_precision_data['CRE_type'] == 'Sec1-Sec2\nReproducible CREs']
cre_precision_data = cre_precision_data[cre_precision_data['y'] == 'precision']
fig, ax = plt.subplots(figsize=(3, 4))
sns.barplot(data=cre_precision_data, x='use', y='precision', ax=ax, alpha=0.8)
ax.set_xlabel('Assay')
ax.set_ylabel('Precision')
ax.set_ylim(0, max(cre_precision_data['precision']) + 0.1)
for idx, (i, row) in enumerate(cre_precision_data.iterrows()):
    ax.text(idx, row['precision'] + 0.01, f"{int(row['TP_count'])}/{int(row['Total_count'])}",
            ha='center', va='bottom', fontsize=9)
fig.savefig('results/expr3/cre_precision_reproducible_cres_barplot.pdf')
fig.show()


# %% do upset plot of ATAC, H3K4me1, H3K27ac, Chromatin A
on_target_cres = {}
for use in ['atac-peak', 'h3k27ac-peak', 'h3k4me1-peak', 'chromatin-a']:
    cre_precision_df_all = get_cre_precision_df(res_q_right.copy(), starrfish3, reproducible_celltypes, use=use)
    on_target_cres[use] = cre_precision_df_all.index[cre_precision_df_all['TP'] > 0]
    overall_tp[use] = cre_precision_df_all['TP'].sum()
    overall_total[use] = len(cre_precision_df_all)
# do upset plot
from upsetplot import UpSet, from_contents
upset_data = from_contents(on_target_cres)
fig = plt.figure(figsize=(6, 4))
upset = UpSet(upset_data, subset_size='count', show_counts='%d', sort_by='degree', sort_categories_by=None)
upset.plot(fig=fig)
fig.savefig('results/expr3/cre_ontarget_upsetplot.pdf')
# %%
