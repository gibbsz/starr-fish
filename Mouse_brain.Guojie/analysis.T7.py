# %%
from turtle import st
import warnings
warnings.filterwarnings("ignore", category=SyntaxWarning, module="docrep")
import scvi
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
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
# # filter by t7 > 0 and cre > 0
# adata3_filtered = adata3[adata3.obsm['T7CRE'].sum(axis=1) > 0, :]
# adata3_filtered = adata3_filtered[adata3_filtered.obsm['CRE'].sum(axis=1) > 0, :]
# adata_cpm = 'Data/ATAC_cpm_peakBysubclass.csv'
# starrfish3 = STARRFISH(adata3, atac_cpm=adata_cpm)
# starrfish3_filtered = STARRFISH(adata3_filtered, atac_cpm=adata_cpm)
starrfish3 = STARRFISH.load('results/starrfish3.pkl')
starrfish3_filtered = STARRFISH.load('results/starrfish3_filtered.pkl')
starrfish2 = STARRFISH.load('results/starrfish2.pkl')
starrfish2_filtered = STARRFISH.load('results/starrfish2_filtered.pkl')
# %% define cell types to use for filtered data
negative_control_cres = starrfish3_filtered.get_negative_control_cres()
cell_types_counts3 = starrfish3_filtered.get_celltypes().value_counts()
cell_types_counts2 = starrfish2_filtered.get_celltypes().value_counts()
cell_types_to_use_3 = cell_types_counts3[cell_types_counts3 > 500].index
cell_types_to_use_2 = cell_types_counts2[cell_types_counts2 > 500].index
cell_types_to_use = cell_types_to_use_3.intersection(cell_types_to_use_2)
# check the negative control counts for those cell types
negative_control_counts3 = starrfish3_filtered.get_cre_expression()[negative_control_cres].groupby(starrfish3_filtered.get_celltypes()).sum()
negative_control_counts2 = starrfish2_filtered.get_cre_expression()[negative_control_cres].groupby(starrfish2_filtered.get_celltypes()).sum()
negative_control_sum_counts3 = starrfish3_filtered.get_cre_expression()[starrfish3_filtered.get_negative_control_cres()].sum(axis=1).groupby(starrfish3_filtered.get_celltypes()).sum()
negative_control_sum_counts2 = starrfish2_filtered.get_cre_expression()[starrfish2_filtered.get_negative_control_cres()].sum(axis=1).groupby(starrfish2_filtered.get_celltypes()).sum()
common_cell_types_sum_20_nc = negative_control_sum_counts3[negative_control_sum_counts3 > 20].index.intersection(negative_control_sum_counts2[negative_control_sum_counts2 > 20].index)
# define the cell types by the negative control counts > 50
cell_types_to_use_nc_3 = negative_control_sum_counts3[negative_control_sum_counts3 > 10].index
cell_types_to_use_nc_2 = negative_control_sum_counts2[negative_control_sum_counts2 > 10].index
cell_types_to_use_nc = cell_types_to_use_nc_3.intersection(cell_types_to_use_nc_2)
target_cres = starrfish2_filtered.get_creinfo().index[starrfish2_filtered.get_creinfo()['best_subclass'].isin(cell_types_to_use_nc_2)]
len(cell_types_to_use), len(cell_types_to_use_nc), len(cell_types_to_use_nc_2), len(target_cres)




# %%
# correlation of cell type counts
cell_type_counts = pd.DataFrame(index=cell_types_counts2.index.intersection(cell_types_counts3.index), columns=['Exp2', 'Exp3'])
cell_type_counts['Exp2'] = cell_types_counts2[cell_type_counts.index]
cell_type_counts['Exp3'] = cell_types_counts3[cell_type_counts.index]
fig, ax = plt.subplots(figsize=(8, 6))
sns.scatterplot(x=cell_type_counts['Exp2'], y=cell_type_counts['Exp3'], ax=ax, alpha=0.5)
ax.set_xlabel('Cell type counts in Exp2')
ax.set_ylabel('Cell type counts in Exp3')
# calculate the correlation
corr, p_value = pearsonr(cell_type_counts['Exp2'], cell_type_counts['Exp3'])
# plot text
ax.text(0.05, 0.95, f'Pearson r: {corr:.2f}\np-value: {p_value:.2e}', transform=ax.transAxes, fontsize=12, verticalalignment='top')
# plot the STR D1 Gaba and STR D2 Gaba cell types
text = []
for cell_type in ['STR D1 Gaba', 'STR D2 Gaba']:
    text.append(ax.text(cell_type_counts.loc[cell_type, 'Exp2'], cell_type_counts.loc[cell_type, 'Exp3'], cell_type, fontsize=12, ha='right', va='bottom'))
adjust_text(text, ax=ax, arrowprops=dict(arrowstyle='->', color='black', lw=0.5))
ax.set_xscale('log')
ax.set_yscale('log')




# %% correlation of CRE counts in each experiment
cre_counts2 = starrfish2.get_cre_expression().groupby(starrfish2.get_celltypes()).mean()
cre_counts3 = starrfish3.get_cre_expression().groupby(starrfish3.get_celltypes()).mean()
cre_corr, celltype_corr = starrfish2.corr_starrfish(cre_counts2, cre_counts3)
celltype_count2 = starrfish2.get_celltypes().value_counts()
celltype_count3 = starrfish3.get_celltypes().value_counts()
celltype_corr['celltype_count2'] = celltype_count2[celltype_corr.index].values
celltype_corr['celltype_count3'] = celltype_count3[celltype_corr.index].values
# %%
# cell type corr
fig, ax = plt.subplots(figsize=(8, 6))
sns.scatterplot(x=celltype_corr['celltype_count2'], y=celltype_corr['celltype_count3'], ax=ax, alpha=0.5,
                hue=celltype_corr['pearson'], palette='coolwarm')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('Cell type counts in Exp2')
ax.set_ylabel('Cell type counts in Exp3')
# %%
# CRE corr
fig, ax = plt.subplots(figsize=(8, 6))
cell_types_to_use = celltype_count2.index[celltype_count2 > 1000].intersection(celltype_count3.index[celltype_count3 > 1000])
cre_corr, celltype_corr = starrfish2.corr_starrfish(cre_counts2, cre_counts3)
cre_corr['lib_size'] = starrfish2.lib_size['counts'].loc[cre_corr.index].values
sns.scatterplot(x=cre_corr['lib_size'], y=cre_corr['pearson'], ax=ax, alpha=0.5)
ax.set_xscale('log')
ax.set_xlabel('AAV library size')
ax.set_ylabel('Pearson correlation')
ax.set_title('CRE counts across cell types (>500 cells) correlation between Exp2 and Exp3')




# %%
# first check the number of transcripts and T7 per cell type.
fig, ax = plt.subplots(ncols=2, figsize=(12, 6))
cre_counts = starrfish3.get_cre_expression().sum(axis=1)
t7_counts = starrfish3.get_t7_expression().sum(axis=1)
cre_celltype = cre_counts.groupby(starrfish3.get_tag('obs:subclass')).mean()
t7_celltype = t7_counts.groupby(starrfish3.get_tag('obs:subclass')).mean()
sns.scatterplot(x=t7_celltype, y=cre_celltype, ax=ax[0], alpha=0.5)
ax[0].set_xscale('log')
ax[0].set_yscale('log')
ax[0].set_xlabel('Average T7 counts per cell type')
ax[0].set_ylabel('Average CRE counts per cell type')
sns.scatterplot(x=cre_counts, y=t7_counts, ax=ax[1], alpha=0.5)
ax[1].set_xlabel('Total T7 counts per cell')
ax[1].set_ylabel('Total CRE counts per cell')



# %% assumption: cell segmentation issue
# get one unmatched cell
t7_expression = starrfish3.get_t7_expression().copy()
cre_expression = starrfish3.get_cre_expression().copy()
dont_match = (t7_expression == 0) & (cre_expression > 0)
# get the cell name and CRE name that don't match with largest number of CREs
cell_id = cre_expression[dont_match].max(axis=1).idxmax()
cre_id = cre_expression.loc[cell_id, dont_match.loc[cell_id]].idxmax()
print(f'Cell ID: {cell_id}, CRE ID: {cre_id}, CREs: {cre_expression.loc[cell_id, cre_id]}, T7: {t7_expression.loc[cell_id, cre_id]}')
# get the cell coordinates and closest neighbors
k_nn = starrfish3.get_k_nearest_neighbors(cell_id, k=1000)
# plot the cre and t7 expression in the region
fig1 = starrfish3.plot_gene(cre_id, use='CRE', 
                     x_region=[k_nn['X'].min(), k_nn['X'].max()], y_region=[k_nn['Y'].min(), k_nn['Y'].max()], 
                     norm_by_negative_control_cell_type_sum=False, sz_background=20, sz_min=20, sz_max=30,
                     log=False, show_celltypes=False,
                     figsize=(8, 6))
fig2 = starrfish3.plot_gene(cre_id, use='T7CRE', 
                     x_region=[k_nn['X'].min(), k_nn['X'].max()], y_region=[k_nn['Y'].min(), k_nn['Y'].max()], 
                     norm_by_negative_control_cell_type_sum=False, sz_background=20, sz_min=20, sz_max=30,
                     log=False, show_celltypes=False,
                     figsize=(8, 6))
fig1, fig2



# %%
# for each CRE, we calculate the correlation between T7 and CRE counts across all cells, and we compare to other CREs
# Vectorized correlation calculation using pandas corr() method
t7_expression.columns = [f'{col}_T7' for col in t7_expression.columns]
cre_expression.columns = [f'{col}_CRE' for col in cre_expression.columns]
t7_cre_combined = pd.concat([t7_expression, cre_expression], axis=1)
# remove all zeros
t7_cre_combined = t7_cre_combined[(t7_cre_combined != 0).any(axis=1)]
corr_matrix = t7_cre_combined.corr()

# Extract cross-correlations between T7 and CRE (exclude self-correlations)
t7_cols = t7_expression.columns
cre_cols = cre_expression.columns
cross_corr = corr_matrix.loc[t7_cols, cre_cols]
# visualize
fig, ax = plt.subplots(figsize=(12, 10))
# filter to any > 0.1
sns.heatmap(cross_corr.loc[(cross_corr > 0.1).any(axis=1), (cross_corr > 0.1).any(axis=0)], 
            ax=ax, cmap='coolwarm', center=0, annot=False, fmt='.2f', cbar_kws={'label': 'Correlation Coefficient'})
        


# %%
t7_unique = (starrfish3.get_t7_expression() > 0).sum(axis=1)
fig, ax = plt.subplots(ncols=3, figsize=(18, 6))
# only show n > 1 cells
sns.histplot(cre_counts[cre_counts > 0], bins=100, ax=ax[0])
sns.histplot(t7_counts[t7_counts > 0], bins=100, ax=ax[1])
sns.histplot(t7_unique[t7_unique > 0], bins=50, ax=ax[2])
# check the relationshape between T7 and CRE
fig, ax = plt.subplots(figsize=(8, 6))
cre_counts = starrfish3.get_cre_expression().sum(axis=1)
t7_counts = starrfish3.get_t7_expression().sum(axis=1)
sns.scatterplot(x=cre_counts, y=t7_counts, ax=ax, alpha=0.5)
ax.set_xlabel('CRE counts')
ax.set_ylabel('T7 counts')




# %% plot the number of t7 counts in each cell type
celltype_t7 = starrfish3.get_t7_expression().sum(axis=1).groupby(starrfish3.get_celltypes()).mean()
# rank the cell types by the number of T7 counts
celltype_t7 = celltype_t7.sort_values(ascending=False)
celltype_counts = starrfish3.get_celltypes().value_counts().loc[celltype_t7.index]
# bar plot
fig, ax = plt.subplots(figsize=(8, 6))
sns.scatterplot(x=celltype_counts.values, y=celltype_t7.values, color='orange', ax=ax)
ax.set_xlabel('Number of cells')
ax.set_xscale('log')
ax.set_ylabel('Average T7 counts per cell')
# mark a few cell types
cell_types = ['Oligo NN', 'Endo NN', 'Astro-NT NN', 'STR D1 Gaba', 'STR D2 Gaba', 'L5 IT CTX Glut', 'L6 CT CTX Glut']
texts = []
for cell_type in cell_types:
    if cell_type in celltype_t7.index:
        texts.append(ax.text(celltype_counts[cell_type], celltype_t7[cell_type], cell_type, fontsize=12, ha='right', va='bottom'))
# use adjust text to adjust the text positions
adjust_text(texts)



# %%
# plot the average T7 counts in each cell type, class and regions
subclass_T7 = starrfish3.get_t7_expression().sum(axis=1).groupby(starrfish3.get_tag('obs:subclass_name')).mean()
subclass_CRE = starrfish3.get_cre_expression().sum(axis=1).groupby(starrfish3.get_tag('obs:subclass_name')).mean()
subclass_NC_T7 = starrfish3.get_t7_expression()[negative_control_cres].sum(axis=1).groupby(starrfish3.get_tag('obs:subclass_name')).mean()
subclass_NC_CRE = starrfish3.get_cre_expression()[negative_control_cres].sum(axis=1).groupby(starrfish3.get_tag('obs:subclass_name')).mean()
class_T7 = starrfish3.get_t7_expression().sum(axis=1).groupby(starrfish3.get_tag('obs:class_name')).mean()
class_CRE = starrfish3.get_cre_expression().sum(axis=1).groupby(starrfish3.get_tag('obs:class_name')).mean()
class_NC_T7 = starrfish3.get_t7_expression()[negative_control_cres].sum(axis=1).groupby(starrfish3.get_tag('obs:class_name')).mean()
class_NC_CRE = starrfish3.get_cre_expression()[negative_control_cres].sum(axis=1).groupby(starrfish3.get_tag('obs:class_name')).mean()
cluster_annotation_term = pd.read_excel('Data/abc_atlas/allen_institute_nominature.xlsx')
# only keep class_id_label, subclass_id_label
cluster_annotation_term = cluster_annotation_term[['class_id_label', 'subclass_id_label', 'nt_type_combo_label']]
# drop that not in subclass
cluster_annotation_term = cluster_annotation_term[cluster_annotation_term['subclass_id_label'].isin(subclass_T7.index)]
unique_classes = np.sort(cluster_annotation_term['class_id_label'].unique())
# select color for 34 classes, combine tab20 and tab20b
palette = sns.color_palette("tab20b", n_colors=20) + sns.color_palette("tab20c", n_colors=20)[:12] + sns.color_palette("tab20c", n_colors=4)[-2:]
class_color_map = dict(zip(unique_classes, palette))
cluster_annotation_term['color'] = cluster_annotation_term['class_id_label'].map(class_color_map)
# assign T7 average counts to the subclass
cluster_annotation_term['T7Avg'] = subclass_T7.loc[cluster_annotation_term['subclass_id_label']].values
# %% plot average CRE counts per class
for p in ['T7', 'CRE', 'ratio_NC']:
    fig, ax = plt.subplots(ncols=1, figsize=(4, 6))
    # use the color map to color the bars
    if p == 'T7':
        sns.barplot(y=class_T7.index, x=class_T7.values, ax=ax, palette=class_color_map, orient='h')
    elif p == 'CRE':
        sns.barplot(y=class_CRE.index, x=class_CRE.values, ax=ax, palette=class_color_map, orient='h')
    else:
        ratio = class_NC_CRE / class_NC_T7
        sns.barplot(y=ratio.index, x=ratio.values, ax=ax, palette=class_color_map, orient='h')
    ax.set_xlabel(f'Average {p} counts per cell')
    # make y-label to the right
    ax.yaxis.set_label_position('right')
    ax.yaxis.tick_right()
    # flip x-axis, log scale x
    ax.set_xscale('log')
    # grey dash line
    if p == 'T7':
        vline_xs = [10, 20]
        ticks = [1, 2, 5, 10, 20, 40]
    elif p == 'CRE':
        vline_xs = [1, 2]
        ticks = [0.5, 1, 2, 5, 10]
    else:
        vline_xs = [0.1, 0.2]
        ticks = [0.05, 0.1, 0.2, 0.5, 1]
    ax.set_xticks(ticks)
    ax.set_xticklabels(ticks)
    for vline_x in vline_xs:
        ax.axvline(x=vline_x, color='grey', linestyle='--', linewidth=1.5)
    ax.set_xlim(ax.get_xlim()[::-1])
# %% neuron type level
cluster_annotation_term['nt_type_combo_label'] = cluster_annotation_term['nt_type_combo_label'].fillna('Non-neuron')
neuron_type_label = cluster_annotation_term['nt_type_combo_label'].groupby(cluster_annotation_term['subclass_id_label']).first().loc[starrfish3.get_tag('obs:subclass_name')]
neuron_type_label.index = starrfish3.adata.obs.index
neuron_type_T7 = starrfish3.get_t7_expression().sum(axis=1).groupby(neuron_type_label).mean()
neuron_type_CRE = starrfish3.get_cre_expression().sum(axis=1).groupby(neuron_type_label).mean()
neuron_type_NC_T7 = starrfish3.get_t7_expression()[negative_control_cres].sum(axis=1).groupby(neuron_type_label).mean()
neuron_type_NC_CRE = starrfish3.get_cre_expression()[negative_control_cres].sum(axis=1).groupby(neuron_type_label).mean()
# Horizontal colored bar for class
# get the color map for neuron types
unique_neuron_types = np.sort(cluster_annotation_term['nt_type_combo_label'].unique())
# select color for neuron types, use terrain color palette
palette = sns.color_palette("terrain", n_colors=len(unique_neuron_types))
neuron_type_color_map = dict(zip(unique_neuron_types, palette))
colors = [neuron_type_color_map[x] for x in neuron_type_T7.index]
for p in ['T7', 'CRE', 'ratio_NC']:
    fig, ax = plt.subplots(ncols=1, figsize=(4, 6))
    if p == 'T7':
        sns.barplot(y=neuron_type_T7.index, x=neuron_type_T7.values, ax=ax, palette=neuron_type_color_map, orient='h')
    elif p == 'CRE':
        sns.barplot(y=neuron_type_CRE.index, x=neuron_type_CRE.values, ax=ax, palette=neuron_type_color_map, orient='h')
    else:
        ratio = neuron_type_NC_CRE / neuron_type_NC_T7
        sns.barplot(y=ratio.index, x=ratio.values, ax=ax, palette=neuron_type_color_map, orient='h')
    ax.margins(y=0)
    ax.set_xlabel(f'Average {p} counts per cell')
    ax.set_ylabel('Neuron type')
    # make y-label to the right 
    ax.yaxis.set_label_position('right')
    ax.yaxis.tick_right()
    # flip x-axis, log scale x
    ax.set_xscale('log')
    # grey dash line
    if p == 'T7':
        vline_xs = [10, 20]
        ticks = [1, 2, 5, 10, 20, 40]
    elif p == 'CRE':
        vline_xs = [1, 2]
        ticks = [0.5, 1, 2, 5, 10]
    else:
        vline_xs = [0.1, 0.2]
        ticks = [0.05, 0.1, 0.2, 0.5, 1]
    ax.set_xticks(ticks)
    ax.set_xticklabels(ticks)
    for vline_x in vline_xs:
        ax.axvline(x=vline_x, color='grey', linestyle='--', linewidth=1.5)
    ax.set_xlim(ax.get_xlim()[::-1])
# %% subclass level
# Horizontal colored bar for class
subclass_color_map = dict(zip(cluster_annotation_term['subclass_id_label'], cluster_annotation_term['color']))
colors = [subclass_color_map[x] for x in subclass_T7.index]
for p in ['T7', 'CRE', 'ratio_NC']:
    fig, ax = plt.subplots(ncols=1, figsize=(4, 15))
    if p == 'T7':
        ax.barh(y=subclass_T7.index, width=subclass_T7.values, height=1.0, color=colors)
    elif p == 'CRE':
        ax.barh(y=subclass_CRE.index, width=subclass_CRE.values, height=1.0, color=colors)
    else:
        ratio = subclass_NC_CRE / subclass_NC_T7
        ax.barh(y=ratio.index, width=ratio.values, height=1.0, color=colors)
    ax.margins(y=0)
    ax.set_xlabel(f'Average {p} counts per cell')
    # remove y-label ticks
    ax.set_yticklabels([])
    ax.set_yticks([])
    # flip x-axis, log scale x
    ax.set_xscale('log')
    # grey dash line
    if p == 'T7':
        vline_xs = [10, 20]
        ticks = [1, 2, 5, 10, 20, 40]
    elif p == 'CRE':
        vline_xs = [1, 2]
        ticks = [0.5, 1, 2, 5, 10]
    else:
        vline_xs = [0.1, 0.2]
        ticks = [0.05, 0.1, 0.2, 0.5, 1]
    ax.set_xticks(ticks)
    ax.set_xticklabels(ticks)
    for vline_x in vline_xs:
        ax.axvline(x=vline_x, color='grey', linestyle='--', linewidth=1.5)
    ax.set_xlim(ax.get_xlim()[::-1])
    ax.set_ylim(ax.get_ylim()[::-1])  # flip y-axis
# %% spatial plot for T7
starrfish3.adata.obsm['T7Avg'] = pd.DataFrame({'T7Avg': subclass_T7.loc[starrfish3.adata.obs['subclass_name']].values},
                                              index=starrfish3.adata.obs.index)
starrfish3.plot_gene('T7Avg', use='T7Avg', norm_by_negative_control_cell_type_sum=False, log=False,
                     sz_min=1, sz_max=1)
# %% spatial plot for CRE
starrfish3.adata.obsm['CREAvg'] = pd.DataFrame({'CREAvg': subclass_CRE.loc[starrfish3.adata.obs['subclass_name']].values},
                                              index=starrfish3.adata.obs.index)
starrfish3.plot_gene('CREAvg', use='CREAvg', norm_by_negative_control_cell_type_sum=False, log=False,
                     sz_min=1, sz_max=1)
# %% spatial plot for T7/CRE ratio at class level
starrfish3.adata.obsm['CRE/T7'] = pd.DataFrame({'CRE/T7': (class_NC_CRE / class_NC_T7).loc[starrfish3.adata.obs['class_name']].values},
                                                 index=starrfish3.adata.obs.index)
starrfish3.plot_gene('CRE/T7', use='CRE/T7', norm_by_negative_control_cell_type_sum=False, log=False,
                     sz_min=1, sz_max=1, nmax=2)
# %% spatial plot for T7/CRE ratio at subclass level
starrfish3.adata.obsm['CRE/T7'] = pd.DataFrame({'CRE/T7': (subclass_NC_CRE / subclass_NC_T7).loc[starrfish3.adata.obs['subclass_name']].values},
                                                 index=starrfish3.adata.obs.index)
starrfish3.plot_gene('CRE/T7', use='CRE/T7', norm_by_negative_control_cell_type_sum=False, log=False,
                     sz_min=1, sz_max=1, nmax=2)
# %% check cell type
starrfish3.plot_cluster(clusters=['27 MY GABA', '18 TH Glut'], use='class_name')







# %% number of cells with CRE and cells with T7
cre_counts.shape[0] / starrfish3.adata.shape[0], t7_counts.shape[0] / starrfish3.adata.shape[0], t7_counts.shape[0] / cre_counts.shape[0]
# %% T7 correlation with AAV libaray size
t7_counts = starrfish3.get_t7_expression().sum(axis=0)
t7_counts = t7_counts.loc[starrfish3.lib_size.index]  # align with library size
fig, ax = plt.subplots(ncols = 2, figsize=(12, 6))
sns.scatterplot(x=np.expm1(starrfish3.lib_size['counts']), y=t7_counts, ax=ax[0], alpha=0.5)
ax[0].set_xlabel('AAV library size')
ax[0].set_ylabel('Total T7 counts in all cells')
# log scale
t7_counts_log = np.log1p(t7_counts)
sns.scatterplot(x=starrfish3.lib_size['counts'], y=t7_counts_log, ax=ax[1], alpha=0.5)
ax[1].set_xlabel('Log(AAV library size)')
ax[1].set_ylabel('Log(T7 counts)')
# draw correlation line
from scipy.stats import linregress
slope, intercept, r_value, p_value, std_err = linregress(np.expm1(starrfish3.lib_size['counts']), t7_counts)
x = np.expm1(np.linspace(starrfish3.lib_size['counts'].min(), starrfish3.lib_size['counts'].max(), 100))
y = slope * x + intercept
ax[0].plot(x, y, color='red', label=f'Correlation: {r_value:.2f}, p-value: {p_value:.2e}')
ax[0].legend()
# draw correlation line for log scale
slope_log, intercept_log, r_value_log, p_value_log, std_err_log = linregress(starrfish3.lib_size['counts'], t7_counts_log)
x_log = np.linspace(starrfish3.lib_size['counts'].min(), starrfish3.lib_size['counts'].max(), 100)
y_log = slope_log * x_log + intercept_log
ax[1].plot(x_log, y_log, color='red', label=f'Correlation: {r_value_log:.2f}, p-value: {p_value_log:.2e}')
ax[1].legend()  
# %% calculate the correlation between T7 and AAV library size in different cell types
corr_df = pd.DataFrame(columns=['celltype', 'correlation', 'p_value'])
for celltype in starrfish3.get_celltypes().unique():
    t7_counts = starrfish3.get_t7_expression().loc[starrfish3.get_celltypes() == celltype].sum(axis=0)
    t7_counts = t7_counts.loc[starrfish3.lib_size.index]  # align with library size
    slope, intercept, r_value, p_value, std_err = linregress(starrfish3.lib_size['counts'], np.log1p(t7_counts))
    corr_df = pd.concat([corr_df, pd.DataFrame({'celltype': [celltype], 'correlation': [r_value], 'p_value': [p_value]})], ignore_index=True)
# sort
corr_df = corr_df.sort_values(by='correlation', ascending=False)
# get number of cells
cell_counts = starrfish3.get_celltypes().value_counts()
corr_df['cell_counts'] = corr_df['celltype'].map(cell_counts)
# plot
fig, ax = plt.subplots(ncols=2, figsize=(12, 6))
sns.histplot(data=corr_df, x='correlation', ax=ax[0], bins=30, kde=True)
ax[0].set_xlabel('Correlation with AAV library size')
sns.scatterplot(data=corr_df, x='cell_counts', y='correlation', ax=ax[1], alpha=0.5)
ax[1].set_xlabel('Number of cells')
ax[1].set_xscale('log')
ax[1].set_ylabel('Correlation with AAV library size')




# %% test plot the negative control CREs in Endo NN, per cell T7 ~ CRE counts
# single cell don't match
cell_type = 'Endo NN'
ncols = 6
# cres = starrfish3.get_negative_control_cres()
cres = starrfish3.get_positive_control_cres(cell_type, use='atac-peak')
nrows = int(np.ceil(len(cres) / ncols))
fig, ax = plt.subplots(ncols = ncols, nrows = nrows, figsize=(5 *ncols, 4 * nrows), gridspec_kw={'hspace': 0.5, 'wspace': 0.5})
for i, cre in enumerate(cres):
    if nrows > 1:
        ax_work = ax[i // ncols, i % ncols]
    else:
        ax_work = ax[i % ncols]
    cre_counts = starrfish3.get_cre_expression().loc[starrfish3.get_celltypes() == cell_type][cre]
    t7_counts = starrfish3.get_t7_expression().loc[starrfish3.get_celltypes() == cell_type][cre]
    # match the cell names
    cre_counts = cre_counts.loc[t7_counts.index]
    sns.scatterplot(x=t7_counts, y=cre_counts, ax=ax_work, alpha=0.5)
    ax_work.set_xlabel(f'sc T7 counts')
    ax_work.set_ylabel(f'sc CRE counts')
    # ax_work.set_xscale('symlog')
    # ax_work.set_yscale('symlog')
    ax_work.set_title(f'{cre}')





# %%
# check if the bulk version of T7 and CRE counts match, do linear correlation
for cell_type in starrfish3.get_celltypes().value_counts().index:
    cre_ct = starrfish3.get_cre_expression().loc[starrfish3.get_celltypes() == cell_type].copy()
    t7_ct = starrfish3.get_t7_expression().loc[starrfish3.get_celltypes() == cell_type].copy()
    # filter out cells with low t7 counts
    # to_filter = t7_ct < 1
    # cre_ct[to_filter] = 0
    # t7_ct[to_filter] = 0
    # plot the sum of cre_endo and t7_endo
    fig, ax = plt.subplots(ncols=2, figsize=(14, 6))
    sns.scatterplot(y=cre_ct.sum(axis=0), x=t7_ct.sum(axis=0), ax=ax[0], alpha=0.5)
    sns.scatterplot(y=(cre_ct > 0).sum(axis=0), x=(t7_ct > 0).sum(axis=0), ax=ax[1], alpha=0.5)
    ax[0].set_xlabel('Total T7 counts')
    ax[0].set_ylabel('Total CRE counts')
    ax[1].set_xlabel('T7 > 0 cells')
    ax[1].set_ylabel('CRE > 0 cells')
    # find the negative control CREs
    neg_control_cres = starrfish3.get_negative_control_cres()
    # fine positive controls
    positive_control_cres = starrfish3.get_positive_control_cres(cell_type, use='atac-peak')
    sns.scatterplot(y=cre_ct[neg_control_cres].sum(axis=0), x=t7_ct[neg_control_cres].sum(axis=0), ax=ax[0], color='orange', label='Negative control CREs')
    sns.scatterplot(y=cre_ct[positive_control_cres].sum(axis=0), x=t7_ct[positive_control_cres].sum(axis=0), ax=ax[0], color='red', label='Positive control CREs')
    sns.scatterplot(y=(cre_ct[neg_control_cres] > 0).sum(axis=0), x=(t7_ct[neg_control_cres] > 0).sum(axis=0), ax=ax[1], color='orange', label='Negative control CREs')
    sns.scatterplot(y=(cre_ct[positive_control_cres] > 0).sum(axis=0), x=(t7_ct[positive_control_cres] > 0).sum(axis=0), ax=ax[1], color='red', label='Positive control CREs')
    # plot the names of positive_control CREs
    txts = []
    for cre in positive_control_cres:
        txts.append(ax[0].text(t7_ct[cre].sum(), cre_ct[cre].sum(), cre, fontsize=8, ha='right', va='bottom', color='red'))
    # adjust_text(txts, ax=ax[0], arrowprops=dict(arrowstyle='->', color='black', lw=0.5))
    ax[0].legend()
    ax[1].legend()
    # set x and y scale to log
    ax[0].set_xscale('symlog')
    ax[0].set_yscale('symlog')
    ax[1].set_xscale('symlog')
    ax[1].set_yscale('symlog')
    ax[0].set_title(f'{cell_type} - Total CREs vs Total T7')
# %% calculate the correlation between T7 and CRE counts in each cell type
corr_df = pd.DataFrame(columns=['cell_type', 'correlation', 'p_value'])
for cell_type in starrfish3.get_tag('obs:subclass_name').value_counts().index:
    cre_ct = starrfish3.get_cre_expression().loc[starrfish3.get_tag('obs:subclass_name') == cell_type].copy().sum(axis=0)
    t7_ct = starrfish3.get_t7_expression().loc[starrfish3.get_tag('obs:subclass_name') == cell_type].copy().sum(axis=0)
    # calculate the correlation
    corr, p_value = pearsonr(t7_ct.values.flatten(), cre_ct.values.flatten())
    corr_df = pd.concat([corr_df, pd.DataFrame({'cell_type': [cell_type], 'correlation': [corr], 'p_value': [p_value]})], ignore_index=True)
# %% plot
corr_df['color'] = corr_df['cell_type'].map(subclass_color_map)
# sort corr_df by cell_type
corr_df = corr_df.sort_values(by='cell_type', ascending=True)
fig, ax = plt.subplots(ncols=1, figsize=(4, 15))
# Horizontal colored bar for class
subclass_color_map = dict(zip(cluster_annotation_term['subclass_id_label'], cluster_annotation_term['color']))
colors = [subclass_color_map[x] for x in corr_df['cell_type']]
ax.barh(y=corr_df['cell_type'], width=corr_df['correlation'], height=1.0, color=colors)
ax.margins(y=0)
ax.set_xlabel('T7-CRE correlation')
# remove y-label ticks
ax.set_yticklabels([])
ax.set_yticks([])
ax.axvline(x=0.4, color='grey', linestyle='--', linewidth=1.5)
ax.set_xlim(ax.get_xlim()[::-1])
ax.set_ylim(ax.get_ylim()[::-1])





# %% # unmatched T7 and CRE counts
t7_expression = starrfish3.get_t7_expression().copy()
cre_expression = starrfish3.get_cre_expression().copy()
dont_match = (t7_expression == 0) & (cre_expression > 0)
proportion = pd.DataFrame({'dont_match': dont_match.sum(axis=0),
                           'cre > 0': (cre_expression > 0).sum(axis=0)}, index=dont_match.columns)
proportion['proportion'] = proportion['dont_match'] / proportion['cre > 0']
# proportion.to_csv('results/dont_match_proportion.csv')
cells_dont_match = pd.DataFrame({'numbers': dont_match.sum(axis=1)})
cells_dont_match['ID'] = np.nan
cells_dont_match = cells_dont_match[cells_dont_match['numbers'] > 0]
for cell in cells_dont_match.index:
    # get the cell IDs that dont match, join them with ";"
    cells_dont_match.loc[cell, 'ID'] = ";".join(dont_match.loc[cell].index[dont_match.loc[cell]].to_list())
x_spatial = pd.DataFrame(starrfish3.adata.obsm['X_spatial'],
                         index = starrfish3.adata.obs.index)
cells_dont_match['X_Spatial_0'] = x_spatial.loc[cells_dont_match.index, 0].values
cells_dont_match['X_Spatial_1'] = x_spatial.loc[cells_dont_match.index, 1].values
cells_dont_match['FOV'] = starrfish3.adata.obs['fov'].loc[cells_dont_match.index].values
cells_dont_match.to_csv('results/cells_dont_match.csv')
# %% plot the proportion
fig, ax = plt.subplots(ncols=2, figsize=(12, 6))
sns.histplot(data=proportion, x='proportion', bins=100, ax=ax[0], kde=False)
ax[0].set_xlabel('Proportion of cells with T7-CRE mismatch for each CRE')
sns.histplot(data=dont_match.sum(axis=1), bins=100, ax=ax[1], kde=False)
ax[1].set_xlabel('Number of CREs with T7-CRE mismatch in each cell')






# %%
# focus on correlation between CRE counts across experiments
corr_df = pd.DataFrame(columns=['cell_type', 'correlation', 'p_value'])
exp2_counts = starrfish2_filtered.get_cre_expression().groupby(starrfish2_filtered.get_celltypes()).sum()
exp3_counts = starrfish3_filtered.get_cre_expression().groupby(starrfish3_filtered.get_celltypes()).sum()
for cell_type in cell_types_to_use:
    corr, p_value = pearsonr(exp2_counts.loc[cell_type], exp3_counts.loc[cell_type])
    corr_df = pd.concat([corr_df, pd.DataFrame({'cell_type': [cell_type], 'correlation': [corr], 'p_value': [p_value]})], ignore_index=True)
# plot the correlation
fig, ax = plt.subplots(figsize=(8, 6))
sns.scatterplot(data=corr_df, x='cell_type', y='correlation', ax=ax, alpha=0.5)
ax.set_title('Correlation between CRE counts in each cell type (Exp2 vs Exp3)')
ax.set_xlabel('Cell Type')
ax.set_ylabel('Pearson Correlation Coefficient')
plt.xticks(rotation=90, fontsize=5)
plt.tight_layout()
plt.show()





# %% calculate average, bootstrap q-value
average_bootstrap_test_config = {
    'cell_types_to_use': None,
    'normalize_by_cell_rna': False,
    'normalize_by_cell_volume': False,
    'normalize_by_cell_t7': False,  # normalize by T7, filter cells with T7 < 4
    'normalize_by_celltype_rna': False,
    'normalize_by_celltype_volume': False,
    'normalize_by_celltype_t7': False,  # normalize by T7
    'filter_by_cell_t7': None,
    'normalize_by_negative_control': False,  # normalize by negative control
    'normalize_by_libsize': True,
    'log_transform': False,
    'bootstrap_number': 10000,
    'bootstrap_to_fixed_pct': 0.5,
    'bootstrap_to_fixed_sample_size': None,
    'load_stored': True,
    'n_jobs': 128,
}
threshold = 'neg_control'
res = starrfish3.average_bootstrap_test(**average_bootstrap_test_config)
res_q, res_df = starrfish3.average_bootstrap_test_q(res, threshold=threshold, norm='libsize', tail='right')





# %% do volcano plot
cell_type = 'Endo NN'
norm = 'libsize'
fig, ax = plt.subplots(figsize=(5, 4))
def plot_volcano(res_q, res_df, res_array, cell_type, starrfish3_obj, ax):
    # log then average
    res_array = np.log(res_array)
    # assign inf to NaN
    res_array[np.isinf(res_array)] = np.nan
    neg_control_array = res_array[:, :, res_df.columns.isin(starrfish3_obj.get_negative_control_cres())]
    neg_control_array = np.nanmean(neg_control_array, axis=2)
    # cap to min 1e-5
    res_q = res_q.clip(lower=1e-5)
    cre_ct1 = starrfish3_obj.get_cre_expression().loc[starrfish3_obj.get_celltypes() == cell_type].copy()
    t7_ct1 = starrfish3_obj.get_t7_expression().loc[starrfish3_obj.get_celltypes() == cell_type].copy()
    # filter out cells with low t7 counts
    total_neg_control_cre1 = cre_ct1[starrfish3_obj.get_negative_control_cres()].sum(axis=0).sum()
    total_neg_control_t71 = t7_ct1[starrfish3_obj.get_negative_control_cres()].sum(axis=0).sum()
    # calculate the fold change
    # calculate the fold change
    if threshold == '0':
        fdc_u = 0
        fdc_l = fdc_u
    elif threshold == 'total':
        if norm == 'T7':
            fdc_u = np.log(cre_ct1.sum(axis=0).sum() / t7_ct1.sum(axis=0).sum())
        elif norm == 'libsize':
            fdc_u = np.log(cre_ct1.sum() / starrfish3_obj.lib_size['counts'].sum())
        fdc_l = fdc_u
    elif threshold == 'total_dist':
        # use the distribution of total CREs to set the threshold
        fdc = np.nanmean(res_array[:, res_df.index == cell_type])
        fdc_std = np.nanstd(res_array[:, res_df.index == cell_type])
        fdc_u = fdc + 2 * fdc_std
        fdc_l = fdc - 2 * fdc_std
    elif threshold == 'neg_control':
        if norm == 'T7':
            fdc_u = np.log(total_neg_control_cre1 / total_neg_control_t71)
        elif norm == 'libsize':
            fdc_u = np.log(total_neg_control_cre1 / starrfish3_obj.lib_size['counts'][starrfish3_obj.get_negative_control_cres()].sum())
        fdc_l = fdc_u
    elif threshold == 'neg_control_dist':
        # use the distribution of negative control CREs to set the threshold
        fdc = np.nanmean(neg_control_array[:, res_df.index == cell_type])
        fdc_std = np.nanstd(neg_control_array[:, res_df.index == cell_type])
        fdc_u = fdc + 2 * fdc_std
        fdc_l = fdc - 2 * fdc_std
    fdc = (fdc_u + fdc_l) / 2
    # fdc = 0
    sns.scatterplot(x=res_df.loc[cell_type].values.flatten(), y=-np.log10(res_q.loc[cell_type].values.flatten().astype(float)), ax=ax, alpha=0.5)
    ax.axhline(y=-np.log10(0.05), color='red', linestyle='--')
    ax.axvline(x=fdc, color='black', linestyle='--')
    ax.axvline(x=fdc_u, color='grey', linestyle='--')
    ax.axvline(x=fdc_l, color='grey', linestyle='--')
    ax.set_xlabel('Fold change (log)')
    ax.set_ylabel('-log10(q-value)')
    # mark the negative control CREs
    neg_control_cres = starrfish3_obj.get_negative_control_cres()
    positive_control_cres = starrfish3_obj.get_positive_control_cres(cell_type, use='atac-peak')
    sns.scatterplot(x=res_df.loc[cell_type, neg_control_cres].values.flatten(), y=-np.log10(res_q.loc[cell_type, neg_control_cres].values.flatten().astype(float)), ax=ax, color='orange', label='Negative control CREs')
    sns.scatterplot(x=res_df.loc[cell_type, positive_control_cres].values.flatten(), y=-np.log10(res_q.loc[cell_type, positive_control_cres].values.flatten().astype(float)), ax=ax, color='red', label='Positive control CREs')
    # plot positive controls
    texts = []
    for cre in positive_control_cres:
        texts.append(ax.text(res_df.loc[cell_type, cre], -np.log10(res_q.loc[cell_type, cre]), cre, fontsize=8, ha='right', va='bottom', color='red'))
    adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle='->', color='black', lw=0.5))
    ax.legend()
plot_volcano(res_q, res_df, res['celltype_activity_array'], cell_type, starrfish3, ax)





# %%
# calculate precision and recall
pr_df = pd.DataFrame(index=res_df.index, columns=['neg_control', 'atac-peak', 'chrom-a', 'all-pred', 'all-atac-peak', 'all-chrom-a'])
for cell_type in res_df.index:
    cres = res_df.columns
    # get the true positive, false positive, false negative, true negative
    all_pred = res_q.columns[(res_q.loc[cell_type] <= 0.05) & np.isfinite(res_df.loc[cell_type])]
    atac_peak = starrfish3.get_positive_control_cres(cell_type, use='atac-peak')
    chrom_a = starrfish3.get_positive_control_cres(cell_type, use='chromatin-a')
    neg_control = starrfish3.get_negative_control_cres()
    pr_df.loc[cell_type, 'neg_control'] = len(np.intersect1d(neg_control, all_pred))
    pr_df.loc[cell_type, 'all-neg_control'] = len(neg_control)
    if atac_peak is not None:
        pr_df.loc[cell_type, 'atac-peak'] = len(np.intersect1d(atac_peak, all_pred))
        pr_df.loc[cell_type, 'all-atac-peak'] = len(atac_peak)
    if chrom_a is not None:
        pr_df.loc[cell_type, 'chrom-a'] = len(np.intersect1d(chrom_a, all_pred))
        pr_df.loc[cell_type, 'all-chrom-a'] = len(chrom_a)
    pr_df.loc[cell_type, 'all-pred'] = len(all_pred)
# fill NaN with 0
pr_df = pr_df.fillna(0)
# plot the number of neg_control in each cell type
fig, ax = plt.subplots(ncols = 3, nrows = 2, figsize=(12, 8), gridspec_kw={'wspace': 0.5})
sns.scatterplot(x=pr_df['all-pred'], y=pr_df['all-neg_control']-pr_df['neg_control'], ax=ax[0, 0], alpha=0.5)
sns.scatterplot(x=pr_df['all-pred'], y=pr_df['atac-peak'], ax=ax[0, 1], alpha=0.5)
sns.scatterplot(x=pr_df['all-pred'], y=pr_df['chrom-a'], ax=ax[0, 2], alpha=0.5)
ax[0, 0].set_xlabel('Number of significant CREs')
ax[0, 0].set_ylabel('Number of negative control CREs')
ax[0, 1].set_xlabel('Number of significant CREs')
ax[0, 1].set_ylabel('Number of ATAC-peak positive control CREs')
ax[0, 2].set_xlabel('Number of significant CREs')
ax[0, 2].set_ylabel('Number of chrom-a positive control CREs')
# barplot of precision
pr_df['atac-peak-precision'] = pr_df['atac-peak'] / pr_df['all-atac-peak']
pr_df['neg_control-precision'] = 1 - pr_df['neg_control'] / pr_df['all-neg_control']
pr_df['chrom-a-precision'] = pr_df['chrom-a'] / pr_df['all-chrom-a']
sns.scatterplot(x=pr_df['neg_control-precision'], y=pr_df['atac-peak-precision'], ax=ax[1, 1], alpha=0.5)
sns.scatterplot(x=pr_df['neg_control-precision'], y=pr_df['chrom-a-precision'], ax=ax[1, 2], alpha=0.5)
print(pr_df['atac-peak'].sum() / pr_df['all-atac-peak'].sum(), (1-pr_df['neg_control'].sum()/pr_df['all-neg_control'].sum()), pr_df['chrom-a'].sum() / pr_df['all-chrom-a'].sum())
# %% Horizontal colored bar for class
class_color_map = dict(zip(cluster_annotation_term['class_id_label'], cluster_annotation_term['color']))
cluster_annotation_term['subclass'] = cluster_annotation_term['subclass_id_label'].str.replace('^[0-9]+ ', '', regex=True)
cluster_annotation_term['subclass'] = cluster_annotation_term['subclass'].str.replace('/', '-')
pr_df['class_name'] = cluster_annotation_term['class_id_label'].groupby(cluster_annotation_term['subclass']).first().loc[pr_df.index].values
pr_df_class = pr_df.groupby('class_name').sum()
pr_df_class['atac-peak-precision'] = pr_df_class['atac-peak'] / pr_df_class['all-atac-peak']
pr_df_class['neg_control-precision'] = 1 - pr_df_class['neg_control'] / pr_df_class['all-neg_control']
pr_df_class['chrom-a-precision'] = pr_df_class['chrom-a'] / pr_df_class['all-chrom-a']
fig, ax = plt.subplots(ncols=3, figsize=(12, 15))
for i, p in enumerate(['atac-peak-precision', 'neg_control-precision', 'chrom-a-precision']):
    ax[i].barh(y=pr_df_class.index, width=pr_df_class[p].values, height=1.0, color=pr_df_class.index.map(class_color_map))
    ax[i].margins(y=0)
    ax[i].set_xlabel(f'{p}')
    # remove y-label ticks
    ax[i].set_yticklabels([])
    ax[i].set_yticks([])
    # grey dash line
    if p == 'atac-peak-precision':
        vline_xs = [pr_df_class['atac-peak'].sum() / pr_df_class['all-atac-peak'].sum()]
    elif p == 'neg_control-precision':
        vline_xs = [(1-pr_df_class['neg_control'].sum()/pr_df_class['all-neg_control'].sum())]
    else:
        vline_xs = [pr_df_class['chrom-a'].sum() / pr_df_class['all-chrom-a'].sum()]
    for vline_x in vline_xs:
        ax[i].axvline(x=vline_x, color='grey', linestyle='--', linewidth=1.5)
    ax[i].set_xlim(ax[i].get_xlim()[::-1])
    ax[i].set_ylim(ax[i].get_ylim()[::-1])  # flip y-axis
# Create legend for subclass colors
from matplotlib.patches import Patch
class_color_map = dict(zip(cluster_annotation_term['class_id_label'], cluster_annotation_term['color']))
legend_elements = [Patch(facecolor=color, label=cls) 
                   for cls, color in class_color_map.items()]
fig.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5))
# %% Horizontal colored bar for subclass
class_color_map = dict(zip(cluster_annotation_term['subclass_id_label'], cluster_annotation_term['color']))
pr_df.index = cluster_annotation_term['subclass_id_label'].groupby(cluster_annotation_term['subclass']).first().loc[pr_df.index].values
pr_df = pr_df.sort_index()
fig, ax = plt.subplots(ncols=3, figsize=(12, 15))
for i, p in enumerate(['atac-peak-precision', 'neg_control-precision', 'chrom-a-precision']):
    ax[i].barh(y=pr_df.index, width=pr_df[p].values, height=1.0, color=pr_df.index.map(class_color_map))
    ax[i].margins(y=0)
    ax[i].set_xlabel(f'{p}')
    # remove y-label ticks
    ax[i].set_yticklabels([])
    ax[i].set_yticks([])
    # grey dash line
    if p == 'atac-peak-precision':
        vline_xs = [pr_df['atac-peak'].sum() / pr_df['all-atac-peak'].sum()]
    elif p == 'neg_control-precision':
        vline_xs = [(1-pr_df['neg_control'].sum()/pr_df['all-neg_control'].sum())]
    else:
        vline_xs = [pr_df['chrom-a'].sum() / pr_df['all-chrom-a'].sum()]
    for vline_x in vline_xs:
        ax[i].axvline(x=vline_x, color='grey', linestyle='--', linewidth=1.5)
    ax[i].set_xlim(ax[i].get_xlim()[::-1])
    ax[i].set_ylim(ax[i].get_ylim()[::-1])  # flip y-axis
# Create legend for subclass colors
from matplotlib.patches import Patch
class_color_map = dict(zip(cluster_annotation_term['class_id_label'], cluster_annotation_term['color']))
legend_elements = [Patch(facecolor=color, label=cls) 
                   for cls, color in class_color_map.items()]
fig.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5))






# %% plot the histogram of the negative control log fold change to t7 in different cell types
activity_df = res_df.copy()
res_array = np.log(res['celltype_activity_array'].copy())
# fill inf with NaN
res_array[np.isinf(res_array)] = np.nan
for cell_type in starrfish3.get_celltypes().value_counts().index[:5]:
    res_array_cell_type = pd.DataFrame(res_array[:, activity_df.index == cell_type, ][:, 0, :], columns = activity_df.columns)
    # add another column of mean log fold change to T7 of negative control CREs
    res_array_neg_control = res_array_cell_type[neg_control_cres]
    res_array_neg_control['mean'] = np.nanmean(res_array_neg_control.values, axis=1)
    fig, ax = plt.subplots(nrows = 2, figsize=(6, 12), sharex=True)
    # make a toplot from res_array
    to_plot1 = pd.DataFrame({'log_fold_change': res_array_neg_control.values.T.flatten(),
                             'cres': np.repeat(neg_control_cres.to_list() + ['mean'], repeats=res_array.shape[0])})
    to_plot2 = pd.DataFrame({'log_fold_change': res_array_cell_type[res_array_cell_type.columns.difference(neg_control_cres)].values.T.flatten()})
    to_plot1['type'] = 'negative_control'
    to_plot1['type'][to_plot1['cres'] == 'mean'] = 'mean_negative_control'
    to_plot2['type'] = 'non_negative_control'
    to_plot2 = pd.concat([to_plot2, to_plot1], axis=0, ignore_index=True)
    # do histogram, bin alpha=0.5, kde=True
    # skip if all values are NaN
    if to_plot1['log_fold_change'].isna().all():
        continue
    sns.histplot(data=to_plot1, x='log_fold_change', ax=ax[0], bins=100, kde=True, hue='cres', fill=False, common_norm=False,
                 palette='tab10', stat='density', alpha=0.5)
    ax[0].set_xlabel('Log fold change to T7')
    ax[0].set_ylabel('Density')
    ax[0].set_title(f'Histogram of log fold change to T7 in {cell_type}')
    # do histogram for non-negative control CREs
    sns.histplot(data=to_plot2, x='log_fold_change', ax=ax[1], bins=100, kde=True, hue='type', fill=False, common_norm=False,
                stat='density', alpha=0.5)
    ax[1].set_xlabel('Log fold change to T7')
    ax[1].set_ylabel('Density')
    ax[1].set_title(f'Histogram of log fold change to T7 in {cell_type}')
    # do t-test for each negative control CRE to the total negative control CRE distribution
    from scipy.stats import ks_2samp
    p_values = []
    for cres in neg_control_cres:
        cres_values = res_array_cell_type[cres].values.flatten()
        all_neg_control_values = res_array_cell_type[neg_control_cres].values.flatten()
        # remove NaN values
        cres_values = cres_values[~np.isnan(cres_values)]
        all_neg_control_values = all_neg_control_values[~np.isnan(all_neg_control_values)]
        ks_stat, p_value = ks_2samp(cres_values, all_neg_control_values)
        p_values.append(p_value)
    print(f'Cell type: {cell_type}, p-values for negative control CREs: {dict(zip(neg_control_cres, p_values))}')









# %%
# mark any CREs with q-value > 0.05 to Nan in the res_df
activity_df = res_df.copy()
activity_df[res_q > 0.05] = np.nan
# fullfill NaN with 0
activity_df = activity_df.fillna(0)
# add cpm data
starrfish3.load_cpm('Data/ATAC_cpm_peakBysubclass.csv', attr_to_add='atac_cpm')
starrfish3.load_cpm('Data/H3K4me1_cpm_peak_pad_500_Bysubclass.csv', attr_to_add='h3k4me1_cpm')
starrfish3.load_cpm('Data/H3K9me3_cpm_peak_pad_500_Bysubclass.csv', attr_to_add='h3k9me3_cpm')
starrfish3.load_cpm('Data/H3K27ac_cpm_peak_pad_500_Bysubclass.csv', attr_to_add='h3k27ac_cpm')
starrfish3.load_cpm('Data/H3K27me3_cpm_peak_pad_500_Bysubclass.csv', attr_to_add='h3k27me3_cpm')
# add chromatin state data
chromatin_o = pd.read_csv('Data/cre_chromatin_state_o.csv', index_col=0)
chromatin_a = pd.read_csv('Data/cre_chromatin_state_a.csv', index_col=0)
starrfish3.chromatin_o = (chromatin_o.copy() + chromatin_a.copy()) / 2
starrfish3.chromatin_a = chromatin_a.copy()
# add snapatac2_de data
snapatac2_de_fc =  pd.read_csv('Data/snapatac2_de_fc.csv', index_col=0)
snapatac2_de_pval =  pd.read_csv('Data/snapatac2_de_pval.csv', index_col=0)
starrfish3.snapatac2_de_fc = snapatac2_de_fc
starrfish3.snapatac2_de_pval = snapatac2_de_pval
# do correlation with cpm
cell_types_to_use_nc_2_common = activity_df.index.copy()
for mod in ['atac_cpm', 'h3k4me1_cpm', 'h3k9me3_cpm', 'h3k27ac_cpm', 'h3k27me3_cpm']:
    cell_types_to_use_nc_2_common = cell_types_to_use_nc_2_common.intersection(
        getattr(starrfish3, mod).index)
# normalize activity_df by library size
cre_corr, celltype_corr = starrfish3.corr_atac_cpm(
    cell_types_to_use=cell_types_to_use_nc_2_common, cres_to_use=None, 
    acvitity_df=activity_df, 
    filter_by_atac_z_threshold=None, filter_by_atac_raw_threshold=None,
    filter_by_negative_control_z_threshold=None,
    log_activity=True,
    log_atac=False)
significant_cres = cre_corr[(cre_corr['pearson_p'] <= 0.05) & (cre_corr['pearson'] > 0)].index
significant_celltypes = celltype_corr[(celltype_corr['pearson_p'] <= 0.05) & (celltype_corr['pearson'] > 0)].index
print(len(significant_cres), len(significant_celltypes))
# %%
# corr with ATAC
cre_atac_peaks = pd.read_csv('Data/cre_atac_peaks.csv', index_col=0)
ct = 0
for cell_type in activity_df.index.intersection(cre_atac_peaks.index):
    peak_cres = cre_atac_peaks.loc[cell_type].index[cre_atac_peaks.loc[cell_type] > 0]
    # how about ATAC z-score > 2
    # atac_z = np.log1p(starrfish2_filtered.atac_cpm.loc[cell_type].copy())
    # atac_z = (atac_z - atac_z.mean()) / atac_z.std()
    # peak_cres = atac_z[atac_z > 2].index
    # violin plot
    toplot = pd.DataFrame({'activity': activity_df.loc[cell_type].values,
                           'peak': activity_df.columns.isin(peak_cres)},
                          index=activity_df.columns)
    # toplot = toplot[toplot['lib_size'] > 100]  # filter by lib size
    # do a t-test between the two groups
    from scipy.stats import ttest_ind
    peak_activity = toplot[toplot['peak']]['activity']
    non_peak_activity = toplot[~toplot['peak']]['activity']
    t_stat, p_value = ttest_ind(peak_activity, non_peak_activity)
    if p_value < 0.05:
        ct += 1
ct
# %%
# try SCVI now
# set CUDA_VISIBLE_DEVICES
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
scvi_test_config = {'use_model': 'STARRFISHVI', 
                    'model_args': {'infection_rate_inference': 'decoder',
                                   'infection_rate_type': 'gene-cell',
                                   'kl_infection_rate_type': '',
                                   'n_latent': 24,
                                   'infection_rate_generative': ''}}
adata_mvi = starrfish3_filtered.scvi(**scvi_test_config)
starrfish4 = STARRFISH(adata_mvi, atac_cpm='Data/ATAC_cpm_peakBysubclass.csv')
# %%
# glm test
glm_test_config = {
    'variate': 'T7',
    'norm_by_volm': False,
    'volm_covariate': True,
    'fov_covariate': True,
    'rna_covariate': False,
    'filter_infected_cells': False,
    'positive_x_or_y': True,
    'only_keep_positive_x': False,
    'only_keep_positive_y': False,
}
glm_res = starrfish3.glm_test(**glm_test_config)
activity_df = glm_res['coef']
# %%
# pseudo bulk glm test
pseudo_bulk_glm_test_config = {
    'variate': 'T7',
    'norm_by_volm': False,
    'volm_covariate': True,
    # 'fov_covariate': True,
    'rna_covariate': False,
    'filter_infected_cells': False,
    'positive_x_or_y': True,
    'only_keep_positive_x': False,
    'only_keep_positive_y': False,
}
glm_res = starrfish3.pseudo_bulk_glm_test(**pseudo_bulk_glm_test_config)
activity_df = glm_res['result']['coef']
# %%
starrfish3.save('results/starrfish3.pkl')
starrfish3_filtered.save('results/starrfish3_filtered.pkl')
# %%
fig, ax = plt.subplots(figsize=(8, 6))
i=7
sns.scatterplot(x=t7_ct[neg_control_cres[i]], y=cre_ct[neg_control_cres[i]], ax=ax, color='red', label='Negative control CREs')
# %%
starrfish4.plot_umap(clusters=cell_types_to_use_nc_2_common.tolist())
# %%
starrfish4.plot_gene('CRE048', norm_by_negative_control_cell_type_sum=False, log=False)
# %%

# %%
fold_change_test_config = {"cell_types_to_use": None,
                           "normalize_by_cell_rna": False,
                           "normalize_by_cell_volume": False,
                           "normalize_by_cell_t7": False, # normalize by T7
                           'filter_by_cell_t7': None,
                           "normalize_by_celltype_rna": False,
                           "normalize_by_celltype_volume": False,
                           "normalize_by_celltype_t7": True, # normalize by T7
                           "normalize_by_negative_control": True, # normalize by negative control
                           'normalize_by_total_cre': False,
                           "normalize_by_infected_cell": False,
                           "normalize_by_libsize": False,
                           "log_transform": False,
                           "filter_zero_counts": False,
                           "bootstrap_number": 10000,
                           'load_stored': True,}
res = starrfish3.fold_change_test(**fold_change_test_config)
res_p = res['pvalue_activity'].loc[cell_types_to_use_nc].copy()
activity_df = res['celltype_activity'].loc[cell_types_to_use_nc].copy()
# q-value correction
res_q = pd.DataFrame(multitest.multipletests(res_p.values.flatten(), method='fdr_bh')[1].reshape(res_p.shape),
                      index=res_p.index, columns=res_p.columns)
neg_q = res_q[starrfish3.get_negative_control_cres()]
q_threshold = 0.01
print(neg_q.loc[(neg_q <= q_threshold).any(axis=1), (neg_q <= q_threshold).any(axis=0)])
# get have target cres
have_target_cres = res_q.loc[:, (res_q <= q_threshold).any(axis=0)].columns
# remove any cre with nan activity
have_target_cres = have_target_cres[~activity_df[have_target_cres].isna().any(axis=0)]
# %% plot the q value cre plot
from plots import cre_pval_dotplot
cre_info = starrfish2_filtered.get_creinfo().copy()
cre_info['best_subclass'] = 'CRE'
cre_info.loc[negative_control_cres, 'best_subclass'] = 'Negative Control'
cre_pval_dotplot(res_q, activity_df, cres_to_use=have_target_cres.union(negative_control_cres), 
                 cell_types_to_use=cell_types_to_use_nc,
                 positive_control_info=cre_info,
                 significant_cutoff=q_threshold, figsize=(12, 20), flip_axis=False)
# %%
# load expr2 results
fold_change_test_config['cell_types_to_use'] = None
fold_change_test_config['fill_nan'] = False  # normalize by T7
res2 = starrfish2_filtered.fold_change_test(**fold_change_test_config)
res2_p = res2['pvalue_activity'].loc[cell_types_to_use_nc_2].copy()
activity_df2 = res2['celltype_activity'].loc[cell_types_to_use_nc_2].copy()
# q-value correction
res2_q = pd.DataFrame(multitest.multipletests(res2_p.values.flatten(), method='fdr_bh')[1].reshape(res2_p.shape),
                      index=res2_p.index, columns=res2_p.columns)
# %% plot reproducibility of CRE activity
reproducibility_df = pd.DataFrame(index=cell_types_to_use_nc_2.intersection(cell_types_to_use_nc_3),
                                  columns=['exp2', 'exp3', 'overlap'])
for cell_type in cell_types_to_use_nc_2.intersection(cell_types_to_use_nc_3):
    cres_exp2 = res2_q.loc[cell_type].index[res2_q.loc[cell_type] <= q_threshold]
    cres_exp3 = res_q.loc[cell_type].index[res_q.loc[cell_type] <= q_threshold]
    reproducibility_df.loc[cell_type, 'exp2'] = len(cres_exp2)
    reproducibility_df.loc[cell_type, 'exp3'] = len(cres_exp3)
    reproducibility_df.loc[cell_type, 'overlap'] = len(set(cres_exp2).intersection(set(cres_exp3)))
reproducibility_df
# %% do the fold change test on expr2 
# %% calculate average, bootstrap
average_bootstrap_test_config = {
    'cell_types_to_use': None,
    'normalize_by_cell_rna': False,
    'normalize_by_cell_volume': False,
    'normalize_by_cell_t7': False,  # normalize by T7, filter cells with T7 < 4
    'normalize_by_celltype_rna': False,
    'normalize_by_celltype_volume': False,
    'normalize_by_celltype_t7': False,  # normalize by T7
    'filter_by_cell_t7': 1,
    'normalize_by_negative_control': False,  # normalize by negative control
    'normalize_by_libsize': True,
    'log_transform': False,
    'bootstrap_number': 10000,
    'bootstrap_to_fixed_pct': 0.5,
    'bootstrap_to_fixed_sample_size': None,
    'load_stored': True,
    'n_jobs': 128,
}
res = starrfish3_filtered.average_bootstrap_test(**average_bootstrap_test_config)
res_q, res_df = starrfish3_filtered.average_bootstrap_test_q(res, threshold='total', norm='libsize', tail='right')
average_bootstrap_test_config = {
    'cell_types_to_use': None,
    'normalize_by_cell_rna': False,
    'normalize_by_cell_volume': False,
    'normalize_by_cell_t7': False,  # normalize by T7, filter cells with T7 < 4
    'normalize_by_celltype_rna': False,
    'normalize_by_celltype_volume': False,
    'normalize_by_celltype_t7': False,  # normalize by T7
    'filter_by_cell_t7': None,
    'normalize_by_negative_control': False,  # normalize by negative control
    'normalize_by_libsize': True,
    'log_transform': False,
    'bootstrap_number': 10000,
    'bootstrap_to_fixed_pct': 0.5,
    'bootstrap_to_fixed_sample_size': None,
    'load_stored': True,
    'n_jobs': 128,
}
res = starrfish2_filtered.average_bootstrap_test(**average_bootstrap_test_config)
# fold change to T7 array
res2_q, res2_df = starrfish2_filtered.average_bootstrap_test_q(res, threshold='total', norm='libsize', tail='right')



# %%
cre_counts_3 = starrfish3_filtered.get_cre_expression().sum(axis=0)
cre_counts_2 = starrfish2_filtered.get_cre_expression().sum(axis=0)
fig, ax = plt.subplots(figsize=(8, 6))
sns.scatterplot(x=cre_counts_2, y=cre_counts_3, ax=ax, alpha=0.5)
ax.set_title('CRE Counts in Exp2 vs Exp3')
ax.set_xlabel('Exp2 CRE Count')
ax.set_ylabel('Exp3 CRE Count')
ax.set_xscale('log')
ax.set_yscale('log')


# %%
# reproducibility of expr2 and expr3
q_threshold = 0.05
reproducibility_df = pd.DataFrame(index=cell_types_to_use_nc_2.intersection(cell_types_to_use_nc_3),
                                  columns=['exp2', 'exp3', 'overlap'])
for cell_type in cell_types_to_use_nc_2.intersection(cell_types_to_use_nc_3):
    cres_exp2 = res2_q.loc[cell_type].index[res2_q.loc[cell_type] <= q_threshold]
    cres_exp3 = res_q.loc[cell_type].index[res_q.loc[cell_type] <= q_threshold]
    reproducibility_df.loc[cell_type, 'exp2'] = len(cres_exp2)
    reproducibility_df.loc[cell_type, 'exp3'] = len(cres_exp3)
    reproducibility_df.loc[cell_type, 'overlap'] = len(set(cres_exp2).intersection(set(cres_exp3)))
reproducibility_df



# %%
# correlation of cell type expression between expr2 and expr3
from scipy import stats
reproducibility_df = pd.DataFrame(index=cell_types_to_use_2.intersection(cell_types_to_use_3),
                                  columns=['pearson', 'spearman', 'pearson_p', 'spearman_p'])
for cell_type in cell_types_to_use_2.intersection(cell_types_to_use_3):
    cres_exp2_df = pd.DataFrame(starrfish2_filtered.get_rna_expression(),
                             index = starrfish2_filtered.get_celltypes().index,
                             columns = starrfish2_filtered.adata.var_names)
    cres_exp3_df = pd.DataFrame(starrfish3.get_rna_expression(),
                             index = starrfish3.get_celltypes().index,
                             columns = starrfish3.adata.var_names)
    cres_exp2 = cres_exp2_df.loc[starrfish2_filtered.get_celltypes() == cell_type].sum(axis=0)
    cres_exp3 = cres_exp3_df.loc[starrfish3.get_celltypes() == cell_type].sum(axis=0)
    # exclude CRE217
    # cres_exp2 = cres_exp2[cres_exp2.index != 'CRE217']
    # cres_exp3 = cres_exp3[cres_exp3.index != 'CRE217']
    reproducibility_df.loc[cell_type, 'pearson'] = stats.pearsonr(cres_exp2, cres_exp3)[0]
    reproducibility_df.loc[cell_type, 'spearman'] = stats.spearmanr(cres_exp2, cres_exp3)[0]
    reproducibility_df.loc[cell_type, 'pearson_p'] = stats.pearsonr(cres_exp2, cres_exp3)[1]
    reproducibility_df.loc[cell_type, 'spearman_p'] = stats.spearmanr(cres_exp2, cres_exp3)[1]
# plot the correlation
fig, ax = plt.subplots(figsize=(8, 6))
sns.scatterplot(data=reproducibility_df, x='pearson', y='spearman', ax=ax, alpha=0.5)
text = []
for cell_type in reproducibility_df.index:
    text.append(ax.text(reproducibility_df.loc[cell_type, 'pearson'], 
                        reproducibility_df.loc[cell_type, 'spearman'], 
                        cell_type, fontsize=8, ha='right', va='bottom'))
adjust_text(text, ax=ax, arrowprops=dict(arrowstyle='->', color='black', lw=0.5))
ax.set_title('Correlation between RNA counts in each cell type (Exp2 vs Exp3)')
ax.set_xlabel('Pearson Correlation Coefficient')
ax.set_ylabel('Spearman Correlation Coefficient')
plt.tight_layout()
plt.show()


# %%
# check STR D1 Gaba cells
cell_type = 'STR D1 Gaba'
cres_exp2 = starrfish2_filtered.get_cre_expression().loc[starrfish2_filtered.get_celltypes() == cell_type].sum(axis=0)
cres_exp3 = starrfish3.get_cre_expression().loc[starrfish3.get_celltypes() == cell_type].sum(axis=0)
cres_expr2 = cres_exp2[cres_exp2.index != 'CRE217']
cres_expr3 = cres_exp3[cres_exp3.index != 'CRE217']
fig, ax = plt.subplots(figsize=(8, 6))
sns.scatterplot(x=cres_exp2, y=cres_exp3, ax=ax, alpha=0.5)
ax.set_title('STR D1 Gaba Cells: Exp2 vs Exp3')
ax.set_xlabel('Exp2 CRE Count')
ax.set_ylabel('Exp3 CRE Count')
ax.set_xscale('symlog')
ax.set_yscale('symlog')
ax.set_xlim(-1, cres_exp2.max() * 1.2)
ax.set_ylim(-1, cres_exp3.max() * 1.2)
plt.tight_layout()
plt.show()

# %%
res_q = res2_q.copy()
# mark any CREs with q-value > 0.05 to Nan in the res_df
activity_df = res_df.copy()
# activity_df[res_q > 0.05] = np.nan
cell_types_to_plot = cell_types_to_use_3.copy()
# Remove rows and columns with all NaN values
activity_df = activity_df.loc[(~activity_df.isna()).any(axis=1), (~activity_df.isna()).any(axis=0)]
# reorder activity_df by the order of cell types
cluster_annotation_term = pd.read_csv('Data/abc_atlas/cluster_annotation_term.csv', index_col=0)
cluster_annotation_term['subclass'] = cluster_annotation_term['subclass'].str.replace('/', '-')
cell_types_to_plot_cluster_number = cluster_annotation_term['subclass_number'].groupby(cluster_annotation_term['subclass']).first().loc[cell_types_to_plot].values
cell_types_to_plot = pd.Index(cell_types_to_plot[np.argsort(cell_types_to_plot_cluster_number)])
# just draw the heatmap of activity_df, draw dendrogram on the top and left
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
# scale the activity to z-score in each cell type
# activity_df = activity_df.sub(np.nanmean(activity_df, axis=1), axis=0).div(np.nanstd(activity_df, axis=1), axis=0)  # Z-score per cell type
# fullfill NaN with 0
activity_df = activity_df.fillna(-3)
# Create a color series: highlighted rows get a color, others get gray or None
col_colors = pd.Series(
    ['red' if r in negative_control_cres else 'lightgray' for r in activity_df.columns],
    index=activity_df.columns
)
# Plot
g = sns.clustermap(activity_df.loc[cell_types_to_plot], 
                   cmap='coolwarm', 
                   center=-3, 
                   figsize=(30, 12),
                   method='average',
                   metric='euclidean',
                   xticklabels=True, 
                   yticklabels=True,
                   linewidths=0.5,
                   cbar_pos=(0, 0, 0.1, 0.01),
                   cbar_kws={'label': 'z-score lfc', 'orientation': 'horizontal'},
                   dendrogram_ratio=(0.03, 0.03),
                   col_colors=col_colors,
                   row_cluster=False)
# Rotate labels for better readability
g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xticklabels(), rotation=90, fontsize=6)
g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_yticklabels(), rotation=0, fontsize=8)
g.ax_heatmap.set_xlabel('CREs')
g.ax_heatmap.set_ylabel('Cell Types')

plt.suptitle('CRE Activity Heatmap with Hierarchical Clustering', y=1.02)
plt.show()








# %%
# plot cumulative correlation versus CREs, we need to see that but not necessarily in the manuscript
from plots import cre_corr_dotplot
infected_cells_threshold = 5
to_filter = (starrfish3.get_cre_expression() > 0).groupby(starrfish3.get_celltypes()).sum() < infected_cells_threshold
cre_blacklist = ['CRE061', 'CRE143', 'CRE001']
# read in the mismatching barcode CREs
mismatching_cres = pd.read_csv('Data/AAV_ONT_Barcode_Counts_vs_Mismatch_Percentage.csv', index_col=0)
cre_blacklist = np.unique(cre_blacklist + mismatching_cres.index[mismatching_cres['MismatchPercent'] > 20].tolist()).tolist()
to_filter[cre_blacklist] = True
# %%
fold_change_test_config = {"cell_types_to_use": None,
                           "normalize_by_cell_rna": False,
                           "normalize_by_cell_volume": False,
                           "normalize_by_cell_t7": False, # normalize by T7
                           'filter_by_cell_t7': None,
                           "normalize_by_celltype_rna": False,
                           "normalize_by_celltype_volume": False,
                           "normalize_by_celltype_t7": True, # normalize by T7
                           "normalize_by_negative_control": False, # normalize by negative control
                           'normalize_by_total_cre': False,
                           "normalize_by_infected_cell": False,
                           "normalize_by_libsize": False,
                           "log_transform": False,
                           "filter_zero_counts": False,
                           "bootstrap_number": 10000,
                           "n_jobs": 196,
                           'load_stored': True,}
res = starrfish3.fold_change_test(**fold_change_test_config)
p = res['pvalue_activity'].copy()
q = p.copy()
q[to_filter] = np.nan
q = q.values.flatten().copy().astype(float)
q[~np.isnan(q)] = multitest.multipletests(q[~np.isnan(q)], method='fdr_bh')[1]
q = pd.DataFrame(q.reshape(p.shape), index=p.index, columns=p.columns)
q_toplot = q.copy()
res_toplot = res['celltype_activity'].copy()
cell_types_to_use = starrfish3.get_celltypes().value_counts().index[starrfish3.get_celltypes().value_counts()>=200]
# cell_types_to_use = starrfish3.get_celltypes().value_counts().index
q_toplot[to_filter] = np.nan
res_toplot[to_filter] = np.nan
# cres_to_use = q_toplot.columns[np.nanmin(q_toplot.loc[cell_types_to_use], axis=0) < 0.05].union(starrfish3.get_negative_control_cres())
cres_to_use = q_toplot.columns
cres_to_use = cres_to_use[~cres_to_use.isin(cre_blacklist)]
# %%
corr_cutoffs = np.linspace(-1, 1, 200)
prob = {'atac_cpm': [], 'h3k4me1_cpm': [], 'h3k9me3_cpm': [], 'h3k27ac_cpm': [], 'h3k27me3_cpm': []}
significant_cres_mod = {}
violin_res = pd.DataFrame()
for mod in ['atac_cpm', 'h3k4me1_cpm', 'h3k27ac_cpm', 'h3k9me3_cpm', 'h3k27me3_cpm']:
    cre_corr, celltype_corr = starrfish3.corr_atac_cpm(
        cell_types_to_use=cell_types_to_use, cres_to_use=cres_to_use, 
        acvitity_df=res_toplot, 
        filter_by_atac_z_threshold=None, filter_by_atac_raw_threshold=None,
        filter_by_negative_control_z_threshold=None,
        log_activity=False, log_atac=False, attr_to_use=mod)
    print(f"Variance explained by {mod}: {(cre_corr['pearson'] ** 2).mean()}")
    cre_corr['mod'] = mod.replace('_cpm', '')
    cre_corr['CRE'] = cre_corr.index
    significant_cres = cre_corr[(cre_corr['pearson_p'] <= 0.05) & (cre_corr['pearson'] > 0)].index
    significant_cres_mod[mod] = set(significant_cres)
    violin_res = pd.concat([violin_res, cre_corr], axis=0, ignore_index=True)
    for corr_cutoff in corr_cutoffs:
        prop = (cre_corr['pearson'] >= corr_cutoff).sum() / len(cre_corr)
        prob[mod].append(prop)
# add lib size
violin_res['lib_size'] = starrfish3.lib_size.loc[violin_res['CRE'], 'counts'].values
fig = cre_corr_dotplot(starrfish3, pd.Series(list(set.union(significant_cres_mod['atac_cpm'], significant_cres_mod['h3k4me1_cpm'], significant_cres_mod['h3k27ac_cpm']))), 
                       cell_types_to_use, mods=['atac_cpm', 'h3k4me1_cpm', 'h3k27ac_cpm'],
                       test_method='fold_change', test_configs=fold_change_test_config, qval_df=q_toplot, log=False,
                       scale_by_cre=True, z_score_by_cre=False, sz_max=100, figsize=(24, 12))
fig.savefig(f'results/expr3/fdc_atac_cpm_corr_dotplot.pdf')
fig
# %% venn
from matplotlib_venn import venn3
mod_dict = {'atac_cpm': f'ATAC ({len(significant_cres_mod['atac_cpm'])} / {len(cres_to_use)})', 
            'h3k4me1_cpm': f'H3K4me1 ({len(significant_cres_mod['h3k4me1_cpm'])} / {len(cres_to_use)})',
            'h3k27ac_cpm': f'H3K27ac ({len(significant_cres_mod['h3k27ac_cpm'])} / {len(cres_to_use)})',}
fig, ax = plt.subplots(figsize=(6, 4))
venn = venn3([significant_cres_mod[i] for i in mod_dict.keys()], set_labels=mod_dict.values(), ax=ax)
fig.savefig(f'results/expr3/expr3_fdc_atac_cpm_corr_significant_venn.pdf')
# %%
# cumulative probability plot
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(corr_cutoffs, prob['atac_cpm'], label='ATAC', color='#A6CEE3')
ax.plot(corr_cutoffs, prob['h3k4me1_cpm'], label='H3K4me1', color='#B2DF8A')
ax.plot(corr_cutoffs, prob['h3k9me3_cpm'], label='H3K9me3', color='#FB8072')
ax.plot(corr_cutoffs, prob['h3k27ac_cpm'], label='H3K27ac', color='#FDB462')
ax.plot(corr_cutoffs, prob['h3k27me3_cpm'], label='H3K27me3', color='#CAB2D6')
ax.set_xlabel('Pearson correlation with epigenomic markers')
ax.set_ylabel('Proportion correlation ≥ cutoff')
ax.set_xlim(0, 1)
ax.set_ylim(0, 0.6)
ax.legend()
fig.tight_layout()
fig.savefig(f'results/expr3/expr2_cre_cumulative_prob.pdf')
# violin plot
fig, ax = plt.subplots(figsize=(6, 4))
lib_size_palette = {'0–100': '#A6CEE3', '100–1000': '#B2DF8A', '>1000': '#FB8072'}
bins = [0, 100, 1000, float('inf')]
labels = ['0–100', '100–1000', '>1000']
violin_res['lib_size_group'] = pd.cut(violin_res['lib_size'], bins=bins, labels=labels)
sns.violinplot(data=violin_res, x='mod', y='pearson', ax=ax, inner='quartile', scale='width', hue='mod',
               palette={'atac': '#A6CEE3', 'h3k4me1': '#B2DF8A', 'h3k9me3': '#FB8072',
                        'h3k27ac': '#FDB462', 'h3k27me3': '#CAB2D6'})
# jittered points
sns.stripplot(data=violin_res, x='mod', y='pearson', color='k', size=2, jitter=True, ax=ax, alpha=0.5)
ax.set_ylabel('Activity correlation with epigenomic markers')
# %% plot a heatmap with the correlation values
# Create correlation heatmap
corr_df = pd.DataFrame(index=violin_res['CRE'].unique(), 
                       columns=['atac', 'h3k4me1', 'h3k27ac'])
p_val_df = corr_df.copy()
for mod in corr_df.columns:
    corr_df[mod] = violin_res[violin_res['mod'] == mod]['pearson'].values
    p_val_df[mod] = violin_res[violin_res['mod'] == mod]['pearson_p'].values
# Define significance criteria
p_threshold = 0.05
corr_threshold = 0.0
# Create significance masks
sig_mask = (p_val_df < p_threshold) & (corr_df > corr_threshold)
# Count significant correlations per CRE
sig_counts = sig_mask.sum(axis=1)
# Heatmap of correlations for each CRE
# Group CREs by significance count
groups = {}
for i in range(4):
    group_cres = sig_counts[sig_counts == i].index.tolist()
    groups[i] = group_cres
# Function to perform hierarchical clustering within a group
def cluster_within_group(cres_list, corr_matrix, pval_matrix):
    """Perform hierarchical clustering on a subset of CREs"""
    if len(cres_list) <= 1:
        return cres_list
    # Extract correlation data for this group
    group_corr = corr_matrix.loc[cres_list]
    group_pval = pval_matrix.loc[cres_list]
    # order the cres based on correlation of first column
    cres_list = group_corr.iloc[:, 0].abs().sort_values(ascending=False).index.tolist()
    group_corr = group_corr.loc[cres_list]
    group_pval = group_pval.loc[group_corr.index]
    # binarize the p-values for clustering
    group_pval = group_pval < p_threshold
    # Calculate distance matrix (1 - correlation for clustering)
    # We'll use the correlation patterns across the three metrics as features
    # fill nans with zeros for group_corr
    group_corr.fillna(0, inplace=True)
    group_pval.fillna(1, inplace=True)
    distance_matrix = pdist(group_corr.astype(float).values, metric='euclidean')
    distance_p_matrix = pdist(group_pval.astype(float).values, metric='euclidean')
    # Perform hierarchical clustering
    linkage_matrix = linkage(distance_matrix + 100*distance_p_matrix, method='ward')
    # Get the order of CREs after clustering, reverse the order to get original order
    clustered_order = leaves_list(linkage_matrix)
    # Return CREs in clustered order
    return [cres_list[i] for i in clustered_order]
# Cluster CREs within each group
clustered_groups = {}
group_names = ['None Significant', '1 Significant', '2 Significant', 'All 3 Significant']
for sig_count in [3, 2, 1]:  # Start with most significant
    group_cres = groups[sig_count]
    if len(group_cres) > 0:
        # Sort by mean absolute correlation first, then cluster
        group_corr = corr_df.loc[group_cres]
        group_pval = p_val_df.loc[group_cres]
        mean_abs_corr = np.abs(group_corr).mean(axis=1)
        sorted_cres = mean_abs_corr.sort_values(ascending=False).index.tolist()
        # Perform clustering within this sorted group
        clustered_cres = cluster_within_group(sorted_cres, corr_df, group_pval)
        clustered_groups[sig_count] = clustered_cres
        print(f"Group '{group_names[sig_count]}': {len(clustered_cres)} CREs clustered")
# Create final ordered list of CREs
ordered_cres = []
group_boundaries = [0]
group_labels = []
for sig_count in [3, 2, 1]:  # Most to least significant
    if sig_count in clustered_groups:
        if sig_count == 3:
            ordered_cres.extend(clustered_groups[sig_count][::-1])
        else:
            ordered_cres.extend(clustered_groups[sig_count])
        group_boundaries.append(len(ordered_cres))
        group_labels.append(group_names[sig_count])
# Reorder correlation matrix according to clustered groups
ordered_corr_matrix = corr_df.loc[ordered_cres].T
# Create the comprehensive heatmap
fig, ax = plt.subplots(figsize=(8, 3))
# Create heatmap
im = ax.imshow(ordered_corr_matrix.astype(float).values, cmap='RdBu_r', aspect='auto', vmin=-0.6, vmax=0.6)
# Set labels
ax.set_yticks(range(len(ordered_corr_matrix.index)))
ax.set_yticklabels(['Activity vs ATAC', 'Activity vs H3K4me1', 'Activity vs H3K27ac'])
# Set x-axis ticks as the ordered CREs
ax.set_xticks(range(len(ordered_cres)))
ax.set_xticklabels(ordered_cres, rotation=45, fontsize=8, ha='right')
# Add group boundaries
for boundary in group_boundaries[1:-1]:  # Skip first (0) and last (end)
    ax.axvline(x=boundary-0.5, color='black', linewidth=2)
# Add colorbar
cbar = plt.colorbar(im, ax=ax, shrink=0.6)
cbar.set_label('Correlation Coefficient', fontsize=12)
# Set title and labels
ax.set_title('STARR-FISH Activity vs Chromatin Profiles', 
             fontsize=16, weight='bold', pad=20)
ax.set_xlabel('CREs', fontsize=12)
ax.set_ylabel('Correlation Metrics', fontsize=12)
fig.savefig('results/expr3/cre_correlation_heatmap.pdf', bbox_inches='tight')
# %%
from plots import cre_corr_heatmap
selected_cres = [pd.Series(list(significant_cres_mod['atac_cpm'])), 
                 pd.Series(list(significant_cres_mod['h3k4me1_cpm'])), 
                 pd.Series(list(significant_cres_mod['h3k27ac_cpm']))]
fig = cre_corr_heatmap(starrfish3, selected_cres, 
                       cell_types_to_use=cell_types_to_use,
                       mods = ['atac_cpm', 'h3k4me1_cpm', 'h3k27ac_cpm'], 
                       test_method='fold_change', test_configs=fold_change_test_config,
                       qval_df = q_toplot, log = False, scale_by_cre=True, z_score_by_cre=False, figsize=(24, 0.3))
fig.savefig(f'results/fold_change/expr2_cre_corr_heatmap.pdf')
# %%
