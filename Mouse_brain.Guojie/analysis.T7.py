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
cre_blacklist = ['CRE061', 'CRE143', 'CRE001']
cre_whitelist = starrfish3.get_creinfo().index[~starrfish3.get_creinfo().index.isin(cre_blacklist)]
# read in the mismatching barcode CREs
mismatching_cres = pd.read_csv('Data/AAV_ONT_Barcode_Counts_vs_Mismatch_Percentage.csv', index_col=0)
cre_whitelist = cre_whitelist[~cre_whitelist.isin(mismatching_cres.index[mismatching_cres['MismatchPercent'] > 20])]
cre_blacklist = np.unique(cre_blacklist + mismatching_cres.index[mismatching_cres['MismatchPercent'] > 20].tolist()).tolist()
# negative control CREs
negative_control_cres = starrfish3.get_negative_control_cres()
negative_control_cres = [cre for cre in negative_control_cres if cre in cre_whitelist]
starrfish3.blacklist_cre = cre_blacklist
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
        vline_xs = [class_T7.mean(), starrfish3.get_t7_expression().sum(axis=1).mean()]
        color_xs = ['grey', 'black']
        ticks = [1, 2, 5, 10, 20, 40]
    elif p == 'CRE':
        vline_xs = [class_CRE.mean(), starrfish3.get_cre_expression().sum(axis=1).mean()]
        color_xs = ['grey', 'black']
        ticks = [0.5, 1, 2, 5, 10]
    else:
        vline_xs = [ratio.mean(), starrfish3.get_cre_expression().sum(axis=1).mean() / starrfish3.get_t7_expression().sum(axis=1).mean()]
        color_xs = ['grey', 'black']
        ticks = [0.05, 0.1, 0.2, 0.5, 1]
    ax.set_xticks(ticks)
    ax.set_xticklabels(ticks)
    for vline_x, color_x in zip(vline_xs, color_xs):
        ax.axvline(x=vline_x, color=color_x, linestyle='--', linewidth=1.5)
    ax.set_xlim(ax.get_xlim()[::-1])
    fig.tight_layout()
    fig.savefig(f'results/expr3/fig3/Avg_{p}_per_class.pdf')
    

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
        vline_xs = [neuron_type_T7.mean(), starrfish3.get_t7_expression().sum(axis=1).mean()]
        color_xs = ['grey', 'black']
        ticks = [1, 2, 5, 10, 20, 40]
    elif p == 'CRE':
        vline_xs = [neuron_type_CRE.mean(), starrfish3.get_cre_expression().sum(axis=1).mean()]
        color_xs = ['grey', 'black']
        ticks = [0.5, 1, 2, 5, 10]
    else:
        vline_xs = [ratio.mean(), starrfish3.get_cre_expression().sum(axis=1).mean() / starrfish3.get_t7_expression().sum(axis=1).mean()]
        color_xs = ['grey', 'black']
        ticks = [0.05, 0.1, 0.2, 0.5, 1]
    ax.set_xticks(ticks)
    ax.set_xticklabels(ticks)
    for vline_x, color_x in zip(vline_xs, color_xs):
        ax.axvline(x=vline_x, color=color_x, linestyle='--', linewidth=1.5)
    ax.set_xlim(ax.get_xlim()[::-1])
    fig.tight_layout()
    fig.savefig(f'results/expr3/fig3/Avg_{p}_per_neuron_type.pdf')
    
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
        vline_xs = [subclass_T7.mean(), starrfish3.get_t7_expression().sum(axis=1).mean()]
        color_xs = ['grey', 'black']
        ticks = [1, 2, 5, 10, 20, 40]
    elif p == 'CRE':
        vline_xs = [subclass_CRE.mean(), starrfish3.get_cre_expression().sum(axis=1).mean()]
        color_xs = ['grey', 'black']
        ticks = [0.5, 1, 2, 5, 10]
    else:
        vline_xs = [subclass_NC_CRE.mean() / subclass_NC_T7.mean(), starrfish3.get_cre_expression().sum(axis=1).mean() / starrfish3.get_t7_expression().sum(axis=1).mean()]
        color_xs = ['grey', 'black']
        ticks = [0.05, 0.1, 0.2, 0.5, 1]
    ax.set_xticks(ticks)
    ax.set_xticklabels(ticks)
    for vline_x, color_x in zip(vline_xs, color_xs):
        ax.axvline(x=vline_x, color=color_x, linestyle='--', linewidth=1.5)
    ax.set_xlim(ax.get_xlim()[::-1])
    ax.set_ylim(ax.get_ylim()[::-1])  # flip y-axis
    fig.tight_layout()
    fig.savefig(f'results/expr3/fig3/Avg_{p}_per_subclass.pdf')
    
# %% spatial plot for T7
starrfish3.adata.obsm['T7Avg'] = pd.DataFrame({'T7Avg': subclass_T7.loc[starrfish3.adata.obs['subclass_name']].values},
                                              index=starrfish3.adata.obs.index)
fig = starrfish3.plot_gene('T7Avg', use='T7Avg', 
                           norm_by_negative_control_cell_type_sum=False, 
                           norm_by_t7_cell_type_mean=False,
                           log=False, sz_min=1, sz_max=1)
fig.savefig('results/expr3/fig3/Spatial_subclass_T7Avg.pdf')

# %% spatial plot for CRE
starrfish3.adata.obsm['CREAvg'] = pd.DataFrame({'CREAvg': subclass_CRE.loc[starrfish3.adata.obs['subclass_name']].values},
                                              index=starrfish3.adata.obs.index)
fig = starrfish3.plot_gene('CREAvg', use='CREAvg', 
                           norm_by_negative_control_cell_type_sum=False, log=False,
                           norm_by_t7_cell_type_mean=False,
                           sz_min=1, sz_max=1)
fig.savefig('results/expr3/fig3/Spatial_subclass_CREAvg.pdf')

# %% spatial plot for T7/CRE ratio at class level
starrfish3.adata.obsm['CRE/T7'] = pd.DataFrame({'CRE/T7': (class_NC_CRE / class_NC_T7).loc[starrfish3.adata.obs['class_name']].values},
                                                 index=starrfish3.adata.obs.index)
fig = starrfish3.plot_gene('CRE/T7', use='CRE/T7', 
                     norm_by_negative_control_cell_type_sum=False, 
                        norm_by_t7_cell_type_mean=False,
                     log=False, sz_min=1, sz_max=1, nmax=2)
fig.savefig('results/expr3/fig3/Spatial_CRET7_class_NC_ratio.pdf')
# %% spatial plot for T7/CRE ratio at subclass level
starrfish3.adata.obsm['CRE/T7'] = pd.DataFrame({'CRE/T7': (subclass_NC_CRE / subclass_NC_T7).loc[starrfish3.adata.obs['subclass_name']].values},
                                                 index=starrfish3.adata.obs.index)
fig = starrfish3.plot_gene('CRE/T7', use='CRE/T7', norm_by_negative_control_cell_type_sum=False, norm_by_t7_cell_type_mean=False,
                     log=False, sz_min=1, sz_max=1, nmax=2)
fig.savefig('results/expr3/fig3/Spatial_CRET7_subclass_NC_ratio.pdf')


# %% number of cells with CRE and cells with T7
cre_counts.shape[0] / starrfish3.adata.shape[0], t7_counts.shape[0] / starrfish3.adata.shape[0], t7_counts.shape[0] / cre_counts.shape[0]
# %% T7 correlation with AAV libaray size
t7_counts = starrfish3.get_t7_expression().sum(axis=0)
t7_counts = t7_counts.loc[starrfish3.lib_size.index]  # align with library size
fig, ax = plt.subplots(ncols = 2, figsize=(12, 6))
sns.scatterplot(x=(starrfish3.lib_size['counts']), y=t7_counts, ax=ax[0], alpha=0.5)
ax[0].set_xlabel('AAV library size')
ax[0].set_ylabel('Total T7 counts in all cells')
# log scale
t7_counts_log = np.log1p(t7_counts)
sns.scatterplot(x=np.log1p(starrfish3.lib_size['counts']), y=t7_counts_log, ax=ax[1], alpha=0.5)
ax[1].set_xlabel('Log(AAV library size)')
ax[1].set_ylabel('Log(T7 counts)')
# draw correlation line
from scipy.stats import linregress
slope, intercept, r_value, p_value, std_err = linregress(starrfish3.lib_size['counts'], t7_counts)
x = np.linspace(starrfish3.lib_size['counts'].min(), starrfish3.lib_size['counts'].max(), 100)
y = slope * x + intercept
ax[0].plot(x, y, color='red', label=f'Correlation: {r_value:.2f}, p-value: {p_value:.2e}')
ax[0].legend()
# draw correlation line for log scale
slope_log, intercept_log, r_value_log, p_value_log, std_err_log = linregress(np.log1p(starrfish3.lib_size['counts']), t7_counts_log)
x_log = np.linspace(np.log1p(starrfish3.lib_size['counts']).min(), np.log1p(starrfish3.lib_size['counts']).max(), 100)
y_log = slope_log * x_log + intercept_log
ax[1].plot(x_log, y_log, color='red', label=f'Correlation: {r_value_log:.2f}, p-value: {p_value_log:.2e}')
ax[1].legend()  
fig.savefig('results/expr3/fig3/T7_vs_AAV_library_size.pdf')

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
fig.savefig('results/expr3/fig3/T7_AAV_library_size_correlation_per_celltype.pdf')



# %% test plot the correlation between total T7 counts and total CRE counts
fig, ax = plt.subplots(figsize=(5, 4))
cre_counts = starrfish3.get_cre_expression().sum(axis=0)
t7_counts = starrfish3.get_t7_expression().sum(axis=0)
sns.scatterplot(x=np.log(cre_counts), y=np.log(t7_counts), ax=ax, alpha=0.5)
# add linear regression line
slope, intercept, r_value, p_value, std_err = linregress(np.log(cre_counts), np.log(t7_counts))
x = np.linspace(np.log(cre_counts).min(), np.log(cre_counts).max(), 100)
y = slope * x + intercept
ax.plot(x, y, color='red', label=f'Correlation: {r_value:.2f}, p-value: {p_value:.2e}')
ax.legend()
ax.set_xlabel('Total CRE counts per CRE (log scale)')
ax.set_ylabel('Total T7 counts per CRE (log scale)')
fig.savefig('results/expr3/fig3/Total_T7_vs_Total_CRE_counts_per_CRE.pdf')
# %% calculate this correlation in each cell type
corr_df = pd.DataFrame(columns=['celltype', 'correlation', 'p_value'])
for celltype in starrfish3.get_celltypes().unique():
    cre_counts = starrfish3.get_cre_expression().loc[starrfish3.get_celltypes() == celltype].sum(axis=0)
    t7_counts = starrfish3.get_t7_expression().loc[starrfish3.get_celltypes() == celltype].sum(axis=0)
    try:
        slope, intercept, r_value, p_value, std_err = linregress(np.log(cre_counts + 1), np.log(t7_counts + 1))
        corr_df = pd.concat([corr_df, pd.DataFrame({'celltype': [celltype], 'correlation': [r_value], 'p_value': [p_value]})], ignore_index=True)
    except Exception as e:
        print(f"Error processing celltype {celltype}: {e}")
# sort
corr_df = corr_df.sort_values(by='correlation', ascending=False)
# get number of cells
cell_counts = starrfish3.get_celltypes().value_counts()
corr_df['cell_counts'] = corr_df['celltype'].map(cell_counts)
# plot
fig, ax = plt.subplots(ncols=2, figsize=(12, 6))
sns.histplot(data=corr_df, x='correlation', ax=ax[0], bins=30, kde=True)
ax[0].set_xlabel('Correlation between total T7 and total CRE counts')
sns.scatterplot(data=corr_df, x='cell_counts', y='correlation', ax=ax[1], alpha=0.5)
ax[1].set_xlabel('Number of cells')
ax[1].set_xscale('log')
ax[1].set_ylabel('Correlation between total T7 and total CRE counts')
fig.savefig('results/expr3/fig3/T7_CRE_total_counts_correlation_per_celltype.pdf')

# %%
