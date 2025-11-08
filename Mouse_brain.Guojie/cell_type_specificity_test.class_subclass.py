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
import pickle
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
    PWD = '/gpfs/commons/groups/ren_lab/guojiezhong/starr-fish/Mouse_brain.Guojie'
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
    # add region label
    cluster_annotation_term = pd.read_excel('Data/abc_atlas/allen_institute_nominature.xlsx')
    adata.obs['region'] = cluster_annotation_term['neighborhood'].groupby(cluster_annotation_term['subclass_label']).first().reindex(adata.obs['subclass']).values
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
# starrfish3 = STARRFISH(adata3)
starrfish3_sec1 = STARRFISH.load('results/starrfish3_sec1.class_subclass.pkl')
starrfish3_sec2 = STARRFISH.load('results/starrfish3_sec2.class_subclass.pkl')
starrfish3 = STARRFISH.load('results/starrfish3.class_subclass.pkl')


# %% define the CREs and Cell Type matric to keep
infected_cells_threshold = 5
to_filter = (starrfish3.get_cre_expression() > 0).groupby(starrfish3.get_celltypes()).sum() < infected_cells_threshold
to_filter_sec1 = (starrfish3_sec1.get_cre_expression() > 0).groupby(starrfish3_sec1.get_celltypes()).sum() < infected_cells_threshold
to_filter_sec2 = (starrfish3_sec2.get_cre_expression() > 0).groupby(starrfish3_sec2.get_celltypes()).sum() < infected_cells_threshold
cre_blacklist = ['CRE061', 'CRE143', 'CRE001']
cre_whitelist = starrfish3_sec1.get_creinfo().index[~starrfish3_sec1.get_creinfo().index.isin(cre_blacklist)]
# read in the mismatching barcode CREs
mismatching_cres = pd.read_csv('Data/AAV_ONT_Barcode_Counts_vs_Mismatch_Percentage.csv', index_col=0)
cre_whitelist = cre_whitelist[~cre_whitelist.isin(mismatching_cres.index[mismatching_cres['MismatchPercent'] > 20])]
cre_blacklist = np.unique(cre_blacklist + mismatching_cres.index[mismatching_cres['MismatchPercent'] > 20].tolist()).tolist()
to_filter[cre_blacklist] = True
to_filter_sec1[cre_blacklist] = True
to_filter_sec2[cre_blacklist] = True


# %% load average fold change test results
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
    'n_jobs': 36,
}
starrfish3_sec1_avg = STARRFISH.load(f'{PWD}/results/starrfish3_sec1.bak.pkl')
starrfish3_sec2_avg = STARRFISH.load(f'{PWD}/results/starrfish3_sec2.bak.pkl')
starrfish3_avg = STARRFISH.load(f'{PWD}/results/starrfish3.bak.pkl')

res1_avg = starrfish3_sec1_avg.average_bootstrap_test(**average_bootstrap_test_config)
res2_avg = starrfish3_sec2_avg.average_bootstrap_test(**average_bootstrap_test_config)
res_avg = starrfish3_avg.average_bootstrap_test(**average_bootstrap_test_config)

del starrfish3_sec1_avg, starrfish3_sec2_avg, starrfish3_avg

# %% define classs for the test
classes = starrfish3.adata.obs['class'].str.split(';').explode().unique().tolist()
for class_name in classes:
    if os.path.exists(f'results/expr3/class_subclass/{class_name}/precision_recall_all.csv'):
        print(f'Skipping class {class_name} as results already exist.')
        continue
    print(f'Processing class: {class_name}')
    cell_types_to_use = starrfish3.get_celltypes()[starrfish3.adata.obs['class'].str.contains(class_name)].unique().tolist()
    fig = starrfish3.plot_cluster(cell_types_to_use)
    fig.savefig(f'results/expr3/class_subclass/class_{class_name}.pdf')
    # run the fold change test for T7
    fold_change_test_config = {
        "cell_types_to_use": cell_types_to_use,
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
        "n_jobs": 72,
        'load_stored': True,
    }
    # check if cell number of cell_types_to_use is larger than 0 for each of the cell type
    if starrfish3_sec1.get_celltypes().value_counts().reindex(cell_types_to_use).fillna(0).min() == 0:
        print(f'Skipping class {class_name} in section 1 due to insufficient cell numbers.')
        continue
    else:
        res1 = starrfish3_sec1.fold_change_test(**fold_change_test_config)
        # starrfish3_sec1.save('results/starrfish3_sec1.class_subclass.pkl')
    if starrfish3_sec2.get_celltypes().value_counts().reindex(cell_types_to_use).fillna(0).min() == 0:
        print(f'Skipping class {class_name} in section 2 due to insufficient cell numbers.')
        continue
    else:
        res2 = starrfish3_sec2.fold_change_test(**fold_change_test_config)
        # starrfish3_sec2.save('results/starrfish3_sec2.class_subclass.pkl')
    if starrfish3.get_celltypes().value_counts().reindex(cell_types_to_use).fillna(0).min() == 0:
        print(f'Skipping class {class_name} in combined due to insufficient cell numbers in one of the sections.')
        continue
    else:
        res = starrfish3.fold_change_test(**fold_change_test_config)
        # starrfish3.save('results/starrfish3.class_subclass.pkl')

    # %% check reproducibility of cell type specificity
    from plots import average_foldchange_specificity_test, q_value_correction
    p_mat_rank_test, p_mat_frequentist = average_foldchange_specificity_test(res_avg, res)
    # sec1 
    p_mat_rank_test1, p_mat_frequentist1 = average_foldchange_specificity_test(res1_avg, res1)
    # sec2
    p_mat_rank_test2, p_mat_frequentist2 = average_foldchange_specificity_test(res2_avg, res2)
    # %% q-value correction
    # filter out to filter
    infected_cells_threshold = 5
    to_filter = (starrfish3.get_cre_expression() > 0).groupby(starrfish3.get_celltypes()).sum() < infected_cells_threshold
    to_filter_sec1 = (starrfish3_sec1.get_cre_expression() > 0).groupby(starrfish3_sec1.get_celltypes()).sum() < infected_cells_threshold
    to_filter_sec2 = (starrfish3_sec2.get_cre_expression() > 0).groupby(starrfish3_sec2.get_celltypes()).sum() < infected_cells_threshold
    to_filter[cre_blacklist] = True
    to_filter_sec1[cre_blacklist] = True
    to_filter_sec2[cre_blacklist] = True
    
    # calculate q-values
    p_mat_rank_test_filter = p_mat_rank_test.copy()
    p_mat_rank_test_filter[to_filter] = np.nan
    q_mat_rank_test = q_value_correction(p_mat_rank_test_filter)

    p_mat_frequentist_filter = p_mat_frequentist.copy()
    p_mat_frequentist_filter[to_filter] = np.nan
    q_mat_frequentist = q_value_correction(p_mat_frequentist_filter)

    p_mat_rank_test1_filter = p_mat_rank_test1.copy()
    p_mat_rank_test1_filter[to_filter_sec1] = np.nan
    q_mat_rank_test1 = q_value_correction(p_mat_rank_test1_filter)

    p_mat_frequentist1_filter = p_mat_frequentist1.copy()
    p_mat_frequentist1_filter[to_filter_sec1] = np.nan
    q_mat_frequentist1 = q_value_correction(p_mat_frequentist1_filter)

    p_mat_rank_test2_filter = p_mat_rank_test2.copy()
    p_mat_rank_test2_filter[to_filter_sec2] = np.nan
    q_mat_rank_test2 = q_value_correction(p_mat_rank_test2_filter)

    p_mat_frequentist2_filter = p_mat_frequentist2.copy()
    p_mat_frequentist2_filter[to_filter_sec2] = np.nan
    q_mat_frequentist2 = q_value_correction(p_mat_frequentist2_filter)

    # %%
    q_res1 = q_mat_frequentist1.copy()
    q_res2 = q_mat_frequentist2.copy()
    q_res = q_mat_frequentist.copy()
    activity_res1 = res1_avg['celltype_activity'].loc[q_res1.index].copy()
    activity_res1[to_filter_sec1] = np.nan
    activity_res2 = res2_avg['celltype_activity'].loc[q_res2.index].copy()
    activity_res2[to_filter_sec2] = np.nan
    activity_res = res_avg['celltype_activity'].loc[q_res.index].copy()
    activity_res[to_filter] = np.nan
    # %% plot reproducibility
    from plots import plot_q_value_cre_reproducibility
    res_compare = plot_q_value_cre_reproducibility(q_res1, q_res2, q_res, starrfish3.lib_size, 0.05)
    # define reproducible CREs
    reproducible_cres = res_compare.index[(res_compare[['Common', 'Common_sec1', 'Common_sec2']] > 0).any(axis=1)]
    fig, ax = plt.subplots(ncols=3, figsize=(12, 4))
    # Determine common size and hue ranges
    size_cols = ['Common', 'Common_sec1', 'Common_sec2']
    hue_cols = ['Percentage', 'Percentage_sec1', 'Percentage_sec2']
    size_range = (res_compare[size_cols].min().min(), res_compare[size_cols].max().max())
    hue_range = (res_compare[hue_cols].min().min(), res_compare[hue_cols].max().max())

    sns.scatterplot(data=res_compare, x='Sec1', y='Sec2', size='Common', hue='Percentage',
                    palette='coolwarm', ax=ax[0], alpha=0.7, legend=False,
                    sizes=(20, 200), size_norm=size_range, hue_norm=hue_range)
    sns.scatterplot(data=res_compare, x='Sec1', y='All', size='Common_sec1', hue='Percentage_sec1',
                    palette='coolwarm', ax=ax[1], alpha=0.7, legend=False,
                    sizes=(20, 200), size_norm=size_range, hue_norm=hue_range)
    sns.scatterplot(data=res_compare, x='Sec2', y='All', size='Common_sec2', hue='Percentage_sec2',
                    palette='coolwarm', ax=ax[2], alpha=0.7,
                    sizes=(20, 200), size_norm=size_range, hue_norm=hue_range)

    # Customize legend titles
    handles, labels = ax[2].get_legend_handles_labels()
    legend = ax[2].legend(handles, labels, bbox_to_anchor=(1.05, 1), loc='upper left')
    legend.set_title('')
    for text in legend.get_texts():
        if text.get_text() in [str(i) for i in range(10)]:  # size legend items
            legend.get_texts()[legend.get_texts().index(text)].set_text('')
    os.makedirs(f'results/expr3/class_subclass/{class_name}/', exist_ok=True)
    fig.savefig(f'results/expr3/class_subclass/{class_name}/reproducibility_significant_scatter.pdf', bbox_inches='tight')
    # %% dot plot of some biology
    from plots import cre_pval_dotplot
    negative_control_cres = starrfish3.get_negative_control_cres()
    cre_info = starrfish3.get_creinfo().copy()
    cre_info['best_subclass'] = 'CRE'
    cre_info.loc[negative_control_cres, 'best_subclass'] = 'Negative Control'
    # plot
    celltypes_to_use = q_res.index.copy()
    cres_to_use = q_res.columns[np.nanmin(q_res.loc[celltypes_to_use], axis=0) < 0.05].union(starrfish3.get_negative_control_cres())
    cres_to_use = cres_to_use[~cres_to_use.isin(cre_blacklist)]
    fig, cre_orders = cre_pval_dotplot(q_res, activity_res, cres_to_use, celltypes_to_use, cre_info, reorder_cres=True, figsize=(15, 15))
    fig.savefig(f'results/expr3/class_subclass/{class_name}/dotplot_all_significant.pdf', bbox_inches='tight')
    # %% visualize some example
    have_target_cres = q_res.columns[np.nanmin(q_res.loc[celltypes_to_use], axis=0) < 0.05]
    os.makedirs(f'results/expr3/class_subclass/{class_name}/cre_significant_celltypes/', exist_ok=True)
    for cre in have_target_cres:
        if os.path.exists(f'results/expr3/class_subclass/{class_name}/cre_significant_celltypes/{cre}.pdf'):
            continue
        print(f'Plotting {cre} ... in {class_name}')
        # rank by q-value
        cre_q_values = q_res.loc[celltypes_to_use, cre]
        cre_q_values = cre_q_values[cre_q_values <= 0.05] 
        # order by rank
        cre_q_values = cre_q_values.sort_values(ascending=True)
        cell_types_to_visualize = cre_q_values.index
        fig = starrfish3.plot_gene(
            cre, average_by_celltype=False, # if true, all cells from same cell type will have same value
            cell_types_to_visualize=cell_types_to_visualize, # only visualize some cell types
            scale_size_by='counts', # scale size by "counts": normalized counts; or "celltype_number": number of cells in the cell type
            log=True, transpose=-1, flipx=-1, sz_max=50,
            cell_types_to_use=celltypes_to_use)
        fig.savefig(f'results/expr3/class_subclass/{class_name}/cre_significant_celltypes/{cre}.pdf')
    # %% prepare a data frame
    q_res_df = pd.DataFrame(index=q_res.columns, columns=['significant_classes', 'num_significant_classes', 'max_activity_class'])
    for cre in q_res.columns:
        sig_classes = q_res.index[q_res[cre] < 0.05].tolist()
        q_res_df.loc[cre, 'significant_classes'] = ' | '.join(sig_classes)
        q_res_df.loc[cre, 'num_significant_classes'] = len(sig_classes)
        if len(sig_classes) > 0:
            max_activity_class = activity_res.loc[sig_classes, cre].idxmax()
            q_res_df.loc[cre, 'max_activity_class'] = max_activity_class
        else:
            q_res_df.loc[cre, 'max_activity_class'] = ''
    q_res_df.to_csv(f'results/expr3/class_subclass/{class_name}/cre_significant_subclasses.csv')
    q_res.to_csv(f'results/expr3/class_subclass/{class_name}/cre_significant_subclasses_qvalues.csv')
    activity_res.to_csv(f'results/expr3/class_subclass/{class_name}/cre_significant_subclasses_activities.csv')
    # %% filter to reproducible cres
    # %% get precision recall in that class
    from plots import get_pr_df, plot_bar
    # %% get precision for ATAC and histone modifications
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

    # %% get precision recall
    pr_df1 = get_pr_df(qvalue_df=q_res.loc[cell_types_to_use, reproducible_cres].copy(), cell_types_to_use=pd.Series(cell_types_to_use),
                    starrfish_obj=starrfish3, metric = ['atac_cpm', 'h3k4me1_cpm', 'h3k27ac_cpm'], z_cutoffs=[2.0])
    pr_df2 = get_pr_df(qvalue_df=q_res.loc[cell_types_to_use, reproducible_cres].copy(), cell_types_to_use=pd.Series(cell_types_to_use),
                    starrfish_obj=starrfish3, 
                    metric=['chromatin_o', 'chromatin_a'], z_cutoffs=[0.5])
    pr_df2 = pr_df2.sort_values(by=['cell_type_rank']).reset_index(drop=True)
    pr_df1 = pr_df1[pr_df1['cell_type'].isin(pr_df2['cell_type'])].copy()
    pr_df2 = pr_df2[pr_df2['cell_type'].isin(pr_df1['cell_type'])].copy()
    df_bar = pr_df1[(pr_df1['z_cutoff'] == 2.0)].copy()
    # add a column for overall precision
    df_bar_all1 = pd.DataFrame({'cell_type': 'ALL',
        'precision': df_bar.groupby(['mod'])['correct'].sum() / df_bar.groupby(['mod'])['all_pred'].sum(),
        'correct': df_bar.groupby(['mod'])['correct'].sum(),
        'all_pred': df_bar.groupby(['mod'])['all_pred'].sum(),
        'target': df_bar.groupby(['mod'])['target'].sum(),
    })
    df_bar_all1['recall'] = df_bar_all1['correct'].astype(str) + '/' + df_bar_all1['all_pred'].astype(str)
    df_bar_all1['mod'] = df_bar_all1.index
    df_bar = df_bar[df_bar['target'] >= 2].copy()
    # fig, ax = plot_bar(df_bar, legend_loc=(0.95, 0.75), figsize=(6, 6), flip_axis=True, fontsize=6)
    df_bar = pr_df2.copy()
    df_bar_all2 = pd.DataFrame({'cell_type': 'ALL',
        'precision': df_bar.groupby(['mod'])['correct'].sum() / df_bar.groupby(['mod'])['all_pred'].sum(),
        'correct': df_bar.groupby(['mod'])['correct'].sum(),
        'all_pred': df_bar.groupby(['mod'])['all_pred'].sum(),
        'target': df_bar.groupby(['mod'])['target'].sum(),
    })
    df_bar_all2['recall'] = df_bar_all2['correct'].astype(str) + '/' + df_bar_all2['all_pred'].astype(str)
    df_bar = df_bar[df_bar['target'] >= 2].copy()
    df_bar_all2['mod'] = df_bar_all2.index
    # fig, ax = plot_bar(df_bar, figsize=(6, 6), flip_axis=True, fontsize=6)
    # ALL cell types
    df_bar_all = pd.concat([df_bar_all1, df_bar_all2], axis=0, ignore_index=True)
    fig, ax = plot_bar(df_bar_all, figsize=(6, 6), flip_axis=True, fontsize=6)
    df_bar_all.to_csv(f'results/expr3/class_subclass/{class_name}/precision_recall_all.csv')
    fig.savefig(f'results/expr3/class_subclass/{class_name}/precision_recall_all.pdf')

# %% get the list of CREs of interest in each class
class_df = pd.read_csv('results/expr3/class/cre_significant_classes.csv', index_col=0)
for class_name in classes:
    q_res_df = pd.read_csv(f'results/expr3/class_subclass/{class_name}/cre_significant_subclasses.csv', index_col=0)
    # check if the cre is significant in the class
    cres_of_interest = class_df.loc[q_res_df.index[q_res_df['significant_classes'].notna()]]
    cres_of_interest = cres_of_interest[cres_of_interest['significant_classes'].notna()]
    cres_of_interest = cres_of_interest.index[cres_of_interest['significant_classes'].str.contains(class_name)]
    cres_of_interest = cres_of_interest[~cres_of_interest.isin(cre_blacklist)]
    cres_of_interest.to_series().to_csv(f'results/expr3/class_subclass/{class_name}/{class_name}_cres_of_interest.csv')
# %%
# %% merge different region precision recall df together
for class_name in classes:
    if not os.path.exists(f'results/expr3/class_subclass/{class_name}/precision_recall_all.csv'):
        print(f'Skipping class {class_name} as results do not exist.')
        continue
    df_bar_all = pd.read_csv(f'results/expr3/class_subclass/{class_name}/precision_recall_all.csv', index_col=0)
    df_bar_all['class'] = class_name
    if class_name == classes[0]:
        df_bar_all_merged = df_bar_all.copy()
    else:
        df_bar_all_merged = pd.concat([df_bar_all_merged, df_bar_all], axis=0, ignore_index=True)
fig, ax = plt.subplots(figsize=(8, 6))
df_bar_all_merged = df_bar_all_merged.fillna(0)
sns.barplot(data=df_bar_all_merged, x='class', y='precision', hue='mod', ax=ax)
# Add text annotations showing correct/all_pred on each bar
# Create a lookup dictionary for the data
data_lookup = {}
for _, row in df_bar_all_merged.iterrows():
    key = (row['class'], row['mod'])
    data_lookup[key] = row

# Get the order of x and hue from the plot
classes_order = [tick.get_text() for tick in ax.get_xticklabels()]
mods_order = df_bar_all_merged['mod'].unique()

# Annotate patches
patch_idx = 0
for mod in mods_order:
    for class_name in classes:
        if patch_idx < len(ax.patches):
            patch = ax.patches[patch_idx]
            height = patch.get_height()
            x = patch.get_x() + patch.get_width() / 2
            if (class_name, mod) in data_lookup:
                row = data_lookup[(class_name, mod)]
                label = f"{int(row['correct'])}/{int(row['all_pred'])}"
                ax.text(x, height, label, ha='center', va='bottom', fontsize=6)
            patch_idx += 1
ax.set_ylabel('Precision')
ax.set_xlabel('Region')
ax.legend(title='Modality', bbox_to_anchor=(1.05, 1), loc='upper left')
# rotate x labels
plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
fig.tight_layout()
fig.savefig('results/expr3/class_subclass/precision_recall_all_classes.pdf', bbox_inches='tight')
# %%
