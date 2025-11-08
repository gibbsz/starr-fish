# %%
import abc
import pandas as pd
import numpy as np
from pathlib import Path
import pyBigWig
import pyranges as pr
import os
import matplotlib.pyplot as plt
import seaborn as sns
PWD = '/gpfs/commons/groups/ren_lab/guojiezhong/starr-fish/Mouse_brain.Guojie'
os.chdir(PWD)

# %% get abc specificity
abc_aav = pd.read_csv('Data/abc_atlas/AAV/Enhancer_AAVs(in).csv')
abc_aav.set_index('Enhancer ID', inplace=True)
# drop nan Coordinates
abc_aav = abc_aav[~abc_aav['Coordinates'].isna()]
# drop duplicated Coordinates
abc_aav = abc_aav[~abc_aav['Coordinates'].duplicated(keep='first')]
cres = abc_aav.index.unique().tolist()
# %% overlap with our cres
import pyranges as pr
our_cres = pd.read_csv('Data/CRE.bed', sep='\t', header=None)
our_cres.columns = ['Chromosome', 'Start', 'End', 'label_type', 'name']
our_cres = pr.PyRanges(our_cres)
abc_aav_bed = abc_aav['Coordinates'].str.split(':', expand=True)
abc_aav_bed.columns = ['Chromosome', 'Position']
abc_aav_bed[['Start', 'End']] = abc_aav_bed['Position'].str.split('-', expand=True).astype(int)
abc_aav_bed['ID'] = abc_aav.index.copy()
abc_aav_bed_gr = pr.PyRanges(abc_aav_bed)
# intersection
our_cres_abc = abc_aav_bed_gr.join(our_cres)
# write to file
our_cres_abc.df.to_csv('Data/abc_atlas/AAV/Enhancer_AAVs_in_ourCREs.bed', sep='\t', index=False)


# %%
# calculate the annotations for abc aavs
# cluster names
cluster_annotation_term = pd.read_csv(f'{PWD}/Data/abc_atlas/cluster_annotation_term.csv', index_col=0)
cluster_annotation_term['subclass'] = cluster_annotation_term['subclass'].str.replace('/', '-')
# load bigwig files
for bigwig_path, modality in zip([f'{PWD}/Data/ATAC/snATACbw_bamCoverage/', f'{PWD}/Data/Histone/DNAbw/'], 
                                 [['ATAC'], ['H3K27ac', 'H3K9me3', 'H3K4me1', 'H3K27me3']]):
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
                # get the cell type name from cluster_annotation_term
                subclass_number = int(f.split('_')[0])
                if subclass_number in cluster_annotation_term['subclass_number'].values:
                    celltype = cluster_annotation_term.loc[cluster_annotation_term['subclass_number'] == subclass_number, 'subclass'].values[0]
                    celltypes.append(celltype)
        histone_df = pd.DataFrame(index=cres, columns=celltypes)
        for f in bigwig_files:
            if f.endswith(pattern):
                print(f)
                # get the cell type name from cluster_annotation_term
                subclass_number = int(f.split('_')[0])
                if subclass_number in cluster_annotation_term['subclass_number'].values:
                    celltype = cluster_annotation_term.loc[cluster_annotation_term['subclass_number'] == subclass_number, 'subclass'].values[0]
                    bw = pyBigWig.open(bigwig_path + f)
                    for i in cres:
                        try:
                            histone_df.loc[i, celltype] = bw.stats(abc_aav.loc[i, 'Coordinates'].split(':')[0], 
                                                                int(abc_aav.loc[i, 'Coordinates'].split(':')[1].split('-')[0])-500, 
                                                                int(abc_aav.loc[i, 'Coordinates'].split(':')[1].split('-')[1])+500, type='sum')[0] / bw.header()['sumData'] * 1e5
                        except Exception as e:
                            print(f"Error processing {celltype} for {i}: {e}")
                    bw.close()
        histone_df.to_csv(f'{PWD}/Data/abc_atlas/AAV/{mod}_cpm_peak_pad_500_Bysubclass.csv')

# %% create chromatin state files
# Load chromatin state data
chromstate = pd.read_csv('Data/allCRE.amb.PairedTag.annot.tsv', sep='\t')
chromstate_a = chromstate[chromstate['chromHMMState'] == 'Chr-A']
chromstate_o = chromstate[chromstate['chromHMMState'] == 'Chr-O']

# Prepare ABC CRE info using existing cres variable
abc_creinfo = abc_aav.loc[cres, ['Coordinates']].copy()
abc_creinfo['Chromosome'] = abc_creinfo['Coordinates'].str.split(':').str[0]
abc_creinfo['Start'] = abc_creinfo['Coordinates'].str.split(':').str[1].str.split('-').str[0].astype(int)
abc_creinfo['End'] = abc_creinfo['Coordinates'].str.split(':').str[1].str.split('-').str[1].astype(int)
abc_creinfo['CRE_ID'] = abc_creinfo.index  # keep track of original rows
abc_cre_gr = pr.PyRanges(abc_creinfo[['Chromosome', 'Start', 'End', 'CRE_ID']])

# Output containers
celltypes = chromstate['subclass'].unique()
cre_chrom_a = pd.DataFrame(index=cres, columns=celltypes)
cre_chrom_o = pd.DataFrame(index=cres, columns=celltypes)

# Loop over cell types
print("Processing chromatin states for ABC AAV CREs...")
for celltype in celltypes:
    for chrom_label, chrom_df, target_df in zip(
        ['a', 'o'], [chromstate_a, chromstate_o], [cre_chrom_a, cre_chrom_o]
    ):
        subset = chrom_df[chrom_df['subclass'] == celltype].copy()
        if len(subset) == 0:
            continue
        subset = subset[['chrom', 'startFrom', 'endTo']].copy()
        subset.columns = ['Chromosome', 'Start', 'End']
        subset['Start'] = subset['Start'].astype(int)
        subset['End'] = subset['End'].astype(int)
        subset_gr = pr.PyRanges(subset)

        # Perform intersection
        intersection = abc_cre_gr.join(subset_gr)
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
        target_df[celltype] = best_overlap.reindex(abc_creinfo['CRE_ID']).fillna(0).values

# Fill na with 0
cre_chrom_a = cre_chrom_a.fillna(0)
cre_chrom_o = cre_chrom_o.fillna(0)

# Change column names to match cluster annotation
cluster_annotation_term = pd.read_csv('Data/abc_atlas/cluster_annotation_term.csv', index_col=0)
cluster_annotation_term['subclass'] = cluster_annotation_term['subclass'].str.replace('/', '-')
clm_idx = [int(col[:3]) for col in celltypes]
cre_chrom_a.columns = cluster_annotation_term['subclass'].groupby(cluster_annotation_term['subclass_number']).first().loc[clm_idx].values
cre_chrom_o.columns = cluster_annotation_term['subclass'].groupby(cluster_annotation_term['subclass_number']).first().loc[clm_idx].values

# Save the results
cre_chrom_a.T.to_csv(f'{PWD}/Data/abc_atlas/AAV/chromatin_a_Bysubclass.csv')
cre_chrom_o.T.to_csv(f'{PWD}/Data/abc_atlas/AAV/chromatin_o_Bysubclass.csv')
print(f"Chromatin state files saved: chromatin_a and chromatin_o")

# %% drop invalid cre
atac_df = pd.read_csv(f'{PWD}/Data/abc_atlas/AAV/ATAC_cpm_peak_pad_500_Bysubclass.csv', index_col=0)
valid_cres = atac_df.index[~atac_df.isna().all(axis=1)].tolist()
abc_aav = pd.read_csv('Data/abc_atlas/AAV/Enhancer_AAVs(in).csv')
abc_aav = abc_aav[abc_aav['Enhancer ID'].isin(valid_cres)]
# %% get cortex cell types as that's what allen only used
cortex_celltypes = atac_df.columns[
    atac_df.columns.str.contains(r'^L[1-6].*|^Lamp5|^Sncg|^Vip Gaba|^Sst Gaba|^Pvalb Gaba|^Pvalb chandelier Gaba', regex=True, na=False)
].unique()

# %%
def get_pr_df(qvalue_df, cell_types_to_use,
              metric = ['ATAC_cpm', 'H3K4me1_cpm', 'H3K9me3_cpm', 'H3K27ac_cpm', 'H3K27me3_cpm'],
              z_cutoffs=np.arange(0, 5, 0.1), q_threshold=0.05):
    res_df = pd.DataFrame()
    # filter cell_types_to_use based on mod
    for mod in metric:
        if mod.endswith('_cpm'):
            file_path = f'{PWD}/Data/abc_atlas/AAV/{mod}_peak_pad_500_Bysubclass.csv'
            mod_data = pd.read_csv(file_path, index_col=0).T
        else:  # chromatin_a or chromatin_o
            file_path = f'{PWD}/Data/abc_atlas/AAV/{mod}_Bysubclass.csv'
            mod_data = pd.read_csv(file_path, index_col=0)
        cell_types_to_use = cell_types_to_use[cell_types_to_use.isin(mod_data.index)]
    # for each CRE, select top rank cell type
    for z in z_cutoffs:
        for mod in metric:
            if mod.endswith('_cpm'):
                # Load CPM data with padding
                mod_cpm = pd.read_csv(f'{PWD}/Data/abc_atlas/AAV/{mod}_peak_pad_500_Bysubclass.csv', index_col=0).T
                qvalue_df = qvalue_df[qvalue_df.columns.intersection(mod_cpm.columns)]
                mod_cpm = mod_cpm.loc[qvalue_df.index.intersection(mod_cpm.index), qvalue_df.columns]
                # log transform
                mod_cpm = np.log1p(mod_cpm.astype(float))
                mod_cpm_z = mod_cpm.sub(mod_cpm.mean(axis=0), axis=1).div(mod_cpm.std(axis=0), axis=1)  # Z-score per CRE
            else:  # chromatin_a or chromatin_o
                # Load chromatin state overlap ratios (already normalized 0-1)
                mod_chromatin = pd.read_csv(f'{PWD}/Data/abc_atlas/AAV/{mod}_Bysubclass.csv', index_col=0)
                qvalue_df = qvalue_df[qvalue_df.columns.intersection(mod_chromatin.columns)]
                mod_chromatin = mod_chromatin.loc[qvalue_df.index.intersection(mod_chromatin.index), qvalue_df.columns]
                # Z-score normalize the chromatin overlap ratios
                mod_chromatin = mod_chromatin.astype(float)
                mod_cpm_z = mod_chromatin.sub(mod_chromatin.mean(axis=0), axis=1).div(mod_chromatin.std(axis=0), axis=1)
            for cell_type in cell_types_to_use:
                target_cres = qvalue_df.loc[cell_type].index[qvalue_df.loc[cell_type] <= q_threshold]
                z_score = mod_cpm_z.loc[cell_type]
                if mod in ['H3K9me3_cpm', 'H3K27me3_cpm']:
                    z_score = -z_score
                pred_cres = z_score.index[z_score >= z]
                # on-target and off-target rates
                correct = target_cres.isin(pred_cres).sum()
                all_pred = len(pred_cres)
                res_df = pd.concat((res_df,
                pd.DataFrame({
                    'cell_type': cell_type,
                    'mod': mod.replace('_cpm', ''),
                    'z_cutoff': z,
                    'precision': correct / all_pred if all_pred > 0 else 0,
                    'recall': f'{correct}/{all_pred}' if all_pred > 0 else '0/0',
                    'all_pred': all_pred,
                    'correct': correct,
                    'target': len(target_cres),
                }, index=[0])), ignore_index=True)
    # drop NaN values
    res_df = res_df.dropna(subset=['precision', 'recall'])
    # order by allen institute's nominature
    cluster_annotation_term = pd.read_csv('Data/abc_atlas/cluster_annotation_term.csv', index_col=0)
    cluster_annotation_term['subclass'] = cluster_annotation_term['subclass'].str.replace('/', '-')
    try:
        res_df['cell_type_rank'] = cluster_annotation_term['subclass_number'].groupby(cluster_annotation_term['subclass']).first().loc[res_df['cell_type']].values
    except KeyError:
        try:
            res_df['cell_type_rank'] = cluster_annotation_term['class_number'].groupby(cluster_annotation_term['class']).first().loc[res_df['cell_type']].values
        except KeyError:
            # alphabetical order
            res_df['cell_type_rank'] = pd.Categorical(res_df['cell_type']).codes
    # reorder by cell type rank
    res_df = res_df.sort_values(by=['cell_type_rank']).reset_index(drop=True)
    return res_df

# %%
def plot_precision_vs_zscore(pr_df, figsize=(8, 6), save_path=None,
                             title='Precision vs Z-score Cutoff',
                             use_lowess=True):
    """
    Plot precision with regard to z-score cutoff.

    Parameters:
    -----------
    pr_df : pd.DataFrame
        Precision-recall dataframe from get_pr_df() with columns:
        cell_type, mod, z_cutoff, precision, recall, all_pred, correct, target
    figsize : tuple
        Figure size (width, height)
    save_path : str, optional
        Path to save the figure. If None, figure is not saved.
    title : str
        Title for the plot
    use_lowess : bool
        Whether to use lowess smoothing (locally weighted regression)

    Returns:
    --------
    fig : matplotlib.figure.Figure
        The figure object
    """
    # Calculate overall precision by grouping by mod and z_cutoff
    overall_precision = pr_df.groupby(['mod', 'z_cutoff'])['correct'].sum() / \
                        pr_df.groupby(['mod', 'z_cutoff'])['all_pred'].sum()

    # Create dataframe for plotting
    toplot = pd.DataFrame(overall_precision).reset_index()
    toplot.rename(columns={0: 'precision'}, inplace=True)

    # Create the plot
    g = sns.lmplot(data=toplot, x='z_cutoff', y='precision', hue='mod',
                   scatter=True, lowess=use_lowess, height=figsize[1],
                   aspect=figsize[0]/figsize[1], legend=True)

    # Customize plot
    g.set_axis_labels("Z-score Cutoff", "Precision")
    g.fig.suptitle(title, y=1.02)

    # Adjust layout
    plt.tight_layout()

    # Save if path provided
    if save_path:
        g.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Figure saved to: {save_path}")

    return g.fig

# %%
# create a fake qvalue df for testing
abc_fake_qvalue = pd.DataFrame(index=np.unique(valid_cres), columns=cortex_celltypes)
for cre in abc_fake_qvalue.index:
    abc_targets = abc_aav.loc[abc_aav['Enhancer ID'] == cre, 'Subclass'].unique()
    for celltype in abc_targets:
        if pd.isna(celltype):
            continue
        for ct in cortex_celltypes:
            if celltype.replace('_', ' ') in ct:
                abc_fake_qvalue.loc[cre, ct] = 0.00  # significant
            else:
                abc_fake_qvalue.loc[cre, ct] = 1.00  # not significant
# fill nan with 1.0
abc_fake_qvalue = abc_fake_qvalue.fillna(1.0)
# %% get precision recall
from plots import plot_bar
pr_df1 = get_pr_df(qvalue_df=abc_fake_qvalue.T.copy(), cell_types_to_use=cortex_celltypes, 
                   metric = ['ATAC_cpm', 'H3K4me1_cpm', 'H3K27ac_cpm'], z_cutoffs=[2.0])
pr_df2 = get_pr_df(qvalue_df=abc_fake_qvalue.T.copy(), cell_types_to_use=cortex_celltypes, 
                   metric=['chromatin_o', 'chromatin_a'], z_cutoffs=[0.5])
pr_df2 = pr_df2.sort_values(by=['cell_type_rank']).reset_index(drop=True)
pr_df1 = pr_df1[pr_df1['cell_type'].isin(pr_df2['cell_type'])].copy()
pr_df2 = pr_df2[pr_df2['cell_type'].isin(pr_df1['cell_type'])].copy()
pr_df1['mod'] = pr_df1['mod'].str.lower()
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
fig, ax = plot_bar(df_bar, legend_loc=(0.95, 0.75), figsize=(6, 6), flip_axis=True, fontsize=6)
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
fig, ax = plot_bar(df_bar, figsize=(6, 6), flip_axis=True, fontsize=6)
# ALL cell type
df_bar_all = pd.concat([df_bar_all1, df_bar_all2], axis=0, ignore_index=True)
fig, ax = plot_bar(df_bar_all, figsize=(6, 6), flip_axis=True, fontsize=6)
fig.savefig('results/expr3/precision_recall_all.pdf')

# %% Plot precision vs z-score cutoff
# Example: Run get_pr_df with multiple z_cutoffs to generate data
pr_df_zscore = get_pr_df(qvalue_df=abc_fake_qvalue.T.copy(),
                         cell_types_to_use=cortex_celltypes,
                         metric=['ATAC_cpm', 'H3K4me1_cpm', 'H3K27ac_cpm'],
                         z_cutoffs=np.arange(0.5, 4.5, 0.5))
#
# # Plot precision vs z-score
fig = plot_precision_vs_zscore(pr_df_zscore,
                               figsize=(4, 4),
                               save_path='results/expr3/precision_vs_zscore.pdf',
                               title='ABC Atlas: Precision vs Z-score Cutoff')

# %%
