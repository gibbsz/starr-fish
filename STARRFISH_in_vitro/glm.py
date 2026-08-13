# %%
import statsmodels.api as sm
import statsmodels.formula.api as smf
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy
import warnings
warnings.filterwarnings('ignore')
# %%
# read in data
def preprocess_experiment(enhancer_file, vector_file=None, nanopore_file=None, rna_seq_file=None, dna_seq_file=None, bed_file=None):
    enhancer = pd.read_csv(enhancer_file)
    if vector_file is None:
        # use enhancer as vector as a place holder, so that I don't have to change the code
        vector = enhancer.copy()
    else:
        vector = pd.read_csv(vector_file)
    if 'masks' in enhancer.columns and 'fov' in enhancer.columns:
        enhancer.index = enhancer['masks'].astype(str) + '_' + enhancer['fov'].astype(str)
        vector.index = vector['masks'].astype(str) + '_' + vector['fov'].astype(str)
    else:
        # infer mask and fov from the index
        enhancer_idx_str = enhancer.index.astype(str)
        vector_idx_str = vector.index.astype(str)
        if enhancer_idx_str.str.contains('-').any():
            enhancer['fov'] = enhancer_idx_str.str.split('-').str[0]
            enhancer['masks'] = enhancer_idx_str.str.split('-').str[1]
        else:
            enhancer['fov'] = enhancer_idx_str
            enhancer['masks'] = enhancer_idx_str
        if vector_idx_str.str.contains('-').any():
            vector['fov'] = vector_idx_str.str.split('-').str[0]
            vector['masks'] = vector_idx_str.str.split('-').str[1]
        else:
            vector['fov'] = vector_idx_str
            vector['masks'] = vector_idx_str
    # find common index and filter
    common_index = enhancer.index.intersection(vector.index)
    enhancer = enhancer.loc[common_index]
    vector = vector.loc[common_index]
    # drop mask and fov columns
    enhancer_drop = enhancer.copy().drop(columns=['masks', 'fov'])
    vector_drop = vector.copy().drop(columns=['masks', 'fov'])
    # check if total transcripts column exists
    if 'total transcripts' not in enhancer_drop.columns:
        enhancer_drop['total transcripts'] = enhancer_drop.sum(axis=1)
        enhancer['total transcripts'] = enhancer_drop['total transcripts']
    if 'total transcripts' not in vector_drop.columns:
        vector_drop['total transcripts'] = vector_drop.sum(axis=1)
        vector['total transcripts'] = vector_drop['total transcripts']
    result = {"enhancer": enhancer_drop, "vector": vector_drop,
              "enhancer_orig": enhancer, "vector_orig": vector}
    result['enhancer_orig'].rename(columns={'total transcripts': 'total_transcripts'}, inplace=True)
    result['vector_orig'].rename(columns={'total transcripts': 'total_transcripts'}, inplace=True)
    # if rna_seq_file and dna_seq_file are provided, add them to the result
    if nanopore_file is not None:
        nanopore = pd.read_csv(nanopore_file, sep=' ', skipinitialspace=True, header=None)
        nanopore.set_index(1, inplace=True)
        # fullfill the index to CRE names
        cre_names = enhancer_drop.columns[enhancer_drop.columns != 'total transcripts']
        result['nanopore'] = nanopore.reindex(cre_names, fill_value=0)
    if rna_seq_file is not None:
        rna_counts = pd.read_csv(rna_seq_file, sep='\t', skipinitialspace=True, skiprows=1)
        rna_counts.set_index('Geneid', inplace=True)
        result['rna_counts'] = rna_counts.iloc[:, -1]
    if dna_seq_file is not None:
        dna_counts = pd.read_csv(dna_seq_file, sep='\t', skipinitialspace=True, skiprows=1)
        dna_counts.set_index('Geneid', inplace=True)
        result['dna_counts'] = dna_counts.iloc[:, -1]
    if bed_file is not None:
        ccre_names = pd.read_csv(bed_file, header=None, sep='\t')
        ccre_names = ccre_names.astype(str)
        ccre_names = (ccre_names[0] + ":" + ccre_names[1] + "-" + ccre_names[2])
        # set index
        ccre_names = pd.DataFrame(ccre_names.values, index=['CRE' + str(i+1).zfill(3) for i in range(len(ccre_names))])
        result['ccre_names'] = ccre_names
    return result

# glm fit on single experiment, fit enhancer ~ T7
# if norm_by_total, then fit enhancer / total_enhancer ~ vector / total_vector
def glm_fit(experiment, family=sm.families.Gaussian(), use_fov_covariate=True, norm_by_total=False):
    # Create the model
    enhancer = experiment["enhancer"]
    vector = experiment["vector"]
    result = pd.DataFrame()
    if norm_by_total:
        enhancer = enhancer.div(enhancer.iloc[:, -1], axis=0)
        vector = vector.div(vector.iloc[:, -1], axis=0)
    for cre in enhancer.columns[:-1]:
        if cre not in vector.columns:
            raise ValueError(f"CRE {cre} not found in vector data.")
        fit_data=pd.DataFrame({'y': enhancer[cre].values, 
                               'x': vector[cre].values,
                               'fov': experiment["enhancer_orig"]['fov'].values})
        if not use_fov_covariate:
            glm_results = smf.ols('y ~ x', fit_data, family=family).fit()
        else:
            glm_results = smf.ols('y ~ x + C(fov)', fit_data, family=family).fit()
        # Fit the model
        glm_summary = pd.read_html(glm_results.summary().tables[1].as_html(), header=0, index_col=0)[0]
        glm_summary = pd.DataFrame(glm_summary.loc['x']).T
        glm_summary.index = [cre]
        result = pd.concat([result, glm_summary], axis=0)
    # do glm fit on poshen data
    poshen = pd.read_excel("data/Poshen_Table_S3.xlsx", sheet_name = "raw counts HCT116")
    poshen.set_index('oligo', inplace=True)
    poshen_deseq2 = pd.read_csv("data/HCT116_activating_oligo.csv")
    poshen_deseq2.index = poshen_deseq2['chr'].astype(str) + ":" + poshen_deseq2['start'].astype(int).astype(str) + "-" + poshen_deseq2['end'].astype(int).astype(str)
    ccre_names = experiment['ccre_names']
    # filter for ccre_names in poshen
    ccre_names = ccre_names.loc[ccre_names[0].isin(poshen.index)]
    poshen = poshen.reindex(ccre_names[0], fill_value=np.nan)
    poshen_deseq2 = poshen_deseq2.reindex(ccre_names[0], fill_value=np.nan)
    poshen.index = ccre_names.index
    poshen_deseq2.index = ccre_names.index
    # poshen fill missing values with NA
    poshen = poshen.reindex(result.index, fill_value=np.nan)
    poshen_deseq2 = poshen_deseq2.reindex(result.index, fill_value=np.nan)
    result['Poshen Activity'] = poshen['Activity'].values
    result['Poshen Activity DESeq2'] = poshen_deseq2['lfc'].values
    if 'rna_counts' in experiment and 'dna_counts' in experiment:
        result['RNA/DNA'] = experiment['rna_counts'].values / experiment['dna_counts'].values
        result['RNA/DNA lfc'] = np.log2(result['RNA/DNA'])
    # rename result
    result = result.rename(columns={'coef': 'STARR-FISH Activity', 'std err': 'STARR-FISH Activity std err'})
    result['log(STARR-FISH Activity)'] = np.log(result['STARR-FISH Activity'])
    experiment['glm_fit_result'] = result
    return experiment

# glm fit on single experiment, fit enhancer ~ total_enhancer, T7 ~ total_T7, 
# then if norm_by_vector, divide the coefficients: enhancer ~ total_enhancer / vector ~ total_vector
# if norm_by_nanopore, divide the enhancer coefficients by log(nanopore counts): enhancer ~ total_enhancer / log(nanopore counts)
def glm_fit_total(experiment, family=sm.families.Gaussian(), norm_by_vector=True, norm_by_nanopore=True, use_fov_covariate=False, key_add='glm_fit_total_result'):
    # Create the model
    enhancer = experiment["enhancer"]
    vector = experiment["vector"]
    enh_result = pd.DataFrame()
    vec_result = pd.DataFrame()
    for cre in enhancer.columns[:-1]:
        if cre not in vector.columns:
            raise ValueError(f"CRE {cre} not found in vector data.")
        fit_data=pd.DataFrame({'y': enhancer[cre].values, 
                               'x': enhancer['total transcripts'].values,
                               'fov': experiment["enhancer_orig"]['fov'].values})
        if not use_fov_covariate:
            glm_results = smf.ols('y ~ x', fit_data, family=family).fit()
        else:
            glm_results = smf.ols('y ~ x + C(fov)', fit_data, family=family).fit()
        # Fit the model
        glm_summary = pd.read_html(glm_results.summary().tables[1].as_html(), header=0, index_col=0)[0]
        glm_summary = pd.DataFrame(glm_summary.loc['x']).T
        glm_summary.index = [cre]
        enh_result = pd.concat([enh_result, glm_summary], axis=0)
        
        fit_data=pd.DataFrame({'y': vector[cre].values, 
                               'x': vector['total transcripts'].values,
                               'fov': experiment["vector_orig"]['fov'].values})
        if not use_fov_covariate:
            glm_results = smf.ols('y ~ x', fit_data, family=family).fit()
        else:
            glm_results = smf.ols('y ~ x + C(fov)', fit_data, family=family).fit()
        # Fit the model
        glm_summary = pd.read_html(glm_results.summary().tables[1].as_html(), header=0, index_col=0)[0]
        glm_summary = pd.DataFrame(glm_summary.loc['x']).T
        glm_summary.index = [cre]
        vec_result = pd.concat([vec_result, glm_summary], axis=0)
    result = enh_result.copy()
    if norm_by_vector:
        result['coef'] = result['coef'].values / vec_result['coef'].values
    if norm_by_nanopore:
        result['nanopore'] = experiment['nanopore'][0].loc[result.index].values
        result['coef'] = result['coef'].values / np.log(result['nanopore'].values)
    # do glm fit on poshen data
    poshen = pd.read_excel("data/Poshen_Table_S3.xlsx", sheet_name = "raw counts HCT116")
    poshen.set_index('oligo', inplace=True)
    poshen_deseq2 = pd.read_csv("data/HCT116_activating_oligo.csv")
    poshen_deseq2.index = poshen_deseq2['chr'].astype(str) + ":" + poshen_deseq2['start'].astype(int).astype(str) + "-" + poshen_deseq2['end'].astype(int).astype(str)
    ccre_names = experiment['ccre_names']
    # filter for ccre_names in poshen
    ccre_names = ccre_names.loc[ccre_names[0].isin(poshen.index)]
    poshen = poshen.reindex(ccre_names[0], fill_value=np.nan)
    poshen_deseq2 = poshen_deseq2.reindex(ccre_names[0], fill_value=np.nan)
    poshen.index = ccre_names.index
    poshen_deseq2.index = ccre_names.index
    # poshen fill missing values with NA
    poshen = poshen.reindex(result.index, fill_value=np.nan)
    poshen_deseq2 = poshen_deseq2.reindex(result.index, fill_value=np.nan)
    result['Poshen Activity'] = poshen['Activity'].values
    result['Poshen Activity DESeq2'] = poshen_deseq2['lfc'].values
    if 'rna_counts' in experiment and 'dna_counts' in experiment:
        result['RNA/DNA'] = experiment['rna_counts'].values / experiment['dna_counts'].values
        result['RNA/DNA lfc'] = np.log2(result['RNA/DNA'])
    result = result.rename(columns={'coef': 'STARR-FISH Activity', 'std err': 'STARR-FISH Activity std err'})
    result['log(STARR-FISH Activity)'] = np.log(result['STARR-FISH Activity'])
    experiment[key_add] = result
    return experiment

# plot glm result correlation with poshen paper
# poshen_col: the column name of poshen activity, 'Poshen Activity' or 'Poshen Activity DESeq2'
def plot_glm_results(glm_result, x_col='Poshen Activity', y_col='STARR-FISH Activity', fig_name='glm_result.pdf', ax=None):
    # drop na and inf values
    glm_result = glm_result.replace([np.inf, -np.inf], np.nan)
    glm_result = glm_result.dropna()
    if ax is None:
        plt.figure(figsize=(5, 5))
        ax = plt.gca()
    else:
        plt.sca(ax)
    sns.scatterplot(glm_result, x=x_col, y=y_col, hue=glm_result.index)
    sns.regplot(glm_result, x=x_col, y=y_col, scatter=False, color='red')
    slope, intercept, r, p, sterr = scipy.stats.linregress(x=glm_result[y_col], y=glm_result[x_col])
    plt.text(min(glm_result[x_col])*1.2, max(glm_result[y_col])*0.8, 'y = ' + str(round(intercept,3)) + ' + ' + str(round(slope,3)) + 'x\nr = ' + str(round(r,3)) + ', p = ' + str(round(p,3)), fontsize=10)
    plt.legend([],[], frameon=False)
    # save figure
    if fig_name is not None:
        plt.savefig(fig_name, bbox_inches='tight')
    return ax

# plot glm result itself
def plot_activity(glm_result, value='STARR-FISH Activity', std='STARR-FISH Activity std err', fig_name='activity.pdf', ax=None):
    if ax is None:
        plt.figure(figsize=(5, 5))
        ax = plt.gca()
    else:
        plt.sca(ax)
    sns.scatterplot(data=glm_result, x=glm_result.index, y=value, hue=glm_result.index)  # `s` adjusts point size
    plt.errorbar(x=glm_result.index, y=glm_result[value], yerr=2*glm_result[std], 
                 fmt='none',  # Do not plot markers (already done by Seaborn)
                 ecolor='gray', elinewidth=1.5, capsize=5)
    plt.ylabel('STARR-FISH Activity')
    # rotate x ticks
    plt.xticks(rotation=45)
    # remove x label
    plt.xlabel('')
    plt.legend([],[], frameon=False)
    # save figure
    if fig_name is not None:
        plt.savefig(fig_name, bbox_inches='tight')
    return ax

# plot correlation between two experiments
def plot_corr_experiments(experiment1, experiment2, value='STARR-FISH Activity', std='STARR-FISH Activity std err', 
                          x_label = None, y_label = None,
                          fig_name='corr_experiment.pdf', ax=None):
    # drop na and inf values
    glm_res1 = experiment1.replace([np.inf, -np.inf], np.nan).dropna().copy()
    glm_res2 = experiment2.replace([np.inf, -np.inf], np.nan).dropna().copy()
    # rename value and std columns
    glm_res1 = glm_res1.rename(columns={value: f'{value}_rep1', std: f'{std}_rep1'})
    glm_res2 = glm_res2.rename(columns={value: f'{value}_rep2', std: f'{std}_rep2'})
    # merge dataframes
    glm_res = pd.merge(glm_res1, glm_res2, left_index=True, right_index=True)
    # drop na values
    glm_res = glm_res.dropna()
    if ax is None:
        plt.figure(figsize=(5, 5))
        ax = plt.gca()
    else:
        plt.sca(ax)
    # scatter plot
    sns.scatterplot(glm_res, x=f'{value}_rep1', y=f'{value}_rep2', hue=glm_res.index)
    # plot regression line
    sns.regplot(glm_res, x=f'{value}_rep1', y=f'{value}_rep2', scatter=False, color='red')
    # calculate slope, intercept, r, p, sterr
    slope, intercept, r, p, sterr = scipy.stats.linregress(x=glm_res[f'{value}_rep1'], 
                                                           y=glm_res[f'{value}_rep2'])
    plt.text(min(glm_res[f'{value}_rep1'])*1.2, max(glm_res[f'{value}_rep2'])*0.8, 
             'y = ' + str(round(intercept,3)) + ' + ' + str(round(slope,3)) + 'x\nr = ' + str(round(r,3)) + ', p = ' + str(round(p,3)), fontsize=10)
    plt.legend([],[], frameon=False)
    if x_label is not None:
        plt.xlabel(x_label)
    if y_label is not None:
        plt.ylabel(y_label)
    # title
    plt.title(f'Correlation between two experiments')
    # save figure
    if fig_name is not None:
        plt.savefig(fig_name, bbox_inches='tight')
    return ax
# %%
if __name__ == "__main__":
    # %%
    july_experiment = preprocess_experiment(enhancer_file='data/SFv4_T7_July_enhancer_cbg.csv', vector_file='data/SFv4_T7_July_T7_cbg.csv', nanopore_file='data/SFv4_T7_20CRE_nanopore_counts',
                                            rna_seq_file='data/Bulk_sequencing/RNA_sequencing/CRE20_T7_July_featureCounts_output.txt',
                                            dna_seq_file='data/Bulk_sequencing/DNA_sequencing/CRE20_July_featureCounts_output.txt', 
                                            bed_file='data/20CRE.bed')
    sept_experiment = preprocess_experiment(enhancer_file='data/SFv4_T7_Sept_enhancer_cbg.csv', vector_file='data/SFv4_T7_Sept_T7_cbg.csv', nanopore_file='data/SFv4_T7_20CRE_nanopore_counts',
                                            rna_seq_file='data/Bulk_sequencing/RNA_sequencing/CRE20_T7_Sept_featureCounts_output.txt',
                                            dna_seq_file='data/Bulk_sequencing/DNA_sequencing/CRE20_Sept_featureCounts_output.txt', 
                                            bed_file='data/20CRE.bed')
    CRE_300_experiment1 = preprocess_experiment(enhancer_file='data/SFv6_cell_by_CRE_01_04_2023.csv', vector_file=None, nanopore_file='data/SFv6_300CRE_nanopore_counts',
                                                bed_file='data/STARR-FISH_300_library.bed')
    CRE_300_experiment2 = preprocess_experiment(enhancer_file='data/SFv6_cell_by_CRE_03_19_2023.csv', vector_file=None, nanopore_file='data/SFv6_300CRE_nanopore_counts',
                                                bed_file='data/STARR-FISH_300_library.bed')
    # %%
    # fit glm
    july_experiment = glm_fit(july_experiment, family=sm.families.Gaussian(), norm_by_total=False)
    sept_experiment = glm_fit(sept_experiment, family=sm.families.Gaussian(), norm_by_total=False)
    july_experiment = glm_fit_total(july_experiment, family=sm.families.Gaussian(), norm_by_vector=False, norm_by_nanopore=True, key_add='glm_fit_total_result_nanopore')
    sept_experiment = glm_fit_total(sept_experiment, family=sm.families.Gaussian(), norm_by_vector=False, norm_by_nanopore=True, key_add='glm_fit_total_result_nanopore')
    july_experiment = glm_fit_total(july_experiment, family=sm.families.Gaussian(), norm_by_vector=True, norm_by_nanopore=False, key_add='glm_fit_total_result_T7')
    sept_experiment = glm_fit_total(sept_experiment, family=sm.families.Gaussian(), norm_by_vector=True, norm_by_nanopore=False, key_add='glm_fit_total_result_T7')
    # %%
    CRE_300_experiment1 = glm_fit_total(CRE_300_experiment1, family=sm.families.Gaussian(), norm_by_vector=False, norm_by_nanopore=True, key_add='glm_fit_total_result_nanopore')
    CRE_300_experiment2 = glm_fit_total(CRE_300_experiment2, family=sm.families.Gaussian(), norm_by_vector=False, norm_by_nanopore=True, key_add='glm_fit_total_result_nanopore')
    # %% plot
    plot_activity(july_experiment['glm_fit_result'], fig_name=None)
    # %%
    def plot_glm_two_experiments(exp1, exp2, key='glm_fit_result'):
        fig, ax = plt.subplots(2, 4, figsize=(24, 10))
        # drop nanopore counts ≤ 10 if use glm_fit_total_result_nanopore
        if key == 'glm_fit_total_result_nanopore':
            toplot1 = exp1[key].dropna().copy()
            toplot1 = toplot1[toplot1['nanopore'] > 10]
            toplot2 = exp2[key].dropna().copy()
            toplot2 = toplot2[toplot2['nanopore'] > 10]
        else:
            toplot1 = exp1[key].dropna().copy()
            toplot2 = exp2[key].dropna().copy()
        plot_glm_results(toplot1, x_col='Poshen Activity', fig_name=None, ax=ax[0, 0])
        plot_glm_results(toplot1, x_col='Poshen Activity DESeq2', fig_name=None, ax=ax[0, 1])
        plot_glm_results(toplot1, x_col='RNA/DNA', fig_name=None, ax=ax[0, 2])
        plot_glm_results(toplot1, x_col='RNA/DNA lfc', fig_name=None, ax=ax[0, 3])
        plot_glm_results(toplot2, x_col='Poshen Activity', fig_name=None, ax=ax[1, 0])
        plot_glm_results(toplot2, x_col='Poshen Activity DESeq2', fig_name=None, ax=ax[1, 1])
        plot_glm_results(toplot2, x_col='RNA/DNA', fig_name=None, ax=ax[1, 2])
        plot_glm_results(toplot2, x_col='RNA/DNA lfc', fig_name=None, ax=ax[1, 3])
        return fig
    # %%
    fig = plot_glm_two_experiments(july_experiment, sept_experiment, key='glm_fit_result')
    fig.show()
    # save figure
    # fig.savefig('fig/glm_result.pdf', bbox_inches='tight')
    # %%
    fig = plot_glm_two_experiments(july_experiment, sept_experiment, key='glm_fit_total_result_nanopore')
    fig.show()
    # save figure
    # fig.savefig('fig/glm_total_result_nanopore.pdf', bbox_inches='tight')
    # %%
    fig = plot_glm_two_experiments(july_experiment, sept_experiment, key='glm_fit_total_result_T7') 
    plt.show()
    # save figure
    # fig.savefig('fig/glm_total_result_T7.pdf', bbox_inches='tight')
    # %%
    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    plot_corr_experiments(july_experiment['glm_fit_result'], sept_experiment['glm_fit_result'], fig_name=None, ax=ax[0],
                        x_label='July Enhancer~T7', y_label='Sept Enhancer~T7')
    plot_corr_experiments(july_experiment['glm_fit_total_result_nanopore'], sept_experiment['glm_fit_total_result_nanopore'], fig_name=None, ax=ax[1],
                        x_label='July (Enhancer~total) / nanorepore', y_label='Sept (Enhancer~total) / nanorepore')
    plot_corr_experiments(july_experiment['glm_fit_total_result_T7'], sept_experiment['glm_fit_total_result_T7'], fig_name=None, ax=ax[2],
                        x_label='July (Enhancer~total) / (T7~total)', y_label='Sept (Enhancer~total) / (T7~total)')
    plt.show()
    # fig.savefig('fig/corr_experiment.pdf', bbox_inches='tight')
    # %%
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    toplot1 = july_experiment['glm_fit_total_result_nanopore'].dropna().copy()
    toplot1 = toplot1[toplot1['nanopore'] > 10]
    toplot2 = sept_experiment['glm_fit_total_result_nanopore'].dropna().copy()
    toplot2 = toplot2[toplot2['nanopore'] > 10]
    plot_corr_experiments(july_experiment['glm_fit_total_result_T7'], toplot1, fig_name=None, ax=ax[0],
                        x_label='July (Enhancer~total) / nanorepore', y_label='July (Enhancer~total) / (T7~total)')
    plot_corr_experiments(sept_experiment['glm_fit_total_result_T7'], toplot2, fig_name=None, ax=ax[1],
                        x_label='Sept (Enhancer~total) / nanorepore', y_label='Sept (Enhancer~total) / (T7~total)')
    plt.show()
    # fig.savefig('fig/corr_normalization_T7_nanorepore.pdf', bbox_inches='tight')
    # %%
    # scatter plot
    july_T7 = july_experiment['vector'].copy().sum(axis=0)
    july_T7 = pd.DataFrame({'T7': july_T7, 'log(nanorepore)': np.log(july_experiment['glm_fit_total_result_nanopore']['nanopore'].values)},
                        index = july_experiment['glm_fit_total_result_nanopore'].index)
    sns.scatterplot(july_T7, x=f'T7', y=f'log(nanorepore)', hue=july_T7.index)
    sns.regplot(july_T7, x=f'T7', y=f'log(nanorepore)', scatter=False, color='red')
    # calculate slope, intercept, r, p, sterr
    slope, intercept, r, p, sterr = scipy.stats.linregress(x=july_T7[f'T7'], y=july_T7[f'log(nanorepore)'])
    plt.text(min(july_T7[f'T7'])*1.2, max(july_T7[f'log(nanorepore)'])*0.8, 
                'y = ' + str(round(intercept,3)) + ' + ' + str(round(slope,3)) + 'x\nr = ' + str(round(r,3)) + ', p = ' + str(round(p,3)), fontsize=10)
    plt.legend([],[], frameon=False)
    plt.xlabel('July T7')
    # %%
    sept_T7 = sept_experiment['vector'].copy().sum(axis=0)
    sept_T7 = pd.DataFrame({'T7': sept_T7, 'log(nanorepore)': np.log(sept_experiment['glm_fit_total_result_nanopore']['nanopore'].values)},
                        index = sept_experiment['glm_fit_total_result_nanopore'].index)
    sns.scatterplot(sept_T7, x=f'T7', y=f'log(nanorepore)', hue=sept_T7.index)
    sns.regplot(sept_T7, x=f'T7', y=f'log(nanorepore)', scatter=False, color='red')
    # calculate slope, intercept, r, p, sterr
    slope, intercept, r, p, sterr = scipy.stats.linregress(x=sept_T7[f'T7'], y=sept_T7[f'log(nanorepore)'])
    plt.text(min(sept_T7[f'T7'])*1.2, max(sept_T7[f'log(nanorepore)'])*0.8, 
                'y = ' + str(round(intercept,3)) + ' + ' + str(round(slope,3)) + 'x\nr = ' + str(round(r,3)) + ', p = ' + str(round(p,3)), fontsize=10)
    plt.legend([],[], frameon=False)
    plt.xlabel('Sept T7')
    # %%
    # check 300 CRE experiment
    fig, ax = plt.subplots(2, 2, figsize=(12, 10))
    key='glm_fit_total_result_nanopore'
    # drop nanopore counts ≤ 10
    toplot1 = CRE_300_experiment1[key].dropna().copy()
    toplot1 = toplot1[toplot1['nanopore'] > 10]
    toplot2 = CRE_300_experiment2[key].dropna().copy()
    toplot2 = toplot2[toplot2['nanopore'] > 10]
    plot_glm_results(toplot1, x_col='Poshen Activity', fig_name=None, ax=ax[0, 0])
    plot_glm_results(toplot1, x_col='Poshen Activity DESeq2', fig_name=None, ax=ax[0, 1])
    plot_glm_results(toplot2, x_col='Poshen Activity', fig_name=None, ax=ax[1, 0])
    plot_glm_results(toplot2, x_col='Poshen Activity DESeq2', fig_name=None, ax=ax[1, 1])
    fig.show()
    # %%
    # plot correlation between Poshen and STARR-FISH activity on different groups of CREs
    cre_groups = []
    libsize = CRE_300_experiment1[key].dropna().copy()['nanopore']
    cre_groups.append(libsize[libsize < 5].index)
    cre_groups.append(libsize[(libsize >= 5) & (libsize < 10)].index)
    cre_groups.append(libsize[(libsize >= 10) & (libsize < 500)].index)
    cre_groups.append(libsize[(libsize >= 25) & (libsize < 50)].index)
    cre_groups.append(libsize[(libsize >= 50) & (libsize < 1000)].index)
    cre_groups.append(libsize[libsize >= 1000].index)
    # 4 x 2 subplots
    fig, ax = plt.subplots(2, len(cre_groups), figsize=(6*len(cre_groups), 12))
    for i, cre_group in enumerate(cre_groups):
        toplot1 = CRE_300_experiment1[key].dropna().copy().loc[cre_group]
        plot_glm_results(toplot1, x_col='Poshen Activity', fig_name=None, ax=ax[0, i])
        plot_glm_results(toplot1, x_col='Poshen Activity DESeq2', fig_name=None, ax=ax[1, i])
    # %%
    sns.regplot(CRE_300_experiment1[key].dropna(), x='Poshen Activity', y='STARR-FISH Activity', scatter=False, color='red')
    slope, intercept, r, p, sterr = scipy.stats.linregress(x=CRE_300_experiment1[key].dropna()['STARR-FISH Activity'], 
                                                        y=CRE_300_experiment1[key].dropna()['Poshen Activity'])
    # %%
