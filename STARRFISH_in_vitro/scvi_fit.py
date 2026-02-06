# %%
import statsmodels.api as sm
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy
import scanpy as sc
from scvi.model import SCVI
import warnings
warnings.filterwarnings('ignore')
# %%
# read in data
def preprocess_experiment(enhancer_file, vector_file):
    enhancer = pd.read_csv(enhancer_file)
    vector = pd.read_csv(vector_file)
    enhancer.index = enhancer['masks'].astype(str) + '_' + enhancer['fov'].astype(str)
    vector.index = vector['masks'].astype(str) + '_' + vector['fov'].astype(str)
    # find common index and filter
    common_index = enhancer.index.intersection(vector.index)
    enhancer = enhancer.loc[common_index]
    vector = vector.loc[common_index]
    # drop mask and fov columns
    enhancer_drop = enhancer.copy().drop(columns=['masks', 'fov', 'total transcripts'])
    vector_drop = vector.copy().drop(columns=['masks', 'fov', 'total transcripts'])
    # append column names
    enhancer_drop.columns = [str(x) + '_enhancer' for x in enhancer_drop.columns]
    vector_drop.columns = [str(x) + '_vector' for x in vector_drop.columns]
    # prepare a scanpy object
    adata = sc.AnnData(X=np.concat([enhancer_drop.values, vector_drop.values], axis=1),
                       obs=enhancer[['masks', 'fov']].copy(), 
                       var=pd.DataFrame(index=enhancer_drop.columns.append(vector_drop.columns)))
    return adata

# fit glm on experiment
# glm fit on single experiment, fit enhancer (normalized by scvi) ~ T7 (normalized by scvi)
# if norm_by_total, then fit enhancer / total_enhancer ~ vector / total_vector
def glm_fit(experiment, family=sm.families.Gaussian(), norm_by_total=False):
    # Create the model
    enhancer_columns = experiment.var_names[experiment.var_names.str.contains('enhancer')]
    vector_columns = experiment.var_names[experiment.var_names.str.contains('vector')]
    enhancer = experiment.obsm['X_scvi'][enhancer_columns].copy()
    vector = experiment.obsm["X_scvi"][vector_columns].copy()
    # remove the _enhancer and _vector suffix
    enhancer.columns = enhancer.columns.str.replace('_enhancer', '')
    vector.columns = vector.columns.str.replace('_vector', '')
    result = pd.DataFrame()
    if norm_by_total:
        enhancer = enhancer.div(enhancer.sum(axis=1), axis=0)
        vector = vector.div(vector.sum(axis=1), axis=0)
    for cre in enhancer.columns[:-1]:
        if cre not in vector.columns:
            raise ValueError(f"CRE {cre} not found in vector data.")
        glm_model = sm.GLM(enhancer[cre], vector[cre], family=family)
        # Fit the model
        glm_results = glm_model.fit()
        glm_summary = glm_results.summary()
        result = pd.concat([result, pd.read_html(glm_summary.tables[1].as_html(), header=0, index_col=0)[0]], axis=0)
    # do glm fit on poshen data
    poshen = pd.read_excel("data/Poshen_Table_S3.xlsx", sheet_name = "raw counts HCT116")
    poshen.set_index('oligo', inplace=True)
    poshen_deseq2 = pd.read_csv("data/HCT116_activating_oligo.csv")
    poshen_deseq2.index = poshen_deseq2['chr'].astype(str) + ":" + poshen_deseq2['start'].astype(int).astype(str) + "-" + poshen_deseq2['end'].astype(int).astype(str)
    ccre_names = pd.read_csv('data/20CRE.bed', header=None, sep='\t')
    ccre_names = ccre_names.astype(str)
    poshen = poshen.loc[(ccre_names[0] + ":" + ccre_names[1] + "-" + ccre_names[2])]
    poshen_deseq2 = poshen_deseq2.loc[(ccre_names[0] + ":" + ccre_names[1] + "-" + ccre_names[2])]
    poshen.index = [f'CRE{str(i+1).zfill(3)}' for i in range(len(poshen))]
    poshen_deseq2.index = [f'CRE{str(i+1).zfill(3)}' for i in range(len(poshen))]
    # poshen fill missing values with NA
    poshen = poshen.reindex(result.index, fill_value=np.nan)
    poshen_deseq2 = poshen_deseq2.reindex(result.index, fill_value=np.nan)
    result['Poshen Activity'] = poshen['Activity'].values
    result['Poshen Activity DESeq2'] = poshen_deseq2['lfc'].values
    # rename result
    result = result.rename(columns={'coef': 'STARR-FISH Activity', 'std err': 'STARR-FISH Activity std err'})
    return result

# plot glm result correlation with poshen paper
# poshen_col: the column name of poshen activity, 'Poshen Activity' or 'Poshen Activity DESeq2'
def plot_glm_results(glm_result, poshen_col='Poshen Activity', fig_name='glm_result.pdf', ax=None):
    if ax is None:
        plt.figure(figsize=(5, 5))
        ax = plt.gca()
    else:
        plt.sca(ax)
    glm_result = glm_result.dropna()
    sns.scatterplot(glm_result, x=poshen_col, y='STARR-FISH Activity', hue=glm_result.index)
    sns.regplot(glm_result, x=poshen_col, y='STARR-FISH Activity', scatter=False, color='red')
    slope, intercept, r, p, sterr = scipy.stats.linregress(x=glm_result['STARR-FISH Activity'], 
                                                           y=glm_result[poshen_col])
    plt.text(min(glm_result[poshen_col])*1.2, max(glm_result['STARR-FISH Activity'])*0.8, 'y = ' + str(round(intercept,3)) + ' + ' + str(round(slope,3)) + 'x\nr = ' + str(round(r,3)) + ', p = ' + str(round(p,3)), fontsize=10)
    plt.ylabel('STARR-FISH Activity')
    plt.legend([],[], frameon=False)
    # save figure
    if fig_name is not None:
        plt.savefig(fig_name, bbox_inches='tight')
    return ax

# plot glm result itself
def plot_activity(glm_result, value='STARR-FISH Activity', std='STARR-FISH Activity std err', fig_name='activity.pdf'):
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
    plt.savefig(fig_name, bbox_inches='tight')
    return plt.gca()

# plot correlation between two experiments
def plot_corr_experiments(glm_res1, glm_res2, value='STARR-FISH Activity', std='STARR-FISH Activity std err', fig_name='corr_experiment.pdf'):
    # rename value and std columns
    glm_res1 = glm_res1.rename(columns={value: f'{value}_rep1', std: f'{std}_rep1'})
    glm_res2 = glm_res2.rename(columns={value: f'{value}_rep2', std: f'{std}_rep2'})
    # merge dataframes
    glm_res = pd.merge(glm_res1, glm_res2, left_index=True, right_index=True)
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
    # title
    plt.title(f'Correlation between two experiments')
    # save figure
    plt.savefig(fig_name, bbox_inches='tight')
    return plt.gca()

# %%
july_experiment = preprocess_experiment('data/SFv4_T7_July_enhancer_cbg.csv', 'data/SFv4_T7_July_T7_cbg.csv')
sept_experiment = preprocess_experiment('data/SFv4_T7_Sept_enhancer_cbg.csv', 'data/SFv4_T7_Sept_T7_cbg.csv')
# fit scvi on experiment
SCVI.setup_anndata(july_experiment, batch_key='fov')
july_scvi = SCVI(july_experiment, n_latent=10, n_layers=2, n_hidden=16,
                 gene_likelihood='nb')
july_scvi.train(devices=[1])
SCVI.setup_anndata(sept_experiment, batch_key='fov')
sept_scvi = SCVI(sept_experiment, n_latent=10, n_layers=2, n_hidden=16,
                 gene_likelihood='nb')
sept_scvi.train(devices=[1])
# %%
july_experiment.obsm['X_scvi'] = july_scvi.get_normalized_expression()
sept_experiment.obsm['X_scvi'] = sept_scvi.get_normalized_expression()
july_glm = glm_fit(july_experiment, family=sm.families.Gaussian(), norm_by_total=False)
sept_glm = glm_fit(sept_experiment, family=sm.families.Gaussian(), norm_by_total=False)
# %%
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
plot_glm_results(july_glm, poshen_col='Poshen Activity', fig_name=None, ax=ax[0])
plot_glm_results(july_glm, poshen_col='Poshen Activity DESeq2', fig_name=None, ax=ax[1])
fig.show()
# %%
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
plot_glm_results(sept_glm, poshen_col='Poshen Activity', fig_name=None, ax=ax[0])
plot_glm_results(sept_glm, poshen_col='Poshen Activity DESeq2', fig_name=None, ax=ax[1])
fig.show()
# %%
plot_corr_experiments(july_glm, sept_glm, value='STARR-FISH Activity', std='STARR-FISH Activity std err', 
                      fig_name='fig/scvi_corr_experiment.pdf')
# %%
