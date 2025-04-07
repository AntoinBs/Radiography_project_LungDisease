import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statannotations
from statannotations.Annotator import Annotator
# Import statistical libraries
from scipy.stats import f_oneway
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd

def VisuMetaD(data):
    # Pie chart of composition of data

    plt.figure(figsize=(15,15))


    total= data['label'].value_counts().values.sum()
    print(total)

    def fmt(x):
        return '{:.1f}%\n{:.0f}'.format(x, total*x/100)

    plt.subplot(3,3,1)
    fig1=plt.pie(data['label'].value_counts().values, labels= data['label'].value_counts().index, autopct=fmt)
    plt.title('Composition du dataset général')


    # Stacked bars with composition of each dataset
    plt.subplot(3,3,2)
    fig2= sns.histplot(data, x='label', hue='origin_data',multiple='stack', stat='percent')
    plt.title('Origine des données')
    plt.xlabel('Dataset')
    plt.ylabel('Pourcentage')
    plt.legend(list(data['origin_data'].unique()),bbox_to_anchor=(1, 1), loc='upper left', frameon= False)
    
    plt.savefig(r'..\reports\figures\dataset_composition.pdf', bbox_inches='tight')
    plt.savefig(r'..\reports\figures\dataset_composition.png', bbox_inches='tight')
    
    return plt.show()



def WhitePix_explo(metadata, Allwhitepix, test = str, test_correction= str):

    ''' This functions explores statistically and graphically the size of masks according to different datasets
    It also performs statistical analysis of differences between mask sizes, as well as identify outliers and count them.
    -------------------------------------------- ARGUMENTS ---------------------------------------------------
    metadata = dataframe containing all the metadata from all datasets (output of LoadMetaD)
    Allwhitepix= dataframe containing the number and area of white pixels in each mask (output of CountWhitePixMasks)

    -------------------------------------------- Parameters ---------------------------------------------------
    test = str | satistical test to do using statannot
        Available tests: - Mann-Whitney
                         - t-test (independant and paired)
                         - Welch's t-test
                         - Levene test
                         - Wilcoxon test
                         - Kruskal-Wallis test
                         - Brunner-Munzel test
    
    test_correstion = str | statistical correction to apply to the test
        Available correctors : - Bonferroni
                               - Holm-Bonferroni
                               - Benjamini-Hochberg
                               - Benjamini-Yekutieli
    ----------------------------------------- RETURN -----------------------------------------------------------

    Plots annotated with statistical significance of differences when relevant.

    '''
    # Plot Size of lungs as boxplots separated by 
    plt.figure(figsize=(15,15))
    plt.subplot(3,3,1)
    Dataset_plt=sns.violinplot(Allwhitepix, x='label', y='relative')
    plt.xlabel('Dataset')
    plt.ylabel('% Pixels blancs')
    plt.title('Taille des masques dans les différents datasets')

    # Statistical analysis
    model=ols('relative ~ C(label)', data= Allwhitepix).fit()
    # Lung size by pathologie
    # Prepare data
    anova_table= sm.stats.anova_lm(model, typ=2)
    anova_table
    anova_table.to_csv(r'../data/processed/stats/ANOVA_maskDataset.csv',sep=',')

    # Perform Tukey's HSD test
    tukey_results = pairwise_tukeyhsd(Allwhitepix['relative'],Allwhitepix['label'])

    print(tukey_results)

    # Indicate statistical significance to comparisons between pairs
    modes= Allwhitepix['label'].unique()
    modes=list(modes)
    pairs = [(a, b) for idx, a in enumerate(modes) for b in modes[idx + 1:]]

    
    annotator = Annotator(
        Dataset_plt,
        pairs,
        data=Allwhitepix,
        x='label',
        y='relative'
    )
    annotator.configure(test=test, text_format='star', loc='inside', verbose=2, comparisons_correction=test_correction)
    annotator.apply_and_annotate()

    # Link Lung size wwith data origin
    # Add origin of data to white pixels dataframe
    Allwhitepix.reset_index(drop=True, inplace= True)
    metadata.reset_index(drop=True, inplace=True)
    Pix_Origin=pd.concat([Allwhitepix, metadata['origin_data']], axis=1, join='outer')

    # Make boxplot separated by dataset and source of datasets
    plt.subplot(3,3,2)
    sns.boxplot(x=Pix_Origin['label'],y=Pix_Origin['relative'],hue=Pix_Origin['origin_data'])
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, title='Source data')
    plt.xlabel('Dataset')
    plt.ylabel('% Pixels blancs')
    plt.title('Taille des masques dans les différentes sources')

    # Make violin plots of COVID dataset separated by origin of dataset
    COVID_OriginPix=Pix_Origin[Pix_Origin['label']=='COVID']
    plt.subplot(3,3,4)
    COVID_originplt=sns.violinplot(x=COVID_OriginPix['origin_data'],y=COVID_OriginPix['relative'])
    sns.color_palette()
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper right', borderaxespad=0)
    plt.xlabel('Origine des données')
    plt.ylabel('% Pixels blancs')
    plt.title('Taille des masques COVID')

    #Statistical analysis COVID Lung size by data source
    COVID_OriginPix=Pix_Origin[Pix_Origin['label']=='COVID']
    model3=ols('relative ~ C(origin_data)',data= Pix_Origin).fit() 

    anova_table3= sm.stats.anova_lm(model3, typ=2)
    anova_table3
    anova_table3.to_csv(r'../data/processed/stats/ANOVA_maskDataset_Origin.csv',sep=',')

    tukey_results = pairwise_tukeyhsd(Pix_Origin['relative'],Pix_Origin['origin_data'])
    print(tukey_results)

    # Indicate statistical significance to comparisons between pairs
    modes= COVID_OriginPix['origin_data'].unique()
    modes=list(modes)
    pairs = [(a, b) for idx, a in enumerate(modes) for b in modes[idx + 1:]]

    annotator = Annotator(
        COVID_originplt,
        pairs,
        data=COVID_OriginPix,
        x='origin_data',
        y='relative'
    )
    annotator.configure(test=test, text_format='star', loc='inside', verbose=2, comparisons_correction=test_correction)
    annotator.apply_and_annotate()
    
    plt.savefig(r'..\reports\figures\whitepix_exploration.pdf', bbox_inches='tight')
    plt.savefig(r'..\reports\figures\whitepix_exploration.png', bbox_inches='tight')

    grouped = Pix_Origin.groupby(['label', 'origin_data'])

    # Function to calculate outliers in 'relative' for each group
    def count_outliers(group):
        Q1 = group['relative'].quantile(0.25)
        Q3 = group['relative'].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = group[(group['relative'] < lower_bound) | (group['relative'] > upper_bound)]
        return len(outliers)

    # Apply function to each group and reset index
    outlier_counts = grouped.apply(count_outliers).reset_index(name='Outlier_Count')

    # Display the results
    outlier_counts.to_csv(r'../data/processed/outlier_count.csv',sep=',')
    print(outlier_counts)
    
    return plt.show()