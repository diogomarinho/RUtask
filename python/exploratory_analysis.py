import pandas as pd
import seaborn as sns
import gcsfs
import matplotlib.pyplot as plt
import numpy as np

import sklearn as sk
from sklearn import decomposition
from sklearn.decomposition import PCA
from sklearn.externals import joblib

from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest


import os
# 20/10/2018

# Aux to plot histograms
def histogram(distribution, xlabel, output, bins=None, clear=True, color=None, label=None):
    try:
        plot = sns.distplot(distribution, kde=False, bins=bins, color=color, label=label)
        plot.set(xlabel=xlabel)
        plot.get_figure().savefig(output)
        if (clear):
            plt.clf()
        return(plot)
    except Exception as e:
        print(e)
    return(False)

# histogram(distribution=uf.total_length, xlabel='log(N)', output='/Users/diogoma/fdmusic/plots/total_length_dist.png')
# main method which generate the plots and create the models
def buil_model(uf, to_plot = False):

    sns.set(style="white")

    if (to_plot):
        try:
        # Create target Directory
            if(to_plot):
                os.mkdir('plots')
                print("Saving plots in ./plots/")
        #print("Directory " , dirName ,  " Created ")
        except FileExistsError:
            if(to_plot):
                print("Saving plots in ./plots/")

        '''
        blobs = '56cd13f3-cd7e-4ead-88af-623e391ec8d8/user_features.csv.gz
        fs = gcsfs.GCSFileSystem(project="56cd13f3-cd7e-4ead-88af-623e391ec8d8")
        with fs.open() as f:
            users_features = pd.read_csv(f, lines=True, compression='gzip')
        '''
        #uf = pd.read_csv('~/fdmusic/uploaded/user_features_P2.csv.gz', index_col=0, compression='gzip')

        # . starting showing the distribution of how many streams each user performed
        print('Total songs distribution across the users')
        histogram(distribution=uf.total_songs, bins=100, xlabel='#Songs per user', output='./plots/total_songs_dist.png')


        # . distributions of total length streaming
        print('Log of total length of streaming distribution across the users')
        histogram(distribution=uf.total_length, bins=100,xlabel='Sum(length) sec', output='./plots/total_length_dist.png')


        # . distribution of the total number of songs / number of artists
        print('Number of #songs/#artists across the users')
        histogram(distribution=uf.songs_per_artist, bins=100,xlabel='#songs/#artists', output='./plots/songs_artists_dist.png')


        # . distribution mean length
        print('Coef of variation length distribution across the users')
        histogram(distribution=uf.cv_length, bins=100,xlabel='length (cv)', output='./plots/cv_length_dist.png')
        uf.cv_length = uf.cv_length


        # . distribution repetition
        print('Rate of repeptition')
        histogram(distribution=uf.repetition_rate, bins=100,xlabel='repetitions rate', output='./plots/repetition_dist.png')


        # . distribution mean time interval
        print('Timestamp interval coefficient of variation')
        histogram(distribution=uf.cv_timestamp, bins=300,xlabel='Timestamp interval (cv)', output='./plots/cv_timestamp_dist.png')


        # entropy of time intervals and lengths can be a good way to classify anomaly, since
        # regular patters might produced low level of entropy
        print('Total entropy distribution')
        histogram(distribution=uf.entropy, bins=100,xlabel='entropy', output='./plots/entropy.png')

        #
        print('Entropy plot')
        entropy_2d = sns.scatterplot(x='timestamp_entropy', y='length_entropy', data=uf)
        plt.ylim(0, None)
        plt.xlim(0, None)
        entropy_2d.get_figure().savefig('./plots/length_entropy_time_interval_entropy.png')
        plt.clf()


        # computing correlatin between normalized_data
        sns.set(font_scale=0.7)
        cor_matrix = uf.drop(['birth_year', 'country', 'gender'], axis=1)
        corr = cor_matrix.corr()
        mask = np.zeros_like(corr, dtype=np.bool)
        mask[np.triu_indices_from(mask)] = True
        cor_plot = sns.heatmap(corr, mask=mask, square=True, center=0, cbar_kws={"shrink": .3 }, annot=True,)
        cor_plot.set_yticklabels(cor_plot.get_yticklabels(), rotation = 20)
        cor_plot.set_xticklabels(cor_plot.get_xticklabels(), rotation = 18)
        cor_plot.get_figure().savefig('./plots/features_correlation.png')
        plt.clf()

    # Starting standardization: removing high correlated and random features
    normalized_data = uf.drop(['total_length', 'birth_year', 'country', 'gender','entropy'], axis=1)

    # factorize account_type features
    labels, uniques = pd.factorize(normalized_data.access)
    normalized_data = normalized_data.drop(['access'], axis=1)
    normalized_data['access'] = labels

    # this need to be stored for new datasets
    normalized_data.total_songs = np.log(normalized_data.total_songs + 1)

    #
    mean = normalized_data.mean()
    std = normalized_data.std()
    model_normalization = pd.DataFrame({'mean':mean, 'std':std})

    # normalizing by it's mean
    print('Data normalization ')
    normalized_data=(normalized_data-normalized_data.mean())/normalized_data.std()
    #uf.total_songs = np.log1p(uf.total_songs + 1)
    # applying PCA in the whole data
    print ('Create principal components')
    pca = decomposition.PCA()
    pca_transformed = pca.fit_transform(normalized_data)
    print('Explained variance: ')
    print(pca.explained_variance_ratio_)

    # let's see pca loadings
    comps = pd.DataFrame(pca.components_, columns=normalized_data.columns.values)

    i = 1
    t_comps = comps.transpose().abs()
    total_features = []
    for cols in t_comps:
        t_comps = t_comps.sort_values([cols], ascending=False)
        print('PC{I}: {A} and {B}'.format(I=i, A=t_comps.index.values[0], B=t_comps.index.values[1]))
        i+=1
        #total_features.append(t_comps.index.values[0])
        #total_features.append(t_comps.index.values[1])
    # .
    # Plotting sample based on the 1 and 2 principal components
    pca_transformed = pd.DataFrame(pca_transformed)
    pca_transformed.index = uf.index.values
    pca_transformed.columns = ['PC' + s for s in list(map(str, range(1 , pca_transformed.shape[1] + 1)))]
    # .
    #pca_transformed.to_csv('~/fdmusic/uploaded/pca_transformed.csv.gz', compression='gzip')
    #normalized_data.to_csv('~/fdmusic/uploaded/normalized_data.csv.gz', compression='gzip')


    # training isolation forest for anomaly detection
    print('Fitting isolation forest')
    rng = np.random.RandomState(2018)
    clf = IsolationForest(behaviour='new', max_samples=200, random_state=rng, contamination='auto')
    clf.fit(pca_transformed.values)
    # clf = joblib.load('filename.joblib')

    #
    try:
        os.mkdir('models')
        print('saving in ./models')
    #print("Directory " , dirName ,  " Created ")
    except FileExistsError:
        print('saving in ./models')
# . saving models
    joblib.dump(clf, './models/isolation_forest.joblib')
    joblib.dump(pca, './models/PCA.joblib')
    model_normalization.to_csv('./models/norm.joblib')

    if (to_plot):
        pca_plot = sns.heatmap(comps, center=0, cbar_kws={"shrink": .3 }, annot=True,)
        pca_plot.set_xticklabels(pca_plot.get_xticklabels(), rotation = 18)
        pca_plot.get_figure().savefig('/Users/diogoma/fdmusic/plots/PCA_features_corr.png')
        plt.clf()
        pca_plot = sns.scatterplot(x='PC1', y='PC2', data=pca_transformed)
        pca_plot.get_figure().savefig('./plots/pca_1_2.png')
        plt.clf()
        histogram(distribution=pca_transformed.PC1, xlabel='PC1', output='./plots/PC1.png')
        histogram(distribution=pca_transformed.PC2, xlabel='PC1', output='./plots/PC2.png')
        histogram(distribution=pca_transformed.PC3, xlabel='PC1', output='./plots/PC3.png')
        histogram(distribution=pca_transformed.PC4, xlabel='PC1', output='./plots/PC4.png')
        histogram(distribution=pca_transformed.PC5, xlabel='PC1', output='./plots/PC5.png')
    #clf = joblib.load('filename.joblib')
    #upload_to_bucket(pca_transformed, normalized_data)
