#!/usr/bin/python

import argparse
import pandas as pd
import gcsfs
import numpy as np

import sklearn as sk
from sklearn import decomposition
from sklearn.decomposition import PCA
from sklearn.externals import joblib

# from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest

import os

import pre_process_data as ppd

parser = argparse.ArgumentParser( description='Detect bots in the system!')
parser.add_argument('-r','--report', action='store', required=True,help='One user id per file')

# .
try:
    args = parser.parse_args()

    pca = joblib.load('./models/pca.joblib')
    isolation_forest = joblib.load('./models/isolation_forest.joblib')
    norm = pd.read_csv('./models/norm.joblib', index_col = 0)

    report = pd.read_csv(args.report, header = None)[0].tolist()

    users, tracks, streams = ppd.load_files_from_disk   (
        '~/fdmusic/downloaded/users.csv.gz',
        '~/fdmusic/downloaded/tracks.csv.gz',
        '~/fdmusic/downloaded/streams.csv.gz'
    )

    users = users[users.user_id.isin(report)]
    streams = streams[streams.user_id.isin(report)]
    streams = streams.merge(tracks, on='track_id')
    # .
    normalized_data = ppd.process(users, streams)
    users = users.set_index('user_id')
    users = users.loc[normalized_data.index.values, :]
    #
    normalized_data['access'] = users.access
    labels, uniques = pd.factorize(normalized_data.access)
    normalized_data = normalized_data.drop(['access'], axis=1)
    normalized_data['access'] = labels
    normalized_data = normalized_data.drop(['total_length', 'entropy'], axis=1)
    normalized_data.total_songs = np.log(normalized_data.total_songs + 1)
    normalized_data=(normalized_data-norm['mean'].values)/norm['std'].values
    #
    pca_transformed = pca.fit_transform(normalized_data.values)
    prediction =  isolation_forest.fit_predict(pca_transformed)
    num_outliers = np.sum(prediction == -1)
    print('Number of outliers {N}'.format(np.sum(prediction == -1)))
    if (num_outliers > 0):
        


except Exception as e:
    print(e)
    print('Something went wrong, bye')
