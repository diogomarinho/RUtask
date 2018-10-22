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
    users = pd.read_csv(args.report, header = None).values

    users, tracks, streams = load_files_from_disk   (
        '~/fdmusic/downloaded/users.csv.gz',
        '~/fdmusic/downloaded/tracks.csv.gz',
        '~/fdmusic/downloaded/streams.csv.gz'
    )



    import pdb; pdb.set_trace()
except Exception as e:
    print(e)
    print('Something went wrong, bye')
