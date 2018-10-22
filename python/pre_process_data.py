#!/usr/bin/python

import entropy_measures as em
import pandas as pd
import numpy as np
import gcsfs
import gc
#from scipy.sparse.linalg import svds
from scipy import stats

import pdb
from statsmodels import robust

'''
This code reads from google storage bucket and preprocess the data using as information
the amoun of stream time (lenght) of user. The data wil composed of users who listem
between one  and seven hours.

The entropy features will suffer a cuttof based on the median absolute deviation on the left
tail

The main function of this file is prepare_data which returns a dataframe with the user features
'''


# Same idea of a z-score but for the median
def median_zscore(x):
    return(1.4826 * ((x - np.median(x)) / robust.mad(x)))

# Assuming that the authentication is already set on the computer
# this function will load from the bucket
def load_files_from_gs():

    blobs = [
        '56cd13f3-cd7e-4ead-88af-623e391ec8d8/users/2017/09/09',
        '56cd13f3-cd7e-4ead-88af-623e391ec8d8/tracks/2017/09/09',
        '56cd13f3-cd7e-4ead-88af-623e391ec8d8/streams/2017/09/09/allcountries'
    ]

    fs = gcsfs.GCSFileSystem(project="56cd13f3-cd7e-4ead-88af-623e391ec8d8")

    with fs.open(blobs[0]) as f:
        users = pd.read_json(f, lines=True, compression='gzip')
        # users.to_csv('./users.csv.gz', index=False, compression='gzip')
    with fs.open(blobs[1]) as f:
        tracks = pd.read_json(f, lines=True, compression='gzip')
        # tracks.to_csv('./tracks.csv.gz', index=False, compression='gzip')
    with fs.open(blobs[2]) as f:
        streams = pd.read_json(f, lines=True, compression='gzip')
        # streams.to_csv('./streams.csv.gz', index=False, compression='gzip')
    return users, tracks, streams


# . Aux file just to avoid the loading from the bucket
# .
def load_files_from_disk(path1, path2, path3):
    try:
        users = pd.read_csv(path1, compression='gzip')
        tracks = pd.read_csv(path2, compression='gzip')
        streams = pd.read_csv(path3, compression='gzip')
        streams.timestamp = streams.timestamp.astype(np.datetime64)
        return users, tracks, streams
    except Exception as e:
        print(e)
        return users, tracks, streams


def process(users, streams):
    # Feature for number of songs per artist
    # recalculating total songs and streaming time
    streaming_time = streams.groupby('user_id').length.sum()
    total_songs = streams.groupby('user_id').size()
    indexes = total_songs.index.values
    print('Sample size of {N}'.format(N=indexes.size))


    user_artist = streams['user_id album_artist'.split()]
    user_artist = user_artist.drop_duplicates()
    counts = user_artist.groupby(['user_id']).album_artist.size()
    # total songs per artists listened
    songs_per_artist = total_songs/counts
    #import pdb; pdb.set_trace()


    # features for length of streaming
    print('Creating features based on the length of stream')
    total_length = streams.groupby('user_id').length.sum()
    mean_length =  streams.groupby('user_id').length.mean()
    std_length = streams.groupby('user_id').length.std().fillna(0.0)
    cv_length =  std_length/mean_length

    #import pdb; pdb.set_trace()
    #cv_length =  mean_length
    #std_length = streams.groupby('user_id').length.std()
    # Maybe use the coefficient of variation instead ...
    #length_cvar = std_length/cv_length

    # features for unique tracks listed
    print('Create tracks features')
    num_unique_tracks = streams.groupby('user_id').track_id.nunique()
    #repetition_rate = (total_songs - num_unique_tracks)
    repetition_rate = (total_songs - num_unique_tracks)/total_songs

    # . Dummy variable for device switching along  the day
    device_switch = (streams.groupby('user_id').device_type.nunique() > 1).astype('int')

    # features related to entropy and mean timestamp interval .
    print('Create entropy features')
    timestamp_entropy = []
    length_entropy = []
    cv_timestamp = []
    intervals = []
    # .
    for user_id, group in streams.groupby('user_id'):
        group = group.sort_values('timestamp')
        #import pdb; pdb.set_trace()
        time_diference = group.timestamp - group.timestamp.shift()
        # getting mean interval time, sort interval, sort lenghts
        mean = ((time_diference[1:]/1e+9).astype(int).values).mean()
        std = np.std((time_diference[1:]/1e+9).astype(int).values)
        if (mean  == 0.0 or std == 0.0):
            cv = 0.0
        else:
            cv = mean/std
        interval = np.sort((time_diference[1:]/1e+9).astype(int).values)
        for i in interval:
            intervals.append(i)
        length = np.sort(group.length.values)
        # import pdb; pdb.set_trace()
        # adding to the lists
        cv_timestamp.append(cv)
        timestamp_entropy.append(em.ApEn(interval, 2, 3))
        length_entropy.append(em.ApEn(length, 2, 3))
        #import pdb; pdb.set_trace()
    # . creating data frame for upload
    user_features_df = pd.DataFrame({
        'songs_per_artist': songs_per_artist,
        'total_length':total_length,
        'cv_length':cv_length,
        'device_switch':device_switch,
        'repetition_rate':repetition_rate,
        'cv_timestamp':cv_timestamp,
        'timestamp_entropy':timestamp_entropy,
        'length_entropy':length_entropy,
        'entropy':(np.array(timestamp_entropy) + np.array(length_entropy)),
        'total_songs':total_songs
    })
    user_features_df.index = indexes
    return(user_features_df)

# main data which filter in or out user based on their IDs and recompute features
def prepare_data():
    print('Loading files')
    users, tracks, streams = load_files_from_gs() # unmcoment for final version
    # users, tracks, streams = load_files_from_disk   (
    #     '~/fdmusic/downloaded/users.csv.gz',
    #         '~/fdmusic/downloaded/tracks.csv.gz',
    #         '~/fdmusic/downloaded/streams.csv.gz'
    # )

    print('Merging stream log with track files')
    # adding track information with the log
    streams = streams.merge(tracks, on='track_id')
    tracks = None; gc.collect()


    # The users that will build up the model will have at least 15 streams
    print('Using users in the range of 3600 seconds and 25200 seconds total stream lenght')
    # A quick google serch suggest that the average music listen per day is around 4.5 hors a day
    # I'm taking a window of 1.5 hours and 6 hours to use as sample
    # remove those who listen less than 1 hour
    streaming_time = streams.groupby('user_id').length.sum()
    valid_users = streaming_time[streaming_time >= 3600 ].index.values
    users = users[users.user_id.isin(valid_users)]
    streams = streams[streams.user_id.isin(valid_users)]
    # remove those who listen more than 7 hors
    streaming_time = streams.groupby('user_id').length.sum()
    valid_users = streaming_time[streaming_time <= 25200 ].index.values
    users = users[users.user_id.isin(valid_users)]
    streams = streams[streams.user_id.isin(valid_users)]


    # . Processing data the first time
    user_features_df = process(users, streams)
    users = users.set_index('user_id')
    users = users.loc[user_features_df.index.values, :]
    user_features_df = pd.concat([user_features_df, users], axis = 1)
    #user_features_df.to_csv('~/fdmusic/uploaded/user_features_P1.csv.gz', compression='gzip')


    # calculate z-scores for songs/artists cv_lengh_dist cv_timestamp entropy
    # perhaps the z-score should be based on the median since
    zscore1 = median_zscore(user_features_df.entropy.values)
    zscore2 = median_zscore(user_features_df.cv_length.values)
    to_remove = user_features_df.index.values[zscore1 < -2].tolist() + user_features_df.index.values[zscore2 < -2].tolist()
    to_remove = list(set(to_remove))



    # recalculating valid users
    valid_users = user_features_df.loc[~user_features_df.index.isin(to_remove), :].index.values



    # re - indexing
    users = users.loc[valid_users, :]
    streams = streams[streams.user_id.isin(valid_users)]



    # . Removing based on the entropy tail
    user_features_df = process(users, streams)
    users = users.loc[user_features_df.index.values, :]
    user_features_df = pd.concat([user_features_df, users], axis = 1)
    # user_features_df.to_csv('~/fdmusic/uploaded/user_features_P2.csv.gz', compression='gzip')


    # upload the data to the bucket:
    return (user_features_df)
    #pdb.set_trace()
