import librosa
import librosa.feature

import numpy as np
import glob

from keras.utils.np_utils import to_categorical

def get_features(file):
    # load the song
    y, _ = librosa.load(file)  # sr = 22050 (default sampling rate)

    # compute MFCC for the song
    mfcc = librosa.feature.mfcc(y)  
    # 20(n_mfcc)*1293(t)
    # 1293帧，每帧内提取20维的MFCC
    
    # normalize to [-1, 1]
    mfcc /= np.amax(np.absolute(mfcc))

    # for every song, only 25000 mfcc values is needed
    # actually it's 25860(=1293*20)
    return np.ndarray.flatten(mfcc)[:25000]


def generate_features_and_labels():
    all_features = []
    all_labels = []

    genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
    for genre in genres:
        sound_files = glob.glob('../data/genres/' + genre + '/*.au')
        for file in sound_files:
            feature = get_features(file)
            all_features.append(feature)
            all_labels.append(genre)

            # print progress
            percent = sound_files.index(file) + 1
            print('\r' + 'Processing %d songs in %s genre...'%(len(sound_files), genre) + ' %d%%'%percent, end='', flush=True)

        print('')

    # convert labels to one-hot encoding
    label_uniq_ids, label_row_ids = np.unique(all_labels, return_inverse=True)
    label_row_ids = label_row_ids.astype(np.int32, copy=False)
    onehot_labels = to_categorical(label_row_ids, len(label_uniq_ids))
    
    return np.stack(all_features), onehot_labels