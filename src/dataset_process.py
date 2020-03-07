import numpy as np
import random

from preprocessing import generate_features_and_labels


def get_train_test_set(type):
    features, labels = generate_features_and_labels()

    features = np.array(features).reshape(1000, 20, 1250)
    labels = np.array(labels).reshape(1000, 10)

    # shuffle feature set and label set
    randnum = random.randint(0,100)
    random.seed(randnum)
    random.shuffle(features)
    random.seed(randnum)
    random.shuffle(labels)

    # split to train set and test set
    training_percent = 0.8
    split_index = int(len(features)*training_percent)
    if type=='train' or type=='both':
        train_input, train_labels = features[:split_index,:,:], labels[:split_index,:]
        print('Shape of train set    : ', np.shape(train_input))
        print('Shape of train labels : ', np.shape(train_labels))
    if type=='test' or type=='both':
        test_input, test_labels = features[split_index:,:,:], labels[split_index:,:]
        print('Shape of test set     : ', np.shape(test_input))
        print('Shape of test labels : ', np.shape(test_labels))

    if type=='train':
        return train_input, train_labels, split_index
    elif type=='test':
        return test_input, test_labels, split_index
    elif type=='both':
        return train_input, train_labels, test_input, test_labels, split_index
    else:
        print('[Error] from get_train_test_set.py: "type" must be "train"/"test"/"both"!')