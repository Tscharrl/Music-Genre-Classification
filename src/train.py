import os
import numpy as np
from keras.models import load_model
from keras.models import model_from_json

from model_cnn import get_cnn_model
from model_crnn import get_crnn_model

import keras.backend as K
# from data_split_to_train_test import get_train_test_set


def train(epo, bat, path, exec_time, train_input, train_labels, test_input, test_labels, split_index):
    # Reshape train and test set
    train_input = np.expand_dims(train_input, axis=split_index-1)
    test_input = np.expand_dims(test_input, axis=1000-split_index-1)

    # Get model
    if os.path.exists('../model/CNN.h5'):
        model = load_model('../model/CNN.h5')
        # Former loss and accuracy
        loss, acc = model.evaluate(test_input, test_labels, batch_size=32)
        print("Before this train:\nLoss: %.4f\nAccuracy: %.4f" % (loss, acc))
    else:
        # model = get_cnn_model()
        model = get_crnn_model()

    K.set_value(model.optimizer.lr, 0.001)

    # Train
    history = model.fit(train_input, train_labels, epochs=epo, batch_size=bat,
          validation_split=0.2)

    # Evaluate
    loss, acc = model.evaluate(test_input, test_labels, batch_size=bat)

    # Print loss and accuracy
    print("Done!")
    print("Loss: %.4f\nAccuracy: %.4f" % (loss, acc))

    # Save weights
    model.save_weights('../model/CNN_weights.h5')
    # Save model
    model.save('../model/CNN.h5')

    # Save model to log
    model.save_weights(path + '/CNN_weights.h5')
    model.save(path + '/CNN.h5')

    # Save loss&acc to log
    exist = os.path.exists('../log/loss&acc.txt')
    f = open('../log/loss&acc.txt', 'a')
    if exist:
        f.write('\n\n')
    f.write('TIME: {}\nloss: {}\nacccuracy: {}'.format(exec_time, loss, acc))
    f.close()

    # Save history to log
    his_f = open(path + '/history.txt', 'a')
    his_f.write(str(history.history))
    his_f.close()

    return history