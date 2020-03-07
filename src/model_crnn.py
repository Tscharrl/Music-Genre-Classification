from keras import backend as K
from keras.layers import Input, Dense, Activation, Dropout, Reshape, Permute
from keras.layers.convolutional import Conv2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import ELU
from keras.layers.convolutional import MaxPooling2D
from keras.layers.recurrent import GRU
from keras.models import Model


def get_crnn_model():
    # Determine input shape
    if K.image_dim_ordering() == 'th':
        input_shape = (1, 20, 1250)
    else:
        input_shape = (20, 1250, 1)

    # Determine input axis
    if K.image_dim_ordering() == 'th':
        channel_axis = 1
        mfcc_axis = 2
        time_axis = 3
    else:
        channel_axis = 3
        mfcc_axis = 1
        time_axis = 2

    mfcc_input = Input(shape=input_shape)

    # Input block
    x = BatchNormalization(axis=mfcc_axis, name='bn0_mfcc')(mfcc_input)

    # Conv block 1
    x = Conv2D(32, (3, 3), name='conv1', padding='same')(x) # (320, 1250, 32)
    x = BatchNormalization(axis=channel_axis, name='bn1')(x)
    x = ELU()(x)
    x = MaxPooling2D(pool_size=(2, 2), name='pool1')(x) # (10, 625, 32)
    x = Dropout(0.1, name='dropout1')(x)

    # Conv block 2
    x = Conv2D(64, (3, 3), name='conv2', padding='same')(x) # (10, 625, 64)
    x = BatchNormalization(axis=channel_axis, name='bn2')(x)
    x = ELU()(x)
    x = MaxPooling2D(pool_size=(2, 5), name='pool2')(x) # (5, 125, 64)
    x = Dropout(0.1, name='dropout2')(x)

    # Conv block 3
    x = Conv2D(128, (3, 3), name='conv3', padding='same')(x)# (5, 125, 128)
    x = BatchNormalization(axis=channel_axis, name='bn3')(x)
    x = ELU()(x)
    x = MaxPooling2D(pool_size=(1, 5), name='pool3')(x) # (5, 25, 128)
    x = Dropout(0.1, name='dropout3')(x)

    # Conv block 4
    x = Conv2D(64, (3, 3), name='conv4', padding='same')(x) # (5, 25, 64)
    x = BatchNormalization(axis=channel_axis, name='bn4')(x)
    x = ELU()(x)
    x = MaxPooling2D(pool_size=(5, 5), name='pool4')(x) # (1, 5, 64)
    x = Dropout(0.1, name='dropout4')(x)

    # Reshaping
    if K.image_dim_ordering() == 'th':
        x = Permute((3, 1, 2))(x)
    x = Reshape((5, 64))(x)

    # GRU block
    x = GRU(32, return_sequences=True, name='gru1')(x)
    x = GRU(32, return_sequences=False, name='gru2')(x)
    x = Dropout(0.3)(x)

    # Conv block 5
    # x = Conv2D(64, (3, 3), name='conv5', padding='same')(x)
    # x = BatchNormalization(axis=channel_axis, name='bn5')(x)
    # x = ELU()(x)
    # x = MaxPooling2D(pool_size=(4, 4), name='pool5')(x)

    # Output
    # x = Flatten()(x)
    x = Dense(10, activation='softmax', name='output')(x)

    # Create model
    model = Model(mfcc_input, x)

    # Compile model
    model.compile(optimizer='adam', 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])

    return model

def print_model(model):
    print(model.summary())