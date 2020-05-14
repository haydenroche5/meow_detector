from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from kapre.time_frequency import Melspectrogram
from kapre.utils import Normalization2D


def LSTM(input_length, num_classes):
    i = layers.Input(shape=(1, input_length), name='input')
    x = Melspectrogram(n_dft=512,
                       n_hop=160,
                       padding='same',
                       sr=16000,
                       n_mels=128,
                       fmin=0.0,
                       fmax=16000 / 2,
                       power_melgram=1.0,
                       return_decibel_melgram=True,
                       trainable_fb=False,
                       trainable_kernel=False,
                       name='melbands')(i)
    x = Normalization2D(str_axis='batch', name='batch_norm')(x)
    x = layers.Permute((2, 1, 3), name='permute')(x)
    x = layers.TimeDistributed(layers.Reshape((-1, )), name='reshape')(x)
    s = layers.TimeDistributed(layers.Dense(64, activation='tanh'),
                               name='td_dense_tanh')(x)
    x = layers.Bidirectional(layers.LSTM(32, return_sequences=True),
                             name='bidirectional_lstm')(s)
    x = layers.concatenate([s, x], axis=2, name='skip_connection')
    x = layers.Dense(64, activation='relu', name='dense_1_relu')(x)
    x = layers.MaxPooling1D(name='max_pool_1d')(x)
    x = layers.Dense(32, activation='relu', name='dense_2_relu')(x)
    x = layers.Flatten(name='flatten')(x)
    x = layers.Dropout(rate=0.2, name='dropout')(x)
    x = layers.Dense(32,
                     activation='relu',
                     activity_regularizer=l2(0.001),
                     name='dense_3_relu')(x)

    o = layers.Dense(num_classes, activation='softmax', name='softmax')(x)

    model = Model(inputs=i, outputs=o, name='long_short_term_memory')

    return model
