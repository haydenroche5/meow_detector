from tensorflow.keras import layers
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from kapre.time_frequency import Melspectrogram
from kapre.utils import Normalization2D


def Conv2D(input_length, num_classes):
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
    x = layers.Conv2D(8,
                      kernel_size=(7, 7),
                      activation='tanh',
                      padding='same',
                      name='conv2d_tanh')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2),
                            padding='same',
                            name='max_pool_2d_1')(x)
    x = layers.Conv2D(16,
                      kernel_size=(5, 5),
                      activation='relu',
                      padding='same',
                      name='conv2d_relu_1')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2),
                            padding='same',
                            name='max_pool_2d_2')(x)
    x = layers.Conv2D(16,
                      kernel_size=(3, 3),
                      activation='relu',
                      padding='same',
                      name='conv2d_relu_2')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2),
                            padding='same',
                            name='max_pool_2d_3')(x)
    x = layers.Conv2D(32,
                      kernel_size=(3, 3),
                      activation='relu',
                      padding='same',
                      name='conv2d_relu_3')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2),
                            padding='same',
                            name='max_pool_2d_4')(x)
    x = layers.Conv2D(32,
                      kernel_size=(3, 3),
                      activation='relu',
                      padding='same',
                      name='conv2d_relu_4')(x)
    x = layers.Flatten(name='flatten')(x)
    x = layers.Dropout(rate=0.2, name='dropout')(x)
    x = layers.Dense(64,
                     activation='relu',
                     activity_regularizer=l2(0.001),
                     name='dense')(x)

    o = layers.Dense(num_classes, activation='softmax', name='softmax')(x)

    model = Model(inputs=i, outputs=o, name='2d_convolution')

    return model
