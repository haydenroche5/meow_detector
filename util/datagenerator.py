# This class comes from this code by Seth Adams: https://github.com/seth814/Audio-Classification.

from tensorflow.keras.utils import Sequence, to_categorical
import numpy as np
from scipy.io import wavfile


class DataGenerator(Sequence):
    def __init__(self,
                 wav_paths,
                 labels,
                 input_length,
                 n_classes,
                 batch_size=32,
                 shuffle=True):
        self.wav_paths = wav_paths
        self.labels = labels
        self.input_length = input_length
        self.n_classes = n_classes
        self.batch_size = batch_size
        self.shuffle = True
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.wav_paths) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size:(index + 1) *
                               self.batch_size]

        wav_paths = [self.wav_paths[k] for k in indexes]
        labels = [self.labels[k] for k in indexes]

        # generate a batch of time data
        X = np.empty((self.batch_size, 1, self.input_length), dtype=np.int16)
        Y = np.empty((self.batch_size, self.n_classes), dtype=np.float32)

        for i, (path, label) in enumerate(zip(wav_paths, labels)):
            rate, wav = wavfile.read(path)
            X[i, ] = wav.reshape(1, -1)
            Y[i, ] = to_categorical(label, num_classes=self.n_classes)

        return X, Y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.wav_paths))
        if self.shuffle:
            np.random.shuffle(self.indexes)