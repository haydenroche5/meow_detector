import tensorflow as tf
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping
import os
from scipy.io import wavfile
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from models import Conv1D, Conv2D, LSTM
from tqdm import tqdm
from glob import glob
import argparse
import tensorflow as tf
from tensorflow.keras.optimizers import SGD
import json
from datetime import datetime
from util import DataGenerator

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


def get_class(s, src_root):
    s = s.split(src_root)[-1]
    s = s.split('/')[0]
    return s


def main(args):
    with open(args.config, 'r') as f:
        config = json.load(f)

    if config['optimizer'] == 'SGD':
        optimizer = SGD(lr=config['learning_rate'],
                        decay=config['learning_rate'] / config['epochs'],
                        momentum=config['momentum'])
    else:
        raise Exception('Unsupported optimizer: {}.'.format(
            config['optimizer']))

    model_name = str.lower(config['model'])
    if model_name == 'lstm':
        model = LSTM(config['input_length'], 2)
    elif model_name == 'conv1d':
        model = Conv1D(config['input_length'], 2)
    elif model_name == 'conv2d':
        model = Conv2D(config['input_length'], 2)
    else:
        raise Exception('Unsupported model: {}.'.format(config['model']))

    model.compile(loss='binary_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])

    wav_paths = glob('{}/**'.format(args.data_dir), recursive=True)
    wav_paths = [x for x in wav_paths if '.wav' in x]
    classes = sorted(os.listdir(args.data_dir))
    le = LabelEncoder()
    le.fit(classes)
    labels = [get_class(x, args.data_dir) for x in wav_paths]
    labels = le.transform(labels)

    print('CLASSES: ', list(le.classes_))
    print(le.transform(list(le.classes_)))

    wav_train, wav_val, label_train, label_val = train_test_split(
        wav_paths,
        labels,
        test_size=config['validation_split'],
        random_state=0)
    tg = DataGenerator(wav_train,
                       label_train,
                       config['input_length'],
                       len(set(label_train)),
                       batch_size=config['batch_size'])
    vg = DataGenerator(wav_val,
                       label_val,
                       config['input_length'],
                       len(set(label_val)),
                       batch_size=config['batch_size'])

    output_sub_dir = os.path.join(args.output_dir, model_name,
                                  datetime.now().strftime('%Y%m%d_%H%M%S'))
    os.makedirs(output_sub_dir)

    callbacks = [
        EarlyStopping(monitor='val_loss',
                      patience=config['patience'],
                      restore_best_weights=True,
                      verbose=1),
        ModelCheckpoint(filepath=os.path.join(
            output_sub_dir, 'model.{epoch:02d}-{val_loss:.4f}.h5'),
                        monitor='val_loss',
                        save_best_only=True,
                        verbose=1),
        CSVLogger(os.path.join(output_sub_dir, 'epochs.csv'))
    ]

    model.fit(tg,
              validation_data=vg,
              epochs=config['epochs'],
              verbose=1,
              callbacks=callbacks)


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(
        description='Train a meow detection model.')
    arg_parser.add_argument('--data-dir',
                            dest='data_dir',
                            required=True,
                            help='Directory containing training data.')
    arg_parser.add_argument('--output-dir',
                            dest='output_dir',
                            required=True,
                            help='Directory to save training results in.')
    arg_parser.add_argument('--config',
                            dest='config',
                            required=True,
                            help='Path to a JSON config file.')
    main(arg_parser.parse_args())
