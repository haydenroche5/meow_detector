from tensorflow.keras.models import load_model
from clean import downsample_mono, trim_silence
from kapre.time_frequency import Melspectrogram
from kapre.utils import Normalization2D
import numpy as np
from glob import glob
import argparse
import os
import tensorflow as tf
from scipy.io import wavfile
from pathlib import Path
import pyaudio
import signal
import sys
import queue
from datetime import datetime
import librosa

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

error_codes = dict([(pyaudio.paInputUnderflow, 'paInputUnderflow'),
                    (pyaudio.paInputOverflow, 'paInputOverflow'),
                    (pyaudio.paOutputUnderflow, 'paOutputUnderflow'),
                    (pyaudio.paOutputOverflow, 'paOutputOverflow'),
                    (pyaudio.paPrimingOutput, 'paPrimingOutput')])

audio_queue = queue.SimpleQueue()

done = False


def signal_handler(sig, frame):
    global done
    print('You pressed Ctrl+C!')
    done = True


signal.signal(signal.SIGINT, signal_handler)


def audio_callback(in_data, frame_count, time_info, status):
    if status in error_codes.keys():
        raise Exception('Error: {}.'.format(error_codes[status]))

    audio_data = np.frombuffer(in_data, dtype=np.int16)
    audio_queue.put_nowait(audio_data)

    return (None, pyaudio.paContinue)


def start_detector(args):
    record_seconds = 3 * args.dt  # get 3 samples at a time
    chunk_size = int(args.sr * record_seconds)
    pa_format = pyaudio.paInt16
    channels = 1

    p = pyaudio.PyAudio()

    model = load_model(args.model,
                       custom_objects={
                           'Melspectrogram': Melspectrogram,
                           'Normalization2D': Normalization2D
                       })

    stream = p.open(format=pa_format,
                    channels=channels,
                    rate=args.sr,
                    input=True,
                    frames_per_buffer=chunk_size,
                    stream_callback=audio_callback)

    classes = ['meow', 'not_meow']

    print('Starting detection loop.', flush=True)

    stream.start_stream()

    while stream.is_active():
        while not audio_queue.empty():
            audio_data = audio_queue.get()
            clean_audio_data = trim_silence(audio_data, args.sr,
                                            args.silence_threshold)
            step = int(args.sr * args.dt)
            batch = []

            for i in range(0, clean_audio_data.shape[0], step):
                sample = clean_audio_data[i:i + step]
                sample = sample.reshape(1, -1)

                if sample.shape[1] < step:
                    continue

                batch.append(sample)

            if len(batch) == 0:
                continue

            X_batch = np.array(batch)
            y_pred = model.predict(X_batch)

            for sample, meow_prob in zip(batch, y_pred[:, 0]):
                if meow_prob > args.meow_threshold:
                    print(
                        'Meow detected with probability {}.'.format(meow_prob),
                        flush=True)
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    dst_path = Path(args.save_dir,
                                    'meow_{}.wav'.format(timestamp))
                    sample = np.squeeze(sample)

                    wavfile.write(dst_path, args.sr, sample)

            # is_meow = (y_pred[:, 0] > args.meow_thresold).any()
            # print(y_pred[:, 0], classes[0 if is_meow else 1], flush=True)

            if done:
                print('Stopping detector.')
                return

    if done:
        print('Stopping detector.')
        return


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Detect meows.')
    parser.add_argument('--model',
                        type=str,
                        default='models/lstm.h5',
                        help='Model file to use for predictions.')
    parser.add_argument('--dt',
                        type=float,
                        default=0.3,
                        help='Length of audio to pass to the model.')
    parser.add_argument('--sr',
                        type=int,
                        default=16000,
                        help='Sample rate to use for recording.')
    parser.add_argument('--silence-threshold',
                        dest='silence_threshold',
                        type=int,
                        default=12,
                        help='dB threshold passed to librosa.effects.trim')
    parser.add_argument(
        '--meow-threshold',
        dest='meow_threshold',
        type=float,
        default=0.5,
        help='Probability threshold for a detection to be considered a meow.')
    parser.add_argument('--save-dir',
                        dest='save_dir',
                        type=str,
                        required=True,
                        help='Directory to save meow detections in.')

    start_detector(parser.parse_args())
