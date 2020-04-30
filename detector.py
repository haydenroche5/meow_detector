import pyaudio
import wave
import time
import numpy as np
import librosa
import joblib

model = joblib.load('models/model.pkl')

threshold = 0.5

RATE = 22050
RECORD_SECONDS = 3
CHUNK = RATE * RECORD_SECONDS
FORMAT = pyaudio.paFloat32
CHANNELS = 1

p = pyaudio.PyAudio()


def callback(in_data, frame_count, time_info, status):
    audio_data = np.frombuffer(in_data, dtype=np.float32)

    mfcc = librosa.feature.mfcc(y=audio_data, sr=RATE, n_mfcc=40)
    mfcc_avgs = np.mean(mfcc, axis=1).reshape(1, -1)

    pred = model.predict(mfcc_avgs)
    print(pred)

    return (None, pyaudio.paContinue)


stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK,
                stream_callback=callback)

print('* recording')

stream.start_stream()

while stream.is_active():
    time.sleep(0.1)

stream.stop_stream()
stream.close()

p.terminate()
