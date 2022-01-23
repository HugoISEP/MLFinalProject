import librosa
import numpy as np
import pandas as pd
import pyaudio
import wave
import librosa.display
import tensorflow as tf
from matplotlib import pyplot as plt
import csv

from sklearn.preprocessing import LabelEncoder, StandardScaler

from Utils import Utils
from audioManager import write_spec_file

recording_file = "file.wav"
recording_image = "file.png"


def predict_audio_record():
    record_audio()
    xo = []
    xo.append(createCsvFile())
    x, sr = librosa.load(recording_file)
    write_spec_file(x, sr, recording_file.split(".")[0])
    cnnModel = tf.keras.models.load_model('saved_model/test')
    mlpModel = tf.keras.models.load_model('saved_model/test2')
    cnnProbability_model = tf.keras.Sequential([cnnModel, tf.keras.layers.Softmax()])
    mlpProbability_model = tf.keras.Sequential(mlpModel)
    image = Utils.format_image(recording_image)
    cnnPredictions = cnnProbability_model.predict(image)
    mlpPredictions = mlpProbability_model.predict(xo)
    print("hello how are u",mlpPredictions[0].shape, np.argmax(mlpPredictions[0]), np.sum(mlpPredictions[0]), mlpPredictions[0])
    plot_value_array(cnnPredictions, mlpPredictions)


def record_audio():
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44100
    CHUNK = 1024
    RECORD_SECONDS = 10
    WAVE_OUTPUT_FILENAME = recording_file

    audio = pyaudio.PyAudio()

    # start Recording
    stream = audio.open(format=FORMAT, channels=CHANNELS,
                        rate=RATE, input=True,
                        frames_per_buffer=CHUNK)
    print("recording...")
    frames = []

    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)
    print("finished recording")

    # stop Recording
    stream.stop_stream()
    stream.close()
    audio.terminate()

    waveFile = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    waveFile.setnchannels(CHANNELS)
    waveFile.setsampwidth(audio.get_sample_size(FORMAT))
    waveFile.setframerate(RATE)
    waveFile.writeframes(b''.join(frames))
    waveFile.close()


def plot_value_array(predictions_array, predictions2_array):
    plt.figure(figsize=(10, 5))
    print('aah non',predictions_array[0])
    plt.bar(Utils.get_music_genre_directories(), predictions_array[0])
    plt.bar(Utils.get_music_genre_directories(), predictions2_array[0])
    plt.xlabel("Music genres")
    plt.ylabel("Probability (SoftMax)")
    plt.title("Probability of the music recorded")
    plt.show()


def createCsvFile():
        path = "./file.wav"
        y, sr = librosa.load(path, mono=True, duration=10)
        chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
        spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
        spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        zcr = librosa.feature.zero_crossing_rate(y)
        mfcc = librosa.feature.mfcc(y=y, sr=sr)
        to_append = f' {np.mean(chroma_stft)} {np.mean(spec_cent)} {np.mean(spec_bw)} {np.mean(rolloff)} {np.mean(zcr)}'
        for e in mfcc:
            to_append += f' {np.mean(e)}'
        imageValue = []
        for cellValue in to_append.split():
            imageValue.append(float(cellValue))
        return imageValue


createCsvFile()