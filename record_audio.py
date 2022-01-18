import librosa
import numpy as np
import pyaudio
import wave
import librosa.display
import tensorflow as tf
from matplotlib import pyplot as plt

from Utils import Utils
from audioManager import write_spec_file

recording_file = "file.wav"
recording_image = "file.png"


def predict_audio_record():
    record_audio()
    x, sr = librosa.load(recording_file)
    write_spec_file(x, sr, recording_file.split(".")[0])
    model = tf.keras.models.load_model('saved_model/test')
    probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
    image = Utils.format_image(recording_image)
    predictions = probability_model.predict(image)
    plot_value_array(predictions)


def record_audio():
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44100
    CHUNK = 1024
    RECORD_SECONDS = 5
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


def plot_value_array(predictions_array):
    plt.figure(figsize=(10, 5))
    plt.bar(Utils.get_music_genre_directories(), predictions_array[0])
    plt.xlabel("Music genres")
    plt.ylabel("Probability (SoftMax)")
    plt.title("Probability of the music recorded")
    plt.show()
