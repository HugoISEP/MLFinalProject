import librosa
import librosa.display
import numpy as np
from matplotlib import pyplot as plt
import scipy.io.wavfile as wavfile

def split_file_in_three_sec(file_path: str):
    file_list = []
    y, sr = librosa.load(file_path)
    file_duration = round(librosa.get_duration(y=y, sr=sr))
    for start_time in range(0, file_duration, 10):
        x, sr = librosa.load(file_path, duration=10, offset=start_time)
        file_list.append((x, sr))
    return file_list


def write_spec_file(x, sr, music_file_path):
    hop_length = 1024
    X = librosa.stft(x, hop_length=hop_length)
    Xdb = librosa.amplitude_to_db(abs(X), ref=np.max)
    plt.figure(figsize=(14, 5))

    librosa.display.specshow(Xdb, sr=sr, hop_length=hop_length, x_axis='time', y_axis='log')
    plt.colorbar()
    plt.clim(0, -80)
    plt.savefig(music_file_path)
    plt.close()


def write_waveplot_file(x, sr, music_file_path):
    plt.figure(figsize=(14, 5))
    librosa.display.waveplot(x, sr=sr)
    plt.clim(-1, 1)
    plt.savefig(music_file_path)
    plt.close()