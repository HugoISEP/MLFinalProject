import os
import librosa
import matplotlib.pyplot as plt
import librosa.display
import numpy as np

# Librosa improvments
# 1. fixer l'échelle db -> -50 à 50 dB
# 2. échelle log pour l'axe des Hz
# 3. Diviser le morceau en 3s d'interval
# 4. Ajuster le modèle


directory_path = "Data/genres_original/"
target_directory = "Data/spec_results/"

files_to_exclude = [".DS_Store"]


def init_music_specs():
    if not os.path.exists(target_directory):
        os.mkdir(target_directory)

    for music_genre_directory in os.listdir(directory_path):
        if not files_to_exclude.__contains__(music_genre_directory):
            if not os.path.exists(target_directory + music_genre_directory):
                os.mkdir(target_directory + music_genre_directory)
            for music_file in os.listdir(directory_path + music_genre_directory):

                if not files_to_exclude.__contains__(music_file):
                    file_split = split_file_in_three_sec(directory_path + music_genre_directory + "/" + music_file)
                    for index, (x, sr) in enumerate(file_split):
                        music_file_name = "".join(music_file.split(".")[:2]) + str(index)
                        write_spec_file(x, sr, music_file_name, music_genre_directory)


def split_file_in_three_sec(file_path: str):
    file_list = []
    y, sr = librosa.load(file_path)
    file_duration = round(librosa.get_duration(y=y, sr=sr))
    for start_time in range(0, file_duration, 3):
        x, sr = librosa.load(file_path, duration=3, offset=start_time)
        file_list.append((x, sr))
    return file_list


def write_spec_file(x, sr, music_file_name: str, music_genre_directory: str):
    hop_length = 1024
    X = librosa.stft(x, hop_length=hop_length)
    Xdb = librosa.amplitude_to_db(abs(X), ref=np.max)

    plt.figure(figsize=(14, 5))

    librosa.display.specshow(Xdb, sr=sr, hop_length=hop_length, x_axis='time', y_axis='log')
    plt.colorbar()
    plt.clim(-80, 0)
    plt.savefig(target_directory + music_genre_directory + "/" + music_file_name)
    plt.close()
