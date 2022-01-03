import os
import librosa
import matplotlib.pyplot as plt
import librosa.display


def init_music_specs():
    directory_path = "Data/genres_original/"
    target_directory = "Data/spec_results/"
    if not os.path.exists(target_directory):
        os.mkdir(target_directory)

    for music_genre_directory in os.listdir(directory_path):
        if not os.path.exists(target_directory + music_genre_directory):
            os.mkdir(target_directory + music_genre_directory)
        for music_file in os.listdir(directory_path + music_genre_directory):
            x, sr = librosa.load(directory_path + music_genre_directory + "/" + music_file)
            X = librosa.stft(x)
            Xdb = librosa.amplitude_to_db(abs(X))

            plt.figure(figsize=(14, 5))
            librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz')
            plt.colorbar()

            music_file_name = "".join(music_file.split(".")[:2])
            plt.savefig(target_directory + music_genre_directory + "/" + music_file_name)
            plt.close()