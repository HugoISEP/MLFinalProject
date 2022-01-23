import csv
import os
import pathlib

import librosa
from matplotlib import pyplot as plt

from Utils import Utils
from audioManager import split_file_in_three_sec, write_spec_file, write_waveplot_file
import numpy as np


def init_music_specs():
    if not os.path.exists(Utils.target_directory_path):
        os.mkdir(Utils.target_directory_path)

    for music_genre_directory in Utils.get_music_genre_directories():
        print('helo', Utils.directory_path + music_genre_directory)

        # Create directory if not exists
        if not os.path.exists(Utils.target_directory_path + music_genre_directory):
            os.mkdir(Utils.target_directory_path + music_genre_directory)
        for music_file in os.listdir(Utils.directory_path + music_genre_directory):
            print(music_file)
            file_split = split_file_in_three_sec(Utils.directory_path + music_genre_directory + "/" + music_file)
            for index, (x, sr) in enumerate(file_split):
                music_file_name = "".join(music_file.split(".")[:2]) + str(index)
                music_file_path = Utils.target_directory_path + music_genre_directory + "/" + music_file_name
                write_spec_file(x, sr, music_file_path)


def csvDataset():
    header = 'chroma_stft spectral_centroid spectral_bandwidth rolloff zero_crossing_rate'
    for i in range(1, 21):
        header += f' mfcc{i}'
    header += ' label'
    header = header.split()

    file = open('data.csv', 'w', newline='')
    with file:
        writer = csv.writer(file)
        writer.writerow(header)
    for music_genre_directory in Utils.get_music_genre_directories():
        for music_file in os.listdir(Utils.directory_path + music_genre_directory):
            specPath = Utils.directory_path + music_genre_directory + '/' + music_file
            y, sr = librosa.load(specPath, mono=True, duration=30)
            chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
            spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
            spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
            rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
            zcr = librosa.feature.zero_crossing_rate(y)
            mfcc = librosa.feature.mfcc(y=y, sr=sr)
            to_append = f' {np.mean(chroma_stft)} {np.mean(spec_cent)} {np.mean(spec_bw)} {np.mean(rolloff)} {np.mean(zcr)}'
            for e in mfcc:
                to_append += f' {np.mean(e)}'

            to_append += f' {music_file.split(".")[0]}'
            file = open('data.csv', 'a', newline='')
            with file:
                writer = csv.writer(file)
                writer.writerow(to_append.split())

csvDataset()
