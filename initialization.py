import os

from Utils import Utils
from audioManager import split_file_in_three_sec, write_spec_file


def init_music_specs():
    if not os.path.exists(Utils.target_directory_path):
        os.mkdir(Utils.target_directory_path)

    for music_genre_directory in Utils.get_music_genre_directories():
        # Create directory if not exists
        if not os.path.exists(Utils.target_directory_path + music_genre_directory):
            os.mkdir(Utils.target_directory_path + music_genre_directory)
        for music_file in os.listdir(Utils.target_directory_path + music_genre_directory):
            file_split = split_file_in_three_sec(Utils.directory_path + music_genre_directory + "/" + music_file)
            for index, (x, sr) in enumerate(file_split):
                music_file_name = "".join(music_file.split(".")[:2]) + str(index)
                music_file_path = Utils.target_directory_path + music_genre_directory + "/" + music_file_name
                write_spec_file(x, sr, music_file_path)
