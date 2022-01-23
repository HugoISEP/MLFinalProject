import os

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image


class Utils:
    files_to_exclude = [".DS_Store"]
    directory_path = "Data/genres_original/"
    target_directory_path = "Data/spec_results/"

    @staticmethod
    def format_image(image_path: str):
        img_height = 180
        img_width = 180
        img = image.load_img(image_path, target_size=(img_width, img_height))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        return x

    @staticmethod
    def get_music_genre_directories():
        return [directory for directory in sorted(os.listdir(Utils.directory_path))
               if not Utils.files_to_exclude.__contains__(directory)]

    @staticmethod
    def get_music_files_from_directory(genre_directory: str):
        return [file for file in sorted(os.listdir(genre_directory))
                if not Utils.files_to_exclude.__contains__(file)]
