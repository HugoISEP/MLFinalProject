import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import numpy as np
from Utils import Utils
from keras import models
from record_audio import createCsvFile
batch_size = 32
img_height = 180
img_width = 180

data = pd.read_csv('data.csv')

###return music genre
genre_list = data.iloc[:, -1]
encoder = LabelEncoder()
y = encoder.fit_transform(genre_list)

scaler = StandardScaler()
X = scaler.fit_transform(np.array(data.iloc[:, :-1], dtype=float))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

train_ds = tf.keras.utils.image_dataset_from_directory(
    Utils.target_directory_path,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

val_ds = tf.keras.utils.image_dataset_from_directory(
    Utils.target_directory_path,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)


AUTOTUNE = tf.data.AUTOTUNE

train_spec_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_spec_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

def create_mlp():
    model = models.Sequential()
    model.add(tf.keras.layers.Dense(512, activation='relu', input_shape=(X_train.shape[1],)))
    model.add(tf.keras.layers.Dense(256, activation='relu'))
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dense(10, activation='softmax'))
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy']
                  )
    model.fit(X_train, y_train, epochs=200,  batch_size=64, validation_data=(X_test, y_test)
)
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print('test_acc: ', test_acc, y_test)
    model.predict(X_test)
    model.save('saved_model/test2')
create_mlp()

def create_cnn():
    # normalization_layer = tf.keras.layers.Rescaling(1./255)
    # normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    # image_batch, labels_batch = next(iter(normalized_ds))
    # first_image = image_batch[0]
    # Notice the pixel values are now in `[0,1]`.
    # print(np.min(first_image), np.max(first_image))

    num_classes = 10

    model = tf.keras.Sequential([
        tf.keras.layers.Rescaling(1. / 255),
        tf.keras.layers.Conv2D(32, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(32, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(32, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(num_classes)
    ])
    print(train_ds, val_ds)
    model.compile(
      optimizer='adam',
      loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
      metrics=['accuracy'])


    model.fit(
      train_ds,
      validation_data=val_ds,
      epochs=5
    )

