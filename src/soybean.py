from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf

from PIL import Image
from PIL import ImageFilter

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true',
                        help='Train model')

    return parser.parse_args()

def load_data(data_path):
    print("Loading data from {}".format(data_path))

    x_data = []
    y_data = []

    labels = pd.read_csv(data_path + '/TrainAnnotations.csv', header=0, index_col=0, squeeze=True).to_dict()

    image_data_path = data_path + '/images/TrainData'
    files = [f for f in os.listdir(image_data_path)]
    for f in files:
        img = Image.open(image_data_path + "/" + f).convert('L')

        x_data.append(tf.convert_to_tensor(np.array(img.filter(ImageFilter.FIND_EDGES)), dtype=tf.int32))
        y_data.append(labels[f])

    print("Finished loading data.")

    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data,
                                                        train_size=0.8,
                                                        random_state=138,
                                                        shuffle=True,
                                                        stratify=y_data)

    return (tf.convert_to_tensor(x_train, dtype=tf.float32), tf.convert_to_tensor(y_train, dtype=tf.int32)), \
           (tf.convert_to_tensor(x_test, dtype=tf.float32), tf.convert_to_tensor(y_test, dtype=tf.int32))


def train():
    (x_train, y_train), (x_test, y_test) = load_data('../data')
    x_train, x_test = tf.realdiv(x_train, 255.0), tf.realdiv(x_test, 255.0)

    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(480, 640)),
        tf.keras.layers.Dense(128, activation='relu'),
        # tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(5)
    ])

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    print('Fitting model...')
    model.fit(x_train, y_train, epochs=10)

    print('Evaluating model...')
    model.evaluate(x_test, y_test, verbose=2)

    probability_model = tf.keras.Sequential([
        model,
        tf.keras.layers.Softmax()
    ])

    print("Result: {}".format(probability_model(x_test[:5])))


if __name__ == '__main__':
    FLAGS = get_args()
    if FLAGS.train:
        train()
