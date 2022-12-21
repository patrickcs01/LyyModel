# https://www.tensorflow.org/datasets/overview
# import tensorflow_datasets as tfds
# import ssl
# import os
#
# os.environ['NO_GCE_CHECK'] = 'true'
#
# os.environ['http_proxy'] = 'http://127.0.0.1:7890'
# os.environ['https_proxy'] = 'https://127.0.0.1:7890'
# ssl._create_default_https_context = ssl._create_unverified_context
#
# ratings = tfds.load('movielens/100k-ratings', split="train")
# movies = tfds.load('movielens/100k-movies', split="train")
# movies = movies.map(lambda x: x["movie_title"])

import pandas as pd
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

from model.reco.dcn_layer import DCNLayer
from model.reco.wide_deep_layer import WideDeepLayer

df = pd.read_csv('../dataset/movies.csv')
column = ['id', 'popularity', ]
feature = column
x_data = df[feature]
y_data = df['vote_count']
x_data = np.asarray(x_data)
y_data = np.asarray(y_data)
print(x_data)
print(y_data)

input_shape = x_data.shape
print(input_shape)

id = tf.keras.Input(shape=(1,))
popularity = tf.keras.Input(shape=(1,))


outputs = DCNLayer(layer_name="w&d", deep_activation="relu")(popularity)

model = tf.keras.Model([id, popularity], outputs)
model.compile(optimizer="adam", loss='mse')

model.fit([x_data[:,0], x_data[:,1]], y_data, epochs=100, )


y_pred = model.predict([x_data[:,0], x_data[:,1]])
print(y_data)
print(y_pred)