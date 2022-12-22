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

from feature.feature_embedding import encode_inputs, create_model_inputs
from feature.feature_tools import FeatureTools
from model.base.dense_layer import DenseLayer
from model.reco.dcn_layer import DCNLayer
from model.reco.wide_deep_layer import WideDeepLayer

df = pd.read_csv('../dataset/movies.csv')
# print(df.columns)
# print(df.director)
feature_head_name = ['budget', 'popularity', 'vote_count']
numeric_feature_names = ['budget', 'popularity']
categorical_features_with_vocabulary = {
    # "homepage": list(df["homepage"].unique()),
    # "id": list(df["id"].unique()),
    # "original_language": list(df["original_language"].unique()),
    # 'status': list(df["status"].unique()),
    # 'director': list(df["director"].unique()),
}
target_feature_labels = list(df["vote_count"].unique())
feature_tools = FeatureTools(feature_head_name=feature_head_name,
                             numeric_feature_names=numeric_feature_names,
                             categorical_features_with_vocabulary=categorical_features_with_vocabulary,
                             target_name='vote_count',
                             target_feature_labels=target_feature_labels)

column = feature_tools.all_feature_name
feature = column
x_data = df[feature]
y_data = df[feature_tools.target_name]
x_data = np.asarray(x_data).astype(np.float32)
y_data = np.asarray(y_data)

input_shape = x_data.shape
print(input_shape)
max_id_length = df.id.max()
max_original_language_length = df.original_language.count()

inputs = create_model_inputs(feature_tools)


train_dataset = tf.data.experimental.make_csv_dataset(
    "../dataset/movies.csv",
    batch_size=32,
    # column_names=feature_tools.feature_head,
    # column_defaults=feature_tools.column_defaults,
    label_name=feature_tools.target_name,
    num_epochs=1,
    header=True,
    shuffle=True,
)


wide = encode_inputs(inputs, feature_tools)
wide = tf.keras.layers.BatchNormalization()(wide)

deep = encode_inputs(inputs, feature_tools)
for units in [32, 32]:
    deep = tf.keras.layers.Dense(units)(deep)
    deep = tf.keras.layers.BatchNormalization()(deep)
    deep = tf.keras.layers.ReLU()(deep)
    deep = tf.keras.layers.Dropout(0.4)(deep)

merged = tf.keras.layers.concatenate([wide, deep])
outputs = tf.keras.layers.Dense(units=1)(merged)
model = tf.keras.Model(inputs=inputs, outputs=outputs)

# model = tf.keras.Model(input, outputs)
model.compile(optimizer="adam", loss='mse')

model.fit(train_dataset, epochs=1000, )

# _, accuracy = model.evaluate(train_dataset, verbose=0)

# print(f"Test accuracy: {round(accuracy * 100, 2)}%")