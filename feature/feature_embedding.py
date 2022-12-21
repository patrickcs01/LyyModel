import math
import tensorflow as tf
from keras.layers import StringLookup

from feature.feature_tools import FeatureTools


def encode_inputs(inputs, feature_tool: FeatureTools, use_embedding=False):
    assert feature_tool is not None
    encoded_features = []
    for feature_name in inputs:
        if feature_name in feature_tool.categorical_features_names:
            vocabulary = feature_tool.categorical_features_with_vocabulary[feature_name]
            # Create a lookup to convert string values to an integer indices.
            # Since we are not using a mask token nor expecting any out of vocabulary
            # (oov) token, we set mask_token to None and  num_oov_indices to 0.
            lookup = StringLookup(
                vocabulary=vocabulary,
                mask_token=None,
                num_oov_indices=0,
                output_mode="int" if use_embedding else "binary",
            )
            if use_embedding:
                # Convert the string input values into integer indices.
                encoded_feature = lookup(inputs[feature_name])
                embedding_dims = int(math.sqrt(len(vocabulary)))
                # Create an embedding layer with the specified dimensions.
                embedding = tf.keras.layers.Embedding(
                    input_dim=len(vocabulary), output_dim=embedding_dims
                )
                # Convert the index values to embedding representations.
                encoded_feature = embedding(encoded_feature)
            else:
                # Convert the string input values into a one hot encoding.
                encoded_feature = lookup(tf.expand_dims(inputs[feature_name], -1))
        else:
            # Use the numerical features as-is.
            encoded_feature = tf.expand_dims(inputs[feature_name], -1)

        encoded_features.append(encoded_feature)

    all_features = tf.keras.layers.concatenate(encoded_features)
    return all_features


def create_model_inputs(feature_tool: FeatureTools):
    assert feature_tool is not None
    inputs = {}
    for feature_name in feature_tool.all_feature_name:
        if feature_name in feature_tool.numeric_feature_names:
            inputs[feature_name] = tf.keras.layers.Input(
                name=feature_name, shape=(), dtype=tf.float32
            )
        else:
            inputs[feature_name] = tf.keras.layers.Input(
                name=feature_name, shape=(), dtype=tf.string
            )
    return inputs
