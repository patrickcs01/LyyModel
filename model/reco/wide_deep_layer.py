import math
import tensorflow as tf

from model.base.dense_layer import DenseLayer
from keras import activations


class WideDeepLayer(tf.keras.layers.Layer):
    def __init__(self, num_deep=2, deep_activation=None, deep_trainable=True,
                 use_deep_bn=True, deep_dropout_rate=0.0,
                 output_dim=1, output_activation="softmax", output_trainable=True, layer_name="", **kwargs):
        super(WideDeepLayer, self).__init__()
        self.num_deep = num_deep
        self.deep_activation = activations.get(deep_activation)
        self.use_deep_bn = use_deep_bn
        self.deep_trainable = deep_trainable
        self.deep_dropout_rate = deep_dropout_rate

        self.output_dim = output_dim
        self.output_activation = activations.get(output_activation)
        self.output_trainable = output_trainable
        self.layer_name = layer_name

    def build(self, input_shape):
        self.dim = input_shape[-1]
        self.deep_list = list()
        self.output_dens = DenseLayer(units=self.output_dim, layer_name="dcn_layer_output_" + self.layer_name,
                                      activation=self.output_activation,
                                      trainable=self.output_trainable)
        for i in range(self.num_deep):
            self.deep_list.append(DenseLayer(units=int(self.dim * pow(0.5, i)) + 1, activation=self.deep_activation,
                                             layer_name="wide_deep_deep_layer_{}_".format(i) + self.layer_name,
                                             trainable=self.deep_trainable))
        super(WideDeepLayer, self).build(input_shape)

    def call(self, inputs):
        wide = inputs
        deep = inputs

        for deep_layer in self.deep_list:
            deep = deep_layer(deep)
            if self.use_deep_bn:
                deep = tf.keras.layers.BatchNormalization()(deep)
            if self.deep_dropout_rate > 0.0:
                deep = tf.keras.layers.Dropout(self.deep_dropout_rate)(deep)
        z = tf.keras.layers.concatenate([wide, deep])
        z = self.output_dens(z)
        return z
