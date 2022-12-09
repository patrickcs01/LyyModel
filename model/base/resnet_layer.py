import tensorflow as tf
from keras import activations

from model.base.dense_layer import DenseLayer


class RestNetLayer1D(tf.keras.layers.Layer):
    def __init__(self, use_dense=False, dense_activation="linear", dense_trainable=False,
                 padding_dense_activation=None, padding_dense_trainable=False,
                 layer_name="", **kwargs):
        super(RestNetLayer1D, self).__init__()
        self.use_dense = use_dense
        self.layer_name = layer_name
        self.dense_activation = activations.get(dense_activation)
        self.dense_trainable = dense_trainable

        self.padding_dense_activation = activations.get(padding_dense_activation)
        self.padding_dense_trainable = padding_dense_trainable

    def build(self, input_shape_list):
        assert isinstance(input_shape_list, list)
        assert len(input_shape_list) == 2
        input_shape = input_shape_list[0]  # left
        x_shape = input_shape_list[1]  # right
        if self.use_dense:
            self.dense = DenseLayer(units=input_shape[-1], activation=self.dense_activation,
                                    layer_name="restnet_layer_dense_" + self.layer_name, trainable=self.dense_trainable)
        else:
            self.dim = input_shape[1] - x_shape[1]
            self.padding_dense = DenseLayer(units=abs(self.dim), activation=self.padding_dense_activation,
                                            layer_name="restnet_layer_padding_dense_" + self.layer_name,
                                            w_initializer=tf.keras.initializers.Zeros(),
                                            b_initializer=tf.keras.initializers.Zeros(),
                                            trainable=self.padding_dense_trainable)

        super(RestNetLayer1D, self).build(input_shape)

    def call(self, inputs_list):
        assert isinstance(inputs_list, list)
        assert len(inputs_list) == 2

        inputs = inputs_list[0]
        x = inputs_list[1]
        if self.use_dense:
            x = self.dense(x)
        else:
            if self.dim > 0:
                x = tf.keras.layers.Concatenate()([x, self.padding_dense(x)])
            elif self.dim < 0:
                inputs = tf.keras.layers.Concatenate()([inputs, self.padding_dense(inputs)])

        inputs = tf.keras.layers.Add()([inputs, x])
        return inputs
