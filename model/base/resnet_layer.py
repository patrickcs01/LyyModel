import tensorflow as tf

from model.base.dense_layer import DenseLayer


class RestNetLayer1D(tf.keras.layers.Layer):
    def __init__(self, use_dense=False, dense_activation="linear",
                 dense_trainable=False, padding_value=0.0,
                 layer_name="", **kwargs):
        super(RestNetLayer1D, self).__init__()
        self.use_dense = use_dense
        self.layer_name = layer_name
        self.dense_activation = dense_activation
        self.dense_trainable = dense_trainable
        self.padding_value = padding_value

    def build(self, input_shape):
        if self.use_dense:
            self.dense = DenseLayer(units=input_shape[-1], activation=self.dense_activation,
                                    layer_name="restnet_layer_dense_" + self.layer_name, trainable=self.dense_trainable)

    def call(self, inputs, x):
        if self.use_dense:
            x = self.dense(x)
        else:
            dim = inputs.shape[1] - x.shape[1]
            paddings = tf.constant([[0, 0], [0, abs(dim)]])
            if dim >= 0:
                x = tf.pad(x, paddings, mode='CONSTANT',
                           constant_values=self.padding_value)
            else:
                inputs = tf.pad(inputs, paddings, mode='CONSTANT',
                                constant_values=self.padding_value)

        inputs = tf.keras.layers.Add()([inputs, x])
        return inputs
