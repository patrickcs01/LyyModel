import tensorflow as tf
import keras.backend as K

from model.base.dense_layer import DenseLayer


class FMLayer(tf.keras.layers.Layer):
    def __init__(self, factor_order=2, trainable=True, activation=None, layer_name="",**kwargs):
        super(FMLayer, self).__init__()
        self.factor_order = factor_order
        self.trainable = trainable
        self.activation = activation
        self.layer_name = layer_name

    def build(self, input_shape):
        self.b = self.add_weight(name='fm_layer_b_' + self.layer_name,
                                 shape=(input_shape[-1],),
                                 initializer='zeros',
                                 trainable=True)
        self.linear = DenseLayer(units=1, layer_name=self.layer_name)
        self.v = self.add_weight(name='fm_layer_v_' + self.layer_name,
                                 shape=(input_shape[-1], self.factor_order),
                                 initializer='glorot_uniform',
                                 trainable=True)
        super(FMLayer, self).build(input_shape)

    def call(self, inputs):
        xv = K.square(K.dot(inputs, self.v))
        interaction = 0.5 * K.sum(xv - K.dot(K.square(inputs), K.square(self.v)), 1)
        z = self.linear(inputs) + interaction
        if self.activation is not None:
            z = self.activation(z)
        return z
