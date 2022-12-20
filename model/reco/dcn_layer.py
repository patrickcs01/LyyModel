import tensorflow as tf

from model.base.dense_layer import DenseLayer
from model.reco.reco_base import RecoBase
from keras import activations

tf.config.experimental_run_functions_eagerly(True)


class DCNLayer(RecoBase):
    def __init__(self, num_deep=3, deep_activation=None, deep_trainable=True,
                 use_deep_bn=True, deep_dropout_rate=0.0, num_cross=2, use_cross_bn=True, cross_activation=None,
                 output_dim=8, output_activation=None, output_trainable=True, **kwargs):
        super(DCNLayer, self).__init__()
        self.num_deep = num_deep
        self.deep_activation = activations.get(deep_activation)
        self.use_deep_bn = use_deep_bn
        self.deep_trainable = deep_trainable
        self.deep_dropout_rate = deep_dropout_rate

        self.num_cross = num_cross
        self.use_cross_bn = use_cross_bn
        self.cross_activation = activations.get(cross_activation)

        self.output_dim = output_dim
        self.output_activation = activations.get(output_activation)
        self.output_trainable = output_trainable

    def build(self, input_shape):
        self.dim = input_shape[-1]
        self.deep_list = list()
        self.output_dens = DenseLayer(units=self.output_dim, layer_name="dcn_layer_output_" + self.layer_name,
                                      activation=self.output_activation,
                                      trainable=self.output_trainable)
        for i in range(self.num_deep):
            self.deep_list.append(DenseLayer(units=int(self.dim * pow(0.5, i)) + 1, activation=self.deep_activation,
                                             layer_name="dcn_deep_layer_{}_".format(i) + self.layer_name,
                                             trainable=self.deep_trainable))

        super(DCNLayer, self).build(input_shape)

    def call(self, inputs):

        deep = inputs
        for deep_layer in self.deep_list:
            deep = deep_layer(deep)
            if self.use_deep_bn:
                deep = tf.keras.layers.BatchNormalization()(deep)
            if self.deep_dropout_rate > 0.0:
                deep = tf.keras.layers.Dropout(self.deep_dropout_rate)(deep)

        cross = inputs
        for i in range(self.num_cross):
            units = cross.shape[-1]
            x = DenseLayer(units=units, layer_name="dcn_cross_layer_{}_".format(i) + self.layer_name,
                           activation=self.cross_activation)(
                cross)
            cross += inputs * x
        if self.use_cross_bn:
            cross = tf.keras.layers.BatchNormalization()(cross)

        z = tf.keras.layers.concatenate([cross, deep])
        z = self.output_dens(z)
        return z
