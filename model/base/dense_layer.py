import tensorflow as tf


class DenseLayer(tf.keras.layers.Layer):
    def __init__(self, units=32,
                 w_initializer=tf.keras.initializers.GlorotNormal(),
                 b_initializer=tf.keras.initializers.Zeros(),
                 activation=None, trainable=True, layer_name="", **kwargs):
        super(DenseLayer, self).__init__()
        self.units = units
        self.w_init = w_initializer
        self.b_init = b_initializer
        self.trainable = trainable
        self.activation = activation
        self.layer_name = layer_name

    def build(self, input_shape):
        self.w = tf.Variable(name="dnn_layer_w_" + self.layer_name,
            initial_value=self.w_init(shape=(input_shape[-1], self.units), dtype="float32"),
            trainable=self.trainable,
        )
        self.b = tf.Variable(name="dnn_layer_b_" + self.layer_name,
            initial_value=self.b_init(shape=(self.units,), dtype="float32"), trainable=self.trainable
        )

    def call(self, inputs):
        z = tf.matmul(inputs, self.w) + self.b
        if self.activation is not None:
            z = self.activation(z)
        return z
