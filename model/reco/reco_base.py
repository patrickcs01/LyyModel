import tensorflow as tf


class RecoBase(tf.keras.layers.Layer):
    def __init__(self, layer_name="", **kwargs):
        super().__init__(**kwargs)
        self.layer_name = layer_name
