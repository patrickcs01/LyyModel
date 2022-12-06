import tensorflow as tf

from model.base.fm_layer import FMLayer

if __name__ == '__main__':
    x = tf.ones((2, 2))
    linear_layer = FMLayer(activation=tf.keras.activations.relu)
    y = linear_layer(x)
    print(y)