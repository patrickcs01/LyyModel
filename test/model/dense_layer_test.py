import tensorflow as tf

from model.base.dense_layer import DenseLayer

if __name__ == '__main__':
    x = tf.ones((2, 2))
    linear_layer = DenseLayer(units=4, activation=tf.keras.activations.linear)
    y = linear_layer(x)
    print(y)

