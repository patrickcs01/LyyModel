# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

import tensorflow as tf

from model.base.dense_layer import DenseLayer

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    x = tf.ones((2, 2))
    linear_layer = DenseLayer(4,2)
    y = linear_layer(x)
    print(y)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
