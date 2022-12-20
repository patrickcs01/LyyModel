import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf

from model.base.dense_layer import DenseLayer
from model.base.fm_layer import FMLayer
from model.base.resnet_layer import RestNetLayer1D
from model.reco.dcn_layer import DCNLayer

raw = np.linspace(0, 0.5, 2000)
x_data = 6 * np.cos(raw) ** 3
noise = np.random.normal(0, 0.02, x_data.shape)
y_data = 13 * np.cos(x_data) - 5 * np.cos(2 * x_data) - 2 * np.cos(3 * x_data) - np.cos(4 * x_data) + \
         np.random.normal(0, 0.3, x_data.shape) * 5 * np.cos(2 * x_data) + x_data * x_data + np.tan(x_data) * 0.0001
y_data *= 0.3 * np.cos(x_data)
print(x_data)
print(y_data)
plt.scatter(x_data, y_data)
plt.show()

input_shape = x_data.shape
inputs = tf.keras.Input(shape=(1,))
k = DCNLayer(num_deep=5, num_cross=7, deep_activation="linear", cross_activation="relu", layer_name="dcn1",
             output_dim=128, output_activation="relu")(inputs)
# x = DenseLayer(units=100, layer_name="a")(k)
x1 = FMLayer(units=64, layer_name="a1", activation="tanh")(k)
x2 = DenseLayer(units=32, layer_name="a2", activation="relu")(x1)
# k = RestNetLayer1D(layer_name="a3", use_dense=True)([x1, x2])

outputs = DenseLayer(units=1, layer_name="o")(x2)
model = tf.keras.Model(inputs, outputs)
model.compile(optimizer="adam", loss='mse')

model.fit(x_data, y_data, epochs=1000,)
          # callbacks=[tf.keras.callbacks.EarlyStopping(monitor='loss', patience=7)])

y_pred = model.predict(x_data)
print(y_pred)
print(y_data)
plt.scatter(x_data, y_data)
plt.plot(x_data, y_pred, 'r-', lw=3)
plt.show()
