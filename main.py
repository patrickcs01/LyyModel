import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf

from model.base.dense_layer import DenseLayer
from model.base.fm_layer import FMLayer
from model.base.resnet_layer import RestNetLayer1D

# generate 2d classification dataset
# X, y = make_blobs(n_samples=13, centers=2, n_features=2, random_state=1)
# scalar = MinMaxScaler()
# scalar.fit(X)
# X = scalar.transform(X)
# # define and fit the final model
# model = Sequential()
# model.add(FMLayer(units=32, factor_order=2))
# model.add(DenseLayer(units=32, activation=tf.keras.activations.tanh))
# model.add(DenseLayer(units=1, activation=tf.keras.activations.relu))
# model.compile(loss='binary_crossentropy', optimizer='adam')
# model.fit(X, y, epochs=50)
# # new instances where we do not know the answer
# Xnew, _ = make_blobs(n_samples=300, centers=2, n_features=2, random_state=1)
# Xnew = scalar.transform(Xnew)
# # make a prediction
# ynew = model.predict(Xnew)
# # show the inputs and predicted outputs
# for i in range(len(Xnew)):
#  print("X=%s, Predicted=%s" % (Xnew[i], ynew[i]))


raw = np.linspace(0, 0.5, 2000)
x_data = 16 * np.sin(raw) ** 3
noise = np.random.normal(0, 0.02, x_data.shape)
y_data = 13 * np.cos(x_data) - 5 * np.cos(2 * x_data) - 2 * np.cos(3 * x_data) - np.cos(4 * x_data)
# np.square(x_data)
plt.scatter(x_data, y_data)
plt.show()

# model = Sequential()
# model.add(DenseLayer(units=100, layer_name="a"))
# model.add(DenseLayer(units=50, layer_name="a1", activation="tanh"))
# model.add(DenseLayer(units=10, layer_name="a2", activation="relu"))
# model.add(DenseLayer(units=1, layer_name="b", activation="relu"))

input_shape = x_data.shape
inputs = tf.keras.Input(shape=(1,))
x = DenseLayer(units=100, layer_name="a")(inputs)
x1 = FMLayer(units=64, layer_name="a1", activation="tanh")(inputs)
x2 = DenseLayer(units=32, layer_name="a2", activation="relu")(inputs)
k = RestNetLayer1D(layer_name="a3", padding_value=0.0, use_dense=True)(x1, x2)

outputs = DenseLayer(units=1, layer_name="o", activation="relu")(k)
model = tf.keras.Model(inputs, outputs)
model.compile(optimizer="sgd", loss='mse')

model.fit(x_data, y_data, epochs=13000)

# print(model.summary())
y_pred = model.predict(x_data)
print(y_pred)
print(y_data)
plt.scatter(x_data, y_data)
plt.plot(x_data, y_pred, 'r-', lw=3)
plt.show()
