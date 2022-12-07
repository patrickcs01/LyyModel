import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense, Activation
from sklearn.datasets import make_blobs
from sklearn.preprocessing import MinMaxScaler

from model.base.dense_layer import DenseLayer
from model.base.fm_layer import FMLayer

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


x_data = np.linspace(0, 0.5, 2000)
noise = np.random.normal(0, 0.02, x_data.shape)
y_data = 100*np.square(x_data)*np.square(x_data) + noise  # 平方

plt.scatter(x_data, y_data)
plt.show()

model = Sequential()
model.add(DenseLayer(units=100, layer_name="a"))
model.add(DenseLayer(units=50, layer_name="a1", activation="tanh"))
model.add(DenseLayer(units=10, layer_name="a2", activation="relu"))
model.add(DenseLayer(units=1, layer_name="b", activation="relu"))

model.compile(optimizer="sgd", loss='mse')

for step in range(30001):
    cost = model.train_on_batch(x_data, y_data)
    if step % 500 == 0:
        print('cost:', cost)
print(model.summary())
y_pred = model.predict(x_data)
print(y_pred)
print(y_data)
plt.scatter(x_data, y_data)
plt.plot(x_data, y_pred, 'r-', lw=3)
plt.show()
