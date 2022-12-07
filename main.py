import tensorflow as tf
# example making new probability predictions for a classification problem
from keras.models import Sequential
from keras.layers import Dense
from sklearn.datasets import make_blobs
from sklearn.preprocessing import MinMaxScaler

from model.base.dense_layer import DenseLayer
from model.base.fm_layer import FMLayer

# generate 2d classification dataset
X, y = make_blobs(n_samples=13, centers=2, n_features=2, random_state=1)
scalar = MinMaxScaler()
scalar.fit(X)
X = scalar.transform(X)
# define and fit the final model
model = Sequential()
model.add(FMLayer(units=32, factor_order=2))
model.add(DenseLayer(units=32, activation=tf.keras.activations.tanh))
model.add(DenseLayer(units=1, activation=tf.keras.activations.relu))
model.compile(loss='binary_crossentropy', optimizer='adam')
model.fit(X, y, epochs=50)
# new instances where we do not know the answer
Xnew, _ = make_blobs(n_samples=300, centers=2, n_features=2, random_state=1)
Xnew = scalar.transform(Xnew)
# make a prediction
ynew = model.predict(Xnew)
# show the inputs and predicted outputs
for i in range(len(Xnew)):
 print("X=%s, Predicted=%s" % (Xnew[i], ynew[i]))