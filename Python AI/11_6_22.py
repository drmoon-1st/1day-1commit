import sys
a = int(sys.stdin.readline())
def factorization(x):
    d = 2
    arr = []
    while d <= x:
        if x % d == 0:
            arr.append(d)
            x = x / d
        else:
            d = d + 1
    return arr
for i in factorization(a):
    print(i)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
print(tf.__version__)

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

# type(tf_data)

# print(tf_data)

# type(X_train)

# X_train.shape

# y_train.shape

# X_test.shape

# y_test.shape

f, ax = plt.subplots()
ax.imshow(X_train[1])

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
    ])
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
model.summary()

model.fit(X_train, y_train, epochs=5)

loss, acc = model.evaluate(X_test, y_test, verbose=2)
model.predict(X_test)

norm = layers.Normalization(input_shape=[1,], axis=None)
norm.adapt(X_train)
model = keras.Sequential([
      norm,
      layers.Dense(units=1)
])
# model.summary()
model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.1), loss='mse')
hist = model.fit(X_train, y_train, epochs=100, verbose=0)

f, ax = plt.subplots()
ax.scatter(X, y)
ax.plot(np.linspace(-0.1, 0.15, 20), model.predict(np.linspace(-0.1, 0.15, 20)))

norm = layers.Normalization(input_shape=[1,], axis=None)
norm.adapt(X_train)
model = keras.Sequential([
      norm,
      layers.Dense(units=64, activation='relu'),
      layers.Dense(units=64, activation='relu'),
      layers.Dense(units=1)
])
# model.summary()
model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.1), loss='mse')
hist = model.fit(X_train, y_train, epochs=1000, verbose=0)

f, ax = plt.subplots()
ax.scatter(X, y)
ax.plot(np.linspace(-0.1, 0.15, 20), model.predict(np.linspace(-0.1, 0.15, 20)), c='red')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import google

pd_data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data',header=None,sep='\t', skipinitialspace=True)
data = []
for i in pd_data[0]:
    data.append(i.strip().split())
# print(data)
pd_data2 = pd.DataFrame(data).dropna()
print(pd_data2.isna().sum())
print(pd_data2)

X_train, X_test, y_train, y_test = train_test_split(pd_data2[0].astype('float64'), pd_data2[7].astype('float64'))
norm = layers.Normalization(input_shape=[1,], axis=None)
norm.adapt(X_train)
model = keras.Sequential([
    norm,
    layers.Dense(units=64, activation='relu'),
    layers.Dense(units=32, activation='relu'),
    layers.Dense(3)
])
model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.1), loss='mse')
hist = model.fit(X_train, y_train, epochs=1000, verbose=0)

f, ax = plt.subplots()
ax.plot(hist.history['loss'])

f, ax = plt.subplots()
ax.scatter(X_test, y_test)
ax.plot(X_test, model.predict(X_test))