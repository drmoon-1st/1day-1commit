import sys

def get_ans(n):
    arr = [True] * (2*n)

    m = int((2*n) ** 0.5)
    for i in range(2, m + 1):
        if arr[i] == True:
            for j in range(i+i, 2*n, i):
                arr[j] = False

    return len([i for i in range(n+1, 2*n) if arr[i] == True])

while True:
    a = int(sys.stdin.readline())
    if a == 0:
        break
    if a == 1:
        print(1)
        continue
#     print(get_ans(a))

from keras.datasets import mnist
from keras import layers
from keras import models
import numpy as np


def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a-c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


(train_image, train_label), (test_image, test_label) = mnist.load_data()
print(train_image.shape)
print(len(train_label))

network = models.Sequential()
network.add(layers.Dense(512, activation='relu', input_shape=(28*28,)))
network.add(layers.Dense(10, activation='softmax'))
network.compile(optimizer='rmsprop',
                loss='categorical_crossentropy', metrics=['accuracy'])