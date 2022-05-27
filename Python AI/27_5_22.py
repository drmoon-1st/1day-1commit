from sklearn import metrics
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import cluster
from sklearn.datasets import load_iris
from types import new_class
import sys
n = int(sys.stdin.readline())
arr1 = sorted(list(map(int, sys.stdin.readline().split())))
m = int(sys.stdin.readline())
arr2 = list(map(int, sys.stdin.readline().split()))


def find(n, i):
    start = 0
    end = n-1
    while start <= end:
        mid = (start + end)//2
        if i == arr1[mid]:
            return 1
        elif i > arr1[mid]:
            start = mid + 1
        else:
            end = mid - 1
    return 0


for i in arr2:
    print(find(n, i))


class fStack:
    class Empty(Exception):
        pass

    class Full(Exception):
        pass

    def __init__(self, capacity: int = 256) -> None:
        self.stk = [None]*capacity
        self.capacity = capacity
        self.ptr = 0

    def __len__(self):
        return self.ptr

    def is_full(self):
        return self.ptr >= self.capacity

    def is_empty(self):
        return self.ptr <= 0

    def push(self, value):
        if self.is_full():
            raise fStack.Full
        else:
            self.stk[self.ptr] = value
            self.ptr += 1

    def pop(self):
        if self.is_empty():
            raise fStack.Empty
        else:
            self.ptr -= 1
            return self.stk[self.ptr]

    def peek(self, idx):
        return self.stk[idx]

    def display(self):
        print(self.stk)


a = fStack(10)
a.push(5)
a.push(3)
a.push(2)
a.pop()
a.push(4)
print(a.__len__())
print(a.peek(2))
a.display()

a = [3, 2, 4, 66, 5, 8, 9, 1]


def insertion_sort(arr):
    for i in range(1, len(arr)):
        for j in range(i, 0, -1):
            if arr[j] < arr[j-1]:
                arr[j], arr[j-1] = arr[j-1], arr[j]
    return arr


def insertion_sort(arr):
    for i in range(len(arr)-1):
        for j in range(i+1, 0, -1):
            if arr[j] < arr[j-1]:
                arr[j], arr[j-1] = arr[j-1], arr[j]
            else:
                break
    return arr


def binary_insertion_sort(arr):
    n = len(arr)
    for i in range(1, n):
        key = arr[i]
        pl = 0
        pr = i - 1

        while True:
            pc = (pl + pr)//2
            if arr[pc] == key:
                break
            elif arr[pc] < key:
                pl = pc + 1
            else:
                pr = pc - 1
            if pl > pr:
                break
        pd = pc + 1 if pl <= pr else pr + 1

        for j in range(i, pd, -1):
            arr[j] = arr[j-1]
        arr[pd] = key
    return arr


print(binary_insertion_sort(a))

#########################


iris = load_iris()
X = iris.data[:, 3]

f, ax = plt.subplots()
ax.scatter(np.arange(len(X)), X, c=iris.target)
plt.show()

cl = cluster.KMeans(n_clusters=3).fit(X[:, np.newaxis])
print(cl.labels_)

f, ax = plt.subplots()
ax.scatter(np.arange(len(X)), X, c=cl.labels_)
plt.show()

af = cluster.AffinityPropagation(damping=0.7).fit(X[:, np.newaxis])
f, ax = plt.subplots()
ax.scatter(np.arange(len(X)), X, c=af.labels_)
plt.show()

ms = cluster.MeanShift(damping=0.7).fit(X[:, np.newaxis])

pd_iris = pd.DataFrame(iris.data)
pd_iris['class'] = iris.target

X = iris.data[:, 2:]

km = cluster.KMeans(n_clusters=3).fit(X)
f, ax = plt.subplots()
ax.scatter(X[:, 0], X[:, 1], c=km.labels_)
plt.show()

print(metrics.homogeneity_score(iris.target, km.labels_))
print(metrics.completeness_score(iris.target, km.labels_))

data = pd.read_csv("abalone.csv", header=None)
X = data[[2, 5, 6]]
y = data[0].astype('category').cat.codes
f, ax = plt.subplots()
sns.heatmap(data[data.corr() > 0.9].corr(), annot=True)
plt.show()

km = cluster.KMeans(n_clusters=3).fit(X)
print(km.labels_)
f, ax = plt.subplots()
ax.scatter(np.arange(0, 4177), y.values, c=km.labels_)
ax.scatter(np.arange(0, 4177), km.labels_)
plt.show()